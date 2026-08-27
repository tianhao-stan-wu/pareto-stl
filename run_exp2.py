import sys, os
import carla
import random
import math
import time
import numpy as np
import gurobipy as gp
import cvxpy as cp

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/CARLA_0.9.15/PythonAPI/carla")
except IndexError:
    pass

from src.config import load_config
from src.client import Client
from src.agents import Vehicle, Walker
from src.utils import (
    set_all_lights_green, setup_logging, setup_camera, save_frame,
    save_stats, save_trajectories, imgs_to_video, save_pareto_log,
    save_robustness_exp2, save_theorem1_results
)

from exp2.mpc import solve_mpc_pareto
from exp2.check_feas import check_feasibility
from exp2.theorem1 import (
    verify_theorem1,
    summarise_theorem1,
    make_exp2_joint_sampler,
    make_exp2_satisfaction_checker,
)


def main():

    cfg = load_config(exp="exp2")

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["project"]["seed"]
    density = int(sys.argv[2]) if len(sys.argv) > 2 else cfg["mpc"]["density"]
    num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    cfg["project"]["seed"] = seed
    cfg["mpc"]["density"] = density
    cfg["project"]["name"] = (
        f"{cfg['project']['exp']}_batch_d{density}_{num_runs}"
    )

    random.seed(seed)
    np.random.seed(seed)

    ego_dx = random.uniform(-3.0, 3.0)
    leader_dx = random.uniform(-3.0, 3.0)
    follower_dx = random.uniform(-3.0, 3.0)
    left_dx = random.uniform(-3.0, 3.0)
    ped_dx = random.uniform(-0.5, 0.5)

    cfg["ego_vehicle"]["spawn"]["location"]["x"] += ego_dx
    cfg["leader"]["spawn"]["location"]["x"] += leader_dx
    cfg["follower"]["spawn"]["location"]["x"] += follower_dx
    cfg["left_vehicle"]["spawn"]["location"]["x"] += left_dx
    cfg["pedestrian2"]["spawn"]["location"]["x"] += ped_dx

    print(f"Trial seed: {seed}")
    print(
        "Initial-condition perturbations:\n"
        f"  ego:       dx={ego_dx:+.3f} m\n"
        f"  leader:    dx={leader_dx:+.3f} m\n"
        f"  follower:  dx={follower_dx:+.3f} m\n"
        f"  left:      dx={left_dx:+.3f} m\n"
        f"  pedestrian dx={ped_dx:+.3f} m"
    )

    # Create log directory after modifying cfg so the actual
    # trial configuration can be saved.
    log_dir, img_dir = setup_logging(cfg)

    print(f"Starting project: {cfg['project']['name']}")


    client = Client(cfg)
    camera, img_queue = setup_camera(client.world, cfg["carla"])

    # spawn agents
    ego = Vehicle(client.world, cfg, "ego_vehicle")
    leader = Vehicle(client.world, cfg, "leader")
    follower = Vehicle(client.world, cfg, "follower")
    left = Vehicle(client.world, cfg, "left_vehicle")

    col1 = Vehicle(client.world, cfg, "collision_v1")
    col2 = Vehicle(client.world, cfg, "collision_v2")

    ped1 = Walker(client.world, cfg, "pedestrian1")
    ped2 = Walker(client.world, cfg, "pedestrian2")

    agents = [ego, leader, follower, left, ped2]

    # timing
    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)
    emergency_tick = int(cfg["mpc"]["emergency_start"] / dt)
    swerve_left_tick = int((cfg["mpc"]["emergency_start"] - 0.7) / dt)
    swerve_right_tick = int((cfg["mpc"]["emergency_start"] - 0.2) / dt)

    agent_trajectories = {agent.key: [] for agent in agents}

    feas_build_times = []
    feas_solve_times = []
    feas_num_constraints = None
    feas_num_variables = None

    pareto_build_times = []
    pareto_solve_times = []
    pareto_solve_all_times = []
    pareto_num_constraints = None
    pareto_num_variables = None

    pareto_records = []

    theorem1_cfg = cfg.get("theorem1", {})
    theorem1_enabled = bool(theorem1_cfg.get("enabled", True))
    theorem1_eps = float(theorem1_cfg.get("eps", 0.05))
    theorem1_beta = float(theorem1_cfg.get("beta", 0.05))
    theorem1_N_approx = int(theorem1_cfg.get("N_approx", 10000))
    theorem1_records = []

    u_prev = None

    tick = 0
    camera_tick = start_tick

    n_infeas = 0
    n_infeas_resolved = 0

    solver = next((s for s in [cp.GUROBI, cp.CPLEX, cp.SCIP, cp.CBC]
                   if s in cp.installed_solvers()), None)
    if solver is None:
        raise RuntimeError(f"No MIP solver found. Installed: {cp.installed_solvers()}")
    else:
        print(f"Installed solver: {cp.installed_solvers()}")
        print(f"Selected solver: {solver}\n")

    try:
        while True:

            print(f"tick: {tick}\n")

            client.tick()

            # camera
            if tick == camera_tick:
                camera.listen(img_queue.put)
            if tick > camera_tick:
                save_frame(img_queue, img_dir, tick - camera_tick)

            # warmup
            if tick < start_tick:
                ego.step(acc=0.85)
                leader.step(acc=1)
                follower.step(acc=1)
                left.step(acc=0.9)

            # mpc phase
            elif tick < end_tick:

                # leader scripted behaviour
                if tick < swerve_left_tick:
                    leader.step()
                elif swerve_left_tick <= tick < swerve_right_tick:
                    control = carla.VehicleControl(
                        throttle=0.35,
                        steer=-0.15 * (tick - swerve_left_tick),
                        brake=0.0,
                    )
                    leader.actor.apply_control(control)
                elif swerve_right_tick <= tick < emergency_tick:
                    control = carla.VehicleControl(
                        throttle=0.35, steer=0.2, brake=0.0,
                    )
                    leader.actor.apply_control(control)
                elif tick == emergency_tick:
                    leader.agent.set_destination(
                        carla.Location(x=-359.331, y=8.909, z=2.803)
                    )
                    leader.step()
                else:
                    if not leader.is_done():
                        leader.step()

                # follower and left
                if start_tick <= tick < emergency_tick:
                    follower.step(acc=0.7)
                    left.step(acc=0.8)
                else:
                    if not follower.is_done():
                        follower.step()
                    if not left.is_done():
                        left.step(steer=0)

                # pedestrian
                ped2.random_step()

                # skip feas and pareto if ego speed stops
                if ego.get_speed() < 1:
                    ego.step(acc=-0.1)  # apply small brake
                    for agent in agents:
                        loc = agent.get_transform().location
                        agent_trajectories[agent.key].append([float(loc.x), float(loc.y)])
                    tick += 1
                    continue

                # ego: gate then pareto
                emergency = tick >= emergency_tick

                feas_result = check_feasibility(
                    client, agents, cfg, emergency=emergency, u_prev=u_prev)

                feas_build_times.append(feas_result["t_build"])
                feas_solve_times.append(feas_result["t_solve"])

                if feas_num_constraints is None:
                    feas_num_constraints = feas_result["num_constraints"]
                    feas_num_variables = feas_result["num_variables"]

                # Theorem 1 validation: only strict-feasible solutions are tested.
                if feas_result["status"] and theorem1_enabled:
                    N = int(round(cfg["mpc"]["horizon"] / dt))
                    sampler = make_exp2_joint_sampler(
                        leader, follower, left, ped2, N, dt
                    )
                    satisfies = make_exp2_satisfaction_checker(
                        cfg["stl"], emergency
                    )
                    theorem_result = verify_theorem1(
                        feas_result["u_star"],
                        feas_result["x_star"],
                        sampler,
                        satisfies,
                        theorem1_eps,
                        theorem1_beta,
                        N_approx=theorem1_N_approx,
                    )
                    theorem_result["tick"] = int(tick)
                    theorem_result["emergency"] = bool(emergency)
                    theorem1_records.append(theorem_result)

                    if theorem_result["validated"]:
                        p_hat = theorem_result["p_viol_hat"]
                        verdict = "<= eps" if p_hat <= theorem1_eps else "> eps"
                        print(
                            f"  [Theorem 1] tick={tick}: VALIDATED "
                            f"(M={theorem_result['M']}), "
                            f"p_viol_hat={p_hat:.4f} ({verdict})"
                        )
                    else:
                        print(
                            f"  [Theorem 1] tick={tick}: validation FAILED "
                            f"({theorem_result['validation_failures']}/{theorem_result['M']})"
                        )

                if not feas_result["status"]:

                    n_infeas += 1

                    pareto_result = solve_mpc_pareto(
                        client, agents, cfg, emergency=emergency, u_prev=u_prev)

                    if pareto_result["status"]:

                        n_infeas_resolved += 1

                        pareto_build_times.append(pareto_result["t_build"])
                        pareto_solve_times.append(pareto_result["t_solve"])
                        pareto_solve_all_times.append(pareto_result["t_solve_all"])

                        if pareto_num_constraints is None:
                            pareto_num_constraints = pareto_result["num_constraints"]
                            pareto_num_variables = pareto_result["num_variables"]

                        pareto_records.append({
                            "tick": tick,
                            **pareto_result["pareto_log"],
                        })

                    ego.apply_control(pareto_result["control"])
                    u_prev = pareto_result["u_applied"]

                else:
                    ego.apply_control(feas_result["control"])
                    u_prev = feas_result["u_applied"]

                for agent in agents:
                    loc = agent.get_transform().location
                    agent_trajectories[agent.key].append([float(loc.x), float(loc.y)])

            else:
                print("End of simulation")
                break

            # background vehicles always brake to stop
            col1.step(acc=-1)
            col2.step(acc=-1)

            tick += 1

    finally:

        camera.stop()
        camera.destroy()
        client.quit(destroy=True)

        total_tick = end_tick - start_tick

        save_stats(
            feas_build_times, feas_solve_times,
            feas_num_constraints, feas_num_variables,
            pareto_build_times, pareto_solve_times, pareto_solve_all_times,
            pareto_num_constraints, pareto_num_variables,
            total_tick, n_infeas, n_infeas_resolved,
            log_dir,
        )

        save_pareto_log(pareto_records, log_dir)

        theorem1_summary = summarise_theorem1(theorem1_records)
        save_theorem1_results(theorem1_records, theorem1_summary, log_dir)

        save_trajectories(agent_trajectories, log_dir)
        imgs_to_video(log_dir)

        save_robustness_exp2(
            agent_trajectories, cfg["stl"], cfg["carla"]["dt"], log_dir
        )


if __name__ == "__main__":
    main()