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
    save_stats, save_trajectories, imgs_to_video, save_robustness_exp1, save_pareto_log,
    save_theorem1_results
)

from exp1.mpc import solve_mpc_pareto
from exp1.check_feas import check_feasibility
from exp1.theorem1 import (
    verify_theorem1,
    summarise_theorem1,
    make_exp1_joint_sampler,
    make_exp1_satisfaction_checker,
)


def main():

    # init
    cfg = load_config(exp="exp1")
    log_dir, img_dir = setup_logging(cfg)
    
    seed = cfg["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

    print(f"Starting project: {cfg['project']['name']}")

    client = Client(cfg)
    camera, img_queue = setup_camera(client.world, cfg["carla"])

    # set all green for demo
    set_all_lights_green(client.world)

    # spawn key agents
    ego = Vehicle(client.world, cfg, "ego_vehicle")
    amb = Vehicle(client.world, cfg, "ambulance")
    ped = Walker(client.world, cfg, "pedestrian")
    agents = [ego, amb, ped]

    # amb.agent._proximity_threshold = cfg["ambulance"]["proximity_threshold"]

    # spawn other road vehicles 
    v1 = Vehicle(client.world, cfg, "v1")
    v2 = Vehicle(client.world, cfg, "v2")
    v3 = Vehicle(client.world, cfg, "v3")
    v4 = Vehicle(client.world, cfg, "v4")

    # setup
    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)
    emergency_tick = int(cfg["mpc"]["emergency_start"] / dt)

    ego_warmup = cfg["ego_vehicle"]["warmup"]
    amb_warmup = cfg["ambulance"]["warmup"]

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
        print(f"Selected solver: {solver} \n")

    try:
        while True:

            print(f"tick: {tick} \n")

            client.tick()

            # save carla image
            if tick == camera_tick:
                camera.listen(img_queue.put)

            if tick > camera_tick:
                save_frame(img_queue, img_dir, tick - camera_tick)
            
            # control step
            if tick < start_tick:

                ego.step(acc=ego_warmup)
                amb.step(acc=amb_warmup)
                ped.step()

                v1.step()
                v2.step()
                v3.step(acc=0.1, steer=0.5)
                v4.step(acc=0.1, steer=0.5)

            elif tick < end_tick:

                amb.random_step()
                ped.random_step()

                v1.step()
                v2.step()
                v3.step(acc=-1, steer=0.5)
                v4.step(acc=-1, steer=0.5)

                emergency = tick >= emergency_tick
       
                feas_result = check_feasibility(client, agents, cfg, emergency=emergency, u_prev=u_prev)

                feas_build_times.append(feas_result["t_build"])
                feas_solve_times.append(feas_result["t_solve"])

                if feas_num_constraints is None:
                    feas_num_constraints = feas_result["num_constraints"]
                    feas_num_variables = feas_result["num_variables"]

                # Theorem 1 validation is performed only for strict-feasible
                # solutions, after x_star and u_star have been fixed.
                if feas_result["status"] and theorem1_enabled:
                    N = int(round(cfg["mpc"]["horizon"] / dt))
                    sampler = make_exp1_joint_sampler(ped, amb, N, dt)
                    satisfies = make_exp1_satisfaction_checker(
                        float(cfg["stl"]["pedestrian"]),
                        float(cfg["stl"]["ambulance"]),
                        cfg["stl"]["lane"],
                        emergency,
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

                # if infeasible
                if not feas_result["status"]:

                    n_infeas += 1

                    pareto_result = solve_mpc_pareto(client, agents, cfg, emergency=emergency, u_prev=u_prev)

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

            tick += 1

    finally:
        
        camera.stop()
        camera.destroy()
        client.quit(destroy=True)

        total_tick = end_tick - start_tick

        save_stats(
            feas_build_times, feas_solve_times, feas_num_constraints, feas_num_variables,
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
        
        agent_dims = {}

        for agent in agents:
            if hasattr(agent, "width") and hasattr(agent, "length"):
                agent_dims[agent.key] = {"width": agent.width, "length": agent.length}

        save_robustness_exp1(
            agent_trajectories, cfg["stl"], cfg["carla"]["dt"], log_dir
        )

        
if __name__ == "__main__":
    main()

