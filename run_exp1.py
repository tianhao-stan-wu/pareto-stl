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
    save_stats, save_trajectories, imgs_to_video, compute_and_save_robustness, save_pareto_log
)

from exp1.mpc import solve_mpc_pareto
from exp1.check_feas import check_feasibility


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
       
                feas_result = check_feasibility(client, agents, cfg, emergency=emergency)

                feas_build_times.append(feas_result["t_build"])
                feas_solve_times.append(feas_result["t_solve"])

                if feas_num_constraints is None:
                    feas_num_constraints = feas_result["num_constraints"]
                    feas_num_variables = feas_result["num_variables"]

                # if infeasible
                if not feas_result["status"]:

                    n_infeas += 1

                    pareto_result = solve_mpc_pareto(client, agents, cfg, emergency=emergency)

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

                else:
                    ego.apply_control(feas_result["control"])

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

        save_trajectories(agent_trajectories, log_dir)
        imgs_to_video(log_dir)   
        
        agent_dims = {}

        for agent in agents:
            if hasattr(agent, "width") and hasattr(agent, "length"):
                agent_dims[agent.key] = {"width": agent.width, "length": agent.length}

        compute_and_save_robustness(
            agent_trajectories, cfg["stl"], agent_dims, cfg["carla"]["dt"], log_dir
        )

        
if __name__ == "__main__":
    main()

