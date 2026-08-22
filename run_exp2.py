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
    save_stats, save_trajectories, imgs_to_video, compute_and_save_robustness, set_traffic_lights_by_location
)

from exp2.mpc import solve_mpc_pareto
from exp2.check_feas import check_feasibility


def main():

    # init
    cfg = load_config(exp="exp2")
    log_dir, img_dir = setup_logging(cfg)
    
    seed = cfg["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

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

    agents = [ego, leader, follower, left]

    # setup
    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)
    emergency_tick = int(cfg["mpc"]["emergency_start"] / dt)
    swerve_left_tick = int((cfg["mpc"]["emergency_start"] - 0.7) / dt)
    swerve_right_tick = int((cfg["mpc"]["emergency_start"] - 0.2) / dt)
    straight_tick = int((cfg["mpc"]["emergency_start"] + 0.5) / dt)

    agent_trajectories = {agent.key: [] for agent in agents}

    build_times = []
    solve_times = []
    num_constraints = None
    num_variables = None

    tick = 0
    camera_tick = start_tick

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

                ego.step(acc=0.9)
                leader.step(acc=1)
                follower.step(acc=1)
                left.step(acc=1)

            else:

                if tick < swerve_left_tick:
                    leader.step()

                elif swerve_left_tick <= tick < swerve_right_tick:
                    control = carla.VehicleControl(
                        throttle=0.35,
                        steer=-0.1*(tick-swerve_left_tick),
                        brake=0.0
                    )
                    leader.actor.apply_control(control)

                elif swerve_right_tick <= tick < emergency_tick:
                    control = carla.VehicleControl(
                        throttle=0.35,
                        steer=0.15,
                        brake=0.0
                    )
                    leader.actor.apply_control(control)

                elif tick == emergency_tick:
                    leader.agent.set_destination(
                        carla.Location(x=-359.331, y=8.909, z=2.803)
                    )

                else:
                    leader.step()

                if start_tick <= tick < emergency_tick:
                    follower.step(acc=0.7)
                    left.step(acc=0.8)
                else:
                    follower.step()
                    left.step(steer=0)

                emergency = tick >= emergency_tick
                result = check_feasibility(client, agents, cfg, emergency=emergency)

                # if infeasible
                if not result["status"]:
                    result = solve_mpc_pareto(client, agents, cfg, emergency=emergency)
    
                ego.apply_control(result["control"])

                build_times.append(result["t_build"])
                solve_times.append(result["t_solve"])
               
                if num_constraints is None:
                    num_constraints = result["num_constraints"]
                    num_variables = result["num_variables"]

                for agent in agents:
                    loc = agent.get_transform().location
                    agent_trajectories[agent.key].append([float(loc.x), float(loc.y)])

            col1.step(acc=-1)
            col2.step(acc=-1)

            tick += 1

            if tick == end_tick:
                print("End of simulation")
                break 

    finally:
        
        camera.stop()
        camera.destroy()
        client.quit(destroy=True)

        save_stats(build_times, solve_times, num_constraints, num_variables, log_dir)
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

