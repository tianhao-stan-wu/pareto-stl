import sys, os
import carla
import random
import math
import time
import numpy as np

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/CARLA_0.9.15/PythonAPI/carla")
except IndexError:
    pass

from src.config import load_config
from src.client import Client
from src.agents import Vehicle, Walker
from src.utils import (
    set_all_lights_green, setup_logging, setup_camera, save_frame, 
    save_stats, save_trajectories, imgs_to_video, compute_and_save_robustness
)
from src.mpc import build_and_solve_mpc
from src.mpc_pareto import solve_mpc_pareto


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
    set_all_lights_green(client.world)

    # spawn agents
    ego = Vehicle(client.world, cfg, "ego_vehicle")
    amb = Vehicle(client.world, cfg, "ambulance")
    ped = Walker(client.world, cfg, "pedestrian")
    agents = [ego, amb, ped]

    amb.agent._proximity_threshold = cfg["ambulance"]["proximity_threshold"]

    v1 = Vehicle(client.world, cfg, "v1")
    v2 = Vehicle(client.world, cfg, "v2")
    v3 = Vehicle(client.world, cfg, "v3")
    v4 = Vehicle(client.world, cfg, "v4")

    # setup
    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)

    ego_warmup = cfg["ego_vehicle"]["warmup"]
    amb_warmup = cfg["ambulance"]["warmup"]

    agent_trajectories = {agent.key: [] for agent in agents}

    build_times = []
    solve_times = []
    num_constraints = None
    num_variables = None

    tick = 0
    camera_tick = start_tick

    try:
        while True:

            print(f"tick: {tick}")

            client.tick()

            speed_kmh = ego.get_speed()
            print(f"Ego Speed: {speed_kmh:.2f} km/h")

            speed_kmh = amb.get_speed()
            print(f"Amb Speed: {speed_kmh:.2f} km/h")

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
                v4.step(acc=0.15, steer=0.5)

            # elif tick <= end_tick:

            #     ego.step()      
            #     amb.random_step()
            #     ped.random_step()

            #     v1.step()
            #     v2.step()
            #     v3.step(acc=-1, steer=0.5)
            #     v4.step(acc=-1, steer=0.5)

            elif tick <= end_tick:

                amb.random_step()
                ped.random_step()

                v1.step()
                v2.step()
                v3.step(acc=-1, steer=0.5)
                v4.step(acc=-1, steer=0.5)
       
                # result = build_and_solve_mpc(client, agents, cfg)
                result = solve_mpc_pareto(client, agents, cfg)
                ego.apply_control(result["control"])

                build_times.append(result["t_build"])
                solve_times.append(result["t_solve"])
               
                if num_constraints is None:
                    num_constraints = result["num_constraints"]
                    num_variables = result["num_variables"]

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

