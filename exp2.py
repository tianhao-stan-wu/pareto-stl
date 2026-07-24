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
from src.carla_client import Client
from src.agents import Vehicle, Walker
from src.utils import SmoothNoise, set_all_lights_green, print_distances, setup_logging
from src.utils import setup_camera, save_frame, save_timing
from src.mpc_soft import build_and_solve_mpc_soft
from src.mpc_hard import build_and_solve_mpc_hard
from src.mpc_pareto import build_and_solve_mpc_pareto


def main():

    # init
    cfg = load_config(exp="exp1")
    log_dir, img_dir = setup_logging(cfg)
    
    # set a seed for reproducibility
    seed = cfg["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

    print(f"Starting project: {cfg['project']['name']}")

    client = Client(cfg)

    # start camera
    camera, img_queue = setup_camera(client.world, cfg["carla"])

    # spawn vehicles
    ego = Vehicle(client.world, cfg, "ego_vehicle")
    amb = Vehicle(client.world, cfg, "ambulance")
    amb.agent._proximity_threshold = cfg["ambulance"]["proximity"]

    ped = Walker(client.world, cfg, "pedestrian")
    # p1 = Vehicle(client.world, cfg, "parked_v1")
    # p2 = Vehicle(client.world, cfg, "parked_v2")
    o1 = Vehicle(client.world, cfg, "opposite_v1")

    agents = [ego, amb, ped, o1]

    set_all_lights_green(client.world)

    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)

    tick = 0

    ego_warmup = cfg["ego_vehicle"]["warmup"]
    amb_warmup = cfg["ambulance"]["warmup"]
    o1_warmup = cfg["opposite_v1"]["warmup"]

    build_times = []
    solve_times = []

    try:
        while True:

            print(f"tick: {tick}")

            client.tick()

            if tick == start_tick - 1:
                camera.listen(img_queue.put)
            
            # warmup: ego accelerates, amb full throttle to target speed
            if tick < start_tick:
                # default step
                ego.step(acc=ego_warmup)
                amb.step(acc=amb_warmup)
                o1.step(acc=o1_warmup)

            # MPC phase
            elif tick <= end_tick:
                # break
                ped.random_step()
                amb.random_step()
                o1.random_step()
       
                if cfg["mpc"]["type"] == "hard":
                    result = build_and_solve_mpc_hard(client, agents, cfg)

                elif cfg["mpc"]["type"] == "soft":
                    result = build_and_solve_mpc_soft(client, agents, cfg)
                    
                elif cfg["mpc"]["type"] == "pareto":
                    result = build_and_solve_mpc_pareto(client, agents, cfg)

                ego.apply_control(result["control"])

                build_times.append(result["t_build"])
                solve_times.append(result["t_solve"])

                save_frame(img_queue, img_dir, tick - start_tick)

            else:
                print("End of simulation")
                break

            tick += 1

    finally:
        
        camera.stop()
        camera.destroy()
        client.quit(destroy=True)

        save_timing(build_times, log_dir / "build_times.txt")
        save_timing(solve_times, log_dir / "solve_times.txt")


if __name__ == "__main__":
    main()