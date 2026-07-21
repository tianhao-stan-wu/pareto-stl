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
from src.mpc import build_and_solve_mpc


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
    camera, img_queue = setup_camera(client.world)

    # spawn vehicles
    ego = Vehicle(client.world, cfg, "ego_vehicle")
    amb = Vehicle(client.world, cfg, "ambulance")
    amb.agent._proximity_threshold = cfg["ambulance"]["proximity"]

    ped = Walker(client.world, cfg, "pedestrian")
    p1 = Vehicle(client.world, cfg, "parked_v1")
    p2 = Vehicle(client.world, cfg, "parked_v2")

    agents = [ego, amb, ped, p1, p2]

    set_all_lights_green(client.world)

    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)
    tick = 0

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
                ego.step(acc=cfg["ego_vehicle"]["warmup"])
                amb.step(acc=cfg["ambulance"]["warmup"])

            # MPC phase
            elif tick <= end_tick:

                ped.random_step()
                amb.random_step()
       
                result = build_and_solve_mpc(client, agents, cfg)
                ego.apply_control(result["control"])

                build_times.append(result["t_build"])
                solve_times.append(result["t_solve"])

                save_frame(img_queue, img_dir, tick - start_tick)

            else:
                print("End of simulation")
                break


            tick += 1

    finally:
        save_timing(build_times, log_dir / "build_times.txt")
        save_timing(solve_times, log_dir / "solve_times.txt")
        
        camera.stop()
        camera.destroy()
        client.quit(destroy=True)


if __name__ == "__main__":
    main()