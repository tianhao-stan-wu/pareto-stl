import sys, os
import carla
import random
import math
import time

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/CARLA_0.9.15/PythonAPI/carla")
except IndexError:
    pass

from src.config import load_config
from src.carla_client import Client
from src.agents import Vehicle, Walker
from src.utils import SmoothNoise, set_all_lights_green, print_distances
from src.mpc import build_and_solve_mpc


def main():
    cfg = load_config(exp="exp1")
    print(f"Starting project: {cfg['project']['name']}")

    client = Client(cfg)

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

    try:
        while True:

            print(f"tick: {tick}")

            client.tick()
            
            # warmup: ego accelerates, amb full throttle to target speed
            if tick < start_tick:
                # default step
                ego.step(acc=0.4)
                amb.step(acc=0.9)

            # MPC phase
            elif tick <= start_tick+1:
                break
                ped.random_step()
                amb.random_step()
       
                result = build_and_solve_mpc(client, agents, cfg)

                if result["status"]:
                    ego.apply_control(result["control"])
                else:
                    # to do: apply control from first step() from mpc
                    ego.step()
                    print(f"MPC failed at tick {tick}, fallback to autopilot")

            else:
                print("End of simulation")
                break

            tick += 1

    finally:
        client.quit(destroy=True)


if __name__ == "__main__":
    main()