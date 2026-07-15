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
from src.vehicle import Vehicle
from src.utils import SmoothNoise, set_all_lights_green
from src.mpc import build_and_solve_mpc


def ped_init(world: carla.World, cfg):
    bp_lib = world.get_blueprint_library()
    ped_bp = bp_lib.find("walker.pedestrian.0001")

    pcfg = cfg["pedestrian"]
    loc = pcfg["spawn"]["location"]
    rot = pcfg["spawn"]["rotation"]
    spawn = carla.Transform(
        carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    ped = world.try_spawn_actor(ped_bp, spawn)
    if ped is None:
        raise RuntimeError("Failed to spawn pedestrian.")
    world.tick()
    return ped


def parked_vehicles_init(world: carla.World, cfg: dict):
    pcfg = cfg["parked_vehicles"]
    loc = pcfg["spawn"]["location"]
    rot = pcfg["spawn"]["rotation"]
    gap = pcfg.get("gap", 6.0)

    blueprints = [
        world.get_blueprint_library().find(pcfg["blueprints"][0]),
        world.get_blueprint_library().find(pcfg["blueprints"][1]),
    ]

    t1 = carla.Transform(
        carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    forward = t1.get_forward_vector()
    t2 = carla.Transform(
        carla.Location(x=loc["x"] - forward.x * gap, y=loc["y"] - forward.y * gap, z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    vehicles = []
    for bp, t in zip(blueprints, [t1, t2]):
        v = world.try_spawn_actor(bp, t)
        if v is None:
            raise RuntimeError(f"Failed to spawn {bp.id} at {t.location}")
        vehicles.append(v)

    world.tick()
    return vehicles


def ped_step(ped, cfg: dict):
    fwd = ped.get_transform().get_forward_vector()
    noise_yaw = random.gauss(0, cfg["pedestrian"]["std_dir"])
    direction = carla.Vector3D(
        x=fwd.x * math.cos(noise_yaw) - fwd.y * math.sin(noise_yaw),
        y=fwd.x * math.sin(noise_yaw) + fwd.y * math.cos(noise_yaw),
        z=0.0
    )
    control = carla.WalkerControl()
    control.speed = max(0.0, random.gauss(cfg["pedestrian"]["mean_speed"], cfg["pedestrian"]["std_speed"]))
    control.direction = direction
    ped.apply_control(control)


def amb_step(amb: Vehicle, cfg: dict):
    if not hasattr(amb_step, "_acc_noise"):
        amb_step._acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=cfg["ambulance"]["std_acc"])
        amb_step._steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=cfg["ambulance"]["std_steer"])

    control = amb.agent.run_step()
    control.manual_gear_shift = False

    acc = control.throttle + amb_step._acc_noise.sample()
    if acc >= 0:
        control.throttle = max(0.0, min(acc, 1.0))
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = min(abs(acc), 1.0)

    control.steer = max(-1.0, min(control.steer + amb_step._steer_noise.sample(), 1.0))
    amb.apply_control(control)


def main():
    cfg = load_config(exp="exp1")
    print(f"Starting project: {cfg['project']['name']}")

    client = Client(cfg)

    ego = Vehicle(client.world, cfg, "ego_vehicle")
    amb = Vehicle(client.world, cfg, "ambulance")
    amb.agent._proximity_threshold = cfg["ambulance"]["proximity"]

    ped = ped_init(client.world, cfg)
    parked = parked_vehicles_init(client.world, cfg)

    set_all_lights_green(client.world)

    dt = cfg["carla"]["dt"]
    start_tick = int(cfg["mpc"]["sim_start"] / dt)
    end_tick = int(cfg["mpc"]["sim_end"] / dt)
    tick = 0

    try:
        while True:

            print(f"tick: {tick}")

            client.tick()
            tick += 1

            # warmup: ego accelerates, amb full throttle to target speed
            if tick < start_tick:
                # default step
                ego.step(acc=0.4)
                amb.step(acc=1.0)

            # MPC phase
            elif tick <= end_tick:
                ped_step(ped, cfg)
                # step with noise
                amb_step(amb, cfg)

                result = build_and_solve_mpc(client, ego, ped, amb, cfg)

                if result["status"]:
                    ego.apply_control(result["control"])
                else:
                    ego.step()
                    print(f"MPC failed at tick {tick}, fallback to autopilot")

                time.sleep(1)

            else:
                print("End of simulation")
                break

    finally:
        client.quit(destroy=True)


if __name__ == "__main__":
    main()