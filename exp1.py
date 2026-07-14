# import sys, os
# import carla
# import random
# import math

# try:
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/CARLA_0.9.15/PythonAPI/carla")
# except IndexError:
#     pass

# from agents.navigation.behavior_agent import BehaviorAgent

# from src.config import load_config
# from src.carla_client import Client
# from src.vehicle import Vehicle
# from src.mpc import bicycle_to_carla, build_and_solve_mpc


# class SmoothNoise:
#     """Ornstein-Uhlenbeck process for smooth random acceleration."""

#     def __init__(self, mean=0.0, theta=0.3, sigma=0.1):
#         self.mean = mean      # reverts toward this value
#         self.theta = theta    # how fast it reverts (higher = faster)
#         self.sigma = sigma    # volatility
#         self.value = 0.0

#     def sample(self):
#         self.value += self.theta * (self.mean - self.value) + self.sigma * random.gauss(0, 1)
#         return self.value


# def get_speed(actor) -> float:
#     """Return vehicle speed in km/h."""
#     v = actor.get_velocity()
#     return 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5


# def set_all_lights_green(world, green_time: float = 30.0):
#     """Set all traffic lights to green with specified duration."""
#     for light in world.get_actors().filter("traffic.traffic_light*"):
#         light.set_state(carla.TrafficLightState.Green)
#         light.set_green_time(green_time)
#         light.set_red_time(0.0)
#         light.set_yellow_time(0.0)


# def ped_init(world: carla.World, cfg):
#     """Spawn a pedestrian and apply control with speed sampled from N(mean, std)."""
#     bp_lib = world.get_blueprint_library()
#     ped_bp = bp_lib.find('walker.pedestrian.0001')

#     cfg = cfg["pedestrian"]
#     loc = cfg["spawn"]["location"]
#     rot = cfg["spawn"]["rotation"]
#     spawn = carla.Transform(
#             carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
#             carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
#         )

#     ped = world.try_spawn_actor(ped_bp, spawn)
#     if ped is None:
#         raise RuntimeError("Failed to spawn pedestrian. Location may be blocked.")

#     world.tick()  # let server register the spawn

#     return ped


# def ambulance_init(world: carla.World, cfg: dict):
#     """Spawn an ambulance and set destination from config."""
#     cfg = cfg["ambulance"]
#     bp_lib = world.get_blueprint_library()
#     amb_bp = bp_lib.find(cfg["blueprint"])
    
#     loc = cfg["spawn"]["location"]
#     rot = cfg["spawn"]["rotation"]
#     spawn = carla.Transform(
#         carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
#         carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
#     )

#     amb = world.try_spawn_actor(amb_bp, spawn)
#     if amb is None:
#         raise RuntimeError("Failed to spawn ambulance. Location may be blocked.")

#     world.tick()

#     agent = BehaviorAgent(amb, behavior=cfg.get("behavior", "normal"))
#     # agent.ignore_traffic_lights(True)
#     # agent.ignore_stop_signs(True)

#     dest = cfg["destination"]["location"]
#     agent.set_destination(carla.Location(x=dest["x"], y=dest["y"], z=dest["z"]))
#     agent.set_target_speed(cfg.get("target_speed", 50))

#     return amb, agent


# def parked_vehicles_init(world: carla.World, cfg: dict):
#     """Spawn two vehicles queued at a stop sign from config."""
#     cfg = cfg["parked_vehicles"]
#     loc = cfg["spawn"]["location"]
#     rot = cfg["spawn"]["rotation"]
#     gap = cfg.get("gap", 6.0)  # meters between vehicles

#     blueprints = [
#         world.get_blueprint_library().find(cfg["blueprints"][0]),
#         world.get_blueprint_library().find(cfg["blueprints"][1]),
#     ]

#     # first vehicle at specified location
#     t1 = carla.Transform(
#         carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
#         carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
#     )

#     # second vehicle placed behind the first using the facing direction
#     forward = t1.get_forward_vector()
#     t2 = carla.Transform(
#         carla.Location(
#             x=loc["x"] - forward.x * gap,
#             y=loc["y"] - forward.y * gap,
#             z=loc["z"]
#         ),
#         carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
#     )

#     vehicles = []
#     for bp, t in zip(blueprints, [t1, t2]):
#         v = world.try_spawn_actor(bp, t)
#         if v is None:
#             raise RuntimeError(f"Failed to spawn {bp.id} at {t.location}")
#         # v.apply_control(carla.VehicleControl(brake=1.0))  # keep stationary
#         vehicles.append(v)

#     world.tick()
#     return vehicles


# def full_throttle_step(agent, actor, target_speed: float):
#     """Apply max throttle until target speed, then hand back to agent."""
#     control = agent.run_step()
#     control.manual_gear_shift = False

#     if get_speed(actor) < target_speed:
#         control.throttle = 1.0
#         control.brake = 0.0

#     actor.apply_control(control)


# def ped_step(ped, cfg: dict):
#     """Apply one step of noisy pedestrian control."""
#     fwd = ped.get_transform().get_forward_vector()
#     noise_yaw = random.gauss(0, cfg["pedestrian"]["std_dir"])

#     direction = carla.Vector3D(
#         x=fwd.x * math.cos(noise_yaw) - fwd.y * math.sin(noise_yaw),
#         y=fwd.x * math.sin(noise_yaw) + fwd.y * math.cos(noise_yaw),
#         z=0.0
#     )

#     control = carla.WalkerControl()
#     control.speed = max(0.0, random.gauss(cfg["pedestrian"]["mean_speed"], cfg["pedestrian"]["std_speed"]))
#     control.direction = direction
#     ped.apply_control(control)


# def amb_step(agent, actor, cfg: dict):
#     """Apply one step of ambulance control with smooth noise on acceleration and steering."""
#     if not hasattr(amb_step, "_acc_noise"):
#         amb_step._acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=cfg["ambulance"]["std_acc"])
#         amb_step._steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=cfg["ambulance"]["std_steer"])

#     control = agent.run_step()
#     control.manual_gear_shift = False

#     acc = control.throttle + amb_step._acc_noise.sample()
#     if acc >= 0:
#         control.throttle = max(0.0, min(acc, 1.0))
#         control.brake = 0.0
#     else:
#         control.throttle = 0.0
#         control.brake = min(abs(acc), 1.0)

#     control.steer = max(-1.0, min(control.steer + amb_step._steer_noise.sample(), 1.0))

#     actor.apply_control(control)


# def main():
#     cfg = load_config(exp="exp1")
#     print(f"Starting project: {cfg['project']['name']}")

#     client = Client(cfg)
#     ego = Ego(client.world, cfg)   
#     ped = ped_init(client.world, cfg)
#     amb, amb_agent = ambulance_init(client.world, cfg)
#     amb_agent._proximity_threshold = cfg["ambulance"]["proximity"]
#     vehicles = parked_vehicles_init(client.world, cfg)

#     set_all_lights_green(client.world)

#     start_tick = cfg['mpc']['sim_start'] * (1 / cfg['carla']['dt'])
#     end_tick = cfg['mpc']['sim_end'] * (1 / cfg['carla']['dt'])

#     tick = 0

#     try:
#         while True:
#             client.tick()

#             tick += 1

#             # before experiment starts, ego accelerates gradually and amb applies full throttle to target speed
#             # both proceed to a preset destination
#             if tick < start_tick:

#                 ego.step(acc=0.4)
#                 full_throttle_step(amb_agent, amb, cfg["ambulance"]["target_speed"])
            
#             # MPC framework
#             if start_tick <= tick <= end_tick:
                
#                 ped_step(ped, cfg)
#                 amb_step(amb_agent, amb, cfg)

#                 result = build_and_solve_mpc(ego, ped, amb, cfg)

#                 if result["status"]:
#                     ego.apply_control(result["control"])
#                 else:
#                     ego.step()
#                     print(f"MPC failed at tick {tick_count}, fallback to autopilot")
                
#             if tick >= end_tick:
#                 print("End of simulation")
#                 break

#             # print(f"tick: {tick} | Ego: {get_speed(ego.actor):.1f} km/h | Amb: {get_speed(amb):.1f} km/h | ped: {get_speed(ped):.1f} km/h | ")

#     finally:
#         client.quit(destroy=True)

    
# if __name__ == "__main__":
#     main()















import sys, os
import carla
import random
import math

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
                print(f"tick: {tick}")
                ego.step()


                # result = build_and_solve_mpc(ego, ped, amb, cfg)

                # if result["status"]:
                #     ego.apply_control(result["control"])
                # else:
                #     ego.step()
                #     print(f"MPC failed at tick {tick}, fallback to autopilot")

            else:
                print("End of simulation")
                break

    finally:
        client.quit(destroy=True)


if __name__ == "__main__":
    main()