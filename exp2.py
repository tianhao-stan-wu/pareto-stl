import sys, os
import carla
import random

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/CARLA_0.9.15/PythonAPI/carla")
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent

from src.config import load_config
from src.carla_client import Client
from src.ego import Ego


def get_speed(actor) -> float:
    """Return vehicle speed in km/h."""
    v = actor.get_velocity()
    return 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5


def boost_cyclist_physics(cyclist):
    """Override cyclist physics for fast acceleration."""
    physics = cyclist.get_physics_control()
    physics.mass = 30  # lighter = faster acceleration

    # boost torque curve
    physics.torque_curve = [
        carla.Vector2D(0, 600),     # high torque from 0 RPM
        carla.Vector2D(5000, 600),  # sustained through rev range
    ]

    physics.max_rpm = 10000
    physics.damping_rate_zero_throttle_clutch_engaged = 0.0
    physics.damping_rate_full_throttle = 0.0

    cyclist.apply_physics_control(physics)


def cyclist_init(world: carla.World, cfg):
    """Spawn a pedestrian and apply control with speed sampled from N(mean, std)."""
    bp_lib = world.get_blueprint_library()
    cyclist_bp = bp_lib.find('vehicle.bh.crossbike')

    cfg = cfg["cyclist"]
    loc = cfg["spawn"]["location"]
    rot = cfg["spawn"]["rotation"]
    spawn = carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
        )

    cyclist = world.try_spawn_actor(cyclist_bp, spawn)
    if cyclist is None:
        raise RuntimeError("Failed to spawn pedestrian. Location may be blocked.")

    boost_cyclist_physics(cyclist)

    world.tick()  # let server register the spawn

    agent = BehaviorAgent(cyclist, behavior=cfg.get("behavior", "normal"))
    # agent.ignore_traffic_lights(True)
    # agent.ignore_stop_signs(True)

    dest = cfg["destination"]["location"]
    agent.set_destination(carla.Location(x=dest["x"], y=dest["y"], z=dest["z"]))
    agent.set_target_speed(cfg.get("target_speed", 20))


    return cyclist, agent


def parked_vehicles_init(world: carla.World, cfg: dict):
    """Spawn two vehicles queued at a stop sign from config."""
    cfg = cfg["parked_vehicles"]
    loc = cfg["spawn"]["location"]
    rot = cfg["spawn"]["rotation"]
    gap = cfg.get("gap", 6.0)  # meters between vehicles

    blueprints = [
        world.get_blueprint_library().find(cfg["blueprints"][0]),
        world.get_blueprint_library().find(cfg["blueprints"][1]),
    ]

    # first vehicle at specified location
    t1 = carla.Transform(
        carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    # second vehicle placed behind the first using the facing direction
    forward = t1.get_forward_vector()
    t2 = carla.Transform(
        carla.Location(
            x=loc["x"] - forward.x * gap,
            y=loc["y"] - forward.y * gap,
            z=loc["z"]
        ),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    vehicles = []
    for bp, t in zip(blueprints, [t1, t2]):
        v = world.try_spawn_actor(bp, t)
        if v is None:
            raise RuntimeError(f"Failed to spawn {bp.id} at {t.location}")
        # v.apply_control(carla.VehicleControl(brake=1.0))  # keep stationary
        vehicles.append(v)

    world.tick()
    return vehicles


def pursuit_vehicle_init(world: carla.World, cfg: dict):
    """Spawn a fast pursuit vehicle and set destination from config."""
    bp_lib = world.get_blueprint_library()
    cfg = cfg["pursuit_vehicle"]
    veh_bp = bp_lib.find(cfg["blueprint"])

    loc = cfg["spawn"]["location"]
    rot = cfg["spawn"]["rotation"]
    spawn = carla.Transform(
        carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    vehicle = world.try_spawn_actor(veh_bp, spawn)
    if vehicle is None:
        raise RuntimeError("Failed to spawn pursuit vehicle. Location may be blocked.")

    world.tick()

    agent = BehaviorAgent(vehicle, behavior=cfg.get("behavior", "aggressive"))
    agent.set_target_speed(cfg.get("target_speed", 80))
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)

    dest = cfg["destination"]["location"]
    agent.set_destination(carla.Location(x=dest["x"], y=dest["y"], z=dest["z"]))

    return vehicle, agent


def full_throttle_step(agent, actor, target_speed: float):
    """Apply max throttle until target speed, then hand back to agent."""
    control = agent.run_step()
    control.manual_gear_shift = False

    if get_speed(actor) < target_speed:
        control.throttle = 1.0
        control.brake = 0.0

    actor.apply_control(control)


def main():
    cfg = load_config(exp="exp2")
    print(f"Starting project: {cfg['project']['name']}")

    client = Client(cfg)
    ego = Ego(client.world, cfg)   
    vehicles = parked_vehicles_init(client.world, cfg)
    cyclist, cyc_agent = cyclist_init(client.world, cfg)
    pursuit, pur_agent = pursuit_vehicle_init(client.world, cfg)
    
    # pursuit_wait_ticks = int(2.0 * 20)  # 2 seconds worth of ticks
    # tick_count = 0

    # # your pipeline here
    try:
        while True:
            client.tick()

            # tick_count += 1

            # if tick_count >= pursuit_wait_ticks:
                

            ego.step()

            # full_throttle_step(ego.agent, ego.actor, cfg["ego_vehicle"]["target_speed"])
            full_throttle_step(cyc_agent, cyclist, cfg["cyclist"]["target_speed"])
            full_throttle_step(pur_agent, pursuit, cfg["pursuit_vehicle"]["target_speed"])

            # if ego.agent:
            #     ego.step()
            # else:
            #     ego.apply_control(my_control)

            control = cyc_agent.run_step()
            # cyclist.apply_control(control)

            if cyc_agent.done():
                print("The target has been reached, stopping the simulation")
                break

    finally:
        client.quit(destroy=True)

    
if __name__ == "__main__":
    main()