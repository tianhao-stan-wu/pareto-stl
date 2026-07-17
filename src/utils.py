import argparse
import carla
import random
import numpy as np
import math


# ------------------------------------------------------------------
# import functions
# ------------------------------------------------------------------

class SmoothNoise:
    """Ornstein-Uhlenbeck process for smooth random noise."""

    def __init__(self, mean=0.0, theta=0.3, sigma=0.1):
        self.mean = mean
        self.theta = theta
        self.sigma = sigma
        self.value = 0.0

    def sample(self):
        self.value += self.theta * (self.mean - self.value) + self.sigma * random.gauss(0, 1)
        return self.value


def dist_2d(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)


def print_distances(agent1, agent2):
    agent1_loc = agent1.get_transform().location
    agent2_loc = agent2.get_transform().location

    dist = dist_2d(agent1_loc, agent2_loc)
    print(f"  dist between {agent1.key} {agent2.key}: {dist:.1f}")


def set_all_lights_green(world, green_time: float = 30.0):
    for light in world.get_actors().filter("traffic.traffic_light*"):
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(green_time)
        light.set_red_time(0.0)
        light.set_yellow_time(0.0)


def draw_sample_traj(world, trajs, color=None, size=0.05, life_time=1.0):
    """
    Draw sample trajectories in CARLA.

    Parameters
    ----------
    world    : carla.World
    trajs    : ndarray (S, N+1, 2) or (N+1, 2)
    color    : carla.Color, default red
    size     : float, point size
    life_time: float, seconds to persist
    """
    
    if color is None:
        color = carla.Color(255, 0, 0)

    trajs = np.asarray(trajs)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]  # (N+1, 2) → (1, N+1, 2)

    debug = world.debug
    S, N1, _ = trajs.shape

    for s in range(S):
        for k in range(N1 - 1):
            start = carla.Location(x=float(trajs[s, k, 0]),   y=float(trajs[s, k, 1]),   z=0.5)
            end   = carla.Location(x=float(trajs[s, k+1, 0]), y=float(trajs[s, k+1, 1]), z=0.5)
            debug.draw_line(start, end, thickness=size, color=color, life_time=life_time)


# ------------------------------------------------------------------
# util functions (main)
# ------------------------------------------------------------------

def get_spectator_transform(world):
    """Print spectator transform in YAML-ready format."""
    t = world.get_spectator().get_transform()
    loc, rot = t.location, t.rotation
    print(
        f"location: {{x: {loc.x:.3f}, y: {loc.y:.3f}, z: {loc.z:.3f}}}\n"
        f"rotation: {{pitch: {rot.pitch:.3f}, yaw: {rot.yaw:.3f}, roll: {rot.roll:.3f}}}"
    )
    return t


def set_spectator_transform(world, transform):
    world.get_spectator().set_transform(transform)


def draw_waypoints(world, ref, min_r=10.0, max_r=30.0, sampling=1.0, min_sep=5.0, life_time=60.0):
    """Draw and print waypoints within a distance band from ref location."""
    debug = world.debug
    waypoints = world.get_map().generate_waypoints(sampling)
    selected = []

    for wp in waypoints:
        loc = wp.transform.location
        dist = loc.distance(ref)

        if not (min_r <= dist <= max_r):
            continue
        if any(loc.distance(s.transform.location) < min_sep for s in selected):
            continue

        selected.append(wp)
        debug.draw_point(loc + carla.Location(z=0.2), size=0.12,
                         color=carla.Color(255, 0, 0), life_time=life_time)
        debug.draw_string(loc + carla.Location(z=1.0), str(wp.id),
                          color=carla.Color(0, 255, 0), life_time=life_time)
        print(f"id={wp.id}, road={wp.road_id}, lane={wp.lane_id}, "
              f"junction={wp.is_junction}, dist={dist:.2f}, "
              f"loc=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

    print(f"\nFound {len(selected)} waypoints in [{min_r}, {max_r}] m")
    return selected


def get_max_steer_angle(actor):
    """Print and return the maximum steer angle (degrees) for each wheel."""
    physics = actor.get_physics_control()
    for i, wheel in enumerate(physics.wheels):
        print(f"Wheel {i}: max_steer_angle = {wheel.max_steer_angle:.1f}°")
    # front wheels have the steer angle, rears are typically 0
    max_angle = max(w.max_steer_angle for w in physics.wheels)
    print(f"Vehicle max steer angle: {max_angle:.1f}°")
    return max_angle


def get_lr_lf(actor):
    """Estimate lr and lf from wheel positions and center of mass."""
    physics = actor.get_physics_control()
    wheels = physics.wheels
    com = physics.center_of_mass

    # wheel positions are world coords (cm), but difference cancels that out
    front_x = (wheels[0].position.x + wheels[1].position.x) / 2.0
    rear_x = (wheels[2].position.x + wheels[3].position.x) / 2.0
    wheelbase = abs(front_x - rear_x) / 100.0  # cm to m

    # CoM.x is local frame (cm), offset from vehicle origin
    # positive = forward of origin
    com_offset = com.x / 100.0  # cm to m

    # split wheelbase around CoM
    lr = wheelbase / 2.0 + com_offset
    lf = wheelbase / 2.0 - com_offset

    print(f"Wheelbase: {wheelbase:.3f} m")
    print(f"CoM offset: {com_offset:.3f} m")
    print(f"lf: {lf:.3f} m")
    print(f"lr: {lr:.3f} m")

    return lr, lf


def get_dimensions(actor):
    """Get vehicle width and length from bounding box."""
    bbox = actor.bounding_box.extent
    width = bbox.y * 2
    length = bbox.x * 2
    height = bbox.z * 2

    print(f"Length: {length:.3f} m")
    print(f"Width:  {width:.3f} m")
    print(f"Height: {height:.3f} m")

    return length, width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", action="store_true", help="get spectator transform (YAML format)")
    parser.add_argument("-s", action="store_true", help="set spectator transform")
    parser.add_argument("-w", action="store_true", help="draw nearby waypoints")
    parser.add_argument("-a", action="store_true", help="get physical params of a vehicle")
    parser.add_argument("--blueprint", type=str, default="vehicle.ford.ambulance",
                    help="blueprint for steer angle check (default: vehicle.tesla.model3)")
    args = parser.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if args.g:
        get_spectator_transform(world)

    if args.s:
        transform = carla.Transform(
            carla.Location(x=18.915, y=28.195, z=41.810),
            carla.Rotation(pitch=-53.682, yaw=-116.643, roll=0.000)
        )
        set_spectator_transform(world, transform)

    if args.w:
        ref = carla.Location(x=0, y=0, z=2)
        draw_waypoints(world, ref, min_r=30.0, max_r=50.0, sampling=5.0, life_time=10.0)

    if args.a:
        bp = world.get_blueprint_library().find(args.blueprint)
        spawn = world.get_map().get_spawn_points()[0]
        actor = world.try_spawn_actor(bp, spawn)
        world.tick()
        get_max_steer_angle(actor)
        get_lr_lf(actor)
        get_dimensions(actor)
        actor.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)