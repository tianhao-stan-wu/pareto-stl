import argparse
import carla
import random


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


def set_all_lights_green(world, green_time: float = 30.0):
    for light in world.get_actors().filter("traffic.traffic_light*"):
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(green_time)
        light.set_red_time(0.0)
        light.set_yellow_time(0.0)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", action="store_true", help="get spectator transform (YAML format)")
    parser.add_argument("-s", action="store_true", help="set spectator transform")
    parser.add_argument("-w", action="store_true", help="draw nearby waypoints")
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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)