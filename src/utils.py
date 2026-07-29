import argparse
import carla
import random
import numpy as np
import math
import json

import shutil
from datetime import datetime
from pathlib import Path


# ------------------------------------------------------------------
# logging
# ------------------------------------------------------------------

def setup_logging(cfg):
    """Create log directory, save config copy, return paths."""
    name = cfg["project"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    type = cfg["mpc"]["type"]
    density = cfg["mpc"]["density"]

    if type != "pareto":
        log_dir = Path(f"./logs/{name}/{type}/{timestamp}")
    else:
        log_dir = Path(f"./logs/{name}/{type}_d_{density}/{timestamp}")

    img_dir = log_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    # save config copy
    shutil.copy(f"configs/{name}.yaml", f"{log_dir}/{name}.yaml")

    print(f"Logging to {log_dir}")
    return log_dir, img_dir


def setup_camera(world, cfg):
    """Attach a persistent camera to the spectator."""
    import queue

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1920")
    bp.set_attribute("image_size_y", "1080")
    bp.set_attribute("fov", "90")

    loc = cfg["spectator"]["location"]
    rot = cfg["spectator"]["rotation"]
    spectator = carla.Transform(
        carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
        carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
    )

    camera = world.spawn_actor(bp, spectator)

    img_queue = queue.Queue()

    return camera, img_queue


def save_frame(img_queue, img_dir, tick, timeout=1.0):
    """Save latest frame from persistent camera."""
    import queue as q
    try:
        image = img_queue.get(timeout=timeout)
        image.save_to_disk(str(img_dir / f"tick_{tick:05d}.png"))
    except q.Empty:
        print(f"Warning: no image at tick {tick}")


def save_stats(build_times, solve_times, num_constraints, num_variables, log_dir):
    """Save timing and problem size stats to stats.json."""
    stats = {
        "build_times": {
            "avg": sum(build_times) / len(build_times),
            "min": min(build_times),
            "max": max(build_times),
            "n": len(build_times),
            "values": build_times,
        },
        "solve_times": {
            "avg": sum(solve_times) / len(solve_times),
            "min": min(solve_times),
            "max": max(solve_times),
            "n": len(solve_times),
            "values": solve_times,
        },
        "num_constraints": int(num_constraints) if num_constraints is not None else None,
        "num_variables": int(num_variables) if num_variables is not None else None,
    }

    path = Path(log_dir) / "complexity.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {path}")
    print(f"  Build: avg={stats['build_times']['avg']:.4f}s, "
          f"min={stats['build_times']['min']:.4f}s, "
          f"max={stats['build_times']['max']:.4f}s")
    print(f"  Solve: avg={stats['solve_times']['avg']:.4f}s, "
          f"min={stats['solve_times']['min']:.4f}s, "
          f"max={stats['solve_times']['max']:.4f}s")


def save_trajectories(agent_trajectories, log_dir):
    """Save agent locations to locations.json."""
    path = Path(log_dir) / "trajectory.json"
    with open(path, "w") as f:
        json.dump(agent_trajectories, f, indent=2)
    print(f"Locations saved to {path}")


def imgs_to_video(log_dir, fps=10):
    """Compile all images in log_dir/imgs into a video."""
    import subprocess
    from pathlib import Path

    img_dir = Path(log_dir) / "imgs"
    output_path = Path(log_dir) / "video.mp4"

    if not img_dir.exists():
        print(f"No imgs folder found in {log_dir}")
        return

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(img_dir / "tick_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path
    ], check=True)

    print(f"Video saved to {output_path}")            


# ------------------------------------------------------------------
# STL robustness
# ------------------------------------------------------------------

def compute_and_save_robustness(agent_locations, stl_cfg, agent_dims, dt, log_dir):
    """
    Compute traditional STL and spatio-temporal robustness for all 5 STL specs:
    (1) safe distance with pedestrian
    (2) safe distance with ambulance
    (3) avoid vehicle rectangle
    (4) avoid opposite lane rectangle
    (5) stop before intersection y-line
    """
    ego_locs = np.array(agent_locations["ego_vehicle"])
    ego_w = agent_dims["ego_vehicle"]["width"]
    ego_l = agent_dims["ego_vehicle"]["length"]
    T = len(ego_locs)

    min_max = {}
    spatio_temporal = {}

    # ------------------------------------------------------------------
    # (1) (2) Safe distance with pedestrian and ambulance
    # ------------------------------------------------------------------
    for key in ["pedestrian", "ambulance"]:
        if key not in agent_locations or key not in stl_cfg:
            continue

        d_safe = stl_cfg[key]
        agent_locs = np.array(agent_locations[key])
        T_k = min(T, len(agent_locs))

        dx = np.abs(ego_locs[:T_k, 0] - agent_locs[:T_k, 0])
        dy = np.abs(ego_locs[:T_k, 1] - agent_locs[:T_k, 1])

        if key in agent_dims:
            margin_x = ego_w / 2.0 + agent_dims[key]["width"] / 2.0 + d_safe
            margin_y = ego_l / 2.0 + agent_dims[key]["length"] / 2.0 + d_safe
        else:
            margin_x = ego_w / 2.0 + d_safe
            margin_y = ego_l / 2.0 + d_safe

        ro_x = np.maximum(dx - margin_x, -d_safe)
        ro_y = np.maximum(dy - margin_y, -d_safe)
        ro = np.maximum(ro_x, ro_y)

        min_idx = int(np.argmin(ro))
        min_max[key] = {"robustness": float(ro[min_idx])}

        violations = np.minimum(ro, 0.0)
        spatio_temporal[key] = {
            "robustness": float(np.sum(violations) * dt),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(T),
        }

    # ------------------------------------------------------------------
    # (3) Avoid vehicle rectangle
    # (4) Avoid opposite lane rectangle
    # ------------------------------------------------------------------
    for rect_key, label in [("vehicles", "vehicles_rect"), ("lane", "lane_rect")]:
        if rect_key not in stl_cfg:
            continue

        rect = stl_cfg[rect_key]
        x_min, x_max = rect["x_min"], rect["x_max"]
        y_min, y_max = rect["y_min"], rect["y_max"]

        dx_min = ego_locs[:T, 0] - x_min  # distance from left edge
        dx_max = x_max - ego_locs[:T, 0]  # distance from right edge
        dy_min = ego_locs[:T, 1] - y_min  # distance from bottom edge
        dy_max = y_max - ego_locs[:T, 1]  # distance from top edge

        inside_x = np.minimum(dx_min, dx_max)  # min distance to x-edge
        inside_y = np.minimum(dy_min, dy_max)  # min distance to y-edge
        penetration = np.minimum(inside_x, inside_y)  # min over both axes

        ro = -penetration

        min_idx = int(np.argmin(ro))
        min_max[label] = {"robustness": float(ro[min_idx])}

        violations = np.minimum(ro, 0.0)
        spatio_temporal[label] = {
            "robustness": float(np.sum(violations) * dt),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(T),
        }

    # ------------------------------------------------------------------
    # (5) Stop before intersection y-line
    # ------------------------------------------------------------------
    if "intersection_y" in stl_cfg:
        y_line = stl_cfg["intersection_y"]

        # robustness: how far ego is from crossing the line
        # positive = ego is above y_line (safe), negative = crossed
        ro = ego_locs[:T, 1] - y_line

        min_idx = int(np.argmin(ro))
        min_max["intersection"] = {"robustness": float(ro[min_idx])}

        violations = np.minimum(ro, 0.0)
        spatio_temporal["intersection"] = {
            "robustness": float(np.sum(violations) * dt),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(T),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "min-max": min_max,
        "spatio-temporal": spatio_temporal,
    }

    path = Path(log_dir) / "robustness.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRobustness saved to {path}")

    summary_path = Path(log_dir) / "robustness_summary.txt"
    with open(summary_path, "w") as f:
        f.write("STL Robustness Summary\n")
        f.write(f"  {'spec':<20} {'min-max':>10} {'spatio-temp':>12} {'violations':>12}\n")
        f.write(f"  {'-'*56}\n")
        for key in min_max:
            rho_mm = min_max[key]["robustness"]
            rho_st = spatio_temporal[key]["robustness"]
            n_viol = spatio_temporal[key]["num_violations"]
            total = spatio_temporal[key]["total_steps"]
            status = "✓" if rho_mm >= 0 else "✗"
            line = f"  {key:<20} {rho_mm:>+10.3f} {rho_st:>+12.3f} {n_viol:>5}/{total:<5} {status}\n"
            f.write(line)

    print(f"Summary saved to {summary_path}")


# ------------------------------------------------------------------
# mpc
# ------------------------------------------------------------------
COLORS = {
    "red":     carla.Color(10, 0, 0),
    "blue":    carla.Color(0, 0, 10),
    "green":   carla.Color(0, 10, 0),
    "yellow":  carla.Color(80, 80, 0),
    "magenta": carla.Color(80, 0, 80),
    "cyan":    carla.Color(0, 80, 80),
    "orange":  carla.Color(80, 40, 0),
    "white":   carla.Color(80, 80, 80),
}

MAP = {
    "ego": "blue",
    "ambulance": "magenta",
    "pedestrian": "red",
    "parked_v1": "yellow",
    "parked_v2": "cyan"
}


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


# opposite conventions between CARLA steer and bicycle model beta, negate to fix
def bicycle_to_carla(u, acc_min, acc_max, beta_min, beta_max):
    a, beta = u
    a = max(acc_min, min(a, acc_max))
    beta = max(beta_min, min(beta, beta_max))

    control = carla.VehicleControl()
    control.manual_gear_shift = False

    if a >= 0:
        control.throttle = min(a / acc_max, 1.0)
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = min(abs(a) / abs(acc_min), 1.0)

    steer_angle = math.degrees(math.atan(2.0 * math.tan(beta)))
    max_steer = math.degrees(math.atan(2.0 * math.tan(beta_max)))
    control.steer = max(-1.0, min(steer_angle / max_steer, 1.0)) 

    return control


def carla_to_bicycle(control, acc_min, acc_max, beta_min, beta_max):
    if control.throttle > 0:
        a = control.throttle * acc_max
    else:
        a = -control.brake * abs(acc_min)

    max_steer_rad = math.radians(70.0)
    steer_angle = control.steer * max_steer_rad 
    beta = math.atan(0.5 * math.tan(steer_angle))

    a = max(acc_min, min(a, acc_max))
    beta = max(beta_min, min(beta, beta_max))

    return a, beta


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
    else:
        trajs = trajs[:10, :, :]   # visualize 10 trajectories

    debug = world.debug
    S, N1, _ = trajs.shape

    for s in range(S):
        for k in range(N1 - 1):
            start = carla.Location(x=float(trajs[s, k, 0]),   y=float(trajs[s, k, 1]),   z=0.5)
            end   = carla.Location(x=float(trajs[s, k+1, 0]), y=float(trajs[s, k+1, 1]), z=0.5)
            debug.draw_line(start, end, thickness=size, color=color, life_time=life_time)


def draw_rectangle_boundary(
    world: carla.World,
    rect,
    z: float = 0.2,
    color: carla.Color = carla.Color(255, 0, 0),
    thickness: float = 0.05,
    life_time: float = 1,
):
    """
    Draw the boundary of an axis-aligned rectangle in CARLA.

    Parameters
    ----------
    world : carla.World
    rect : [x_min, x_max, y_min, y_max]
    z : float
        Height at which to draw the rectangle.
    color : carla.Color
    thickness : float
    life_time : float
        0.0 means persistent until the simulator resets.
    """

    x_min, x_max, y_min, y_max = rect

    # Rectangle corners
    p1 = carla.Location(x=x_min, y=y_min, z=z)
    p2 = carla.Location(x=x_max, y=y_min, z=z)
    p3 = carla.Location(x=x_max, y=y_max, z=z)
    p4 = carla.Location(x=x_min, y=y_max, z=z)

    debug = world.debug

    debug.draw_line(p1, p2,
                    thickness=thickness,
                    color=color,
                    life_time=life_time)

    debug.draw_line(p2, p3,
                    thickness=thickness,
                    color=color,
                    life_time=life_time)

    debug.draw_line(p3, p4,
                    thickness=thickness,
                    color=color,
                    life_time=life_time)

    debug.draw_line(p4, p1,
                    thickness=thickness,
                    color=color,
                    life_time=life_time)

# ------------------------------------------------------------------
# carla world (callable in main)
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
    parser.add_argument("--blueprint", type=str, default="vehicle.audi.a2",
                    help="blueprint for steer angle check (default: vehicle.tesla.model3)")

    parser.add_argument("--log_dir", type=str, help="path to logs folder")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("-v", action="store_true", help="save video")

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

    if args.v:
        imgs_to_video(args.log_dir, args.fps)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)