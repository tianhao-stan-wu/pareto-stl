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
    exp = cfg["project"]["exp"]
    shutil.copy(f"configs/{exp}.yaml", f"{log_dir}/{exp}.yaml")

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


def save_stats(
    feas_build_times, feas_solve_times, feas_num_constraints, feas_num_variables,
    pareto_build_times, pareto_solve_times, pareto_solve_all_times,
    pareto_num_constraints, pareto_num_variables,
    total_tick, n_infeas, n_infeas_resolved,
    log_dir,
):
    def summarise(times):
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "n": 0, "values": []}
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "n": len(times),
            "values": times,
        }

    infeas_rate = n_infeas / total_tick if total_tick > 0 else 0
    completion_rate = (total_tick - n_infeas + n_infeas_resolved) / total_tick if total_tick > 0 else 0

    stats = {
        "simulation": {
            "simulation_steps": total_tick,
            "n_infeas": n_infeas,
            "infeas_rate": round(infeas_rate, 4),
            "n_infeas_resolved": n_infeas_resolved,
            "completion_rate": round(completion_rate, 4),
        },
        "feasibility": {
            "build_times": summarise(feas_build_times),
            "solve_times": summarise(feas_solve_times),
            "num_constraints": int(feas_num_constraints) if feas_num_constraints is not None else None,
            "num_variables": int(feas_num_variables) if feas_num_variables is not None else None,
        },
        "pareto": {
            "build_times": summarise(pareto_build_times),
            "solve_times": summarise(pareto_solve_times),
            "solve_all_times": summarise(pareto_solve_all_times),
            "num_constraints": int(pareto_num_constraints) if pareto_num_constraints is not None else None,
            "num_variables": int(pareto_num_variables) if pareto_num_variables is not None else None,
        },
    }

    path = Path(log_dir) / "solve_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {path}")
    print(f"  Simulation: {total_tick} steps, "
          f"{n_infeas} infeasible ({infeas_rate:.1%}), "
          f"{n_infeas_resolved} resolved, "
          f"completion {completion_rate:.1%}")
    s = stats["feasibility"]
    print(f"  Feasibility:  build {s['build_times']['avg']:.4f}s  "
          f"solve {s['solve_times']['avg']:.4f}s  "
          f"({s['solve_times']['n']} calls)")
    s = stats["pareto"]
    if s["solve_times"]["n"] > 0:
        print(f"  Pareto (per MILP): build {s['build_times']['avg']:.4f}s  "
              f"solve {s['solve_times']['avg']:.4f}s  "
              f"({s['solve_times']['n']} calls)")
        print(f"  Pareto (per tick): sweep {s['solve_all_times']['avg']:.4f}s  "
              f"max {s['solve_all_times']['max']:.4f}s  "
              f"({s['solve_all_times']['n']} ticks)")


def save_pareto_log(pareto_records, log_dir):
    path = Path(log_dir) / "pareto.json"
    with open(path, "w") as f:
        json.dump(pareto_records, f, indent=2)
    print(f"Pareto log saved to {path} ({len(pareto_records)} ticks)")

    
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
def save_robustness_exp1(agent_locations, stl_cfg, dt, log_dir):
    """
    Compute and save STL robustness for three specifications:
      (1) safe distance with pedestrian  (d_safe box)
      (2) safe distance with ambulance   (d_safe box)
      (3) stay in lane                   (x_min <= px <= x_max when y_min <= py <= y_max)
    """
    ego = np.array(agent_locations["ego_vehicle"])
    T = len(ego)

    results = {}

    # (1) (2) safe distance: robustness = max(|dx|, |dy|) - d_safe
    #         positive = outside the box, negative = inside
    for key in ["pedestrian", "ambulance"]:
        if key not in agent_locations or key not in stl_cfg:
            continue

        agent = np.array(agent_locations[key])
        d = float(stl_cfg[key])
        Tk = min(T, len(agent))

        dx = np.abs(ego[:Tk, 0] - agent[:Tk, 0])
        dy = np.abs(ego[:Tk, 1] - agent[:Tk, 1])
        linf = np.maximum(dx, dy)
        ro = linf - d

        results[key] = {
            "min_robustness": float(np.min(ro)),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(Tk),
            "per_step": ro.tolist(),
        }

    # (3) stay in lane: robustness = min distance to lane boundary (negative = outside)
    if "lane" in stl_cfg:
        lane = stl_cfg["lane"]
        x_min, x_max = float(lane["x_min"]), float(lane["x_max"])
        y_min, y_max = float(lane["y_min"]), float(lane["y_max"])

        in_y = (ego[:T, 1] >= y_min) & (ego[:T, 1] <= y_max)
        dx_left = ego[:T, 0] - x_min
        dx_right = x_max - ego[:T, 0]

        ro = np.zeros(T)
        for k in range(T):
            if not in_y[k]:
                ro[k] = 1.0  # outside y range, implication vacuously true
            else:
                ro[k] = min(dx_left[k], dx_right[k])  # negative = outside lane

        results["lane"] = {
            "min_robustness": float(np.min(ro)),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(T),
            "per_step": ro.tolist(),
        }

    # save
    path = Path(log_dir) / "robustness_summary.txt"
    with open(path, "w") as f:
        f.write("STL Robustness Summary\n\n")

        f.write(f"{'spec':<15} {'min_rho':>10} {'violations':>12}\n")
        f.write(f"{'-'*40}\n")
        for key, r in results.items():
            status = "pass" if r["min_robustness"] >= 0 else "FAIL"
            f.write(f"{key:<15} {r['min_robustness']:>+10.3f} "
                    f"{r['num_violations']:>5}/{r['total_steps']:<5} {status}\n")

        f.write(f"\n\nPer-step robustness\n\n")
        keys = list(results.keys())
        header = f"{'step':>6}" + "".join(f"{k:>12}" for k in keys)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        max_t = max(len(results[k]["per_step"]) for k in keys)
        for t in range(max_t):
            row = f"{t:>6}"
            for k in keys:
                steps = results[k]["per_step"]
                row += f"{steps[t]:>+12.3f}" if t < len(steps) else f"{'':>12}"
            f.write(row + "\n")

    print(f"\nRobustness saved to {path}")
    for key, r in results.items():
        status = "pass" if r["min_robustness"] >= 0 else "FAIL"
        print(f"  {key:<15} min_rho={r['min_robustness']:>+.3f}  "
              f"violations={r['num_violations']}/{r['total_steps']}  {status}")


def save_robustness_exp2(agent_locations, stl_cfg, dt, log_dir):
    """
    Compute and save STL robustness for exp2 specifications:
      (1) safe distance with leader   (L-inf, d_safe_y along travel axis)
      (2) safe distance with follower
      (3) safe distance with left vehicle
      (4) stay in lane                (y_min relaxable)
      (5) crash scene box             (inflated rectangle)
    """
    from pathlib import Path

    ego = np.array(agent_locations["ego_vehicle"])
    T = len(ego)

    results = {}

    # (1)(2)(3) safe distance per vehicle — L-inf with separate x/y margins
    d_safe_x = float(stl_cfg["d_safe_x"])
    d_safe_y = float(stl_cfg["d_safe_y"])

    for key in ["leader", "follower", "left_vehicle"]:
        if key not in agent_locations:
            continue

        agent = np.array(agent_locations[key])
        Tk = min(T, len(agent))

        dx = np.abs(ego[:Tk, 0] - agent[:Tk, 0])
        dy = np.abs(ego[:Tk, 1] - agent[:Tk, 1])

        ro_x = dx - d_safe_x
        ro_y = dy - d_safe_y
        ro = np.maximum(ro_x, ro_y)   # positive = outside keep-out box

        results[key] = {
            "min_robustness": float(np.min(ro)),
            "num_violations": int(np.sum(ro < 0)),
            "total_steps": int(Tk),
            "per_step": ro.tolist(),
        }

    # (4) stay in lane — robustness = min(px - x_min, x_max - px, py - y_min, y_max - py)
    #     only y_min is relaxable, but robustness is computed for all four bounds
    x_min = float(stl_cfg["x_min"])
    x_max = float(stl_cfg["x_max"])
    y_min = float(stl_cfg["y_min"])
    y_max = float(stl_cfg["y_max"])

    ro_lane = np.minimum(
        np.minimum(ego[:T, 0] - x_min, x_max - ego[:T, 0]),
        np.minimum(ego[:T, 1] - y_min, y_max - ego[:T, 1]),
    )

    results["lane"] = {
        "min_robustness": float(np.min(ro_lane)),
        "num_violations": int(np.sum(ro_lane < 0)),
        "total_steps": int(T),
        "per_step": ro_lane.tolist(),
    }

    # (5) crash scene box — robustness = distance from inflated box boundary
    #     positive = outside, negative = inside
    d_crash = float(stl_cfg["d_crash"])
    cx_min = float(stl_cfg["crash_x_min"]) - d_crash
    cx_max = float(stl_cfg["crash_x_max"]) + d_crash
    cy_min = float(stl_cfg["crash_y_min"]) - d_crash
    cy_max = float(stl_cfg["crash_y_max"]) + d_crash

    inside_x = np.minimum(ego[:T, 0] - cx_min, cx_max - ego[:T, 0])
    inside_y = np.minimum(ego[:T, 1] - cy_min, cy_max - ego[:T, 1])
    penetration = np.minimum(inside_x, inside_y)
    ro_crash = -penetration   # positive = outside box

    results["crash"] = {
        "min_robustness": float(np.min(ro_crash)),
        "num_violations": int(np.sum(ro_crash < 0)),
        "total_steps": int(T),
        "per_step": ro_crash.tolist(),
    }

    # save
    path = Path(log_dir) / "robustness_summary.txt"
    with open(path, "w") as f:
        f.write("STL Robustness Summary (Exp 2)\n\n")

        f.write(f"{'spec':<20} {'min_rho':>10} {'violations':>12}\n")
        f.write(f"{'-'*45}\n")
        for key, r in results.items():
            status = "pass" if r["min_robustness"] >= 0 else "FAIL"
            f.write(f"{key:<20} {r['min_robustness']:>+10.3f} "
                    f"{r['num_violations']:>5}/{r['total_steps']:<5} {status}\n")

        f.write(f"\n\nPer-step robustness\n\n")
        keys = list(results.keys())
        header = f"{'step':>6}" + "".join(f"{k:>14}" for k in keys)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        max_t = max(len(results[k]["per_step"]) for k in keys)
        for t in range(max_t):
            row = f"{t:>6}"
            for k in keys:
                steps = results[k]["per_step"]
                row += f"{steps[t]:>+14.3f}" if t < len(steps) else f"{'':>14}"
            f.write(row + "\n")

    print(f"\nRobustness saved to {path}")
    for key, r in results.items():
        status = "pass" if r["min_robustness"] >= 0 else "FAIL"
        print(f"  {key:<20} min_rho={r['min_robustness']:>+.3f}  "
              f"violations={r['num_violations']}/{r['total_steps']}  {status}")


def save_theorem1_results(records, summary, log_dir):
    """Save per-tick and aggregate Theorem 1 verification results."""
    from pathlib import Path
    import json

    path = Path(log_dir) / "theorem1_validation.json"

    output = {
        "summary": summary,
        "records": records,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTheorem 1 validation saved to {path}")

    # Print exactly what summarise_theorem1() produced.
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
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


# ------------------------------------------------------------------
# other helper functions
# ------------------------------------------------------------------

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


def print_and_draw_traffic_light_ids(world, duration=10.0, z_offset=2.0):
    """
    Print all traffic-light IDs and locations to the terminal,
    and draw each ID above the corresponding traffic light in CARLA.

    Args:
        world: carla.World
        duration: How long the ID label remains visible, in seconds.
        z_offset: Height above the traffic light for the label.
    """
    traffic_lights = world.get_actors().filter("traffic.traffic_light*")

    print(f"\nFound {len(traffic_lights)} traffic lights:")

    for tl in traffic_lights:
        loc = tl.get_location()

        print(
            f"ID: {tl.id:4d} | "
            f"Location: "
            f"({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f})"
        )

        # Draw ID above traffic light
        label_loc = carla.Location(
            x=loc.x,
            y=loc.y,
            z=loc.z + z_offset,
        )

        world.debug.draw_string(
            label_loc,
            str(tl.id),
            draw_shadow=True,
            color=carla.Color(255, 0, 0),
            life_time=duration,
            persistent_lines=False,
        )

    print(f"Traffic-light IDs drawn for {duration:.1f} seconds.\n")


def set_traffic_lights_by_location(
    world,
    green_locations,
    red_locations,
    tolerance=5,
    freeze=True,
):
    """
    Find traffic lights by world location and set their states.

    Args:
        world: carla.World
        green_locations: list of (x, y, z) tuples
        red_locations: list of (x, y, z) tuples
        tolerance: matching distance in meters
        freeze: freeze traffic lights after setting the state

    Returns:
        dict with matched traffic-light IDs.
    """

    traffic_lights = world.get_actors().filter("traffic.traffic_light*")

    def find_closest_light(target):
        target_loc = carla.Location(*target)

        best_tl = None
        best_dist = float("inf")

        for tl in traffic_lights:
            dist = tl.get_location().distance(target_loc)

            if dist < best_dist:
                best_dist = dist
                best_tl = tl

        if best_tl is None or best_dist > tolerance:
            return None

        return best_tl

    result = {
        "green": [],
        "red": [],
    }

    # Set green lights
    for location in green_locations:
        tl = find_closest_light(location)

        if tl is None:
            print(f"No traffic light found near {location}")
            continue

        if freeze:
            tl.freeze(True)

        tl.set_state(carla.TrafficLightState.Green)

        result["green"].append(tl.id)

        print(
            f"GREEN: ID={tl.id}, "
            f"location={tl.get_location()}"
        )

    # Set red lights
    for location in red_locations:
        tl = find_closest_light(location)

        if tl is None:
            print(f"No traffic light found near {location}")
            continue

        if freeze:
            tl.freeze(True)

        tl.set_state(carla.TrafficLightState.Red)

        result["red"].append(tl.id)

        print(
            f"RED: ID={tl.id}, "
            f"location={tl.get_location()}"
        )

    return result


def draw_sample_traj(world, trajs, color=None, size=0.05, life_time=1.0, z=0.5):
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
            start = carla.Location(x=float(trajs[s, k, 0]),   y=float(trajs[s, k, 1]),   z=z)
            end   = carla.Location(x=float(trajs[s, k+1, 0]), y=float(trajs[s, k+1, 1]), z=z)
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
    parser.add_argument("-tl", action="store_true", help="draw traffic light id in simuator")

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

    if args.tl:
        print_and_draw_traffic_light_ids(world, duration=120)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)