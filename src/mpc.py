import carla
import numpy as np
import cvxpy as cp
import math
import random

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance
from src.utils import SmoothNoise, draw_sample_traj


def bicycle_to_carla(u, ego_acc_min, ego_acc_max, ego_beta_min, ego_beta_max, lr=1.5):
    """Convert bicycle model [a, beta] to CARLA VehicleControl."""
    a, beta = u

    a = max(ego_acc_min, min(a, ego_acc_max))
    beta = max(ego_beta_min, min(beta, ego_beta_max))

    control = carla.VehicleControl()
    control.manual_gear_shift = False

    if a >= 0:
        control.throttle = min(a / ego_acc_max, 1.0)
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = min(abs(a) / abs(ego_acc_min), 1.0)

    steer_angle = math.degrees(math.atan(2.0 * math.tan(beta)))
    max_steer = math.degrees(math.atan(2.0 * math.tan(ego_beta_max)))
    control.steer = max(-1.0, min(steer_angle / max_steer, 1.0))

    return control


def carla_to_bicycle(control, ego_acc_min, ego_acc_max, ego_beta_min, ego_beta_max):
    """Convert CARLA VehicleControl to bicycle model [a, beta]."""
    if control.throttle > 0:
        a = control.throttle * ego_acc_max
    else:
        a = -control.brake * abs(ego_acc_min)

    # max steer angle is 70 degrees in carla, checked in utils.py
    max_steer_rad = math.radians(70.0)
    steer_angle = control.steer * max_steer_rad
    # lr/(lf+lr) ~= 0.5
    beta = math.atan(0.5 * math.tan(steer_angle))

    a = max(ego_acc_min, min(a, ego_acc_max))
    beta = max(ego_beta_min, min(beta, ego_beta_max))

    return a, beta


def sample_ped_trajectories(ped, cfg, N: int, dt: float, S: int):
    """Sample S pedestrian trajectories with noisy speed and direction."""
    tf = ped.get_transform()
    px0, py0 = tf.location.x, tf.location.y
    yaw0 = math.radians(tf.rotation.yaw)

    mean_speed = cfg["pedestrian"]["mean_speed"]
    std_speed = cfg["pedestrian"]["std_speed"]
    std_dir = cfg["pedestrian"]["std_dir"]

    trajs = np.zeros((S, N + 1, 2))

    for s in range(S):
        px, py, yaw = px0, py0, yaw0
        trajs[s, 0] = [px, py]

        for k in range(N):
            yaw += random.gauss(0, std_dir)
            speed = max(0.0, random.gauss(mean_speed, std_speed))
            px += speed * math.cos(yaw) * dt
            py += speed * math.sin(yaw) * dt
            trajs[s, k + 1] = [px, py]

    return trajs  # (S, N+1, 2)


def sample_amb_trajectories(amb, cfg, N: int, dt: float, S: int):
    """Sample S ambulance trajectories with smooth noise."""
    tf = amb.get_transform()
    px0, py0 = tf.location.x, tf.location.y
    yaw0 = math.radians(tf.rotation.yaw)
    speed0 = amb.get_speed() / 3.6

    amb_cfg = cfg["ambulance"]
    lr = amb_cfg["lr"]

    control = amb.agent.run_step()
    base_throttle = control.throttle
    base_steer = control.steer
    max_steer_rad = math.radians(70.0)

    trajs = np.zeros((S, N + 1, 2))

    for s in range(S):
        px, py, yaw, speed = px0, py0, yaw0, speed0
        acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=amb_cfg.get("std_acc", 0.1))
        steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=amb_cfg.get("std_steer", 0.02))

        trajs[s, 0] = [px, py]

        for k in range(N):
            acc = base_throttle + acc_noise.sample()
            if acc >= 0:
                a = min(acc, 1.0) * amb_cfg["acc_max"]
            else:
                a = -min(abs(acc), 1.0) * abs(amb_cfg["acc_min"])

            steer = max(-1.0, min(base_steer + steer_noise.sample(), 1.0))
            beta = math.atan(0.5 * math.tan(steer * max_steer_rad))

            speed += a * dt
            speed = max(0.0, speed)
            yaw += (speed / lr) * beta * dt
            px += speed * math.cos(yaw) * dt
            py += speed * math.sin(yaw) * dt
            trajs[s, k + 1] = [px, py]

    return trajs  # (S, N+1, 2)


def build_and_solve_mpc(client, ego, ped, amb, cfg):
    # extract parameters
    T = cfg["mpc"]["T"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
    S = cfg["mpc"]["S"]

    ego_cfg = cfg["ego_vehicle"]
    ego_acc_min = ego_cfg["acc_min"]
    ego_acc_max = ego_cfg["acc_max"]
    ego_beta_min = ego_cfg["beta_min"]
    ego_beta_max = ego_cfg["beta_max"]
    ego_w = ego_cfg["width"]
    ego_l = ego_cfg["length"]
    ego_lr = ego_cfg["lr"]

    # set up model
    model = KinematicBicycle(lr=ego_lr, dt=dt)

    # get ego's current state
    tf = ego.get_transform()
    vel = ego.get_velocity()
    ego_init = np.array([
        tf.location.x,
        tf.location.y,
        math.radians(tf.rotation.yaw),
        math.sqrt(vel.x**2 + vel.y**2)
    ])

    # get nominal control from carla autopilot
    control_nom = ego.agent.run_step()
    a_nom, beta_nom = carla_to_bicycle(control_nom, ego_acc_min, ego_acc_max, ego_beta_min, ego_beta_max)
    U_nom = np.tile([a_nom, beta_nom], (N, 1))

    # nominal trajectory and linearization
    X_nom = np.zeros((N + 1, 4), dtype=float)
    X_nom[0] = ego_init.copy()
    A_seq, B_seq, c_seq = [], [], []

    for k in range(N):
        A_k, B_k = model.linearize(X_nom[k], U_nom[k])
        X_nom[k + 1] = model.step(X_nom[k], U_nom[k])
        c_k = X_nom[k + 1] - A_k @ X_nom[k] - B_k @ U_nom[k]
        A_seq.append(A_k)
        B_seq.append(B_k)
        c_seq.append(c_k)

    # sample agent trajectories
    ped_traj = sample_ped_trajectories(ped, cfg, N, dt, S)
    amb_traj = sample_amb_trajectories(amb, cfg, N, dt, S)

    # draw all pedestrian samples in red
    draw_sample_traj(client.world, ped_traj, color=carla.Color(255, 0, 0))

    # draw ambulance samples in blue
    draw_sample_traj(client.world, amb_traj, color=carla.Color(0, 0, 255))

    return {
            "status": False,
            "control": None,
            "ped_delta": None,
            "amb_delta": None,
        }

    # cvxpy variables
    x_var = cp.Variable((4, N + 1), name="x")
    u_var = cp.Variable((2, N), name="u")

    constraints = []
    constraints.append(x_var[:, 0] == ego_init)

    # dynamics constraints
    for k in range(N):
        constraints.append(
            x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
        )

    # control bounds
    for k in range(N):
        constraints += [
            u_var[0, k] >= ego_acc_min,
            u_var[0, k] <= ego_acc_max,
            u_var[1, k] >= ego_beta_min,
            u_var[1, k] <= ego_beta_max,
        ]

    # safe distance constraints
    ped_cons, ped_delta = safe_distance(x_var, ped_traj, ego_w, ego_l, d_safe=2.0, label="ped")
    amb_cons, amb_delta = safe_distance(x_var, amb_traj, ego_w, ego_l, d_safe=5.0, label="amb")
    constraints += ped_cons + amb_cons

    # objective
    objective = cp.Minimize(ped_delta + amb_delta)
    prob = cp.Problem(objective, constraints)

    # select MIP solver
    solver = None
    for s in [cp.GUROBI, cp.CPLEX, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]:
        if s in cp.installed_solvers():
            solver = s
            break
    if solver is None:
        raise RuntimeError(
            f"No MIP solver found. Install GUROBI, CPLEX, GLPK, or SCIP. "
            f"Installed: {cp.installed_solvers()}"
        )

    prob.solve(solver=solver, verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: solver returned status '{prob.status}'")
        return {
            "status": False,
            "control": None,
            "ped_delta": None,
            "amb_delta": None,
        }

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego_acc_min, ego_acc_max, ego_beta_min, ego_beta_max, ego_lr)

    return {
        "status": True,
        "control": control,
        "ped_delta": float(ped_delta.value),
        "amb_delta": float(amb_delta.value),
    }