# import numpy as np
# import cvxpy as cp
# import math

# from bicycle_model import KinematicBicycle
# from stl_constraints import *


# def bicycle_to_carla(u, acc_min, acc_max, beta_min, beta_max, lr=1.5):
#     """Convert bicycle model [a, beta] to CARLA VehicleControl."""
#     a, beta = u

#     a = max(acc_min, min(a, acc_max))
#     beta = max(beta_min, min(beta, beta_max))

#     control = carla.VehicleControl()
#     control.manual_gear_shift = False

#     if a >= 0:
#         control.throttle = min(a / acc_max, 1.0)
#         control.brake = 0.0
#     else:
#         control.throttle = 0.0
#         control.brake = min(abs(a) / abs(acc_min), 1.0)

#     steer_angle = math.degrees(math.atan(2.0 * lr * math.tan(beta) / (2.0 * lr)))
#     max_steer = math.degrees(math.atan(2.0 * lr * math.tan(beta_max) / (2.0 * lr)))
#     control.steer = max(-1.0, min(steer_angle / max_steer, 1.0))

#     return control


# def carla_to_bicycle(control, acc_min, acc_max, beta_min, beta_max):
#     """Convert CARLA VehicleControl to bicycle model [a, beta]."""
#     if control.throttle > 0:
#         a = control.throttle * acc_max
#     else:
#         a = -control.brake * abs(acc_min)

#     max_steer_rad = math.radians(70.0)
#     steer_angle = control.steer * max_steer_rad
#     beta = math.atan(0.5 * math.tan(steer_angle))

#     a = max(acc_min, min(a, acc_max))
#     beta = max(beta_min, min(beta, beta_max))

#     return a, beta


# def sample_ped_trajectory(ped, cfg, N: int, dt: float):
#     """Sample pedestrian trajectory with noisy speed and direction for N steps."""
#     tf = ped.get_transform()
#     px, py = tf.location.x, tf.location.y
#     yaw = math.radians(tf.rotation.yaw)

#     mean_speed = cfg["pedestrian"]["mean_speed"]
#     std_speed = cfg["pedestrian"]["std_speed"]
#     std_dir = cfg["pedestrian"]["std_dir"]

#     traj = np.zeros((N + 1, 2))
#     traj[0] = [px, py]

#     for k in range(N):
#         noise_yaw = random.gauss(0, std_dir)
#         yaw += noise_yaw
#         speed = max(0.0, random.gauss(mean_speed, std_speed))
#         px += speed * math.cos(yaw) * dt
#         py += speed * math.sin(yaw) * dt
#         traj[k + 1] = [px, py]

#     return traj


# def sample_amb_trajectory(amb, amb_agent, cfg, N: int, dt: float):
#     """Sample ambulance trajectory with smooth noise on acceleration and steering."""
#     tf = amb.get_transform()
#     vel = amb.get_velocity()

#     px, py = tf.location.x, tf.location.y
#     yaw = math.radians(tf.rotation.yaw)
#     speed = math.sqrt(vel.x**2 + vel.y**2)

#     acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=cfg["ambulance"].get("std_acc", 0.1))
#     steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=cfg["ambulance"].get("std_steer", 0.02))

#     # get base control from agent
#     control = amb_agent.run_step()
#     base_throttle = control.throttle
#     base_brake = control.brake
#     base_steer = control.steer

#     traj = np.zeros((N + 1, 2))
#     traj[0] = [px, py]

#     for k in range(N):
#         # noisy acceleration
#         acc = base_throttle + acc_noise.sample()
#         if acc >= 0:
#             a = min(acc, 1.0) * cfg["mpc"]["acc_max"]
#         else:
#             a = -min(abs(acc), 1.0) * abs(cfg["mpc"]["acc_min"])

#         # noisy steering → beta
#         steer = max(-1.0, min(base_steer + steer_noise.sample(), 1.0))
#         max_steer_rad = math.radians(70.0)
#         beta = math.atan(0.5 * math.tan(steer * max_steer_rad))

#         # step with bicycle dynamics
#         speed += a * dt
#         speed = max(0.0, speed)
#         yaw += (speed / cfg["mpc"]["lr"]) * beta * dt
#         px += speed * math.cos(yaw) * dt
#         py += speed * math.sin(yaw) * dt
#         traj[k + 1] = [px, py]

#     return traj


# def build_and_solve_mpc(ego, ped, amb, cfg):
#     # extract parameters from yaml
#     T = cfg['mpc']['T']
#     dt = cfg['carla']['dt']
#     N = T / dt

#     acc_min = cfg['mpc']['acc_min']
#     acc_max = cfg['mpc']['acc_max']
#     beta_min = cfg['mpc']['beta_min']
#     beta_max = cfg['mpc']['beta_max']

#     ego_w = cfg['mpc']['width']
#     ego_l = cfg['mpc']['length']
#     lr = cfg['mpc']['lr']

#     # set up model
#     model = KinematicBicycle(lr=lr, dt=dt)

#     # nominal trajectory and linearization
#     U_nom = np.zeros((N, 2), dtype=float)
#     X_nom = np.zeros((N+1, 4), dtype=float)

#     # get ego's current location as init
#     tf = ego.actor.get_transform()
#     vel = ego.actor.get_velocity()
#     ego_init = np.array([
#         tf.location.x,
#         tf.location.y,
#         math.radians(tf.rotation.yaw),
#         math.sqrt(vel.x**2 + vel.y**2)
#     ])

#     # get nominal control from carla autopilot
#     control_nom = ego.agent.run_step()
#     a_nom, beta_nom = carla_to_bicycle(control_nom, acc_min, acc_max, beta_min, beta_max)
#     U_nom = np.tile([a_nom, beta_nom], (N, 1))

#     X_nom[0] = ego_init.copy()
#     A_seq, B_seq, c_seq = [], [], []

#     for k in range(N):
#         A_k, B_k = model.linearize(X_nom[k], U_nom[k])
#         X_nom[k+1] = model.step(X_nom[k], U_nom[k])
#         c_k = X_nom[k+1] - A_k @ X_nom[k] - B_k @ U_nom[k]
#         A_seq.append(A_k); B_seq.append(B_k); c_seq.append(c_k)

#     # cvxpy variables
#     x_var = cp.Variable((4, N+1), name="x")
#     u_var = cp.Variable((2, N), name="u")

#     constraints: list = []
#     constraints.append(x_var[:, 0] == ego_init)

#     # vehicle constraints
#     for k in range(N):
#         constraints.append(x_var[:, k+1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k])

#     for k in range(N):
#         constraints += [
#             u_var[0, k] >= acc_min,
#             u_var[0, k] <= acc_max,
#             u_var[1, k] >= beta_min,
#             u_var[1, k] <= beta_max,
#         ]

#     # build constraints
#     ped_cons, ped_delta = safe_distance(x_var, ped_traj, ego_w, ego_l, d_safe=2.0, label="ped")
#     amb_cons, amb_delta = safe_distance(x_var, amb_traj, ego_w, ego_l, d_safe=5.0, label="amb")
#     constraints += ped_cons + amb_cons

#     # objective
#     objective = cp.Minimize(ped_delta + amb_delta)
#     prob = cp.Problem(objective, constraints)

#     # select MIP solver
#     solver = None
#     for s in [cp.GUROBI, cp.CPLEX, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]:
#         if s in cp.installed_solvers():
#             solver = s
#             break
#     if solver is None:
#         raise RuntimeError(
#             f"No MIP solver found. Install GUROBI, CPLEX, GLPK, or SCIP. "
#             f"Installed: {cp.installed_solvers()}"
#         )

#     prob.solve(solver=solver, verbose=False)

#     if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#         print(f"Warning: solver returned status '{prob.status}'")
#         return {
#             "status": False,
#             "control": None,
#             "ped_delta": None,
#             "amb_delta": None,
#         }

#     a, beta = u_var.value[:, 0]
#     control = bicycle_to_carla([a, beta], acc_min, acc_max, beta_min, beta_max, lr)

#     return {
#         "status": True,
#         "control": control,
#         "ped_delta": float(ped_delta.value),
#         "amb_delta": float(amb_delta.value),
#     }




import carla
import numpy as np
import cvxpy as cp
import math
import random

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance
from src.utils import SmoothNoise


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

    steer_angle = math.degrees(math.atan(2.0 * lr * math.tan(beta) / (2.0 * lr)))
    max_steer = math.degrees(math.atan(2.0 * lr * math.tan(ego_beta_max) / (2.0 * lr)))
    control.steer = max(-1.0, min(steer_angle / max_steer, 1.0))

    return control


def carla_to_bicycle(control, ego_acc_min, ego_acc_max, ego_beta_min, ego_beta_max):
    """Convert CARLA VehicleControl to bicycle model [a, beta]."""
    if control.throttle > 0:
        a = control.throttle * ego_acc_max
    else:
        a = -control.brake * abs(ego_acc_min)

    max_steer_rad = math.radians(70.0)
    steer_angle = control.steer * max_steer_rad
    beta = math.atan(0.5 * math.tan(steer_angle))

    a = max(ego_acc_min, min(a, ego_acc_max))
    beta = max(ego_beta_min, min(beta, ego_beta_max))

    return a, beta


def sample_ped_trajectory(ped, cfg, N: int, dt: float):
    """Sample pedestrian trajectory with noisy speed and direction for N steps."""
    tf = ped.get_transform()
    px, py = tf.location.x, tf.location.y
    yaw = math.radians(tf.rotation.yaw)

    mean_speed = cfg["pedestrian"]["mean_speed"]
    std_speed = cfg["pedestrian"]["std_speed"]
    std_dir = cfg["pedestrian"]["std_dir"]

    traj = np.zeros((N + 1, 2))
    traj[0] = [px, py]

    for k in range(N):
        yaw += random.gauss(0, std_dir)
        speed = max(0.0, random.gauss(mean_speed, std_speed))
        px += speed * math.cos(yaw) * dt
        py += speed * math.sin(yaw) * dt
        traj[k + 1] = [px, py]

    return traj


def sample_amb_trajectory(amb, cfg, N: int, dt: float):
    """Sample ambulance trajectory with smooth noise on acceleration and steering."""
    tf = amb.get_transform()

    px, py = tf.location.x, tf.location.y
    yaw = math.radians(tf.rotation.yaw)
    speed = amb.get_speed() / 3.6  # convert km/h to m/s

    ego_cfg = cfg["ego_vehicle"]
    lr = ego_cfg["lr"]

    acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=cfg["ambulance"].get("std_acc", 0.1))
    steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=cfg["ambulance"].get("std_steer", 0.02))

    # get base control from agent
    control = amb.agent.run_step()
    base_throttle = control.throttle
    base_steer = control.steer

    traj = np.zeros((N + 1, 2))
    traj[0] = [px, py]

    for k in range(N):
        acc = base_throttle + acc_noise.sample()
        if acc >= 0:
            a = min(acc, 1.0) * ego_cfg["acc_max"]
        else:
            a = -min(abs(acc), 1.0) * abs(ego_cfg["acc_min"])

        steer = max(-1.0, min(base_steer + steer_noise.sample(), 1.0))
        max_steer_rad = math.radians(70.0)
        beta = math.atan(0.5 * math.tan(steer * max_steer_rad))

        speed += a * dt
        speed = max(0.0, speed)
        yaw += (speed / lr) * beta * dt
        px += speed * math.cos(yaw) * dt
        py += speed * math.sin(yaw) * dt
        traj[k + 1] = [px, py]

    return traj


def build_and_solve_mpc(ego, ped, amb, cfg):
    # extract parameters
    T = cfg["mpc"]["T"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))

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
    ped_traj = sample_ped_trajectory(ped, cfg, N, dt)
    amb_traj = sample_amb_trajectory(amb, cfg, N, dt)

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