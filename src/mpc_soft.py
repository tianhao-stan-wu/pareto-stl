"""
mpc with stl constraints (with relaxation)
"""

import carla
import numpy as np
import cvxpy as cp
import math
import random
import time

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance_vehicle, safe_distance_walker
from src.utils import SmoothNoise, draw_sample_traj, bicycle_to_carla, carla_to_bicycle


COLORS = {
    "red":     carla.Color(150, 0, 0),
    "blue":    carla.Color(0, 0, 150),
    "green":   carla.Color(0, 80, 0),
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


def build_and_solve_mpc_soft(client, agents, cfg):

    # extract parameters
    T = cfg["mpc"]["horizon"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
    lt = dt * 1.5
    S = cfg["mpc"]["num_samples"]

    # set up model
    ego = agents[0]
    model = KinematicBicycle(lr=ego.lr, dt=dt)

    # get ego's current state
    tf = ego.get_transform()
    vel = ego.get_velocity()
    ego_init = np.array([
        tf.location.x,
        tf.location.y,
        math.radians(tf.rotation.yaw),
        math.sqrt(vel.x**2 + vel.y**2)
    ])

    # center coordinates at ego's initial position
    px_offset = ego_init[0]
    py_offset = ego_init[1]
    ego_init[0] = 0.0
    ego_init[1] = 0.0

    # get nominal control from carla autopilot
    control_nom = ego.agent.run_step()

    a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)
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

    t_build_start = time.perf_counter()

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
            u_var[0, k] >= ego.acc_min,
            u_var[0, k] <= ego.acc_max,
            u_var[1, k] >= ego.beta_min,
            u_var[1, k] <= ego.beta_max,
        ]

    # add STL constraints
    deltas = {}
    all_binaries = {}

    for i, agent in enumerate(agents[1:]):

        trajs = agent.sample_trajectories(N, dt, S)
        draw_sample_traj(client.world, trajs, color=COLORS[MAP[agent.key]], life_time=lt)

        # center agent trajectories
        trajs[:, :, 0] -= px_offset
        trajs[:, :, 1] -= py_offset

        traj_mean = trajs.mean(axis=0)
        d_safe = cfg["stl"][agent.key]

        if agent.agent_type == "vehicle":
            cons, delta = safe_distance_vehicle(
                x_var, traj_mean, ego.width, ego.length,
                agent.width, agent.length, d_safe=d_safe, label=agent.key
            )
            # all_binaries[agent.key] = bins
        else:
            cons, delta = safe_distance_walker(
                x_var, traj_mean, ego.width, ego.length,
                d_safe=d_safe, label=agent.key
            )

        constraints += cons
        deltas[agent.key] = delta
    
    # w_safe = cfg["mpc"]["w_safe"]
    # w_control = cfg["mpc"]["w_control"]
    # control_cost = cp.sum_squares(u_var[:, 0] - U_nom[0])
    # control_cost = cp.sum_squares(u_var - U_nom.T)
    # control_cost = cp.norm(u_var - U_nom.T, 1)

    w_safe = cfg["mpc"]["w_safe"]
    w_control = cfg["mpc"]["w_control"]
    w_smooth = cfg["mpc"]["w_smooth"]

    # control deviation from nominal
    control_cost = cp.norm(u_var - U_nom.T, 1)

    # control rate — penalize change between consecutive controls
    control_rate = 0
    for k in range(N - 1):
        control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

    # trajectory smoothness — penalize curvature (second derivative)
    traj_smooth = 0
    for k in range(1, N):
        # x_{k+1} - 2*x_k + x_{k-1} ≈ acceleration in position
        traj_smooth += cp.norm(x_var[:2, k+1] - 2 * x_var[:2, k] + x_var[:2, k-1], 1)

    objective = cp.Minimize(
        w_safe * sum(deltas.values())
        + w_control * control_cost
        + w_smooth * (control_rate + traj_smooth)
    )
    # objective = cp.Minimize(w_safe * sum(deltas.values()) + w_control * control_cost)
    
    prob = cp.Problem(objective, constraints)

    t_build = time.perf_counter() - t_build_start

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

    t_solve_start = time.perf_counter()
    prob.solve(solver=solver, verbose=False)
    t_solve = time.perf_counter() - t_solve_start

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: solver returned status '{prob.status}', apply nominal control")
        return {
            "status": False,
            "control": control_nom,
            "deltas": None,
            "t_build": t_build, 
            "t_solve": t_solve,
        }

    # draw ego planned trajectory
    ego_traj = x_var.value[:2, :].T  # (N+1, 2) — extract px, py
    ego_traj[:, 0] += px_offset
    ego_traj[:, 1] += py_offset
    draw_sample_traj(client.world, ego_traj, color=COLORS[MAP["ego"]], life_time=lt)

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)


    delta_values = {key: float(d.value) for key, d in deltas.items()}
    print(", ".join(f"{key}: {val:.3f}" for key, val in delta_values.items()))



    return {
        "status": True,
        "control": control,
        "deltas": delta_values,
        "t_build": t_build, 
        "t_solve": t_solve,
    }