"""
mpc with stl constraints (no relaxation)
"""

import carla
import numpy as np
import cvxpy as cp
import math
import random
import time

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance_vehicle_hard, safe_distance_walker_hard
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



def build_and_solve_mpc_hard(client, agents, cfg):

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

    # get nominal control from carla autopilot
    control_nom = ego.agent.run_step()

    a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)
    U_nom = np.tile([a_nom, 0], (N, 1))

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

    # draw nominal trajectory in white
    nom_traj = X_nom[:, :2]  # (N+1, 2)
    # draw_sample_traj(client.world, nom_traj, color=COLORS["white"], life_time=lt)

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

    for i, agent in enumerate(agents[1:]):

        trajs = agent.sample_trajectories(N, dt, S)
        draw_sample_traj(client.world, trajs, color=COLORS[MAP[agent.key]], life_time=lt)

        traj_mean = trajs.mean(axis=0)
        d_safe = cfg["stl"][agent.key]

        if agent.key in ["parked_v1", "parked_v2", "ambulance"]:
            cons = safe_distance_vehicle_hard(
                x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
                d_safe=d_safe, label=agent.key
            )
        else:
            cons = safe_distance_walker_hard(
                x_var, traj_mean, ego.width, ego.length,
                d_safe=d_safe, label=agent.key
            )

        constraints += cons

    # control_cost = cp.sum_squares(u_var[:, 0] - U_nom[0])
    control_cost = cp.sum_squares(u_var - U_nom.T)

    # add small penalty for deviation from nominal control
    objective = cp.Minimize(control_cost)
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
    draw_sample_traj(client.world, ego_traj, color=COLORS[MAP["ego"]], life_time=lt)

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    return {
        "status": True,
        "control": control,
        "deltas": None,
        "t_build": t_build, 
        "t_solve": t_solve,
    }