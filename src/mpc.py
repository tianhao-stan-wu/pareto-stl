import carla
import numpy as np
import cvxpy as cp
import math
import random

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance_vehicle, safe_distance_walker
from src.utils import SmoothNoise, draw_sample_traj, bicycle_to_carla, carla_to_bicycle


COLORS = [
        carla.Color(255, 0, 0),     # red
        carla.Color(0, 0, 255),     # blue
        carla.Color(255, 255, 0),   # yellow
        carla.Color(255, 0, 255),   # magenta
        carla.Color(0, 255, 255),   # cyan
        carla.Color(255, 128, 0),   # orange
    ]


def build_and_solve_mpc(client, agents, cfg):
    # extract parameters
    T = cfg["mpc"]["horizon"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
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

    for i, agent in enumerate(agents[1:]):
        trajs = agent.sample_trajectories(N, dt, S)
        traj_mean = trajs.mean(axis=0)

        draw_sample_traj(client.world, trajs, color=COLORS[i % len(COLORS)], life_time=T)

        if agent.agent_type == "vehicle":
            cons, delta = safe_distance_vehicle(
                x_var, traj_mean, ego.length, agent.length,
                d_safe=cfg["stl"][agent.key], label=agent.key
            )
        else:
            cons, delta = safe_distance_walker(
                x_var, traj_mean, ego.width, ego.length,
                d_safe=cfg["stl"][agent.key], label=agent.key
            )

        constraints += cons
        deltas[agent.key] = delta


    # objective
    objective = cp.Minimize(deltas["pedestrian"] + deltas["ambulance"] + deltas["parked_v1"] + deltas["parked_v2"])
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
            "deltas": None,
        }

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    delta_values = {key: float(d.value) for key, d in deltas.items()}
    print(", ".join(f"{key}: {val:.3f}" for key, val in delta_values.items()))

    return {
        "status": True,
        "control": control,
        "deltas": delta_values,
    }