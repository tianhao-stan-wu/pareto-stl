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
from src.stl_constraints_hard import (
    safe_distance_vehicle_hard, safe_distance_walker_hard, avoid_rectangle_hard,
    stay_below_line_hard
)
from src.utils import (
    SmoothNoise, draw_sample_traj, draw_rectangle_boundary, bicycle_to_carla, carla_to_bicycle,
    COLORS, MAP
)


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

    # get nominal control from autopilot + planned waypoints
    control_nom = ego.agent.run_step()
    a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    plan = list(ego.agent.get_local_planner().get_plan())

    U_nom = np.zeros((N, 2))
    for k in range(N):
        if k < len(plan) - 1:
            wp_curr = plan[k][0].transform
            wp_next = plan[k + 1][0].transform

            dx = wp_next.location.x - wp_curr.location.x
            dy = wp_next.location.y - wp_curr.location.y
            yaw_next = math.atan2(dy, dx)

            if k == 0:
                dyaw = yaw_next - ego_init[2]
            else:
                wp_prev = plan[k - 1][0].transform
                dx_p = wp_curr.location.x - wp_prev.location.x
                dy_p = wp_curr.location.y - wp_prev.location.y
                yaw_prev = math.atan2(dy_p, dx_p)
                dyaw = yaw_next - yaw_prev

            # estimate beta from heading change
            v_est = max(ego_init[3] + a_nom * k * dt, 0.5)
            beta_k = dyaw * ego.lr / (v_est * dt)
            beta_k = max(ego.beta_min, min(beta_k, ego.beta_max))

            U_nom[k] = [a_nom, beta_k]
        else:
            U_nom[k] = [a_nom, beta_nom]

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
    rectangles = ["vehicles", "lane"]
    forbidden_rectangles = []

    for item in rectangles:
        cfg_item = cfg["stl"][item]
        rect = [cfg_item['x_min'], cfg_item['x_max'], cfg_item['y_min'], cfg_item['y_max']]
        forbidden_rectangles.append(rect)

    for rect in forbidden_rectangles:
        cons = avoid_rectangle_hard(x_var=x_var, rect=rect, N=N+1)
        constraints += cons

    # intersection constraint — ego must not enter intersection
    y_line = cfg["stl"]["intersection_y"]

    cons = stay_below_line_hard(x_var=x_var, y_line=y_line, N=N+1)
    constraints += cons

    for i, agent in enumerate(agents[1:]):

        trajs = agent.sample_trajectories(N, dt, S)

        traj_mean = trajs.mean(axis=0)
        d_safe = cfg["stl"][agent.key]

        if agent.key in ["ambulance"]:
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

    traj_cost = cp.sum_squares(x_var - X_nom.T)

    control_rate = 0
    for k in range(N - 1):
        control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

    # add small penalty for deviation from nominal control
    objective = cp.Minimize(traj_cost + 0.1 * control_rate)
    prob = cp.Problem(objective, constraints)

    # get number of constraints/variables
    num_constraints = sum(c.size for c in constraints)
    num_variables = sum(v.size for v in prob.variables())
    print(f"  Problem size: {num_constraints} constraints, {num_variables} variables")

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

        print(f"Warning: solver returned status '{prob.status}', apply fallback control_fallback to come to stop")

        control_fallback = carla.VehicleControl()
        control_fallback.throttle = 0.0
        control_fallback.brake = 0.5
        control_fallback.steer = 0.0
        control_fallback.manual_gear_shift = False

        return {
            "status": False,
            "control": control_fallback,
            "deltas": None,
            "t_build": t_build, 
            "t_solve": t_solve,
            "num_constraints": num_constraints,
            "num_variables": num_variables,
        }

    # draw ego planned trajectory
    ego_traj = x_var.value[:2, :].T  # (N+1, 2) — extract px, py
    draw_sample_traj(client.world, ego_traj, color=COLORS[MAP["ego"]], life_time=lt)

    return {
        "status": True,
        "control": control,
        "deltas": None,
        "t_build": t_build, 
        "t_solve": t_solve,
        "num_constraints": num_constraints,
        "num_variables": num_variables,
    }
