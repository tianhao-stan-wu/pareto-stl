import carla
import numpy as np
import cvxpy as cp
import math
import random
import time

from src.bicycle import KinematicBicycle

from src.stl import (
    safe_distance_vehicle, safe_distance_walker, stay_in_lane, clear_intersection
)

from src.utils import (
    draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
)

def solve_mpc_hard(A_seq, B_seq, c_seq, ego, agents, N):

    x_var = cp.Variable((4, N + 1), name="x")
    u_var = cp.Variable((2, N), name="u")

    constraints = []
    constraints.append(x_var[:, 0] == ego_init)

    # dynamics and control constraints
    for k in range(N):

        constraints.append(
            x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
        )

        constraints += [
            u_var[0, k] >= ego.acc_min,
            u_var[0, k] <= ego.acc_max,
            u_var[1, k] >= ego.beta_min,
            u_var[1, k] <= ego.beta_max,
        ]

    # STL constraints
    for i, agent in enumerate(agents[1:]):

        trajs = agent.sample_trajectories(N, dt, S)
        traj_mean = trajs.mean(axis=0)
        d_safe = cfg["stl"][agent.key]

        if agent.key in ["ambulance"]:
            cons, delta_x, delta_y = safe_distance_vehicle(
                x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
                d_safe=d_safe, label=agent.key
            )
        else:
            cons, delta_x, delta_y = safe_distance_walker(
                x_var, traj_mean, ego.width, ego.length,
                d_safe=d_safe, label=agent.key
            )

        constraints += cons

        # no relaxation
        constraints.append(delta_x <= 0)
        constraints.append(delta_y <= 0)

    x_min, x_max, y_min, y_max = [-46.3, -43.4, 40.9, 74.8]
    cons, delta_lane = stay_in_lane(x_var, x_min, x_max, y_min, y_max, N)
    constraints += cons
    constraints.append(delta_lane <= 0)

    y_exit = 0
    cons, delta_inter = clear_intersection(x_var, y_exit, N)
    constraints += cons
    constraints.append(delta_inter <= 0)

    traj_cost = cp.norm(x_var - X_nom.T)
    control_rate = 0

    for k in range(N - 1):
        control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

    # add small penalty for deviation from nominal control
    objective = cp.Minimize(traj_cost + 0.1 * control_rate)
    prob = cp.Problem(objective, constraints)

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
    draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    return {
        "status": True,
        "control": control,
        "deltas": None,
        "t_build": t_build, 
        "t_solve": t_solve,
        "num_constraints": num_constraints,
        "num_variables": num_variables,
    }


def solve_mpc_soft(computed, agents, cfg, client):

    T = cfg["mpc"]["horizon"]
    S = cfg["mpc"]["num_samples"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
    lt = dt * 1.5

    A_seq, B_seq, c_seq, ego_init, X_nom, U_nom = computed
    ego = agents[0]
    
    t_build_start = time.perf_counter()

    x_var = cp.Variable((4, N + 1), name="x")
    u_var = cp.Variable((2, N), name="u")

    constraints = []
    constraints.append(x_var[:, 0] == ego_init)

    # dynamics and control constraints
    for k in range(N):

        constraints.append(
            x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
        )

        constraints += [
            u_var[0, k] >= ego.acc_min,
            u_var[0, k] <= ego.acc_max,
            u_var[1, k] >= ego.beta_min,
            u_var[1, k] <= ego.beta_max,
        ]

    deltas = {}

    # STL constraints
    for i, agent in enumerate(agents[1:]):

        trajs = agent.sample_trajectories(N, dt, S)
        traj_mean = trajs.mean(axis=0)
        d_safe = cfg["stl"][agent.key]

        if agent.key in ["ambulance"]:
            cons, delta_x, delta_y = safe_distance_vehicle(
                x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
                d_safe=d_safe, label=agent.key
            )
        else:
            cons, delta_x, delta_y = safe_distance_walker(
                x_var, traj_mean, ego.width, ego.length,
                d_safe=d_safe, label=agent.key
            )

        constraints += cons
        deltas[agent.key + "_x"] = delta_x
        deltas[agent.key + "_y"] = delta_y

    # x_min, x_max, y_min, y_max = [-46.3, -43.4, 40.9, 74.8]
    # cons, delta_lane = stay_in_lane(x_var, x_min, x_max, y_min, y_max, N)
    # constraints += cons
    # constraints.append(delta_lane <= 4)
    # deltas["lane"] = delta_lane

    y_exit = 0
    cons, delta_inter = clear_intersection(x_var, y_exit, N)
    constraints += cons
    deltas["intersection"] = delta_inter

    # control deviation from nominal
    traj_cost = cp.norm(x_var - X_nom.T, 1)

    # control rate — penalize change between consecutive controls
    control_rate = 0
    for k in range(N - 1):
        control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

    eps = 1e-2  
    # objective = cp.Minimize(sum(deltas.values()) + eps * (control_rate + traj_cost))
    objective = cp.Minimize(sum(deltas.values()))

    prob = cp.Problem(objective, constraints)

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
    draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    delta_values = {key: float(d.value) for key, d in deltas.items()}
    print(", ".join(f"{key}: {val:.3f}" for key, val in delta_values.items()))

    return {
        "status": True,
        "control": control,
        "deltas": None,
        "t_build": t_build, 
        "t_solve": t_solve,
        "num_constraints": num_constraints,
        "num_variables": num_variables,
    }

def solve_mpc_pareto():
    pass


def build_and_solve_mpc(client, agents, cfg):

    # extract parameters
    T = cfg["mpc"]["horizon"]
    S = cfg["mpc"]["num_samples"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
    lt = dt * 1.5

    # build bicycle model
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

    # draw_sample_traj(client.world, X_nom[:, :2], color=COLORS["white"], life_time=lt)

    computed = [A_seq, B_seq, c_seq, ego_init, X_nom, U_nom]

    mpc_type = cfg['mpc']['type']

    if mpc_type == "hard":
        results = solve_mpc_hard(A_seq, B_seq, c_seq, ego, agents, cfg)

    elif mpc_type == "soft":
        results = solve_mpc_soft(computed, agents, cfg, client)

    elif mpc_type == "pareto":
        results = solve_mpc_pareto()

    return results







