"""
mpc with pareto tradeoff between relaxations
"""

import carla
import numpy as np
import cvxpy as cp
import math
import random
import time

from src.bicycle_model import KinematicBicycle
from src.stl_constraints import safe_distance_vehicle_pareto, safe_distance_walker_pareto
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
    "parked_v2": "cyan",
    "opposite_v1": "green"
}


def build_and_solve_mpc_pareto(client, agents, cfg):
    """
    Epsilon-constraint MPC: grid over epsilon bounds for each delta,
    minimize one delta while constraining others <= epsilon.
    Returns Pareto front of solutions.
    """
    
    t_solve_start = time.perf_counter()

    # extract parameters
    T = cfg["mpc"]["horizon"]
    dt = cfg["carla"]["dt"]
    N = int(round(T / dt))
    lt = dt * 1.5
    S = cfg["mpc"]["num_samples"]
    density = cfg["mpc"]["density"]

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

    # nominal
    control_nom = ego.agent.run_step()
    a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)
    U_nom = np.tile([a_nom, beta_nom], (N, 1))

    X_nom = np.zeros((N + 1, 4), dtype=float)
    X_nom[0] = ego_init.copy()
    A_seq, B_seq, c_seq = [], [], []
    for k in range(N):
        A_k, B_k = model.linearize(X_nom[k], U_nom[k])
        X_nom[k + 1] = model.step(X_nom[k], U_nom[k])
        c_k = X_nom[k + 1] - A_k @ X_nom[k] - B_k @ U_nom[k]
        A_seq.append(A_k); B_seq.append(B_k); c_seq.append(c_k)

    # sample trajectories once — reuse for all solves
    agent_trajs = {}
    for agent in agents[1:]:
        trajs = agent.sample_trajectories(N, dt, S)
        draw_sample_traj(client.world, trajs, color=COLORS[MAP[agent.key]], life_time=lt)
        agent_trajs[agent.key] = trajs.mean(axis=0)

    # build epsilon grids for each agent
    d_safes = {agent.key: cfg["stl"][agent.key] for agent in agents[1:]}
    agent_keys = list(d_safes.keys())
    epsilon_grids = {key: np.linspace(d_safes[key] / density, d_safes[key], density) for key in agent_keys}

    print("epsilon_grids:")
    for key, grid in epsilon_grids.items():
        print(f"  {key}: {[f'{v:.3f}' for v in grid]}")

    # solver selection
    solver = None
    for s in [cp.GUROBI, cp.CPLEX, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]:
        if s in cp.installed_solvers():
            solver = s
            break
    if solver is None:
        raise RuntimeError(f"No MIP solver. Installed: {cp.installed_solvers()}")

    # iterate: minimize each delta, constrain the rest
    pareto_front = []

    for obj_key in agent_keys:
        # grid over all OTHER deltas
        other_keys = [k for k in agent_keys if k != obj_key]
        other_grids = [epsilon_grids[k] for k in other_keys]

        # cartesian product of epsilon values
        grid_points = np.array(np.meshgrid(*other_grids)).T.reshape(-1, len(other_keys))
        print(f"Minimizing {obj_key}: {len(grid_points)} grid points over {other_keys}")

        for eps_values in grid_points:
            eps_map = dict(zip(other_keys, eps_values))

            # build fresh problem
            x_var = cp.Variable((4, N + 1), name="x")
            u_var = cp.Variable((2, N), name="u")

            constraints = [x_var[:, 0] == ego_init]

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

            # safe distance constraints + deltas
            deltas = {}
            for agent in agents[1:]:
                traj_mean = agent_trajs[agent.key]
                d_safe = d_safes[agent.key]

                if agent.agent_type == "vehicle":
                    cons, delta = safe_distance_vehicle_pareto(
                        x_var, traj_mean, ego.width, ego.length,
                        agent.width, agent.length, d_safe=d_safe, label=agent.key
                    )
                else:
                    cons, delta = safe_distance_walker_pareto(
                        x_var, traj_mean, ego.width, ego.length,
                        d_safe=d_safe, label=agent.key
                    )

                constraints += cons
                deltas[agent.key] = delta

                if agent.key == obj_key:
                    # this is the objective — only bound by d_safe
                    constraints.append(delta <= d_safe)
                else:
                    # constrained by epsilon
                    constraints.append(delta <= eps_map[agent.key])


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
                traj_smooth += cp.norm(x_var[:2, k+1] - 2 * x_var[:2, k] + x_var[:2, k-1], 1)

            objective = cp.Minimize(
                w_safe * deltas[obj_key]
                + w_control * control_cost
                + w_smooth * (control_rate + traj_smooth)
            )

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=solver, verbose=False)

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                delta_values = {k: float(d.value) for k, d in deltas.items()}
                a, beta = u_var.value[:, 0]
                control = bicycle_to_carla(
                    [a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
                )

                # print epsilon and delta for this solve
                eps_str = ", ".join(f"{k}: {v:.3f}" for k, v in eps_map.items())
                delta_str = ", ".join(f"{k}: {v:.3f}" for k, v in delta_values.items())
                print(f"  min({obj_key}) | eps=[{eps_str}] → delta=[{delta_str}]")

                pareto_front.append({
                    "obj_key": obj_key,
                    "epsilons": eps_map,
                    "deltas": delta_values,
                    "control": control,
                    "u_opt": u_var.value.copy(),
                    "x_opt": x_var.value.copy(),
                    "objective": prob.value,
                })

            else:
                eps_str = ", ".join(f"{k}: {v:.3f}" for k, v in eps_map.items())
                print(f"  min({obj_key}) | eps=[{eps_str}] → {prob.status}")

    # select best point: min pedestrian delta, then min ambulance delta
    if not pareto_front:
        return {
            "status": False,
            "control": control_nom,
            "deltas": None,
            "t_build": 0,
            "t_solve": t_solve,
        }

    # first: minimal pedestrian relaxation
    min_ped = min(p["deltas"]["pedestrian"] for p in pareto_front)
    ped_best = [p for p in pareto_front if abs(p["deltas"]["pedestrian"] - min_ped) < 1e-4]

    # second: among those, minimal ambulance relaxation
    best = min(ped_best, key=lambda p: p["deltas"]["ambulance"])

    # draw best trajectory
    ego_traj = best["x_opt"][:2, :].T
    draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

    t_solve = time.perf_counter() - t_solve_start

    delta_values = best["deltas"]
    print(f"Pareto front: {len(pareto_front)} points | "
          + ", ".join(f"{k}: {v:.3f}" for k, v in delta_values.items()) + f" | solve time: {t_solve}")

    return {
        "status": True,
        "control": best["control"],
        "deltas": delta_values,
        "t_build": 0,
        "t_solve": t_solve,
    }