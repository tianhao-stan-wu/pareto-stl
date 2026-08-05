# import carla
# import numpy as np
# import cvxpy as cp
# import math
# import random
# import time

# from src.bicycle import KinematicBicycle

# from src.stl import (
#     safe_distance_vehicle, safe_distance_walker, stay_in_lane, clear_intersection
# )

# from src.utils import (
#     draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
# )

# def solve_mpc_hard(client, agents, cfg):

#     x_var = cp.Variable((4, N + 1), name="x")
#     u_var = cp.Variable((2, N), name="u")

#     constraints = []
#     constraints.append(x_var[:, 0] == ego_init)

#     # dynamics and control constraints
#     for k in range(N):

#         constraints.append(
#             x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
#         )

#         constraints += [
#             u_var[0, k] >= ego.acc_min,
#             u_var[0, k] <= ego.acc_max,
#             u_var[1, k] >= ego.beta_min,
#             u_var[1, k] <= ego.beta_max,
#         ]

#     # STL constraints
#     for i, agent in enumerate(agents[1:]):

#         trajs = agent.sample_trajectories(N, dt, S)
#         traj_mean = trajs.mean(axis=0)
#         d_safe = cfg["stl"][agent.key]

#         if agent.key in ["ambulance"]:
#             cons, delta_x, delta_y = safe_distance_vehicle(
#                 x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
#                 d_safe=d_safe, label=agent.key
#             )
#         else:
#             cons, delta_x, delta_y = safe_distance_walker(
#                 x_var, traj_mean, ego.width, ego.length,
#                 d_safe=d_safe, label=agent.key
#             )

#         constraints += cons

#         # no relaxation
#         constraints.append(delta_x <= 0)
#         constraints.append(delta_y <= 0)

#     x_min, x_max, y_min, y_max = [-46.3, -43.4, 40.9, 74.8]
#     cons, delta_lane = stay_in_lane(x_var, x_min, x_max, y_min, y_max, N)
#     constraints += cons
#     constraints.append(delta_lane <= 0)

#     y_exit = 0
#     cons, delta_inter = clear_intersection(x_var, y_exit, N)
#     constraints += cons
#     constraints.append(delta_inter <= 0)

#     traj_cost = cp.norm(x_var - X_nom.T)
#     control_rate = 0

#     for k in range(N - 1):
#         control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

#     # add small penalty for deviation from nominal control
#     objective = cp.Minimize(traj_cost + 0.1 * control_rate)
#     prob = cp.Problem(objective, constraints)

#     num_constraints = sum(c.size for c in constraints)
#     num_variables = sum(v.size for v in prob.variables())
#     print(f"  Problem size: {num_constraints} constraints, {num_variables} variables")

#     t_build = time.perf_counter() - t_build_start

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

#     t_solve_start = time.perf_counter()
#     prob.solve(solver=solver, verbose=False)
#     t_solve = time.perf_counter() - t_solve_start

#     if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:

#         print(f"Warning: solver returned status '{prob.status}', apply fallback control_fallback to come to stop")

#         control_fallback = carla.VehicleControl()
#         control_fallback.throttle = 0.0
#         control_fallback.brake = 0.5
#         control_fallback.steer = 0.0
#         control_fallback.manual_gear_shift = False

#         return {
#             "status": False,
#             "control": control_fallback,
#             "deltas": None,
#             "t_build": t_build, 
#             "t_solve": t_solve,
#             "num_constraints": num_constraints,
#             "num_variables": num_variables,
#         }

#     # draw ego planned trajectory
#     ego_traj = x_var.value[:2, :].T  # (N+1, 2) — extract px, py
#     draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

#     a, beta = u_var.value[:, 0]
#     control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

#     return {
#         "status": True,
#         "control": control,
#         "deltas": None,
#         "t_build": t_build, 
#         "t_solve": t_solve,
#         "num_constraints": num_constraints,
#         "num_variables": num_variables,
#     }


# def solve_mpc_soft(client, agents, cfg):

#     # extract parameters
#     T = cfg["mpc"]["horizon"]
#     S = cfg["mpc"]["num_samples"]
#     dt = cfg["carla"]["dt"]
#     N = int(round(T / dt))
#     lt = dt * 1.5

#     # build bicycle model
#     ego = agents[0]
#     model = KinematicBicycle(lr=ego.lr, dt=dt)

#     # get ego's current state
#     tf = ego.get_transform()
#     vel = ego.get_velocity()
#     ego_init = np.array([
#         tf.location.x,
#         tf.location.y,
#         math.radians(tf.rotation.yaw),
#         math.sqrt(vel.x**2 + vel.y**2)
#     ])

#     # get nominal control from autopilot + planned waypoints
#     control_nom = ego.agent.run_step()
#     a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

#     plan = list(ego.agent.get_local_planner().get_plan())

#     U_nom = np.zeros((N, 2))

#     for k in range(N):

#         if k < len(plan) - 1:

#             wp_curr = plan[k][0].transform
#             wp_next = plan[k + 1][0].transform

#             dx = wp_next.location.x - wp_curr.location.x
#             dy = wp_next.location.y - wp_curr.location.y
#             yaw_next = math.atan2(dy, dx)

#             if k == 0:
#                 dyaw = yaw_next - ego_init[2]

#             else:
#                 wp_prev = plan[k - 1][0].transform
#                 dx_p = wp_curr.location.x - wp_prev.location.x
#                 dy_p = wp_curr.location.y - wp_prev.location.y
#                 yaw_prev = math.atan2(dy_p, dx_p)
#                 dyaw = yaw_next - yaw_prev

#             # estimate beta from heading change
#             v_est = max(ego_init[3] + a_nom * k * dt, 0.5)
#             beta_k = dyaw * ego.lr / (v_est * dt)
#             beta_k = max(ego.beta_min, min(beta_k, ego.beta_max))

#             U_nom[k] = [a_nom, beta_k]

#         else:
#             U_nom[k] = [a_nom, beta_nom]

#     # nominal trajectory and linearization
#     X_nom = np.zeros((N + 1, 4), dtype=float)
#     X_nom[0] = ego_init.copy()
#     A_seq, B_seq, c_seq = [], [], []

#     for k in range(N):

#         A_k, B_k = model.linearize(X_nom[k], U_nom[k])
#         X_nom[k + 1] = model.step(X_nom[k], U_nom[k])
#         c_k = X_nom[k + 1] - A_k @ X_nom[k] - B_k @ U_nom[k]
#         A_seq.append(A_k)
#         B_seq.append(B_k)
#         c_seq.append(c_k)

#     # draw_sample_traj(client.world, X_nom[:, :2], color=COLORS["white"], life_time=lt)
    
#     t_build_start = time.perf_counter()

#     x_var = cp.Variable((4, N + 1), name="x")
#     u_var = cp.Variable((2, N), name="u")

#     constraints = []
#     constraints.append(x_var[:, 0] == ego_init)

#     # dynamics and control constraints
#     for k in range(N):

#         constraints.append(
#             x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
#         )

#         constraints += [
#             u_var[0, k] >= ego.acc_min,
#             u_var[0, k] <= ego.acc_max,
#             u_var[1, k] >= ego.beta_min,
#             u_var[1, k] <= ego.beta_max,
#         ]

#     deltas = {}

#     # STL constraints
#     for i, agent in enumerate(agents[1:]):

#         trajs = agent.sample_trajectories(N, dt, S)
#         traj_mean = trajs.mean(axis=0)
#         d_safe = cfg["stl"][agent.key]

#         if agent.key in ["ambulance"]:
#             cons, delta_x, delta_y = safe_distance_vehicle(
#                 x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
#                 d_safe=d_safe, label=agent.key
#             )
#         else:
#             cons, delta_x, delta_y = safe_distance_walker(
#                 x_var, traj_mean, ego.width, ego.length,
#                 d_safe=d_safe, label=agent.key
#             )

#         constraints += cons
#         deltas[agent.key + "_x"] = delta_x
#         deltas[agent.key + "_y"] = delta_y

#     # x_min, x_max, y_min, y_max = [-46.3, -43.4, 40.9, 74.8]
#     # cons, delta_lane = stay_in_lane(x_var, x_min, x_max, y_min, y_max, N)
#     # constraints += cons
#     # constraints.append(delta_lane <= 4)
#     # deltas["lane"] = delta_lane

#     y_exit = 0
#     cons, delta_inter = clear_intersection(x_var, y_exit, N)
#     constraints += cons
#     deltas["intersection"] = delta_inter

#     # control deviation from nominal
#     traj_cost = cp.norm(x_var - X_nom.T, 1)

#     # control rate — penalize change between consecutive controls
#     control_rate = 0
#     for k in range(N - 1):
#         control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

#     eps = 1e-2  
#     # objective = cp.Minimize(sum(deltas.values()) + eps * (control_rate + traj_cost))
#     objective = cp.Minimize(sum(deltas.values()))

#     prob = cp.Problem(objective, constraints)

#     num_constraints = sum(c.size for c in constraints)
#     num_variables = sum(v.size for v in prob.variables())
#     print(f"  Problem size: {num_constraints} constraints, {num_variables} variables")

#     t_build = time.perf_counter() - t_build_start

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

#     t_solve_start = time.perf_counter()
#     prob.solve(solver=solver, verbose=False)
#     t_solve = time.perf_counter() - t_solve_start

#     if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:

#         print(f"Warning: solver returned status '{prob.status}', apply fallback control_fallback to come to stop")

#         control_fallback = carla.VehicleControl()
#         control_fallback.throttle = 0.0
#         control_fallback.brake = 0.5
#         control_fallback.steer = 0.0
#         control_fallback.manual_gear_shift = False

#         return {
#             "status": False,
#             "control": control_fallback,
#             "deltas": None,
#             "t_build": t_build, 
#             "t_solve": t_solve,
#             "num_constraints": num_constraints,
#             "num_variables": num_variables,
#         }

#     # draw ego planned trajectory
#     ego_traj = x_var.value[:2, :].T  # (N+1, 2) — extract px, py
#     draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

#     a, beta = u_var.value[:, 0]
#     control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

#     delta_values = {key: float(d.value) for key, d in deltas.items()}
#     print(", ".join(f"{key}: {val:.3f}" for key, val in delta_values.items()))

#     return {
#         "status": True,
#         "control": control,
#         "deltas": None,
#         "t_build": t_build, 
#         "t_solve": t_solve,
#         "num_constraints": num_constraints,
#         "num_variables": num_variables,
#     }


# def solve_mpc_pareto(client, agents, cfg):
    
#     # extract parameters
#     T = cfg["mpc"]["horizon"]
#     S = cfg["mpc"]["num_samples"]
#     dt = cfg["carla"]["dt"]
#     N = int(round(T / dt))
#     lt = dt * 1.5

#     # build bicycle model
#     ego = agents[0]
#     model = KinematicBicycle(lr=ego.lr, dt=dt)

#     # get ego's current state
#     tf = ego.get_transform()
#     vel = ego.get_velocity()
#     ego_init = np.array([
#         tf.location.x,
#         tf.location.y,
#         math.radians(tf.rotation.yaw),
#         math.sqrt(vel.x**2 + vel.y**2)
#     ])

#     # get nominal control from autopilot + planned waypoints
#     control_nom = ego.agent.run_step()
#     a_nom, beta_nom = carla_to_bicycle(control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

#     plan = list(ego.agent.get_local_planner().get_plan())

#     U_nom = np.zeros((N, 2))

#     for k in range(N):

#         if k < len(plan) - 1:

#             wp_curr = plan[k][0].transform
#             wp_next = plan[k + 1][0].transform

#             dx = wp_next.location.x - wp_curr.location.x
#             dy = wp_next.location.y - wp_curr.location.y
#             yaw_next = math.atan2(dy, dx)

#             if k == 0:
#                 dyaw = yaw_next - ego_init[2]

#             else:
#                 wp_prev = plan[k - 1][0].transform
#                 dx_p = wp_curr.location.x - wp_prev.location.x
#                 dy_p = wp_curr.location.y - wp_prev.location.y
#                 yaw_prev = math.atan2(dy_p, dx_p)
#                 dyaw = yaw_next - yaw_prev

#             # estimate beta from heading change
#             v_est = max(ego_init[3] + a_nom * k * dt, 0.5)
#             beta_k = dyaw * ego.lr / (v_est * dt)
#             beta_k = max(ego.beta_min, min(beta_k, ego.beta_max))

#             U_nom[k] = [a_nom, beta_k]

#         else:
#             U_nom[k] = [a_nom, beta_nom]

#     # nominal trajectory and linearization
#     X_nom = np.zeros((N + 1, 4), dtype=float)
#     X_nom[0] = ego_init.copy()
#     A_seq, B_seq, c_seq = [], [], []

#     for k in range(N):

#         A_k, B_k = model.linearize(X_nom[k], U_nom[k])
#         X_nom[k + 1] = model.step(X_nom[k], U_nom[k])
#         c_k = X_nom[k + 1] - A_k @ X_nom[k] - B_k @ U_nom[k]
#         A_seq.append(A_k)
#         B_seq.append(B_k)
#         c_seq.append(c_k)

#     # draw_sample_traj(client.world, X_nom[:, :2], color=COLORS["white"], life_time=lt)
    
#     t_build_start = time.perf_counter()

#     x_var = cp.Variable((4, N + 1), name="x")
#     u_var = cp.Variable((2, N), name="u")

#     constraints = []
#     constraints.append(x_var[:, 0] == ego_init)

#     # dynamics and control constraints
#     for k in range(N):

#         constraints.append(
#             x_var[:, k + 1] == A_seq[k] @ x_var[:, k] + B_seq[k] @ u_var[:, k] + c_seq[k]
#         )

#         constraints += [
#             u_var[0, k] >= ego.acc_min,
#             u_var[0, k] <= ego.acc_max,
#             u_var[1, k] >= ego.beta_min,
#             u_var[1, k] <= ego.beta_max,
#         ]

#     deltas = {}

#     # STL constraints
#     for i, agent in enumerate(agents[1:]):

#         trajs = agent.sample_trajectories(N, dt, S)
#         traj_mean = trajs.mean(axis=0)
#         d_safe = cfg["stl"][agent.key]

#         if agent.key in ["ambulance"]:
#             cons, delta_x, delta_y = safe_distance_vehicle(
#                 x_var, traj_mean, ego.width, ego.length, agent.width, agent.length,
#                 d_safe=d_safe, label=agent.key
#             )
#         else:
#             cons, delta_x, delta_y = safe_distance_walker(
#                 x_var, traj_mean, ego.width, ego.length,
#                 d_safe=d_safe, label=agent.key
#             )

#         constraints += cons
#         deltas[agent.key + "_x"] = delta_x
#         deltas[agent.key + "_y"] = delta_y

#     # x_min, x_max, y_min, y_max = [-46.3, -43.4, 40.9, 74.8]
#     # cons, delta_lane = stay_in_lane(x_var, x_min, x_max, y_min, y_max, N)
#     # constraints += cons
#     # constraints.append(delta_lane <= 4)
#     # deltas["lane"] = delta_lane

#     y_exit = 0
#     cons, delta_inter = clear_intersection(x_var, y_exit, N)
#     constraints += cons
#     deltas["intersection"] = delta_inter

#     # control deviation from nominal
#     traj_cost = cp.norm(x_var - X_nom.T, 1)

#     # control rate — penalize change between consecutive controls
#     control_rate = 0
#     for k in range(N - 1):
#         control_rate += cp.norm(u_var[:, k+1] - u_var[:, k], 1)

#     eps = 1e-2  
#     # objective = cp.Minimize(sum(deltas.values()) + eps * (control_rate + traj_cost))
#     objective = cp.Minimize(sum(deltas.values()))

#     prob = cp.Problem(objective, constraints)

#     num_constraints = sum(c.size for c in constraints)
#     num_variables = sum(v.size for v in prob.variables())
#     print(f"  Problem size: {num_constraints} constraints, {num_variables} variables")

#     t_build = time.perf_counter() - t_build_start

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

#     t_solve_start = time.perf_counter()
#     prob.solve(solver=solver, verbose=False)
#     t_solve = time.perf_counter() - t_solve_start

#     if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:

#         print(f"Warning: solver returned status '{prob.status}', apply fallback control_fallback to come to stop")

#         control_fallback = carla.VehicleControl()
#         control_fallback.throttle = 0.0
#         control_fallback.brake = 0.5
#         control_fallback.steer = 0.0
#         control_fallback.manual_gear_shift = False

#         return {
#             "status": False,
#             "control": control_fallback,
#             "deltas": None,
#             "t_build": t_build, 
#             "t_solve": t_solve,
#             "num_constraints": num_constraints,
#             "num_variables": num_variables,
#         }

#     # draw ego planned trajectory
#     ego_traj = x_var.value[:2, :].T  # (N+1, 2) — extract px, py
#     draw_sample_traj(client.world, ego_traj, color=COLORS["blue"], life_time=lt)

#     a, beta = u_var.value[:, 0]
#     control = bicycle_to_carla([a, beta], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

#     delta_values = {key: float(d.value) for key, d in deltas.items()}
#     print(", ".join(f"{key}: {val:.3f}" for key, val in delta_values.items()))

#     return {
#         "status": True,
#         "control": control,
#         "deltas": None,
#         "t_build": t_build, 
#         "t_solve": t_solve,
#         "num_constraints": num_constraints,
#         "num_variables": num_variables,
#     }




import carla
import numpy as np
import cvxpy as cp
import math
import time
import itertools

from src.bicycle import KinematicBicycle
from src.stl import (
    safe_distance_vehicle, safe_distance_walker, clear_intersection
)
from src.utils import (
    draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
)

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (fixed across all calls)
# ──────────────────────────────────────────────────────────────────────────────
_M_EGO = 1850.0   # kg  Tesla Model 3
_M_PED =   70.0   # kg  average adult
_M_AMB = 4500.0   # kg  Ford Ambulance

# Vulnerability weights  (higher = more weight in risk objective)
_V_PED = 1.0      # pedestrian is most vulnerable
_V_AMB = 0.5      # ambulance occupants
_V_EGO = 0.5      # ego occupants


# ──────────────────────────────────────────────────────────────────────────────
# Helper: CVaR LP over K pre-selected scenarios
# ──────────────────────────────────────────────────────────────────────────────

def _build_cvar(x_var, trajs, r_weights, collision_dist, label):
    """
    Build a CVaR expression (convex, no binary variables) over K worst-case
    agent trajectories.

    The α=0.95 selection already happened upstream (worst K out of 100).
    Here we average all K selected scenarios uniformly (α_inner = 0),
    which equals E[loss | top-(1-α) scenarios] = CVaR_0.95 of the full set.

    Soft loss per scenario n:
        loss_n = r_weights[n] * pos(collision_dist - min_k ||ego_k - agent_nk||_inf) 
                                / collision_dist

    The L∞ distance is convex in x_var.
    pos(collision_dist - convex) is convex (nondecreasing outer function).

    CVaR LP (Rockafellar & Uryasev 2000):
        CVaR = τ + (1/K) * Σ s_n
        s.t.  s_n ≥ loss_n − τ,  s_n ≥ 0

    τ is a free scalar; the optimizer sets it to min(losses) automatically.

    Parameters
    ----------
    x_var          : cp.Variable (4, N+1)    ego state trajectory
    trajs          : np.ndarray  (K, N+1, 2) worst-K agent position trajectories
    r_weights      : np.ndarray  (K,)        per-scenario risk weight
    collision_dist : float                   effective safe-distance radius (L∞)
    label          : str                     name prefix for cvxpy variables

    Returns
    -------
    cvar_expr   : cp.Expression  (scalar, convex)
    cvar_cons   : list[cp.Constraint]
    """
    K, T, _ = trajs.shape

    losses = []
    for n in range(K):
        w = float(r_weights[n])
        if w == 0.0:
            losses.append(cp.Constant(0.0))
            continue

        # L∞ distance between ego and agent at each timestep
        l_inf_steps = [
            cp.norm(x_var[:2, k] - trajs[n, k].astype(float), "inf")
            for k in range(T)
        ]

        # closest approach across the horizon (concave in x_var)
        d_min = cp.min(cp.vstack(l_inf_steps))

        # penetration depth (convex): > 0 only when inside collision zone
        hinge = cp.pos(collision_dist - d_min) / collision_dist   # ∈ [0, 1]

        losses.append(w * hinge)

    # CVaR LP variables
    tau = cp.Variable(name=f"tau_{label}")                        # tail threshold
    s   = cp.Variable(K, nonneg=True, name=f"s_{label}")         # exceedances

    # K scenarios, α_inner = 0  →  weight = 1/((1-0)*K) = 1/K
    cvar_cons = [s[n] >= losses[n] - tau for n in range(K)]
    cvar_expr = tau + (1.0 / K) * cp.sum(s)

    return cvar_expr, cvar_cons


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute per-scenario risk weights from nominal ego trajectory
# ──────────────────────────────────────────────────────────────────────────────

def _compute_risk_weights(
    ego_pos_nom,   # (N+1, 2)
    ego_vel_nom,   # (N+1, 2)
    agent_trajs,   # (K, N+1, 2)
    collision_dist,
    mu,            # reduced mass (kg)
    V_agent,       # vulnerability of agent
    V_ego,         # vulnerability of ego
    sev_scale,     # divisor for severity normalisation
    dt,
):
    """
    For each of K agent trajectories evaluate whether the nominal ego path
    enters the collision zone and, if so, estimate:
        severity = mu * ||v_ego − v_agent|| / sev_scale

    Returns r_agent (K,), r_ego (K,) — per-scenario risk contributions.
    These are used as fixed weights in the CVaR expression so that the
    optimiser sees the marginal risk of each scenario, computed on the
    *nominal* trajectory and kept constant during optimisation.
    """
    K = agent_trajs.shape[0]
    r_agent = np.zeros(K)
    r_ego   = np.zeros(K)

    for n in range(K):
        agent_pos = agent_trajs[n, :, :2]                           # (N+1, 2)
        dists     = np.linalg.norm(ego_pos_nom - agent_pos, axis=1) # (N+1,)
        hit_idx   = np.where(dists <= collision_dist)[0]

        if hit_idx.size == 0:
            # no collision on nominal trajectory → small non-zero weight so
            # the scenario still contributes a gradient signal
            r_agent[n] = 0.0
            r_ego[n]   = 0.0
            continue

        t_hit = int(hit_idx[0])

        # agent velocity via finite differences (trajs only carry positions)
        if t_hit > 0:
            v_agent = (agent_pos[t_hit] - agent_pos[t_hit - 1]) / dt
        else:
            v_agent = np.zeros(2)

        rel_v    = ego_vel_nom[t_hit] - v_agent
        severity = mu * float(np.linalg.norm(rel_v)) / sev_scale

        # divide by K so that sum(r_agent) ≈ E[risk] over selected scenarios
        r_agent[n] = severity * V_agent / K
        r_ego[n]   = severity * V_ego   / K

    return r_agent, r_ego


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def solve_mpc_pareto(client, agents, cfg):
    """
    Epsilon-constraint Pareto MPC with CVaR risk objectives.

    Pipeline
    --------
    1.  Build nominal ego trajectory from autopilot plan + bicycle linearisation.
    2.  Sample S=100 trajectories for each agent (amb, ped).
    3.  Select worst K=5 scenarios per agent (CVaR α=0.95 pre-selection)
        ranked by minimum L∞ distance to the nominal ego path.
    4.  Compute fixed per-scenario risk weights on the nominal trajectory.
    5.  For each mode ∈ {ped, ego, amb} and each of `density` random ε samples:
            - minimise that agent's CVaR risk
            - enforce the other two risks as ε-constraints
            - STL soft constraints (safe distance + clear intersection) remain
    6.  Pareto-filter feasible solutions.
    7.  Return the Pareto-optimal solution with minimum control effort.

    agents : [ego (Vehicle), amb (Vehicle), ped (Walker)]
    """

    # ── 0. Unpack config ──────────────────────────────────────────────────────
    T_sim   = cfg["mpc"]["horizon"]
    S       = cfg["mpc"]["num_samples"]     # 100
    dt      = cfg["carla"]["dt"]
    N       = int(round(T_sim / dt))
    lt      = dt * 1.5                      # life-time for CARLA debug draw
    density = cfg["mpc"]["density"]         # epsilon samples per mode

    alpha   = 0.95
    K_tail  = max(1, int(round((1.0 - alpha) * S)))  # 5 worst scenarios

    d_safe_ped = float(cfg["stl"]["pedestrian"])   # 2 m
    d_safe_amb = float(cfg["stl"]["ambulance"])    # 4 m

    ego = agents[0]
    amb = agents[1]
    ped = agents[2]

    # effective L∞ collision radius (accounts for vehicle half-extents)
    coll_dist_ped = ego.width / 2.0 + d_safe_ped
    coll_dist_amb = ego.width / 2.0 + amb.width / 2.0 + d_safe_amb

    # reduced masses
    mu_ped = (_M_EGO * _M_PED) / (_M_EGO + _M_PED)
    mu_amb = (_M_EGO * _M_AMB) / (_M_EGO + _M_AMB)

    # ── 1. Ego initial state ──────────────────────────────────────────────────
    tf  = ego.get_transform()
    vel = ego.get_velocity()
    ego_init = np.array([
        tf.location.x,
        tf.location.y,
        math.radians(tf.rotation.yaw),
        math.sqrt(vel.x**2 + vel.y**2),
    ])

    # ── 2. Nominal control sequence from autopilot waypoint plan ─────────────
    model       = KinematicBicycle(lr=ego.lr, dt=dt)
    control_nom = ego.agent.run_step()
    a_nom, beta_nom = carla_to_bicycle(
        control_nom, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
    )
    plan    = list(ego.agent.get_local_planner().get_plan())
    U_nom   = np.zeros((N, 2))

    for k in range(N):
        if k < len(plan) - 1:
            wp_c = plan[k][0].transform
            wp_n = plan[k + 1][0].transform
            dx   = wp_n.location.x - wp_c.location.x
            dy   = wp_n.location.y - wp_c.location.y
            yaw_n = math.atan2(dy, dx)
            if k == 0:
                dyaw = yaw_n - ego_init[2]
            else:
                wp_p  = plan[k - 1][0].transform
                yaw_p = math.atan2(
                    wp_c.location.y - wp_p.location.y,
                    wp_c.location.x - wp_p.location.x
                )
                dyaw = yaw_n - yaw_p
            v_est  = max(ego_init[3] + a_nom * k * dt, 0.5)
            beta_k = np.clip(dyaw * ego.lr / (v_est * dt), ego.beta_min, ego.beta_max)
            U_nom[k] = [a_nom, beta_k]
        else:
            U_nom[k] = [a_nom, beta_nom]

    # ── 3. Nominal trajectory + linearisation ────────────────────────────────
    X_nom             = np.zeros((N + 1, 4))
    X_nom[0]          = ego_init.copy()
    A_seq, B_seq, c_seq = [], [], []

    for k in range(N):
        A_k, B_k      = model.linearize(X_nom[k], U_nom[k])
        X_nom[k + 1]  = model.step(X_nom[k], U_nom[k])
        c_k           = X_nom[k + 1] - A_k @ X_nom[k] - B_k @ U_nom[k]
        A_seq.append(A_k)
        B_seq.append(B_k)
        c_seq.append(c_k)

    ego_pos_nom = X_nom[:, :2]                              # (N+1, 2)
    ego_vel_nom = np.stack([                                # (N+1, 2)
        X_nom[:, 3] * np.cos(X_nom[:, 2]),
        X_nom[:, 3] * np.sin(X_nom[:, 2]),
    ], axis=1)

    # ── 4. Sample 100 trajectories per agent ─────────────────────────────────
    print(f"  Sampling {S} trajectories per agent...")
    ped_trajs_all = ped.sample_trajectories(N, dt, S)   # (S, N+1, 2)
    amb_trajs_all = amb.sample_trajectories(N, dt, S)   # (S, N+1, 2)

    # ── 5. Select worst K_tail scenarios (CVaR α=0.95 pre-selection) ─────────
    # Worst = smallest minimum L∞ distance to nominal ego across the horizon.
    # These are the scenarios most likely to produce a collision.

    def _worst_k(ego_pos, agent_trajs_all, k_tail):
        diffs      = np.abs(agent_trajs_all - ego_pos[np.newaxis])  # (S, N+1, 2)
        l_inf      = diffs.max(axis=2)                               # (S, N+1)
        min_dist   = l_inf.min(axis=1)                               # (S,) per scenario
        worst_idx  = np.argsort(min_dist)[:k_tail]                  # ascending → closest
        return worst_idx, min_dist[worst_idx]

    ped_idx, ped_dists = _worst_k(ego_pos_nom, ped_trajs_all, K_tail)
    amb_idx, amb_dists = _worst_k(ego_pos_nom, amb_trajs_all, K_tail)

    ped_trajs = ped_trajs_all[ped_idx]   # (K_tail, N+1, 2)
    amb_trajs = amb_trajs_all[amb_idx]   # (K_tail, N+1, 2)

    print(f"  Ped worst-{K_tail} min dists: {np.round(ped_dists, 2).tolist()}")
    print(f"  Amb worst-{K_tail} min dists: {np.round(amb_dists, 2).tolist()}")

    # ── 6. Compute per-scenario risk weights (fixed, from nominal trajectory) ─
    # r_ped[n]  : marginal pedestrian risk if scenario n materialises
    # r_amb[n]  : marginal ambulance risk
    # r_ego_p[n]: marginal ego risk from ped collision in scenario n
    # r_ego_a[n]: marginal ego risk from amb collision in scenario n

    r_ped,   r_ego_p = _compute_risk_weights(
        ego_pos_nom, ego_vel_nom,
        ped_trajs, coll_dist_ped,
        mu_ped, _V_PED, _V_EGO,
        sev_scale=1000.0, dt=dt,
    )
    r_amb,   r_ego_a = _compute_risk_weights(
        ego_pos_nom, ego_vel_nom,
        amb_trajs, coll_dist_amb,
        mu_amb, _V_AMB, _V_EGO,
        sev_scale=50.0, dt=dt,
    )

    print(f"  r_ped weights:   {np.round(r_ped,   6).tolist()}")
    print(f"  r_amb weights:   {np.round(r_amb,   6).tolist()}")
    print(f"  r_ego_ped:       {np.round(r_ego_p, 6).tolist()}")
    print(f"  r_ego_amb:       {np.round(r_ego_a, 6).tolist()}")

    # ── 7. Solver selection (done once) ──────────────────────────────────────
    solver = None
    for _s in [cp.GUROBI, cp.CPLEX, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]:
        if _s in cp.installed_solvers():
            solver = _s
            break
    if solver is None:
        raise RuntimeError(
            "No MIP solver available. Install GUROBI, CPLEX, GLPK, or SCIP.\n"
            f"Installed: {cp.installed_solvers()}"
        )

    # ── 8. Inner build-and-solve for one (mode, ε) pair ──────────────────────

    def _solve_one(mode, eps_ped, eps_ego, eps_amb):
        """
        Build and solve one epsilon-constraint MILP.

        mode     : "ped" | "ego" | "amb"  → which risk is minimised
        eps_*    : upper bound on non-minimised risks (np.inf = unconstrained)

        Returns a result dict or None if infeasible.
        """
        t0_build = time.perf_counter()

        x_var = cp.Variable((4, N + 1), name="x")
        u_var = cp.Variable((2, N),     name="u")

        cons = []

        # initial condition
        cons.append(x_var[:, 0] == ego_init)

        # linearised dynamics + control bounds
        for k in range(N):
            cons.append(
                x_var[:, k + 1] == A_seq[k] @ x_var[:, k]
                                  + B_seq[k] @ u_var[:, k]
                                  + c_seq[k]
            )
            cons += [
                u_var[0, k] >= ego.acc_min,
                u_var[0, k] <= ego.acc_max,
                u_var[1, k] >= ego.beta_min,
                u_var[1, k] <= ego.beta_max,
            ]

        # ── STL soft constraints (binary, structural) ────────────────────────
        # Use the mean of the worst-K scenarios as the representative trajectory.
        # delta variables absorb any unavoidable violation.
        stl_deltas = {}

        ped_mean = ped_trajs.mean(axis=0)                       # (N+1, 2)
        c_ped, dx_ped, dy_ped = safe_distance_walker(
            x_var, ped_mean, ego.width, ego.length,
            d_safe=d_safe_ped, label="pedestrian",
        )
        cons += c_ped
        stl_deltas["ped_x"] = dx_ped
        stl_deltas["ped_y"] = dy_ped

        amb_mean = amb_trajs.mean(axis=0)                       # (N+1, 2)
        c_amb, dx_amb, dy_amb = safe_distance_vehicle(
            x_var, amb_mean,
            ego.width, ego.length, amb.width, amb.length,
            d_safe=d_safe_amb, label="ambulance",
        )
        cons += c_amb
        stl_deltas["amb_x"] = dx_amb
        stl_deltas["amb_y"] = dy_amb

        y_exit = 0.0
        c_int, d_int = clear_intersection(x_var, y_exit, N)
        cons += c_int
        stl_deltas["intersection"] = d_int

        stl_penalty = sum(stl_deltas.values())

        # ── CVaR risk expressions (convex, no binary variables) ──────────────
        # Pedestrian CVaR risk
        r_ped_cvar, c_rp = _build_cvar(
            x_var, ped_trajs, r_ped, coll_dist_ped, "ped"
        )
        # Ambulance CVaR risk
        r_amb_cvar, c_ra = _build_cvar(
            x_var, amb_trajs, r_amb, coll_dist_amb, "amb"
        )
        # Ego CVaR risk = contribution from ped scenarios + amb scenarios
        r_ego_ped_cvar, c_rep = _build_cvar(
            x_var, ped_trajs, r_ego_p, coll_dist_ped, "ego_ped"
        )
        r_ego_amb_cvar, c_rea = _build_cvar(
            x_var, amb_trajs, r_ego_a, coll_dist_amb, "ego_amb"
        )
        r_ego_cvar = r_ego_ped_cvar + r_ego_amb_cvar

        cons += c_rp + c_ra + c_rep + c_rea

        # ── Epsilon constraints on the non-minimised objectives ──────────────
        if mode != "ped" and not np.isinf(eps_ped):
            cons.append(r_ped_cvar <= eps_ped)
        if mode != "ego" and not np.isinf(eps_ego):
            cons.append(r_ego_cvar <= eps_ego)
        if mode != "amb" and not np.isinf(eps_amb):
            cons.append(r_amb_cvar <= eps_amb)

        # ── Objective ────────────────────────────────────────────────────────
        # Primary: minimise chosen CVaR risk
        # Secondary (small weight): penalise STL slack so trajectory stays
        #   feasible w.r.t. safe distance even when risk itself is zero
        W_STL = 1e-3

        if mode == "ped":
            obj = cp.Minimize(r_ped_cvar + W_STL * stl_penalty)
        elif mode == "ego":
            obj = cp.Minimize(r_ego_cvar + W_STL * stl_penalty)
        else:
            obj = cp.Minimize(r_amb_cvar + W_STL * stl_penalty)

        prob = cp.Problem(obj, cons)

        n_cons = sum(c.size for c in cons)
        n_vars = sum(v.size for v in prob.variables())
        t_build = time.perf_counter() - t0_build

        t0_solve = time.perf_counter()
        prob.solve(solver=solver, verbose=False)
        t_solve = time.perf_counter() - t0_solve

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        return {
            "status":          prob.status,
            "mode":            mode,
            "x_opt":           x_var.value,
            "u_opt":           u_var.value,
            "r_ped":           float(r_ped_cvar.value),
            "r_ego":           float(r_ego_cvar.value),
            "r_amb":           float(r_amb_cvar.value),
            "stl_deltas":      {k: float(v.value) for k, v in stl_deltas.items()
                                 if v.value is not None},
            "t_build":         t_build,
            "t_solve":         t_solve,
            "num_constraints": n_cons,
            "num_variables":   n_vars,
        }

    # ── 9. Epsilon-constraint sweep ───────────────────────────────────────────
    # For each mode, sample `density` random epsilon vectors and solve.
    # The epsilon for the minimised objective is set to ∞ (unconstrained).
    # The other two are sampled uniformly in [0, eps_max].
    #
    # eps_max is set from the nominal-trajectory risks so epsilons are
    # in a meaningful range rather than an arbitrary [0, 1].

    r_nom_ped = float(np.sum(r_ped))    # total ped risk on nominal trajectory
    r_nom_amb = float(np.sum(r_amb))
    r_nom_ego = float(np.sum(r_ego_p) + np.sum(r_ego_a))

    # use 2× nominal as upper bound for epsilon range
    eps_max_ped = max(r_nom_ped * 2.0, 1e-4)
    eps_max_amb = max(r_nom_amb * 2.0, 1e-4)
    eps_max_ego = max(r_nom_ego * 2.0, 1e-4)

    rng     = np.random.default_rng(seed=cfg["project"]["seed"])
    results = []

    for mode in ["ped", "ego", "amb"]:
        print(f"\n  ── Mode: minimise r_{mode} ──")
        for i in range(density):

            eps_ped = np.inf if mode == "ped" else rng.uniform(0, eps_max_ped)
            eps_ego = np.inf if mode == "ego" else rng.uniform(0, eps_max_ego)
            eps_amb = np.inf if mode == "amb" else rng.uniform(0, eps_max_amb)

            print(
                f"    [{mode} | ε-sample {i+1}/{density}] "
                f"ε_ped={eps_ped:.4f}  ε_ego={eps_ego:.4f}  ε_amb={eps_amb:.4f}",
                end="  ... ",
                flush=True,
            )

            sol = _solve_one(mode, eps_ped, eps_ego, eps_amb)

            if sol is None:
                print("INFEASIBLE")
            else:
                print(
                    f"OK  r_ped={sol['r_ped']:.5f}  "
                    f"r_ego={sol['r_ego']:.5f}  "
                    f"r_amb={sol['r_amb']:.5f}  "
                    f"({sol['t_solve']:.2f}s)"
                )
                results.append(sol)

    # ── 10. Pareto filter ────────────────────────────────────────────────────
    if not results:
        print("  All solves infeasible — applying emergency braking.")
        fb = carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0,
                                  manual_gear_shift=False)
        return {
            "status": False, "control": fb,
            "deltas": None, "t_build": 0.0, "t_solve": 0.0,
            "num_constraints": None, "num_variables": None,
        }

    pts      = np.array([[r["r_ped"], r["r_ego"], r["r_amb"]] for r in results])
    n_sol    = len(pts)
    is_ndom  = np.ones(n_sol, dtype=bool)

    for i in range(n_sol):
        for j in range(n_sol):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_ndom[i] = False
                break

    pareto = [results[i] for i in range(n_sol) if is_ndom[i]]
    print(f"\n  Pareto front: {len(pareto)} / {n_sol} solutions")

    # among Pareto-optimal solutions pick minimum control effort
    best = min(pareto, key=lambda r: float(np.linalg.norm(r["u_opt"])))

    # ── 11. CARLA debug draw ─────────────────────────────────────────────────
    ego_planned = best["x_opt"][:2, :].T                   # (N+1, 2)
    draw_sample_traj(client.world, ego_planned,
                     color=COLORS["blue"], life_time=lt)
    # draw_sample_traj(client.world, ped_trajs,
    #                  color=COLORS["green"], life_time=lt)
    # draw_sample_traj(client.world, amb_trajs,
    #                  color=COLORS["red"],   life_time=lt)

    # ── 12. Convert first control step to CARLA format ───────────────────────
    a_opt, beta_opt = best["u_opt"][:, 0]
    control = bicycle_to_carla(
        [a_opt, beta_opt],
        ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max,
    )

    print(
        f"\n  Best: r_ped={best['r_ped']:.5f}  "
        f"r_ego={best['r_ego']:.5f}  r_amb={best['r_amb']:.5f}  "
        f"mode={best['mode']}  "
        f"deltas={best['stl_deltas']}"
    )

    return {
        "status":          True,
        "control":         control,
        "deltas":          best["stl_deltas"],
        "t_build":         best["t_build"],
        "t_solve":         best["t_solve"],
        "num_constraints": best["num_constraints"],
        "num_variables":   best["num_variables"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point for build_and_solve_mpc (dispatcher used in exp1.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_and_solve_mpc(client, agents, cfg):
    mpc_type = cfg["mpc"]["type"]
    if mpc_type == "pareto":
        return solve_mpc_pareto(client, agents, cfg)
    else:
        raise ValueError(f"Unknown mpc.type='{mpc_type}'. Expected 'pareto'.")