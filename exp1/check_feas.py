# """
# check_feas.py --  whether the min relaxation is zero:
#     sum(delta) == 0   ->  STL satisfiable   ->  apply solved control
#     sum(delta)  > 0   ->  STL violated      ->  run the Pareto sweep

# """

# import math
# import time

# import numpy as np
# import cvxpy as cp

# from src.bicycle import KinematicBicycle
# from src.utils import draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
# from exp1.stl import safe_distance_vehicle, safe_distance_walker, clear_intersection, stay_in_lane
# from exp1.mpc import ego_state, build_nominal

# import gurobipy as gp


# DELTA_TOL = 1e-5

# def check_feasibility(client, agents, cfg, emergency):
#     """
#     Solve the STL-feasibility MILP for the current tick.
#     """
#     T_sim   = cfg["mpc"]["horizon"]
#     S       = cfg["mpc"]["num_samples"]   
#     dt      = cfg["carla"]["dt"]
#     N       = int(round(T_sim / dt))
#     density = cfg["mpc"]["density"]
#     lt      = dt * 1.5

#     d_ped   = float(cfg["stl"]["pedestrian"])
#     d_amb   = float(cfg["stl"]["ambulance"])

#     ego, amb, ped = agents[0], agents[1], agents[2]

#     # ── 1. Ego state + nominal trajectory ─────────────────────────────────────
#     model    = KinematicBicycle(lr=ego.lr, dt=dt)
#     ego_init = ego_state(ego)
#     X_nom, U_nom, A_seq, B_seq, c_seq = build_nominal(ego, model, dt, N, ego_init)

#     ego_pos_nom = X_nom[:, :2]                                       
#     ego_vel_nom = np.stack([X_nom[:,3]*np.cos(X_nom[:,2]),           
#                             X_nom[:,3]*np.sin(X_nom[:,2])], axis=1)

#     ped_trajs = ped.sample_trajectories(N, dt, S)                            
#     amb_trajs = amb.sample_trajectories(N, dt, S)                                

#     t0 = time.perf_counter()

#     x_var = cp.Variable((4, N+1), name="x")
#     u_var = cp.Variable((2, N),   name="u")
#     cons  = [x_var[:, 0] == ego_init]

#     # linearised dynamics + control bounds (identical for every solve)
#     for k in range(N):
#         cons += [x_var[:,k+1] == A_seq[k]@x_var[:,k] + B_seq[k]@u_var[:,k] + c_seq[k],
#                  u_var[0,k] >= ego.acc_min,  u_var[0,k] <= ego.acc_max,
#                  u_var[1,k] >= ego.beta_min, u_var[1,k] <= ego.beta_max]

#     # STL soft constraints (structural; shared across all solves)
#     deltas = {}
#     if emergency:
#         c, d_ped_stl = safe_distance_walker(x_var, ped_trajs, d_ped, label="ped")
#         cons += c
#         cons += [d_ped_stl <= 0]
#         deltas["ped"] = d_ped_stl

#     c, d_amb_stl = safe_distance_vehicle(x_var, amb_trajs, d_amb, label="amb")
#     cons += c
#     cons += [d_amb_stl <= 0]
#     deltas["amb"] = d_amb_stl

#     c, d_lane = stay_in_lane(x_var, x_min=-45.5, x_max=-44, y_min=0, y_max=100, N=N)
#     cons += c 
#     cons += [d_lane <= 0]
#     deltas["lane"] = d_lane

#     # c, d_int = clear_intersection(x_var, y_exit=0.0, N=N)
#     # cons += c 
#     # deltas["inter"] = d_int

#     # objective = cp.Minimize(sum(deltas.values()))
#     objective = cp.Minimize(cp.norm(u_var - U_nom.T, 1)+ cp.norm(x_var - X_nom.T, 1))
#     prob = cp.Problem(objective, cons)

#     n_cons  = sum(c.size for c in cons)
#     n_vars  = sum(v.size for v in prob.variables())

#     t_build = time.perf_counter() - t0

#     t1 = time.perf_counter()

#     prob.solve(solver=cp.GUROBI)

#     t_solve = time.perf_counter() - t1

#     print(f"t_solve: {t_solve:.3f}, n_cons: {n_cons}, n_vars: {n_vars}")

#     if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#         print(f"The MPC-STL problem is infeasible: [{prob.status}]")
#         return {
#         "status": False,
#     }

#     a, beta = u_var.value[:, 0]
#     control = bicycle_to_carla([a, beta],
#                                ego.acc_min, ego.acc_max,
#                                ego.beta_min, ego.beta_max)

#     delta_vals = {k: float(v.value) for k, v in deltas.items()
#                   if v.value is not None}
#     delta_sum  = float(sum(delta_vals.values()))
    
#     print(f"delta_sum: {delta_sum:.2f}\n")
#     for k, v in delta_vals.items():
#         print(f"  {k:8s}: {v:.2f}")
#     print(f"The MPC-STL problem is feasible: [{prob.status}]")

#     draw_sample_traj(client.world, x_var.value[:2, :].T,
#                      color=COLORS["blue"],  life_time=lt)

#     return {
#         "status":          True,
#         "control":         control,
#         "deltas":          deltas,
#         "t_build":         t_build,
#         "t_solve":         t_solve,
#         "num_constraints": n_cons,
#         "num_variables":   n_vars,
#     }



import time
import numpy as np
import cvxpy as cp
import gurobipy as gp

from src.utils import draw_sample_traj, bicycle_to_carla, COLORS
from exp1.stl import safe_distance_vehicle, safe_distance_walker, stay_in_lane
from exp1.mpc import ego_state, build_nominal, _ENV

_GATE_PARAMS = dict(
    MIPGapAbs=1e-6, MIPGap=0.0,
    IntFeasTol=1e-6, Threads=2, Seed=0,
)

DELTA_TOL = 1e-5


def check_feasibility(client, agents, cfg, emergency):

    # config
    S = cfg["mpc"]["num_samples"]
    dt = cfg["carla"]["dt"]
    N = int(round(cfg["mpc"]["horizon"] / dt))
    lt = dt * 1.5

    d_ped = float(cfg["stl"]["pedestrian"])
    d_amb = float(cfg["stl"]["ambulance"])

    ego, amb, ped = agents[0], agents[1], agents[2]

    # nominal trajectory
    x0 = ego_state(ego)
    X_nom, U_nom, A, B, c = build_nominal(ego, dt, N, x0)

    # sample trajectories for STL
    ped_trajs = ped.sample_trajectories(N, dt, S)
    amb_trajs = amb.sample_trajectories(N, dt, S)

    t0 = time.perf_counter()

    x = cp.Variable((4, N + 1), name="x")
    u = cp.Variable((2, N), name="u")
    cons = [x[:, 0] == x0]

    # dynamics and control bounds
    for k in range(N):
        cons += [
            x[:, k+1] == A[k] @ x[:, k] + B[k] @ u[:, k] + c[k],
            u[0, k] >= ego.acc_min, u[0, k] <= ego.acc_max,
            u[1, k] >= ego.beta_min, u[1, k] <= ego.beta_max,
        ]

    # STL constraints, pinned to zero slack
    deltas = {}

    if emergency:
        c_ped, d_ped_stl = safe_distance_walker(x, ped_trajs, d_ped, label="ped")
        cons += c_ped
        cons += [d_ped_stl <= 0]
        deltas["ped"] = d_ped_stl

    c_amb, d_amb_stl = safe_distance_vehicle(x, amb_trajs, d_amb, label="amb")
    cons += c_amb
    cons += [d_amb_stl <= 0]
    deltas["amb"] = d_amb_stl

    c_lane, d_lane = stay_in_lane(x, x_min=-45.5, x_max=-44, y_min=0, y_max=100, N=N)
    cons += c_lane
    cons += [d_lane <= 0]
    deltas["lane"] = d_lane

    # objective: track nominal, no slack term
    W_DELTA = 1e4
    objective = cp.Minimize(
        W_DELTA * sum(deltas.values())
        + cp.norm(u - U_nom.T, 1)
        + cp.norm(x - X_nom.T, 1)
    )
    prob = cp.Problem(objective, cons)

    n_cons = sum(ci.size for ci in cons)
    n_vars = sum(vi.size for vi in prob.variables())
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob.solve(solver=cp.GUROBI, env=_ENV, **_GATE_PARAMS)
    t_solve = time.perf_counter() - t1

    print(f"  [gate] {t_solve:.3f}s  vars: {n_vars}  cons: {n_cons}  [{prob.status}]")

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"  [gate] infeasible -> PARETO")
        return {"status": False}

    delta_vals = {k: float(v.value) for k, v in deltas.items() if v.value is not None}
    delta_sum = float(sum(delta_vals.values()))

    print(f"  [gate] delta_sum: {delta_sum:.6f}")
    for k, v in delta_vals.items():
        print(f"         {k:8s}: {v:.6f}")
    print(f"  [gate] -> {'NOMINAL' if delta_sum <= DELTA_TOL else 'PARETO'}")

    draw_sample_traj(client.world, x.value[:2, :].T, color=COLORS["blue"], life_time=lt)

    a_opt, beta_opt = u.value[:, 0]
    control = bicycle_to_carla(
        [a_opt, beta_opt], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
    )

    return {
        "status": delta_sum <= DELTA_TOL,
        "control": control,
        "deltas": delta_vals,
        "t_build": t_build,
        "t_solve": t_solve,
        "num_constraints": n_cons,
        "num_variables": n_vars,
    }

    