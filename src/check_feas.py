"""
check_feas.py --  whether the min relaxation is zero:
    sum(delta) == 0   ->  STL satisfiable   ->  apply solved control
    sum(delta)  > 0   ->  STL violated      ->  run the Pareto sweep

"""

import math
import time

import numpy as np
import cvxpy as cp

from src.bicycle import KinematicBicycle
from src.stl import safe_distance_vehicle, safe_distance_walker, clear_intersection, stay_in_lane
from src.utils import draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
from src.mpc_pareto import _ego_state, _build_nominal

import gurobipy as gp


DELTA_TOL = 1e-5

def check_feasibility(client, agents, cfg):
    """
    Solve the STL-feasibility MILP for the current tick.
    """
    T_sim   = cfg["mpc"]["horizon"]
    S       = cfg["mpc"]["num_samples"]     # 100
    dt      = cfg["carla"]["dt"]
    N       = int(round(T_sim / dt))
    density = cfg["mpc"]["density"]
    lt      = dt * 1.5

    d_ped   = float(cfg["stl"]["pedestrian"])
    d_amb   = float(cfg["stl"]["ambulance"])

    ego, amb, ped = agents[0], agents[1], agents[2]

    # ── 1. Ego state + nominal trajectory ─────────────────────────────────────
    model    = KinematicBicycle(lr=ego.lr, dt=dt)
    ego_init = _ego_state(ego)
    X_nom, U_nom, A_seq, B_seq, c_seq = _build_nominal(ego, model, dt, N, ego_init)

    ego_pos_nom = X_nom[:, :2]                                       
    ego_vel_nom = np.stack([X_nom[:,3]*np.cos(X_nom[:,2]),           
                            X_nom[:,3]*np.sin(X_nom[:,2])], axis=1)

    # ── 2. Sample 100 trajectories
    ped_trajs = ped.sample_trajectories(N, dt, S*10)                            
    amb_trajs = amb.sample_trajectories(N, dt, S*10)                                

    t0 = time.perf_counter()

    x_var = cp.Variable((4, N+1), name="x")
    u_var = cp.Variable((2, N),   name="u")
    cons  = [x_var[:, 0] == ego_init]

    # linearised dynamics + control bounds (identical for every solve)
    for k in range(N):
        cons += [x_var[:,k+1] == A_seq[k]@x_var[:,k] + B_seq[k]@u_var[:,k] + c_seq[k],
                 u_var[0,k] >= ego.acc_min,  u_var[0,k] <= ego.acc_max,
                 u_var[1,k] >= ego.beta_min, u_var[1,k] <= ego.beta_max]

    # STL soft constraints (structural; shared across all solves)
    deltas = {}
    c, d_ped_stl = safe_distance_walker(
        x_var, ped_trajs.mean(axis=0), ego.width, ego.length, d_ped, label="ped")
    cons += c
    deltas["ped"] = d_ped_stl

    c, d_amb_stl = safe_distance_vehicle(
        x_var, amb_trajs.mean(axis=0), ego.width, ego.length,
        amb.width, amb.length, d_amb, label="amb")
    cons += c
    deltas["amb"] = d_amb_stl

    c, d_lane = stay_in_lane(x_var, x_min=-45.5, x_max=-44, y_min=0, y_max=100, N=N)
    cons += c 
    deltas["lane"] = d_lane

    c, d_int = clear_intersection(x_var, y_exit=0.0, N=N)
    cons += c 
    deltas["inter"] = d_int

    objective = cp.Minimize(sum(deltas.values()))
    prob = cp.Problem(objective, cons)

    n_cons  = sum(c.size for c in cons)
    n_vars  = sum(v.size for v in prob.variables())
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()

    prob.solve(solver=cp.GUROBI)

    t_solve = time.perf_counter() - t1

    print(f"t_solve: {t_solve:.3f}, n_cons: {n_cons}, n_vars: {n_vars}")

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Feasibility checking status: [{prob.status}]", end=" ")
        return None

    a, beta = u_var.value[:, 0]
    control = bicycle_to_carla([a, beta],
                               ego.acc_min, ego.acc_max,
                               ego.beta_min, ego.beta_max)

    delta_vals = {k: float(v.value) for k, v in deltas.items()
                  if v.value is not None}
    delta_sum  = float(sum(delta_vals.values()))
    feasible   = delta_sum <= DELTA_TOL
    
    print(f"delta_sum: {delta_sum:.2f}\n")
    for k, v in delta_vals.items():
        print(f"  {k:8s}: {v:.2f}")
    print(f"\nfeasible: {feasible}")

    if feasible:
        draw_sample_traj(client.world, x_var.value[:2, :].T,
                     color=COLORS["blue"],  life_time=lt)

    return {
        "status":          feasible,
        "control":         control,
        "deltas":          deltas,
        "t_build":         t_build,
        "t_solve":         t_solve,
        "num_constraints": n_cons,
        "num_variables":   n_vars,
    }


    