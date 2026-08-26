import time
import numpy as np
import cvxpy as cp

from src.utils import draw_sample_traj, bicycle_to_carla, COLORS
from exp2.stl import (
    safe_distance_vehicle, safe_distance_walker, stay_in_lane,
    bounded_deceleration, bounded_steering_rate, safe_distance_box,
)
from exp2.mpc import ego_state, build_nominal, _ENV

_GATE_PARAMS = dict(
    MIPGapAbs=1e-6, MIPGap=0.0,
    IntFeasTol=1e-6, Threads=2, Seed=0,
)

DELTA_TOL = 1e-5


def check_feasibility(client, agents, cfg, emergency, u_prev=None):

    S = cfg["mpc"]["num_samples"]
    dt = cfg["carla"]["dt"]
    N = int(round(cfg["mpc"]["horizon"] / dt))
    lt = dt * 1.5

    d_safe_x = float(cfg["stl"]["d_safe_x"])
    d_safe_y = float(cfg["stl"]["d_safe_y"])
    d_crash = float(cfg["stl"]["d_crash"])
    d_ped = float(cfg["stl"]["d_ped"])
    a_comfort = float(cfg["stl"]["a_comfort"])
    beta_rate = float(cfg["stl"]["beta_rate_max"])

    ego, leader, follower, left, ped2 = agents[0], agents[1], agents[2], agents[3], agents[4]
    z_ego = ego.get_transform().location.z

    x0 = ego_state(ego)
    X_nom, U_nom, A, B, c = build_nominal(ego, dt, N, x0, u_prev=u_prev)

    draw_sample_traj(client.world, X_nom[:, :2], color=COLORS["magenta"], life_time=lt, z=z_ego)

    leader_trajs = leader.sample_trajectories(N, dt, S)
    follower_trajs = follower.sample_trajectories(N, dt, S)
    left_trajs = left.sample_trajectories(N, dt, S)
    ped2_stl = ped2.sample_trajectories(N, dt, S)

    t0 = time.perf_counter()

    x = cp.Variable((4, N + 1), name="x")
    u = cp.Variable((2, N), name="u")
    cons = [x[:, 0] == x0]

    for k in range(N):
        cons += [
            x[:, k+1] == A[k] @ x[:, k] + B[k] @ u[:, k] + c[k],
            u[0, k] >= ego.acc_min, u[0, k] <= ego.acc_max,
            u[1, k] >= ego.beta_min, u[1, k] <= ego.beta_max,
        ]

    deltas = {}

    # always: stay in lane
    c_lane, d_lane = stay_in_lane(
        x, x_min=cfg["stl"]["x_min"], x_max=cfg["stl"]["x_max"],
        y_min=cfg["stl"]["y_min"], y_max=cfg["stl"]["y_max"], N=N,
    )
    cons += c_lane
    cons += [d_lane <= 0]
    deltas["lane"] = d_lane

    # always: bounded deceleration
    c_dec, d_dec = bounded_deceleration(u, a_comfort, N)
    cons += c_dec
    cons += [d_dec <= 0]
    deltas["decel"] = d_dec

    # always: bounded steering rate
    c_sr, d_sr = bounded_steering_rate(u, beta_rate, N)
    cons += c_sr
    cons += [d_sr <= 0]
    deltas["steer"] = d_sr

    # leader
    c_ld, dx_ld, dy_ld = safe_distance_vehicle(
        x, leader_trajs, d_safe_x, d_safe_y, label="leader")
    cons += c_ld
    cons += [dx_ld <= 0, dy_ld <= 0]
    deltas["leader_x"] = dx_ld
    deltas["leader_y"] = dy_ld

    # follower
    c_f, dx_f, dy_f = safe_distance_vehicle(
        x, follower_trajs, d_safe_x, d_safe_y, label="follower")
    cons += c_f
    cons += [dx_f <= 0, dy_f <= 0]
    deltas["follower_x"] = dx_f
    deltas["follower_y"] = dy_f

    if emergency:
        # left vehicle
        c_l, dx_l, dy_l = safe_distance_vehicle(
            x, left_trajs, d_safe_x, d_safe_y, label="left")
        cons += c_l
        cons += [dx_l <= 0, dy_l <= 0]
        deltas["left_x"] = dx_l
        deltas["left_y"] = dy_l

        # ped2
        c_p, d_p = safe_distance_walker(x, ped2_stl, d_ped, label="ped2")
        cons += c_p
        cons += [d_p <= 0]
        deltas["ped2"] = d_p

        # crash scene box
        c_cr, d_cr = safe_distance_box(
            x,
            x_min=cfg["stl"]["crash_x_min"],
            x_max=cfg["stl"]["crash_x_max"],
            y_min=cfg["stl"]["crash_y_min"],
            y_max=cfg["stl"]["crash_y_max"],
            d_safe=d_crash,
        )
        cons += c_cr
        cons += [d_cr <= 0]
        deltas["crash"] = d_cr

    # objective
    objective = cp.Minimize(
        cp.norm(u - U_nom.T, 1)
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
        print(f"  [gate] solver failed -> PARETO")
        return {
            "status": False,
            "t_build": t_build,
            "t_solve": t_solve,
            "num_constraints": n_cons,
            "num_variables": n_vars,
        }

    delta_vals = {k: float(v.value) for k, v in deltas.items() if v.value is not None}
    delta_sum = float(sum(delta_vals.values()))

    print(f"  [gate] delta_sum: {delta_sum:.6f}")
    for k, v in delta_vals.items():
        print(f"         {k:12s}: {v:.6f}")
    print(f"  [gate] -> {'NOMINAL' if delta_sum <= DELTA_TOL else 'PARETO'}")
    print()

    draw_sample_traj(client.world, x.value[:2, :].T, color=COLORS["blue"], life_time=lt, z=z_ego)

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
        "u_applied": [a_opt, beta_opt],
    }