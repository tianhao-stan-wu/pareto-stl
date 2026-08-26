import itertools
import carla
import numpy as np
import cvxpy as cp
import math
import time
import gurobipy as gp

from src.bicycle import KinematicBicycle
from src.utils import draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
from exp2.stl import (
    safe_distance_vehicle, safe_distance_walker, stay_in_lane,
    bounded_deceleration, bounded_steering_rate, safe_distance_box,
    _promote,
)

_ENV = gp.Env(params={"OutputFlag": 0})

_SOLVER_PARAMS = dict(
    TimeLimit=10.0, MIPGap=0.0,
    IntFeasTol=1e-7, NumericFocus=2, Presolve=2,
    MIPFocus=1, Heuristics=0.1, Cuts=2,
    Threads=4, Seed=0,
)


def ego_state(ego):
    tf = ego.get_transform()
    v = ego.get_velocity()
    return np.array([
        tf.location.x, tf.location.y,
        math.radians(tf.rotation.yaw),
        math.sqrt(v.x**2 + v.y**2),
    ])


def build_nominal(ego, dt, N, x0, u_prev=None):
    model = KinematicBicycle(lr=ego.lr, dt=dt)

    # use previous acceleration, but always zero steering
    a_nom = u_prev[0] if u_prev is not None else 0.0

    U = np.zeros((N, 2))
    X = np.zeros((N + 1, 4))
    X[0] = x0.copy()
    ROAD_YAW = math.radians(-180.0)   # −π, heading in −x direction
    LANE_Y   = 5.5

    X[0, 1] = LANE_Y
    X[0, 2] = ROAD_YAW

    for k in range(N):
        U[k] = [a_nom, 0.0]
        X[k + 1] = model.step(X[k], U[k])

    A_list, B_list, c_list = [], [], []
    for k in range(N):
        A, B = model.linearize(X[k], U[k])
        c = X[k + 1] - A @ X[k] - B @ U[k]
        A_list.append(A)
        B_list.append(B)
        c_list.append(c)

    return X, U, A_list, B_list, c_list


def encode_collision_vehicle(x_var, trajs, d_safe_x, d_safe_y, label, M=500):
    K, T, _ = trajs.shape

    pp = cp.Variable((K, T), boolean=True, name=f"pp_{label}")
    pn = cp.Variable((K, T), boolean=True, name=f"pn_{label}")
    qp = cp.Variable((K, T), boolean=True, name=f"qp_{label}")
    qn = cp.Variable((K, T), boolean=True, name=f"qn_{label}")
    z = cp.Variable(K, boolean=True, name=f"z_{label}")

    PX = _promote(x_var[0, :T], K)
    PY = _promote(x_var[1, :T], K)
    AX = trajs[:, :T, 0].astype(float)
    AY = trajs[:, :T, 1].astype(float)
    DX = PX - AX
    DY = PY - AY

    cons = [
        DX >= d_safe_x - M * (1 - pp),
        -DX >= d_safe_x - M * (1 - pn),
        DY >= d_safe_y - M * (1 - qp),
        -DY >= d_safe_y - M * (1 - qn),
        cp.vstack([z for _ in range(T)]).T >= 1 - pp - pn - qp - qn,
    ]

    prob = cp.sum(z) / float(K)
    return z, prob, cons


def pareto_filter(pts):
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                mask[i] = False
                break
    return mask


def solve_mpc_pareto(client, agents, cfg, emergency, u_prev=None):

    T_sim = cfg["mpc"]["horizon"]
    S = cfg["mpc"]["num_samples"]
    P = cfg["mpc"]["num_prob"]
    dt = cfg["carla"]["dt"]
    N = int(round(T_sim / dt))
    density = cfg["mpc"]["density"]
    lt = dt * 1.5

    d_safe_x = float(cfg["stl"]["d_safe_x"])
    d_safe_y = float(cfg["stl"]["d_safe_y"])
    d_crash = float(cfg["stl"]["d_crash"])
    d_ped = float(cfg["stl"]["d_ped"])
    a_comfort = float(cfg["stl"]["a_comfort"])
    beta_rate = float(cfg["stl"]["beta_rate_max"])

    crash_box = {
        "x_min": float(cfg["stl"]["crash_x_min"]),
        "x_max": float(cfg["stl"]["crash_x_max"]),
        "y_min": float(cfg["stl"]["crash_y_min"]),
        "y_max": float(cfg["stl"]["crash_y_max"]),
    }

    ego, leader, follower, left, ped2 = agents[0], agents[1], agents[2], agents[3], agents[4]
    z_ego = ego.get_transform().location.z

    x0 = ego_state(ego)
    X_nom, U_nom, A, B, c = build_nominal(ego, dt, N, x0, u_prev=u_prev)

    # sample trajectories for STL
    follower_stl = follower.sample_trajectories(N, dt, S)
    left_stl = left.sample_trajectories(N, dt, S)
    ped2_stl = ped2.sample_trajectories(N, dt, S)

    # sample trajectories for probability encoding
    follower_prob = follower.sample_trajectories(N, dt, P)
    left_prob = left.sample_trajectories(N, dt, P)
    ped2_prob = ped2.sample_trajectories(N, dt, P)

    # draw_sample_traj(client.world, ped2_prob, color=COLORS["green"], life_time=lt, z=z_ego)
    # draw_sample_traj(client.world, left_prob, color=COLORS["red"], life_time=lt, z=z_ego)
    # draw_sample_traj(client.world, follower_prob, color=COLORS["red"], life_time=lt, z=z_ego)

    # epsilon grid: uniform for all three objectives
    eps_grid = np.linspace(0, 1, density + 1)[1:]

    modes = ["follower", "left", "ped2"]

    free_axes = {
        "follower": ["left", "ped2"],
        "left": ["follower", "ped2"],
        "ped2": ["follower", "left"],
    }

    t_sweep_start = time.perf_counter()

    def solve_one(mode, eps_dict, warm=None):

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

        # always: lane, deceleration, steering rate
        c_lane, d_lane = stay_in_lane(
            x, x_min=cfg["stl"]["x_min"], x_max=cfg["stl"]["x_max"],
            y_min=cfg["stl"]["y_min"], y_max=cfg["stl"]["y_max"], N=N,
        )
        cons += c_lane
        cons += [d_lane <= 2.5]
        deltas["lane"] = d_lane

        c_dec, d_dec = bounded_deceleration(u, a_comfort, N)
        cons += c_dec
        deltas["decel"] = d_dec

        c_sr, d_sr = bounded_steering_rate(u, beta_rate, N)
        cons += c_sr
        deltas["steer"] = d_sr

        # always: follower
        c_f, dx_f, dy_f = safe_distance_vehicle(
            x, follower_stl, d_safe_x, d_safe_y, label="follower")
        cons += c_f
        cons += [dx_f <= d_safe_x, dy_f <= d_safe_y]
        deltas["follower_x"] = dx_f
        deltas["follower_y"] = dy_f

        if emergency:
            # left vehicle
            c_l, dx_l, dy_l = safe_distance_vehicle(
                x, left_stl, d_safe_x, d_safe_y, label="left")
            cons += c_l
            cons += [dx_l <= d_safe_x, dy_l <= d_safe_y]
            deltas["left_x"] = dx_l
            deltas["left_y"] = dy_l

            # ped2
            c_p, d_p = safe_distance_walker(x, ped2_stl, d_ped, label="ped2")
            cons += c_p
            cons += [d_p <= d_ped]
            deltas["ped2"] = d_p

            # crash scene box (STL only, no probability objective)
            c_cr, d_cr = safe_distance_box(
                x, crash_box["x_min"], crash_box["x_max"],
                crash_box["y_min"], crash_box["y_max"], d_crash,
            )
            cons += c_cr
            cons += [d_cr <= d_crash]
            deltas["crash"] = d_cr

        # probability encoding: follower, left, ped2
        z_fo, p_follower, c_pfo = encode_collision_vehicle(
            x, follower_prob, d_safe_x, d_safe_y, "p_follower")
        z_le, p_left, c_ple = encode_collision_vehicle(
            x, left_prob, d_safe_x, d_safe_y, "p_left")
        z_pe, p_ped2, c_ppe = encode_collision_vehicle(
            x, ped2_prob, d_ped, d_ped, "p_ped2")
        cons += c_pfo + c_ple + c_ppe

        probs = {
            "follower": p_follower,
            "left": p_left,
            "ped2": p_ped2,
        }

        for name, eps_val in eps_dict.items():
            cons.append(probs[name] <= float(eps_val))

        ws = 1e-4
        wd = 1e-2
        wp = 1e-6 

        delta_cost = cp.sum(
            cp.hstack([deltas[name] for name in deltas])
        )

        smoothness = (
            cp.norm(cp.diff(u[0, :]), 1)       # acceleration rate (jerk)
            + cp.norm(cp.diff(u[1, :]), 1)     # steering rate
        )       

        all_probs = sum(probs.values())

        cost = ws * smoothness

        # objective = cp.Minimize(probs[mode] + ws * smoothness + wd * delta_cost + wp * all_probs)
        # objective = cp.Minimize(probs[mode] + ws * smoothness + wd * delta_cost)
        objective = cp.Minimize(probs[mode] + cost)
        prob = cp.Problem(objective, cons)

        t_build = time.perf_counter() - t0
        n_cons = sum(ci.size for ci in cons)
        n_vars = sum(vi.size for vi in prob.variables())

        t1 = time.perf_counter()
        if warm is not None:
            x.value = warm["x_opt"]
            u.value = warm["u_opt"]
            prob.solve(solver=cp.GUROBI, env=_ENV, warm_start=True, **_SOLVER_PARAMS)
        else:
            prob.solve(solver=cp.GUROBI, env=_ENV, **_SOLVER_PARAMS)
        t_solve = time.perf_counter() - t1

        print(f"  t_solve: {t_solve:.3f}s  vars: {n_vars}  cons: {n_cons}  [{prob.status}]")

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        return {
            "mode": mode,
            "x_opt": x.value,
            "u_opt": u.value,
            "p_follower": float(p_follower.value),
            "p_left": float(p_left.value),
            "p_ped2": float(p_ped2.value),
            "deltas": {k: float(v.value) for k, v in deltas.items() if v.value is not None},
            "t_build": t_build,
            "t_solve": t_solve,
            "num_constraints": n_cons,
            "num_variables": n_vars,
        }

    # epsilon-constraint sweep: 3 modes x density^2
    results = []
    for mode, axes in free_axes.items():
        combos = list(itertools.product(eps_grid, repeat=len(axes)))
        print(f"\n  -- min p_{mode}  ({len(combos)} grid pts) --")
        warm = None
        for combo in combos:
            eps_dict = dict(zip(axes, combo))
            tag = "  ".join(f"{k}={v:.2f}" for k, v in eps_dict.items())
            print(f"    {tag}")
            sol = solve_one(mode, eps_dict, warm=warm)
            if sol is None:
                print("    INFEASIBLE")
            else:
                warm = sol
                results.append(sol)
                print(f"    p=({sol['p_follower']:.3f}, {sol['p_left']:.3f}, "
                      f"{sol['p_ped2']:.3f})")

    t_solve_all = time.perf_counter() - t_sweep_start

    if not results:
        print("  All infeasible -- emergency braking.")
        return {
            "status": False,
            "control": carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0),
            "u_applied": None,
        }

    # pareto filter over three objectives
    pts = np.array([
        [r["p_follower"], r["p_left"], r["p_ped2"]]
        for r in results
    ])
    mask = pareto_filter(pts)
    pareto = [results[i] for i in range(len(results)) if mask[i]]
    print(f"\n  Pareto: {len(pareto)} / {len(results)} retained")

    best = min(pareto, key=lambda r: np.linalg.norm(r["u_opt"][:, 0] - U_nom[0]))

    draw_sample_traj(client.world, best["x_opt"][:2, :].T,
                     color=COLORS["blue"], life_time=lt, z=z_ego)

    a_opt, beta_opt = best["u_opt"][:, 0]
    control = bicycle_to_carla(
        [a_opt, beta_opt], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
    )

    print(f"  Best [{best['mode']}]: "
          f"p=({best['p_follower']:.3f}, {best['p_left']:.3f}, {best['p_ped2']:.3f})  "
          f"deltas={best['deltas']}")
    print()

    best_idx = next(i for i in range(len(results)) if results[i] is best)

    return {
        "status": True,
        "control": control,
        "deltas": best["deltas"],
        "t_build": best["t_build"],
        "t_solve": best["t_solve"],
        "t_solve_all": t_solve_all,
        "num_constraints": best["num_constraints"],
        "num_variables": best["num_variables"],
        "pareto_log": {
            "all_points": pts.tolist(),
            "pareto_mask": mask.tolist(),
            "selected_idx": int(best_idx),
        },
        "u_applied": [a_opt, beta_opt],
    }