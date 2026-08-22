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
    safe_distance_vehicle, stay_in_lane,
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


def build_nominal(ego, dt, N, x0):
    model = KinematicBicycle(lr=ego.lr, dt=dt)
    ctrl = ego.agent.run_step()
    a0, _ = carla_to_bicycle(ctrl, ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max)

    U = np.zeros((N, 2))
    for k in range(N):
        U[k] = [a0, 0.0]

    X = np.zeros((N + 1, 4))
    X[0] = x0.copy()
    A_list, B_list, c_list = [], [], []

    for k in range(N):
        A, B = model.linearize(X[k], U[k])
        X[k + 1] = model.step(X[k], U[k])
        c = X[k + 1] - A @ X[k] - B @ U[k]
        A_list.append(A)
        B_list.append(B)
        c_list.append(c)

    return X, U, A_list, B_list, c_list


def encode_collision_vehicle(x_var, trajs, d_safe_x, d_safe_y, label, M=500):
    """
    P = (1/K) * sum(z_n). Rectangular keep-out box with separate x/y margins.
    """
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


def encode_collision_box(x_var, x_min, x_max, y_min, y_max, d_safe, label, M=500):
    """
    P = z (0 or 1). Ego enters the inflated static box at any step -> z = 1.
    """
    lx, ux = x_min - d_safe, x_max + d_safe
    ly, uy = y_min - d_safe, y_max + d_safe
    T = x_var.shape[1]

    pp = cp.Variable(T, boolean=True, name=f"pp_{label}")
    pn = cp.Variable(T, boolean=True, name=f"pn_{label}")
    qp = cp.Variable(T, boolean=True, name=f"qp_{label}")
    qn = cp.Variable(T, boolean=True, name=f"qn_{label}")
    z = cp.Variable(1, boolean=True, name=f"z_{label}")

    cons = []
    for k in range(T):
        px = x_var[0, k]
        py = x_var[1, k]

        cons += [
            px <= lx + M * (1 - pp[k]),
            px >= ux - M * (1 - pn[k]),
            py <= ly + M * (1 - qp[k]),
            py >= uy - M * (1 - qn[k]),
            z[0] >= 1 - pp[k] - pn[k] - qp[k] - qn[k],
        ]

    prob = z[0]  # 0 or 1
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


def solve_mpc_pareto(client, agents, cfg, emergency):

    # config
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
    a_comfort = float(cfg["stl"]["a_comfort"])
    beta_rate = float(cfg["stl"]["beta_rate_max"])

    crash_box = {
        "x_min": float(cfg["stl"]["crash_x_min"]),
        "x_max": float(cfg["stl"]["crash_x_max"]),
        "y_min": float(cfg["stl"]["crash_y_min"]),
        "y_max": float(cfg["stl"]["crash_y_max"]),
    }

    ego, leader, follower, left = agents[0], agents[1], agents[2], agents[3]

    z_ego = ego.get_transform().location.z
    z_ld = leader.get_transform().location.z
    z_f = follower.get_transform().location.z
    z_lt = left.get_transform().location.z

    # nominal trajectory
    x0 = ego_state(ego)
    X_nom, U_nom, A, B, c = build_nominal(ego, dt, N, x0)

    # sample trajectories
    leader_stl = leader.sample_trajectories(N, dt, S)
    follower_stl = follower.sample_trajectories(N, dt, S)
    left_stl = left.sample_trajectories(N, dt, S)

    leader_prob = leader.sample_trajectories(N, dt, P)
    follower_prob = follower.sample_trajectories(N, dt, P)
    left_prob = left.sample_trajectories(N, dt, P)

    draw_sample_traj(client.world, leader_stl, color=COLORS["green"], life_time=lt, z=z_ld)
    # draw_sample_traj(client.world, follower_stl, color=COLORS["red"], life_time=lt, z=z_f)
    # draw_sample_traj(client.world, left_stl, color=COLORS["yellow"], life_time=lt, z=z_lt)

    # epsilon grids
    # vehicles: uniform over (0, 1]
    eps_veh = np.linspace(0, 1, density + 1)[1:]
    # crash: only two values — 0.5 means "no collision", 1.0 means "allow collision"
    eps_crash = [0.5, 1.0]

    # four objectives, each mode minimises one and constrains the other three
    modes = ["leader", "follower", "left", "crash"]

    def solve_one(mode, eps_dict, warm=None):

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

        # STL constraints (always active)
        deltas = {}

        c_lane, d_lane = stay_in_lane(
            x, x_min=cfg["stl"]["x_min"], x_max=cfg["stl"]["x_max"],
            y_min=cfg["stl"]["y_min"], y_max=cfg["stl"]["y_max"], N=N,
        )
        cons += c_lane
        cons += [d_lane <= 3]
        deltas["lane"] = d_lane

        c_dec, d_dec = bounded_deceleration(u, a_comfort, N)
        cons += c_dec
        deltas["decel"] = d_dec

        c_sr, d_sr = bounded_steering_rate(u, beta_rate, N)
        cons += c_sr
        deltas["steer"] = d_sr

        # leader
        c_ld, dx_ld, dy_ld = safe_distance_vehicle(
            x, leader_stl, d_safe_x, d_safe_y, label="leader")
        cons += c_ld
        cons += [dx_ld <= d_safe_x]
        cons += [dy_ld <= d_safe_y]
        deltas["leader_x"] = dx_ld
        deltas["leader_y"] = dy_ld

        # follower
        c_f, dx_f, dy_f = safe_distance_vehicle(
            x, follower_stl, d_safe_x, d_safe_y, label="follower")
        cons += c_f
        cons += [dx_f <= d_safe_x]
        cons += [dy_f <= d_safe_y]
        deltas["follower_x"] = dx_f
        deltas["follower_y"] = dy_f

        if emergency:

            # left
            c_l, dx_l, dy_l = safe_distance_vehicle(
                x, left_stl, d_safe_x, d_safe_y, label="left")
            cons += c_l
            cons += [dx_l <= d_safe_x]
            cons += [dy_l <= d_safe_y]
            deltas["left_x"] = dx_l
            deltas["left_y"] = dy_l

            # crash scene box
            c_cr, d_cr = safe_distance_box(
                x, crash_box["x_min"], crash_box["x_max"],
                crash_box["y_min"], crash_box["y_max"], d_crash,
            )
            cons += c_cr
            cons += [d_cr <= d_crash]
            deltas["crash"] = d_cr

        # probability encoding — vehicles
        z_ld, p_leader, c_pld = encode_collision_vehicle(
            x, leader_prob, d_safe_x, d_safe_y, "p_leader")
        z_fo, p_follower, c_pfo = encode_collision_vehicle(
            x, follower_prob, d_safe_x, d_safe_y, "p_follower")
        z_le, p_left, c_ple = encode_collision_vehicle(
            x, left_prob, d_safe_x, d_safe_y, "p_left")
        cons += c_pld + c_pfo + c_ple

        # probability encoding — crash scene box
        z_cr, p_crash, c_pcr = encode_collision_box(
            x, crash_box["x_min"], crash_box["x_max"],
            crash_box["y_min"], crash_box["y_max"], d_crash, "p_crash",
        )
        cons += c_pcr

        probs = {
            "leader": p_leader,
            "follower": p_follower,
            "left": p_left,
            "crash": p_crash,
        }

        # epsilon constraints on non-minimised objectives
        for name, eps_val in eps_dict.items():
            cons.append(probs[name] <= float(eps_val))

        # objective
        W = 1e-4
        smoothness = (
            cp.norm(cp.diff(u[0, :]), 1)
            + cp.norm(cp.diff(u[1, :]), 1)
            + cp.norm(cp.diff(x[0, :]), 1)
            + cp.norm(cp.diff(x[1, :]), 1)
        )
        objective = cp.Minimize(probs[mode] + W * smoothness)

        prob = cp.Problem(objective, cons)

        n_cons = sum(ci.size for ci in cons)
        n_vars = sum(vi.size for vi in prob.variables())
        t_build = time.perf_counter() - t0

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
            "p_leader": float(p_leader.value),
            "p_follower": float(p_follower.value),
            "p_left": float(p_left.value),
            "p_crash": float(p_crash.value),
            "deltas": {k: float(v.value) for k, v in deltas.items() if v.value is not None},
            "t_build": t_build,
            "t_solve": t_solve,
            "num_constraints": n_cons,
            "num_variables": n_vars,
        }

    # epsilon-constraint sweep
    # for each mode, build the grid over the other three axes:
    #   vehicle axes -> eps_veh (uniform)
    #   crash axis   -> eps_crash ([0.5, 1.0])
    results = []
    for mode in modes:
        other = [m for m in modes if m != mode]
        grids = [eps_crash if ax == "crash" else eps_veh for ax in other]
        combos = list(itertools.product(*grids))

        print(f"\n  -- min p_{mode}  ({len(combos)} grid pts) --")
        warm = None
        for combo in combos:
            eps_dict = dict(zip(other, combo))
            tag = "  ".join(f"{k}={v:.2f}" for k, v in eps_dict.items())
            print(f"    {tag}")
            sol = solve_one(mode, eps_dict, warm=warm)
            if sol is None:
                print("    INFEASIBLE")
            else:
                warm = sol
                results.append(sol)
                print(f"    p=({sol['p_leader']:.3f}, {sol['p_follower']:.3f}, "
                      f"{sol['p_left']:.3f}, {sol['p_crash']:.3f})")

    # fallback
    if not results:
        print("  All infeasible -- emergency braking.")
        return {
            "status": False,
            "control": carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0),
            "deltas": None,
            "t_build": 0.0, "t_solve": 0.0,
            "num_constraints": None, "num_variables": None,
        }

    # pareto filter over four objectives
    pts = np.array([
        [r["p_leader"], r["p_follower"], r["p_left"], r["p_crash"]]
        for r in results
    ])
    mask = pareto_filter(pts)
    pareto = [results[i] for i in range(len(results)) if mask[i]]
    print(f"\n  Pareto: {len(pareto)} / {len(results)} retained")

    best = min(pareto, key=lambda r: np.linalg.norm(r["u_opt"][:, 0] - U_nom[0]))

    # draw and apply
    draw_sample_traj(client.world, best["x_opt"][:2, :].T, color=COLORS["blue"], life_time=lt, z=z_ego)

    a_opt, beta_opt = best["u_opt"][:, 0]
    control = bicycle_to_carla(
        [a_opt, beta_opt], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
    )

    print(f"  Best [{best['mode']}]: "
          f"p=({best['p_leader']:.3f}, {best['p_follower']:.3f}, "
          f"{best['p_left']:.3f}, {best['p_crash']:.3f})  "
          f"deltas={best['deltas']}")

    return {
        "status": True,
        "control": control,
        "deltas": best["deltas"],
        "t_build": best["t_build"],
        "t_solve": best["t_solve"],
        "num_constraints": best["num_constraints"],
        "num_variables": best["num_variables"],
    }