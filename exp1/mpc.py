import itertools
import carla
import numpy as np
import cvxpy as cp
import math
import time
import gurobipy as gp

from src.bicycle import KinematicBicycle
from src.utils import draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS
from exp1.stl import safe_distance_vehicle, safe_distance_walker, stay_in_lane, _promote

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


def encode_collision(x_var, trajs, d_safe, label, M=500):
    """
    P = (1/K) * sum(z_n).  z_n = 1 iff scenario n enters the box at any step.
    Four separation binaries per (n, k); no partition gap, no strict epsilon.
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
        DX >= d_safe - M * (1 - pp),
        -DX >= d_safe - M * (1 - pn),
        DY >= d_safe - M * (1 - qp),
        -DY >= d_safe - M * (1 - qn),
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


def solve_mpc_pareto(client, agents, cfg, emergency):

    # config
    T_sim = cfg["mpc"]["horizon"]
    S = cfg["mpc"]["num_samples"]
    P = cfg["mpc"]["num_prob"]
    dt = cfg["carla"]["dt"]
    N = int(round(T_sim / dt))
    density = cfg["mpc"]["density"]
    lt = dt * 1.5

    d_ped = float(cfg["stl"]["pedestrian"])
    d_amb = float(cfg["stl"]["ambulance"])

    ego, amb, ped = agents[0], agents[1], agents[2]

    # nominal trajectory and linearisation
    x0 = ego_state(ego)
    X_nom, U_nom, A, B, c = build_nominal(ego, dt, N, x0)

    # sample trajectories for STL and probability encoding
    ped_stl = ped.sample_trajectories(N, dt, S)
    amb_stl = amb.sample_trajectories(N, dt, S)
    ped_prob = ped.sample_trajectories(N, dt, P)
    amb_prob = amb.sample_trajectories(N, dt, P)

    # epsilon grid over [0, 1]
    eps_grid = np.linspace(0, 1, density + 1)[1:]

    free_axes = {
        "ped": ["amb"],
        "amb": ["ped"],
    }

    t2 = time.perf_counter()

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

        # STL constraints
        deltas = {}

        if emergency:
            c_ped, d_ped_stl = safe_distance_walker(x, ped_stl, d_ped, label="ped")
            cons += c_ped
            cons += [d_ped_stl <= d_ped]
            deltas["ped"] = d_ped_stl

        c_amb, d_amb_stl = safe_distance_vehicle(x, amb_stl, d_amb, label="amb")
        cons += c_amb
        cons += [d_amb_stl <= d_amb]
        deltas["amb"] = d_amb_stl

        c_lane, d_lane = stay_in_lane(x, x_min=-45.5, x_max=-44, y_min=0, y_max=100, N=N)
        cons += c_lane
        cons += [d_lane <= 3]
        deltas["lane"] = d_lane

        # probability encoding
        z_ped, p_ped, c_pe = encode_collision(x, ped_prob, d_ped, "ped")
        z_amb, p_amb, c_ae = encode_collision(x, amb_prob, d_amb, "amb")
        cons += c_pe + c_ae

        z_any = cp.Variable(P, boolean=True, name="z_any")
        cons += [z_any >= z_ped, z_any >= z_amb]
 
        probs = {"ped": p_ped, "amb": p_amb}

        # epsilon constraints on non-minimised objectives
        for name, eps_val in eps_dict.items():
            cons.append(probs[name] <= float(eps_val))

        W = 1e-4

        smoothness = (
            cp.norm(cp.diff(u[0, :]), 1)       # acceleration rate (jerk)
            + cp.norm(cp.diff(u[1, :]), 1)     # steering rate
            + cp.norm(cp.diff(x[0, :]), 1)     # position smoothness x
            + cp.norm(cp.diff(x[1, :]), 1)     # position smoothness y
        )

        objective = cp.Minimize(probs[mode] + W * smoothness)
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
            "p_ped": float(p_ped.value),
            "p_amb": float(p_amb.value),
            "deltas": {k: float(v.value) for k, v in deltas.items() if v.value is not None},
            "t_build": t_build,
            "t_solve": t_solve,
            "num_constraints": n_cons,
            "num_variables": n_vars,
        }

    # epsilon-constraint sweep
    results = []
    for mode, axes in free_axes.items():
        combos = list(itertools.product(eps_grid, repeat=len(axes)))
        print(f"\n  -- min p_{mode}  ({len(combos)} grid pts) --")
        warm = None
        for combo in combos:
            eps_dict = dict(zip(axes, combo))
            tag = "  ".join(f"e_{k}={v:.2f}" for k, v in eps_dict.items())
            print(f"    {tag}")
            sol = solve_one(mode, eps_dict, warm=warm)
            if sol is None:
                print("    INFEASIBLE")
            else:
                warm = sol
                results.append(sol)
                print(f"    p=({sol['p_ped']:.3f}, {sol['p_amb']:.3f})")

    t_solve_all = time.perf_counter() - t2

    # fallback
    if not results:
        print("  All infeasible -- emergency braking.")
        return {
            "status": False,
            "control": carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0),
        }

    # pareto filter
    pts = np.array([[r["p_ped"], r["p_amb"]] for r in results])
    mask = pareto_filter(pts)
    pareto = [results[i] for i in range(len(results)) if mask[i]]
    print(f"\n  Pareto: {len(pareto)} / {len(results)} retained")

    best = min(pareto, key=lambda r: np.linalg.norm(r["u_opt"][:, 0] - U_nom[0]))

    # draw and apply
    draw_sample_traj(client.world, best["x_opt"][:2, :].T, color=COLORS["blue"], life_time=lt)

    a_opt, beta_opt = best["u_opt"][:, 0]
    control = bicycle_to_carla(
        [a_opt, beta_opt], ego.acc_min, ego.acc_max, ego.beta_min, ego.beta_max
    )

    print(f"  Best [{best['mode']}]: "
          f"p=({best['p_ped']:.3f}, {best['p_amb']:.3f})  "
          f"deltas={best['deltas']}")

    # record all points for later plotting
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
    }
