import itertools
import carla
import numpy as np
import cvxpy as cp
import math
import time

from src.bicycle import KinematicBicycle
from src.stl import safe_distance_vehicle, safe_distance_walker, clear_intersection, stay_in_lane
from src.utils import draw_sample_traj, bicycle_to_carla, carla_to_bicycle, COLORS

import gurobipy as gp

# module-level, created once
_GRB_ENV = gp.Env(params={"OutputFlag": 0})

_GRB_PARAMS = dict(
    # termination
    TimeLimit    = 10.0,   
    MIPGap       = 0.0,    

    # big-M numerics
    IntFeasTol   = 1e-7,   
    NumericFocus = 2,      
    Presolve     = 2,      

    # search strategy
    MIPFocus     = 1,      
    Heuristics   = 0.1,    
    Cuts         = 2,      

    # determinism for reproducibility
    Threads      = 4,      
    Seed         = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
_M_EGO, _M_PED, _M_AMB = 1850.0,  70.0, 4500.0
_V_PED, _V_AMB, _V_EGO = 1.0,   0.1,    0.2
_S_PED, _S_AMB         = 1000, 6000

_BIG_M, _EPS_STRICT     =  500.0,  1e-3      # Big-M and strict-inequality gap
_K_TAIL                 =    20              # CVaR tail size (worst 5 of 100, alpha=0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _ego_state(ego) -> np.ndarray:
    """Return ego state [px, py, theta_rad, speed_mps]."""
    tf  = ego.get_transform()
    vel = ego.get_velocity()
    return np.array([tf.location.x, tf.location.y,
                     math.radians(tf.rotation.yaw),
                     math.sqrt(vel.x**2 + vel.y**2)])


def _build_nominal(ego, model, dt, N, ego_init):
    """
    Derive U_nom from the autopilot waypoint plan, roll out X_nom,
    and linearise the bicycle dynamics at each step.
    Returns X_nom (N+1,4), U_nom (N,2), A_seq, B_seq, c_seq.
    """
    ctrl         = ego.agent.run_step()
    a_nom, b_nom = carla_to_bicycle(ctrl, ego.acc_min, ego.acc_max,
                                    ego.beta_min, ego.beta_max)
    plan = list(ego.agent.get_local_planner().get_plan())

    U_nom = np.zeros((N, 2))
    for k in range(N):
        if k < len(plan) - 1:
            wp_c  = plan[k][0].transform
            wp_n  = plan[k+1][0].transform
            yaw_n = math.atan2(wp_n.location.y - wp_c.location.y,
                               wp_n.location.x - wp_c.location.x)
            if k == 0:
                dyaw = yaw_n - ego_init[2]
            else:
                wp_p  = plan[k-1][0].transform
                yaw_p = math.atan2(wp_c.location.y - wp_p.location.y,
                                   wp_c.location.x - wp_p.location.x)
                dyaw  = yaw_n - yaw_p
            v_est    = max(ego_init[3] + a_nom * k * dt, 0.5)
            beta_k   = np.clip(dyaw * ego.lr / (v_est * dt), ego.beta_min, ego.beta_max)
            U_nom[k] = [a_nom, beta_k]
        else:
            U_nom[k] = [a_nom, b_nom]

    X_nom = np.zeros((N+1, 4));  X_nom[0] = ego_init.copy()
    A_seq, B_seq, c_seq = [], [], []
    for k in range(N):
        A_k, B_k    = model.linearize(X_nom[k], U_nom[k])
        X_nom[k+1]  = model.step(X_nom[k], U_nom[k])
        c_k         = X_nom[k+1] - A_k @ X_nom[k] - B_k @ U_nom[k]
        A_seq.append(A_k);  B_seq.append(B_k);  c_seq.append(c_k)

    return X_nom, U_nom, A_seq, B_seq, c_seq


def _select_worst(ego_pos_nom, trajs, k) -> np.ndarray:
    """
    CVaR worst-k pre-selection.
    Rank all S scenarios by their minimum L-inf distance to the nominal ego
    position across the horizon; return the k closest (most dangerous) indices.

    ego_pos_nom : (N+1, 2)
    trajs       : (S, N+1, 2)
    """
    l_inf    = np.abs(trajs - ego_pos_nom[np.newaxis]).max(axis=2)  # (S, N+1)
    min_dist = l_inf.min(axis=1)                                     # (S,)
    return np.argsort(min_dist)[:k]


def _risk_weights(ego_pos_nom, ego_vel_nom, trajs,
                  d_coll, mu, V_agent, V_ego, sev_scale, dt, S):
    """
    Compute fixed per-scenario risk weights from the *nominal* ego trajectory.
    Severity = mu * ||v_ego - v_agent|| / sev_scale.
    Divided by K so that sum(r_agent) approximates E[risk] over the tail.

    Returns r_agent (K,), r_ego (K,).
    """
    K = trajs.shape[0]
    r_agent, r_ego = np.zeros(K), np.zeros(K)
    for n in range(K):
        pos  = trajs[n, :, :2]
        hits = np.where(np.linalg.norm(ego_pos_nom - pos, axis=1) <= d_coll)[0]
        if hits.size == 0:
            continue
        t       = hits[0]
        v_agent = (pos[t] - pos[t-1]) / dt if t > 0 else np.zeros(2)
        sev     = mu * np.linalg.norm(ego_vel_nom[t] - v_agent) / sev_scale
        r_agent[n] = sev * V_agent / S
        r_ego[n]   = sev * V_ego   / S
    return r_agent, r_ego


def _encode_collision(x_var, trajs, r_per_z, margin_x, margin_y, label):
    """
    MILP encoding of empirical collision risk for one agent type.

    Per (n, k):
      x-axis partition (exclusive, sum = 1):
        s_x=1  iff  |dx| <= margin_x          (ego inside danger band)
        p_pos=1 iff  dx  >= margin_x + eps    (ego safely right of agent)
        p_neg=1 iff -dx  >= margin_x + eps    (ego safely left  of agent)
      y-axis: symmetric with margin_y
      z_nk[n,k] = s_x[n,k] AND s_y[n,k]      (collision at step k)

    Per n:
      z_n[n] = OR_k z_nk[n,k]                 (any collision in scenario n)

    Risk:
      r_approx = sum_n  r_per_z[n] * z_n[n]  (scalar cp.Expression)

    Returns z_n (cp.Variable K), r_approx (cp.Expression), constraints (list).
    """
    K, T, _ = trajs.shape
    M, e    = _BIG_M, _EPS_STRICT

    # declare all binary variables upfront (avoids repeated cp.Variable calls)
    s_x   = cp.Variable((K, T), boolean=True, name=f"sx_{label}")
    p_pos = cp.Variable((K, T), boolean=True, name=f"pp_{label}")
    p_neg = cp.Variable((K, T), boolean=True, name=f"pn_{label}")
    s_y   = cp.Variable((K, T), boolean=True, name=f"sy_{label}")
    q_pos = cp.Variable((K, T), boolean=True, name=f"qp_{label}")
    q_neg = cp.Variable((K, T), boolean=True, name=f"qn_{label}")
    z_nk  = cp.Variable((K, T), boolean=True, name=f"znk_{label}")
    z_n   = cp.Variable(K,      boolean=True, name=f"zn_{label}")

    cons = []
    for n in range(K):
        px = trajs[n, :, 0].astype(float)   # (T,) precomputed agent positions
        py = trajs[n, :, 1].astype(float)

        for k in range(T):
            dx = x_var[0, k] - px[k]
            dy = x_var[1, k] - py[k]

            # x-axis exclusive partition + big-M region constraints
            cons += [s_x[n,k] + p_pos[n,k] + p_neg[n,k] == 1,
                      dx <=  margin_x     + M*(1 - s_x[n,k]),   # inside: right bound
                     -dx <=  margin_x     + M*(1 - s_x[n,k]),   # inside: left bound
                      dx >=  margin_x + e - M*(1 - p_pos[n,k]), # outside right
                     -dx >=  margin_x + e - M*(1 - p_neg[n,k])] # outside left

            # y-axis exclusive partition
            cons += [s_y[n,k] + q_pos[n,k] + q_neg[n,k] == 1,
                      dy <=  margin_y     + M*(1 - s_y[n,k]),
                     -dy <=  margin_y     + M*(1 - s_y[n,k]),
                      dy >=  margin_y + e - M*(1 - q_pos[n,k]),
                     -dy >=  margin_y + e - M*(1 - q_neg[n,k])]

            # AND gate: z_nk[n,k] = s_x[n,k] AND s_y[n,k]
            cons += [z_nk[n,k] <= s_x[n,k],
                     z_nk[n,k] <= s_y[n,k],
                     z_nk[n,k] >= s_x[n,k] + s_y[n,k] - 1]

        # OR gate over time: z_n[n] = any z_nk[n,k] over k=0..T-1
        for k in range(T):
            cons.append(z_nk[n, k] <= z_n[n])       # any hit forces z_n on
        cons.append(cp.sum(z_nk[n, :]) >= z_n[n])   # z_n on only if some k hit

    r_approx = cp.sum(cp.multiply(r_per_z, z_n))
    return z_n, r_approx, cons


def _pareto_filter(pts) -> np.ndarray:
    """
    Return boolean mask of non-dominated rows in pts (M, 3).
    Row i is dominated iff some row j has all values <= and at least one <.
    """
    n, mask = len(pts), np.ones(len(pts), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                mask[i] = False
                break
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def solve_mpc_pareto(client, agents, cfg):
    """
    Multi-objective MPC via epsilon-constraint method with MILP risk encoding.

    Objectives  : r_ped, r_ego, r_amb
    STL cons    : safe_distance (ped, amb) + clear_intersection + lane_keeping
    Epsilon grid: density evenly-spaced values in (0,1] per free axis
                  -> density^2 grid points per mode, 3*density^2 total solves
    Selection   : Pareto-filter then pick control closest to nominal first step.

    agents : [ego (Vehicle), amb (Vehicle), ped (Walker)]
    """

    # ── 0. Config ─────────────────────────────────────────────────────────────
    T_sim   = cfg["mpc"]["horizon"]
    S       = cfg["mpc"]["num_samples"]     # 100
    dt      = cfg["carla"]["dt"]
    N       = int(round(T_sim / dt))
    density = cfg["mpc"]["density"]
    lt      = dt * 1.5

    d_ped   = float(cfg["stl"]["pedestrian"])
    d_amb   = float(cfg["stl"]["ambulance"])

    ego, amb, ped = agents[0], agents[1], agents[2]

    mu_ped = (_M_EGO * _M_PED) / (_M_EGO + _M_PED)
    mu_amb = (_M_EGO * _M_AMB) / (_M_EGO + _M_AMB)

    # ── 1. Ego state + nominal trajectory ─────────────────────────────────────
    model    = KinematicBicycle(lr=ego.lr, dt=dt)
    ego_init = _ego_state(ego)
    X_nom, U_nom, A_seq, B_seq, c_seq = _build_nominal(ego, model, dt, N, ego_init)

    ego_pos_nom = X_nom[:, :2]                                       
    ego_vel_nom = np.stack([X_nom[:,3]*np.cos(X_nom[:,2]),           
                            X_nom[:,3]*np.sin(X_nom[:,2])], axis=1)

    # ── 2. Sample 100 trajectories, select worst 5 per agent (CVaR alpha=0.95) ──
    ped_trajs = ped.sample_trajectories(N, dt, S)                            
    amb_trajs = amb.sample_trajectories(N, dt, S)                            
    ped_trajs = ped_trajs[_select_worst(ego_pos_nom, ped_trajs, _K_TAIL)]    
    amb_trajs = amb_trajs[_select_worst(ego_pos_nom, amb_trajs, _K_TAIL)]    

    # ── 3. Per-scenario risk weights (fixed scalars, computed on nominal traj) ──
    r_ped, r_ego_p = _risk_weights(ego_pos_nom, ego_vel_nom, ped_trajs,
                                   d_ped, mu_ped, _V_PED, _V_EGO, _S_PED., dt, _K_TAIL)
    r_amb, r_ego_a = _risk_weights(ego_pos_nom, ego_vel_nom, amb_trajs,
                                   d_amb, mu_amb, _V_AMB, _V_EGO, _S_AMB., dt, _K_TAIL)
    r_ego = r_ego_p + r_ego_a   # (5,) combined ego risk per scenario index

    print(f"r_ped: {r_ped} \nr_amb: {r_amb} \nr_ego:{r_ego}")

    # ── 4. Solver ─────────────────────────────────────────────────────────────
    solver = next((s for s in [cp.GUROBI, cp.CPLEX, cp.SCIP, cp.CBC]
                   if s in cp.installed_solvers()), None)
    if solver is None:
        raise RuntimeError(f"No MIP solver found. Installed: {cp.installed_solvers()}")
    else:
        print(f"Installed solver: {cp.installed_solvers()}")
        print(f"Selected solver: {solver}")
        
    # ── 5. Epsilon grid: density points in (0, 1] per axis ───────────────────
    eps_grid = np.linspace(0, 1, density + 1)[1:]   # e.g. [0.5, 1.0] for density=2

    # each mode frees its own epsilon; the other two axes form the grid
    _free_axes = {"ped": ["ego", "amb"],
                  "ego": ["ped", "amb"],
                  "amb": ["ped", "ego"]}

    # ── 6. Single MPC solve ───────────────────────────────────────────────────
    def _solve_one(mode, eps_dict, warm=None):
        """
        Build and solve one epsilon-constraint MILP.

        mode     : "ped" | "ego" | "amb"   objective to minimise
        eps_dict : {name: value} upper bounds on the other two risk objectives
        Returns result dict, or None if infeasible.
        """
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

        # MILP risk encoding
        z_ped_var, r_ped_expr, c_ped = _encode_collision(
            x_var, ped_trajs, r_ped, d_ped, d_ped, "ped")
        z_amb_var, r_amb_expr, c_amb = _encode_collision(
            x_var, amb_trajs, r_amb, d_amb, d_amb, "amb")
        cons += c_ped + c_amb

        # z_any[n] = z_ped[n] OR z_amb[n]  -> ego risk indicator
        z_any = cp.Variable(_K_TAIL, boolean=True, name="z_any")
        for n in range(_K_TAIL):
            cons += [z_any[n] >= z_ped_var[n],
                     z_any[n] >= z_amb_var[n],
                     z_any[n] <= z_ped_var[n] + z_amb_var[n]]
        r_ego_expr = cp.sum(cp.multiply(r_ego, z_any))

        # epsilon constraints on non-minimised objectives
        risk_exprs = {"ped": r_ped_expr, "ego": r_ego_expr, "amb": r_amb_expr}
        for name, eps_val in eps_dict.items():
            cons.append(risk_exprs[name] <= float(eps_val))

        W_BETA      = 5e-2    # steering magnitude
        W_DBETA     = 5e-2    # steering rate  (10x magnitude — this is the key term)
        W_STL       = 1e-3 

        # L1 version — keeps the problem a pure MILP
        s_beta = cp.Variable(N - 1, nonneg=True)
        cons += [s_beta >=  cp.diff(u_var[1, :]),
                 s_beta >= -cp.diff(u_var[1, :])]
        smoothness = W_DBETA * cp.sum(s_beta) + W_BETA * cp.norm(u_var[1, :], 1)

        objective = cp.Minimize(
            risk_exprs[mode]
            + W_STL * sum(deltas.values())
            + smoothness
        )

        # minimise chosen risk; small delta penalty keeps STL slack tight
        # objective = cp.Minimize(risk_exprs[mode] + 1e-3 * sum(deltas.values()))
        # objective = cp.Minimize(sum(deltas.values()))
        prob = cp.Problem(objective, cons)

        n_cons  = sum(c.size for c in cons)
        n_vars  = sum(v.size for v in prob.variables())
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()

        if warm is not None:
            x_var.value = warm["x_opt"]        # cvxpy passes these as MIP start
            u_var.value = warm["u_opt"]
            prob.solve(solver=cp.GUROBI, env=_GRB_ENV, warm_start=True, **_GRB_PARAMS)
        else:
            prob.solve(solver=cp.GUROBI, env=_GRB_ENV, **_GRB_PARAMS)

        t_solve = time.perf_counter() - t1
        print(f"t_solve: {t_solve:.3f}, n_cons: {n_cons}, n_vars: {n_vars}")

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"[{prob.status}]", end=" ")
            return None

        return {
            "mode":            mode,
            "x_opt":           x_var.value,
            "u_opt":           u_var.value,
            "r_ped":           float(r_ped_expr.value),
            "r_ego":           float(r_ego_expr.value),
            "r_amb":           float(r_amb_expr.value),
            "deltas":          {k: float(v.value) for k, v in deltas.items()
                                if v.value is not None},
            "t_build":         t_build,
            "t_solve":         t_solve,
            "num_constraints": n_cons,
            "num_variables":   n_vars,
        }

    # ── 7. Epsilon-constraint sweep ───────────────────────────────────────────
    results = []
    for mode, axes in _free_axes.items():
        combos = list(itertools.product(eps_grid, repeat=len(axes)))   # density^2
        print(f"\n  -- min r_{mode}  ({len(combos)} grid pts) --")
        warm = None
        for combo in combos:
            eps_dict = dict(zip(axes, combo))
            tag = "  ".join(f"e_{k}={v:.2f}" for k, v in eps_dict.items())
            print(f"    {tag}", end=" ... \n", flush=True)
            sol = _solve_one(mode, eps_dict, warm=warm)
            if sol is None:
                print("INFEASIBLE")
            else:
                print(f"OK  ({sol['r_ped']:.3f}, {sol['r_ego']:.3f}, {sol['r_amb']:.3f})"
                      f"  {sol['t_solve']:.2f}s")
                warm = sol 
                results.append(sol)

    # ── 8. Fallback if all infeasible ─────────────────────────────────────────
    if not results:
        print("  All solves infeasible -- emergency braking.")
        fb = carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0,
                                  manual_gear_shift=False)
        return {"status": False, "control": fb, "deltas": None,
                "t_build": 0., "t_solve": 0.,
                "num_constraints": None, "num_variables": None}

    # ── 9. Pareto filter ──────────────────────────────────────────────────────
    pts    = np.array([[r["r_ped"], r["r_ego"], r["r_amb"]] for r in results])
    mask   = _pareto_filter(pts)
    pareto = [results[i] for i in range(len(results)) if mask[i]]
    print(f"\n  Pareto: {len(pareto)} / {len(results)} solutions retained")

    # pick solution whose first control step deviates least from nominal
    best = min(pareto, key=lambda r: np.linalg.norm(r["u_opt"][:, 0] - U_nom[0]))

    # ── 10. Debug draw ────────────────────────────────────────────────────────
    draw_sample_traj(client.world, best["x_opt"][:2, :].T,
                     color=COLORS["blue"],  life_time=lt)

    # ── 11. Convert first control step to CARLA VehicleControl ───────────────
    a, beta = best["u_opt"][:, 0]
    control = bicycle_to_carla([a, beta],
                               ego.acc_min, ego.acc_max,
                               ego.beta_min, ego.beta_max)

    print(f"  Best [{best['mode']}]: "
          f"r=({best['r_ped']:.4f}, {best['r_ego']:.4f}, {best['r_amb']:.4f})  "
          f"deltas={best['deltas']}")
    print("max beta diff:", np.abs(np.diff(best["u_opt"][1, :])).max())
    print("\n")

    return {
        "status":          True,
        "control":         control,
        "deltas":          best["deltas"],
        "t_build":         best["t_build"],
        "t_solve":         best["t_solve"],
        "num_constraints": best["num_constraints"],
        "num_variables":   best["num_variables"],
    }


