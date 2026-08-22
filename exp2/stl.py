from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def _promote(row, S):
    """Broadcast a length-T cvxpy row expression to shape (S, T)."""
    return cp.vstack([row for _ in range(S)])


def _as_3d(agent_traj):
    """Accept (T,2) or (S,T,2); always return (S,T,2)."""
    a = np.asarray(agent_traj, dtype=float)
    return a[np.newaxis] if a.ndim == 2 else a


def safe_distance_vehicle(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    d_safe_x: float,
    d_safe_y: float,
    big_M: float = 200.0,
    label: str = "vehicle",
):
    """
    Safe distance from a vehicle, enforced over EVERY sampled trajectory.

    Uses a square keep-out box of half-side d_safe in both axes; vehicle
    extents are folded into d_safe by the caller.

    agent_traj : (S, T, 2) sampled trajectories, or (T, 2) for a single one.

    Per scenario s and step k, ego must be clear on at least one axis:
        z0: ax - px <= -d_safe + delta      (ego to the +x side)
        z1: ax - px >=  d_safe - delta      (ego to the -x side)
        z2: ay - py <= -d_safe + delta      (ego to the +y side)
        z3: ay - py >=  d_safe - delta      (ego to the -y side)
        z0 + z1 + z2 + z3 >= 1

    One delta is shared across all scenarios and steps, so it measures the
    worst-case violation over the sample set rather than an average.
    """
    traj = _as_3d(agent_traj)
    S    = traj.shape[0]
    T    = min(traj.shape[1], x_var.shape[1])
    traj = traj[:, :T, :]

    AX = traj[:, :, 0]                    # (S, T)
    AY = traj[:, :, 1]                    # (S, T)
    PX = _promote(x_var[0, :T], S)        # (S, T)
    PY = _promote(x_var[1, :T], S)        # (S, T)

    delta_x = cp.Variable(nonneg=True, name=f"delta_x_{label}")
    delta_y = cp.Variable(nonneg=True, name=f"delta_y_{label}")
    z0 = cp.Variable((S, T), boolean=True, name=f"z0_{label}")
    z1 = cp.Variable((S, T), boolean=True, name=f"z1_{label}")
    z2 = cp.Variable((S, T), boolean=True, name=f"z2_{label}")
    z3 = cp.Variable((S, T), boolean=True, name=f"z3_{label}")

    constraints = [
        z0 + z1 + z2 + z3 >= 1,
        AX - PX <= -d_safe_x + delta_x + big_M * (1 - z0),
        AX - PX >=  d_safe_x - delta_x - big_M * (1 - z1),
        AY - PY <= -d_safe_y + delta_y + big_M * (1 - z2),
        AY - PY >=  d_safe_y - delta_y - big_M * (1 - z3),
    ]
    return constraints, delta_x, delta_y


def stay_in_lane(
    x_var: cp.Variable,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    N: int,
    label: str = "lane",
):
    """
    G[0,T]( x_min <= px <= x_max  AND  y_min - delta <= py <= y_max )
    Only y_min is relaxed; the other three bounds are hard.
    """
    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    cons = []

    for k in range(N + 1):
        cons += [
            x_var[0, k] >= x_min,
            x_var[0, k] <= x_max,
            x_var[1, k] >= y_min - delta,
            x_var[1, k] <= y_max,
        ]

    return cons, delta


def bounded_deceleration(
    u_var: cp.Variable,
    a_comfort: float,
    N: int,
    label: str = "decel",
):
    """
    G[0,T]( a_ego >= a_comfort - delta )
    """
    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    cons = [u_var[0, k] >= a_comfort - delta for k in range(N)]
    return cons, delta


def bounded_steering_rate(
    u_var: cp.Variable,
    beta_rate_max: float,
    N: int,
    label: str = "steer_rate",
):
    """
    G[0,T-1]( |beta_{k+1} - beta_k| <= beta_rate_max + delta )
    """
    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    dbeta = cp.diff(u_var[1, :])
    cons = [
        dbeta <= beta_rate_max + delta,
        -dbeta <= beta_rate_max + delta,
    ]
    return cons, delta


def safe_distance_box(
    x_var: cp.Variable,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    d_safe: float,
    big_M: float = 200.0,
    label: str = "crash",
):
    """
    G[0,T]( ego outside the box [x_min, x_max] x [y_min, y_max] inflated by d_safe )
    """
    lx = x_min - d_safe
    ux = x_max + d_safe
    ly = y_min - d_safe
    uy = y_max + d_safe
    T = x_var.shape[1]

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    cons = []

    for k in range(T):
        px = x_var[0, k]
        py = x_var[1, k]

        z = cp.Variable(4, boolean=True, name=f"z_{label}_k{k}")
        cons.append(cp.sum(z) >= 1)

        cons.append(px <= lx + delta + big_M * (1 - z[0]))   # ego left of box
        cons.append(px >= ux - delta - big_M * (1 - z[1]))   # ego right of box
        cons.append(py <= ly + delta + big_M * (1 - z[2]))   # ego below box
        cons.append(py >= uy - delta - big_M * (1 - z[3]))   # ego above box

    return cons, delta