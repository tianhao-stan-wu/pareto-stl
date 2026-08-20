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
    d_safe: float,
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

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    z0 = cp.Variable((S, T), boolean=True, name=f"z0_{label}")
    z1 = cp.Variable((S, T), boolean=True, name=f"z1_{label}")
    z2 = cp.Variable((S, T), boolean=True, name=f"z2_{label}")
    z3 = cp.Variable((S, T), boolean=True, name=f"z3_{label}")

    constraints = [
        z0 + z1 + z2 + z3 >= 1,
        AX - PX <= -d_safe + delta + big_M * (1 - z0),
        AX - PX >=  d_safe - delta - big_M * (1 - z1),
        AY - PY <= -d_safe + delta + big_M * (1 - z2),
        AY - PY >=  d_safe - delta - big_M * (1 - z3),
    ]
    return constraints, delta


def safe_distance_walker(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    d_safe: float,
    big_M: float = 200.0,
    label: str = "walker",
):
    """
    Safe distance from a pedestrian, enforced over EVERY sampled trajectory.
    Identical structure to safe_distance_vehicle; kept separate so the two
    agent classes can carry independent slack variables and margins.
    """
    traj = _as_3d(agent_traj)
    S    = traj.shape[0]
    T    = min(traj.shape[1], x_var.shape[1])
    traj = traj[:, :T, :]

    AX = traj[:, :, 0]
    AY = traj[:, :, 1]
    PX = _promote(x_var[0, :T], S)
    PY = _promote(x_var[1, :T], S)

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    z0 = cp.Variable((S, T), boolean=True, name=f"z0_{label}")
    z1 = cp.Variable((S, T), boolean=True, name=f"z1_{label}")
    z2 = cp.Variable((S, T), boolean=True, name=f"z2_{label}")
    z3 = cp.Variable((S, T), boolean=True, name=f"z3_{label}")

    constraints = [
        z0 + z1 + z2 + z3 >= 1,
        AX - PX <= -d_safe + delta + big_M * (1 - z0),
        AX - PX >=  d_safe - delta - big_M * (1 - z1),
        AY - PY <= -d_safe + delta + big_M * (1 - z2),
        AY - PY >=  d_safe - delta - big_M * (1 - z3),
    ]
    return constraints, delta


def stay_in_lane(
    x_var: cp.Variable,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    N: int,
    big_M: float = 200.0,
    label: str = "lane"
):
    """
    G[] (y_min <= ego_py <= y_max) → (x_min <= ego_px <= x_max)
    """
    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    constraints = []

    for k in range(N + 1):
        px = x_var[0, k]
        py = x_var[1, k]

        z = cp.Variable(3, boolean=True, name=f"z_{label}_k{k}")
        constraints.append(cp.sum(z) >= 1)

        constraints.append(py <= y_min + big_M * (1 - z[0]))
        constraints.append(py >= y_max - big_M * (1 - z[1]))

        constraints.append(px >= x_min - delta - big_M * (1 - z[2]))
        constraints.append(px <= x_max + big_M * (1 - z[2]))

    return constraints, delta


def clear_intersection(
    x_var: cp.Variable,
    y_exit: float,
    N: int,
    big_M: float = 200.0,
    label: str = "clear"
):
    """
    STL: F[0,T](py <= y_exit)
    """
    delta_cross = cp.Variable(nonneg=True, name=f"delta_cross_{label}")
    constraints = []

    z_exit = cp.Variable(N + 1, boolean=True, name=f"z_exit_{label}")
    constraints.append(cp.sum(z_exit) >= 1)

    for k in range(N + 1):
        py = x_var[1, k]
        constraints.append(py <= y_exit + delta_cross + big_M * (1 - z_exit[k]))

    return constraints, delta_cross


