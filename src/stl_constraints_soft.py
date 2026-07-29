# stl_constraints.py
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def safe_distance_vehicle_soft(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    ego_w: float,
    ego_l: float,
    veh_w: float,
    veh_l: float,
    d_safe: float,
    big_M: float = 200.0,
    label: str = "vehicle"
):
    """
    Safe distance between ego and another vehicle.
    Lateral (x): margin = ego_w/2 + veh_w/2 + d_safe
    Longitudinal (y): margin = ego_l/2 + veh_l/2 + d_safe
    """
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]
    margin_x = ego_w / 2.0 + veh_w / 2.0 + d_safe
    margin_y = ego_l / 2.0 + veh_l / 2.0 + d_safe

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    constraints = []

    for k in range(N):
        ax = float(agent_traj[k, 0])
        ay = float(agent_traj[k, 1])

        b_left  = cp.Variable(boolean=True, name=f"b_{label}_left_k{k}")
        b_right = cp.Variable(boolean=True, name=f"b_{label}_right_k{k}")
        b_below = cp.Variable(boolean=True, name=f"b_{label}_below_k{k}")
        b_above = cp.Variable(boolean=True, name=f"b_{label}_above_k{k}")

        constraints.append(b_left + b_right + b_below + b_above >= 1)

        px = x_var[0, k]
        py = x_var[1, k]

        # lateral separation (x) uses width
        constraints.append((ax - px) <= -margin_x + delta + big_M * (1 - b_left))
        constraints.append((ax - px) >=  margin_x - delta - big_M * (1 - b_right))

        # longitudinal separation (y) uses length
        constraints.append((ay - py) <= -margin_y + delta + big_M * (1 - b_below))
        constraints.append((ay - py) >=  margin_y - delta - big_M * (1 - b_above))

    constraints.append(delta <= margin_y)

    return constraints, delta


def safe_distance_walker_soft(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    ego_w: float,
    ego_l: float,
    d_safe: float,
    big_M: float = 200,
    label: str = "walker"
):
    """
    Safe distance between ego and a pedestrian.
    Ego represented as rectangle (width x length), walker as a point.

    Parameters
    ----------
    x_var      : cp.Variable (4, N+1) — ego state trajectory
    agent_traj : ndarray (S, N+1, 2) or (N+1, 2) — walker [px, py]
    ego_w      : float — ego vehicle width (m)
    ego_l      : float — ego vehicle length (m)
    d_safe     : float — minimum safe distance (m)
    """
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]
    half_w = ego_w / 2.0
    half_l = ego_l / 2.0

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")
    constraints = []

    for k in range(N):
        ax = float(agent_traj[k, 0])
        ay = float(agent_traj[k, 1])

        b_left  = cp.Variable(boolean=True, name=f"b_{label}_left_k{k}")
        b_right = cp.Variable(boolean=True, name=f"b_{label}_right_k{k}")
        b_below = cp.Variable(boolean=True, name=f"b_{label}_below_k{k}")
        b_above = cp.Variable(boolean=True, name=f"b_{label}_above_k{k}")

        constraints.append(b_left + b_right + b_below + b_above >= 1)

        px = x_var[0, k]
        py = x_var[1, k]

        constraints.append((ax - px) <= -(half_w + d_safe) + delta + big_M * (1 - b_left))
        constraints.append((ax - px) >=  (half_w + d_safe) - delta - big_M * (1 - b_right))
        constraints.append((ay - py) <= -(half_l + d_safe) + delta + big_M * (1 - b_below))
        constraints.append((ay - py) >=  (half_l + d_safe) - delta - big_M * (1 - b_above))

    constraints.append(delta <= d_safe + half_l)

    return constraints, delta


def avoid_rectangle_soft(
    x_var: cp.Variable,
    rect: Sequence[float],   # [x_min, x_max, y_min, y_max]
    N: int,
    max_relax: int,
    name_prefix: str = "avoid_rect",
    M: float = 500.0,
    eps: float = 1e-3,
) -> Tuple[List[cp.Constraint], cp.Variable]:
    """
    Enforce that p_k = x_var[0:2, k] stays OUTSIDE an axis-aligned rectangle.

    Parameters
    ----------
    rect : [x_min, x_max, y_min, y_max]
    """

    if len(rect) != 4:
        raise ValueError("rect must be [x_min, x_max, y_min, y_max]")

    x_min, x_max, y_min, y_max = rect
    cons: List[cp.Constraint] = []

    delta = cp.Variable(nonneg=True, name=f"delta_{name_prefix}")

    for k in range(N):

        p_k = x_var[0:2, k]
        x, y = p_k[0], p_k[1]

       
        # Binary variables indicating which outside condition is active
        z = cp.Variable(4, boolean=True, name=f"{name_prefix}_{k}")

        # Must satisfy at least one outside condition
        cons.append(cp.sum(z) >= 1)

        # Left of rectangle
        cons.append(x <= x_min - eps + M * (1 - z[0]))

        # Right of rectangle
        cons.append(x >= x_max - delta + eps - M * (1 - z[1]))

        # Below rectangle
        cons.append(y <= y_min - eps + M * (1 - z[2]))

        # Above rectangle
        cons.append(y >= y_max + eps - M * (1 - z[3]))

    # cons.append(delta <= max_relax)

    return cons, delta


def yline_soft(x_var, y_line, N, max_relax, label="intersection"):
    """
    G[] ego_py <= y_line
    Hard version — no relaxation.
    """
    constraints = []

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")

    for k in range(N):
        py = x_var[1, k]
        constraints.append(py >= y_line - delta)

    # constraints.append(delta <= max_relax)

    return constraints, delta


def xline_soft(x_var, x_line, N, max_relax, label="lane"):
    """
    G[] ego_py <= y_line
    Hard version — no relaxation.
    """
    constraints = []

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")

    for k in range(N):
        px = x_var[0, k]
        constraints.append(px >= x_line - delta)

    # constraints.append(delta <= max_relax)

    return constraints, delta


def xline_hard_l(x_var, x_line, N, label="vehicle"):
    """
    G[] ego_py <= y_line
    Hard version — no relaxation.
    """
    constraints = []

    delta = cp.Variable(nonneg=True, name=f"delta_{label}")

    for k in range(N):
        px = x_var[0, k]
        constraints.append(px <= x_line)

    return constraints, delta


def avoid_rectangle_hard(
    x_var: cp.Variable,
    rect: Sequence[float],   # [x_min, x_max, y_min, y_max]
    N: int,
    label: str = "avoid_rect",
    M: float = 500.0,
    eps: float = 1e-3,
) -> Tuple[List[cp.Constraint], cp.Variable]:
    """
    Enforce that p_k = x_var[0:2, k] stays OUTSIDE an axis-aligned rectangle.

    Parameters
    ----------
    rect : [x_min, x_max, y_min, y_max]
    """

    if len(rect) != 4:
        raise ValueError("rect must be [x_min, x_max, y_min, y_max]")

    x_min, x_max, y_min, y_max = rect
    cons: List[cp.Constraint] = []

    for k in range(N):

        p_k = x_var[0:2, k]
        x, y = p_k[0], p_k[1]

       
        # Binary variables indicating which outside condition is active
        b_left  = cp.Variable(boolean=True, name=f"b_{label}_left_k{k}")
        b_right = cp.Variable(boolean=True, name=f"b_{label}_right_k{k}")
        b_below = cp.Variable(boolean=True, name=f"b_{label}_below_k{k}")
        b_above = cp.Variable(boolean=True, name=f"b_{label}_above_k{k}")

        # cons.append(b_left + b_right + b_below + b_above >= 1)
        cons.append(b_left + b_below + b_above >= 1)

        cons.append(x <= x_min - eps + M * (1 - b_left))
        # cons.append(x >= x_max + eps - M * (1 - b_right))
        cons.append(y <= y_min - eps + M * (1 - b_below))
        cons.append(y >= y_max + eps - M * (1 - b_above))

    return cons