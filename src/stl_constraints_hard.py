# stl_constraints.py
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def safe_distance_vehicle_hard(
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
    Hard safe distance between ego and another vehicle. No relaxation.
    """
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]
    margin_x = ego_w / 2.0 + veh_w / 2.0 + d_safe
    margin_y = ego_l / 2.0 + veh_l / 2.0 + d_safe

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

        constraints.append((ax - px) <= -margin_x + big_M * (1 - b_left))
        constraints.append((ax - px) >=  margin_x - big_M * (1 - b_right))
        constraints.append((ay - py) <= -margin_y + big_M * (1 - b_below))
        constraints.append((ay - py) >=  margin_y - big_M * (1 - b_above))

    return constraints


def safe_distance_walker_hard(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    ego_w: float,
    ego_l: float,
    d_safe: float,
    big_M: float = 200.0,
    label: str = "walker"
):
    """
    Hard safe distance between ego and a pedestrian. No relaxation.
    """
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]
    half_w = ego_w / 2.0
    half_l = ego_l / 2.0

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

        constraints.append((ax - px) <= -(half_w + d_safe) + big_M * (1 - b_left))
        constraints.append((ax - px) >=  (half_w + d_safe) - big_M * (1 - b_right))
        constraints.append((ay - py) <= -(half_l + d_safe) + big_M * (1 - b_below))
        constraints.append((ay - py) >=  (half_l + d_safe) - big_M * (1 - b_above))

    return constraints


def avoid_rectangle_hard(
    x_var: cp.Variable,
    rect: Sequence[float],   # [x_min, x_max, y_min, y_max]
    N: int,
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
        cons.append(x >= x_max + eps - M * (1 - z[1]))

        # Below rectangle
        cons.append(y <= y_min - eps + M * (1 - z[2]))

        # Above rectangle
        cons.append(y >= y_max + eps - M * (1 - z[3]))

    return cons


def stay_below_line_hard(x_var, y_line, N, label="intersection"):
    """
    G[] ego_py <= y_line
    Hard version — no relaxation.
    """
    constraints = []

    for k in range(N):
        py = x_var[1, k]
        constraints.append(py >= y_line)

    return constraints