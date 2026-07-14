# stl_constraints.py
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def safe_distance(
    x_var: cp.Variable,
    agent_traj: np.ndarray,
    ego_w: float,
    ego_l: float,
    d_safe: float,
    big_M: float = 1e3,
    label: str = "agent"
):
    """
    Encode minimum safe distance between ego and an agent as MILP constraints.

    Parameters
    ----------
    x_var    : cp.Variable (4, N+1) — ego state trajectory
    agent_traj : ndarray (N+1, 2)  — agent [px, py] over horizon
    ego_w    : float — ego vehicle width (m)
    ego_l    : float — ego vehicle length (m)
    d_safe   : float — minimum safe distance (m)
    big_M    : float — big-M constant for disjunction
    label    : str   — name prefix for binary variables

    Returns
    -------
    constraints : list of cp.Constraint
    delta       : cp.Variable (nonneg slack)
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
        constraints.append((ax - px) >= (half_w + d_safe) - delta - big_M * (1 - b_right))
        constraints.append((ay - py) <= -(half_l + d_safe) + delta + big_M * (1 - b_below))
        constraints.append((ay - py) >= (half_l + d_safe) - delta - big_M * (1 - b_above))

    return constraints, delta