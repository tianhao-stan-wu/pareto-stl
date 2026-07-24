# stl_constraints.py
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def safe_distance_vehicle_pareto(
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

    constraints.append(delta <= d_safe)

    return constraints, delta


def safe_distance_walker_pareto(
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

    constraints.append(delta <= d_safe)

    return constraints, delta


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
    # binaries = []

    for k in range(N):
        ax = float(agent_traj[k, 0])
        ay = float(agent_traj[k, 1])

        b_left  = cp.Variable(boolean=True, name=f"b_{label}_left_k{k}")
        b_right = cp.Variable(boolean=True, name=f"b_{label}_right_k{k}")
        b_below = cp.Variable(boolean=True, name=f"b_{label}_below_k{k}")
        b_above = cp.Variable(boolean=True, name=f"b_{label}_above_k{k}")

        # binaries.append({"left": b_left, "right": b_right, "below": b_below, "above": b_above})

        constraints.append(b_left + b_right + b_below + b_above >= 1)

        px = x_var[0, k]
        py = x_var[1, k]

        # lateral separation (x) uses width
        constraints.append((ax - px) <= -margin_x + delta + big_M * (1 - b_left))
        constraints.append((ax - px) >=  margin_x - delta - big_M * (1 - b_right))

        # longitudinal separation (y) uses length
        constraints.append((ay - py) <= -margin_y + delta + big_M * (1 - b_below))
        constraints.append((ay - py) >=  margin_y - delta - big_M * (1 - b_above))

    # if d_safe <= delta <= margin_y, treat as d_safe = delta for post processing (physically impossible)
    constraints.append(delta <= margin_y)

    return constraints, delta
    # return constraints, delta, binaries


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

    constraints.append(delta <= d_safe)

    return constraints, delta


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