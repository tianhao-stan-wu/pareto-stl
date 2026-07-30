from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import cvxpy as cp


def safe_distance_vehicle(
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
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]
    margin_x = ego_w / 2.0 + veh_w / 2.0 + d_safe
    margin_y = ego_l / 2.0 + veh_l / 2.0 + d_safe

    delta_x = cp.Variable(nonneg=True, name=f"delta_x_{label}")
    delta_y = cp.Variable(nonneg=True, name=f"delta_y_{label}")
    constraints = []

    for k in range(N):
        ax = float(agent_traj[k, 0])
        ay = float(agent_traj[k, 1])

        z = cp.Variable(4, boolean=True, name=f"z_{label}_k{k}")
        constraints.append(cp.sum(z) >= 1)

        px = x_var[0, k]
        py = x_var[1, k]

        constraints.append((ax - px) <= -margin_x + delta_x + big_M * (1 - z[0]))
        constraints.append((ax - px) >=  margin_x - delta_x - big_M * (1 - z[1]))
        constraints.append((ay - py) <= -margin_y + delta_y + big_M * (1 - z[2]))
        constraints.append((ay - py) >=  margin_y - delta_y - big_M * (1 - z[3]))

    # option A: guaranteed feasible, allows full overlap
    # constraints.append(delta_x <= margin_x)
    # constraints.append(delta_y <= margin_y)

    # option B: tighter bound, may be infeasible (triggers fallback)
    constraints.append(delta_x <= d_safe)
    constraints.append(delta_y <= d_safe)

    return constraints, delta_x, delta_y


def safe_distance_walker(
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
    """
    agent_traj = np.asarray(agent_traj)
    if agent_traj.ndim == 3:
        agent_traj = agent_traj.mean(axis=0)

    N = agent_traj.shape[0]

    margin_x = ego_w / 2.0 + d_safe
    margin_y = ego_l / 2.0 + d_safe

    delta_x = cp.Variable(nonneg=True, name=f"delta_x_{label}")
    delta_y = cp.Variable(nonneg=True, name=f"delta_y_{label}")
    constraints = []

    for k in range(N):

        ax = float(agent_traj[k, 0])
        ay = float(agent_traj[k, 1])

        z = cp.Variable(4, boolean=True, name=f"z_{label}_k{k}")
        constraints.append(cp.sum(z) >= 1)

        px = x_var[0, k]
        py = x_var[1, k]

        constraints.append((ax - px) <= -margin_x + delta_x + big_M * (1 - z[0]))  # left
        constraints.append((ax - px) >=  margin_x - delta_x - big_M * (1 - z[1]))  # right
        constraints.append((ay - py) <= -margin_y + delta_y + big_M * (1 - z[2]))  # below
        constraints.append((ay - py) >=  margin_y - delta_y - big_M * (1 - z[3]))  # above

    # option A: guaranteed feasible, allows full overlap
    # constraints.append(delta_x <= margin_x)
    # constraints.append(delta_y <= margin_y)

    # option B: tighter bound, may be infeasible (triggers fallback)
    constraints.append(delta_x <= d_safe)
    constraints.append(delta_y <= d_safe)

    return constraints, delta_x, delta_y


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


