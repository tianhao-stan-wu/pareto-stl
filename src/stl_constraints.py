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
    delta_max: float = 0.0,
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

    constraints.append(delta <= delta_max)

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