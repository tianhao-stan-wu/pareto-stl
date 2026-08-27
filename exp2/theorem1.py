"""Empirical verification of Theorem 1 for Experiment 2."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import math
import numpy as np

ScenarioSampler = Callable[[], Any]
SatisfactionChecker = Callable[[np.ndarray, np.ndarray, Any], bool]


def make_exp2_joint_sampler(leader, follower, left, ped2, N: int, dt: float) -> ScenarioSampler:
    """Return a sampler for one joint Exp. 2 scenario."""
    def sampler():
        return {
            "leader": leader.sample_trajectories(N, dt, 1)[0],
            "follower": follower.sample_trajectories(N, dt, 1)[0],
            "left": left.sample_trajectories(N, dt, 1)[0],
            "ped2": ped2.sample_trajectories(N, dt, 1)[0],
        }

    return sampler


def make_exp2_satisfaction_checker(
    cfg_stl: Dict[str, Any],
    emergency: bool,
) -> SatisfactionChecker:
    """Return the exact Boolean STL checker corresponding to Exp. 2 gate constraints."""
    d_safe_x = float(cfg_stl["d_safe_x"])
    d_safe_y = float(cfg_stl["d_safe_y"])
    d_crash = float(cfg_stl["d_crash"])
    d_ped = float(cfg_stl["d_ped"])
    a_comfort = float(cfg_stl["a_comfort"])
    beta_rate_max = float(cfg_stl["beta_rate_max"])

    x_min = float(cfg_stl["x_min"])
    x_max = float(cfg_stl["x_max"])
    y_min = float(cfg_stl["y_min"])
    y_max = float(cfg_stl["y_max"])

    crash_x_min = float(cfg_stl["crash_x_min"])
    crash_x_max = float(cfg_stl["crash_x_max"])
    crash_y_min = float(cfg_stl["crash_y_min"])
    crash_y_max = float(cfg_stl["crash_y_max"])

    def safe_vehicle(ego_xy, other_xy) -> bool:
        T = min(len(ego_xy), len(other_xy))
        dx = np.abs(ego_xy[:T, 0] - other_xy[:T, 0])
        dy = np.abs(ego_xy[:T, 1] - other_xy[:T, 1])
        # Outside the keep-out rectangle iff at least one axis exceeds its margin.
        return bool(np.all(np.maximum(dx - d_safe_x, dy - d_safe_y) >= 0.0))

    def safe_walker(ego_xy, other_xy) -> bool:
        T = min(len(ego_xy), len(other_xy))
        dx = np.abs(ego_xy[:T, 0] - other_xy[:T, 0])
        dy = np.abs(ego_xy[:T, 1] - other_xy[:T, 1])
        return bool(np.all(np.maximum(dx, dy) - d_ped >= 0.0))

    def outside_crash_box(ego_xy) -> bool:
        lx = crash_x_min - d_crash
        ux = crash_x_max + d_crash
        ly = crash_y_min - d_crash
        uy = crash_y_max + d_crash
        px = ego_xy[:, 0]
        py = ego_xy[:, 1]
        inside_x = (px >= lx) & (px <= ux)
        inside_y = (py >= ly) & (py <= uy)
        return bool(np.all(~(inside_x & inside_y)))

    def satisfies(x_star: np.ndarray, u_star: np.ndarray, scenario: Dict[str, np.ndarray]) -> bool:
        ego_xy = np.asarray(x_star[:2, :].T, dtype=float)
        u = np.asarray(u_star, dtype=float)

        # Always-active static STLs.
        px = ego_xy[:, 0]
        py = ego_xy[:, 1]
        if not np.all((px >= x_min) & (px <= x_max) &
                      (py >= y_min) & (py <= y_max)):
            return False

        # G(a >= a_comfort).
        if not np.all(u[0, :] >= a_comfort):
            return False

        # G(|beta[k+1] - beta[k]| <= beta_rate_max).
        if u.shape[1] > 1 and not np.all(np.abs(np.diff(u[1, :])) <= beta_rate_max):
            return False

        # Jointly sampled leader and follower constraints are always active.
        if not safe_vehicle(ego_xy, scenario["leader"]):
            return False
        if not safe_vehicle(ego_xy, scenario["follower"]):
            return False

        if emergency:
            if not safe_vehicle(ego_xy, scenario["left"]):
                return False
            if not safe_walker(ego_xy, scenario["ped2"]):
                return False
            if not outside_crash_box(ego_xy):
                return False

        return True

    return satisfies


def validation_sample_size(eps: float, beta: float) -> int:
    """Smallest M satisfying (1 - eps)^M <= beta."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    return math.ceil(math.log(1.0 / beta) / (-math.log(1.0 - eps)))


def verify_theorem1(
    u_star: np.ndarray,
    x_star: np.ndarray,
    sampler: ScenarioSampler,
    satisfies: SatisfactionChecker,
    eps: float,
    beta: float,
    N_approx: int = 10000,
) -> Dict[str, Any]:
    """Validate one fixed feasible solution, then estimate its violation rate."""
    if N_approx <= 0:
        raise ValueError(f"N_approx must be positive, got {N_approx}")

    x_star = np.asarray(x_star, dtype=float)
    u_star = np.asarray(u_star, dtype=float)
    M = validation_sample_size(eps, beta)

    validation_failures = 0
    for _ in range(M):
        if not satisfies(x_star, u_star, sampler()):
            validation_failures += 1

    validated = validation_failures == 0

    result: Dict[str, Any] = {
        "eps": float(eps),
        "beta": float(beta),
        "M": int(M),
        "N_approx": int(N_approx),
        "validated": bool(validated),
        "certificate": bool(validated),
        "validation_failures": int(validation_failures),
        "approx_violations": None,
        "approx_samples": None,
        "p_viol_hat": None,
        "empirically_below_eps": None,
    }

    violations = sum(
        not satisfies(x_star, u_star, sampler()) for _ in range(N_approx)
    )
    result["approx_violations"] = int(violations)
    result["approx_samples"] = int(N_approx)
    result["p_viol_hat"] = float(violations / N_approx)
    result["empirically_below_eps"] = bool(result["p_viol_hat"] <= eps)

    return result


def summarise_theorem1(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-feasible-tick validation outcomes."""
    if not results:
        return {
            "num_feasible_ticks": 0,
            "num_validated": 0,
            "num_validation_failed": 0,
            "num_validated_below_eps": 0,
            "p_viol_hat": [],
            "eps": None,
            "beta": None,
            "M": None,
            "N_approx": None,
        }

    validated = [r for r in results if r["validated"]]
    p_hats = [r["p_viol_hat"] for r in validated if r["p_viol_hat"] is not None]
    eps = float(results[0]["eps"])
    below_eps = sum(p <= eps for p in p_hats)

    return {
        "num_feasible_ticks": int(len(results)),
        "num_validated": int(len(validated)),
        "num_validation_failed": int(len(results) - len(validated)),
        "num_validated_below_eps": int(below_eps),
        "p_viol_hat": [float(p) for p in p_hats],
        "eps": eps,
        "beta": float(results[0]["beta"]),
        "M": int(results[0]["M"]),
        "N_approx": int(results[0]["N_approx"]),
    }
