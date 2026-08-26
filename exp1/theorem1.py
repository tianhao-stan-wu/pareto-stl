"""Generic empirical verification of Theorem 1."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np

ScenarioSampler = Callable[[], Any]
SatisfactionChecker = Callable[[np.ndarray, Any], bool]


def make_exp1_joint_sampler(ped, amb, N: int, dt: float) -> ScenarioSampler:
    """Return a sampler for one joint (pedestrian, ambulance) scenario."""
    def sampler():
        # Each call creates one sample for each agent; the two are packaged as
        # one joint scenario. This is the distribution used by the verifier.
        return {
            "pedestrian": ped.sample_trajectories(N, dt, 1)[0],
            "ambulance": amb.sample_trajectories(N, dt, 1)[0],
        }

    return sampler


def make_exp1_satisfaction_checker(
    d_ped: float,
    d_amb: float,
    lane: Dict[str, float],
    emergency: bool,
) -> SatisfactionChecker:
    """Return the Exp. 1 STL checker for a fixed ego trajectory."""
    def satisfies(x_star: np.ndarray, scenario: Dict[str, np.ndarray]) -> bool:
        ego_xy = np.asarray(x_star[:2, :].T, dtype=float)
        ped_xy = np.asarray(scenario["pedestrian"], dtype=float)
        amb_xy = np.asarray(scenario["ambulance"], dtype=float)

        T = min(len(ego_xy), len(ped_xy), len(amb_xy))

        # Same L_inf keep-out robustness used by Exp. 1's STL constraints.
        amb_rho = np.maximum(
            np.abs(ego_xy[:T, 0] - amb_xy[:T, 0]),
            np.abs(ego_xy[:T, 1] - amb_xy[:T, 1]),
        ) - d_amb
        if np.min(amb_rho) < 0.0:
            return False

        if emergency:
            ped_rho = np.maximum(
                np.abs(ego_xy[:T, 0] - ped_xy[:T, 0]),
                np.abs(ego_xy[:T, 1] - ped_xy[:T, 1]),
            ) - d_ped
            if np.min(ped_rho) < 0.0:
                return False

        # Deterministic lane STL: the same implication encoded in stl.py.
        px = ego_xy[:, 0]
        py = ego_xy[:, 1]
        in_y = (py >= float(lane["y_min"])) & (py <= float(lane["y_max"]))
        lane_rho = np.where(
            in_y,
            np.minimum(
                px - float(lane["x_min"]),
                float(lane["x_max"]) - px,
            ),
            1.0,
        )
        return bool(np.min(lane_rho) >= 0.0)

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
    """Validate one fixed feasible solution and estimate its violation rate.

    The sampler must return one joint scenario. ``satisfies`` must evaluate all
    active STL specifications on the fixed trajectory and return a single
    Boolean. No optimization or resampling of the control occurs here.
    """
    if N_approx <= 0:
        raise ValueError(f"N_approx must be positive, got {N_approx}")

    x_star = np.asarray(x_star, dtype=float)
    u_star = np.asarray(u_star, dtype=float)
    M = validation_sample_size(eps, beta)

    # Independent validation set. Fail-fast is safe because certification
    # requires every validation scenario to satisfy every STL specification.
    validation_failures = 0
    for _ in range(M):
        if not satisfies(x_star, sampler()):
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

    # Only a solution that passes the theorem's validation set receives a
    # certificate and the expensive Monte Carlo estimate.
    if validated:
        violations = sum(
            not satisfies(x_star, sampler()) for _ in range(N_approx)
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
