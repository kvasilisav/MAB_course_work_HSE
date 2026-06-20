from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(slots=True)
class WeightedABInferenceResult:
    ips_control: float
    ips_treatment: float
    ate_ips: float
    snips_control: float
    snips_treatment: float
    ate_snips: float
    standard_error: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    reject_null: bool
    effective_sample_size: float
    n_observations: int


def _ips_arm_estimates(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensity_control: np.ndarray,
    propensity_treatment: np.ndarray,
    *,
    propensity_floor: float,
) -> tuple[float, float, float, float]:
    n = arms.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    p0 = np.maximum(propensity_control, propensity_floor)
    p1 = np.maximum(propensity_treatment, propensity_floor)

    control_mask = arms == 0
    treatment_mask = arms == 1

    ips_control = float(np.sum(rewards[control_mask] / p0[control_mask]) / n)
    ips_treatment = float(np.sum(rewards[treatment_mask] / p1[treatment_mask]) / n)

    w0 = rewards[control_mask] / p0[control_mask]
    w1 = rewards[treatment_mask] / p1[treatment_mask]
    sum_w0 = float(np.sum(1.0 / p0[control_mask])) if control_mask.any() else 0.0
    sum_w1 = float(np.sum(1.0 / p1[treatment_mask])) if treatment_mask.any() else 0.0
    snips_control = float(np.sum(w0) / sum_w0) if sum_w0 > 0 else 0.0
    snips_treatment = float(np.sum(w1) / sum_w1) if sum_w1 > 0 else 0.0

    return ips_control, ips_treatment, snips_control, snips_treatment


def _ips_ate_per_row(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensity_control: np.ndarray,
    propensity_treatment: np.ndarray,
    *,
    propensity_floor: float,
) -> np.ndarray:
    p0 = np.maximum(propensity_control, propensity_floor)
    p1 = np.maximum(propensity_treatment, propensity_floor)
    contrib_control = np.where(arms == 0, rewards / p0, 0.0)
    contrib_treatment = np.where(arms == 1, rewards / p1, 0.0)
    return contrib_treatment - contrib_control


def run_weighted_ab_inference(
    frame: pd.DataFrame,
    *,
    arm_col: str = "arm",
    reward_col: str = "reward",
    propensity_control_col: str = "propensity_control",
    propensity_treatment_col: str = "propensity_treatment",
    alpha: float = 0.05,
    two_sided: bool = True,
    propensity_floor: float = 0.01,
    n_bootstrap: int = 0,
    bootstrap_seed: int | None = None,
) -> WeightedABInferenceResult:
    required = {arm_col, reward_col, propensity_control_col, propensity_treatment_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame misses columns: {sorted(missing)}")

    arms = frame[arm_col].astype(int).to_numpy()
    rewards = frame[reward_col].astype(float).to_numpy()
    p0 = frame[propensity_control_col].astype(float).to_numpy()
    p1 = frame[propensity_treatment_col].astype(float).to_numpy()
    n = int(arms.shape[0])
    if n == 0:
        raise ValueError("frame must contain at least one observation.")

    ips_control, ips_treatment, snips_control, snips_treatment = _ips_arm_estimates(
        arms,
        rewards,
        p0,
        p1,
        propensity_floor=propensity_floor,
    )
    ate_ips = ips_treatment - ips_control
    ate_snips = snips_treatment - snips_control

    row_ate = _ips_ate_per_row(arms, rewards, p0, p1, propensity_floor=propensity_floor)
    ate_point = float(row_ate.mean())

    if n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap_estimates = np.empty(n_bootstrap, dtype=float)
        for idx in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            bootstrap_estimates[idx] = float(row_ate[sample_idx].mean())
        standard_error = float(np.std(bootstrap_estimates, ddof=1))
    else:
        standard_error = float(np.std(row_ate, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z_score = float(ate_point / standard_error) if standard_error > 0 else 0.0
    if two_sided:
        p_value = float(2 * (1 - norm.cdf(abs(z_score))))
        z_alpha = float(norm.ppf(1 - alpha / 2))
    else:
        p_value = float(1 - norm.cdf(z_score))
        z_alpha = float(norm.ppf(1 - alpha))
    ci_low = float(ate_point - z_alpha * standard_error)
    ci_high = float(ate_point + z_alpha * standard_error)
    reject_null = bool(p_value < alpha)

    p0_clipped = np.maximum(p0, propensity_floor)
    p1_clipped = np.maximum(p1, propensity_floor)
    weights = np.where(arms == 0, 1.0 / p0_clipped, 1.0 / p1_clipped)
    weight_sum = float(weights.sum())
    weight_sq_sum = float(np.sum(np.square(weights)))
    ess = float((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0 else 0.0

    return WeightedABInferenceResult(
        ips_control=ips_control,
        ips_treatment=ips_treatment,
        ate_ips=ate_ips,
        snips_control=snips_control,
        snips_treatment=snips_treatment,
        ate_snips=ate_snips,
        standard_error=standard_error,
        z_score=z_score,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        reject_null=reject_null,
        effective_sample_size=ess,
        n_observations=n,
    )
