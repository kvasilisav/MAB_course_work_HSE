from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.ab_testing.inference import ABInferenceResult
from src.bandits.fixed_ab import FixedABPolicy


@dataclass(slots=True)
class CUPEDABInferenceResult(ABInferenceResult):
    theta: float
    pre_covariate_mean: float


def estimate_cuped_theta(rewards: np.ndarray, covariate: np.ndarray) -> float:
    x = np.asarray(covariate, dtype=float)
    y = np.asarray(rewards, dtype=float)
    if x.size < 2:
        return 0.0
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0.0:
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def run_cuped_ab_inference(
    frame: pd.DataFrame,
    *,
    group_col: str = "group",
    reward_col: str = "reward",
    covariate_col: str = "pre_reward",
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05,
    two_sided: bool = True,
) -> CUPEDABInferenceResult:
    rewards = frame[reward_col].astype(float).to_numpy()
    covariate = frame[covariate_col].astype(float).to_numpy()
    groups = frame[group_col].astype(str).to_numpy()

    control_mask = groups == control_label
    treatment_mask = groups == treatment_label
    if not control_mask.any() or not treatment_mask.any():
        raise ValueError("Both control and treatment groups must contain at least one observation.")

    theta = estimate_cuped_theta(rewards, covariate)
    x_mean = float(covariate.mean())
    adjusted = rewards - theta * (covariate - x_mean)

    control_adj = adjusted[control_mask]
    treatment_adj = adjusted[treatment_mask]
    n_control = int(control_adj.shape[0])
    n_treatment = int(treatment_adj.shape[0])
    control_rate = float(control_adj.mean())
    treatment_rate = float(treatment_adj.mean())
    ate = treatment_rate - control_rate
    relative_lift = float(ate / control_rate) if control_rate != 0 else 0.0

    var_control = float(np.var(control_adj, ddof=1)) if n_control > 1 else 0.0
    var_treatment = float(np.var(treatment_adj, ddof=1)) if n_treatment > 1 else 0.0
    standard_error = float((var_control / n_control + var_treatment / n_treatment) ** 0.5)
    z_score = float(ate / standard_error) if standard_error > 0 else 0.0
    if two_sided:
        p_value = float(2 * (1 - norm.cdf(abs(z_score))))
        z_alpha = float(norm.ppf(1 - alpha / 2))
    else:
        p_value = float(1 - norm.cdf(z_score))
        z_alpha = float(norm.ppf(1 - alpha))
    ci_low = float(ate - z_alpha * standard_error)
    ci_high = float(ate + z_alpha * standard_error)
    reject_null = bool(p_value < alpha)

    return CUPEDABInferenceResult(
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        ate=ate,
        relative_lift=relative_lift,
        standard_error=standard_error,
        z_score=z_score,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        reject_null=reject_null,
        theta=theta,
        pre_covariate_mean=x_mean,
    )


def simulate_fixed_ab_with_covariate(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
    latent_strength: float = 0.25,
) -> pd.DataFrame:
    """Fixed 50/50 A/B with a pre-experiment covariate correlated with clicks."""
    rng = np.random.default_rng(seed)
    policy = FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=seed + 17)
    policy.reset(seed=seed + 17)
    rows: list[dict[str, float | int | str]] = []

    for step in range(horizon):
        latent = float(rng.beta(2.0, 5.0))
        pre_reward = latent
        arm = int(policy.select_arm())
        base_p = p_treatment if arm == 1 else p_control
        click_prob = float(np.clip(base_p + latent_strength * (latent - 0.29), 0.0, 1.0))
        reward = float(rng.binomial(1, click_prob))
        rows.append(
            {
                "step": step,
                "arm": arm,
                "group": "treatment" if arm == 1 else "control",
                "pre_reward": pre_reward,
                "reward": reward,
            }
        )
        policy.update(arm, reward)

    return pd.DataFrame(rows)
