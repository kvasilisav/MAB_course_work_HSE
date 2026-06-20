from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm

from .inference import run_ab_inference


@dataclass(slots=True)
class SequentialInferenceResult:
    method: str
    reject_null: bool
    stopped_at_look: int
    stopped_at_n: int
    stop_fraction: float
    z_score: float
    p_value: float


def obf_critical_value(*, alpha: float, look_index: int, n_looks: int, two_sided: bool = True) -> float:
    """O'Brien-Fleming boundary at look k of K (information fraction k/K)."""
    if look_index < 1 or look_index > n_looks:
        raise ValueError("look_index must be in [1, n_looks]")
    tau = look_index / n_looks
    z_alpha = float(norm.ppf(1 - alpha)) if not two_sided else float(norm.ppf(1 - alpha / 2))
    return z_alpha / math.sqrt(tau)


def _look_sizes(horizon: int, look_fractions: list[float]) -> list[int]:
    if not look_fractions:
        raise ValueError("look_fractions must be non-empty")
    if any(f <= 0 or f > 1 for f in look_fractions):
        raise ValueError("look_fractions must be in (0, 1]")
    sizes = [max(2, int(round(horizon * fraction))) for fraction in look_fractions]
    sizes[-1] = horizon
    for idx in range(1, len(sizes)):
        if sizes[idx] <= sizes[idx - 1]:
            sizes[idx] = min(horizon, sizes[idx - 1] + 1)
    return sizes


def _prefix_frame(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame
    return frame.iloc[:n].copy()


def run_fixed_horizon_inference(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.05,
    group_col: str = "group",
    reward_col: str = "reward",
) -> SequentialInferenceResult:
    result = run_ab_inference(frame, alpha=alpha, group_col=group_col, reward_col=reward_col)
    n = int(len(frame))
    return SequentialInferenceResult(
        method="fixed_horizon",
        reject_null=result.reject_null,
        stopped_at_look=1,
        stopped_at_n=n,
        stop_fraction=1.0,
        z_score=result.z_score,
        p_value=result.p_value,
    )


def run_naive_peeking_inference(
    frame: pd.DataFrame,
    *,
    look_fractions: list[float],
    alpha: float = 0.05,
    group_col: str = "group",
    reward_col: str = "reward",
) -> SequentialInferenceResult:
    horizon = len(frame)
    look_sizes = _look_sizes(horizon, look_fractions)
    last_result = None

    for look_idx, n in enumerate(look_sizes, start=1):
        prefix = _prefix_frame(frame, n)
        if (prefix[group_col] == "control").sum() == 0 or (prefix[group_col] == "treatment").sum() == 0:
            continue
        last_result = run_ab_inference(prefix, alpha=alpha, group_col=group_col, reward_col=reward_col)
        if last_result.reject_null:
            return SequentialInferenceResult(
                method="naive_peek",
                reject_null=True,
                stopped_at_look=look_idx,
                stopped_at_n=n,
                stop_fraction=n / horizon,
                z_score=last_result.z_score,
                p_value=last_result.p_value,
            )

    if last_result is None:
        return SequentialInferenceResult(
            method="naive_peek",
            reject_null=False,
            stopped_at_look=len(look_sizes),
            stopped_at_n=look_sizes[-1],
            stop_fraction=look_sizes[-1] / horizon,
            z_score=0.0,
            p_value=1.0,
        )

    return SequentialInferenceResult(
        method="naive_peek",
        reject_null=False,
        stopped_at_look=len(look_sizes),
        stopped_at_n=look_sizes[-1],
        stop_fraction=look_sizes[-1] / horizon,
        z_score=last_result.z_score,
        p_value=last_result.p_value,
    )


def run_group_sequential_obf_inference(
    frame: pd.DataFrame,
    *,
    look_fractions: list[float],
    alpha: float = 0.05,
    group_col: str = "group",
    reward_col: str = "reward",
) -> SequentialInferenceResult:
    horizon = len(frame)
    look_sizes = _look_sizes(horizon, look_fractions)
    n_looks = len(look_sizes)
    last_result = None

    for look_idx, n in enumerate(look_sizes, start=1):
        prefix = _prefix_frame(frame, n)
        if (prefix[group_col] == "control").sum() == 0 or (prefix[group_col] == "treatment").sum() == 0:
            continue
        last_result = run_ab_inference(prefix, alpha=alpha, group_col=group_col, reward_col=reward_col)
        boundary = obf_critical_value(alpha=alpha, look_index=look_idx, n_looks=n_looks)
        if abs(last_result.z_score) >= boundary:
            return SequentialInferenceResult(
                method="group_sequential_obf",
                reject_null=True,
                stopped_at_look=look_idx,
                stopped_at_n=n,
                stop_fraction=n / horizon,
                z_score=last_result.z_score,
                p_value=last_result.p_value,
            )

    if last_result is None:
        return SequentialInferenceResult(
            method="group_sequential_obf",
            reject_null=False,
            stopped_at_look=n_looks,
            stopped_at_n=look_sizes[-1],
            stop_fraction=look_sizes[-1] / horizon,
            z_score=0.0,
            p_value=1.0,
        )

    return SequentialInferenceResult(
        method="group_sequential_obf",
        reject_null=False,
        stopped_at_look=n_looks,
        stopped_at_n=look_sizes[-1],
        stop_fraction=look_sizes[-1] / horizon,
        z_score=last_result.z_score,
        p_value=last_result.p_value,
    )


def bernoulli_msprt_mixture_likelihood(
    *,
    n: int,
    theta_hat: float,
    variance_sum: float,
    tau: float,
) -> float:
    """Mixture likelihood ratio for two-stream Bernoulli data (Johari et al. 2022, §6.1)."""
    if n <= 0 or variance_sum <= 0 or tau <= 0:
        return 0.0
    denom = variance_sum + n * tau**2
    ratio = math.sqrt(variance_sum / denom)
    exponent = (n**2 * tau**2 * theta_hat**2) / (2 * variance_sum * denom)
    return ratio * math.exp(exponent)


def run_always_valid_msprt_inference(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.05,
    tau: float = 0.02,
    min_per_arm: int = 100,
    theta0: float = 0.0,
    group_col: str = "group",
    reward_col: str = "reward",
    control_label: str = "control",
    treatment_label: str = "treatment",
) -> SequentialInferenceResult:
    """Always-valid mSPRT with continuous monitoring (Johari et al. 2022, §6.1).

    Logs are converted to two arrival-order Bernoulli streams; at each new pair
    (control_i, treatment_i) the mixture likelihood ratio is updated.
    """
    if min_per_arm < 1:
        raise ValueError("min_per_arm must be >= 1")
    if tau <= 0:
        raise ValueError("tau must be positive")

    horizon = len(frame)
    control_stream: list[float] = []
    treatment_stream: list[float] = []
    always_valid_p = 1.0
    last_theta = 0.0
    variance_sum = 0.0
    n_pairs = 0
    stop_step = horizon

    for step, row in enumerate(frame.itertuples(index=False), start=1):
        group = getattr(row, group_col)
        reward = float(getattr(row, reward_col))
        if group == control_label:
            control_stream.append(reward)
        elif group == treatment_label:
            treatment_stream.append(reward)
        else:
            continue

        new_pairs = min(len(control_stream), len(treatment_stream))
        if new_pairs <= n_pairs or new_pairs < min_per_arm:
            continue
        n_pairs = new_pairs

        control_vals = control_stream[:n_pairs]
        treatment_vals = treatment_stream[:n_pairs]
        mu_control = sum(control_vals) / n_pairs
        mu_treatment = sum(treatment_vals) / n_pairs
        theta_hat = mu_treatment - mu_control - theta0
        last_theta = theta_hat
        variance_sum = mu_control * (1 - mu_control) + mu_treatment * (1 - mu_treatment)
        if variance_sum <= 0:
            continue

        mixture_lr = bernoulli_msprt_mixture_likelihood(
            n=n_pairs,
            theta_hat=theta_hat,
            variance_sum=variance_sum,
            tau=tau,
        )
        if mixture_lr <= 0:
            continue

        always_valid_p = min(always_valid_p, 1.0 / mixture_lr)
        if always_valid_p < alpha:
            stop_step = step
            return SequentialInferenceResult(
                method="always_valid_msprt",
                reject_null=True,
                stopped_at_look=step,
                stopped_at_n=step,
                stop_fraction=step / horizon,
                z_score=last_theta / math.sqrt(variance_sum / n_pairs) if variance_sum > 0 else 0.0,
                p_value=always_valid_p,
            )

    se = math.sqrt(variance_sum / n_pairs) if variance_sum > 0 and n_pairs > 0 else 0.0
    return SequentialInferenceResult(
        method="always_valid_msprt",
        reject_null=False,
        stopped_at_look=stop_step,
        stopped_at_n=stop_step,
        stop_fraction=stop_step / horizon,
        z_score=last_theta / se if se > 0 else 0.0,
        p_value=always_valid_p,
    )
