from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm


@dataclass(slots=True)
class ABInferenceResult:
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    ate: float
    relative_lift: float
    standard_error: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    reject_null: bool


def run_ab_inference(
    frame: pd.DataFrame,
    *,
    group_col: str = "group",
    reward_col: str = "reward",
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05,
    two_sided: bool = True,
) -> ABInferenceResult:
    control = frame[frame[group_col] == control_label][reward_col].astype(float)
    treatment = frame[frame[group_col] == treatment_label][reward_col].astype(float)
    if control.empty or treatment.empty:
        raise ValueError("Both control and treatment groups must contain at least one observation.")

    n_control = int(control.shape[0])
    n_treatment = int(treatment.shape[0])
    p_control = float(control.mean())
    p_treatment = float(treatment.mean())
    ate = p_treatment - p_control
    relative_lift = float(ate / p_control) if p_control != 0 else 0.0

    pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    standard_error = float((pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment)) ** 0.5)
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

    return ABInferenceResult(
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=p_control,
        treatment_rate=p_treatment,
        ate=ate,
        relative_lift=relative_lift,
        standard_error=standard_error,
        z_score=z_score,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        reject_null=reject_null,
    )
