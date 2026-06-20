from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm


@dataclass(slots=True)
class ABDesignSpec:
    baseline_rate: float
    minimum_detectable_effect: float
    alpha: float = 0.05
    power: float = 0.8
    two_sided: bool = True


def required_sample_size_two_proportions(spec: ABDesignSpec) -> int:
    p1 = spec.baseline_rate
    p2 = p1 + spec.minimum_detectable_effect
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("baseline_rate and baseline+MDE must be in (0, 1).")
    alpha_tail = spec.alpha / 2.0 if spec.two_sided else spec.alpha
    z_alpha = norm.ppf(1 - alpha_tail)
    z_beta = norm.ppf(spec.power)
    pooled = (p1 + p2) / 2.0
    num = (
        z_alpha * (2 * pooled * (1 - pooled)) ** 0.5
        + z_beta * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5
    ) ** 2
    den = (p2 - p1) ** 2
    return int(num / den) + 1
