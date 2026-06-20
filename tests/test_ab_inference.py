import pandas as pd

from src.ab_testing.design import ABDesignSpec, required_sample_size_two_proportions
from src.ab_testing.inference import run_ab_inference


def test_ab_inference_detects_clear_effect() -> None:
    frame = pd.DataFrame(
        {
            "group": ["control"] * 1000 + ["treatment"] * 1000,
            "reward": [0] * 930 + [1] * 70 + [0] * 880 + [1] * 120,
        }
    )
    result = run_ab_inference(frame, alpha=0.05)
    assert result.treatment_rate > result.control_rate
    assert result.p_value < 0.05
    assert result.reject_null


def test_sample_size_is_positive() -> None:
    spec = ABDesignSpec(baseline_rate=0.05, minimum_detectable_effect=0.01, alpha=0.05, power=0.8)
    n = required_sample_size_two_proportions(spec)
    assert n > 0
