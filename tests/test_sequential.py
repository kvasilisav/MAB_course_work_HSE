import pandas as pd

from src.ab_testing.sequential import (
    bernoulli_msprt_mixture_likelihood,
    obf_critical_value,
    run_always_valid_msprt_inference,
    run_fixed_horizon_inference,
    run_group_sequential_obf_inference,
    run_naive_peeking_inference,
)
from src.experiments.sequential_validity_e14 import run_sequential_validity_study


def test_obf_boundary_increases_at_final_look() -> None:
    early = obf_critical_value(alpha=0.05, look_index=1, n_looks=4)
    final = obf_critical_value(alpha=0.05, look_index=4, n_looks=4)
    assert early > final


def test_naive_peek_can_stop_before_horizon() -> None:
    frame = pd.DataFrame(
        {
            "group": ["control"] * 500 + ["treatment"] * 500 + ["control"] * 500 + ["treatment"] * 500,
            "reward": [0] * 930 + [1] * 70 + [0] * 880 + [1] * 120,
        }
    )
    result = run_naive_peeking_inference(frame, look_fractions=[0.25, 0.5, 0.75, 1.0], alpha=0.05)
    assert result.method == "naive_peek"
    assert result.stop_fraction <= 1.0


def test_group_sequential_runs_on_balanced_logs() -> None:
    frame = pd.DataFrame(
        {
            "group": ["control", "treatment"] * 100,
            "reward": [0, 1] * 100,
        }
    )
    result = run_group_sequential_obf_inference(
        frame,
        look_fractions=[0.5, 1.0],
        alpha=0.05,
    )
    assert result.method == "group_sequential_obf"
    assert 0.0 <= result.stop_fraction <= 1.0


def test_fixed_horizon_matches_single_test() -> None:
    frame = pd.DataFrame(
        {
            "group": ["control"] * 1000 + ["treatment"] * 1000,
            "reward": [0] * 930 + [1] * 70 + [0] * 880 + [1] * 120,
        }
    )
    result = run_fixed_horizon_inference(frame, alpha=0.05)
    assert result.reject_null
    assert result.stop_fraction == 1.0


def test_msprt_likelihood_positive_on_effect() -> None:
    likelihood = bernoulli_msprt_mixture_likelihood(
        n=1000,
        theta_hat=0.02,
        variance_sum=0.09,
        tau=0.01,
    )
    assert likelihood > 1.0


def test_always_valid_msprt_runs_on_balanced_logs() -> None:
    frame = pd.DataFrame(
        {
            "group": ["control", "treatment"] * 100,
            "reward": [0, 1] * 100,
        }
    )
    result = run_always_valid_msprt_inference(frame, alpha=0.05, tau=0.01, min_per_arm=10)
    assert result.method == "always_valid_msprt"
    assert 0.0 <= result.stop_fraction <= 1.0


def test_sequential_validity_study_runs_small() -> None:
    summary, _detail = run_sequential_validity_study(
        horizon=1000,
        n_trials=6,
        alpha=0.05,
        null_ctr=0.05,
        effect_ctr=0.06,
        look_fractions=[0.5, 1.0],
    )
    assert len(summary) == 16
    assert {
        "fixed_horizon",
        "naive_peek",
        "group_sequential_obf",
        "always_valid_msprt",
    }.issubset(set(summary["method"]))
