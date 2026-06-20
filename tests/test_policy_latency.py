from src.experiments.policy_latency import benchmark_policy


def test_latency_benchmark_returns_positive_values() -> None:
    row = benchmark_policy(
        policy_name="fixed_ab",
        n_arms=5,
        context_dim=1,
        n_warmup=10,
        n_calls=100,
    )
    assert row["select_arm_us_mean"] > 0.0
    assert row["update_us_mean"] >= 0.0


def test_linucb_slower_than_thompson_on_many_arms() -> None:
    lin = benchmark_policy(
        policy_name="linucb",
        n_arms=80,
        context_dim=7,
        n_warmup=5,
        n_calls=50,
    )
    ts = benchmark_policy(
        policy_name="thompson_sampling",
        n_arms=80,
        context_dim=7,
        n_warmup=5,
        n_calls=50,
    )
    assert lin["select_arm_us_mean"] > ts["select_arm_us_mean"]
