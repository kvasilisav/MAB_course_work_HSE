import numpy as np

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.linucb import LinUCBPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy


def test_policies_return_valid_arm_indices() -> None:
    policies = [
        FixedABPolicy(n_arms=3, seed=1),
        EpsilonGreedyPolicy(n_arms=3, epsilon=0.2, seed=1),
        UCB1Policy(n_arms=3, seed=1),
        ThompsonSamplingPolicy(n_arms=3, seed=1),
        LinUCBPolicy(n_arms=3, context_dim=2, seed=1),
    ]
    for policy in policies:
        for _ in range(20):
            arm = policy.select_arm()
            assert 0 <= arm < 3
            policy.update(arm, reward=float(np.random.binomial(1, 0.5)))


def test_linucb_uses_context_dimension() -> None:
    policy = LinUCBPolicy(n_arms=2, context_dim=2, seed=1)
    arm = policy.select_arm(context=np.array([1.0, 0.0]))
    policy.update(arm=arm, reward=1.0, context=np.array([1.0, 0.0]))
    state = policy.snapshot()
    assert state["context_dim"] == 2


def test_ucb1_explores_all_arms_first() -> None:
    policy = UCB1Policy(n_arms=4, seed=1)
    seen = set()
    for _ in range(4):
        arm = policy.select_arm()
        seen.add(arm)
        policy.update(arm, reward=0.0)
    assert seen == {0, 1, 2, 3}


def test_epsilon_zero_is_deterministic_greedy() -> None:
    policy = EpsilonGreedyPolicy(n_arms=2, epsilon=0.0, seed=1)
    # Arm 1 has higher estimate after updates.
    policy.update(arm=0, reward=0.0)
    policy.update(arm=1, reward=1.0)
    assert policy.select_arm() == 1


def test_thompson_sampling_posterior_updates() -> None:
    policy = ThompsonSamplingPolicy(n_arms=2, alpha_prior=1.0, beta_prior=1.0, seed=1)
    policy.update(arm=0, reward=1.0)
    policy.update(arm=0, reward=0.0)
    state = policy.snapshot()
    assert state["alphas"][0] == 2.0
    assert state["betas"][0] == 2.0
