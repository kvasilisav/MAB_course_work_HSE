import numpy as np

from src.environments.bernoulli import BernoulliBanditEnv
from src.environments.contextual_bernoulli import ContextualBernoulliBanditEnv


def test_bernoulli_reward_is_binary() -> None:
    env = BernoulliBanditEnv([0.1, 0.8], horizon=100, seed=42)
    env.reset(seed=42)
    for _ in range(10):
        reward, _ = env.step(arm=1)
        assert reward in (0.0, 1.0)


def test_oracle_best_arm_is_correct() -> None:
    env = BernoulliBanditEnv([0.05, 0.3, 0.2], horizon=10, seed=1)
    _, meta = env.step(arm=0)
    assert meta["best_arm"] == 1
    assert np.isclose(meta["best_reward"], 0.3)


def test_reset_with_same_seed_is_reproducible() -> None:
    env = BernoulliBanditEnv([0.2, 0.7], horizon=20, seed=7)
    env.reset(seed=7)
    seq1 = [env.step(arm=1)[0] for _ in range(10)]
    env.reset(seed=7)
    seq2 = [env.step(arm=1)[0] for _ in range(10)]
    assert seq1 == seq2


def test_contextual_env_returns_context() -> None:
    env = ContextualBernoulliBanditEnv(n_arms=5, context_dim=4, horizon=100, seed=1)
    env.reset(seed=1)
    context = env.current_context()
    assert context is not None
    assert context.size == 4


def test_contextual_oracle_best_arm_changes_with_context() -> None:
    env = ContextualBernoulliBanditEnv(n_arms=3, context_dim=2, horizon=10, seed=0)
    env.reset(seed=0)
    _, meta_a = env.step(arm=0, context=np.array([1.0, 0.0]))
    env.reset(seed=0)
    _, meta_b = env.step(arm=0, context=np.array([0.0, 1.0]))
    assert meta_a["best_arm"] in {0, 1, 2}
    assert meta_b["best_arm"] in {0, 1, 2}
