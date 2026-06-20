from __future__ import annotations

import pandas as pd

from src.ab_testing.propensity import thompson_sampling_propensities
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.environments.bernoulli import BernoulliBanditEnv


def simulate_thompson_logs_with_propensity(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
    propensity_mc_samples: int = 100,
) -> pd.DataFrame:
    env = BernoulliBanditEnv([p_control, p_treatment], horizon=horizon, seed=seed)
    policy = ThompsonSamplingPolicy(n_arms=2, seed=seed + 17)
    env.reset()
    policy.reset()

    rows: list[dict[str, float | int | str]] = []
    for step in range(horizon):
        propensities = thompson_sampling_propensities(
            policy,
            n_mc=propensity_mc_samples,
            seed=seed + step,
        )
        arm = int(policy.select_arm())
        reward, _ = env.step(arm)
        rows.append(
            {
                "step": step,
                "arm": arm,
                "reward": float(reward),
                "propensity_control": float(propensities[0]),
                "propensity_treatment": float(propensities[1]),
                "group": "treatment" if arm == 1 else "control",
            }
        )
        policy.update(arm, reward)
    return pd.DataFrame(rows)


def logs_to_ab_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["group", "reward"]].copy()
