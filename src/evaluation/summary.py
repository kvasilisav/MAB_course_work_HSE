from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    cumulative_regret,
    cumulative_reward,
    empirical_arm_posterior_means,
    suboptimal_arm_share,
    time_to_first_oracle_optimal,
    time_to_stable_best_arm,
)


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "policy_name",
                "final_cumulative_reward",
                "final_cumulative_regret",
                "suboptimal_share",
                "time_to_first_oracle_optimal",
                "time_to_stable_best_arm",
                "seed",
            ]
        )

    rows: list[dict[str, float | int | str | None]] = []
    for (run_id, policy_name, seed), group in results.groupby(["run_id", "policy_name", "seed"]):
        rewards = group["reward"].to_numpy()
        oracle = group["oracle_best_reward"].to_numpy()
        chosen = group["chosen_arm"].to_numpy(dtype=int)
        optimal = group["oracle_best_arm"].to_numpy(dtype=int)
        n_arms = int(max(chosen.max(initial=0), optimal.max(initial=0)) + 1)
        posterior_means = empirical_arm_posterior_means(chosen, rewards, n_arms)
        rows.append(
            {
                "run_id": run_id,
                "policy_name": policy_name,
                "final_cumulative_reward": float(cumulative_reward(rewards)[-1]),
                "final_cumulative_regret": float(cumulative_regret(rewards, oracle)[-1]),
                "suboptimal_share": float(suboptimal_arm_share(chosen, optimal)),
                "time_to_first_oracle_optimal": time_to_first_oracle_optimal(chosen, optimal),
                "time_to_stable_best_arm": time_to_stable_best_arm(posterior_means),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "seed"]).reset_index(drop=True)
