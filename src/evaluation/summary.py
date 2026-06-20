from __future__ import annotations

import pandas as pd

from .metrics import cumulative_regret, cumulative_reward, suboptimal_arm_share


def _time_to_best_arm(chosen_arms: pd.Series) -> int | None:
    if chosen_arms.empty:
        return None
    final_best = int(chosen_arms.mode().iloc[0])
    for idx, arm in enumerate(chosen_arms.to_list()):
        if arm == final_best:
            suffix = chosen_arms.iloc[idx:]
            if (suffix == final_best).mean() >= 0.95:
                return idx
    return None


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "policy_name",
                "final_cumulative_reward",
                "final_cumulative_regret",
                "suboptimal_share",
                "time_to_best_arm",
                "seed",
            ]
        )

    rows: list[dict[str, float | int | str]] = []
    for (run_id, policy_name, seed), group in results.groupby(["run_id", "policy_name", "seed"]):
        rewards = group["reward"].to_numpy()
        oracle = group["oracle_best_reward"].to_numpy()
        chosen = group["chosen_arm"].to_numpy()
        optimal = group["oracle_best_arm"].to_numpy()
        rows.append(
            {
                "run_id": run_id,
                "policy_name": policy_name,
                "final_cumulative_reward": float(cumulative_reward(rewards)[-1]),
                "final_cumulative_regret": float(cumulative_regret(rewards, oracle)[-1]),
                "suboptimal_share": float(suboptimal_arm_share(chosen, optimal)),
                "time_to_best_arm": _time_to_best_arm(group["chosen_arm"]),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "seed"]).reset_index(drop=True)
