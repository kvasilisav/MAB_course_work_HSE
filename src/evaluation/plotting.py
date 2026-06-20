from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_policy(
    results: pd.DataFrame,
    metric: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, group in results.groupby("policy_name"):
        grouped = group.groupby("step")[metric].mean()
        ax.plot(grouped.index, grouped.values, label=policy)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.25)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_arm_share(results: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    arm_share = (
        results.groupby(["policy_name", "chosen_arm"])
        .size()
        .div(results.groupby("policy_name").size(), level=0)
        .reset_index(name="share")
    )
    for policy, group in arm_share.groupby("policy_name"):
        ax.plot(group["chosen_arm"], group["share"], marker="o", label=policy)
    ax.set_title("Traffic Share by Arm")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Share")
    ax.legend()
    ax.grid(alpha=0.25)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig
