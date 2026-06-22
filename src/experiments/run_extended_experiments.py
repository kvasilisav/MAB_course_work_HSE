from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.experiments.policy_latency import run_latency_matrix
from src.pipeline.loader import load_events


def _run_command(command: list[str], cwd: Path) -> None:
    print(">", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _prepare_obd(
    *,
    root: Path,
    behavior_policy: str,
    campaign: str,
    output_path: Path,
) -> None:
    if output_path.exists():
        print(f"skip prepare (exists): {output_path}")
        return
    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.prepare_open_bandit",
            "--download",
            "--include-context",
            "--behavior-policy",
            behavior_policy,
            "--campaign",
            campaign,
            "--output-path",
            str(output_path),
        ],
        cwd=root,
    )


def _run_compare(
    *,
    root: Path,
    mode: str,
    policies: str,
    events_path: Path,
    output_dir: Path,
    seeds: int,
    horizon: int | None = None,
    env: str = "bernoulli",
    context_dim: int = 4,
    batch_size: int = 200,
    n_bootstrap: int = 0,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.experiments.compare_ab_vs_bandits",
        "--mode",
        mode,
        "--policies",
        policies,
        "--events-path",
        str(events_path),
        "--output-dir",
        str(output_dir),
        "--seeds",
        str(seeds),
        "--env",
        env,
        "--context-dim",
        str(context_dim),
        "--batch-size",
        str(batch_size),
    ]
    if horizon is not None:
        command.extend(["--horizon", str(horizon)])
    if n_bootstrap > 0:
        command.extend(["--bootstrap", str(n_bootstrap)])
    _run_command(command, cwd=root)

    if mode == "ope":
        return pd.read_csv(output_dir / "ope_summary.csv")
    return pd.read_csv(output_dir / "summary.csv")


def _summarize_ope(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    grouped = (
        frame.groupby("policy_name")[
            ["acceptance_rate", "ips_estimate", "snips_estimate", "effective_sample_size"]
        ]
        .mean()
        .reset_index()
    )
    grouped["dataset"] = label
    return grouped


def _summarize_batch(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    grouped = (
        frame.groupby("policy_name")[
            ["final_cumulative_reward", "final_cumulative_regret", "suboptimal_share"]
        ]
        .mean()
        .reset_index()
    )
    grouped["dataset"] = label
    return grouped


def _summarize_synthetic(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    return _summarize_batch(frame, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended coursework experiments (E8, E2b/c, latency).")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds-synthetic", type=int, default=15)
    parser.add_argument("--seeds-obd-batch", type=int, default=5)
    parser.add_argument("--seeds-ope", type=int, default=5)
    parser.add_argument("--horizon-contextual", type=int, default=5000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output-dir", default="outputs/extended")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    contextual_policies = "fixed_ab,thompson_sampling,linucb"
    ope_policies = "fixed_ab,thompson_sampling,epsilon_greedy,ucb1,linucb"

    # E8b — contextual synthetic
    e8b_summary = _run_compare(
        root=root,
        mode="synthetic",
        policies=contextual_policies,
        events_path=root / "data/processed/obd_events.csv",
        output_dir=out / "e8b_contextual_synthetic",
        seeds=args.seeds_synthetic,
        horizon=args.horizon_contextual,
        env="contextual",
        context_dim=4,
    )
    e8b_agg = _summarize_synthetic(e8b_summary, "contextual_synthetic")

    # E8 — OBD batch with real context
    obd_random_all = root / "data/processed/obd_events.csv"
    if not obd_random_all.exists() and not args.skip_download:
        _prepare_obd(
            root=root,
            behavior_policy="random",
            campaign="all",
            output_path=obd_random_all,
        )
    events = load_events(obd_random_all)
    context_dim = int(events[0].context.size) if events and events[0].context is not None else 7

    e8_summary = _run_compare(
        root=root,
        mode="batch",
        policies=contextual_policies,
        events_path=obd_random_all,
        output_dir=out / "e8_obd_batch_contextual",
        seeds=args.seeds_obd_batch,
        env="bernoulli",
        context_dim=context_dim,
        batch_size=200,
    )
    e8_agg = _summarize_batch(e8_summary, "obd_random_all_batch")

    # E2b/c — OPE matrix
    ope_specs = [
        ("random", "all", "ope_random_all"),
        ("bts", "all", "ope_bts_all"),
        ("random", "men", "ope_random_men"),
        ("random", "women", "ope_random_women"),
    ]
    ope_frames: list[pd.DataFrame] = []
    for behavior, campaign, folder in ope_specs:
        events_path = root / f"data/processed/obd_{behavior}_{campaign}.csv"
        if not events_path.exists() and not args.skip_download:
            _prepare_obd(
                root=root,
                behavior_policy=behavior,
                campaign=campaign,
                output_path=events_path,
            )
        if not events_path.exists():
            print(f"skip OPE (missing): {events_path}")
            continue
        ope_summary = _run_compare(
            root=root,
            mode="ope",
            policies=ope_policies,
            events_path=events_path,
            output_dir=out / folder,
            seeds=args.seeds_ope,
        )
        ope_frames.append(_summarize_ope(ope_summary, f"{behavior}_{campaign}"))

    ope_agg = pd.concat(ope_frames, ignore_index=True) if ope_frames else pd.DataFrame()

    # P1 — latency benchmark
    latency = run_latency_matrix(n_calls=3000)
    latency_path = out / "policy_latency.csv"
    latency.to_csv(latency_path, index=False)

    # Aggregate master tables
    e8b_agg.to_csv(out / "e8b_contextual_synthetic_summary.csv", index=False)
    e8_agg.to_csv(out / "e8_obd_batch_summary.csv", index=False)
    if not ope_agg.empty:
        ope_agg.to_csv(out / "ope_matrix_summary.csv", index=False)

    report = {
        "e8b_contextual_synthetic": e8b_agg.to_dict(orient="records"),
        "e8_obd_batch": e8_agg.to_dict(orient="records"),
        "ope_matrix": ope_agg.to_dict(orient="records") if not ope_agg.empty else [],
        "latency": latency.to_dict(orient="records"),
    }
    report_path = out / "extended_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== E8b contextual synthetic (mean regret) ===")
    print(e8b_agg.sort_values("final_cumulative_regret"))
    print("\n=== E8 OBD batch contextual (mean regret) ===")
    print(e8_agg.sort_values("final_cumulative_regret"))
    if not ope_agg.empty:
        print("\n=== OPE matrix (mean SNIPS) ===")
        print(ope_agg.sort_values(["dataset", "snips_estimate"], ascending=[True, False]))
    print("\n=== Latency (us per decision) ===")
    print(latency.sort_values("total_decision_us_mean"))


if __name__ == "__main__":
    main()
