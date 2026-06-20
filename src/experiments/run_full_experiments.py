from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.experiments.obd_streaming_ope import run_streaming_obd_ope
from src.experiments.policy_latency import run_latency_matrix
from src.experiments.run_extended_experiments import (
    _prepare_obd,
    _run_compare,
    _summarize_batch,
    _summarize_ope,
    _summarize_synthetic,
)


def _run_streaming_ope_matrix(
    *,
    root: Path,
    specs: list[tuple[str, str, str]],
    seeds: int,
    include_linucb: bool,
    max_events: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for behavior, campaign, label in specs:
        input_path = root / "data/raw" / f"obd_{behavior}_{campaign}.csv"
        if not input_path.exists():
            print(f"skip streaming OPE (missing): {input_path}")
            continue
        print(f"streaming OPE: {label}, seeds={seeds}, max_events={max_events}")
        summary = run_streaming_obd_ope(
            input_path=input_path,
            behavior_policy=behavior,
            campaign=campaign,
            n_arms=None,
            seeds=seeds,
            chunksize=100_000,
            max_events=max_events,
            include_context=include_linucb,
            include_linucb=include_linucb,
        )
        out_dir = root / "outputs/extended_full" / f"streaming_ope_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "ope_summary.csv", index=False)
        agg = _summarize_ope(summary, label)
        frames.append(agg)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-scale extended experiments (20 seeds, full small-OBD rows, separate output dir).",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds-synthetic", type=int, default=20)
    parser.add_argument("--seeds-obd-batch", type=int, default=20)
    parser.add_argument("--seeds-ope", type=int, default=20)
    parser.add_argument("--horizon-contextual", type=int, default=10_000)
    parser.add_argument("--latency-calls", type=int, default=20_000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-linucb-ope", action="store_true", help="Skip LinUCB in streaming OPE (much faster).")
    parser.add_argument("--output-dir", default="outputs/extended_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    contextual_policies = "fixed_ab,thompson_sampling,linucb"
    ope_policies = "fixed_ab,thompson_sampling,epsilon_greedy,ucb1"
    if not args.skip_linucb_ope:
        ope_policies += ",linucb"

    # Ensure all small-OBD raw files exist (10k rows each = full small release).
    if not args.skip_download:
        for behavior, campaign in [("random", "all"), ("bts", "all"), ("random", "men"), ("random", "women")]:
            raw_path = root / f"data/raw/obd_{behavior}_{campaign}.csv"
            processed_path = root / f"data/processed/obd_{behavior}_{campaign}.csv"
            if not raw_path.exists():
                _prepare_obd(
                    root=root,
                    behavior_policy=behavior,
                    campaign=campaign,
                    output_path=processed_path,
                )
                # prepare writes processed; copy raw cache if needed
                cache = root / f"data/raw/obd_{behavior}_{campaign}.csv"
                if not cache.exists() and processed_path.exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.prepare_open_bandit",
                            "--download",
                            "--behavior-policy",
                            behavior,
                            "--campaign",
                            campaign,
                            "--output-path",
                            str(processed_path),
                        ],
                        cwd=root,
                        check=True,
                    )

    # E8b — contextual synthetic, full horizon
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

    # E8 — OBD batch, all 10k rows
    obd_random_all = root / "data/processed/obd_events.csv"
    if not obd_random_all.exists():
        _prepare_obd(
            root=root,
            behavior_policy="random",
            campaign="all",
            output_path=obd_random_all,
        )
    from src.pipeline.loader import load_events

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

    # E2b/c — in-memory OPE on full 10k per dataset (all rows, no max_events)
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
            events_path = obd_random_all if behavior == "random" and campaign == "all" else None
        if events_path is None or not events_path.exists():
            print(f"skip OPE (missing): {behavior}/{campaign}")
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

    # Streaming OPE duplicate path (validates pipeline on raw CSV, full 10k rows)
    streaming_agg = _run_streaming_ope_matrix(
        root=root,
        specs=[
            ("random", "all", "random_all"),
            ("bts", "all", "bts_all"),
            ("random", "men", "random_men"),
            ("random", "women", "random_women"),
        ],
        seeds=args.seeds_ope,
        include_linucb=not args.skip_linucb_ope,
        max_events=None,
    )

    # E1-scale synthetic baseline (non-contextual) for comparison
    e1_summary = _run_compare(
        root=root,
        mode="synthetic",
        policies="fixed_ab,epsilon_greedy,ucb1,thompson_sampling",
        events_path=obd_random_all,
        output_dir=out / "e1_synthetic_full",
        seeds=args.seeds_synthetic,
        horizon=5000,
        env="bernoulli",
    )
    e1_agg = _summarize_synthetic(e1_summary, "bernoulli_synthetic")

    latency = run_latency_matrix(n_calls=args.latency_calls)

    e8b_agg.to_csv(out / "e8b_contextual_synthetic_summary.csv", index=False)
    e8_agg.to_csv(out / "e8_obd_batch_summary.csv", index=False)
    e1_agg.to_csv(out / "e1_synthetic_summary.csv", index=False)
    if not ope_agg.empty:
        ope_agg.to_csv(out / "ope_matrix_summary.csv", index=False)
    if not streaming_agg.empty:
        streaming_agg.to_csv(out / "streaming_ope_matrix_summary.csv", index=False)
    latency.to_csv(out / "policy_latency.csv", index=False)

    meta = {
        "config": {
            "seeds_synthetic": args.seeds_synthetic,
            "seeds_obd_batch": args.seeds_obd_batch,
            "seeds_ope": args.seeds_ope,
            "horizon_contextual": args.horizon_contextual,
            "obd_rows_per_dataset": 10_000,
            "note": "Full small-OBD release (10k rows). For millions of rows, place full CSV under data/raw/full_obd/ and use obd_streaming_ope --max-events.",
        },
        "e8b_contextual_synthetic": e8b_agg.to_dict(orient="records"),
        "e8_obd_batch": e8_agg.to_dict(orient="records"),
        "e1_synthetic": e1_agg.to_dict(orient="records"),
        "ope_matrix": ope_agg.to_dict(orient="records") if not ope_agg.empty else [],
        "streaming_ope_matrix": streaming_agg.to_dict(orient="records") if not streaming_agg.empty else [],
        "latency": latency.to_dict(orient="records"),
    }
    (out / "full_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FULL E8b contextual synthetic ===")
    print(e8b_agg.sort_values("final_cumulative_regret"))
    print("\n=== FULL E8 OBD batch ===")
    print(e8_agg.sort_values("final_cumulative_regret"))
    print("\n=== FULL E1 synthetic ===")
    print(e1_agg.sort_values("final_cumulative_regret"))
    if not ope_agg.empty:
        print("\n=== FULL OPE matrix ===")
        print(ope_agg.sort_values(["dataset", "snips_estimate"], ascending=[True, False]))


if __name__ == "__main__":
    main()
