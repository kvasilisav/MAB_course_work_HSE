from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiments.obd_streaming_ope import run_streaming_obd_ope
from src.experiments.run_extended_experiments import (
    _run_compare,
    _summarize_batch,
    _summarize_ope,
    _summarize_synthetic,
)


def _agg_regret_with_std(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    grouped = (
        frame.groupby("policy_name")
        .agg(
            final_cumulative_reward=("final_cumulative_reward", "mean"),
            final_cumulative_regret=("final_cumulative_regret", "mean"),
            regret_std=("final_cumulative_regret", "std"),
            suboptimal_share=("suboptimal_share", "mean"),
        )
        .reset_index()
    )
    return grouped


def _ensure_e1(*, root: Path, seeds: int) -> pd.DataFrame:
    out_dir = root / "outputs/extended_full/e1_synthetic_full"
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        print(f"skip E1 (exists): {summary_path}")
        return _summarize_synthetic(pd.read_csv(summary_path), "bernoulli_synthetic")
    print("running E1 synthetic full...")
    summary = _run_compare(
        root=root,
        mode="synthetic",
        policies="fixed_ab,epsilon_greedy,ucb1,thompson_sampling",
        events_path=root / "data/processed/obd_events.csv",
        output_dir=out_dir,
        seeds=seeds,
        horizon=5000,
        env="bernoulli",
    )
    return _summarize_synthetic(summary, "bernoulli_synthetic")


def _ensure_streaming(
    *,
    root: Path,
    behavior: str,
    campaign: str,
    label: str,
    seeds: int,
    include_linucb: bool,
) -> pd.DataFrame | None:
    out_dir = root / f"outputs/extended_full/streaming_ope_{label}"
    summary_path = out_dir / "ope_summary.csv"
    if summary_path.exists():
        print(f"skip streaming {label} (exists)")
        return _summarize_ope(pd.read_csv(summary_path), label)

    input_path = root / f"data/raw/obd_{behavior}_{campaign}.csv"
    if not input_path.exists():
        print(f"skip streaming {label} (missing {input_path})")
        return None

    print(f"running streaming OPE: {label}, seeds={seeds}")
    summary = run_streaming_obd_ope(
        input_path=input_path,
        behavior_policy=behavior,
        campaign=campaign,
        n_arms=None,
        seeds=seeds,
        chunksize=100_000,
        max_events=None,
        include_context=include_linucb,
        include_linucb=include_linucb,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return _summarize_ope(summary, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble full_report.json from extended_full outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--run-missing", action="store_true", help="Run E1 and missing streaming OPE jobs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out = root / "outputs/extended_full"
    out.mkdir(parents=True, exist_ok=True)

    if args.run_missing:
        _ensure_e1(root=root, seeds=args.seeds)
        for behavior, campaign, label in [
            ("random", "men", "random_men"),
            ("random", "women", "random_women"),
        ]:
            _ensure_streaming(
                root=root,
                behavior=behavior,
                campaign=campaign,
                label=label,
                seeds=args.seeds,
                include_linucb=True,
            )

    e8b_path = out / "e8b_contextual_synthetic/summary.csv"
    e8_path = out / "e8_obd_batch_contextual/summary.csv"
    if not e8b_path.exists() or not e8_path.exists():
        raise FileNotFoundError("Missing E8b/E8 summaries. Run run_full_experiments first.")

    e8b_agg = _summarize_synthetic(pd.read_csv(e8b_path), "contextual_synthetic")
    e8_agg = _summarize_batch(pd.read_csv(e8_path), "obd_random_all_batch")

    e1_agg = pd.DataFrame()
    e1_path = out / "e1_synthetic_full/summary.csv"
    if e1_path.exists():
        e1_agg = _summarize_synthetic(pd.read_csv(e1_path), "bernoulli_synthetic")

    ope_frames: list[pd.DataFrame] = []
    for behavior, campaign in [
        ("random", "all"),
        ("bts", "all"),
        ("random", "men"),
        ("random", "women"),
    ]:
        p = out / f"ope_{behavior}_{campaign}" / "ope_summary.csv"
        if p.exists():
            ope_frames.append(_summarize_ope(pd.read_csv(p), f"{behavior}_{campaign}"))
    ope_agg = pd.concat(ope_frames, ignore_index=True) if ope_frames else pd.DataFrame()

    stream_frames: list[pd.DataFrame] = []
    for label in ["random_all", "bts_all", "random_men", "random_women"]:
        p = out / f"streaming_ope_{label}" / "ope_summary.csv"
        if p.exists():
            stream_frames.append(_summarize_ope(pd.read_csv(p), label))
    streaming_agg = pd.concat(stream_frames, ignore_index=True) if stream_frames else pd.DataFrame()

    latency_path = out / "policy_latency.csv"
    latency = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()

    e8b_agg.to_csv(out / "e8b_contextual_synthetic_summary.csv", index=False)
    e8_agg.to_csv(out / "e8_obd_batch_summary.csv", index=False)
    if not e1_agg.empty:
        e1_agg.to_csv(out / "e1_synthetic_summary.csv", index=False)
    if not ope_agg.empty:
        ope_agg.to_csv(out / "ope_matrix_summary.csv", index=False)
    if not streaming_agg.empty:
        streaming_agg.to_csv(out / "streaming_ope_matrix_summary.csv", index=False)

    e8b_detail = _agg_regret_with_std(e8b_path)
    e8_detail = _agg_regret_with_std(e8_path)
    e1_detail = _agg_regret_with_std(e1_path) if e1_path.exists() else pd.DataFrame()

    report = {
        "config": {
            "seeds": args.seeds,
            "horizon_contextual": 10_000,
            "horizon_e1": 5_000,
            "obd_rows_per_dataset": 10_000,
            "output_dir": "outputs/extended_full",
            "small_run_dir": "outputs/extended",
            "note": (
                "Full small-OBD release (10k rows per dataset, 20 seeds). "
                "Pilot results preserved in outputs/extended/."
            ),
        },
        "e8b_contextual_synthetic_summary": e8b_agg.to_dict(orient="records"),
        "e8b_contextual_synthetic_detail": e8b_detail.to_dict(orient="records"),
        "e8_obd_batch_summary": e8_agg.to_dict(orient="records"),
        "e8_obd_batch_detail": e8_detail.to_dict(orient="records"),
        "e1_synthetic_summary": e1_agg.to_dict(orient="records") if not e1_agg.empty else [],
        "e1_synthetic_detail": e1_detail.to_dict(orient="records") if not e1_detail.empty else [],
        "ope_matrix": ope_agg.to_dict(orient="records") if not ope_agg.empty else [],
        "streaming_ope_matrix": streaming_agg.to_dict(orient="records") if not streaming_agg.empty else [],
        "latency": latency.to_dict(orient="records") if not latency.empty else [],
    }
    report_path = out / "full_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"OPE datasets: {len(ope_frames)}, streaming: {len(stream_frames)}, E1: {e1_path.exists()}")


if __name__ == "__main__":
    main()
