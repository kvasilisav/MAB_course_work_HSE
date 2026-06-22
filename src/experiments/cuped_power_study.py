from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ab_testing.cuped import run_cuped_ab_inference, simulate_fixed_ab_with_covariate
from src.ab_testing.inference import run_ab_inference


def _evaluate_trial(
    *,
    scenario: str,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
    alpha: float,
) -> list[dict[str, float | int | str | bool]]:
    logs = simulate_fixed_ab_with_covariate(
        p_control=p_control,
        p_treatment=p_treatment,
        horizon=horizon,
        seed=seed,
    )
    naive = run_ab_inference(logs, alpha=alpha)
    cuped = run_cuped_ab_inference(logs, alpha=alpha)
    rows: list[dict[str, float | int | str | bool]] = []
    for method, result in (("naive_ab", naive), ("cuped_ab", cuped)):
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "p_control": p_control,
                "p_treatment": p_treatment,
                "reject_null": bool(result.reject_null),
                "p_value": float(result.p_value),
                "ate": float(result.ate),
                "standard_error": float(result.standard_error),
                "seed": seed,
            }
        )
    if cuped.theta != 0.0:
        rows[-1]["theta"] = float(cuped.theta)
    return rows


def run_cuped_power_study(
    *,
    horizon: int,
    n_trials: int,
    alpha: float,
    null_ctr: float,
    effect_ctrs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, float | int | str | bool]] = []
    scenarios: list[tuple[str, float, float]] = [("null", null_ctr, null_ctr)]
    scenarios.extend(
        (f"effect_{p_treatment:.3f}", null_ctr, p_treatment) for p_treatment in effect_ctrs
    )

    for scenario, p_control, p_treatment in scenarios:
        for seed in tqdm(
            range(n_trials),
            desc=f"CUPED {scenario} (p={p_control:.2f}/{p_treatment:.2f})",
            unit="trial",
        ):
            detail_rows.extend(
                _evaluate_trial(
                    scenario=scenario,
                    p_control=p_control,
                    p_treatment=p_treatment,
                    horizon=horizon,
                    seed=seed,
                    alpha=alpha,
                )
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["scenario", "method"], as_index=False)
        .agg(
            rejection_rate=("reject_null", "mean"),
            mean_p_value=("p_value", "mean"),
            mean_se=("standard_error", "mean"),
            trials=("reject_null", "count"),
            p_control=("p_control", "first"),
            p_treatment=("p_treatment", "first"),
        )
        .sort_values(["scenario", "method"])
        .reset_index(drop=True)
    )
    summary["target_rate"] = summary["scenario"].map({"null": alpha}).fillna(0.8)
    return summary, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E15: CUPED vs naive A/B on fixed split with correlated pre-period covariate.",
    )
    parser.add_argument("--horizon", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-ctr", type=float, default=0.05)
    parser.add_argument(
        "--effect-ctrs",
        default="0.06,0.08",
        help="Comma-separated treatment CTRs for effect scenarios.",
    )
    parser.add_argument("--output-dir", default="outputs/cuped_power")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effect_ctrs = [float(x.strip()) for x in args.effect_ctrs.split(",") if x.strip()]
    summary, detail = run_cuped_power_study(
        horizon=args.horizon,
        n_trials=args.trials,
        alpha=args.alpha,
        null_ctr=args.null_ctr,
        effect_ctrs=effect_ctrs,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "e15_summary.csv", index=False)
    detail.to_csv(out_dir / "e15_detail.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
