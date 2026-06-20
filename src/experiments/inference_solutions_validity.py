from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ab_testing.bandit_logs import logs_to_ab_frame, simulate_thompson_logs_with_propensity
from src.ab_testing.inference import run_ab_inference
from src.ab_testing.weighted_inference import run_weighted_ab_inference
from src.bandits.fixed_ab import FixedABPolicy
from src.environments.bernoulli import BernoulliBanditEnv


def _simulate_fixed_ab_logs(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    env = BernoulliBanditEnv([p_control, p_treatment], horizon=horizon, seed=seed)
    policy = FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=seed + 17)
    env.reset()
    policy.reset()
    rows: list[dict[str, float | int | str]] = []
    for step in range(horizon):
        arm = int(policy.select_arm())
        reward, _ = env.step(arm)
        rows.append(
            {
                "step": step,
                "arm": arm,
                "reward": float(reward),
                "group": "treatment" if arm == 1 else "control",
            }
        )
        policy.update(arm, reward)
    return pd.DataFrame(rows)


def _evaluate_trial(
    *,
    scenario: str,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
    alpha: float,
    propensity_mc_samples: int,
    n_bootstrap: int,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []

    ts_logs = simulate_thompson_logs_with_propensity(
        p_control=p_control,
        p_treatment=p_treatment,
        horizon=horizon,
        seed=seed,
        propensity_mc_samples=propensity_mc_samples,
    )
    fixed_logs = _simulate_fixed_ab_logs(
        p_control=p_control,
        p_treatment=p_treatment,
        horizon=horizon,
        seed=seed + 200_003,
    )

    evaluations: list[tuple[str, object]] = []

    try:
        evaluations.append(
            (
                "naive_ab_full_ts",
                run_ab_inference(logs_to_ab_frame(ts_logs), alpha=alpha),
            )
        )
    except ValueError:
        pass

    try:
        evaluations.append(
            (
                "naive_ab_fixed_only",
                run_ab_inference(logs_to_ab_frame(fixed_logs), alpha=alpha),
            )
        )
    except ValueError:
        pass

    try:
        weighted = run_weighted_ab_inference(
            ts_logs,
            alpha=alpha,
            bootstrap_seed=seed,
            n_bootstrap=n_bootstrap,
        )
        evaluations.append(("ips_weighted_ab", weighted))
    except ValueError:
        pass

    for method, result in evaluations:
        row: dict[str, float | int | str | bool] = {
            "scenario": scenario,
            "method": method,
            "p_control": p_control,
            "p_treatment": p_treatment,
            "reject_null": bool(result.reject_null),
            "p_value": float(result.p_value),
            "seed": seed,
        }
        if method == "ips_weighted_ab":
            row["ate"] = float(result.ate_ips)
            row["ess"] = float(result.effective_sample_size)
            row["n_observations"] = int(result.n_observations)
        else:
            row["ate"] = float(result.ate)
            row["n_control"] = int(result.n_control)
            row["n_treatment"] = int(result.n_treatment)
        rows.append(row)

    return rows


def run_inference_solutions_study(
    *,
    horizon: int,
    n_trials: int,
    alpha: float,
    null_ctr: float,
    effect_ctrs: list[float],
    propensity_mc_samples: int = 100,
    n_bootstrap: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, float | int | str | bool]] = []
    scenarios: list[tuple[str, float, float]] = [("null", null_ctr, null_ctr)]
    scenarios.extend(
        (f"effect_{p_treatment:.3f}", null_ctr, p_treatment) for p_treatment in effect_ctrs
    )

    for scenario, p_control, p_treatment in scenarios:
        for seed in tqdm(
            range(n_trials),
            desc=f"E11 {scenario} (p={p_control:.2f}/{p_treatment:.2f})",
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
                    propensity_mc_samples=propensity_mc_samples,
                    n_bootstrap=n_bootstrap,
                )
            )
        scenario_summary = (
            pd.DataFrame(detail_rows)
            .query("scenario == @scenario")
            .groupby("method")["reject_null"]
            .mean()
        )
        tqdm.write(f"  interim {scenario}: {scenario_summary.to_dict()}")

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["scenario", "method"], as_index=False)
        .agg(
            rejection_rate=("reject_null", "mean"),
            mean_p_value=("p_value", "mean"),
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
        description="E11: IPS-weighted inference vs naive A/B on bandit logs.",
    )
    parser.add_argument("--horizon", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-ctr", type=float, default=0.05)
    parser.add_argument(
        "--effect-ctrs",
        default="0.06,0.08,0.10",
        help="Comma-separated treatment CTRs for effect scenarios (control = --null-ctr).",
    )
    parser.add_argument("--propensity-mc", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap draws for IPS SE; 0 = analytical.")
    parser.add_argument("--output-dir", default="outputs/inference_valid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"E11 inference solutions: horizon={args.horizon}, "
        f"trials={args.trials}, alpha={args.alpha}, propensity_mc={args.propensity_mc}"
    )
    effect_ctrs = [float(x.strip()) for x in args.effect_ctrs.split(",") if x.strip()]
    summary, detail = run_inference_solutions_study(
        horizon=args.horizon,
        n_trials=args.trials,
        alpha=args.alpha,
        null_ctr=args.null_ctr,
        effect_ctrs=effect_ctrs,
        propensity_mc_samples=args.propensity_mc,
        n_bootstrap=args.bootstrap,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "e11_summary.csv", index=False)
    detail.to_csv(out_dir / "e11_detail.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
