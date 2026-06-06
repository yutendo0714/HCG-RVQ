import argparse
import csv
import json
from pathlib import Path

from analyze_gate_selector import (
    SEEDS,
    best_thresholds,
    corr,
    fmt,
    mean,
    read_csv,
    selector_metrics,
    summarize,
    value,
)
from analyze_gate_selector_cv import DEPLOYABLE_FEATURES, choose_policy, evaluate_policy, policy_delta


def step_plan(seed: int, mode: str) -> tuple[int, int, int]:
    hcs_step = 500 if seed == 1234 else 250
    if mode == "transfer":
        old_step = 500 if seed == 3456 else 250
        risk_step = 500 if seed in {1234, 3456} else 250
    elif mode == "slice_best":
        old_step = 250
        risk_step = 500 if seed == 1234 else 250
    else:
        raise ValueError(f"unknown mode: {mode}")
    return hcs_step, old_step, risk_step


def existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def paths_for(analysis_dir: Path, seed: int, mode: str) -> dict[str, Path]:
    hcs_step, old_step, risk_step = step_plan(seed, mode)
    return {
        "hcs_features": analysis_dir / f"per_image_features_hcs_seed{seed}_step{hcs_step}_val4096_start0_current_recheck.csv",
        "old_features": existing_path(
            analysis_dir / f"per_image_features_hcg_h_gate025_seed{seed}_step{old_step}_val4096_start0_current_recheck.csv",
            analysis_dir / f"per_image_features_hcg_h_gate025_seed{seed}_step{old_step}_val4096_reeval_current.csv",
        ),
        "risk_features": existing_path(
            analysis_dir
            / f"per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed{seed}_step{risk_step}_val4096_start0_current_recheck.csv",
            analysis_dir
            / f"per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed{seed}_step{risk_step}_val4096_reeval_current.csv",
        ),
    }


def feature(row: dict[str, str], key: str) -> float:
    return value(row, key)


def load_rows(analysis_dir: Path, mode: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        paths = paths_for(analysis_dir, seed, mode)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("\n".join(missing))
        hcs_rows = read_csv(paths["hcs_features"])
        old_rows = read_csv(paths["old_features"])
        risk_rows = read_csv(paths["risk_features"])
        old_by_path = {row["path"]: row for row in old_rows}
        risk_by_path = {row["path"]: row for row in risk_rows}
        for hcs in hcs_rows:
            path = hcs["path"]
            old = old_by_path[path]
            risk = risk_by_path[path]
            hcs_rd = feature(hcs, "rd_score")
            old_rd = feature(old, "rd_score")
            risk_rd = feature(risk, "rd_score")
            rows.append(
                {
                    "seed": seed,
                    "index": int(hcs["index"]),
                    "path": path,
                    "hcs_rd": hcs_rd,
                    "old_rd": old_rd,
                    "risk_rd": risk_rd,
                    "old_delta_rd": old_rd - hcs_rd,
                    "risk_delta_rd": risk_rd - hcs_rd,
                    "risk_minus_old_delta_rd": risk_rd - old_rd,
                    "hcs_s_q_mean": feature(hcs, "s_q_mean"),
                    "hcs_latent_quant_mse": feature(hcs, "rvq_latent_quant_mse"),
                    "hcs_y_error_rms": feature(hcs, "y_error_rms"),
                    "old_s_q_mean": feature(old, "s_q_mean"),
                    "old_raw_gate_mean": feature(old, "householder_gate_raw_mean"),
                    "old_strength_mean": feature(old, "householder_strength_mean"),
                    "old_delta_rms": feature(old, "householder_delta_rms"),
                    "old_latent_quant_mse": feature(old, "rvq_latent_quant_mse"),
                    "old_y_error_rms": feature(old, "y_error_rms"),
                    "risk_s_q_mean": feature(risk, "s_q_mean"),
                    "risk_raw_gate_mean": feature(risk, "householder_gate_raw_mean"),
                    "risk_strength_mean": feature(risk, "householder_strength_mean"),
                    "risk_delta_rms": feature(risk, "householder_delta_rms"),
                    "risk_latent_quant_mse": feature(risk, "rvq_latent_quant_mse"),
                    "risk_y_error_rms": feature(risk, "y_error_rms"),
                    "risk_multiplier_mean": feature(risk, "householder_risk_multiplier_mean"),
                }
            )
    return rows


def absolute_summary(rows: list[dict[str, float | int | str]], deltas: list[float] | None = None) -> dict[str, float | None]:
    if deltas is None:
        selected_mean = None
        selected_delta = None
    else:
        selected_delta = mean(deltas)
        selected_mean = mean(float(row["hcs_rd"]) + delta for row, delta in zip(rows, deltas))
    return {
        "hcs_mean_rd": mean(float(row["hcs_rd"]) for row in rows),
        "old_mean_rd": mean(float(row["old_rd"]) for row in rows),
        "risk_mean_rd": mean(float(row["risk_rd"]) for row in rows),
        "oracle_old_or_min090_mean_rd": mean(min(float(row["old_rd"]), float(row["risk_rd"])) for row in rows),
        "oracle_hcs_or_old_or_min090_mean_rd": mean(
            min(float(row["hcs_rd"]), float(row["old_rd"]), float(row["risk_rd"])) for row in rows
        ),
        "selected_mean_rd": selected_mean,
        "selected_delta_rd": selected_delta,
    }


def fixed_policy_deltas(rows: list[dict[str, float | int | str]], feature: str, threshold: float, direction: str) -> tuple[list[float], list[float]]:
    deltas = []
    flags = []
    for row in rows:
        delta, use_risk = policy_delta(row, feature, threshold, direction)
        deltas.append(delta)
        flags.append(1.0 if use_risk else 0.0)
    return deltas, flags


def cross_validated(rows: list[dict[str, float | int | str]], folds: int) -> tuple[dict[str, float], list[dict[str, object]], list[float], list[float]]:
    fold_results = []
    cv_deltas = []
    cv_flags = []
    for fold in range(folds):
        train_rows = [row for row in rows if int(row["index"]) % folds != fold]
        test_rows = [row for row in rows if int(row["index"]) % folds == fold]
        selected = choose_policy(train_rows, DEPLOYABLE_FEATURES)
        result = evaluate_policy(test_rows, selected["feature"], float(selected["threshold"]), selected["direction"])
        for row in test_rows:
            delta, use_risk = policy_delta(row, selected["feature"], float(selected["threshold"]), selected["direction"])
            cv_deltas.append(delta)
            cv_flags.append(1.0 if use_risk else 0.0)
        fold_results.append(
            {
                "fold": fold,
                "selected_feature": selected["feature"],
                "selected_direction": selected["direction"],
                "selected_threshold": float(selected["threshold"]),
                "selected_train_delta": float(selected["mean_delta"]),
                "selected_test": result,
            }
        )
    summary = {"mean_delta": mean(cv_deltas), "risk_fraction": mean(cv_flags)}
    for seed in SEEDS:
        seed_deltas = []
        for row in rows:
            if int(row["seed"]) != seed:
                continue
            fold = int(row["index"]) % folds
            selected = fold_results[fold]
            delta, _ = policy_delta(
                row,
                str(selected["selected_feature"]),
                float(selected["selected_threshold"]),
                str(selected["selected_direction"]),
            )
            seed_deltas.append(delta)
        summary[f"seed{seed}_delta"] = mean(seed_deltas)
    return summary, fold_results, cv_deltas, cv_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--mode", choices=["transfer", "slice_best"], default="transfer")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fixed-feature", default="old_raw_gate_mean")
    parser.add_argument("--fixed-direction", choices=["le", "ge"], default="ge")
    parser.add_argument("--fixed-threshold", type=float, default=0.260788)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_prefix = args.output_prefix or f"gate025_min090_selector_start0_current_recheck_{args.mode}"
    rows = load_rows(analysis_dir, args.mode)

    baseline = summarize(rows)
    per_seed = {str(seed): summarize([row for row in rows if int(row["seed"]) == seed]) for seed in SEEDS}
    absolute = absolute_summary(rows)

    feature_names = ["hcs_rd", "hcs_s_q_mean", "hcs_latent_quant_mse", *DEPLOYABLE_FEATURES]
    all_thresholds = []
    best_by_feature = []
    for name in feature_names:
        ranked = best_thresholds(rows, name)
        all_thresholds.extend(ranked)
        best_by_feature.append(ranked[0])
    all_thresholds.sort(key=lambda row: row["mean_delta"])
    best_by_feature.sort(key=lambda row: row["mean_delta"])

    fixed_deltas, fixed_flags = fixed_policy_deltas(rows, args.fixed_feature, args.fixed_threshold, args.fixed_direction)
    fixed_summary = {
        "feature": args.fixed_feature,
        "direction": args.fixed_direction,
        "threshold": args.fixed_threshold,
        "mean_delta": mean(fixed_deltas),
        "vs_old_gate025": mean(fixed_deltas) - baseline["old_gate_mean_delta"],
        "risk_fraction": mean(fixed_flags),
    }
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_deltas, seed_flags = fixed_policy_deltas(seed_rows, args.fixed_feature, args.fixed_threshold, args.fixed_direction)
        fixed_summary[f"seed{seed}_delta"] = mean(seed_deltas)
        fixed_summary[f"seed{seed}_risk_fraction"] = mean(seed_flags)

    cv_summary, fold_results, cv_deltas, cv_flags = cross_validated(rows, args.folds)
    cv_summary["vs_old_gate025"] = cv_summary["mean_delta"] - baseline["old_gate_mean_delta"]
    cv_summary["risk_fraction"] = mean(cv_flags)

    correlations = {
        name: {
            "corr_hcs_rd": corr([float(row["hcs_rd"]) for row in rows], [float(row[name]) for row in rows]),
            "corr_min090_minus_old": corr(
                [float(row["risk_minus_old_delta_rd"]) for row in rows],
                [float(row[name]) for row in rows],
            ),
        }
        for name in feature_names
    }

    selected_absolute = absolute_summary(rows, fixed_deltas)
    cv_absolute = absolute_summary(rows, cv_deltas)

    payload = {
        "mode": args.mode,
        "step_plan": {str(seed): dict(zip(("hcs_step", "old_step", "risk_step"), step_plan(seed, args.mode))) for seed in SEEDS},
        "baseline": baseline,
        "per_seed_summary": per_seed,
        "absolute": absolute,
        "fixed_policy": fixed_summary,
        "fixed_policy_absolute": selected_absolute,
        "cross_validated_policy": cv_summary,
        "cross_validated_absolute": cv_absolute,
        "fold_results": fold_results,
        "best_by_feature": best_by_feature,
        "best_overall": all_thresholds[:20],
        "correlations": correlations,
    }

    json_path = analysis_dir / f"{output_prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    thresholds_path = analysis_dir / f"{output_prefix}_thresholds.csv"
    with thresholds_path.open("w", newline="") as handle:
        fields = [
            "feature",
            "direction",
            "threshold",
            "mean_delta",
            "risk_fraction",
            "hcs_win_rate",
            "seed1234_delta",
            "seed2345_delta",
            "seed3456_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_thresholds)

    old_delta = baseline["old_gate_mean_delta"]
    risk_delta = baseline["min090_mean_delta"]
    oracle_delta = baseline["oracle_old_or_min090_delta"]
    fixed_delta = fixed_summary["mean_delta"]
    cv_delta = cv_summary["mean_delta"]
    lines = [
        "# Gate0.25 vs min090 selector start0 current recheck",
        "",
        f"Mode: `{args.mode}`. Deltas are method minus HCS RD on OpenImages start_index=0, max_images=4096; lower is better.",
        "",
        "## Step Plan",
        "",
        "| seed | HCS step | old gate0.25 step | min090 risk step |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        hcs_step, old_step, risk_step = step_plan(seed, args.mode)
        lines.append(f"| {seed} | {hcs_step} | {old_step} | {risk_step} |")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| policy | mean RD | mean delta RD | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| HCS | {fmt(absolute['hcs_mean_rd'])} | {fmt(0.0, True)} | n/a | 0.000000 | n/a | n/a | n/a |",
            f"| old gate0.25 | {fmt(absolute['old_mean_rd'])} | {fmt(old_delta, True)} | {fmt(0.0, True)} | 0.000000 | "
            f"{fmt(per_seed['1234']['old_gate_mean_delta'], True)} | {fmt(per_seed['2345']['old_gate_mean_delta'], True)} | {fmt(per_seed['3456']['old_gate_mean_delta'], True)} |",
            f"| min090 risk | {fmt(absolute['risk_mean_rd'])} | {fmt(risk_delta, True)} | {fmt(risk_delta - old_delta, True)} | 1.000000 | "
            f"{fmt(per_seed['1234']['min090_mean_delta'], True)} | {fmt(per_seed['2345']['min090_mean_delta'], True)} | {fmt(per_seed['3456']['min090_mean_delta'], True)} |",
            f"| oracle old/min090 | {fmt(absolute['oracle_old_or_min090_mean_rd'])} | {fmt(oracle_delta, True)} | {fmt(oracle_delta - old_delta, True)} | {fmt(baseline['min090_beats_old_rate'])} | n/a | n/a | n/a |",
            f"| fixed old_raw_gate_mean >= {args.fixed_threshold:.6f} | {fmt(selected_absolute['selected_mean_rd'])} | {fmt(fixed_delta, True)} | {fmt(fixed_summary['vs_old_gate025'], True)} | {fmt(fixed_summary['risk_fraction'])} | "
            f"{fmt(fixed_summary['seed1234_delta'], True)} | {fmt(fixed_summary['seed2345_delta'], True)} | {fmt(fixed_summary['seed3456_delta'], True)} |",
            f"| CV selected deployable threshold | {fmt(cv_absolute['selected_mean_rd'])} | {fmt(cv_delta, True)} | {fmt(cv_summary['vs_old_gate025'], True)} | {fmt(cv_summary['risk_fraction'])} | "
            f"{fmt(cv_summary['seed1234_delta'], True)} | {fmt(cv_summary['seed2345_delta'], True)} | {fmt(cv_summary['seed3456_delta'], True)} |",
            "",
            "## Best Single-Feature Thresholds",
            "",
            "| feature | direction | threshold | mean delta RD | vs old gate0.25 | min090 fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_feature[:12]:
        lines.append(
            f"| {row['feature']} | {row['direction']} | {fmt(float(row['threshold']))} | "
            f"{fmt(float(row['mean_delta']), True)} | {fmt(float(row['mean_delta']) - old_delta, True)} | {fmt(float(row['risk_fraction']))} |"
        )
    lines.extend(
        [
            "",
            "## Fold Details",
            "",
            "| fold | selected feature | dir | threshold | train delta | test delta | test min090 fraction |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in fold_results:
        lines.append(
            f"| {row['fold']} | {row['selected_feature']} | {row['selected_direction']} | {float(row['selected_threshold']):.6f} | "
            f"{fmt(float(row['selected_train_delta']), True)} | {fmt(float(row['selected_test']['mean_delta']), True)} | {fmt(float(row['selected_test']['risk_fraction']))} |"
        )
    lines.extend(
        [
            "",
            "## Feature Correlations",
            "",
            "| feature | corr(feature, HCS RD) | corr(feature, min090-old delta) |",
            "|---|---:|---:|",
        ]
    )
    for feature_name, vals in sorted(correlations.items(), key=lambda item: abs(item[1]["corr_min090_minus_old"]), reverse=True):
        lines.append(
            f"| {feature_name} | {fmt(vals['corr_hcs_rd'], True)} | {fmt(vals['corr_min090_minus_old'], True)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This recheck uses the current code path and regenerates per-image metrics from checkpoint files. It is meant to catch stale start0 CSVs and to test whether a selector calibrated on the paper-facing holdout slice transfers to a different image slice.",
        ]
    )
    md_path = analysis_dir / f"{output_prefix}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(md_path)
    print(json_path)
    print(thresholds_path)


if __name__ == "__main__":
    main()
