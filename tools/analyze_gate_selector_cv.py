import argparse
import json
from pathlib import Path

from analyze_gate_selector import SEEDS, best_thresholds, fmt, load_rows, mean, summarize


DEPLOYABLE_FEATURES = [
    "old_s_q_mean",
    "old_raw_gate_mean",
    "old_strength_mean",
    "old_delta_rms",
    "old_latent_quant_mse",
    "old_y_error_rms",
    "risk_s_q_mean",
    "risk_raw_gate_mean",
    "risk_strength_mean",
    "risk_delta_rms",
    "risk_latent_quant_mse",
    "risk_y_error_rms",
    "risk_multiplier_mean",
]


def choose_policy(train_rows, feature_names):
    candidates = []
    for feature in feature_names:
        ranked = best_thresholds(train_rows, feature)
        if ranked:
            candidates.append(ranked[0])
    candidates.sort(key=lambda row: row["mean_delta"])
    return candidates[0]


def policy_delta(row, feature, threshold, direction):
    use_risk = row[feature] <= threshold if direction == "le" else row[feature] >= threshold
    return row["risk_delta_rd"] if use_risk else row["old_delta_rd"], use_risk


def evaluate_policy(rows, feature, threshold, direction):
    deltas = []
    risk_flags = []
    for row in rows:
        delta, use_risk = policy_delta(row, feature, threshold, direction)
        deltas.append(delta)
        risk_flags.append(1.0 if use_risk else 0.0)
    result = {
        "mean_delta": mean(deltas),
        "risk_fraction": mean(risk_flags),
        "hcs_win_rate": mean(1.0 if delta < 0.0 else 0.0 for delta in deltas),
    }
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if seed_rows:
            seed_deltas = [policy_delta(row, feature, threshold, direction)[0] for row in seed_rows]
            result[f"seed{seed}_delta"] = mean(seed_deltas)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--protocol", choices=["legacy", "current_holdout"], default="current_holdout")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output-prefix", default="gate025_min090_selector_cv_val4096_holdout4096_current")
    parser.add_argument("--fixed-feature", default="old_raw_gate_mean")
    parser.add_argument("--fixed-direction", choices=["le", "ge"], default="ge")
    parser.add_argument("--fixed-threshold", type=float, default=0.260788)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    rows = load_rows(analysis_dir, args.protocol)
    if args.folds < 2:
        raise ValueError("folds must be at least 2")

    fold_results = []
    cv_deltas = []
    cv_risk_flags = []
    fixed_deltas = []
    fixed_risk_flags = []

    for fold in range(args.folds):
        train_rows = [row for row in rows if row["index"] % args.folds != fold]
        test_rows = [row for row in rows if row["index"] % args.folds == fold]
        selected = choose_policy(train_rows, DEPLOYABLE_FEATURES)
        selected_feature = selected["feature"]
        selected_direction = selected["direction"]
        selected_threshold = float(selected["threshold"])
        selected_test = evaluate_policy(test_rows, selected_feature, selected_threshold, selected_direction)
        fixed_test = evaluate_policy(test_rows, args.fixed_feature, args.fixed_threshold, args.fixed_direction)
        for row in test_rows:
            delta, use_risk = policy_delta(row, selected_feature, selected_threshold, selected_direction)
            cv_deltas.append(delta)
            cv_risk_flags.append(1.0 if use_risk else 0.0)
            fixed_delta, fixed_use_risk = policy_delta(row, args.fixed_feature, args.fixed_threshold, args.fixed_direction)
            fixed_deltas.append(fixed_delta)
            fixed_risk_flags.append(1.0 if fixed_use_risk else 0.0)
        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "selected_feature": selected_feature,
                "selected_direction": selected_direction,
                "selected_threshold": selected_threshold,
                "selected_train_delta": float(selected["mean_delta"]),
                "selected_test": selected_test,
                "fixed_test": fixed_test,
            }
        )

    baseline = summarize(rows)
    cv_summary = {
        "mean_delta": mean(cv_deltas),
        "vs_old_gate025": mean(cv_deltas) - baseline["old_gate_mean_delta"],
        "risk_fraction": mean(cv_risk_flags),
    }
    fixed_summary = {
        "feature": args.fixed_feature,
        "direction": args.fixed_direction,
        "threshold": args.fixed_threshold,
        "mean_delta": mean(fixed_deltas),
        "vs_old_gate025": mean(fixed_deltas) - baseline["old_gate_mean_delta"],
        "risk_fraction": mean(fixed_risk_flags),
    }
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        cv_seed = []
        fixed_seed = []
        for row in seed_rows:
            fold = row["index"] % args.folds
            selected = fold_results[fold]
            delta, _ = policy_delta(
                row,
                selected["selected_feature"],
                selected["selected_threshold"],
                selected["selected_direction"],
            )
            cv_seed.append(delta)
            fixed_delta, _ = policy_delta(row, args.fixed_feature, args.fixed_threshold, args.fixed_direction)
            fixed_seed.append(fixed_delta)
        cv_summary[f"seed{seed}_delta"] = mean(cv_seed)
        fixed_summary[f"seed{seed}_delta"] = mean(fixed_seed)

    payload = {
        "protocol": args.protocol,
        "folds": args.folds,
        "baseline": baseline,
        "fold_results": fold_results,
        "cross_validated_selected_policy": cv_summary,
        "fixed_policy": fixed_summary,
    }

    json_path = analysis_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    old_delta = baseline["old_gate_mean_delta"]
    min090_delta = baseline["min090_mean_delta"]
    oracle_delta = baseline["oracle_old_or_min090_delta"]
    oracle_frac = baseline["min090_beats_old_rate"]
    cv_delta = cv_summary["mean_delta"]
    fixed_delta = fixed_summary["mean_delta"]
    cv_vs_old = cv_summary["vs_old_gate025"]
    cv_fraction = cv_summary["risk_fraction"]
    cv_seed1234 = cv_summary["seed1234_delta"]
    cv_seed2345 = cv_summary["seed2345_delta"]
    cv_seed3456 = cv_summary["seed3456_delta"]
    fixed_vs_old = fixed_summary["vs_old_gate025"]
    fixed_fraction = fixed_summary["risk_fraction"]
    fixed_seed1234 = fixed_summary["seed1234_delta"]
    fixed_seed2345 = fixed_summary["seed2345_delta"]
    fixed_seed3456 = fixed_summary["seed3456_delta"]
    lines = [
        "# Gate0.25 vs min090 selector cross-validation",
        "",
        "The threshold is calibrated on train folds and evaluated on held-out image-index folds. Deltas are method minus HCS RD; lower is better.",
        "",
        "## Overall",
        "",
        "| policy | mean delta RD | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| old gate0.25 | {fmt(old_delta, True)} | {fmt(0.0, True)} | 0.000000 | n/a | n/a | n/a |",
        f"| min090 risk | {fmt(min090_delta, True)} | {fmt(min090_delta - old_delta, True)} | 1.000000 | n/a | n/a | n/a |",
        f"| oracle old/min090 | {fmt(oracle_delta, True)} | {fmt(oracle_delta - old_delta, True)} | {fmt(oracle_frac)} | n/a | n/a | n/a |",
        f"| CV selected deployable threshold | {fmt(cv_delta, True)} | {fmt(cv_vs_old, True)} | {fmt(cv_fraction)} | {fmt(cv_seed1234, True)} | {fmt(cv_seed2345, True)} | {fmt(cv_seed3456, True)} |",
        f"| fixed old_raw_gate_mean >= 0.260788 | {fmt(fixed_delta, True)} | {fmt(fixed_vs_old, True)} | {fmt(fixed_fraction)} | {fmt(fixed_seed1234, True)} | {fmt(fixed_seed2345, True)} | {fmt(fixed_seed3456, True)} |",
        "",
        "## Fold Details",
        "",
        "| fold | selected feature | dir | threshold | train delta | test delta | test min090 fraction | fixed test delta |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fold_results:
        fold = row["fold"]
        feature = row["selected_feature"]
        direction = row["selected_direction"]
        threshold = row["selected_threshold"]
        train_delta = row["selected_train_delta"]
        test_delta = row["selected_test"]["mean_delta"]
        test_fraction = row["selected_test"]["risk_fraction"]
        fixed_test_delta = row["fixed_test"]["mean_delta"]
        lines.append(
            f"| {fold} | {feature} | {direction} | {threshold:.6f} | {fmt(train_delta, True)} | {fmt(test_delta, True)} | {fmt(test_fraction)} | {fmt(fixed_test_delta, True)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Cross-validation is a guard against over-reading a threshold chosen on the same holdout images. If the CV selected policy remains better than old gate0.25 and close to the full-data threshold, the selector direction is more credible. If it collapses, the threshold was likely overfit and the next method should use stronger calibration or a frozen-evidence classifier.",
        ]
    )
    md_path = analysis_dir / f"{args.output_prefix}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
