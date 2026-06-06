import argparse
import csv
import json
import math
from pathlib import Path


SEEDS = (1234, 2345, 3456)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row, key):
    raw = row.get(key, "")
    return float(raw) if raw != "" else float("nan")


def prefixed_value(row, key, prefixes):
    for prefix in prefixes:
        value_key = f"{prefix}_{key}"
        if value_key in row:
            return value(row, value_key)
    return value(row, key)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return float("nan")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(x_var * y_var)


def fmt(number, signed=False):
    if not math.isfinite(number):
        return "nan"
    return ("{:+.6f}" if signed else "{:.6f}").format(number)


def percentile_thresholds(values, bins=101):
    values = sorted(v for v in values if math.isfinite(v))
    thresholds = []
    if not values:
        return thresholds
    for index in range(1, bins):
        pos = index * (len(values) - 1) / bins
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            thresholds.append(values[lo])
        else:
            alpha = pos - lo
            thresholds.append(values[lo] * (1.0 - alpha) + values[hi] * alpha)
    return sorted(set(thresholds))


def summarize(rows):
    return {
        "n": len(rows),
        "old_gate_mean_delta": mean(row["old_delta_rd"] for row in rows),
        "min090_mean_delta": mean(row["risk_delta_rd"] for row in rows),
        "oracle_old_or_min090_delta": mean(
            min(row["old_delta_rd"], row["risk_delta_rd"]) for row in rows
        ),
        "oracle_hcs_or_old_or_min090_delta": mean(
            min(0.0, row["old_delta_rd"], row["risk_delta_rd"]) for row in rows
        ),
        "min090_beats_old_rate": mean(
            1.0 if row["risk_delta_rd"] < row["old_delta_rd"] else 0.0 for row in rows
        ),
        "old_win_rate_vs_hcs": mean(1.0 if row["old_delta_rd"] < 0.0 else 0.0 for row in rows),
        "min090_win_rate_vs_hcs": mean(1.0 if row["risk_delta_rd"] < 0.0 else 0.0 for row in rows),
    }


def selector_metrics(rows, feature, threshold, direction):
    deltas = []
    risk_flags = []
    for row in rows:
        use_risk = row[feature] <= threshold if direction == "le" else row[feature] >= threshold
        risk_flags.append(1.0 if use_risk else 0.0)
        deltas.append(row["risk_delta_rd"] if use_risk else row["old_delta_rd"])
    return {
        "mean_delta": mean(deltas),
        "risk_fraction": mean(risk_flags),
        "hcs_win_rate": mean(1.0 if delta < 0.0 else 0.0 for delta in deltas),
    }


def best_thresholds(rows, feature):
    results = []
    for threshold in percentile_thresholds([row[feature] for row in rows]):
        for direction in ("le", "ge"):
            result = {
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                **selector_metrics(rows, feature, threshold, direction),
            }
            for seed in SEEDS:
                subset = [row for row in rows if row["seed"] == seed]
                result[f"seed{seed}_delta"] = selector_metrics(
                    subset, feature, threshold, direction
                )["mean_delta"]
            results.append(result)
    results.sort(key=lambda row: row["mean_delta"])
    return results


def current_holdout_paths(analysis_dir, seed):
    hcs_step = 500 if seed == 1234 else 250
    old_step = 500 if seed == 3456 else 250
    risk_step = 500 if seed in {1234, 3456} else 250
    return {
        "old_rows": analysis_dir / f"per_image_seed{seed}_hcs{hcs_step}_vs_hcgh_gate025_step{old_step}_val4096_holdout4096_current.csv",
        "risk_rows": analysis_dir / f"per_image_seed{seed}_hcs{hcs_step}_vs_hcgh_gate025_risk_inv_detach_s044_min090_step{risk_step}_val4096_holdout4096_current.csv",
        "old_features": analysis_dir / f"per_image_features_hcg_h_gate025_seed{seed}_step{old_step}_val4096_holdout4096_current.csv",
        "risk_features": analysis_dir / f"per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed{seed}_step{risk_step}_val4096_holdout4096_current.csv",
    }


def legacy_paths(analysis_dir, seed):
    risk_step = 500 if seed == 1234 else 250
    return {
        "old_rows": analysis_dir / f"per_image_seed{seed}_hcsbest_vs_hcgh_gate025_best_val4096.csv",
        "risk_rows": analysis_dir / f"per_image_seed{seed}_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min090_best_val4096.csv",
        "old_features": analysis_dir / f"per_image_features_hcg_h_gate025_seed{seed}_step250_val4096.csv",
        "risk_features": analysis_dir / f"per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed{seed}_step{risk_step}_val4096.csv",
    }


def load_rows(analysis_dir, protocol):
    rows = []
    path_fn = current_holdout_paths if protocol == "current_holdout" else legacy_paths
    for seed in SEEDS:
        paths = path_fn(analysis_dir, seed)
        old_rows = read_csv(paths["old_rows"])
        risk_rows = read_csv(paths["risk_rows"])
        old_features = read_csv(paths["old_features"])
        risk_features = read_csv(paths["risk_features"])
        risk_by_path = {row["path"]: row for row in risk_rows}
        old_feat_by_path = {row["path"]: row for row in old_features}
        risk_feat_by_path = {row["path"]: row for row in risk_features}
        for old in old_rows:
            path = old["path"]
            risk = risk_by_path[path]
            old_feat = old_feat_by_path[path]
            risk_feat = risk_feat_by_path[path]
            rows.append(
                {
                    "seed": seed,
                    "index": int(old["index"]),
                    "path": path,
                    "hcs_rd": prefixed_value(old, "rd_score", ["HCS", "hcs"]),
                    "old_delta_rd": value(old, "delta_rd_score"),
                    "risk_delta_rd": value(risk, "delta_rd_score"),
                    "risk_minus_old_delta_rd": value(risk, "delta_rd_score")
                    - value(old, "delta_rd_score"),
                    "old_s_q_mean": value(old_feat, "s_q_mean"),
                    "old_raw_gate_mean": value(old_feat, "householder_gate_raw_mean"),
                    "old_strength_mean": value(old_feat, "householder_strength_mean"),
                    "old_delta_rms": value(old_feat, "householder_delta_rms"),
                    "old_latent_quant_mse": value(old_feat, "rvq_latent_quant_mse"),
                    "old_y_error_rms": value(old_feat, "y_error_rms"),
                    "risk_s_q_mean": value(risk_feat, "s_q_mean"),
                    "risk_raw_gate_mean": value(risk_feat, "householder_gate_raw_mean"),
                    "risk_strength_mean": value(risk_feat, "householder_strength_mean"),
                    "risk_delta_rms": value(risk_feat, "householder_delta_rms"),
                    "risk_latent_quant_mse": value(risk_feat, "rvq_latent_quant_mse"),
                    "risk_y_error_rms": value(risk_feat, "y_error_rms"),
                    "risk_multiplier_mean": value(risk_feat, "householder_risk_multiplier_mean"),
                }
            )
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--output-prefix", default="gate025_min090_selector_val4096")
    parser.add_argument("--protocol", choices=["legacy", "current_holdout"], default="legacy")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    rows = load_rows(analysis_dir, args.protocol)
    feature_names = [
        "hcs_rd",
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

    all_thresholds = []
    best_by_feature = []
    for feature in feature_names:
        ranked = best_thresholds(rows, feature)
        all_thresholds.extend(ranked)
        best_by_feature.append(ranked[0])
    all_thresholds.sort(key=lambda row: row["mean_delta"])
    best_by_feature.sort(key=lambda row: row["mean_delta"])

    summary = summarize(rows)
    per_seed = {str(seed): summarize([row for row in rows if row["seed"] == seed]) for seed in SEEDS}
    correlations = {
        feature: {
            "corr_hcs_rd": corr([row["hcs_rd"] for row in rows], [row[feature] for row in rows]),
            "corr_min090_minus_old": corr(
                [row["risk_minus_old_delta_rd"] for row in rows],
                [row[feature] for row in rows],
            ),
        }
        for feature in feature_names
    }

    csv_path = analysis_dir / f"{args.output_prefix}_thresholds.csv"
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
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_thresholds)

    json_path = analysis_dir / f"{args.output_prefix}_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "per_seed_summary": per_seed,
                "best_by_feature": best_by_feature,
                "best_overall": all_thresholds[:20],
                "correlations": correlations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    md_path = analysis_dir / f"{args.output_prefix}_summary.md"
    lines = [
        "# Gate0.25 vs min090 selector analysis",
        "",
        "All deltas are method minus HCS RD score on OpenImages val4096; lower is better. Use --protocol current_holdout for the paper-facing start_index=4096 holdout slice.",
        "`hcs_rd` uses the HCS result as an oracle difficulty proxy and is not deployable without a predictor.",
        "",
        "## Overall",
        "",
        "| policy | mean delta RD | vs old gate0.25 | min090 fraction |",
        "|---|---:|---:|---:|",
        f"| old gate0.25 | {fmt(summary['old_gate_mean_delta'], True)} | {fmt(0.0, True)} | 0.000000 |",
        f"| min090 risk | {fmt(summary['min090_mean_delta'], True)} | {fmt(summary['min090_mean_delta'] - summary['old_gate_mean_delta'], True)} | 1.000000 |",
        f"| oracle old/min090 | {fmt(summary['oracle_old_or_min090_delta'], True)} | {fmt(summary['oracle_old_or_min090_delta'] - summary['old_gate_mean_delta'], True)} | {fmt(summary['min090_beats_old_rate'])} |",
        f"| oracle HCS/old/min090 | {fmt(summary['oracle_hcs_or_old_or_min090_delta'], True)} | {fmt(summary['oracle_hcs_or_old_or_min090_delta'] - summary['old_gate_mean_delta'], True)} | n/a |",
        "",
        "## Per Seed",
        "",
        "| seed | old gate0.25 | min090 | oracle old/min090 | min090 beats old |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = per_seed[str(seed)]
        lines.append(
            f"| {seed} | {fmt(row['old_gate_mean_delta'], True)} | {fmt(row['min090_mean_delta'], True)} | "
            f"{fmt(row['oracle_old_or_min090_delta'], True)} | {fmt(row['min090_beats_old_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Single-Feature Thresholds",
            "",
            "| feature | direction | threshold | mean delta RD | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_feature[:12]:
        lines.append(
            f"| {row['feature']} | {row['direction']} | {fmt(float(row['threshold']))} | "
            f"{fmt(float(row['mean_delta']), True)} | {fmt(float(row['mean_delta']) - summary['old_gate_mean_delta'], True)} | "
            f"{fmt(float(row['risk_fraction']))} | {fmt(float(row['seed1234_delta']), True)} | "
            f"{fmt(float(row['seed2345_delta']), True)} | {fmt(float(row['seed3456_delta']), True)} |"
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
    for feature, vals in sorted(
        correlations.items(), key=lambda item: abs(item[1]["corr_min090_minus_old"]), reverse=True
    ):
        lines.append(
            f"| {feature} | {fmt(vals['corr_hcs_rd'], True)} | "
            f"{fmt(vals['corr_min090_minus_old'], True)} |"
        )

    best_deployable = next((row for row in best_by_feature if row["feature"] != "hcs_rd"), None)
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "The oracle old/min090 mixture estimates how much image-level reliability control could recover if it knew "
        "which gate variant would win on each image. A large oracle gap means the failure is controllable in principle."
    )
    if best_deployable is not None:
        improvement = float(best_deployable["mean_delta"]) - summary["old_gate_mean_delta"]
        lines.append(
            f"The best non-oracle single-feature threshold is `{best_deployable['feature']}` "
            f"`{best_deployable['direction']}` {float(best_deployable['threshold']):.6f}, "
            f"with mean delta {fmt(float(best_deployable['mean_delta']), True)} "
            f"({fmt(improvement, True)} vs old gate0.25) and min090 fraction "
            f"{fmt(float(best_deployable['risk_fraction']))}."
        )
    lines.append(
        "If deployable thresholds remain much weaker than the oracle, the next method should learn a residual/local "
        "reliability branch instead of applying another global multiplicative gate floor."
    )
    md_path.write_text("\n".join(lines) + "\n")

    print(md_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
