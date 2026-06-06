from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

from analyze_beta005_decoder_safe_selector import (
    DECODER_SAFE_FEATURES,
    DIAGNOSTIC_FEATURES,
    OUT as SELECTOR_OUT,
    best_thresholds,
    build_rows,
    fmt,
    mean,
    summarize_rd,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "beta005_teacher_target_audit"


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranked = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranked[order[k]] = rank
        i = j
    return ranked


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return float("nan")
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def metric_counts(labels: list[bool], preds: list[bool]) -> dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, preds) if label and pred)
    fp = sum(1 for label, pred in zip(labels, preds) if not label and pred)
    tn = sum(1 for label, pred in zip(labels, preds) if not label and not pred)
    fn = sum(1 for label, pred in zip(labels, preds) if label and not pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "positive_fraction": mean(1.0 if pred else 0.0 for pred in preds),
    }


def percentile_thresholds(values: list[float], bins: int = 101) -> list[float]:
    values = sorted(v for v in values if finite(v))
    if not values:
        return []
    thresholds = []
    n = len(values) - 1
    for i in range(bins):
        pos = n * i / (bins - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            thresholds.append(values[lo])
        else:
            alpha = pos - lo
            thresholds.append(values[lo] * (1.0 - alpha) + values[hi] * alpha)
    return sorted(set(thresholds))


def selected_rd(rows: list[dict[str, float | str]], preds: list[bool]) -> float:
    return mean(
        float(row["previous_local_rd"] if pred else row["beta005_rd"])
        for row, pred in zip(rows, preds)
    )


def best_binary_thresholds(
    rows: list[dict[str, float | str]],
    labels: list[bool],
    features: list[str],
    target_name: str,
) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    hcs_rd = summarize_rd(rows, "hcs_rd")
    beta_rd = summarize_rd(rows, "beta005_rd")
    previous_rd = summarize_rd(rows, "previous_local_rd")
    for feature in features:
        values = [float(row[feature]) for row in rows]
        for threshold in percentile_thresholds(values):
            for direction in ["le", "ge"]:
                preds = [value <= threshold if direction == "le" else value >= threshold for value in values]
                metrics = metric_counts(labels, preds)
                rd = selected_rd(rows, preds)
                out.append(
                    {
                        "target": target_name,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "rd": rd,
                        "delta_vs_hcs": rd - hcs_rd,
                        "delta_vs_beta005": rd - beta_rd,
                        "delta_vs_previous_local": rd - previous_rd,
                        **metrics,
                    }
                )
    return sorted(out, key=lambda row: (-float(row["f1"]), float(row["rd"])))


def cohen_d(rows: list[dict[str, float | str]], labels: list[bool], feature: str) -> float:
    pos = [float(row[feature]) for row, label in zip(rows, labels) if label]
    neg = [float(row[feature]) for row, label in zip(rows, labels) if not label]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    mp = mean(pos)
    mn = mean(neg)
    vp = sum((x - mp) ** 2 for x in pos) / (len(pos) - 1)
    vn = sum((x - mn) ** 2 for x in neg) / (len(neg) - 1)
    pooled = math.sqrt(((len(pos) - 1) * vp + (len(neg) - 1) * vn) / (len(pos) + len(neg) - 2))
    return (mp - mn) / pooled if pooled > 0.0 else float("nan")


def group_summary(rows: list[dict[str, float | str]], labels: list[bool]) -> dict[str, float]:
    margins = [float(row["beta005_rd"]) - float(row["previous_local_rd"]) for row in rows]
    return {
        "count": float(len(rows)),
        "previous_local_win_fraction": mean(1.0 if label else 0.0 for label in labels),
        "margin_beta_minus_previous_mean": mean(margins),
        "margin_beta_minus_previous_median": median(margins),
        "hcs_rd": summarize_rd(rows, "hcs_rd"),
        "old_gate025_rd": summarize_rd(rows, "old_rd"),
        "min090_rd": summarize_rd(rows, "min090_rd"),
        "previous_local_rd": summarize_rd(rows, "previous_local_rd"),
        "beta005_rd": summarize_rd(rows, "beta005_rd"),
        "oracle_rd": mean(min(float(row["previous_local_rd"]), float(row["beta005_rd"])) for row in rows),
        "delta_rms_mean": mean(float(row["rvq_householder_delta_rms"]) for row in rows),
        "raw_gate_mean": mean(float(row["rvq_householder_gate_raw"]) for row in rows),
        "strength_mean": mean(float(row["rvq_householder_strength"]) for row in rows),
        "latent_qmse_mean": mean(float(row["rvq_latent_quant_mse"]) for row in rows),
    }


def label_from_policy(rows: list[dict[str, float | str]], policy: dict[str, float | str]) -> list[bool]:
    feature = str(policy["feature"])
    threshold = float(policy["threshold"])
    direction = str(policy["direction"])
    use_beta = [
        float(row[feature]) <= threshold if direction == "le" else float(row[feature]) >= threshold
        for row in rows
    ]
    return [not value for value in use_beta]


def top_correlations(
    rows: list[dict[str, float | str]],
    features: list[str],
    target: list[float],
    target_name: str,
) -> list[dict[str, float | str]]:
    out = []
    for feature in features:
        values = [float(row[feature]) for row in rows]
        out.append(
            {
                "target": target_name,
                "feature": feature,
                "pearson": pearson(values, target),
                "spearman": spearman(values, target),
                "feature_mean": mean(values),
            }
        )
    return sorted(out, key=lambda row: abs(float(row["spearman"])), reverse=True)


def main() -> None:
    rows = build_rows()
    labels_previous_wins = [float(row["previous_local_rd"]) < float(row["beta005_rd"]) for row in rows]
    margins = [float(row["beta005_rd"]) - float(row["previous_local_rd"]) for row in rows]

    safe_policies = best_thresholds(rows, DECODER_SAFE_FEATURES)
    diagnostic_policies = best_thresholds(rows, DIAGNOSTIC_FEATURES)
    best_safe = safe_policies[0]
    best_diagnostic = diagnostic_policies[0]
    best_raw_gate = next(policy for policy in safe_policies if policy["feature"] == "rvq_householder_gate_raw")
    best_delta_rms = next(policy for policy in diagnostic_policies if policy["feature"] == "rvq_householder_delta_rms")

    labels_diag_fallback = label_from_policy(rows, best_delta_rms)
    labels_raw_fallback = label_from_policy(rows, best_raw_gate)

    safe_vs_oracle = best_binary_thresholds(
        rows, labels_previous_wins, DECODER_SAFE_FEATURES, "previous_local_wins"
    )
    safe_vs_diag = best_binary_thresholds(
        rows, labels_diag_fallback, DECODER_SAFE_FEATURES, "diagnostic_delta_fallback"
    )
    feature_rows = []
    for feature in DECODER_SAFE_FEATURES + DIAGNOSTIC_FEATURES:
        values = [float(row[feature]) for row in rows]
        pos = [value for value, label in zip(values, labels_previous_wins) if label]
        neg = [value for value, label in zip(values, labels_previous_wins) if not label]
        feature_rows.append(
            {
                "feature": feature,
                "group": "decoder_safe" if feature in DECODER_SAFE_FEATURES else "diagnostic",
                "cohen_d_previous_wins": cohen_d(rows, labels_previous_wins, feature),
                "mean_previous_wins": mean(pos),
                "mean_beta005_wins": mean(neg),
                "pearson_margin": pearson(values, margins),
                "spearman_margin": spearman(values, margins),
            }
        )
    feature_rows = sorted(feature_rows, key=lambda row: abs(float(row["cohen_d_previous_wins"])), reverse=True)

    by_seed = {}
    for seed in sorted({str(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        seed_labels = [float(row["previous_local_rd"]) < float(row["beta005_rd"]) for row in seed_rows]
        by_seed[seed] = group_summary(seed_rows, seed_labels)

    cross_label = {
        "raw_gate_fallback_vs_previous_local_wins": metric_counts(labels_previous_wins, labels_raw_fallback),
        "delta_rms_fallback_vs_previous_local_wins": metric_counts(labels_previous_wins, labels_diag_fallback),
        "raw_gate_fallback_vs_delta_rms_fallback": metric_counts(labels_diag_fallback, labels_raw_fallback),
    }

    result = {
        "num_rows": len(rows),
        "selector_source_json": str(SELECTOR_OUT.with_suffix(".json")),
        "summary": group_summary(rows, labels_previous_wins),
        "by_seed": by_seed,
        "best_safe_policy": best_safe,
        "best_diagnostic_policy": best_diagnostic,
        "best_raw_gate_policy": best_raw_gate,
        "best_delta_rms_policy": best_delta_rms,
        "cross_label_metrics": cross_label,
        "top_safe_predictors_previous_local_wins": safe_vs_oracle[:20],
        "top_safe_predictors_diagnostic_delta_fallback": safe_vs_diag[:20],
        "top_feature_separations": feature_rows[:30],
        "top_correlations_with_margin": top_correlations(
            rows, DECODER_SAFE_FEATURES + DIAGNOSTIC_FEATURES, margins, "beta_minus_previous_margin"
        )[:30],
        "decision": (
            "The holdout oracle and diagnostic delta-rms labels are useful for target design, but they must not be "
            "used as validation-label supervision in a paper-facing training run. The next trainable direction should "
            "prefer train-split teacher generation or tail-only distribution regularization over broad raw-gate "
            "multiplicative backoff."
        ),
    }

    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "target",
        "feature",
        "direction",
        "threshold",
        "rd",
        "delta_vs_hcs",
        "delta_vs_beta005",
        "delta_vs_previous_local",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "positive_fraction",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in safe_vs_oracle[:100] + safe_vs_diag[:100]:
            writer.writerow({key: row[key] for key in fieldnames})

    def policy_line(name: str, policy: dict[str, float | str]) -> str:
        return (
            f"| {name} | {policy['feature']} | {policy['direction']} | "
            f"{fmt(float(policy['threshold']))} | {fmt(float(policy['rd']))} | "
            f"{fmt(float(policy['delta_vs_hcs']), signed=True)} | "
            f"{fmt(float(policy['delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(policy['beta_fraction']))} |"
        )

    lines = [
        "# Beta005 Teacher Target Audit",
        "",
        "This audit uses existing OpenImages holdout4096 per-image rows to convert the beta005/previous-local selector headroom into trainable target choices. It is not a new training result; it is a protocol guard against turning validation-set oracle behavior into a paper-facing method.",
        "",
        "## Core Summary",
        "",
        "| split | n | previous-local win frac | HCS RD | old gate0.25 RD | previous-local RD | beta005 RD | oracle RD | mean margin beta-prev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = result["summary"]
    lines.append(
        f"| all | {int(summary['count'])} | {fmt(summary['previous_local_win_fraction'])} | "
        f"{fmt(summary['hcs_rd'])} | {fmt(summary['old_gate025_rd'])} | "
        f"{fmt(summary['previous_local_rd'])} | {fmt(summary['beta005_rd'])} | "
        f"{fmt(summary['oracle_rd'])} | {fmt(summary['margin_beta_minus_previous_mean'], signed=True)} |"
    )
    for seed, seed_summary in by_seed.items():
        lines.append(
            f"| seed{seed} | {int(seed_summary['count'])} | {fmt(seed_summary['previous_local_win_fraction'])} | "
            f"{fmt(seed_summary['hcs_rd'])} | {fmt(seed_summary['old_gate025_rd'])} | "
            f"{fmt(seed_summary['previous_local_rd'])} | {fmt(seed_summary['beta005_rd'])} | "
            f"{fmt(seed_summary['oracle_rd'])} | {fmt(seed_summary['margin_beta_minus_previous_mean'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Selector Policies",
            "",
            "| policy | feature | dir | threshold | selected RD | vs HCS | vs beta005 | beta fraction |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            policy_line("best decoder-safe", best_safe),
            policy_line("best diagnostic", best_diagnostic),
            policy_line("raw gate decoder-safe", best_raw_gate),
            policy_line("delta RMS diagnostic", best_delta_rms),
            "",
            "## Cross-Label Checks",
            "",
            "| predicted label | target label | precision | recall | F1 | accuracy | positive frac |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    cross_rows = [
        ("raw-gate fallback", "previous-local wins", cross_label["raw_gate_fallback_vs_previous_local_wins"]),
        ("delta-rms fallback", "previous-local wins", cross_label["delta_rms_fallback_vs_previous_local_wins"]),
        ("raw-gate fallback", "delta-rms fallback", cross_label["raw_gate_fallback_vs_delta_rms_fallback"]),
    ]
    for pred_name, target_name, metrics in cross_rows:
        lines.append(
            f"| {pred_name} | {target_name} | {fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
            f"{fmt(metrics['f1'])} | {fmt(metrics['accuracy'])} | {fmt(metrics['positive_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Decoder-Safe Predictors Of Previous-Local Wins",
            "",
            "| feature | dir | threshold | F1 | precision | recall | selected RD | vs beta005 | fallback frac |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in safe_vs_oracle[:10]:
        lines.append(
            f"| {row['feature']} | {row['direction']} | {fmt(float(row['threshold']))} | "
            f"{fmt(float(row['f1']))} | {fmt(float(row['precision']))} | {fmt(float(row['recall']))} | "
            f"{fmt(float(row['rd']))} | {fmt(float(row['delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(row['positive_fraction']))} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Feature Separations",
            "",
            "| feature | group | Cohen d | previous-win mean | beta-win mean | Spearman margin |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in feature_rows[:12]:
        lines.append(
            f"| {row['feature']} | {row['group']} | {fmt(float(row['cohen_d_previous_wins']), signed=True)} | "
            f"{fmt(float(row['mean_previous_wins']))} | {fmt(float(row['mean_beta005_wins']))} | "
            f"{fmt(float(row['spearman_margin']), signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- The diagnostic delta-RMS teacher gives the clearest headroom, but it is diagnostic-only because it can depend on latent/codebook outcomes.",
            "- Raw gate remains useful as a decoder-safe reliability proxy, but E070 showed that multiplying it broadly is too blunt.",
            "- The next training direction should be either train-split teacher generation for a reliability head, or a tail-only distribution regularizer that only touches the risky high-geometry tail.",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary": result["summary"],
        "best_safe_policy": result["best_safe_policy"],
        "best_delta_rms_policy": result["best_delta_rms_policy"],
        "cross_label_metrics": result["cross_label_metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
