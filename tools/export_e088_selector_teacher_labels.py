#!/usr/bin/env python3
"""Export teacher labels from the E088 decoder-safe transfer selector."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_beta005_decoder_safe_selector import DECODER_SAFE_FEATURES, fmt, mean
from analyze_e088_transfer_learned_selector import (
    OUT as E088_OUT,
    evaluate_scores,
    feature_matrix,
    load_transfer_rows,
    logreg_policy,
    selected_rds,
    sigmoid,
    standardize,
    train_logreg,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "e088_decoder_safe_selector_teacher_labels_transfer8192"


def confidence_weight(score: float, threshold: float) -> float:
    # Keep this deliberately mild. The target itself is already high precision;
    # a large weight would repeat the earlier margin-BCE overdrive failure mode.
    distance = abs(score - threshold)
    return 1.0 + min(2.0, distance * 4.0)


def main() -> None:
    rows = load_transfer_rows()
    x, y, margin, valid_rows = feature_matrix(rows, DECODER_SAFE_FEATURES)
    x_std, _x_test, loc, scale = standardize(x, x)
    weights = np.ones_like(y)
    weight, bias = train_logreg(x_std, y, weights)
    scores = sigmoid(x_std @ weight + bias)
    policy = evaluate_scores(
        "decoder_safe_logreg_uniform",
        "decoder_safe_logreg",
        valid_rows,
        valid_rows,
        scores,
        scores,
    )
    threshold = float(policy["threshold"])
    _rds, selected = selected_rds(valid_rows, scores, threshold)

    out_rows = []
    for row, score, flag, abs_margin in zip(valid_rows, scores, selected, margin, strict=True):
        out_rows.append(
            {
                "seed": row["seed"],
                "path": row["path"],
                "e088_decoder_safe_score": float(score),
                "e088_decoder_safe_threshold": threshold,
                "e088_decoder_safe_selected_previous_local": float(flag),
                "e088_decoder_safe_reliability_keep": 0.0 if flag else 1.0,
                "e088_decoder_safe_confidence_weight": confidence_weight(float(score), threshold),
                "margin_abs": float(abs_margin),
                "beta005_rd": float(row["beta005_rd"]),
                "previous_local_rd": float(row["previous_local_rd"]),
                "previous_local_wins": float(row["previous_local_wins"]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    by_seed = {}
    for seed in sorted({str(row["seed"]) for row in out_rows}):
        subset = [row for row in out_rows if str(row["seed"]) == seed]
        by_seed[seed] = {
            "rows": len(subset),
            "selected_fraction": mean(float(row["e088_decoder_safe_selected_previous_local"]) for row in subset),
            "positive_keep_fraction": mean(float(row["e088_decoder_safe_reliability_keep"]) for row in subset),
            "previous_local_win_fraction": mean(float(row["previous_local_wins"]) for row in subset),
            "mean_confidence_weight": mean(float(row["e088_decoder_safe_confidence_weight"]) for row in subset),
        }

    summary = {
        "rows": len(out_rows),
        "policy": policy,
        "threshold": threshold,
        "score_mean": mean(float(row["e088_decoder_safe_score"]) for row in out_rows),
        "selected_fraction": mean(float(row["e088_decoder_safe_selected_previous_local"]) for row in out_rows),
        "positive_keep_fraction": mean(float(row["e088_decoder_safe_reliability_keep"]) for row in out_rows),
        "previous_local_win_fraction": mean(float(row["previous_local_wins"]) for row in out_rows),
        "by_seed": by_seed,
        "model": {
            "features": DECODER_SAFE_FEATURES,
            "mean": [float(v) for v in loc],
            "scale": [float(v) for v in scale],
            "weight": [float(v) for v in weight],
            "bias": float(bias),
        },
        "source": str(E088_OUT.with_suffix(".json").relative_to(ROOT)),
    }
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E088 Decoder-Safe Selector Teacher Labels",
        "",
        "These labels distill the E088 transfer-trained decoder-safe logistic selector. `e088_decoder_safe_reliability_keep=1` means keep beta005-like reliability; `0` means the selector chose the previous-local/local-cap behavior for this transfer image.",
        "",
        "| split | rows | selected previous-local | keep fraction | previous-local win frac | mean confidence weight |",
        "|---|---:|---:|---:|---:|---:|",
        f"| aggregate | {len(out_rows)} | {fmt(summary['selected_fraction'])} | {fmt(summary['positive_keep_fraction'])} | {fmt(summary['previous_local_win_fraction'])} | {fmt(mean(float(row['e088_decoder_safe_confidence_weight']) for row in out_rows))} |",
    ]
    for seed, item in by_seed.items():
        lines.append(
            f"| seed{seed} | {item['rows']} | {fmt(item['selected_fraction'])} | {fmt(item['positive_keep_fraction'])} | {fmt(item['previous_local_win_fraction'])} | {fmt(item['mean_confidence_weight'])} |"
        )
    lines.extend(
        [
            "",
            f"Threshold: `{threshold:.6f}`",
            f"CSV: `{OUT.with_suffix('.csv').relative_to(ROOT)}`",
            "",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(out_rows), "threshold": threshold, "selected_fraction": summary["selected_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
