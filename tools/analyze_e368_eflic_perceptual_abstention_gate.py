#!/usr/bin/env python3
"""Audit simple perceptual abstention gates on E366 EF-LIC policy outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _fmt(value: float) -> str:
    return f"{value:+.6f}"


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "image": row["image"],
                    "chosen": row["chosen"],
                    "chosen_score": float(row["chosen_score"]),
                    "predicted_score": float(row["predicted_score"]),
                    "oracle": row["oracle"],
                    "oracle_score": float(row["oracle_score"]),
                }
            )
    return rows


def _summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores: list[float] = []
    choices: dict[str, int] = {}
    for row in rows:
        use_hcg = row["predicted_score"] <= threshold
        score = row["chosen_score"] if use_hcg else 0.0
        chosen = row["chosen"] if use_hcg else "noop"
        scores.append(score)
        choices[chosen] = choices.get(chosen, 0) + 1
    return {
        "threshold": threshold,
        "mean_score": sum(scores) / len(scores),
        "worst_score": max(scores),
        "score_wins": sum(1 for value in scores if value < 0.0),
        "selected": sum(1 for value in scores if value != 0.0),
        "images": len(scores),
        "choices": choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-image",
        type=Path,
        default=Path("experiments/analysis/e366_eflic_perceptual_candidate_policy_loo_kodak24_riskm060.per_image.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e368_eflic_perceptual_abstention_gate_kodak24"),
    )
    args = parser.parse_args()

    rows = _load(args.per_image)
    thresholds = sorted({row["predicted_score"] for row in rows})
    thresholds = [min(thresholds) - 1e-9, *thresholds, max(thresholds) + 1e-9]
    summaries = [_summarize(rows, threshold) for threshold in thresholds]
    by_mean = sorted(summaries, key=lambda row: (row["mean_score"], row["worst_score"]))
    no_positive_tail = [row for row in summaries if row["worst_score"] <= 0.0]
    by_tail = sorted(no_positive_tail, key=lambda row: (row["mean_score"], -row["score_wins"]))
    payload = {
        "metric_policy": "PSNR excluded; score = delta_DISTS + 3 * delta_LPIPS, lower is better",
        "best_mean": by_mean[:10],
        "best_no_positive_tail": by_tail[:10],
    }

    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with args.output_prefix.with_suffix(".md").open("w") as f:
        f.write("# E368 EF-LIC Perceptual Abstention Gate Audit\n\n")
        f.write("This audit uses only the E366 predicted perceptual score and chosen candidate score. ")
        f.write("It is a diagnostic for controller design, not a final held-out claim. PSNR is excluded.\n\n")
        f.write("## Best Mean Thresholds\n\n")
        f.write("| threshold | mean score | worst score | score wins | selected | choices |\n")
        f.write("|---:|---:|---:|---:|---:|---|\n")
        for row in by_mean[:8]:
            choices = ", ".join(f"{k}:{v}" for k, v in sorted(row["choices"].items()))
            f.write(
                f"| {row['threshold']:+.6f} | {_fmt(row['mean_score'])} | {_fmt(row['worst_score'])} | "
                f"{row['score_wins']}/{row['images']} | {row['selected']}/{row['images']} | {choices} |\n"
            )
        f.write("\n## Best Thresholds With No Positive Tail\n\n")
        f.write("| threshold | mean score | worst score | score wins | selected | choices |\n")
        f.write("|---:|---:|---:|---:|---:|---|\n")
        for row in by_tail[:8]:
            choices = ", ".join(f"{k}:{v}" for k, v in sorted(row["choices"].items()))
            f.write(
                f"| {row['threshold']:+.6f} | {_fmt(row['mean_score'])} | {_fmt(row['worst_score'])} | "
                f"{row['score_wins']}/{row['images']} | {row['selected']}/{row['images']} | {choices} |\n"
            )
        f.write("\nInterpretation: a scalar abstention threshold can improve tail safety only by discarding most useful HCG decisions. ")
        f.write("The next EF-LIC controller should therefore move to local/slice-level reliability rather than image-level thresholding.\n")


if __name__ == "__main__":
    main()
