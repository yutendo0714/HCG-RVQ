#!/usr/bin/env python3
"""Split-audit simple EF-LIC perceptual feature gates.

This diagnostic searches for simple rules of the form:

    use a fixed slice candidate if feature >=/<= threshold, otherwise noop

Thresholds are selected on a train image split and evaluated on a held-out split.
The score is perceptual-only: delta_DISTS + lpips_weight * delta_LPIPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TARGET_COLUMNS = {
    "active_dists",
    "active_lpips",
    "active_ms_ssim",
    "active_psnr",
    "base_dists",
    "base_lpips",
    "base_ms_ssim",
    "base_psnr",
    "bpp",
    "contract_ok",
    "delta_bpp",
    "delta_dists",
    "delta_lpips",
    "delta_ms_ssim",
    "delta_psnr",
    "max_baseline_diff",
    "max_decode_diff",
    "mean_baseline_diff",
    "mean_decode_diff",
    "nonfinite",
    "payload_equal",
    "payload_len_equal",
    "perceptual_score",
    "perceptual_score_win",
    "triple_perceptual_win",
}

STRING_COLUMNS = {"image", "active_slices", "direction_source", "mode", "omitted_slice", "single_slice"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("experiments/analysis/e364_eflic_perceptual_slice_isolation_kodak24_riskm060.rows.csv"),
    )
    parser.add_argument("--lpips-weight", type=float, default=3.0)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument("--min-train-selected", type=int, default=2)
    parser.add_argument("--max-positive-train", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e383_eflic_perceptual_feature_gate_split_kodak16_8"),
    )
    return parser.parse_args()


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def read_rows(path: Path, lpips_weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for key, value in raw.items():
                if key in STRING_COLUMNS:
                    continue
                parsed = as_float(value)
                if math.isfinite(parsed):
                    row[key] = parsed
            if int(row.get("contract_ok", 0)) != 1:
                continue
            if int(row.get("nonfinite", 1)) != 0:
                continue
            if abs(float(row.get("delta_bpp", math.nan))) > 1e-12:
                continue
            if abs(float(row.get("max_decode_diff", math.nan))) > 1e-12:
                continue
            row["score"] = float(row.get("delta_dists", 0.0)) + lpips_weight * float(row.get("delta_lpips", 0.0))
            rows.append(row)
    return rows


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for key, value in rows[0].items():
        if key in TARGET_COLUMNS:
            continue
        if key in {"image", "active_slices", "direction_source", "mode", "omitted_slice", "single_slice", "score"}:
            continue
        if isinstance(value, float) and math.isfinite(value):
            names.append(key)
    return sorted(names)


def eval_gate(rows: list[dict[str, Any]], feature: str, direction: str, threshold: float) -> dict[str, Any]:
    scores: list[float] = []
    selected = 0
    choices: dict[str, int] = defaultdict(int)
    for row in rows:
        value = float(row[feature])
        use = value >= threshold if direction == ">=" else value <= threshold
        if use:
            score = float(row["score"])
            choice = str(row["active_slices"])
            selected += 1
        else:
            score = 0.0
            choice = "noop"
        scores.append(score)
        choices[choice] += 1
    positives = sum(score > 0.0 for score in scores)
    wins = sum(score < 0.0 for score in scores)
    return {
        "mean_score": finite_mean(scores),
        "worst_score": max(scores) if scores else math.nan,
        "selected": selected,
        "wins": wins,
        "positives": positives,
        "images": len(rows),
        "choices": dict(sorted(choices.items())),
    }


def candidate_thresholds(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = sorted({float(row[feature]) for row in rows if math.isfinite(float(row[feature]))})
    if not values:
        return []
    mids = [(a + b) * 0.5 for a, b in zip(values, values[1:])]
    eps = 1e-12
    return [values[0] - eps, *mids, values[-1] + eps]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.rows, args.lpips_weight)
    images = sorted({str(row["image"]) for row in rows})
    start = args.train_start_index
    stop = start + args.train_count
    train_images = set(images[start:stop])
    eval_images = set(images[:start] + images[stop:])

    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row["active_slices"])].append(row)

    features = feature_names(rows)
    selected_policies: list[dict[str, Any]] = []
    all_policies: list[dict[str, Any]] = []
    for action, action_rows in sorted(by_action.items()):
        train_rows = [row for row in action_rows if str(row["image"]) in train_images]
        eval_rows = [row for row in action_rows if str(row["image"]) in eval_images]
        if not train_rows or not eval_rows:
            continue
        for feature in features:
            if not all(feature in row and isinstance(row[feature], float) for row in train_rows + eval_rows):
                continue
            for direction in (">=", "<="):
                for threshold in candidate_thresholds(train_rows, feature):
                    train = eval_gate(train_rows, feature, direction, threshold)
                    if train["selected"] < args.min_train_selected:
                        continue
                    policy = {
                        "action": action,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "train": train,
                        "eval": eval_gate(eval_rows, feature, direction, threshold),
                    }
                    all_policies.append(policy)
                    if train["positives"] <= args.max_positive_train:
                        selected_policies.append(policy)

    def rank(policy: dict[str, Any]) -> tuple[float, float, int, int]:
        ev = policy["eval"]
        return (
            float(ev["mean_score"]),
            float(ev["worst_score"]),
            -int(ev["wins"]),
            -int(ev["selected"]),
        )

    best = sorted(selected_policies, key=rank)[: args.top_k]
    safe = [
        policy
        for policy in selected_policies
        if policy["eval"]["worst_score"] <= 0.0 and policy["eval"]["positives"] == 0
    ]
    best_safe = sorted(safe, key=rank)[: args.top_k]
    payload = {
        "metric_policy": "PSNR excluded; score = delta_DISTS + lpips_weight * delta_LPIPS, lower is better",
        "rows": len(rows),
        "train_images": sorted(train_images),
        "eval_images": sorted(eval_images),
        "train_start_index": args.train_start_index,
        "lpips_weight": args.lpips_weight,
        "all_policy_count": len(all_policies),
        "selected_policy_count": len(selected_policies),
        "best": best,
        "best_safe": best_safe,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write("# E383 EF-LIC Perceptual Feature Gate Split\n\n")
        f.write("This split audit searches simple fixed-action feature gates on the first Kodak images and evaluates on the remaining images. ")
        f.write("PSNR is excluded; lower score is better.\n\n")
        f.write(f"- rows: `{len(rows)}`\n")
        f.write(f"- train start index: `{args.train_start_index}`\n")
        f.write(f"- train images: `{len(train_images)}`\n")
        f.write(f"- eval images: `{len(eval_images)}`\n")
        f.write(f"- selected policies with train positive rows <= {args.max_positive_train}: `{len(selected_policies)}`\n\n")

        def table(title: str, policies: list[dict[str, Any]]) -> None:
            f.write(f"## {title}\n\n")
            f.write("| action | feature | dir | threshold | train mean | train worst | train selected | eval mean | eval worst | eval wins | eval selected | eval positives |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for policy in policies:
                tr = policy["train"]
                ev = policy["eval"]
                f.write(
                    f"| {policy['action']} | {policy['feature']} | {policy['direction']} | {policy['threshold']:.6g} | "
                    f"{tr['mean_score']:+.6f} | {tr['worst_score']:+.6f} | {tr['selected']}/{tr['images']} | "
                    f"{ev['mean_score']:+.6f} | {ev['worst_score']:+.6f} | {ev['wins']}/{ev['images']} | "
                    f"{ev['selected']}/{ev['images']} | {ev['positives']} |\n"
                )
            f.write("\n")

        table("Best Eval Mean Under Train-Tail Constraint", best[:10])
        table("Best Eval-Safe Policies", best_safe[:10])
        f.write(
            "Interpretation: policies here are design diagnostics for a local/slice reliability controller. "
            "A useful EF-LIC promotion candidate needs held-out negative mean and no positive tail without relying on PSNR.\n"
        )


if __name__ == "__main__":
    main()
