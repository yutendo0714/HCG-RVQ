#!/usr/bin/env python3
"""Cross-dataset EF-LIC perceptual feature-gate audit.

This extends the E383 split diagnostic from image splits to dataset splits:
fit simple decoder-visible feature gates on one dataset and evaluate them on
another dataset.  The decision score is perceptual-only:

    delta_DISTS + lpips_weight * delta_LPIPS

PSNR columns may exist in legacy rows, but are not used for ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_e383_eflic_perceptual_feature_gate_split import (
    candidate_thresholds,
    eval_gate,
    feature_names,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Dataset rows in dataset=csv format. Pass at least two.",
    )
    parser.add_argument("--lpips-weight", type=float, default=3.0)
    parser.add_argument("--min-train-selected", type=int, default=3)
    parser.add_argument("--max-positive-train", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e388_eflic_cross_dataset_feature_gate"),
    )
    return parser.parse_args()


def read_labeled_rows(inputs: list[str], lpips_weight: float) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for spec in inputs:
        if "=" not in spec:
            raise SystemExit(f"--input must be dataset=csv, got {spec!r}")
        dataset, csv_path = spec.split("=", 1)
        for row in read_rows(Path(csv_path), lpips_weight):
            row = dict(row)
            row["dataset"] = dataset
            labeled.append(row)
    return labeled


def rank_policy(policy: dict[str, Any]) -> tuple[float, float, int, int]:
    ev = policy["eval"]
    return (
        float(ev["mean_score"]),
        float(ev["worst_score"]),
        -int(ev["wins"]),
        -int(ev["selected"]),
    )


def search_policies(
    rows: list[dict[str, Any]],
    train_dataset: str,
    eval_dataset: str,
    *,
    min_train_selected: int,
    max_positive_train: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_all = [row for row in rows if row["dataset"] == train_dataset]
    eval_all = [row for row in rows if row["dataset"] == eval_dataset]
    if not train_all or not eval_all:
        raise SystemExit(f"empty train/eval split: train={train_dataset}, eval={eval_dataset}")

    names = feature_names(rows)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row["active_slices"])].append(row)

    all_policies: list[dict[str, Any]] = []
    selected_policies: list[dict[str, Any]] = []
    for action, action_rows in sorted(by_action.items()):
        train_rows = [row for row in action_rows if row["dataset"] == train_dataset]
        eval_rows = [row for row in action_rows if row["dataset"] == eval_dataset]
        if not train_rows or not eval_rows:
            continue
        for feature in names:
            if not all(feature in row and isinstance(row[feature], float) for row in train_rows + eval_rows):
                continue
            for direction in (">=", "<="):
                for threshold in candidate_thresholds(train_rows, feature):
                    train = eval_gate(train_rows, feature, direction, threshold)
                    if train["selected"] < min_train_selected:
                        continue
                    policy = {
                        "train_dataset": train_dataset,
                        "eval_dataset": eval_dataset,
                        "action": action,
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "train": train,
                        "eval": eval_gate(eval_rows, feature, direction, threshold),
                    }
                    all_policies.append(policy)
                    if train["positives"] <= max_positive_train:
                        selected_policies.append(policy)
    return all_policies, selected_policies


def main() -> None:
    args = parse_args()
    rows = read_labeled_rows(args.input, args.lpips_weight)
    datasets = sorted({str(row["dataset"]) for row in rows})
    if len(datasets) < 2:
        raise SystemExit("need at least two datasets")

    summaries: list[dict[str, Any]] = []
    best_by_pair: dict[str, list[dict[str, Any]]] = {}
    safe_by_pair: dict[str, list[dict[str, Any]]] = {}
    for train_dataset in datasets:
        for eval_dataset in datasets:
            if train_dataset == eval_dataset:
                continue
            all_policies, selected = search_policies(
                rows,
                train_dataset,
                eval_dataset,
                min_train_selected=args.min_train_selected,
                max_positive_train=args.max_positive_train,
            )
            best = sorted(selected, key=rank_policy)[: args.top_k]
            safe = [
                policy
                for policy in selected
                if policy["eval"]["worst_score"] <= 0.0 and policy["eval"]["positives"] == 0
            ]
            best_safe = sorted(safe, key=rank_policy)[: args.top_k]
            key = f"{train_dataset}_to_{eval_dataset}"
            best_by_pair[key] = best
            safe_by_pair[key] = best_safe
            summaries.append(
                {
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "all_policy_count": len(all_policies),
                    "selected_policy_count": len(selected),
                    "safe_policy_count": len(best_safe),
                    "best_eval_mean": best[0]["eval"]["mean_score"] if best else math.nan,
                    "best_eval_worst": best[0]["eval"]["worst_score"] if best else math.nan,
                    "best_eval_selected": best[0]["eval"]["selected"] if best else 0,
                    "best_eval_positives": best[0]["eval"]["positives"] if best else 0,
                    "best_safe_eval_mean": best_safe[0]["eval"]["mean_score"] if best_safe else math.nan,
                    "best_safe_eval_worst": best_safe[0]["eval"]["worst_score"] if best_safe else math.nan,
                    "best_safe_eval_selected": best_safe[0]["eval"]["selected"] if best_safe else 0,
                    "best_safe_eval_positives": best_safe[0]["eval"]["positives"] if best_safe else 0,
                }
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_policy": "PSNR excluded; score = delta_DISTS + lpips_weight * delta_LPIPS, lower is better",
        "lpips_weight": args.lpips_weight,
        "inputs": args.input,
        "datasets": datasets,
        "rows": len(rows),
        "summaries": summaries,
        "best_by_pair": best_by_pair,
        "safe_by_pair": safe_by_pair,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = args.output_prefix.with_suffix(".summary.csv")
    fields = sorted({key for row in summaries for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write("# E388 EF-LIC Cross-Dataset Feature Gate\n\n")
        f.write("This audit fits simple fixed-action feature gates on one dataset and evaluates on another. ")
        f.write("PSNR is excluded; lower perceptual score is better.\n\n")
        f.write("| train | eval | selected policies | safe policies | best eval mean | best eval worst | best eval selected | best eval positives | best safe mean | best safe worst | best safe selected |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                f"| {row['train_dataset']} | {row['eval_dataset']} | {row['selected_policy_count']} | "
                f"{row['safe_policy_count']} | {row['best_eval_mean']:+.6f} | "
                f"{row['best_eval_worst']:+.6f} | {row['best_eval_selected']} | "
                f"{row['best_eval_positives']} | {row['best_safe_eval_mean']:+.6f} | "
                f"{row['best_safe_eval_worst']:+.6f} | {row['best_safe_eval_selected']} |\n"
            )
        f.write("\n## Best Safe Policies\n\n")
        for key, policies in safe_by_pair.items():
            f.write(f"### {key}\n\n")
            f.write("| action | feature | dir | threshold | train mean | train selected | eval mean | eval worst | eval selected | eval wins |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for policy in policies[:10]:
                tr = policy["train"]
                ev = policy["eval"]
                f.write(
                    f"| {policy['action']} | {policy['feature']} | {policy['direction']} | "
                    f"{policy['threshold']:.6g} | {tr['mean_score']:+.6f} | "
                    f"{tr['selected']}/{tr['images']} | {ev['mean_score']:+.6f} | "
                    f"{ev['worst_score']:+.6f} | {ev['selected']}/{ev['images']} | "
                    f"{ev['wins']}/{ev['images']} |\n"
                )
            f.write("\n")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
