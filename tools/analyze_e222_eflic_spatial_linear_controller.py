#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e221_eflic_spatial_quant_mse_probe import safe_float, summarize_local, write_csv  # noqa: E402


EXCLUDED_FEATURES = {
    "dataset",
    "image",
    "flat_index",
    "y_pos",
    "x_pos",
    "base_mse",
    "active_mse",
    "delta_mse",
    "oracle_delta_mse",
    "helpful",
}


@dataclass(frozen=True)
class RidgePolicy:
    l2: float
    threshold: float
    train_delta_mse: float
    weights: list[float]
    mean: list[float]
    std: list[float]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [dict(row) for row in csv.DictReader(f)]


def infer_feature_names(rows: list[dict[str, Any]], min_finite_frac: float) -> list[str]:
    names: list[str] = []
    for key in rows[0].keys():
        if key in EXCLUDED_FEATURES:
            continue
        vals = np.array([safe_float(row.get(key)) for row in rows], dtype=float)
        if np.isfinite(vals).mean() >= min_finite_frac:
            names.append(key)
    return names


def make_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.array([[safe_float(row.get(f)) for f in feature_names] for row in rows], dtype=float)


def standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(x), x, mean[None, :])
    std = filled.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (filled - mean[None, :]) / std[None, :], mean, std


def standardize_eval(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, mean[None, :])
    return (filled - mean[None, :]) / std[None, :]


def fit_weights(x_std: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=float), x_std], axis=1)
    reg = np.eye(design.shape[1], dtype=float) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def predict(x_std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=float), x_std], axis=1)
    return design @ weights


def best_threshold(rows: list[dict[str, Any]], pred: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(pred)
    if not finite.any():
        return -float("inf"), 0.0, 0.0
    deltas = np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)
    finite = finite & np.isfinite(deltas)
    if not finite.any():
        return -float("inf"), 0.0, 0.0

    values = pred[finite]
    delta_values = deltas[finite]
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_deltas = delta_values[order]
    unique, counts = np.unique(sorted_values, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_deltas = np.cumsum(sorted_deltas)
    grouped_delta_sums = cum_deltas[cum_counts - 1]

    if unique.size == 1:
        thresholds = np.array([unique[0] - 1e-12, unique[0] + 1e-12], dtype=float)
    else:
        mids = (unique[:-1] + unique[1:]) * 0.5
        thresholds = np.concatenate(([unique[0] - 1e-12], mids, [unique[-1] + 1e-12]))

    active_delta_sums = np.concatenate(([0.0], grouped_delta_sums))
    active_counts = np.concatenate(([0], cum_counts))
    n_rows = max(1, len(rows))
    best_t = -float("inf")
    best_score = 0.0
    best_active = 0.0
    for threshold, delta_sum, active_count in zip(thresholds, active_delta_sums, active_counts):
        score = float(delta_sum) / n_rows
        if score < best_score - 1e-15:
            best_t = float(threshold)
            best_score = score
            best_active = float(active_count) / n_rows
    return best_t, best_score, best_active

def fit_policy(rows: list[dict[str, Any]], feature_names: list[str], l2_values: list[float]) -> RidgePolicy:
    x = make_matrix(rows, feature_names)
    y = np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)
    x_std, mean, std = standardize_train(x)
    best: RidgePolicy | None = None
    for l2 in l2_values:
        weights = fit_weights(x_std, y, l2)
        pred = predict(x_std, weights)
        threshold, train_score, _ = best_threshold(rows, pred)
        if best is None or train_score < best.train_delta_mse - 1e-15:
            best = RidgePolicy(
                l2=float(l2),
                threshold=float(threshold),
                train_delta_mse=float(train_score),
                weights=[float(v) for v in weights],
                mean=[float(v) for v in mean],
                std=[float(v) for v in std],
            )
    assert best is not None
    return best


def eval_policy(rows: list[dict[str, Any]], feature_names: list[str], policy: RidgePolicy, threshold: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    x = make_matrix(rows, feature_names)
    mean = np.array(policy.mean, dtype=float)
    std = np.array(policy.std, dtype=float)
    weights = np.array(policy.weights, dtype=float)
    pred = predict(standardize_eval(x, mean, std), weights)
    t = policy.threshold if threshold is None else threshold
    active = np.isfinite(pred) & (pred <= t)
    summary = summarize_local(rows, active)
    summary.update({"l2": policy.l2, "threshold": float(t), "pred_mean": float(np.nanmean(pred)), "pred_std": float(np.nanstd(pred))})
    return active, summary


def top_weights(policy: RidgePolicy, feature_names: list[str], k: int = 12) -> list[dict[str, Any]]:
    weights = np.array(policy.weights[1:], dtype=float)
    order = np.argsort(np.abs(weights))[::-1]
    return [{"feature": feature_names[i], "weight": float(weights[i])} for i in order[:k]]


def add_summary(summaries: list[dict[str, Any]], group: str, policy: str, rows: list[dict[str, Any]], active: np.ndarray, **extra: Any) -> None:
    summary = summarize_local(rows, active)
    summary.update({"group": group, "policy": policy})
    summary.update(extra)
    summaries.append(summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--l2-grid", default="0.001,0.01,0.1,1,10,100,1000")
    p.add_argument("--min-finite-frac", type=float, default=0.95)
    args = p.parse_args()

    rows = read_rows(args.samples)
    if not rows:
        raise SystemExit("no samples")
    feature_names = infer_feature_names(rows, args.min_finite_frac)
    l2_values = [float(x) for x in args.l2_grid.split(",") if x]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    top_weight_groups: dict[str, list[dict[str, Any]]] = {}
    datasets = sorted({row["dataset"] for row in rows})
    groups: list[tuple[str, list[dict[str, Any]]]] = [("pooled", rows)]
    groups.extend((d, [r for r in rows if r["dataset"] == d]) for d in datasets)

    for name, group_rows in groups:
        delta = np.array([safe_float(r["delta_mse"]) for r in group_rows], dtype=float)
        add_summary(summaries, name, "all_off", group_rows, np.zeros(len(group_rows), dtype=bool))
        add_summary(summaries, name, "all_on", group_rows, np.ones(len(group_rows), dtype=bool))
        add_summary(summaries, name, "oracle_local", group_rows, delta < 0.0)
        policy = fit_policy(group_rows, feature_names, l2_values)
        top_weight_groups[name] = top_weights(policy, feature_names)
        _, train_summary = eval_policy(group_rows, feature_names, policy)
        train_summary.update({"group": name, "policy": "same_table_ridge_train_threshold"})
        summaries.append(train_summary)
        _, zero_summary = eval_policy(group_rows, feature_names, policy, threshold=0.0)
        zero_summary.update({"group": name, "policy": "same_table_ridge_zero_threshold"})
        summaries.append(zero_summary)

    for held in datasets:
        train = [r for r in rows if r["dataset"] != held]
        eval_rows = [r for r in rows if r["dataset"] == held]
        policy = fit_policy(train, feature_names, l2_values)
        top_weight_groups[f"lodo_{held}"] = top_weights(policy, feature_names)
        _, train_summary = eval_policy(eval_rows, feature_names, policy)
        train_summary.update({"group": held, "policy": "leave_dataset_out_ridge_train_threshold", "train_group": "+".join(sorted({r["dataset"] for r in train}))})
        summaries.append(train_summary)
        _, zero_summary = eval_policy(eval_rows, feature_names, policy, threshold=0.0)
        zero_summary.update({"group": held, "policy": "leave_dataset_out_ridge_zero_threshold", "train_group": "+".join(sorted({r["dataset"] for r in train}))})
        summaries.append(zero_summary)

    image_groups = sorted({(row["dataset"], row["image"]) for row in rows})
    active_by_row = np.zeros(len(rows), dtype=bool)
    row_index = {(row["dataset"], row["image"], row["slice_id"], row["flat_index"]): i for i, row in enumerate(rows)}
    for dataset, image in image_groups:
        eval_rows = [r for r in rows if r["dataset"] == dataset and r["image"] == image]
        train = [r for r in rows if not (r["dataset"] == dataset and r["image"] == image)]
        policy = fit_policy(train, feature_names, l2_values)
        active, _ = eval_policy(eval_rows, feature_names, policy)
        for r, act in zip(eval_rows, active):
            idx = row_index[(r["dataset"], r["image"], r["slice_id"], r["flat_index"])]
            active_by_row[idx] = bool(act)
    add_summary(summaries, "pooled", "leave_image_out_ridge_train_threshold", rows, active_by_row)

    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(summary_csv, summaries)

    payload = {
        "samples": len(rows),
        "features": feature_names,
        "l2_grid": l2_values,
        "summaries": summaries,
        "top_weights": top_weight_groups,
        "interpretation": "Ridge diagnostic on E221 spatial local quant-MSE labels; not a deployed codec controller.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E222 EF-LIC Spatial Linear Controller Probe",
        "",
        "This consumes E221 local quant-MSE labels and fits a ridge score predictor using only decoder-reproducible sample features.",
        "Negative `dMSE` means selected active HCG positions reduce normalized quantization MSE versus fixed EF-LIC RVQ.",
        "",
        f"Samples: `{len(rows)}`",
        f"Features: `{len(feature_names)}`",
        "",
        "| group | policy | samples | dMSE | all-on dMSE | oracle dMSE | active | helpful | precision | recall | l2 | threshold | train |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row.get('group', '')} | {row.get('policy', '')} | {int(row.get('samples', 0))} | "
            f"{float(row.get('delta_mse', 0.0)):+.8f} | {float(row.get('all_on_delta_mse', 0.0)):+.8f} | "
            f"{float(row.get('oracle_delta_mse', 0.0)):+.8f} | {float(row.get('active_frac', 0.0)):.3f} | "
            f"{float(row.get('helpful_frac', 0.0)):.3f} | {float(row.get('precision', 0.0)):.3f} | "
            f"{float(row.get('recall', 0.0)):.3f} | {row.get('l2', '')} | {row.get('threshold', '')} | {row.get('train_group', '')} |"
        )
    lines.extend(["", "Top absolute ridge weights:", ""])
    for group, weights in top_weight_groups.items():
        joined = ", ".join(f"{w['feature']}={w['weight']:+.3g}" for w in weights[:8])
        lines.append(f"- {group}: {joined}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Same-table rows are capacity diagnostics and should not be used as final claims.",
            "- Leave-dataset-out and leave-image-out rows are the useful transfer checks for deciding whether local map features are already enough for a deterministic controller.",
            "- If transfer remains much weaker than the oracle, the next method should train a nonlinear/local HCG head with codec loss instead of hard-coding a linear post-hoc policy.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
