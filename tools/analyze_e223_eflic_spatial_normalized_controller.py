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
from analyze_e222_eflic_spatial_linear_controller import EXCLUDED_FEATURES, read_rows  # noqa: E402


@dataclass(frozen=True)
class RidgePolicy:
    l2: float
    threshold: float
    train_delta_mse: float
    weights: list[float]
    mean: list[float]
    std: list[float]
    threshold_objective: str


def infer_feature_names(rows: list[dict[str, Any]], min_finite_frac: float) -> list[str]:
    names: list[str] = []
    for key in rows[0].keys():
        if key in EXCLUDED_FEATURES:
            continue
        vals = np.array([safe_float(row.get(key)) for row in rows], dtype=float)
        if np.isfinite(vals).mean() >= min_finite_frac:
            names.append(key)
    return names


def group_key(row: dict[str, Any], mode: str) -> tuple[str, ...]:
    if mode == "none":
        return ("all",)
    if mode == "image":
        return (str(row["dataset"]), str(row["image"]))
    if mode == "image_slice":
        return (str(row["dataset"]), str(row["image"]), str(row["slice_id"]))
    raise ValueError(f"unknown group mode: {mode}")


def base_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.array([[safe_float(row.get(f)) for f in feature_names] for row in rows], dtype=float)


def relative_matrix(rows: list[dict[str, Any]], x: np.ndarray, group_mode: str) -> np.ndarray:
    if group_mode == "none":
        return x.copy()
    out = np.zeros_like(x, dtype=float)
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(group_key(row, group_mode), []).append(i)
    global_mean = np.nanmean(x, axis=0)
    global_mean = np.where(np.isfinite(global_mean), global_mean, 0.0)
    for idxs in groups.values():
        idx = np.array(idxs, dtype=int)
        sub = x[idx]
        mean = np.nanmean(sub, axis=0)
        mean = np.where(np.isfinite(mean), mean, global_mean)
        filled = np.where(np.isfinite(sub), sub, mean[None, :])
        std = filled.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        out[idx] = (filled - mean[None, :]) / std[None, :]
    return out


def make_design(rows: list[dict[str, Any]], feature_names: list[str], mode: str) -> tuple[np.ndarray, list[str]]:
    x = base_matrix(rows, feature_names)
    if mode == "raw":
        return x, feature_names
    if mode == "image_rel":
        return relative_matrix(rows, x, "image"), [f"image_rel:{f}" for f in feature_names]
    if mode == "image_slice_rel":
        return relative_matrix(rows, x, "image_slice"), [f"image_slice_rel:{f}" for f in feature_names]
    if mode == "raw_plus_image_rel":
        rel = relative_matrix(rows, x, "image")
        return np.concatenate([x, rel], axis=1), feature_names + [f"image_rel:{f}" for f in feature_names]
    if mode == "raw_plus_image_slice_rel":
        rel = relative_matrix(rows, x, "image_slice")
        return np.concatenate([x, rel], axis=1), feature_names + [f"image_slice_rel:{f}" for f in feature_names]
    raise ValueError(f"unknown feature mode: {mode}")


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


def best_threshold(rows: list[dict[str, Any]], pred: np.ndarray, objective: str) -> tuple[float, float, float]:
    deltas = np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)
    finite = np.isfinite(pred) & np.isfinite(deltas)
    if not finite.any():
        return -float("inf"), 0.0, 0.0

    values = pred[finite]
    delta_values = deltas[finite]
    dataset_values = np.array([row["dataset"] for row, ok in zip(rows, finite) if ok], dtype=object)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_deltas = delta_values[order]
    sorted_datasets = dataset_values[order]
    unique, counts = np.unique(sorted_values, return_counts=True)
    cum_counts = np.cumsum(counts)
    cum_deltas = np.cumsum(sorted_deltas)

    if unique.size == 1:
        thresholds = np.array([unique[0] - 1e-12, unique[0] + 1e-12], dtype=float)
    else:
        mids = (unique[:-1] + unique[1:]) * 0.5
        thresholds = np.concatenate(([unique[0] - 1e-12], mids, [unique[-1] + 1e-12]))

    active_counts = np.concatenate(([0], cum_counts))
    if objective == "pooled":
        active_scores = np.concatenate(([0.0], cum_deltas[cum_counts - 1])) / max(1, len(rows))
    elif objective == "dataset_mean":
        per_dataset_scores = []
        all_datasets = sorted({row["dataset"] for row in rows})
        for dataset in all_datasets:
            group_total = max(1, sum(1 for row in rows if row["dataset"] == dataset))
            group_cum = np.cumsum(np.where(sorted_datasets == dataset, sorted_deltas, 0.0))
            group_scores = np.concatenate(([0.0], group_cum[cum_counts - 1])) / group_total
            per_dataset_scores.append(group_scores)
        active_scores = np.mean(np.stack(per_dataset_scores, axis=0), axis=0)
    else:
        raise ValueError(f"unknown threshold objective: {objective}")

    best_idx = int(np.argmin(active_scores))
    best_score = float(active_scores[best_idx])
    if best_score >= -1e-15:
        return -float("inf"), 0.0, 0.0
    return float(thresholds[best_idx]), best_score, float(active_counts[best_idx]) / max(1, len(rows))


def fit_policy(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    feature_mode: str,
    l2_values: list[float],
    threshold_objective: str,
) -> RidgePolicy:
    x, _ = make_design(rows, feature_names, feature_mode)
    y = np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)
    x_std, mean, std = standardize_train(x)
    best: RidgePolicy | None = None
    for l2 in l2_values:
        weights = fit_weights(x_std, y, l2)
        pred = predict(x_std, weights)
        threshold, train_score, _ = best_threshold(rows, pred, threshold_objective)
        if best is None or train_score < best.train_delta_mse - 1e-15:
            best = RidgePolicy(
                l2=float(l2),
                threshold=float(threshold),
                train_delta_mse=float(train_score),
                weights=[float(v) for v in weights],
                mean=[float(v) for v in mean],
                std=[float(v) for v in std],
                threshold_objective=threshold_objective,
            )
    assert best is not None
    return best


def eval_policy(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    feature_mode: str,
    policy: RidgePolicy,
    threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    x, _ = make_design(rows, feature_names, feature_mode)
    mean = np.array(policy.mean, dtype=float)
    std = np.array(policy.std, dtype=float)
    weights = np.array(policy.weights, dtype=float)
    pred = predict(standardize_eval(x, mean, std), weights)
    t = policy.threshold if threshold is None else threshold
    active = np.isfinite(pred) & (pred <= t)
    summary = summarize_local(rows, active)
    summary.update(
        {
            "l2": policy.l2,
            "threshold": float(t),
            "threshold_objective": policy.threshold_objective,
            "pred_mean": float(np.nanmean(pred)),
            "pred_std": float(np.nanstd(pred)),
        }
    )
    return active, summary, pred


def add_summary(summaries: list[dict[str, Any]], group: str, policy: str, rows: list[dict[str, Any]], active: np.ndarray, **extra: Any) -> None:
    summary = summarize_local(rows, active)
    summary.update({"group": group, "policy": policy})
    summary.update(extra)
    summaries.append(summary)


def top_weights(policy: RidgePolicy, design_names: list[str], k: int = 10) -> list[dict[str, Any]]:
    weights = np.array(policy.weights[1:], dtype=float)
    order = np.argsort(np.abs(weights))[::-1]
    return [{"feature": design_names[i], "weight": float(weights[i])} for i in order[: min(k, len(order))]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--l2-grid", default="0.001,0.01,0.1,1,10,100,1000")
    p.add_argument("--feature-modes", default="raw,image_slice_rel,raw_plus_image_slice_rel,raw_plus_image_rel")
    p.add_argument("--threshold-objectives", default="pooled,dataset_mean")
    p.add_argument("--min-finite-frac", type=float, default=0.95)
    args = p.parse_args()

    rows = read_rows(args.samples)
    if not rows:
        raise SystemExit("no samples")
    feature_names = infer_feature_names(rows, args.min_finite_frac)
    l2_values = [float(x) for x in args.l2_grid.split(",") if x]
    feature_modes = [x for x in args.feature_modes.split(",") if x]
    threshold_objectives = [x for x in args.threshold_objectives.split(",") if x]
    datasets = sorted({row["dataset"] for row in rows})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    weight_payload: dict[str, list[dict[str, Any]]] = {}
    delta = np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)
    add_summary(summaries, "pooled", "all_off", rows, np.zeros(len(rows), dtype=bool))
    add_summary(summaries, "pooled", "all_on", rows, np.ones(len(rows), dtype=bool))
    add_summary(summaries, "pooled", "oracle_local", rows, delta < 0.0)

    for feature_mode in feature_modes:
        _, design_names = make_design(rows, feature_names, feature_mode)
        for objective in threshold_objectives:
            policy = fit_policy(rows, feature_names, feature_mode, l2_values, objective)
            active, summary, _ = eval_policy(rows, feature_names, feature_mode, policy)
            policy_name = f"same_table_ridge_{feature_mode}_{objective}"
            summary.update({"group": "pooled", "policy": policy_name, "feature_mode": feature_mode})
            summaries.append(summary)
            weight_payload[policy_name] = top_weights(policy, design_names)

            active_by_row = np.zeros(len(rows), dtype=bool)
            row_index = {(row["dataset"], row["image"], row["slice_id"], row["flat_index"]): i for i, row in enumerate(rows)}
            for held in datasets:
                train = [r for r in rows if r["dataset"] != held]
                eval_rows = [r for r in rows if r["dataset"] == held]
                held_policy = fit_policy(train, feature_names, feature_mode, l2_values, objective)
                held_active, held_summary, _ = eval_policy(eval_rows, feature_names, feature_mode, held_policy)
                held_summary.update(
                    {
                        "group": held,
                        "policy": f"leave_dataset_out_ridge_{feature_mode}_{objective}",
                        "feature_mode": feature_mode,
                        "train_group": "+".join(sorted({r["dataset"] for r in train})),
                    }
                )
                summaries.append(held_summary)

            for dataset, image in sorted({(row["dataset"], row["image"]) for row in rows}):
                eval_rows = [r for r in rows if r["dataset"] == dataset and r["image"] == image]
                train = [r for r in rows if not (r["dataset"] == dataset and r["image"] == image)]
                image_policy = fit_policy(train, feature_names, feature_mode, l2_values, objective)
                image_active, _, _ = eval_policy(eval_rows, feature_names, feature_mode, image_policy)
                for r, act in zip(eval_rows, image_active):
                    idx = row_index[(r["dataset"], r["image"], r["slice_id"], r["flat_index"])]
                    active_by_row[idx] = bool(act)
            add_summary(
                summaries,
                "pooled",
                f"leave_image_out_ridge_{feature_mode}_{objective}",
                rows,
                active_by_row,
                feature_mode=feature_mode,
                threshold_objective=objective,
            )

    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(summary_csv, summaries)
    payload = {
        "samples": len(rows),
        "base_features": feature_names,
        "feature_modes": feature_modes,
        "threshold_objectives": threshold_objectives,
        "summaries": summaries,
        "top_weights": weight_payload,
        "interpretation": "Tests whether relative normalization and dataset-balanced thresholding rescue E222 transfer.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E223 EF-LIC Spatial Normalized Controller Probe",
        "",
        "This is a diagnostic over E221 spatial labels. Negative dMSE means selected projected-HCG positions reduce normalized quantization MSE.",
        "It tests whether image/slice-relative features or dataset-balanced thresholding rescue E222 transfer.",
        "",
        f"Samples: `{len(rows)}`",
        f"Base features: `{len(feature_names)}`",
        "",
        "| group | policy | samples | dMSE | all-on dMSE | oracle dMSE | active | helpful | precision | recall | l2 | threshold | objective | train |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {group} | {policy} | {samples} | {delta_mse:+.8f} | {all_on_delta_mse:+.8f} | {oracle_delta_mse:+.8f} | {active_frac:.3f} | {helpful_frac:.3f} | {precision:.3f} | {recall:.3f} | {l2} | {threshold} | {threshold_objective} | {train_group} |".format(
                group=row.get("group", ""),
                policy=row.get("policy", ""),
                samples=int(row.get("samples", 0)),
                delta_mse=float(row.get("delta_mse", 0.0)),
                all_on_delta_mse=float(row.get("all_on_delta_mse", 0.0)),
                oracle_delta_mse=float(row.get("oracle_delta_mse", 0.0)),
                active_frac=float(row.get("active_frac", 0.0)),
                helpful_frac=float(row.get("helpful_frac", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                l2=row.get("l2", ""),
                threshold=row.get("threshold", ""),
                threshold_objective=row.get("threshold_objective", ""),
                train_group=row.get("train_group", ""),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Same-table rows remain capacity diagnostics.",
            "- Leave-dataset-out and leave-image-out rows decide whether normalized decoder-safe features are sufficient for a deterministic hand controller.",
            "- If transfer remains harmful, the next method should train the local HCG head with the codec objective rather than post-hoc labels.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
