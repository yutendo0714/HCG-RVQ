#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SINGLE_SLICE_SCHEDULES = {f"slice{i}only020": i for i in range(4)}
GLOBAL_FEATURE_KEYS = [
    "z_hat_abs_mean",
    "z_hat_mean",
    "z_hat_std",
    "z_hat_rms",
    "z_hat_min",
    "z_hat_max",
    "z_index_entropy",
    "z_index_perplexity",
    "z_index_used_frac",
    "z_index_max_prob",
]
SLICE_STAT_SUFFIXES = [
    "mean_abs_mean",
    "mean_mean",
    "mean_std",
    "mean_rms",
    "mean_min",
    "mean_max",
    "scale_abs_mean",
    "scale_mean",
    "scale_std",
    "scale_rms",
    "scale_min",
    "scale_max",
]


@dataclass(frozen=True)
class Stump:
    feature: str
    threshold: float
    polarity: str
    train_score: float
    train_active_frac: float


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def read_rows(specs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--input must be dataset=csv, got {spec!r}")
        dataset, csv_path = spec.split("=", 1)
        with Path(csv_path).open() as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["dataset"] = dataset
                rows.append(row)
    return rows


def build_samples(rows: list[dict[str, Any]], lpips_weight: float) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    feature_names = ["slice_id", "is_slice0", "is_slice1", "is_slice2", "is_slice3"]
    feature_names.extend(GLOBAL_FEATURE_KEYS)
    for suffix in SLICE_STAT_SUFFIXES:
        feature_names.append(f"local_{suffix}")
    feature_names.extend(
        [
            "local_scale_over_mean_rms",
            "local_scale_over_mean_abs",
            "local_scale_std_over_mean_std",
        ]
    )

    for row in rows:
        schedule = row.get("alpha_schedule", "")
        if schedule not in SINGLE_SLICE_SCHEDULES:
            continue
        sid = SINGLE_SLICE_SCHEDULES[schedule]
        d_dists = to_float(row["delta_dists"])
        d_lpips = to_float(row["delta_lpips"])
        d_psnr = to_float(row["delta_psnr"])
        score = d_dists + lpips_weight * d_lpips
        sample: dict[str, Any] = {
            "dataset": row["dataset"],
            "image": row["image"],
            "slice_id": sid,
            "alpha_schedule": schedule,
            "alpha_values": row["alpha_values"],
            "delta_dists": d_dists,
            "delta_lpips": d_lpips,
            "delta_psnr": d_psnr,
            "score": score,
            "helpful": int(score < 0.0),
            "both_helpful": int(d_dists < 0.0 and d_lpips < 0.0),
            "dists_helpful": int(d_dists < 0.0),
            "lpips_helpful": int(d_lpips < 0.0),
            "nonfinite": int(to_float(row.get("nonfinite", 0.0))),
            "max_decode_diff": to_float(row.get("max_decode_diff", 0.0)),
        }
        sample["slice_id"] = float(sid)
        for i in range(4):
            sample[f"is_slice{i}"] = 1.0 if sid == i else 0.0
        for key in GLOBAL_FEATURE_KEYS:
            sample[key] = to_float(row.get(key))
        for suffix in SLICE_STAT_SUFFIXES:
            sample[f"local_{suffix}"] = to_float(row.get(f"slice{sid}_{suffix}"))

        mean_rms = abs(to_float(sample.get("local_mean_rms")))
        mean_abs = abs(to_float(sample.get("local_mean_abs_mean")))
        mean_std = abs(to_float(sample.get("local_mean_std")))
        scale_rms = abs(to_float(sample.get("local_scale_rms")))
        scale_abs = abs(to_float(sample.get("local_scale_abs_mean")))
        scale_std = abs(to_float(sample.get("local_scale_std")))
        sample["local_scale_over_mean_rms"] = scale_rms / max(mean_rms, 1e-8)
        sample["local_scale_over_mean_abs"] = scale_abs / max(mean_abs, 1e-8)
        sample["local_scale_std_over_mean_std"] = scale_std / max(mean_std, 1e-8)
        samples.append(sample)
    return samples, feature_names


def summarize_policy(rows: list[dict[str, Any]], active: np.ndarray) -> dict[str, Any]:
    scores = np.array([to_float(r["score"]) for r in rows], dtype=float)
    d_dists = np.array([to_float(r["delta_dists"]) for r in rows], dtype=float)
    d_lpips = np.array([to_float(r["delta_lpips"]) for r in rows], dtype=float)
    d_psnr = np.array([to_float(r["delta_psnr"]) for r in rows], dtype=float)
    helpful = scores < 0.0
    active = active.astype(bool)
    selected_score = np.where(active, scores, 0.0)
    selected_dists = np.where(active, d_dists, 0.0)
    selected_lpips = np.where(active, d_lpips, 0.0)
    selected_psnr = np.where(active, d_psnr, 0.0)
    true_pos = active & helpful
    return {
        "samples": len(rows),
        "score": float(selected_score.mean()) if len(rows) else 0.0,
        "delta_dists": float(selected_dists.mean()) if len(rows) else 0.0,
        "delta_lpips": float(selected_lpips.mean()) if len(rows) else 0.0,
        "delta_psnr": float(selected_psnr.mean()) if len(rows) else 0.0,
        "active_frac": float(active.mean()) if len(rows) else 0.0,
        "helpful_frac": float(helpful.mean()) if len(rows) else 0.0,
        "precision": float(true_pos.sum() / max(1, active.sum())),
        "recall": float(true_pos.sum() / max(1, helpful.sum())),
        "both_win_frac_when_active": float(
            np.mean([(to_float(r["delta_dists"]) < 0 and to_float(r["delta_lpips"]) < 0) for r, a in zip(rows, active) if a])
        ) if active.any() else 0.0,
    }


def valid_feature_values(rows: list[dict[str, Any]], feature: str) -> np.ndarray:
    vals = np.array([to_float(r.get(feature)) for r in rows], dtype=float)
    return vals


def evaluate_stump(rows: list[dict[str, Any]], stump: Stump) -> tuple[np.ndarray, dict[str, Any]]:
    vals = valid_feature_values(rows, stump.feature)
    if stump.polarity == "le":
        active = vals <= stump.threshold
    elif stump.polarity == "ge":
        active = vals >= stump.threshold
    else:
        raise ValueError(stump.polarity)
    active = active & np.isfinite(vals)
    return active, summarize_policy(rows, active)


def fit_stump(rows: list[dict[str, Any]], feature_names: list[str]) -> Stump:
    best = Stump("__off__", 0.0, "ge", 0.0, 0.0)
    best_score = 0.0
    scores = np.array([to_float(r["score"]) for r in rows], dtype=float)
    n_rows = max(1, len(rows))
    for feature in feature_names:
        vals = valid_feature_values(rows, feature)
        finite = np.isfinite(vals)
        if not finite.any():
            continue

        finite_vals = vals[finite]
        finite_scores = scores[finite]
        order = np.argsort(finite_vals, kind="mergesort")
        sorted_vals = finite_vals[order]
        sorted_scores = finite_scores[order]
        unique, counts = np.unique(sorted_vals, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_scores = np.cumsum(sorted_scores)
        group_score_sums = cum_scores[cum_counts - 1]
        total_count = int(cum_counts[-1])
        total_score = float(cum_scores[-1])

        if unique.size == 1:
            mids = np.array([], dtype=float)
        else:
            mids = (unique[:-1] + unique[1:]) * 0.5
        thresholds = np.concatenate(([unique[0] - 1e-9], mids, [unique[-1] + 1e-9]))

        # <= candidates: no finite rows, then rows up through each unique value.
        le_sums = np.concatenate(([0.0], group_score_sums))
        le_counts = np.concatenate(([0], cum_counts))
        for threshold, score_sum, active_count in zip(thresholds, le_sums, le_counts):
            score = float(score_sum) / n_rows
            if score < best_score - 1e-15:
                best_score = score
                best = Stump(feature, float(threshold), "le", score, float(active_count) / n_rows)

        # >= candidates: all finite rows, then rows after each unique value, then none.
        ge_sums = np.concatenate(([total_score], total_score - group_score_sums))
        ge_counts = np.concatenate(([total_count], total_count - cum_counts))
        for threshold, score_sum, active_count in zip(thresholds, ge_sums, ge_counts):
            score = float(score_sum) / n_rows
            if score < best_score - 1e-15:
                best_score = score
                best = Stump(feature, float(threshold), "ge", score, float(active_count) / n_rows)
    return best


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def feature_correlations(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    score = np.array([to_float(r["score"]) for r in rows], dtype=float)
    out: list[dict[str, Any]] = []
    for feature in feature_names:
        vals = valid_feature_values(rows, feature)
        out.append(
            {
                "feature": feature,
                "pearson_score": pearson(vals, score),
                "spearman_score": spearman(vals, score),
                "finite_frac": float(np.isfinite(vals).mean()),
            }
        )
    out.sort(key=lambda r: abs(r["spearman_score"]) if math.isfinite(r["spearman_score"]) else -1.0, reverse=True)
    return out


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["dataset"]), str(row["image"]), int(float(row["slice_id"])))


def run_loocv(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    active = []
    selected_features: dict[str, int] = defaultdict(int)
    for idx, row in enumerate(rows):
        train = rows[:idx] + rows[idx + 1 :]
        stump = fit_stump(train, feature_names)
        selected_features[stump.feature] += 1
        act, _ = evaluate_stump([row], stump)
        active.append(bool(act[0]))
    summary = summarize_policy(rows, np.array(active, dtype=bool))
    summary["selected_features"] = dict(sorted(selected_features.items(), key=lambda kv: (-kv[1], kv[0])))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="dataset=csv")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    args = p.parse_args()

    raw_rows = read_rows(args.input)
    samples, feature_names = build_samples(raw_rows, args.lpips_weight)
    if not samples:
        raise SystemExit("no single-slice samples found")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_prefix.with_suffix(".samples.csv")
    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    corr_csv = args.output_prefix.with_suffix(".correlations.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    sample_fields = [
        "dataset",
        "image",
        "slice_id",
        "alpha_schedule",
        "delta_dists",
        "delta_lpips",
        "delta_psnr",
        "score",
        "helpful",
        "both_helpful",
    ] + feature_names
    with sample_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    summaries: list[dict[str, Any]] = []
    datasets = sorted({r["dataset"] for r in samples})
    groups: list[tuple[str, list[dict[str, Any]]]] = [("pooled", samples)]
    groups.extend((d, [r for r in samples if r["dataset"] == d]) for d in datasets)
    for name, rows in groups:
        scores = np.array([to_float(r["score"]) for r in rows], dtype=float)
        policies = {
            "all_off": np.zeros(len(rows), dtype=bool),
            "all_on_single_slice": np.ones(len(rows), dtype=bool),
            "oracle_single_slice": scores < 0.0,
        }
        for policy, active in policies.items():
            summary = summarize_policy(rows, active)
            summary.update({"dataset": name, "policy": policy, "feature": "", "threshold": "", "polarity": ""})
            summaries.append(summary)
        stump = fit_stump(rows, feature_names)
        active, summary = evaluate_stump(rows, stump)
        summary.update(
            {
                "dataset": name,
                "policy": "same_table_best_stump",
                "feature": stump.feature,
                "threshold": f"{stump.threshold:.10g}",
                "polarity": stump.polarity,
            }
        )
        summaries.append(summary)

    loocv = run_loocv(samples, feature_names)
    loocv.update({"dataset": "pooled", "policy": "sample_loocv_stump", "feature": "mixed", "threshold": "", "polarity": ""})
    summaries.append(loocv)

    for held in datasets:
        train = [r for r in samples if r["dataset"] != held]
        eval_rows = [r for r in samples if r["dataset"] == held]
        stump = fit_stump(train, feature_names)
        active, summary = evaluate_stump(eval_rows, stump)
        summary.update(
            {
                "dataset": held,
                "policy": "leave_dataset_out_stump",
                "feature": stump.feature,
                "threshold": f"{stump.threshold:.10g}",
                "polarity": stump.polarity,
                "train_dataset": "+".join(sorted({r["dataset"] for r in train})),
            }
        )
        summaries.append(summary)

    summary_fields = [
        "dataset",
        "policy",
        "samples",
        "score",
        "delta_dists",
        "delta_lpips",
        "delta_psnr",
        "active_frac",
        "helpful_frac",
        "precision",
        "recall",
        "both_win_frac_when_active",
        "feature",
        "threshold",
        "polarity",
        "train_dataset",
        "selected_features",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    correlations = feature_correlations(samples, feature_names)
    with corr_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "pearson_score", "spearman_score", "finite_frac"])
        writer.writeheader()
        writer.writerows(correlations)

    payload = {
        "lpips_weight": args.lpips_weight,
        "inputs": args.input,
        "samples": len(samples),
        "features": feature_names,
        "summaries": summaries,
        "top_correlations": correlations[:20],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E218 EF-LIC Local Slice Predictability Probe",
        "",
        "This probe uses E217 single-slice schedules as marginal labels for a future decoder-safe local controller.",
        f"The objective is `score = dDISTS + {args.lpips_weight:g}*dLPIPS`; negative is better.",
        "Only decoder-safe predecision features are used: z/z-index statistics, slice id, and the current slice mean/scale statistics.",
        "",
        "| dataset | policy | samples | score | dDISTS | dLPIPS | active | helpful | precision | recall | both-active | feature | condition |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        condition = ""
        if row.get("feature"):
            condition = f"{row.get('feature')} {row.get('polarity')} {row.get('threshold')}"
        lines.append(
            f"| {row['dataset']} | {row['policy']} | {row['samples']} | {row['score']:+.6f} | "
            f"{row['delta_dists']:+.6f} | {row['delta_lpips']:+.6f} | {row['active_frac']:.3f} | "
            f"{row['helpful_frac']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['both_win_frac_when_active']:.3f} | {row.get('feature', '')} | {condition} |"
        )
    lines.extend(
        [
            "",
            "Top absolute Spearman correlations with marginal single-slice score:",
            "",
            "| rank | feature | spearman | pearson | finite |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for i, row in enumerate(correlations[:15], 1):
        lines.append(
            f"| {i} | {row['feature']} | {row['spearman_score']:+.4f} | {row['pearson_score']:+.4f} | {row['finite_frac']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `oracle_single_slice` is the marginal upper bound for choosing each slice independently; it is not a deployable codec policy.",
            "- A leave-dataset-out stump checks whether a very small decoder-safe rule transfers between the two E217 domains.",
            "- If same-table stumps help but leave-dataset-out collapses, the next implementation should train a local controller on a larger mixed-domain label set rather than hard-coding a threshold from these 24 images.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
