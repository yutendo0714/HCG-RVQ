#!/usr/bin/env python3
"""Cross-validate conservative EF-LIC HCG image-level all/none gates.

E319 shows that slice-level single-threshold policies are too brittle. This
diagnostic asks a simpler question: can decoder-available summary features at
least decide whether to use all HCG slices or exact fallback for an image?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.rows.csv",
    )
    p.add_argument(
        "--slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e320_eflic_image_gate_cv_kodak24",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def candidate_slice_features(label_rows: list[dict[str, str]]) -> list[str]:
    if not label_rows:
        return []
    prefixes = ("allctx_", "single_")
    excluded = {
        "single_delta_psnr",
        "single_positive",
        "single_vs_context_agree",
    }
    out: list[str] = []
    for key in sorted(label_rows[0]):
        if not key.startswith(prefixes) or key in excluded:
            continue
        values = [fnum(row, key) for row in label_rows]
        finite = [v for v in values if math.isfinite(v)]
        if len(finite) >= 8 and float(np.std(finite)) > 0.0:
            out.append(key)
    return out


def aggregate_features(label_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        by_image.setdefault(row["image"], []).append(row)
    features = candidate_slice_features(label_rows)
    out: dict[str, dict[str, float]] = {}
    for image, rows in by_image.items():
        row_out: dict[str, float] = {}
        for feature in features:
            vals = [fnum(row, feature) for row in rows]
            finite = [v for v in vals if math.isfinite(v)]
            if not finite:
                continue
            row_out[f"{feature}_mean"] = mean(finite)
            row_out[f"{feature}_min"] = min(finite)
            row_out[f"{feature}_max"] = max(finite)
            row_out[f"{feature}_std"] = std(finite)
        out[image] = row_out
    return out


def build_delta_maps(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    all_delta: dict[str, float] = {}
    none_delta: dict[str, float] = {}
    full_oracle: dict[str, float] = {}
    for row in rows:
        image = row["image"]
        label = row["active_slices"]
        delta = fnum(row, "delta_psnr")
        if label == "all":
            all_delta[image] = delta
        if label == "none":
            none_delta[image] = delta
        full_oracle[image] = max(delta, full_oracle.get(image, -float("inf")))
    return all_delta, none_delta, full_oracle


def policy_delta(image: str, active: bool, all_delta: dict[str, float], none_delta: dict[str, float]) -> float:
    return all_delta[image] if active else none_delta[image]


def score(
    images: list[str],
    feature_by_image: dict[str, dict[str, float]],
    all_delta: dict[str, float],
    none_delta: dict[str, float],
    feature: str,
    direction: str,
    threshold: float,
) -> dict[str, float]:
    vals: list[float] = []
    active: list[float] = []
    for image in images:
        value = feature_by_image[image][feature]
        use_all = value <= threshold if direction == "le" else value >= threshold
        vals.append(policy_delta(image, use_all, all_delta, none_delta))
        active.append(float(use_all))
    return {
        "mean_delta_psnr": mean(vals),
        "worst_delta_psnr": min(vals),
        "active_frac": mean(active),
    }


def train(
    images: list[str],
    feature_by_image: dict[str, dict[str, float]],
    all_delta: dict[str, float],
    none_delta: dict[str, float],
) -> dict[str, Any]:
    features = sorted({k for image in images for k in feature_by_image[image]})
    best: dict[str, Any] | None = None
    for feature in features:
        values = sorted({feature_by_image[image][feature] for image in images if math.isfinite(feature_by_image[image][feature])})
        for direction in ("le", "ge"):
            for threshold in values:
                scored = score(images, feature_by_image, all_delta, none_delta, feature, direction, threshold)
                row: dict[str, Any] = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    **{f"train_{k}": v for k, v in scored.items()},
                }
                key = (
                    row["train_mean_delta_psnr"],
                    row["train_worst_delta_psnr"],
                    -abs(row["train_active_frac"] - 0.5),
                )
                if best is None:
                    best = row
                else:
                    best_key = (
                        best["train_mean_delta_psnr"],
                        best["train_worst_delta_psnr"],
                        -abs(best["train_active_frac"] - 0.5),
                    )
                    if key > best_key:
                        best = row
    if best is None:
        raise RuntimeError("no image gate candidate selected")
    return best


def main() -> None:
    args = parse_args()
    sweep_rows = read_csv(args.rows)
    label_rows = read_csv(args.slice_labels)
    features = aggregate_features(label_rows)
    all_delta, none_delta, full_oracle = build_delta_maps(sweep_rows)
    images = sorted(all_delta)

    folds: list[dict[str, Any]] = []
    for heldout in images:
        train_images = [image for image in images if image != heldout]
        policy = train(train_images, features, all_delta, none_delta)
        value = features[heldout][policy["feature"]]
        use_all = value <= policy["threshold"] if policy["direction"] == "le" else value >= policy["threshold"]
        delta = policy_delta(heldout, use_all, all_delta, none_delta)
        best_all_none = max(all_delta[heldout], none_delta[heldout])
        folds.append(
            {
                "image": heldout,
                "feature": policy["feature"],
                "direction": policy["direction"],
                "threshold": policy["threshold"],
                "feature_value": value,
                "predicted_policy": "all" if use_all else "none",
                "predicted_delta_psnr": delta,
                "all_delta_psnr": all_delta[heldout],
                "none_delta_psnr": none_delta[heldout],
                "best_all_none_delta_psnr": best_all_none,
                "full_oracle_delta_psnr": full_oracle[heldout],
                "gain_over_all": delta - all_delta[heldout],
                "gap_to_best_all_none": best_all_none - delta,
                "gap_to_full_oracle": full_oracle[heldout] - delta,
                "train_mean_delta_psnr": policy["train_mean_delta_psnr"],
                "train_worst_delta_psnr": policy["train_worst_delta_psnr"],
                "train_active_frac": policy["train_active_frac"],
            }
        )

    summary = [
        {
            "policy": "all",
            "images": len(images),
            "mean_delta_psnr": mean([all_delta[i] for i in images]),
            "worst_delta_psnr": min(all_delta[i] for i in images),
            "mean_gain_over_all": 0.0,
            "mean_gap_to_full_oracle": mean([full_oracle[i] - all_delta[i] for i in images]),
        },
        {
            "policy": "best_all_none_oracle",
            "images": len(images),
            "mean_delta_psnr": mean([max(all_delta[i], none_delta[i]) for i in images]),
            "worst_delta_psnr": min(max(all_delta[i], none_delta[i]) for i in images),
            "mean_gain_over_all": mean([max(all_delta[i], none_delta[i]) - all_delta[i] for i in images]),
            "mean_gap_to_full_oracle": mean([full_oracle[i] - max(all_delta[i], none_delta[i]) for i in images]),
        },
        {
            "policy": "loocv_image_gate",
            "images": len(images),
            "mean_delta_psnr": mean([fnum(r, "predicted_delta_psnr") for r in folds]),
            "worst_delta_psnr": min(fnum(r, "predicted_delta_psnr") for r in folds),
            "mean_gain_over_all": mean([fnum(r, "gain_over_all") for r in folds]),
            "mean_gap_to_full_oracle": mean([fnum(r, "gap_to_full_oracle") for r in folds]),
        },
        {
            "policy": "full_subset_oracle",
            "images": len(images),
            "mean_delta_psnr": mean([full_oracle[i] for i in images]),
            "worst_delta_psnr": min(full_oracle[i] for i in images),
            "mean_gain_over_all": mean([full_oracle[i] - all_delta[i] for i in images]),
            "mean_gap_to_full_oracle": 0.0,
        },
    ]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    folds_csv = args.output_prefix.with_suffix(".folds.csv")
    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    for path, rows in ((folds_csv, folds), (summary_csv, summary)):
        with path.open("w", newline="") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps({"folds": folds, "summary": summary}, indent=2), encoding="utf-8")

    md_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# E320 EF-LIC Image-Level Gate LOOCV",
        "",
        "This tests a conservative all/none gate before learning slice-level control.",
        "",
        "| policy | images | mean_delta_psnr | worst_delta_psnr | mean_gain_over_all | mean_gap_to_full_oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['images']} | {row['mean_delta_psnr']:+.6f} | "
            f"{row['worst_delta_psnr']:+.6f} | {row['mean_gain_over_all']:+.6f} | "
            f"{row['mean_gap_to_full_oracle']:+.6f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {folds_csv}, {summary_csv}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
