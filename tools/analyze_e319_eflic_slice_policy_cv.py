#!/usr/bin/env python3
"""Cross-validate simple EF-LIC HCG slice policies from powerset sweeps.

E317/E318 expose per-image slice-subset outcomes and slice-level features. This
diagnostic asks whether a very small, explainable controller can recover useful
headroom without using the held-out image outcome. It is intentionally an
offline policy audit, not final deployable evidence.
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
ALL_SLICES = (0, 1, 2, 3)


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
        default=ROOT / "experiments/analysis/e319_eflic_slice_policy_cv_kodak24",
    )
    p.add_argument("--min-train-active-frac", type=float, default=0.0)
    return p.parse_args()


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def label_for_slices(slices: set[int]) -> str:
    if not slices:
        return "none"
    if slices == set(ALL_SLICES):
        return "all"
    return ",".join(str(s) for s in sorted(slices))


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def candidate_features(label_rows: list[dict[str, str]]) -> list[str]:
    if not label_rows:
        return []
    excluded = {
        "image",
        "slice",
        "all_delta_psnr",
        "single_delta_psnr",
        "leave_one_out_delta_psnr",
        "contextual_margin_psnr",
        "oracle_active",
        "single_positive",
        "contextual_positive",
        "best_slice_set",
        "best_delta_psnr",
        "best_gain_over_all",
        "single_vs_context_agree",
        "oracle_vs_single_agree",
        "oracle_vs_context_agree",
    }
    out: list[str] = []
    for key in sorted(label_rows[0]):
        if key in excluded:
            continue
        values = [fnum(r, key) for r in label_rows]
        finite = [v for v in values if math.isfinite(v)]
        if len(finite) >= 8 and float(np.std(finite)) > 0.0:
            out.append(key)
    return out


def build_delta_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        lookup[(row["image"], row["active_slices"])] = fnum(row, "delta_psnr")
    return lookup


def predict_label(rows: list[dict[str, str]], feature: str, direction: str, threshold: float) -> str:
    active: set[int] = set()
    for row in rows:
        value = fnum(row, feature)
        if not math.isfinite(value):
            continue
        if (direction == "le" and value <= threshold) or (direction == "ge" and value >= threshold):
            active.add(int(row["slice"]))
    return label_for_slices(active)


def score_policy(
    images: list[str],
    rows_by_image: dict[str, list[dict[str, str]]],
    deltas: dict[tuple[str, str], float],
    feature: str,
    direction: str,
    threshold: float,
) -> dict[str, float]:
    vals: list[float] = []
    gains: list[float] = []
    active_fracs: list[float] = []
    for image in images:
        label = predict_label(rows_by_image[image], feature, direction, threshold)
        delta = deltas[(image, label)]
        vals.append(delta)
        gains.append(delta - deltas[(image, "all")])
        if label == "none":
            active_fracs.append(0.0)
        elif label == "all":
            active_fracs.append(1.0)
        else:
            active_fracs.append(len(label.split(",")) / 4.0)
    return {
        "mean_delta_psnr": mean(vals),
        "worst_delta_psnr": min(vals) if vals else float("nan"),
        "mean_gain_over_all": mean(gains),
        "mean_active_frac": mean(active_fracs),
    }


def train_threshold(
    train_images: list[str],
    rows_by_image: dict[str, list[dict[str, str]]],
    deltas: dict[tuple[str, str], float],
    features: list[str],
    min_train_active_frac: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in features:
        values = [fnum(row, feature) for image in train_images for row in rows_by_image[image]]
        finite = sorted({v for v in values if math.isfinite(v)})
        if not finite:
            continue
        thresholds = finite
        if len(thresholds) > 128:
            quantiles = np.linspace(0.0, 1.0, 128)
            thresholds = sorted({float(np.quantile(np.asarray(finite), q)) for q in quantiles})
        for direction in ("le", "ge"):
            for threshold in thresholds:
                score = score_policy(train_images, rows_by_image, deltas, feature, direction, threshold)
                if score["mean_active_frac"] < min_train_active_frac:
                    continue
                row: dict[str, Any] = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    **{f"train_{k}": v for k, v in score.items()},
                }
                key = (
                    row["train_mean_delta_psnr"],
                    row["train_worst_delta_psnr"],
                    -abs(row["train_mean_active_frac"] - 0.5),
                )
                if best is None:
                    best = row
                else:
                    best_key = (
                        best["train_mean_delta_psnr"],
                        best["train_worst_delta_psnr"],
                        -abs(best["train_mean_active_frac"] - 0.5),
                    )
                    if key > best_key:
                        best = row
    if best is None:
        raise RuntimeError("no threshold candidate selected")
    return best


def main() -> None:
    args = parse_args()
    sweep_rows = read_csv(args.rows)
    label_rows = read_csv(args.slice_labels)
    features = candidate_features(label_rows)
    rows_by_image: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        rows_by_image.setdefault(row["image"], []).append(row)
    images = sorted(rows_by_image)
    deltas = build_delta_lookup(sweep_rows)

    fold_rows: list[dict[str, Any]] = []
    for heldout in images:
        train_images = [image for image in images if image != heldout]
        policy = train_threshold(train_images, rows_by_image, deltas, features, args.min_train_active_frac)
        predicted_label = predict_label(rows_by_image[heldout], policy["feature"], policy["direction"], policy["threshold"])
        predicted_delta = deltas[(heldout, predicted_label)]
        all_delta = deltas[(heldout, "all")]
        none_delta = deltas[(heldout, "none")]
        oracle_label, oracle_delta = max(
            ((label, delta) for (image, label), delta in deltas.items() if image == heldout),
            key=lambda item: item[1],
        )
        fold_rows.append(
            {
                "image": heldout,
                "selected_feature": policy["feature"],
                "selected_direction": policy["direction"],
                "selected_threshold": policy["threshold"],
                "predicted_slice_set": predicted_label,
                "predicted_delta_psnr": predicted_delta,
                "all_delta_psnr": all_delta,
                "none_delta_psnr": none_delta,
                "oracle_slice_set": oracle_label,
                "oracle_delta_psnr": oracle_delta,
                "gain_over_all": predicted_delta - all_delta,
                "oracle_gap": oracle_delta - predicted_delta,
                "train_mean_delta_psnr": policy["train_mean_delta_psnr"],
                "train_worst_delta_psnr": policy["train_worst_delta_psnr"],
                "train_mean_active_frac": policy["train_mean_active_frac"],
            }
        )

    summary = [
        {
            "policy": "all",
            "images": len(images),
            "mean_delta_psnr": mean([deltas[(image, "all")] for image in images]),
            "worst_delta_psnr": min(deltas[(image, "all")] for image in images),
            "mean_gain_over_all": 0.0,
            "mean_oracle_gap": mean([
                max(delta for (img, _), delta in deltas.items() if img == image) - deltas[(image, "all")]
                for image in images
            ]),
        },
        {
            "policy": "none",
            "images": len(images),
            "mean_delta_psnr": 0.0,
            "worst_delta_psnr": 0.0,
            "mean_gain_over_all": mean([0.0 - deltas[(image, "all")] for image in images]),
            "mean_oracle_gap": mean([
                max(delta for (img, _), delta in deltas.items() if img == image)
                for image in images
            ]),
        },
        {
            "policy": "loocv_threshold",
            "images": len(images),
            "mean_delta_psnr": mean([fnum(r, "predicted_delta_psnr") for r in fold_rows]),
            "worst_delta_psnr": min(fnum(r, "predicted_delta_psnr") for r in fold_rows),
            "mean_gain_over_all": mean([fnum(r, "gain_over_all") for r in fold_rows]),
            "mean_oracle_gap": mean([fnum(r, "oracle_gap") for r in fold_rows]),
        },
        {
            "policy": "best_per_image_oracle",
            "images": len(images),
            "mean_delta_psnr": mean([
                max(delta for (img, _), delta in deltas.items() if img == image) for image in images
            ]),
            "worst_delta_psnr": min([
                max(delta for (img, _), delta in deltas.items() if img == image) for image in images
            ]),
            "mean_gain_over_all": mean([
                max(delta for (img, _), delta in deltas.items() if img == image) - deltas[(image, "all")]
                for image in images
            ]),
            "mean_oracle_gap": 0.0,
        },
    ]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fold_csv = args.output_prefix.with_suffix(".folds.csv")
    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    for path, rows in ((fold_csv, fold_rows), (summary_csv, summary)):
        with path.open("w", newline="") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "source_rows": str(args.rows),
        "source_slice_labels": str(args.slice_labels),
        "features": features,
        "folds": fold_rows,
        "summary": summary,
    }
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# E319 EF-LIC Slice Policy LOOCV",
        "",
        "This is an offline controller-feasibility diagnostic from E317/E318 powerset rows.",
        "",
        "## Summary",
        "",
        "| policy | images | mean_delta_psnr | worst_delta_psnr | mean_gain_over_all | mean_oracle_gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['images']} | {row['mean_delta_psnr']:+.6f} | "
            f"{row['worst_delta_psnr']:+.6f} | {row['mean_gain_over_all']:+.6f} | {row['mean_oracle_gap']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The LOOCV policy selects one feature/direction/threshold on the other 23 images.",
            "- It can choose `none`, preserving the EF-LIC fixed payload when HCG is predicted unsafe.",
            "- This is not final RD evidence; it is a bridge from oracle headroom to a trainable controller.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {fold_csv}, {summary_csv}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
