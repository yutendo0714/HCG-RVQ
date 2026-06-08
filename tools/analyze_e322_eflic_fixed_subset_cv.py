#!/usr/bin/env python3
"""LOOCV fixed-subset baseline for EF-LIC HCG powerset rows.

This complements E321. If a learned controller fails, we still need to know
whether a train-selected fixed slice subset is a better conservative baseline
than all-on activation, and whether it fixes the hard negative tail.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
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
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e322_eflic_fixed_subset_cv_kodak24",
    )
    p.add_argument(
        "--tail-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="Train objective = mean + tail_weight * worst.",
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


def summarize(policy: str, images: list[str], deltas: dict[tuple[str, str], float], values: list[float]) -> dict[str, Any]:
    full_oracle = {
        image: max(delta for (img, _), delta in deltas.items() if img == image)
        for image in images
    }
    return {
        "policy": policy,
        "images": len(images),
        "mean_delta_psnr": mean(values),
        "worst_delta_psnr": min(values),
        "mean_gain_over_all": mean([v - deltas[(image, "all")] for v, image in zip(values, images, strict=True)]),
        "mean_gap_to_full_oracle": mean([full_oracle[image] - v for v, image in zip(values, images, strict=True)]),
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.rows)
    images = sorted({row["image"] for row in rows})
    labels = sorted({row["active_slices"] for row in rows})
    deltas = {(row["image"], row["active_slices"]): fnum(row, "delta_psnr") for row in rows}
    full_oracle = {
        image: max(delta for (img, _), delta in deltas.items() if img == image)
        for image in images
    }

    folds: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = [
        summarize("all", images, deltas, [deltas[(image, "all")] for image in images]),
        summarize(
            "best_all_none_oracle",
            images,
            deltas,
            [max(deltas[(image, "all")], deltas[(image, "none")]) for image in images],
        ),
        summarize("full_subset_oracle", images, deltas, [full_oracle[image] for image in images]),
    ]

    for tail_weight in args.tail_weights:
        values: list[float] = []
        selected: list[str] = []
        for heldout in images:
            train_images = [image for image in images if image != heldout]
            best_label = max(
                labels,
                key=lambda label: (
                    mean([deltas[(image, label)] for image in train_images])
                    + float(tail_weight) * min(deltas[(image, label)] for image in train_images),
                    mean([deltas[(image, label)] for image in train_images]),
                    min(deltas[(image, label)] for image in train_images),
                ),
            )
            delta = deltas[(heldout, best_label)]
            values.append(delta)
            selected.append(best_label)
            folds.append(
                {
                    "tail_weight": tail_weight,
                    "image": heldout,
                    "selected_slice_set": best_label,
                    "predicted_delta_psnr": delta,
                    "all_delta_psnr": deltas[(heldout, "all")],
                    "none_delta_psnr": deltas[(heldout, "none")],
                    "full_oracle_delta_psnr": full_oracle[heldout],
                    "gain_over_all": delta - deltas[(heldout, "all")],
                    "gap_to_full_oracle": full_oracle[heldout] - delta,
                }
            )
        row = summarize(f"loocv_fixed_subset_tail{tail_weight:g}", images, deltas, values)
        row["selected_counts"] = dict(Counter(selected))
        summary.append(row)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.output_prefix.with_suffix(".folds.csv"): folds,
        args.output_prefix.with_suffix(".summary.csv"): summary,
    }
    for path, out_rows in outputs.items():
        with path.open("w", newline="") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=sorted({k for row in out_rows for k in row}))
            writer.writeheader()
            writer.writerows(out_rows)

    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "folds": folds}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# E322 EF-LIC Fixed-Subset LOOCV",
        "",
        "Train-fold fixed subset selection from E317 powerset rows.",
        "",
        "| policy | images | mean_delta_psnr | worst_delta_psnr | mean_gain_over_all | mean_gap_to_full_oracle | selected_counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['images']} | {row['mean_delta_psnr']:.6f} | "
            f"{row['worst_delta_psnr']:.6f} | {row['mean_gain_over_all']:.6f} | "
            f"{row['mean_gap_to_full_oracle']:.6f} | `{row.get('selected_counts', '')}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is not a deployable local controller. It tests whether a fixed subset chosen without the held-out image is a stronger baseline than all-on activation.",
            "",
        ]
    )
    md_path = args.output_prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
