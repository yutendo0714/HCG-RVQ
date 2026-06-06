#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_rows(specs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"input must be dataset=csv, got {spec!r}")
        dataset, csv_path = spec.split("=", 1)
        with Path(csv_path).open() as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["dataset"] = dataset
                rows.append(row)
    return rows


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    d_dists = np.array([f(r, "delta_dists") for r in rows], dtype=float)
    d_lpips = np.array([f(r, "delta_lpips") for r in rows], dtype=float)
    d_psnr = np.array([f(r, "delta_psnr") for r in rows], dtype=float)
    score = d_dists + 3.0 * d_lpips
    y_mismatch = np.array([f(r, "y_mismatch") for r in rows], dtype=float)
    y_total = np.array([f(r, "y_total") for r in rows], dtype=float)
    return {
        "images": n,
        "delta_dists": float(d_dists.mean()),
        "delta_lpips": float(d_lpips.mean()),
        "delta_psnr": float(d_psnr.mean()),
        "score_dists_3lpips": float(score.mean()),
        "dists_win_frac": float((d_dists < 0).mean()),
        "lpips_win_frac": float((d_lpips < 0).mean()),
        "both_win_frac": float(((d_dists < 0) & (d_lpips < 0)).mean()),
        "nonfinite_rows": int(sum(int(float(r["nonfinite"])) for r in rows)),
        "max_decode_diff": float(max(f(r, "max_decode_diff") for r in rows)),
        "y_mismatch_frac": float(y_mismatch.sum() / max(1.0, y_total.sum())),
        "geometry_delta_rms": float(np.mean([f(r, "y_avg_geometry_delta_rms") for r in rows])),
    }


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[(row["dataset"], row["image"])].append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in by_image.values():
        selected.append(min(image_rows, key=lambda r: f(r, "delta_dists") + 3.0 * f(r, "delta_lpips")))
    out = summarize_group(selected)
    out["active_nonzero_frac"] = float(np.mean([any(abs(float(v)) > 0 for v in r["alpha_values"].split(",")) for r in selected]))
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[row["alpha_schedule"]] += 1
    out["selected_schedules"] = dict(sorted(counts.items()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="dataset=csv")
    p.add_argument("--output-prefix", type=Path, required=True)
    args = p.parse_args()

    rows = read_rows(args.input)
    grouped: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        drows = [r for r in rows if r["dataset"] == dataset]
        for schedule in sorted({r["alpha_schedule"] for r in drows}):
            subset = [r for r in drows if r["alpha_schedule"] == schedule]
            summary = summarize_group(subset)
            summary.update({"dataset": dataset, "alpha_schedule": schedule, "alpha_values": subset[0]["alpha_values"]})
            grouped.append(summary)
        oracle = oracle_summary(drows)
        oracle.update({"dataset": dataset, "alpha_schedule": "oracle_by_score", "alpha_values": "per-image"})
        grouped.append(oracle)

    for schedule in sorted({r["alpha_schedule"] for r in rows}):
        subset = [r for r in rows if r["alpha_schedule"] == schedule]
        summary = summarize_group(subset)
        summary.update({"dataset": "pooled", "alpha_schedule": schedule, "alpha_values": subset[0]["alpha_values"]})
        grouped.append(summary)
    pooled_oracle = oracle_summary(rows)
    pooled_oracle.update({"dataset": "pooled", "alpha_schedule": "oracle_by_score", "alpha_values": "per-image"})
    grouped.append(pooled_oracle)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = [
        "dataset", "alpha_schedule", "alpha_values", "images", "delta_dists", "delta_lpips", "score_dists_3lpips",
        "delta_psnr", "dists_win_frac", "lpips_win_frac", "both_win_frac", "y_mismatch_frac", "max_decode_diff",
        "nonfinite_rows", "geometry_delta_rms", "active_nonzero_frac", "selected_schedules",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(grouped)
    json_path.write_text(json.dumps({"rows": grouped}, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E217 EF-LIC Slice Alpha Schedule Probe Summary",
        "",
        "This aggregates fixed decoder-reproducible per-slice HCG strength schedules. Positive deltas are worse; `score` is `dDISTS + 3*dLPIPS`.",
        "",
        "| dataset | schedule | alpha values | images | dDISTS | dLPIPS | score | dPSNR | DISTS win | LPIPS win | both win | y mismatch | decode max | nonfinite |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            f"| {row['dataset']} | {row['alpha_schedule']} | {row['alpha_values']} | {row['images']} | "
            f"{row['delta_dists']:+.6f} | {row['delta_lpips']:+.6f} | {row['score_dists_3lpips']:+.6f} | "
            f"{row['delta_psnr']:+.5f} | {row['dists_win_frac']:.3f} | {row['lpips_win_frac']:.3f} | "
            f"{row['both_win_frac']:.3f} | {row['y_mismatch_frac']:.6f} | {row['max_decode_diff']:.1e} | {row['nonfinite_rows']} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- The schedules are bitstream-valid diagnostics: the same fixed slice strengths are known to encoder and decoder, so no side bits are added.",
        "- If different datasets prefer different slice locations, the next method should be local/context-conditioned rather than a single global fixed schedule.",
        "- The per-image oracle is an upper bound for a future decoder-safe local controller, not a publishable method by itself.",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
