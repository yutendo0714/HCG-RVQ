#!/usr/bin/env python3
"""Summarize E234 EF-LIC branch-controller scaffold runs.

This keeps E234 honest: fixed preset rows are implementation/ablation evidence,
while the per-image oracle over the same decoder-safe preset vocabulary measures
how much a learned controller could still harvest.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        dataset = "kodak24" if "kodak24" in path.name else "clicpro41" if "clicpro41" in path.name else path.stem
        with path.open(newline="") as fobj:
            for raw in csv.DictReader(fobj):
                row: dict[str, Any] = dict(raw)
                row["dataset"] = dataset
                for key in [
                    "delta_bpp",
                    "max_decode_diff",
                    "nonfinite",
                    "delta_psnr",
                    "delta_dists",
                    "delta_lpips",
                    "score_dists_3lpips",
                    "y_mismatch",
                    "y_total",
                    "y_alpha_mean",
                    "y_alpha_active_frac",
                    "y_avg_geometry_delta_rms",
                ]:
                    row[key] = as_float(raw, key)
                if "score_dists_3lpips" not in raw or raw.get("score_dists_3lpips", "") == "":
                    row["score_dists_3lpips"] = row["delta_dists"] + 3.0 * row["delta_lpips"]
                rows.append(row)
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if abs(float(row["delta_bpp"])) < 1e-12
        and float(row["max_decode_diff"]) == 0.0
        and int(float(row["nonfinite"])) == 0
    ]


def fixed_summary(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    subset = [row for row in rows if dataset == "pooled" or row["dataset"] == dataset]
    for preset in sorted({row["preset"] for row in subset}):
        rows_p = [row for row in subset if row["preset"] == preset]
        if not rows_p:
            continue
        out.append(
            {
                "dataset": dataset,
                "preset": preset,
                "family": rows_p[0]["family"],
                "images": len(rows_p),
                "delta_psnr": mean(rows_p, "delta_psnr"),
                "delta_dists": mean(rows_p, "delta_dists"),
                "delta_lpips": mean(rows_p, "delta_lpips"),
                "score_dists_3lpips": mean(rows_p, "score_dists_3lpips"),
                "y_mismatch_frac": float(sum(row["y_mismatch"] for row in rows_p) / max(1.0, sum(row["y_total"] for row in rows_p))),
                "alpha_mean": mean(rows_p, "y_alpha_mean"),
                "alpha_active_frac": mean(rows_p, "y_alpha_active_frac"),
                "geometry_delta_rms": mean(rows_p, "y_avg_geometry_delta_rms"),
            }
        )
    out.sort(key=lambda item: (item["dataset"], item["score_dists_3lpips"]))
    return out


def oracle_summary(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    subset = [row for row in rows if dataset == "pooled" or row["dataset"] == dataset]
    by_image: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in subset:
        by_image.setdefault((row["dataset"], row["image"]), []).append(row)
    chosen = [min(items, key=lambda row: row["score_dists_3lpips"]) for items in by_image.values()]
    family_counts = Counter(row["family"] for row in chosen)
    preset_counts = Counter(row["preset"] for row in chosen)
    return {
        "dataset": dataset,
        "images": len(chosen),
        "delta_psnr": mean(chosen, "delta_psnr"),
        "delta_dists": mean(chosen, "delta_dists"),
        "delta_lpips": mean(chosen, "delta_lpips"),
        "score_dists_3lpips": mean(chosen, "score_dists_3lpips"),
        "score_win_frac": float(np.mean([row["score_dists_3lpips"] < 0.0 for row in chosen])) if chosen else float("nan"),
        "family_counts": dict(sorted(family_counts.items())),
        "preset_counts": dict(sorted(preset_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            ROOT / "experiments" / "analysis" / "e234_eflic_kodak24_branch_controller_scaffold.csv",
            ROOT / "experiments" / "analysis" / "e234_eflic_clicpro41_branch_controller_scaffold.csv",
        ],
    )
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e234_eflic_branch_controller_scaffold_summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    rows_v = valid_rows(rows)
    datasets = ["pooled"] + sorted({row["dataset"] for row in rows_v})
    fixed = [item for dataset in datasets for item in fixed_summary(rows_v, dataset)]
    oracle = [oracle_summary(rows_v, dataset) for dataset in datasets]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fixed_path = args.output_prefix.with_name(args.output_prefix.name + "_fixed.csv")
    oracle_path = args.output_prefix.with_name(args.output_prefix.name + "_oracle.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    with fixed_path.open("w", newline="") as fobj:
        fieldnames = list(fixed[0].keys()) if fixed else []
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed)

    with oracle_path.open("w", newline="") as fobj:
        fieldnames = ["dataset", "images", "delta_psnr", "delta_dists", "delta_lpips", "score_dists_3lpips", "score_win_frac", "family_counts", "preset_counts"]
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(oracle)

    payload = {
        "experiment": "E234 EF-LIC branch-controller scaffold summary",
        "inputs": [str(path) for path in args.inputs],
        "rows": len(rows),
        "valid_rows": len(rows_v),
        "invalid_rows": len(rows) - len(rows_v),
        "fixed": fixed,
        "oracle": oracle,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with md_path.open("w") as fobj:
        fobj.write("# E234 EF-LIC Branch-Controller Scaffold Summary\n\n")
        fobj.write(f"- Rows: `{len(rows)}`\n")
        fobj.write(f"- Codec-valid rows: `{len(rows_v)}`\n")
        fobj.write(f"- Invalid rows: `{len(rows) - len(rows_v)}`\n\n")
        fobj.write("## Best Fixed Presets\n\n")
        fobj.write("| dataset | preset | family | score | dDISTS | dLPIPS | dPSNR | y_mismatch_frac |\n")
        fobj.write("|---|---|---|---|---|---|---|---|\n")
        for dataset in datasets:
            best = min([item for item in fixed if item["dataset"] == dataset], key=lambda item: item["score_dists_3lpips"])
            fobj.write(
                f"| {dataset} | {best['preset']} | {best['family']} | "
                f"{best['score_dists_3lpips']:+.8f} | {best['delta_dists']:+.8f} | "
                f"{best['delta_lpips']:+.8f} | {best['delta_psnr']:+.8f} | {best['y_mismatch_frac']:.8f} |\n"
            )
        fobj.write("\n## Per-Image Oracle Over E234 Presets\n\n")
        fobj.write("| dataset | images | score | dDISTS | dLPIPS | dPSNR | win_frac | family_counts |\n")
        fobj.write("|---|---|---|---|---|---|---|---|\n")
        for item in oracle:
            fobj.write(
                f"| {item['dataset']} | {item['images']} | {item['score_dists_3lpips']:+.8f} | "
                f"{item['delta_dists']:+.8f} | {item['delta_lpips']:+.8f} | "
                f"{item['delta_psnr']:+.8f} | {item['score_win_frac']:.8f} | `{item['family_counts']}` |\n"
            )
        fobj.write("\nInterpretation: fixed rows are ablations; the oracle is headroom for a trained decoder-safe controller over the same preset vocabulary.\n")

    print(f"wrote {fixed_path}, {oracle_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
