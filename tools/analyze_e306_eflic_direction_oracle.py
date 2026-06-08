#!/usr/bin/env python3
"""Analyze EF-LIC HCG direction-source headroom.

E303-E305 showed that the HCG branch is sensitive to the geometry direction
source (`mean`, `logscale`, `fixed`). This script aggregates matched codec-loop
CSV rows and reports both single-direction performance and the oracle headroom
of a decoder-safe direction/fallback selector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_GROUPS = {
    "kodak24": {
        "mean": ROOT / "experiments/analysis/e305_eflic_hcg_e296_threshold095_mean_kodak24.csv",
        "logscale": ROOT / "experiments/analysis/e302_eflic_hcg_e296_threshold095_logscale_kodak24.csv",
        "fixed": ROOT / "experiments/analysis/e304_eflic_hcg_e296_threshold095_fixed_kodak24.csv",
    },
    "clicpro16": {
        "mean": ROOT / "experiments/analysis/e305_eflic_hcg_e296_threshold095_mean_clicpro16.csv",
        "logscale": ROOT / "experiments/analysis/e302_eflic_hcg_e296_threshold095_logscale_clicpro16.csv",
        "fixed": ROOT / "experiments/analysis/e304_eflic_hcg_e296_threshold095_fixed_clicpro16.csv",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e306_eflic_direction_oracle",
    )
    return p.parse_args()


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(vals: list[float]) -> dict[str, float]:
    return {
        "mean_delta_psnr": mean(vals),
        "worst_delta_psnr": min(vals),
        "best_delta_psnr": max(vals),
        "win_frac": mean([float(v > 0.0) for v in vals]),
        "nonnegative_frac": mean([float(v >= 0.0) for v in vals]),
    }


def load_direction_rows(files: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    by_image: dict[str, dict[str, dict[str, Any]]] = {}
    for direction, path in files.items():
        with path.open(newline="", encoding="utf-8") as fobj:
            for row in csv.DictReader(fobj):
                if row.get("mode") != "trained_hard":
                    continue
                image = row["image"]
                by_image.setdefault(image, {})[direction] = row
    missing = {
        image: sorted(set(files) - set(rows))
        for image, rows in by_image.items()
        if set(rows) != set(files)
    }
    if missing:
        raise RuntimeError(f"direction rows are not matched: {missing}")
    return by_image


def analyze_dataset(dataset: str, files: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_image = load_direction_rows(files)
    per_image: list[dict[str, Any]] = []
    direction_summary: dict[str, Any] = {}
    for direction in files:
        vals = [float(rows[direction]["delta_psnr"]) for rows in by_image.values()]
        direction_summary[direction] = summarize(vals)

    oracle_vals: list[float] = []
    oracle_nonfallback_vals: list[float] = []
    choice_counts: dict[str, int] = {}
    choice_nonfallback_counts: dict[str, int] = {}
    for image, rows in sorted(by_image.items()):
        deltas = {direction: float(row["delta_psnr"]) for direction, row in rows.items()}
        with_fallback = {"fallback": 0.0, **deltas}
        best_choice = max(with_fallback, key=with_fallback.get)
        best_nonfallback = max(deltas, key=deltas.get)
        oracle_vals.append(with_fallback[best_choice])
        oracle_nonfallback_vals.append(deltas[best_nonfallback])
        choice_counts[best_choice] = choice_counts.get(best_choice, 0) + 1
        choice_nonfallback_counts[best_nonfallback] = choice_nonfallback_counts.get(best_nonfallback, 0) + 1
        ref = next(iter(rows.values()))
        per_image.append(
            {
                "dataset": dataset,
                "image": image,
                **{f"{direction}_delta_psnr": deltas[direction] for direction in files},
                "best_with_fallback": best_choice,
                "best_with_fallback_delta_psnr": with_fallback[best_choice],
                "best_nonfallback": best_nonfallback,
                "best_nonfallback_delta_psnr": deltas[best_nonfallback],
                "gate_mean": float(ref.get("y_gate_mean", 0.0)),
                "alpha_mean": float(ref.get("y_alpha_mean", 0.0)),
                "delta_bpp": float(ref.get("delta_bpp", 0.0)),
                "nonfinite": int(float(ref.get("nonfinite", 0.0))),
            }
        )
    summary = {
        "dataset": dataset,
        "images": len(by_image),
        "directions": direction_summary,
        "oracle_with_fallback": summarize(oracle_vals) | {"choice_counts": choice_counts},
        "oracle_nonfallback": summarize(oracle_nonfallback_vals) | {"choice_counts": choice_nonfallback_counts},
    }
    return summary, per_image


def write_outputs(args: argparse.Namespace, summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    with csv_path.open("w", newline="", encoding="utf-8") as fobj:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "E306 EF-LIC HCG direction oracle",
        "purpose": "Quantify matched mean/logscale/fixed direction headroom and fallback-selector upper bound.",
        "summaries": summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E306 EF-LIC HCG Direction Oracle\n\n")
        fobj.write(
            "This is an oracle/headroom analysis over matched codec-loop CSVs. "
            "It is not a deployable selector; it tells us whether a decoder-safe "
            "direction/fallback selector is worth implementing.\n\n"
        )
        for summary in summaries:
            fobj.write(f"## {summary['dataset']}\n\n")
            fobj.write("| policy | mean dPSNR | worst | best | win | nonnegative | choices |\n")
            fobj.write("|---|---:|---:|---:|---:|---:|---|\n")
            for name, stats in summary["directions"].items():
                fobj.write(
                    f"| {name} | {stats['mean_delta_psnr']:+.6f} | {stats['worst_delta_psnr']:+.6f} | "
                    f"{stats['best_delta_psnr']:+.6f} | {stats['win_frac']:.6f} | "
                    f"{stats['nonnegative_frac']:.6f} | - |\n"
                )
            for name in ("oracle_with_fallback", "oracle_nonfallback"):
                stats = summary[name]
                fobj.write(
                    f"| {name} | {stats['mean_delta_psnr']:+.6f} | {stats['worst_delta_psnr']:+.6f} | "
                    f"{stats['best_delta_psnr']:+.6f} | {stats['win_frac']:.6f} | "
                    f"{stats['nonnegative_frac']:.6f} | {stats['choice_counts']} |\n"
                )
            fobj.write("\n")
        fobj.write("Interpretation:\n\n")
        fobj.write(
            "- If `oracle_with_fallback` is near zero, a direction selector is mainly a safety mechanism.\n"
            "- If it is meaningfully above the best single direction, learning direction/fallback is a useful next target.\n"
            "- Any practical selector must be decoder-safe or explicitly signaled and charged.\n"
        )
    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for dataset, files in DEFAULT_GROUPS.items():
        summary, dataset_rows = analyze_dataset(dataset, files)
        summaries.append(summary)
        rows.extend(dataset_rows)
    write_outputs(args, summaries, rows)


if __name__ == "__main__":
    main()
