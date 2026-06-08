#!/usr/bin/env python3
"""Summarize E325 controller codec-loop evaluations against E317 headroom.

E323-E325 established that stale spatial labels should not be paper-main and
that an E318-aligned controller is trainable. E326/E327 then evaluated that
controller in the actual EF-LIC loop. This script turns those smoke/eval runs
into a compact, reusable audit table.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--controller-csv",
        type=Path,
        nargs="+",
        default=[
            ROOT / "experiments/analysis/e326_eflic_e325_controller_codec_eval_kodak24_eval8_thr050.csv",
            ROOT / "experiments/analysis/e327_eflic_e325_controller_codec_eval_kodak24_eval8_thr025.csv",
        ],
    )
    p.add_argument(
        "--oracle-by-image",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.by_image.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e328_eflic_e325_controller_vs_e317_headroom_eval8",
    )
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def summarize_rows(rows: list[dict[str, str]], *, label: str) -> dict[str, Any]:
    deltas = [f(r, "delta_psnr") for r in rows]
    dbpps = [f(r, "delta_bpp") for r in rows]
    alpha = [f(r, "y_alpha_mean") for r in rows]
    gate = [f(r, "y_gate_mean") for r in rows]
    geom = [f(r, "y_avg_geometry_delta_rms") for r in rows]
    mismatches = [f(r, "y_mismatch") for r in rows]
    nonfinite = sum(int(float(r.get("nonfinite", "0") or 0)) for r in rows)
    payload_equal = [int(float(r.get("payload_equal", "0") or 0)) for r in rows]
    return {
        "label": label,
        "images": len(rows),
        "mean_delta_psnr": mean(deltas) if deltas else 0.0,
        "worst_delta_psnr": min(deltas) if deltas else 0.0,
        "best_delta_psnr": max(deltas) if deltas else 0.0,
        "positive_images": sum(1 for v in deltas if v > 0),
        "negative_images": sum(1 for v in deltas if v < 0),
        "zero_images": sum(1 for v in deltas if v == 0),
        "mean_delta_bpp": mean(dbpps) if dbpps else 0.0,
        "mean_alpha": mean(alpha) if alpha else 0.0,
        "mean_gate": mean(gate) if gate else 0.0,
        "mean_geometry_delta_rms": mean(geom) if geom else 0.0,
        "mean_y_mismatch": mean(mismatches) if mismatches else 0.0,
        "payload_equal_frac": mean(payload_equal) if payload_equal else 0.0,
        "nonfinite_rows": nonfinite,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    oracle_rows = read_rows(args.oracle_by_image)
    oracle_by_image = {r["image"]: r for r in oracle_rows}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    per_image_rows: list[dict[str, Any]] = []
    eval_images: set[str] = set()
    for csv_path in args.controller_csv:
        rows = read_rows(csv_path)
        run_label = csv_path.stem.replace("eflic_e325_controller_codec_eval_kodak24_eval8_", "")
        for row in rows:
            mode = row["mode"]
            if mode == "force_zero":
                continue
            image = row["image"]
            eval_images.add(image)
            label = f"{run_label}:{mode}"
            grouped[label].append(row)
            oracle = oracle_by_image.get(image)
            if oracle is None:
                continue
            per_image_rows.append(
                {
                    "label": label,
                    "image": image,
                    "controller_delta_psnr": f(row, "delta_psnr"),
                    "controller_delta_bpp": f(row, "delta_bpp"),
                    "controller_alpha_mean": f(row, "y_alpha_mean"),
                    "controller_gate_mean": f(row, "y_gate_mean"),
                    "controller_y_mismatch": f(row, "y_mismatch"),
                    "e317_all_delta_psnr": f(oracle, "all_delta_psnr"),
                    "e317_best_delta_psnr": f(oracle, "best_delta_psnr"),
                    "e317_best_slice_set": oracle.get("best_slice_set", ""),
                    "e317_best_gain_over_all": f(oracle, "best_gain_over_all"),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for label, rows in sorted(grouped.items()):
        summary_rows.append(summarize_rows(rows, label=label))

    eval_oracle = [oracle_by_image[i] for i in sorted(eval_images) if i in oracle_by_image]
    all_mean = mean([f(r, "all_delta_psnr") for r in eval_oracle]) if eval_oracle else 0.0
    best_mean = mean([f(r, "best_delta_psnr") for r in eval_oracle]) if eval_oracle else 0.0
    none_mean = 0.0
    headroom = best_mean - none_mean
    all_vs_none = all_mean - none_mean
    for row in summary_rows:
        row["e317_eval_all_mean"] = all_mean
        row["e317_eval_best_mean"] = best_mean
        row["e317_eval_headroom"] = headroom
        row["controller_over_e317_all"] = row["mean_delta_psnr"] - all_mean
        row["recovered_headroom_frac"] = (row["mean_delta_psnr"] / headroom) if headroom else 0.0
        row["e317_all_recovered_frac"] = (all_vs_none / headroom) if headroom else 0.0

    summary_fields = [
        "label",
        "images",
        "mean_delta_psnr",
        "worst_delta_psnr",
        "best_delta_psnr",
        "positive_images",
        "negative_images",
        "zero_images",
        "mean_delta_bpp",
        "mean_alpha",
        "mean_gate",
        "mean_geometry_delta_rms",
        "mean_y_mismatch",
        "payload_equal_frac",
        "nonfinite_rows",
        "e317_eval_all_mean",
        "e317_eval_best_mean",
        "e317_eval_headroom",
        "controller_over_e317_all",
        "recovered_headroom_frac",
        "e317_all_recovered_frac",
    ]
    per_image_fields = [
        "label",
        "image",
        "controller_delta_psnr",
        "controller_delta_bpp",
        "controller_alpha_mean",
        "controller_gate_mean",
        "controller_y_mismatch",
        "e317_all_delta_psnr",
        "e317_best_delta_psnr",
        "e317_best_slice_set",
        "e317_best_gain_over_all",
    ]
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary_rows, summary_fields)
    write_csv(args.output_prefix.with_suffix(".per_image.csv"), per_image_rows, per_image_fields)

    payload = {
        "controller_csv": [str(p) for p in args.controller_csv],
        "oracle_by_image": str(args.oracle_by_image),
        "eval_images": sorted(eval_images),
        "e317_eval_all_mean": all_mean,
        "e317_eval_best_mean": best_mean,
        "e317_eval_headroom": headroom,
        "summary": summary_rows,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# EF-LIC Controller vs E317 Headroom",
        "",
        "This audit compares controller codec-loop runs against the E317 powerset headroom on the same image set.",
        "",
        f"- E317 all-on mean delta PSNR: `{all_mean:+.8f}`",
        f"- E317 best powerset oracle mean delta PSNR: `{best_mean:+.8f}`",
        f"- E317 best-vs-fallback headroom: `{headroom:+.8f}`",
        "",
        "| label | mean dPSNR | worst dPSNR | positive | negative | alpha mean | gate mean | payload equal | recovered headroom | vs E317 all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            "| {label} | {mean_delta_psnr:+.8f} | {worst_delta_psnr:+.8f} | {positive_images} | {negative_images} | "
            "{mean_alpha:.8f} | {mean_gate:.8f} | {payload_equal_frac:.8f} | {recovered_headroom_frac:.8f} | {controller_over_e317_all:+.8f} |".format(
                **row
            )
        )
    md.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The E325 hard controller at threshold 0.5 recovers a small positive mean while preserving exact fixed-length bpp and decode agreement, but it is far below the E317 oracle headroom.",
            "- Lowering the threshold to 0.25 increases activation but worsens both mean and tail, so the problem is not simply under-activation.",
            "- The next controller should improve selectivity from decoder-available local/sequential context rather than adding many auxiliary losses.",
            "",
        ]
    )
    args.output_prefix.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
