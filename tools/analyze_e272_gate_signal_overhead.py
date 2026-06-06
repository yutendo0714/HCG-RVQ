#!/usr/bin/env python3
"""E272 gate-signal overhead audit for GLC rate-cap soft rows.

E271 shows that capped soft/progressive output remains useful while capped
all-on remains harmful.  The remaining question is how much gate side information
can be paid before that soft benefit disappears.  This script audits simple
signaling profiles over existing E271 per-image rows.  It is intentionally an
accounting audit, not a final entropy-coded codec.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--labels", nargs="*", default=["trained_rate_cap_soft"])
    p.add_argument("--include-unselected-overhead", action="store_true")
    return p.parse_args()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def profile_bits(name: str, h: int, w: int) -> float:
    if name == "none":
        return 0.0
    if name == "scalar8":
        return 8.0
    if name == "scalar16":
        return 16.0
    if name == "tile64_1bit":
        return float(math.ceil(h / 64) * math.ceil(w / 64))
    if name == "tile32_1bit":
        return float(math.ceil(h / 32) * math.ceil(w / 32))
    if name == "tile16_1bit":
        return float(math.ceil(h / 16) * math.ceil(w / 16))
    if name == "tile32_2bit":
        return 2.0 * float(math.ceil(h / 32) * math.ceil(w / 32))
    if name == "tile16_2bit":
        return 2.0 * float(math.ceil(h / 16) * math.ceil(w / 16))
    raise KeyError(name)


PROFILES = [
    "none",
    "scalar8",
    "scalar16",
    "tile64_1bit",
    "tile32_1bit",
    "tile16_1bit",
    "tile32_2bit",
    "tile16_2bit",
]


def source_name(path: Path) -> str:
    stem = path.stem
    if "clic" in stem.lower():
        return "clic"
    if "kodak" in stem.lower():
        return "kodak"
    return stem


def read_rows(paths: list[Path], labels: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("label") not in labels:
                    continue
                h = int(finite(row.get("height"), 0.0))
                w = int(finite(row.get("width"), 0.0))
                selected = int(finite(row.get("selected"), 0.0) > 0.0)
                rows.append(
                    {
                        "source": source_name(path),
                        "input": str(path),
                        "label": row.get("label", ""),
                        "image": row.get("image", ""),
                        "height": h,
                        "width": w,
                        "pixels": float(max(1, h * w)),
                        "selected": selected,
                        "bpp": finite(row.get("bpp"), float("nan")),
                        "delta_bpp": finite(row.get("delta_bpp"), float("nan")),
                        "score": finite(row.get("score"), float("nan")),
                        "delta_psnr": finite(row.get("delta_psnr"), float("nan")),
                        "delta_ms_ssim": finite(row.get("delta_ms_ssim"), float("nan")),
                        "delta_lpips": finite(row.get("delta_lpips"), float("nan")),
                        "delta_dists": finite(row.get("delta_dists"), float("nan")),
                        "gate_mean": finite(row.get("gate_mean"), float("nan")),
                        "active_mse_ratio": finite(row.get("active_mse_ratio"), float("nan")),
                        "index_entropy_mean": finite(row.get("index_entropy_mean"), float("nan")),
                    }
                )
    return rows


def expand_profiles(rows: list[dict[str, Any]], include_unselected_overhead: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for profile in PROFILES:
            charge = row["selected"] or include_unselected_overhead
            bits = profile_bits(profile, int(row["height"]), int(row["width"])) if charge else 0.0
            overhead_bpp = bits / row["pixels"]
            out.append(
                {
                    **row,
                    "overhead_profile": profile,
                    "gate_bits": bits,
                    "gate_overhead_bpp": overhead_bpp,
                    "adjusted_bpp": row["bpp"] + overhead_bpp,
                    "adjusted_delta_bpp": row["delta_bpp"] + overhead_bpp,
                    "adjusted_score": row["score"] + overhead_bpp,
                    "adjusted_win": int((row["score"] + overhead_bpp) < 0.0),
                }
            )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    sources = sorted({row["source"] for row in rows}) + ["all"]
    profiles = PROFILES
    for source in sources:
        source_rows = rows if source == "all" else [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        for profile in profiles:
            subset = [row for row in source_rows if row["overhead_profile"] == profile]
            if not subset:
                continue
            summaries.append(
                {
                    "source": source,
                    "overhead_profile": profile,
                    "images": len(subset),
                    "selected_frac": mean([float(row["selected"]) for row in subset]),
                    "base_score": mean([float(row["score"]) for row in subset]),
                    "gate_overhead_bpp": mean([float(row["gate_overhead_bpp"]) for row in subset]),
                    "adjusted_score": mean([float(row["adjusted_score"]) for row in subset]),
                    "adjusted_win_frac": mean([float(row["adjusted_win"]) for row in subset]),
                    "max_gate_overhead_bpp": max(float(row["gate_overhead_bpp"]) for row in subset),
                    "delta_psnr": mean([float(row["delta_psnr"]) for row in subset]),
                    "delta_ms_ssim": mean([float(row["delta_ms_ssim"]) for row in subset]),
                    "delta_lpips": mean([float(row["delta_lpips"]) for row in subset]),
                    "delta_dists": mean([float(row["delta_dists"]) for row in subset]),
                }
            )
    return summaries


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], expanded: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in expanded for key in row})
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(expanded)
    payload = {
        "experiment": "E272 GLC gate-signal overhead audit",
        "note": "Accounting audit over E271 rows. Not a final entropy-coded codec.",
        "inputs": [str(path) for path in args.inputs],
        "labels": args.labels,
        "profiles": PROFILES,
        "source_rows": rows,
        "summary": summaries,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E272 GLC Gate-Signal Overhead Audit",
        "",
        "Accounting audit over E271 `rate_cap_soft` rows. It asks how much gate side information can be charged before the selected soft/progressive benefit disappears.",
        "This is not a final entropy-coded codec; it is a design check for the next selected/progressive implementation.",
        "",
        "| source | overhead | images | selected | base score | gate bpp | adjusted score | win | max gate bpp | dPSNR | dMS-SSIM | dLPIPS | dDISTS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['source']} | {row['overhead_profile']} | {row['images']} | {row['selected_frac']:.3f} | "
            f"{row['base_score']:+.6f} | {row['gate_overhead_bpp']:.6f} | {row['adjusted_score']:+.6f} | "
            f"{row['adjusted_win_frac']:.3f} | {row['max_gate_overhead_bpp']:.6f} | {row['delta_psnr']:+.6f} | "
            f"{row['delta_ms_ssim']:+.6f} | {row['delta_lpips']:+.6f} | {row['delta_dists']:+.6f} |"
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs, set(args.labels))
    if not rows:
        raise SystemExit("no matching rows")
    expanded = expand_profiles(rows, args.include_unselected_overhead)
    summaries = summarize(expanded)
    write_outputs(args, rows, expanded, summaries)


if __name__ == "__main__":
    main()
