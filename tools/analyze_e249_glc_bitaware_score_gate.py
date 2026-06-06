#!/usr/bin/env python3
"""E249 bit-aware score gate for the GLC tail VQ/HCG track.

This is a paper-safety audit over existing E181 rows.  It asks whether the
decoder-aware q0 tail branch remains attractive once empirical index bpp and
perceptual regressions are charged explicitly.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
SRC = ANALYSIS / "e181_glc_decoder_aware_tail_vq_split_train_q0_oi16_kodak8.csv"
OUT_PREFIX = ANALYSIS / "e249_glc_bitaware_score_gate"
BPP_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def finite(x: float) -> bool:
    return math.isfinite(x)


def mean_delta(rows: list[dict[str, str]], branch_key: str, base_key: str) -> float:
    vals = [f(r, branch_key) - f(r, base_key) for r in rows]
    vals = [v for v in vals if finite(v)]
    return mean(vals) if vals else math.nan


def mean_key(rows: list[dict[str, str]], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if finite(v)]
    return mean(vals) if vals else math.nan


def perceptual_score(delta_dists: float, delta_lpips: float) -> float:
    return delta_dists + 3.0 * delta_lpips


def break_even_bpp_weight(delta_dists: float, delta_lpips: float, delta_bpp: float) -> float:
    base = perceptual_score(delta_dists, delta_lpips)
    if not finite(base) or not finite(delta_bpp) or delta_bpp <= 0:
        return math.inf
    if base >= 0:
        return 0.0
    return -base / delta_bpp


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("label", "")].append(row)

    out = []
    for label, rs in sorted(by_label.items()):
        if not label:
            continue
        delta_bpp = mean_key(rs, "empirical_bpp_delta")
        delta_psnr = mean_delta(rs, "branch_psnr", "base_psnr")
        delta_ms = mean_delta(rs, "branch_ms_ssim", "base_ms_ssim")
        delta_lpips = mean_delta(rs, "branch_lpips", "base_lpips")
        delta_dists = mean_delta(rs, "branch_dists", "base_dists")
        delta_perc = perceptual_score(delta_dists, delta_lpips)
        row: dict[str, Any] = {
            "label": label,
            "images": len(rs),
            "active_mse_ratio": mean_key(rs, "active_mse_ratio"),
            "empirical_bpp_delta": delta_bpp,
            "fixed_bpp_delta": mean_key(rs, "fixed_bpp_delta"),
            "delta_psnr": delta_psnr,
            "delta_ms_ssim": delta_ms,
            "delta_lpips": delta_lpips,
            "delta_dists": delta_dists,
            "delta_dists_plus_3lpips": delta_perc,
            "break_even_bpp_weight": break_even_bpp_weight(delta_dists, delta_lpips, delta_bpp),
            "nonfinite_rows": sum(int(float(r.get("nonfinite") or 0)) for r in rs),
        }
        for weight in BPP_WEIGHTS:
            row[f"score_bppw_{weight:g}"] = delta_perc + weight * delta_bpp
        row["passes_strict"] = bool(
            row["nonfinite_rows"] == 0
            and delta_bpp <= 0.005
            and delta_dists <= 0.0
            and delta_lpips <= 0.0
            and delta_ms >= 0.0
        )
        row["passes_perceptual_tolerant"] = bool(
            row["nonfinite_rows"] == 0
            and delta_bpp <= 0.015
            and delta_dists <= 0.015
            and delta_lpips <= 0.0
            and delta_ms >= 0.0
            and row["break_even_bpp_weight"] >= 1.0
        )
        out.append(row)
    return out


def fmt(x: float) -> str:
    if not finite(x):
        return "NA"
    return f"{x:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E249 GLC Bit-Aware Score Gate",
        "",
        "This audit re-scores the E181 decoder-aware q0 tail VQ split-train rows",
        "with empirical index-rate penalties.  It is not a new full-training row;",
        "it decides what the next GLC implementation must fix before promotion.",
        "",
        "| label | images | active MSE ratio | emp bpp d | PSNR d | MS-SSIM d | LPIPS d | DISTS d | DISTS+3LPIPS d | break-even bpp weight | strict | tolerant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['images']} | {fmt(row['active_mse_ratio'])} | "
            f"{fmt(row['empirical_bpp_delta'])} | {fmt(row['delta_psnr'])} | "
            f"{fmt(row['delta_ms_ssim'])} | {fmt(row['delta_lpips'])} | "
            f"{fmt(row['delta_dists'])} | {fmt(row['delta_dists_plus_3lpips'])} | "
            f"{fmt(row['break_even_bpp_weight'])} | {row['passes_strict']} | "
            f"{row['passes_perceptual_tolerant']} |"
        )
    lines.extend(
        [
            "",
            "## Score Sweep",
            "",
            "Lower is better.  `score = delta_DISTS + 3 * delta_LPIPS + w * delta_bpp`.",
            "",
            "| label | " + " | ".join(f"w={w:g}" for w in BPP_WEIGHTS) + " |",
            "|---|" + "---:|" * len(BPP_WEIGHTS),
        ]
    )
    for row in rows:
        vals = [fmt(row[f"score_bppw_{w:g}"]) for w in BPP_WEIGHTS]
        lines.append(f"| {row['label']} | " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The trained E181 branch is more credible than the initial branch because it",
            "improves the perceptual combined score before bpp is charged.  However, the",
            "break-even bpp weight is only around the low single digits, and strict DISTS",
            "or bpp constraints are not satisfied.  Therefore the next GLC experiment",
            "should add explicit bit-aware/index-aware control rather than scaling this",
            "residual-MSE/perceptual branch as-is.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_csv(SRC)
    summary = summarize(rows)
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_PREFIX.with_suffix(".csv"), summary)
    OUT_PREFIX.with_suffix(".json").write_text(
        json.dumps({"source": str(SRC), "bpp_weights": BPP_WEIGHTS, "summary": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_md(OUT_PREFIX.with_suffix(".md"), summary)
    print(f"wrote {OUT_PREFIX.with_suffix('.md')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.json')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
