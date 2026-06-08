#!/usr/bin/env python3
"""Analyze EF-LIC HCG perceptual metric rows.

This complements PSNR/RD smoke tables with paper-facing perceptual diagnostics.
PSNR is treated as a codec-health signal; DISTS/LPIPS/MS-SSIM are the relevant
low-bitrate generative/perceptual evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    return p.parse_args()


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def finite(x: float) -> bool:
    return math.isfinite(x)


def mean(values: list[float]) -> float:
    vals = [x for x in values if finite(x)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def load_rows(path: Path, lpips_weight: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            dpsnr = f(row, "delta_psnr")
            dms = f(row, "delta_ms_ssim")
            dlpips = f(row, "delta_lpips")
            ddists = f(row, "delta_dists")
            score = ddists + lpips_weight * dlpips
            item = {
                "image": row.get("image", ""),
                "mode": row.get("mode", ""),
                "delta_psnr": dpsnr,
                "delta_ms_ssim": dms,
                "delta_lpips": dlpips,
                "delta_dists": ddists,
                "score_dists_lpips": score,
                "psnr_win": dpsnr > 0.0,
                "ms_ssim_win": dms > 0.0,
                "lpips_win": dlpips < 0.0,
                "dists_win": ddists < 0.0,
                "score_win": score < 0.0,
                "triple_perceptual_win": (dms > 0.0 and dlpips < 0.0 and ddists < 0.0),
                "delta_bpp": f(row, "delta_bpp"),
                "nonfinite": int(float(row.get("nonfinite", "0") or 0)),
                "max_decode_diff": f(row, "max_decode_diff"),
                "y_mismatch_frac": f(row, "y_mismatch", 0.0) / max(1.0, f(row, "y_total", 1.0)),
                "alpha_mean": f(row, "y_alpha_mean"),
                "alpha_max": f(row, "y_alpha_max"),
                "gate_mean": f(row, "y_gate_mean"),
                "geometry_delta_rms": f(row, "y_avg_geometry_delta_rms"),
                "index_entropy": f(row, "y_avg_index_entropy"),
            }
            out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], lpips_weight: float) -> dict[str, Any]:
    score_winners = [r for r in rows if r["score_win"]]
    psnr_winners = [r for r in rows if r["psnr_win"]]
    psnr_bad_score_good = [r for r in rows if (not r["psnr_win"] and r["score_win"])]
    psnr_good_score_bad = [r for r in rows if (r["psnr_win"] and not r["score_win"])]
    keys = ["delta_psnr", "delta_ms_ssim", "delta_lpips", "delta_dists", "score_dists_lpips"]
    metrics = {f"mean_{key}": mean([r[key] for r in rows]) for key in keys}
    metrics.update({f"oracle_mean_{key}": mean([r[key] if r["score_win"] else 0.0 for r in rows]) for key in keys})
    return {
        "n": len(rows),
        "lpips_weight": lpips_weight,
        **metrics,
        "psnr_win_count": len(psnr_winners),
        "ms_ssim_win_count": sum(1 for r in rows if r["ms_ssim_win"]),
        "lpips_win_count": sum(1 for r in rows if r["lpips_win"]),
        "dists_win_count": sum(1 for r in rows if r["dists_win"]),
        "score_win_count": len(score_winners),
        "triple_perceptual_win_count": sum(1 for r in rows if r["triple_perceptual_win"]),
        "psnr_bad_score_good_count": len(psnr_bad_score_good),
        "psnr_good_score_bad_count": len(psnr_good_score_bad),
        "max_abs_delta_bpp": max(abs(r["delta_bpp"]) for r in rows) if rows else float("nan"),
        "max_decode_diff": max(r["max_decode_diff"] for r in rows) if rows else float("nan"),
        "nonfinite_rows": sum(r["nonfinite"] for r in rows),
        "mean_alpha": mean([r["alpha_mean"] for r in rows]),
        "mean_gate": mean([r["gate_mean"] for r in rows]),
        "corr_psnr_score": corr([r["delta_psnr"] for r in rows], [r["score_dists_lpips"] for r in rows]),
        "corr_psnr_dists": corr([r["delta_psnr"] for r in rows], [r["delta_dists"] for r in rows]),
        "corr_psnr_lpips": corr([r["delta_psnr"] for r in rows], [r["delta_lpips"] for r in rows]),
        "corr_gate_score": corr([r["gate_mean"] for r in rows], [r["score_dists_lpips"] for r in rows]),
        "corr_alpha_score": corr([r["alpha_mean"] for r in rows], [r["score_dists_lpips"] for r in rows]),
        "best_score_rows": sorted(rows, key=lambda r: r["score_dists_lpips"])[:5],
        "worst_score_rows": sorted(rows, key=lambda r: r["score_dists_lpips"], reverse=True)[:5],
        "psnr_conflict_examples": {
            "psnr_bad_score_good": sorted(psnr_bad_score_good, key=lambda r: r["score_dists_lpips"])[:5],
            "psnr_good_score_bad": sorted(psnr_good_score_bad, key=lambda r: r["score_dists_lpips"], reverse=True)[:5],
        },
    }


def fmt(x: Any, digits: int = 6) -> str:
    if isinstance(x, float):
        if not finite(x):
            return "nan"
        return f"{x:+.{digits}f}"
    return str(x)


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "image",
        "delta_psnr",
        "delta_ms_ssim",
        "delta_lpips",
        "delta_dists",
        "score_dists_lpips",
        "psnr_win",
        "ms_ssim_win",
        "lpips_win",
        "dists_win",
        "score_win",
        "triple_perceptual_win",
        "alpha_mean",
        "gate_mean",
        "geometry_delta_rms",
    ]
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# E350 EF-LIC Perceptual Protocol Analysis",
        "",
        "This analysis treats PSNR as a codec-health diagnostic and uses DISTS/LPIPS/MS-SSIM to judge the generative/perceptual compression behavior.",
        "",
        f"Rows: `{summary['n']}`",
        f"Score: `delta_DISTS + {summary['lpips_weight']:g} * delta_LPIPS` (lower is better).",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean dPSNR | {fmt(summary['mean_delta_psnr'])} |",
        f"| mean dMS-SSIM | {fmt(summary['mean_delta_ms_ssim'])} |",
        f"| mean dLPIPS | {fmt(summary['mean_delta_lpips'])} |",
        f"| mean dDISTS | {fmt(summary['mean_delta_dists'])} |",
        f"| mean perceptual score | {fmt(summary['mean_score_dists_lpips'])} |",
        f"| no-op/active oracle score | {fmt(summary['oracle_mean_score_dists_lpips'])} |",
        f"| max abs dBPP | {fmt(summary['max_abs_delta_bpp'])} |",
        f"| max decode diff | {fmt(summary['max_decode_diff'])} |",
        f"| nonfinite rows | {summary['nonfinite_rows']} |",
        "",
        "| win type | count |",
        "|---|---:|",
        f"| PSNR win | {summary['psnr_win_count']} |",
        f"| MS-SSIM win | {summary['ms_ssim_win_count']} |",
        f"| LPIPS win | {summary['lpips_win_count']} |",
        f"| DISTS win | {summary['dists_win_count']} |",
        f"| DISTS+LPIPS score win | {summary['score_win_count']} |",
        f"| all three perceptual metrics win | {summary['triple_perceptual_win_count']} |",
        f"| PSNR loss but perceptual-score win | {summary['psnr_bad_score_good_count']} |",
        f"| PSNR win but perceptual-score loss | {summary['psnr_good_score_bad_count']} |",
        "",
        "| correlation | value |",
        "|---|---:|",
        f"| dPSNR vs perceptual score | {fmt(summary['corr_psnr_score'])} |",
        f"| dPSNR vs dDISTS | {fmt(summary['corr_psnr_dists'])} |",
        f"| dPSNR vs dLPIPS | {fmt(summary['corr_psnr_lpips'])} |",
        f"| gate mean vs perceptual score | {fmt(summary['corr_gate_score'])} |",
        f"| alpha mean vs perceptual score | {fmt(summary['corr_alpha_score'])} |",
        "",
        "## Best Score Rows",
        "",
        "| image | score | dPSNR | dMS | dLPIPS | dDISTS | gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["best_score_rows"]:
        lines.append(
            f"| {row['image']} | {fmt(row['score_dists_lpips'])} | {fmt(row['delta_psnr'])} | "
            f"{fmt(row['delta_ms_ssim'])} | {fmt(row['delta_lpips'])} | {fmt(row['delta_dists'])} | {fmt(row['gate_mean'])} |"
        )
    lines.extend(["", "## Worst Score Rows", "", "| image | score | dPSNR | dMS | dLPIPS | dDISTS | gate |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary["worst_score_rows"]:
        lines.append(
            f"| {row['image']} | {fmt(row['score_dists_lpips'])} | {fmt(row['delta_psnr'])} | "
            f"{fmt(row['delta_ms_ssim'])} | {fmt(row['delta_lpips'])} | {fmt(row['delta_dists'])} | {fmt(row['gate_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- If dPSNR and perceptual score disagree, PSNR should not drive the generative compression claim.",
            "- The no-op/active oracle is an upper bound for a reliability controller that chooses whether to apply HCG per image without changing the fixed payload contract.",
            "- A positive mean dPSNR with near-zero or worse DISTS means the next selector/full-training decision must be metric-aware, not PSNR-only.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.lpips_weight)
    if not rows:
        raise SystemExit(f"no rows loaded from {args.csv}")
    summary = summarize(rows, args.lpips_weight)
    write_outputs(rows, summary, args.output_prefix)


if __name__ == "__main__":
    main()
