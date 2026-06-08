#!/usr/bin/env python3
"""Audit EF-LIC perceptual slice-oracle choices from E313-style rows.

This is a paper-protocol diagnostic for the generative/perceptual branch.  It
compares the slice set chosen by a perceptual objective against the slice set
chosen by PSNR, while treating bpp/decode/nonfinite consistency as hard codec
constraints.  Lower score = delta_DISTS + w * delta_LPIPS is better.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/analysis/e364_eflic_perceptual_slice_isolation_kodak24_riskm060.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e365_eflic_perceptual_slice_oracle_audit_kodak24_riskm060",
    )
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--top-traps", type=int, default=16)
    return p.parse_args()


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def read_rows(path: Path, lpips_weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for raw in csv.DictReader(fobj):
            row: dict[str, Any] = dict(raw)
            for key in (
                "delta_psnr",
                "delta_ms_ssim",
                "delta_lpips",
                "delta_dists",
                "delta_bpp",
                "max_decode_diff",
                "mean_gate",
                "mean_alpha",
                "mean_slice_enabled",
            ):
                row[key] = f(raw, key)
            row["nonfinite"] = int(f(raw, "nonfinite"))
            row["perceptual_score"] = f(
                raw,
                "perceptual_score",
                row["delta_dists"] + lpips_weight * row["delta_lpips"],
            )
            row["contract_ok"] = int(
                abs(row["delta_bpp"]) <= 1e-12
                and abs(row["max_decode_diff"]) <= 1e-12
                and row["nonfinite"] == 0
            )
            rows.append(row)
    return rows


def summarize_choice(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_score": finite_mean(r["perceptual_score"] for r in rows),
        f"{prefix}_delta_psnr": finite_mean(r["delta_psnr"] for r in rows),
        f"{prefix}_delta_ms_ssim": finite_mean(r["delta_ms_ssim"] for r in rows),
        f"{prefix}_delta_lpips": finite_mean(r["delta_lpips"] for r in rows),
        f"{prefix}_delta_dists": finite_mean(r["delta_dists"] for r in rows),
        f"{prefix}_score_wins": sum(r["perceptual_score"] < 0.0 for r in rows),
        f"{prefix}_psnr_wins": sum(r["delta_psnr"] > 0.0 for r in rows),
        f"{prefix}_triple_perceptual_wins": sum(
            r["delta_ms_ssim"] > 0.0 and r["delta_lpips"] < 0.0 and r["delta_dists"] < 0.0 for r in rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.rows, args.lpips_weight)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image"])].append(row)

    per_image: list[dict[str, Any]] = []
    best_score_rows: list[dict[str, Any]] = []
    best_psnr_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for image, subset in sorted(by_image.items()):
        best_score = min(subset, key=lambda r: r["perceptual_score"])
        best_psnr = max(subset, key=lambda r: r["delta_psnr"])
        worst_score = max(subset, key=lambda r: r["perceptual_score"])
        all_row = next((r for r in subset if str(r["active_slices"]) == "all"), None)
        if all_row is not None:
            all_rows.append(all_row)
        best_score_rows.append(best_score)
        best_psnr_rows.append(best_psnr)
        per_image.append(
            {
                "image": image,
                "all_score": all_row["perceptual_score"] if all_row else float("nan"),
                "all_delta_psnr": all_row["delta_psnr"] if all_row else float("nan"),
                "best_score_slice_set": best_score["active_slices"],
                "best_score": best_score["perceptual_score"],
                "best_score_delta_psnr": best_score["delta_psnr"],
                "best_psnr_slice_set": best_psnr["active_slices"],
                "best_psnr_delta_psnr": best_psnr["delta_psnr"],
                "best_psnr_score": best_psnr["perceptual_score"],
                "worst_score_slice_set": worst_score["active_slices"],
                "worst_score": worst_score["perceptual_score"],
                "score_gain_over_all": (best_score["perceptual_score"] - all_row["perceptual_score"]) if all_row else float("nan"),
                "psnr_choice_score_gap": best_psnr["perceptual_score"] - best_score["perceptual_score"],
                "oracle_choice_match": int(best_score["active_slices"] == best_psnr["active_slices"]),
                "contract_ok_frac": finite_mean(r["contract_ok"] for r in subset),
            }
        )

    traps = [r for r in rows if r["delta_psnr"] > 0.0 and r["perceptual_score"] > 0.0]
    traps = sorted(traps, key=lambda r: r["perceptual_score"], reverse=True)[: args.top_traps]
    perceptual_psnr_losses = [r for r in rows if r["delta_psnr"] < 0.0 and r["perceptual_score"] < 0.0]

    summary: dict[str, Any] = {
        "rows": len(rows),
        "images": len(by_image),
        "lpips_weight": args.lpips_weight,
        "contract_ok": all(r["contract_ok"] for r in rows),
        "max_abs_delta_bpp": max(abs(r["delta_bpp"]) for r in rows) if rows else float("nan"),
        "max_decode_diff": max(abs(r["max_decode_diff"]) for r in rows) if rows else float("nan"),
        "nonfinite_rows": sum(r["nonfinite"] for r in rows),
        "psnr_positive_perceptual_bad_rows": sum(r["delta_psnr"] > 0.0 and r["perceptual_score"] > 0.0 for r in rows),
        "perceptual_good_psnr_negative_rows": len(perceptual_psnr_losses),
        "oracle_choice_mismatch_images": sum(not item["oracle_choice_match"] for item in per_image),
        "mean_score_gain_score_oracle_over_all": finite_mean(item["score_gain_over_all"] for item in per_image),
        "mean_score_gap_psnr_oracle_minus_score_oracle": finite_mean(item["psnr_choice_score_gap"] for item in per_image),
        "score_oracle_choice_counts": dict(Counter(str(r["active_slices"]) for r in best_score_rows)),
        "psnr_oracle_choice_counts": dict(Counter(str(r["active_slices"]) for r in best_psnr_rows)),
    }
    summary.update(summarize_choice(all_rows, "all"))
    summary.update(summarize_choice(best_score_rows, "score_oracle"))
    summary.update(summarize_choice(best_psnr_rows, "psnr_oracle"))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".per_image.csv"), per_image)
    write_csv(args.output_prefix.with_suffix(".psnr_traps.csv"), traps)
    with args.output_prefix.with_suffix(".json").open("w", encoding="utf-8") as fobj:
        json.dump({"summary": summary, "per_image": per_image, "top_psnr_traps": traps}, fobj, indent=2)
    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as fobj:
        fobj.write("# E365 EF-LIC Perceptual Slice Oracle Audit\n\n")
        fobj.write(f"Rows: `{args.rows}`\n\n")
        fobj.write("Lower `score = delta_DISTS + w * delta_LPIPS` is better. PSNR is diagnostic only.\n\n")
        fobj.write("## Summary\n\n")
        for key in [
            "images",
            "rows",
            "contract_ok",
            "all_score",
            "all_delta_psnr",
            "score_oracle_score",
            "score_oracle_delta_psnr",
            "psnr_oracle_score",
            "psnr_oracle_delta_psnr",
            "oracle_choice_mismatch_images",
            "mean_score_gain_score_oracle_over_all",
            "mean_score_gap_psnr_oracle_minus_score_oracle",
            "psnr_positive_perceptual_bad_rows",
            "perceptual_good_psnr_negative_rows",
        ]:
            fobj.write(f"- {key}: `{summary.get(key)}`\n")
        fobj.write("\n## Choice Counts\n\n")
        fobj.write(f"- score oracle: `{summary['score_oracle_choice_counts']}`\n")
        fobj.write(f"- PSNR oracle: `{summary['psnr_oracle_choice_counts']}`\n")
        fobj.write("\n## Top PSNR-Positive Perceptual-Bad Rows\n\n")
        fobj.write("| image | slices | dPSNR | score | dLPIPS | dDISTS |\n")
        fobj.write("|---|---|---:|---:|---:|---:|\n")
        for row in traps[:8]:
            fobj.write(
                f"| {row['image']} | {row['active_slices']} | {row['delta_psnr']:+.6f} | "
                f"{row['perceptual_score']:+.6f} | {row['delta_lpips']:+.6f} | {row['delta_dists']:+.6f} |\n"
            )
    print(f"wrote {args.output_prefix}.{{json,md,per_image.csv,psnr_traps.csv}}")


if __name__ == "__main__":
    main()
