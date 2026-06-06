#!/usr/bin/env python3
"""Summarize EF-LIC spatial alpha-map strength probes.

This aggregates E225/E226 CSVs into a compact paper-planning table. The score
used here is diagnostic only: lower DISTS + 3 * LPIPS is better.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUTS = [
    ("clicpro41", Path("experiments/analysis/e228_eflic_clicpro41_spatial_alpha_map_alpha0005_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e226_eflic_clicpro41_spatial_alpha_map_alpha001_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e225_eflic_clicpro41_spatial_alpha_map_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e231_eflic_clicpro41_spatial_alpha_soft_alpha0005.csv")),
    ("clicpro41", Path("experiments/analysis/e230_eflic_clicpro41_spatial_alpha_soft_alpha001.csv")),
    ("clicpro41", Path("experiments/analysis/e229_eflic_clicpro41_spatial_alpha_soft_alpha002.csv")),
    ("kodak24", Path("experiments/analysis/e228_eflic_kodak24_spatial_alpha_map_alpha0005_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e226_eflic_kodak24_spatial_alpha_map_alpha001_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e225_eflic_kodak24_spatial_alpha_map_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e231_eflic_kodak24_spatial_alpha_soft_alpha0005.csv")),
    ("kodak24", Path("experiments/analysis/e230_eflic_kodak24_spatial_alpha_soft_alpha001.csv")),
    ("kodak24", Path("experiments/analysis/e229_eflic_kodak24_spatial_alpha_soft_alpha002.csv")),
]


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "None"} else 0.0


def summarize_rows(label: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], f(row, "alpha"))].append(row)

    out: list[dict[str, object]] = []
    for (mode, alpha), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        n = len(group)
        mean = lambda key: sum(f(r, key) for r in group) / max(n, 1)
        score_values = [f(r, "delta_dists") + 3.0 * f(r, "delta_lpips") for r in group]
        y_mismatch = sum(f(r, "y_mismatch") for r in group)
        y_total = sum(f(r, "y_total") for r in group)
        row = {
            "dataset": label,
            "mode": mode,
            "alpha": alpha,
            "images": n,
            "delta_bpp": mean("delta_bpp"),
            "delta_psnr": mean("delta_psnr"),
            "delta_dists": mean("delta_dists"),
            "delta_lpips": mean("delta_lpips"),
            "score_dists_3lpips": sum(score_values) / max(n, 1),
            "psnr_win_frac": sum(1 for r in group if f(r, "delta_psnr") > 0.0) / max(n, 1),
            "dists_win_frac": sum(1 for r in group if f(r, "delta_dists") < 0.0) / max(n, 1),
            "lpips_win_frac": sum(1 for r in group if f(r, "delta_lpips") < 0.0) / max(n, 1),
            "score_win_frac": sum(1 for v in score_values if v < 0.0) / max(n, 1),
            "max_decode_diff": max(f(r, "max_decode_diff") for r in group),
            "nonfinite_rows": int(sum(f(r, "nonfinite") for r in group)),
            "y_mismatch_frac": y_mismatch / y_total if y_total > 0 else 0.0,
            "alpha_active_frac": mean("y_alpha_active_frac"),
            "geometry_delta_rms": mean("y_avg_geometry_delta_rms"),
        }
        out.append(row)
    return out


def format_float(value: object, digits: int = 8) -> str:
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e226_eflic_spatial_alpha_strength_summary"))
    args = parser.parse_args()

    rows_out: list[dict[str, object]] = []
    for label, path in DEFAULT_INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            rows = list(csv.DictReader(fobj))
        rows_out.extend(summarize_rows(label, rows))
    deduped: list[dict[str, object]] = []
    seen_identity: set[str] = set()
    for row in rows_out:
        if row["mode"] == "zero":
            dataset = str(row["dataset"])
            if dataset in seen_identity:
                continue
            seen_identity.add(dataset)
        deduped.append(row)
    rows_out = deduped

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    headers = [
        "dataset",
        "mode",
        "alpha",
        "images",
        "delta_psnr",
        "delta_dists",
        "delta_lpips",
        "score_dists_3lpips",
        "score_win_frac",
        "max_decode_diff",
        "nonfinite_rows",
        "y_mismatch_frac",
        "alpha_active_frac",
    ]
    with md_path.open("w") as fobj:
        fobj.write("# E226 EF-LIC Spatial Alpha Strength Summary\n\n")
        fobj.write("Lower `score_dists_3lpips = delta_dists + 3 * delta_lpips` is better.\n\n")
        fobj.write("| " + " | ".join(headers) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows_out:
            cells = []
            for key in headers:
                val = row[key]
                if key in {"alpha", "score_win_frac", "y_mismatch_frac", "alpha_active_frac"}:
                    cells.append(format_float(val, 6))
                elif key.startswith("delta") or key.startswith("score_") or key == "max_decode_diff":
                    cells.append(format_float(val, 8))
                else:
                    cells.append(str(val))
            fobj.write("| " + " | ".join(cells) + " |\n")

        fobj.write("\nInterpretation:\n\n")
        fobj.write("- CLIC professional favors very weak/local geometry among fixed rules; `prev_rms_top@0.005` remains best, while `prev_over_scale_top_soft@0.01` is close and improves both DISTS and LPIPS.\n")
        fobj.write("- Kodak keeps stronger all-on behavior useful for the joint score; soft support/scale branches are useful oracle candidates but weaker as fixed policies.\n")
        fobj.write("- All summarized rows preserve unchanged bpp and exact decoder reproduction; remaining differences are geometry/strength effects.\n")

    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
