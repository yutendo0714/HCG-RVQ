#!/usr/bin/env python3
"""Aggregate GLC replacement-row pilots.

This is a paper-readiness diagnostic, not a new model. It combines the direct
codec-loop rows from multiple held-out slices and reports whether the selected
replacement framing remains useful as the evaluation slice grows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

LABELS = [
    "trained_soft_gate",
    "trained_progressive_extra_soft",
    "trained_replacement_soft",
    "trained_rate_cap_replacement_soft",
    "trained_rate_cap_replacement_soft_cap0p0035",
    "trained_rate_cap_replacement_soft_cap0p004",
    "trained_all_on",
    "trained_replacement_all_on",
]

MEAN_FIELDS = [
    "bpp",
    "delta_bpp",
    "score",
    "delta_psnr",
    "delta_ms_ssim",
    "delta_lpips",
    "delta_dists",
    "gate_mean",
    "selected",
    "active_mse_ratio",
    "active_scalar_bpp",
    "active_rvq_extra_bpp",
    "active_replacement_delta_bpp",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "nonfinite",
]

CAPS = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050]


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def source_from_path(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "clicpro16held32" in name:
        return "clic_tail9", "clic"
    if "clicpro16held" in name:
        return "clic16", "clic"
    if "clicpro8held" in name:
        return "clic8", "clic"
    if "kodak16held" in name:
        return "kodak16", "kodak"
    if "kodak4held" in name:
        return "kodak4", "kodak"
    if "clic" in name:
        return "clic_unknown", "clic"
    if "kodak" in name:
        return "kodak_unknown", "kodak"
    return path.stem, "unknown"


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        source, domain = source_from_path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source"] = source
                row["_domain"] = domain
                row["_artifact"] = str(path)
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {"images": len(rows)}
    for field in MEAN_FIELDS:
        out[field] = mean(fval(row, field) for row in rows)
    scores = [fval(row, "score") for row in rows]
    finite_scores = [v for v in scores if math.isfinite(v)]
    out["win_frac"] = mean(1.0 if v < 0.0 else 0.0 for v in finite_scores)
    out["positive_frac"] = mean(1.0 if v > 0.0 else 0.0 for v in finite_scores)
    nonfinite = [fval(row, "nonfinite", 0.0) for row in rows]
    out["nonfinite_rows"] = int(sum(v for v in nonfinite if math.isfinite(v)))
    return out


def label_table(rows: list[dict[str, str]], group_key: str) -> list[dict[str, float | int | str]]:
    table: list[dict[str, float | int | str]] = []
    for group_value in sorted({row[group_key] for row in rows}):
        for label in LABELS:
            subset = [row for row in rows if row[group_key] == group_value and row.get("label") == label]
            if not subset:
                continue
            item = summarize(subset)
            item[group_key.strip("_")] = group_value
            item["label"] = label
            table.append(item)
    return table


def pooled_label_table(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    table: list[dict[str, float | int | str]] = []
    for label in LABELS:
        subset = [row for row in rows if row.get("label") == label]
        if not subset:
            continue
        item = summarize(subset)
        item["domain"] = "all"
        item["label"] = label
        table.append(item)
    return table


def cap_sweep(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    base_rows = [row for row in rows if row.get("label") == "trained_replacement_soft"]
    out: list[dict[str, float | int | str]] = []
    for group_key, group_value in [("_domain", "clic"), ("_domain", "kodak"), ("_domain", "all")]:
        if group_value == "all":
            subset = base_rows
        else:
            subset = [row for row in base_rows if row.get(group_key) == group_value]
        if not subset:
            continue
        for cap in CAPS:
            selected = [row for row in subset if fval(row, "active_replacement_delta_bpp") <= cap]
            score_values = [fval(row, "score") if row in selected else 0.0 for row in subset]
            dbpp_values = [fval(row, "active_replacement_delta_bpp") if row in selected else 0.0 for row in subset]
            selected_scores = [fval(row, "score") for row in selected]
            out.append(
                {
                    "domain": group_value,
                    "cap": cap,
                    "images": len(subset),
                    "selected": len(selected) / len(subset),
                    "score": mean(score_values),
                    "delta_bpp": mean(dbpp_values),
                    "win_frac": mean(1.0 if v < 0.0 else 0.0 for v in score_values),
                    "selected_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in selected_scores),
                }
            )
    return out


def worst_cases(rows: list[dict[str, str]], label: str, n: int = 12) -> list[dict[str, float | int | str]]:
    subset = [row for row in rows if row.get("label") == label]
    subset.sort(key=lambda row: fval(row, "score"), reverse=True)
    fields = [
        "score",
        "delta_bpp",
        "active_replacement_delta_bpp",
        "active_scalar_bpp",
        "active_rvq_extra_bpp",
        "active_mse_ratio",
        "index_entropy_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
        "delta_psnr",
        "delta_ms_ssim",
        "delta_lpips",
        "delta_dists",
        "gate_mean",
        "selected",
    ]
    out: list[dict[str, float | int | str]] = []
    for row in subset[:n]:
        item: dict[str, float | int | str] = {
            "domain": row.get("_domain", ""),
            "source": row.get("_source", ""),
            "image": row.get("image", ""),
            "label": label,
        }
        for field in fields:
            item[field] = fval(row, field)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    if abs(value) < 1 and value != 0:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def md_table(rows: list[dict[str, float | int | str]], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    domain_summary = label_table(rows, "_domain") + pooled_label_table(rows)
    source_summary = label_table(rows, "_source")
    caps = cap_sweep(rows)
    failures_repl = worst_cases(rows, "trained_replacement_soft")
    failures_cap = worst_cases(rows, "trained_rate_cap_replacement_soft")
    failures_prog = worst_cases(rows, "trained_progressive_extra_soft")

    payload = {
        "inputs": [str(path) for path in args.inputs],
        "domain_summary": domain_summary,
        "source_summary": source_summary,
        "cap_sweep": caps,
        "worst_trained_replacement_soft": failures_repl,
        "worst_trained_rate_cap_replacement_soft": failures_cap,
        "worst_trained_progressive_extra_soft": failures_prog,
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_suffix(".domain_summary.csv"), domain_summary)
    write_csv(prefix.with_suffix(".source_summary.csv"), source_summary)
    write_csv(prefix.with_suffix(".cap_sweep.csv"), caps)
    write_csv(prefix.with_suffix(".worst_cases.csv"), failures_repl + failures_cap + failures_prog)

    md: list[str] = []
    md.append("# GLC Replacement Scaling Diagnostics")
    md.append("")
    md.append(
        "Aggregates direct replacement-row pilots. The purpose is to test whether "
        "the GLC low-rate HCG-RVQ branch still looks useful when moving from small held-out "
        "slices to larger CLIC/Kodak slices, and whether the evidence points to replacement "
        "coding rather than additive enhancement."
    )
    md.append("")
    md.append("## Domain Summary")
    md.extend(
        md_table(
            domain_summary,
            [
                "domain",
                "label",
                "images",
                "score",
                "win_frac",
                "delta_bpp",
                "delta_psnr",
                "delta_ms_ssim",
                "delta_lpips",
                "delta_dists",
                "selected",
                "active_mse_ratio",
                "index_entropy_mean",
                "index_dead_frac_mean",
                "nonfinite_rows",
            ],
        )
    )
    md.append("")
    md.append("## Source Summary")
    md.extend(
        md_table(
            source_summary,
            [
                "source",
                "label",
                "images",
                "score",
                "win_frac",
                "delta_bpp",
                "selected",
                "active_replacement_delta_bpp",
                "active_scalar_bpp",
                "active_rvq_extra_bpp",
                "index_entropy_mean",
                "index_used_frac_mean",
                "index_dead_frac_mean",
            ],
        )
    )
    md.append("")
    md.append("## Replacement Delta-Bpp Cap Sweep")
    md.extend(md_table(caps, ["domain", "cap", "images", "selected", "score", "delta_bpp", "win_frac", "selected_win_frac"]))
    md.append("")
    md.append("## Worst Cases: trained_replacement_soft")
    md.extend(
        md_table(
            failures_repl,
            [
                "domain",
                "source",
                "image",
                "score",
                "delta_bpp",
                "active_replacement_delta_bpp",
                "active_mse_ratio",
                "index_entropy_mean",
                "index_dead_frac_mean",
                "delta_dists",
                "gate_mean",
            ],
        )
    )
    md.append("")
    md.append("## Worst Cases: trained_progressive_extra_soft")
    md.extend(
        md_table(
            failures_prog,
            [
                "domain",
                "source",
                "image",
                "score",
                "delta_bpp",
                "active_replacement_delta_bpp",
                "active_mse_ratio",
                "index_entropy_mean",
                "index_dead_frac_mean",
                "delta_dists",
                "gate_mean",
            ],
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "`trained_replacement_soft` is the main short-cycle proxy because it charges the active RVQ "
        "stream as a replacement for active scalar residual bits. `trained_progressive_extra_soft` "
        "is the additive-enhancement negative check, and all-on rows test whether dense activation "
        "destroys the reconstruction manifold. A useful HCG-RVQ port should preserve negative "
        "replacement score, keep all-on harmful as an ablation, and then make replacement accounting "
        "bit-exact in the full codec."
    )
    md.append("")
    prefix.with_suffix(".md").write_text("\n".join(md).rstrip() + "\n")


if __name__ == "__main__":
    main()
