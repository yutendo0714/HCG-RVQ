#!/usr/bin/env python3
"""Audit GLC replacement-mode rate accounting.

E278/E280 showed that the low-rate GLC HCG-RVQ branch is useful when interpreted
as an active-scalar-to-active-RVQ replacement mode. This script tightens that
claim by separating empirical-index and fixed-index accounting, cap sensitivity,
and failure correlations. It is an analysis artifact, not a new model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

CAPS = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050]

SUMMARY_FIELDS = [
    "score",
    "fixed_replacement_score",
    "delta_bpp",
    "fixed_replacement_delta_bpp",
    "active_scalar_bpp",
    "active_rvq_extra_bpp",
    "active_rvq_fixed_bpp",
    "active_replacement_delta_bpp",
    "fixed_index_penalty_bpp",
    "scalar_coverage_frac",
    "replacement_over_progressive_frac",
    "active_mse_ratio",
    "index_entropy_mean",
    "index_dead_frac_mean",
    "delta_psnr",
    "delta_ms_ssim",
    "delta_lpips",
    "delta_dists",
]

CORRELATION_FIELDS = [
    "active_replacement_delta_bpp",
    "fixed_replacement_delta_bpp",
    "fixed_index_penalty_bpp",
    "active_scalar_bpp",
    "active_rvq_extra_bpp",
    "scalar_coverage_frac",
    "replacement_over_progressive_frac",
    "active_mse_ratio",
    "index_entropy_mean",
    "delta_psnr",
    "delta_ms_ssim",
    "delta_lpips",
    "delta_dists",
]


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return num / den


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1e-18 or vy <= 1e-18:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


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


def enrich_row(row: dict[str, str], path: Path) -> dict[str, float | str]:
    source, domain = source_from_path(path)
    score = fval(row, "score")
    empirical_delta = fval(row, "active_replacement_delta_bpp")
    active_scalar = fval(row, "active_scalar_bpp")
    active_rvq_emp = fval(row, "active_rvq_extra_bpp")
    active_rvq_fixed = fval(row, "active_rvq_fixed_bpp", active_rvq_emp)
    fixed_delta = active_rvq_fixed - active_scalar
    fixed_penalty = active_rvq_fixed - active_rvq_emp
    fixed_score = score + (fixed_delta - empirical_delta)
    out: dict[str, float | str] = {
        "domain": domain,
        "source": source,
        "artifact": str(path),
        "image": row.get("image", ""),
        "label": row.get("label", ""),
        "score": score,
        "fixed_replacement_score": fixed_score,
        "delta_bpp": fval(row, "delta_bpp"),
        "fixed_replacement_delta_bpp": fixed_delta,
        "active_scalar_bpp": active_scalar,
        "active_rvq_extra_bpp": active_rvq_emp,
        "active_rvq_fixed_bpp": active_rvq_fixed,
        "active_replacement_delta_bpp": empirical_delta,
        "fixed_index_penalty_bpp": fixed_penalty,
        "scalar_coverage_frac": safe_div(active_scalar, active_rvq_emp),
        "replacement_over_progressive_frac": safe_div(empirical_delta, active_rvq_emp),
        "active_mse_ratio": fval(row, "active_mse_ratio"),
        "index_entropy_mean": fval(row, "index_entropy_mean"),
        "index_dead_frac_mean": fval(row, "index_dead_frac_mean"),
        "delta_psnr": fval(row, "delta_psnr"),
        "delta_ms_ssim": fval(row, "delta_ms_ssim"),
        "delta_lpips": fval(row, "delta_lpips"),
        "delta_dists": fval(row, "delta_dists"),
    }
    return out


def read_replacement_rows(paths: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("label") == "trained_replacement_soft":
                    rows.append(enrich_row(row, path))
    return rows


def num(row: dict[str, float | str], key: str) -> float:
    value = row.get(key, math.nan)
    return float(value) if isinstance(value, (float, int)) else math.nan


def summarize(rows: list[dict[str, float | str]], domain: str, source: str = "all") -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {"domain": domain, "source": source, "images": len(rows)}
    for field in SUMMARY_FIELDS:
        out[field] = mean(num(row, field) for row in rows)
    out["empirical_win_frac"] = mean(1.0 if num(row, "score") < 0.0 else 0.0 for row in rows)
    out["fixed_win_frac"] = mean(1.0 if num(row, "fixed_replacement_score") < 0.0 else 0.0 for row in rows)
    out["empirical_positive_frac"] = mean(1.0 if num(row, "score") > 0.0 else 0.0 for row in rows)
    out["fixed_positive_frac"] = mean(1.0 if num(row, "fixed_replacement_score") > 0.0 else 0.0 for row in rows)
    return out


def summary_tables(rows: list[dict[str, float | str]]) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    domain_rows = []
    for domain in sorted({str(row["domain"]) for row in rows}):
        subset = [row for row in rows if row["domain"] == domain]
        domain_rows.append(summarize(subset, domain))
    domain_rows.append(summarize(rows, "all"))

    source_rows = []
    for source in sorted({str(row["source"]) for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        domain = str(subset[0]["domain"]) if subset else "unknown"
        source_rows.append(summarize(subset, domain, source))
    return domain_rows, source_rows


def cap_accounting(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for domain in sorted({str(row["domain"]) for row in rows}) + ["all"]:
        subset = rows if domain == "all" else [row for row in rows if row["domain"] == domain]
        if not subset:
            continue
        for cap in CAPS:
            selected = [row for row in subset if num(row, "active_replacement_delta_bpp") <= cap]
            empirical_scores = [num(row, "score") if row in selected else 0.0 for row in subset]
            fixed_scores = [num(row, "fixed_replacement_score") if row in selected else 0.0 for row in subset]
            empirical_dbpp = [num(row, "active_replacement_delta_bpp") if row in selected else 0.0 for row in subset]
            fixed_dbpp = [num(row, "fixed_replacement_delta_bpp") if row in selected else 0.0 for row in subset]
            out.append(
                {
                    "domain": domain,
                    "cap": cap,
                    "images": len(subset),
                    "selected": len(selected) / len(subset),
                    "empirical_score": mean(empirical_scores),
                    "fixed_score_same_selection": mean(fixed_scores),
                    "empirical_delta_bpp": mean(empirical_dbpp),
                    "fixed_delta_bpp_same_selection": mean(fixed_dbpp),
                    "empirical_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in empirical_scores),
                    "fixed_win_frac_same_selection": mean(1.0 if v < 0.0 else 0.0 for v in fixed_scores),
                    "selected_empirical_win_frac": mean(1.0 if num(row, "score") < 0.0 else 0.0 for row in selected),
                    "selected_fixed_win_frac": mean(1.0 if num(row, "fixed_replacement_score") < 0.0 else 0.0 for row in selected),
                }
            )
    return out


def correlations(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for domain in sorted({str(row["domain"]) for row in rows}) + ["all"]:
        subset = rows if domain == "all" else [row for row in rows if row["domain"] == domain]
        for target in ["score", "fixed_replacement_score"]:
            target_values = [num(row, target) for row in subset]
            for field in CORRELATION_FIELDS:
                out.append(
                    {
                        "domain": domain,
                        "target": target,
                        "feature": field,
                        "pearson": pearson([num(row, field) for row in subset], target_values),
                    }
                )
    return out


def worst_cases(rows: list[dict[str, float | str]], key: str, n: int = 12) -> list[dict[str, float | str]]:
    subset = sorted(rows, key=lambda row: num(row, key), reverse=True)
    keep = [
        "domain",
        "source",
        "image",
        "score",
        "fixed_replacement_score",
        "active_replacement_delta_bpp",
        "fixed_replacement_delta_bpp",
        "fixed_index_penalty_bpp",
        "scalar_coverage_frac",
        "active_mse_ratio",
        "index_entropy_mean",
        "delta_lpips",
        "delta_dists",
    ]
    return [{field: row.get(field, "") for field in keep} for row in subset[:n]]


def cap_disagreements(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    keep = [
        "domain",
        "source",
        "image",
        "score",
        "fixed_replacement_score",
        "active_replacement_delta_bpp",
        "fixed_replacement_delta_bpp",
        "scalar_coverage_frac",
        "active_mse_ratio",
        "delta_lpips",
        "delta_dists",
    ]
    band = [
        row
        for row in rows
        if 0.0035 < num(row, "active_replacement_delta_bpp") <= 0.0040
    ]
    band.sort(key=lambda row: num(row, "score"), reverse=True)
    return [{field: row.get(field, "") for field in keep} for row in band]


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

    rows = read_replacement_rows(args.inputs)
    if not rows:
        raise SystemExit("No trained_replacement_soft rows found.")

    domain_summary, source_summary = summary_tables(rows)
    caps = cap_accounting(rows)
    corr = correlations(rows)
    worst_emp = worst_cases(rows, "score")
    worst_fixed = worst_cases(rows, "fixed_replacement_score")
    cap_band = cap_disagreements(rows)

    payload = {
        "inputs": [str(path) for path in args.inputs],
        "rows": rows,
        "domain_summary": domain_summary,
        "source_summary": source_summary,
        "cap_accounting": caps,
        "correlations": corr,
        "worst_empirical_cases": worst_emp,
        "worst_fixed_cases": worst_fixed,
        "cap_0035_to_0040_band": cap_band,
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_suffix(".domain_summary.csv"), domain_summary)
    write_csv(prefix.with_suffix(".source_summary.csv"), source_summary)
    write_csv(prefix.with_suffix(".cap_accounting.csv"), caps)
    write_csv(prefix.with_suffix(".correlations.csv"), corr)
    write_csv(prefix.with_suffix(".worst_cases.csv"), worst_emp + worst_fixed)
    write_csv(prefix.with_suffix(".cap0035_to_0040_band.csv"), cap_band)

    md: list[str] = []
    md.append("# GLC Replacement Accounting Audit")
    md.append("")
    md.append(
        "This audit separates the empirical-index replacement estimate from a "
        "more conservative fixed-index estimate. It checks whether the current "
        "GLC HCG-RVQ replacement evidence can be safely promoted toward a real "
        "selected-index codec design."
    )
    md.append("")
    md.append("## Domain Summary")
    md.extend(
        md_table(
            domain_summary,
            [
                "domain",
                "images",
                "score",
                "fixed_replacement_score",
                "empirical_win_frac",
                "fixed_win_frac",
                "active_replacement_delta_bpp",
                "fixed_replacement_delta_bpp",
                "fixed_index_penalty_bpp",
                "scalar_coverage_frac",
                "active_mse_ratio",
                "index_entropy_mean",
            ],
        )
    )
    md.append("")
    md.append("## Cap Accounting")
    md.extend(
        md_table(
            caps,
            [
                "domain",
                "cap",
                "images",
                "selected",
                "empirical_score",
                "fixed_score_same_selection",
                "empirical_delta_bpp",
                "fixed_delta_bpp_same_selection",
                "selected_empirical_win_frac",
                "selected_fixed_win_frac",
            ],
        )
    )
    md.append("")
    md.append("## Strongest Correlations With Empirical Score")
    for domain in ["clic", "kodak", "all"]:
        subset = [
            row
            for row in corr
            if row["domain"] == domain and row["target"] == "score" and math.isfinite(float(row["pearson"]))
        ]
        subset.sort(key=lambda row: abs(float(row["pearson"])), reverse=True)
        md.append("")
        md.append(f"### {domain}")
        md.extend(md_table(subset[:8], ["feature", "pearson"]))
    md.append("")
    md.append("## Worst Empirical Replacement Cases")
    md.extend(
        md_table(
            worst_emp,
            [
                "domain",
                "source",
                "image",
                "score",
                "fixed_replacement_score",
                "active_replacement_delta_bpp",
                "fixed_index_penalty_bpp",
                "scalar_coverage_frac",
                "active_mse_ratio",
                "delta_lpips",
                "delta_dists",
            ],
        )
    )
    md.append("")
    md.append("## Cap 0.0035-0.0040 Decision Band")
    md.extend(
        md_table(
            cap_band,
            [
                "domain",
                "source",
                "image",
                "score",
                "fixed_replacement_score",
                "active_replacement_delta_bpp",
                "scalar_coverage_frac",
                "active_mse_ratio",
                "delta_lpips",
                "delta_dists",
            ],
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "Negative empirical replacement score means the active RVQ branch still "
        "wins after charging the active RVQ index stream minus the active scalar "
        "bits it replaces. The fixed-index columns are deliberately conservative: "
        "they show how much margin disappears if the branch cannot exploit index "
        "statistics or coarse entropy coding. A paper-main GLC implementation "
        "should therefore either implement real selected-index coding or report "
        "the fixed-index ablation as a bound."
    )
    md.append("")
    prefix.with_suffix(".md").write_text("\n".join(md).rstrip() + "\n")


if __name__ == "__main__":
    main()
