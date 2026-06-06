#!/usr/bin/env python3
"""Audit whether head-only reliability concentrates on fallback-needed images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_VARIANTS = {
    "rho005_step250": "experiments/analysis/teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.csv",
    "rho005_step500": "experiments/analysis/teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
    "rho050_step250": "experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.csv",
    "rho050_step500": "experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
    "marginw_rho050_step250": "experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.csv",
    "marginw_rho050_step500": "experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
}

REFERENCE_CSV = "experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv"
OUT_JSON = "experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.json"
OUT_MD = "experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.md"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_variants(items: list[str] | None, include_defaults: bool) -> dict[str, Path]:
    variants = {name: Path(path) for name, path in DEFAULT_VARIANTS.items()} if include_defaults else {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--variant must be NAME=CSV, got {item!r}")
        name, path = item.split("=", 1)
        variants[name] = Path(path)
    return variants


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def auc_low_reliability_for_fallback(rows: list[dict[str, float]]) -> float:
    positives = [row for row in rows if row["fallback_needed"] == 1.0]
    negatives = [row for row in rows if row["fallback_needed"] == 0.0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos["reliability"] < neg["reliability"]:
                wins += 1.0
            elif pos["reliability"] == neg["reliability"]:
                wins += 0.5
    return wins / total


def finite_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def summarize_variant(path: Path, refs: dict[str, dict[str, str]]) -> dict[str, float | str]:
    rows = []
    for row in load_csv(path):
        ref = refs[row["path"]]
        fallback_needed = 1.0 if float(ref["previous_local_rd"]) < float(ref["variant500_rd"]) else 0.0
        reliability = finite_float(row, "rvq_householder_reliability_multiplier")
        rd_delta = float(row["rd_score"]) - float(ref["variant500_rd"])
        rows.append(
            {
                "fallback_needed": fallback_needed,
                "reliability": reliability,
                "rd_delta": rd_delta,
                "raw_gate": finite_float(row, "rvq_householder_gate_raw"),
                "delta_rms": finite_float(row, "rvq_householder_delta_rms"),
            }
        )
    keep = [row for row in rows if row["fallback_needed"] == 0.0]
    fallback = [row for row in rows if row["fallback_needed"] == 1.0]
    reliability_keep = mean(row["reliability"] for row in keep)
    reliability_fallback = mean(row["reliability"] for row in fallback)
    rd_keep = mean(row["rd_delta"] for row in keep)
    rd_fallback = mean(row["rd_delta"] for row in fallback)
    rd_delta = [row["rd_delta"] for row in rows]
    return {
        "csv": str(path),
        "fallback_frac": mean(row["fallback_needed"] for row in rows),
        "reliability_mean": mean(row["reliability"] for row in rows),
        "reliability_keep_mean": reliability_keep,
        "reliability_fallback_mean": reliability_fallback,
        "reliability_gap_fallback_minus_keep": reliability_fallback - reliability_keep,
        "auc_low_reliability_for_fallback": auc_low_reliability_for_fallback(rows),
        "rd_delta_mean": mean(rd_delta),
        "rd_delta_keep_mean": rd_keep,
        "rd_delta_fallback_mean": rd_fallback,
        "corr_reliability_rd_delta_vs_beta005": pearson([row["reliability"] for row in rows], rd_delta),
        "corr_rawgate_rd_delta_vs_beta005": pearson([row["raw_gate"] for row in rows], rd_delta),
        "corr_delta_rms_rd_delta_vs_beta005": pearson([row["delta_rms"] for row in rows], rd_delta),
    }


def fmt(value: float, signed: bool = False) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def render_markdown(report: dict[str, object]) -> str:
    variants = report["variants"]
    assert isinstance(variants, dict)
    lines = [
        "# Head-Only Teacher Reliability Signal Audit",
        "",
        "Holdout4096 reference labels mark fallback-needed images where previous-local step250 beats beta005 step500. Lower reliability should ideally concentrate on those images.",
        "",
        "| variant | fallback frac | rel mean | rel keep | rel fallback | fallback-keep gap | AUC low-rel fallback | RD delta mean | RD delta keep | RD delta fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in variants.items():
        assert isinstance(data, dict)
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(float(data["fallback_frac"])),
                    fmt(float(data["reliability_mean"])),
                    fmt(float(data["reliability_keep_mean"])),
                    fmt(float(data["reliability_fallback_mean"])),
                    fmt(float(data["reliability_gap_fallback_minus_keep"]), signed=True),
                    fmt(float(data["auc_low_reliability_for_fallback"])),
                    fmt(float(data["rd_delta_mean"]), signed=True),
                    fmt(float(data["rd_delta_keep_mean"]), signed=True),
                    fmt(float(data["rd_delta_fallback_mean"]), signed=True),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: reliability separation is useful only if it preserves beta005's RVQ assignment regime. Large RD deltas with high qMSE/dead-code are rejected even when reliability is lower on fallback-needed images.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", default=REFERENCE_CSV)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--variant", action="append", default=None, help="NAME=CSV. May be repeated.")
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--out-json", default=OUT_JSON)
    parser.add_argument("--out-md", default=OUT_MD)
    args = parser.parse_args()

    refs = {
        row["path"]: row
        for row in load_csv(Path(args.reference_csv))
        if str(row.get("seed", "")) == str(args.seed)
    }
    variants = parse_variants(args.variant, include_defaults=not args.no_defaults)
    report = {
        "reference_csv": args.reference_csv,
        "rows": len(refs),
        "variants": {name: summarize_variant(path, refs) for name, path in variants.items()},
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
