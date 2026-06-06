#!/usr/bin/env python3
"""E269 progressive-rate margin audit for GLC fallback-gate rows.

E266/E268 charge selected soft-gated rows the full all-on branch bpp.  This
script asks the next paper-facing question: how much branch rate can the current
soft reconstruction afford, which rows fail under full-rate accounting, and how
stable is the result if a selected/progressive bitstream pays a fraction or a
multiple of the observed full branch rate?

The audit still evaluates the existing soft-gated reconstruction.  It is not a
replacement for an entropy-coded implementation, but it defines the rate budget
that such an implementation must satisfy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "experiments/analysis/e268_glc_lowrate_fallback_gate_kodak4first_k4_parts01_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e267_glc_lowrate_fallback_gate_kodak4held_k4_parts01_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e268_glc_lowrate_fallback_gate_clicpro8first_k4_parts01_t8_e8_s8.csv",
    ROOT / "experiments/analysis/e267_glc_lowrate_fallback_gate_clicpro8held_k4_parts01_t8_e8_s8.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e269_glc_lowrate_progressive_rate_margin",
    )
    p.add_argument("--lpips-score-weight", type=float, default=3.0)
    p.add_argument("--rate-multipliers", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    p.add_argument("--failure-topk", type=int, default=8)
    return p.parse_args()


def finite(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def dataset_from_path(path: Path) -> str:
    name = path.name
    if "clicpro8held" in name:
        return "clicpro8_held"
    if "clicpro8first" in name or "clicpro8" in name:
        return "clicpro8_first"
    if "kodak4held" in name:
        return "kodak4_held"
    if "kodak4first" in name or "kodak4" in name:
        return "kodak4_first"
    if "kodak2" in name:
        return "kodak2_first"
    if "smoke" in name:
        return "kodak1_smoke"
    return path.stem


def read_source(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.exists():
            continue
        dataset = dataset_from_path(path)
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                item = dict(row)
                item["dataset"] = dataset
                item["phase"] = "trained" if str(item.get("label", "")).startswith("trained_") else "init"
                try:
                    item["source_csv"] = str(path.relative_to(ROOT))
                except ValueError:
                    item["source_csv"] = str(path)
                rows.append(item)
    return rows


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["dataset"]), str(row["phase"]), str(row.get("image", "")), str(row.get("q_index", "0")))


def build_rows(source_rows: list[dict[str, Any]], lpips_weight: float) -> list[dict[str, Any]]:
    all_on_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in source_rows:
        if str(row.get("label", "")).endswith("_all_on"):
            all_on_by_key[key(row)] = row

    out_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if not str(row.get("label", "")).endswith("_soft_gate"):
            continue
        paired = all_on_by_key.get(key(row))
        if paired is None:
            continue
        no_bpp_score = finite(row, "delta_dists") + lpips_weight * finite(row, "delta_lpips")
        diag_dbpp = finite(row, "delta_bpp")
        full_dbpp = finite(paired, "delta_bpp")
        diag_score = no_bpp_score + diag_dbpp
        gate_dbpp = finite(row, "gate_mean") * full_dbpp
        gate_score = no_bpp_score + gate_dbpp
        full_score = no_bpp_score + full_dbpp
        affordable_dbpp = max(0.0, -no_bpp_score)
        if full_dbpp > 0.0:
            max_rate_fraction = affordable_dbpp / full_dbpp
            diag_rate_fraction = diag_dbpp / full_dbpp
            gate_rate_fraction = gate_dbpp / full_dbpp
        else:
            max_rate_fraction = float("inf") if no_bpp_score < 0.0 else 0.0
            diag_rate_fraction = 0.0
            gate_rate_fraction = 0.0
        out_rows.append(
            {
                "dataset": key(row)[0],
                "phase": key(row)[1],
                "image": key(row)[2],
                "q_index": key(row)[3],
                "label": row.get("label", ""),
                "score_no_bpp": no_bpp_score,
                "score_diag": diag_score,
                "score_gate_full_rate_fraction": gate_score,
                "score_full_branch_bpp": full_score,
                "diagnostic_dbpp": diag_dbpp,
                "gate_fraction_dbpp": gate_dbpp,
                "full_branch_dbpp": full_dbpp,
                "affordable_dbpp": affordable_dbpp,
                "extra_dbpp_margin_after_full": -full_score,
                "max_rate_fraction": max_rate_fraction,
                "diagnostic_rate_fraction": diag_rate_fraction,
                "gate_rate_fraction": gate_rate_fraction,
                "gate_mean": finite(row, "gate_mean"),
                "active_mse_ratio": finite(row, "active_mse_ratio"),
                "active_rvq_mse": finite(row, "active_rvq_mse"),
                "active_scalar_mse": finite(row, "active_scalar_mse"),
                "index_entropy_mean": finite(row, "index_entropy_mean"),
                "index_used_frac_mean": finite(row, "index_used_frac_mean"),
                "index_dead_frac_mean": finite(row, "index_dead_frac_mean"),
                "full_win": int(full_score < 0.0),
                "diag_win": int(diag_score < 0.0),
                "source_csv": row.get("source_csv", ""),
            }
        )
    return out_rows


def group_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("all", rows),
        ("trained", [r for r in rows if r["phase"] == "trained"]),
        ("trained_kodak", [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).startswith("kodak")]),
        ("trained_clic", [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).startswith("clic")]),
        ("trained_first", [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).endswith("_first")]),
        ("trained_held", [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).endswith("_held")]),
    ]


def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group": name, "rows": 0}
    max_fracs = [float(r["max_rate_fraction"]) for r in rows if math.isfinite(float(r["max_rate_fraction"]))]
    return {
        "group": name,
        "rows": len(rows),
        "score_no_bpp": mean([float(r["score_no_bpp"]) for r in rows]),
        "score_diag": mean([float(r["score_diag"]) for r in rows]),
        "score_gate_fraction": mean([float(r["score_gate_full_rate_fraction"]) for r in rows]),
        "score_full": mean([float(r["score_full_branch_bpp"]) for r in rows]),
        "full_win_rate": mean([float(r["full_win"]) for r in rows]),
        "full_branch_dbpp": mean([float(r["full_branch_dbpp"]) for r in rows]),
        "diagnostic_dbpp": mean([float(r["diagnostic_dbpp"]) for r in rows]),
        "diag_rate_fraction": mean([float(r["diagnostic_rate_fraction"]) for r in rows]),
        "gate_rate_fraction": mean([float(r["gate_rate_fraction"]) for r in rows]),
        "max_rate_fraction_mean": mean(max_fracs),
        "max_rate_fraction_p10": quantile(max_fracs, 0.10),
        "max_rate_fraction_min": min(max_fracs) if max_fracs else float("nan"),
        "extra_dbpp_margin_after_full": mean([float(r["extra_dbpp_margin_after_full"]) for r in rows]),
    }


def summarize_curve(rows: list[dict[str, Any]], multipliers: list[float]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for group, subset in group_rows(rows):
        if not subset:
            continue
        for mult in multipliers:
            scores = [float(r["score_no_bpp"]) + mult * float(r["full_branch_dbpp"]) for r in subset]
            curve.append(
                {
                    "group": group,
                    "rate_multiplier": mult,
                    "mean_score": mean(scores),
                    "win_rate": mean([float(s < 0.0) for s in scores]),
                    "p90_score": quantile(scores, 0.90),
                    "max_score": max(scores),
                }
            )
    return curve


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    rows = build_rows(read_source(args.inputs), args.lpips_score_weight)
    if not rows:
        raise SystemExit("no soft/all-on row pairs found")

    groups = [summarize_group(name, subset) for name, subset in group_rows(rows) if subset]
    curve = summarize_curve(rows, args.rate_multipliers)
    failures = sorted(rows, key=lambda r: float(r["score_full_branch_bpp"]), reverse=True)[: args.failure_topk]

    out_prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    curve_path = out_prefix.with_name(out_prefix.name + "_curve.csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with curve_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    json_path.write_text(
        json.dumps({"groups": groups, "curve": curve, "failures": failures, "rows": rows}, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# E269 Progressive-Rate Margin Audit",
        "",
        "Purpose: measure how much branch rate the current soft-gated GLC reconstruction can afford before the guarded score becomes non-negative.",
        "",
        "## Group Summary",
        "",
        "| group | rows | no-bpp score | diagnostic score | gate-fraction score | full-branch score | full win | full dbpp | diag frac | max rate frac mean | max rate frac p10 | extra dbpp margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        lines.append(
            f"| {row['group']} | {row['rows']} | {row['score_no_bpp']:+.6f} | {row['score_diag']:+.6f} | "
            f"{row['score_gate_fraction']:+.6f} | {row['score_full']:+.6f} | {row['full_win_rate']:.3f} | "
            f"{row['full_branch_dbpp']:+.6f} | {row['diag_rate_fraction']:.3f} | {row['max_rate_fraction_mean']:.3f} | "
            f"{row['max_rate_fraction_p10']:.3f} | {row['extra_dbpp_margin_after_full']:+.6f} |"
        )
    lines.extend([
        "",
        "## Rate-Multiplier Curve (trained)",
        "",
        "| group | rate multiplier | mean score | win rate | p90 score | max score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in curve:
        if not str(row["group"]).startswith("trained"):
            continue
        lines.append(
            f"| {row['group']} | {row['rate_multiplier']:.2f} | {row['mean_score']:+.6f} | "
            f"{row['win_rate']:.3f} | {row['p90_score']:+.6f} | {row['max_score']:+.6f} |"
        )
    lines.extend([
        "",
        "## Highest-Risk Rows Under Full Branch Bpp",
        "",
        "| dataset | phase | image | full score | no-bpp score | full dbpp | max rate frac | gate | active rvq mse | H |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in failures:
        lines.append(
            f"| {row['dataset']} | {row['phase']} | {row['image']} | {float(row['score_full_branch_bpp']):+.6f} | "
            f"{float(row['score_no_bpp']):+.6f} | {float(row['full_branch_dbpp']):+.6f} | "
            f"{float(row['max_rate_fraction']):.3f} | {float(row['gate_mean']):.3f} | "
            f"{float(row['active_rvq_mse']):.6f} | {float(row['index_entropy_mean']):.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A max rate fraction above 1.0 means the current soft-gated reconstruction can pay the full observed branch bpp and still stay score-negative. Values below 1.0 identify rows that need selected/progressive bit savings or fallback.",
        "",
        "This is still an accounting audit over existing soft reconstructions, not a final entropy-coded codec. It defines the target budget for the next selected-index/progressive implementation.",
        "",
        "## Artifacts",
        "",
        f"- `{display_path(csv_path)}`",
        f"- `{display_path(curve_path)}`",
        f"- `{display_path(json_path)}`",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {display_path(md_path)}")


if __name__ == "__main__":
    main()
