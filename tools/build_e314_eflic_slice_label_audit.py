#!/usr/bin/env python3
"""Build EF-LIC HCG slice-label diagnostics from E313 subset sweeps.

E313 measures active-slice subsets in the real codec loop. This script converts
those rows into slice-level label candidates for a future sequential controller.
It is intentionally an audit artifact, not a deployable policy trainer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL_SLICES = (0, 1, 2, 3)
FEATURE_SUFFIXES = (
    "active_logit_mean",
    "alpha_active_frac",
    "alpha_mean",
    "avg_geometry_delta_rms",
    "avg_index_entropy",
    "avg_index_used_frac",
    "avg_residual_error_rms",
    "family_zero_prob_mean",
    "gate_mean",
    "local_score_mean",
    "risk_score_mean",
    "stage0_geometry_delta_rms",
    "stage0_index_entropy",
    "stage0_index_used_frac",
    "stage0_residual_error_rms",
    "strength_mean",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/analysis/e313_eflic_slice_isolation_sweep_kodak4.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e314_eflic_slice_label_audit_kodak4",
    )
    p.add_argument("--eps", type=float, default=1e-9)
    return p.parse_args()


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def active_set(label: str) -> set[int]:
    if label == "all":
        return set(ALL_SLICES)
    if label == "none" or not label:
        return set()
    return {int(part) for part in label.split(",") if part != ""}


def label_for_slices(slices: set[int]) -> str:
    if slices == set(ALL_SLICES):
        return "all"
    if not slices:
        return "none"
    return ",".join(str(s) for s in sorted(slices))


def mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def pearson(x: list[float], y: list[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return float("nan")
    xa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ya = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_image.setdefault(row["image"], {})[row["active_slices"]] = row

    slice_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for image, table in sorted(by_image.items()):
        all_row = table["all"]
        all_delta = fnum(all_row, "delta_psnr")
        best_label, best_row = max(table.items(), key=lambda item: fnum(item[1], "delta_psnr"))
        best_delta = fnum(best_row, "delta_psnr")
        worst_label, worst_row = min(table.items(), key=lambda item: fnum(item[1], "delta_psnr"))
        policy_rows.append(
            {
                "image": image,
                "all_delta_psnr": all_delta,
                "best_slice_set": best_label,
                "best_delta_psnr": best_delta,
                "best_gain_over_all": best_delta - all_delta,
                "worst_slice_set": worst_label,
                "worst_delta_psnr": fnum(worst_row, "delta_psnr"),
                "best_slice_count": len(active_set(best_label)),
            }
        )
        best_slices = active_set(best_label)
        for s in ALL_SLICES:
            single_label = str(s)
            leave_label = label_for_slices(set(ALL_SLICES) - {s})
            single_row = table[single_label]
            leave_row = table[leave_label]
            single_delta = fnum(single_row, "delta_psnr")
            leave_delta = fnum(leave_row, "delta_psnr")
            contextual_margin = all_delta - leave_delta
            oracle_active = int(s in best_slices)
            single_positive = int(single_delta > 0.0)
            contextual_positive = int(contextual_margin > 0.0)
            row: dict[str, Any] = {
                "image": image,
                "slice": s,
                "all_delta_psnr": all_delta,
                "single_delta_psnr": single_delta,
                "leave_one_out_delta_psnr": leave_delta,
                "contextual_margin_psnr": contextual_margin,
                "oracle_active": oracle_active,
                "single_positive": single_positive,
                "contextual_positive": contextual_positive,
                "best_slice_set": best_label,
                "best_delta_psnr": best_delta,
                "best_gain_over_all": best_delta - all_delta,
                "single_vs_context_agree": int(single_positive == contextual_positive),
                "oracle_vs_single_agree": int(oracle_active == single_positive),
                "oracle_vs_context_agree": int(oracle_active == contextual_positive),
            }
            for suffix in FEATURE_SUFFIXES:
                row[f"single_{suffix}"] = fnum(single_row, f"slice{s}_{suffix}")
                row[f"allctx_{suffix}"] = fnum(all_row, f"slice{s}_{suffix}")
            slice_rows.append(row)

    feature_rows: list[dict[str, Any]] = []
    if slice_rows:
        candidate_features = [f"single_{suffix}" for suffix in FEATURE_SUFFIXES] + [f"allctx_{suffix}" for suffix in FEATURE_SUFFIXES]
        for feature in candidate_features:
            values = [fnum(r, feature) for r in slice_rows]
            feature_rows.append(
                {
                    "feature": feature,
                    "corr_oracle_active": pearson(values, [fnum(r, "oracle_active") for r in slice_rows]),
                    "corr_contextual_margin": pearson(values, [fnum(r, "contextual_margin_psnr") for r in slice_rows]),
                    "corr_single_delta": pearson(values, [fnum(r, "single_delta_psnr") for r in slice_rows]),
                    "mean_active": mean([v for v, r in zip(values, slice_rows) if int(r["oracle_active"]) == 1]),
                    "mean_inactive": mean([v for v, r in zip(values, slice_rows) if int(r["oracle_active"]) == 0]),
                    "rows": len(values),
                }
            )
        def corr_sort_key(row: dict[str, Any]) -> float:
            value = fnum(row, "corr_oracle_active", float("nan"))
            return abs(value) if math.isfinite(value) else -1.0

        feature_rows.sort(key=corr_sort_key, reverse=True)
    return slice_rows, policy_rows, feature_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def policy_summary(policy_rows: list[dict[str, Any]], slice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        {
            "policy": "all",
            "mean_delta_psnr": mean([fnum(r, "all_delta_psnr") for r in policy_rows]),
            "worst_delta_psnr": min(fnum(r, "all_delta_psnr") for r in policy_rows),
            "mean_gain_over_all": 0.0,
            "images": len(policy_rows),
        }
    )
    out.append(
        {
            "policy": "best_per_image_oracle",
            "mean_delta_psnr": mean([fnum(r, "best_delta_psnr") for r in policy_rows]),
            "worst_delta_psnr": min(fnum(r, "best_delta_psnr") for r in policy_rows),
            "mean_gain_over_all": mean([fnum(r, "best_gain_over_all") for r in policy_rows]),
            "images": len(policy_rows),
        }
    )
    for s in ALL_SLICES:
        vals = [fnum(r, "single_delta_psnr") for r in slice_rows if int(r["slice"]) == s]
        all_vals = [fnum(r, "all_delta_psnr") for r in slice_rows if int(r["slice"]) == s]
        out.append(
            {
                "policy": f"single_slice{s}",
                "mean_delta_psnr": mean(vals),
                "worst_delta_psnr": min(vals),
                "mean_gain_over_all": mean([v - a for v, a in zip(vals, all_vals)]),
                "images": len(vals),
            }
        )
    return out


def write_outputs(args: argparse.Namespace, slice_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    policies = policy_summary(policy_rows, slice_rows)
    write_csv(args.output_prefix.with_suffix(".slice_labels.csv"), slice_rows)
    write_csv(args.output_prefix.with_suffix(".policy_summary.csv"), policies)
    write_csv(args.output_prefix.with_suffix(".feature_correlations.csv"), feature_rows)
    summary = {
        "experiment": "E314 EF-LIC slice label audit",
        "source_rows": str(args.rows),
        "images": len(policy_rows),
        "slice_labels": len(slice_rows),
        "oracle_active_frac": mean([fnum(r, "oracle_active") for r in slice_rows]),
        "single_context_agreement": mean([fnum(r, "single_vs_context_agree") for r in slice_rows]),
        "oracle_single_agreement": mean([fnum(r, "oracle_vs_single_agree") for r in slice_rows]),
        "oracle_context_agreement": mean([fnum(r, "oracle_vs_context_agree") for r in slice_rows]),
        "policies": policies,
        "top_feature_correlations": feature_rows[:12],
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    md = args.output_prefix.with_suffix(".md")
    with md.open("w") as fobj:
        fobj.write("# E314 EF-LIC Slice Label Audit\n\n")
        fobj.write("This converts E313 subset sweeps into slice-level label diagnostics for a future sequential controller. It is not final RD evidence.\n\n")
        fobj.write(f"- Source: `{args.rows}`\n")
        fobj.write(f"- Images: `{summary['images']}`\n")
        fobj.write(f"- Slice labels: `{summary['slice_labels']}`\n")
        fobj.write(f"- Oracle active fraction: `{summary['oracle_active_frac']:.6f}`\n")
        fobj.write(f"- Single/context sign agreement: `{summary['single_context_agreement']:.6f}`\n")
        fobj.write(f"- Oracle/single agreement: `{summary['oracle_single_agreement']:.6f}`\n")
        fobj.write(f"- Oracle/context agreement: `{summary['oracle_context_agreement']:.6f}`\n\n")
        fobj.write("## Policy Summary\n\n")
        fobj.write("| policy | images | mean_delta_psnr | worst_delta_psnr | mean_gain_over_all |\n")
        fobj.write("|---|---:|---:|---:|---:|\n")
        for row in policies:
            fobj.write(f"| {row['policy']} | {row['images']} | {row['mean_delta_psnr']:+.6f} | {row['worst_delta_psnr']:+.6f} | {row['mean_gain_over_all']:+.6f} |\n")
        fobj.write("\n## Top Feature Correlations With Oracle Active\n\n")
        fobj.write("| feature | corr_oracle_active | corr_contextual_margin | corr_single_delta | mean_active | mean_inactive |\n")
        fobj.write("|---|---:|---:|---:|---:|---:|\n")
        for row in feature_rows[:12]:
            fobj.write(
                f"| {row['feature']} | {fnum(row, 'corr_oracle_active'):+.6f} | {fnum(row, 'corr_contextual_margin'):+.6f} | {fnum(row, 'corr_single_delta'):+.6f} | {fnum(row, 'mean_active'):+.6f} | {fnum(row, 'mean_inactive'):+.6f} |\n"
            )
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- The oracle row is an upper bound over tested E313 subsets, not deployable.\n")
        fobj.write("- Low oracle/single or oracle/context agreement means controller labels need sequential context rather than single-slice signs.\n")
        fobj.write("- Feature correlations are tiny-sample diagnostics only; use them to choose candidate inputs, not to claim transfer.\n")
    print(f"wrote {args.output_prefix}.{{slice_labels,policy_summary,feature_correlations}}.csv and {args.output_prefix}.{{json,md}}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.rows)
    slice_rows, policy_rows, feature_rows = build(rows)
    write_outputs(args, slice_rows, policy_rows, feature_rows)


if __name__ == "__main__":
    main()
