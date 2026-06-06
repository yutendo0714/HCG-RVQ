#!/usr/bin/env python3
"""Export teacher labels from two path-aligned split evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else float("nan")


def fmt(value: float) -> str:
    return f"{value:.6f}"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row.get("has_nonfinite", "0")).lower() not in {"1", "true", "yes"}
    ]


def key_for(row: dict[str, str]) -> tuple[str, str]:
    return str(row["seed"]), str(row["path"])


def maybe_float(value: str | None) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=ANALYSIS / "beta005_transfer_openimages_start8192_n4096.csv",
        help="CSV containing the main/reference method rows.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=ANALYSIS / "local_cap080_rho1_transfer8192_checkpoint_sweep.csv",
        help="CSV containing the safer/fallback candidate rows.",
    )
    parser.add_argument("--reference-method", default="beta005 guard")
    parser.add_argument("--reference-name", default="beta005")
    parser.add_argument("--candidate-name", default="previous_local")
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS / "beta005_previous_local_teacher_labels_transfer8192.csv",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ANALYSIS / "beta005_previous_local_teacher_labels_transfer8192.md",
    )
    parser.add_argument(
        "--delta-rms-threshold",
        type=float,
        default=0.047937,
        help="Diagnostic-only threshold used for audit columns.",
    )
    args = parser.parse_args()

    reference_rows = [
        row
        for row in finite_rows(load_rows(args.reference_csv))
        if row.get("method") == args.reference_method
    ]
    candidate_rows = finite_rows(load_rows(args.candidate_csv))

    reference_by_key = {key_for(row): row for row in reference_rows}
    candidate_by_key = {key_for(row): row for row in candidate_rows}
    common_keys = sorted(set(reference_by_key) & set(candidate_by_key))
    if not common_keys:
        raise RuntimeError("no common (seed, path) rows between reference and candidate CSVs")
    missing_reference = len(candidate_by_key) - len(common_keys)
    missing_candidate = len(reference_by_key) - len(common_keys)

    out_rows: list[dict[str, object]] = []
    for seed, path in common_keys:
        reference = reference_by_key[(seed, path)]
        candidate = candidate_by_key[(seed, path)]
        reference_rd = float(reference["rd_score"])
        candidate_rd = float(candidate["rd_score"])
        candidate_wins = candidate_rd < reference_rd
        delta_rms = float(reference.get("rvq_householder_delta_rms") or 0.0)
        diagnostic_fallback = delta_rms > args.delta_rms_threshold

        out_rows.append(
            {
                "seed": seed,
                "path": path,
                f"{args.reference_name}_rd": reference_rd,
                f"{args.candidate_name}_rd": candidate_rd,
                f"margin_{args.reference_name}_minus_{args.candidate_name}": reference_rd
                - candidate_rd,
                f"{args.candidate_name}_wins": float(candidate_wins),
                "householder_reliability_keep": 0.0 if candidate_wins else 1.0,
                "diagnostic_delta_fallback": float(diagnostic_fallback),
                "diagnostic_reliability_keep": 0.0 if diagnostic_fallback else 1.0,
                "rvq_householder_gate_raw": maybe_float(reference.get("rvq_householder_gate_raw")),
                "rvq_householder_delta_rms": maybe_float(reference.get("rvq_householder_delta_rms")),
                "rvq_householder_strength": maybe_float(reference.get("rvq_householder_strength")),
                "rvq_latent_quant_mse": maybe_float(reference.get("rvq_latent_quant_mse")),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    by_seed: dict[str, dict[str, float]] = {}
    rows_by_seed: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in out_rows:
        rows_by_seed[str(row["seed"])].append(row)

    reference_key = f"{args.reference_name}_rd"
    candidate_key = f"{args.candidate_name}_rd"
    candidate_win_key = f"{args.candidate_name}_wins"
    for seed in sorted(rows_by_seed):
        seed_rows = rows_by_seed[seed]
        by_seed[seed] = {
            "rows": len(seed_rows),
            "candidate_win_fraction": mean(float(row[candidate_win_key]) for row in seed_rows),
            "diagnostic_delta_fallback_fraction": mean(
                float(row["diagnostic_delta_fallback"]) for row in seed_rows
            ),
            "oracle_mean_rd": mean(
                min(float(row[reference_key]), float(row[candidate_key])) for row in seed_rows
            ),
            "reference_mean_rd": mean(float(row[reference_key]) for row in seed_rows),
            "candidate_mean_rd": mean(float(row[candidate_key]) for row in seed_rows),
        }

    aggregate = {
        "rows": len(out_rows),
        "candidate_win_fraction": mean(float(row[candidate_win_key]) for row in out_rows),
        "diagnostic_delta_fallback_fraction": mean(
            float(row["diagnostic_delta_fallback"]) for row in out_rows
        ),
        "oracle_mean_rd": mean(
            min(float(row[reference_key]), float(row[candidate_key])) for row in out_rows
        ),
        "reference_mean_rd": mean(float(row[reference_key]) for row in out_rows),
        "candidate_mean_rd": mean(float(row[candidate_key]) for row in out_rows),
        "missing_reference_rows": missing_reference,
        "missing_candidate_rows": missing_candidate,
    }

    result = {
        "aggregate": aggregate,
        "by_seed": by_seed,
        "reference_csv": str(args.reference_csv),
        "candidate_csv": str(args.candidate_csv),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {args.reference_name} / {args.candidate_name} teacher labels",
        "",
        "These labels are generated on a split separate from the fixed holdout protocol. "
        "`householder_reliability_keep=1` means keep the reference geometry; `0` means the candidate row had lower RD for that image.",
        "",
        "| split | rows | candidate win frac | diagnostic delta fallback frac | oracle RD | reference RD | candidate RD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| aggregate | {aggregate['rows']} | {fmt(aggregate['candidate_win_fraction'])} | {fmt(aggregate['diagnostic_delta_fallback_fraction'])} | {fmt(aggregate['oracle_mean_rd'])} | {fmt(aggregate['reference_mean_rd'])} | {fmt(aggregate['candidate_mean_rd'])} |",
    ]
    for seed, item in by_seed.items():
        lines.append(
            f"| seed{seed} | {item['rows']} | {fmt(item['candidate_win_fraction'])} | {fmt(item['diagnostic_delta_fallback_fraction'])} | {fmt(item['oracle_mean_rd'])} | {fmt(item['reference_mean_rd'])} | {fmt(item['candidate_mean_rd'])} |"
        )
    lines.extend(
        [
            "",
            f"Missing reference rows: `{missing_reference}`",
            f"Missing candidate rows: `{missing_candidate}`",
            f"CSV: `{args.output.relative_to(ROOT)}`",
            "",
        ]
    )
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
