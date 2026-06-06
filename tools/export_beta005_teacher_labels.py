#!/usr/bin/env python3
"""Export split-explicit teacher labels for a deployable reliability controller."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_beta005_decoder_safe_selector import build_rows, mean

ANALYSIS = ROOT / "experiments" / "analysis"
DEFAULT_OUT = ANALYSIS / "beta005_previous_local_teacher_labels.csv"
DEFAULT_MD = ANALYSIS / "beta005_previous_local_teacher_labels.md"
DELTA_RMS_THRESHOLD = 0.047937


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--delta-rms-threshold", type=float, default=DELTA_RMS_THRESHOLD)
    args = parser.parse_args()

    rows = build_rows()
    out_rows = []
    for row in rows:
        beta_rd = float(row["beta005_rd"])
        previous_rd = float(row["previous_local_rd"])
        previous_wins = previous_rd < beta_rd
        diagnostic_fallback = float(row["rvq_householder_delta_rms"]) > args.delta_rms_threshold
        out_rows.append(
            {
                "seed": row["seed"],
                "path": row["path"],
                "hcs_rd": row["hcs_rd"],
                "old_rd": row["old_rd"],
                "min090_rd": row["min090_rd"],
                "previous_local_rd": previous_rd,
                "beta005_rd": beta_rd,
                "margin_beta_minus_previous": beta_rd - previous_rd,
                "previous_local_wins": float(previous_wins),
                "householder_reliability_keep": 0.0 if previous_wins else 1.0,
                "diagnostic_delta_fallback": float(diagnostic_fallback),
                "diagnostic_reliability_keep": 0.0 if diagnostic_fallback else 1.0,
                "rvq_householder_gate_raw": row["rvq_householder_gate_raw"],
                "rvq_householder_delta_rms": row["rvq_householder_delta_rms"],
                "rvq_householder_strength": row["rvq_householder_strength"],
                "rvq_latent_quant_mse": row["rvq_latent_quant_mse"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    by_seed = {}
    for seed in sorted({str(row["seed"]) for row in out_rows}):
        seed_rows = [row for row in out_rows if str(row["seed"]) == seed]
        by_seed[seed] = {
            "rows": len(seed_rows),
            "previous_local_win_fraction": mean(float(row["previous_local_wins"]) for row in seed_rows),
            "diagnostic_delta_fallback_fraction": mean(float(row["diagnostic_delta_fallback"]) for row in seed_rows),
            "oracle_mean_rd": mean(min(float(row["beta005_rd"]), float(row["previous_local_rd"])) for row in seed_rows),
            "beta005_mean_rd": mean(float(row["beta005_rd"]) for row in seed_rows),
            "previous_local_mean_rd": mean(float(row["previous_local_rd"]) for row in seed_rows),
        }

    aggregate = {
        "rows": len(out_rows),
        "previous_local_win_fraction": mean(float(row["previous_local_wins"]) for row in out_rows),
        "diagnostic_delta_fallback_fraction": mean(float(row["diagnostic_delta_fallback"]) for row in out_rows),
        "oracle_mean_rd": mean(min(float(row["beta005_rd"]), float(row["previous_local_rd"])) for row in out_rows),
        "beta005_mean_rd": mean(float(row["beta005_rd"]) for row in out_rows),
        "previous_local_mean_rd": mean(float(row["previous_local_rd"]) for row in out_rows),
    }
    result = {"aggregate": aggregate, "by_seed": by_seed, "output": str(args.output)}

    lines = [
        "# Beta005 / previous-local teacher labels",
        "",
        "These labels are for training or auditing a single-checkpoint reliability controller. `householder_reliability_keep=1` means keep beta005-like geometry; `0` means the previous-local checkpoint had lower RD for that image.",
        "",
        "| split | rows | previous-local win frac | diagnostic delta fallback frac | oracle RD | beta005 RD | previous-local RD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| aggregate | {aggregate['rows']} | {fmt(aggregate['previous_local_win_fraction'])} | {fmt(aggregate['diagnostic_delta_fallback_fraction'])} | {fmt(aggregate['oracle_mean_rd'])} | {fmt(aggregate['beta005_mean_rd'])} | {fmt(aggregate['previous_local_mean_rd'])} |",
    ]
    for seed, item in by_seed.items():
        lines.append(
            f"| seed{seed} | {item['rows']} | {fmt(item['previous_local_win_fraction'])} | {fmt(item['diagnostic_delta_fallback_fraction'])} | {fmt(item['oracle_mean_rd'])} | {fmt(item['beta005_mean_rd'])} | {fmt(item['previous_local_mean_rd'])} |"
        )
    lines.extend(["", f"CSV: `{args.output.relative_to(ROOT)}`", ""])
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")

    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
