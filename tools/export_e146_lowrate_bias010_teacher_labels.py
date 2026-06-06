#!/usr/bin/env python3
"""Export E146 teacher labels from E144 low-rate HCS/HCG transfer pairs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
INPUT = ANALYSIS / "e144_lowrate_bias010_transfer_to_holdout_controller.transfer_start8192_pairs.csv"
OUTPUT = ANALYSIS / "e146_lowrate_bias010_transfer8192_reliability_teacher_labels.csv"
SUMMARY_MD = ANALYSIS / "e146_lowrate_bias010_transfer8192_reliability_teacher_labels.md"


def parse(value: str) -> object:
    if value == "":
        return ""
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        if value == "True":
            return True
        if value == "False":
            return False
        return value


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return statistics.fmean(vals) if vals else float("nan")


def fmt(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "n/a"


def balanced_weight(rows: list[dict[str, object]], keep: float) -> float:
    positives = sum(float(row["householder_reliability_keep"]) == keep for row in rows)
    if positives == 0:
        return 0.0
    return len(rows) / (2.0 * positives)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    pairs = load_rows(INPUT)
    out_rows: list[dict[str, object]] = []
    abs_margins = [abs(float(row["hcg_minus_hcs"])) for row in pairs]
    margin_scale = statistics.median(abs_margins) if abs_margins else 1.0
    margin_scale = max(float(margin_scale), 1e-6)

    for row in pairs:
        hcg_minus_hcs = float(row["hcg_minus_hcs"])
        use_hcg = hcg_minus_hcs < 0.0
        margin_weight = min(abs(hcg_minus_hcs) / margin_scale, 4.0)
        out_rows.append(
            {
                "seed": row["seed"],
                "path": row["path"],
                "hcs_rd": float(row["hcs_rd"]),
                "hcg_rd": float(row["hcg_rd"]),
                "hcg_minus_hcs": hcg_minus_hcs,
                "householder_reliability_keep": 1.0 if use_hcg else 0.0,
                "householder_reliability_suppress": 0.0 if use_hcg else 1.0,
                "householder_reliability_weight_margin": margin_weight,
                "hcg_rvq_householder_strength": float(row["hcg_rvq_householder_strength"]),
                "hcg_rvq_s_q_mean": float(row["hcg_rvq_s_q_mean"]),
                "hcg_rvq_latent_quant_mse": float(row["hcg_rvq_latent_quant_mse"]),
                "hcg_rvq_dead_code_ratio": float(row["hcg_rvq_dead_code_ratio"]),
                "hcg_rvq_perplexity": float(row["hcg_rvq_perplexity"]),
            }
        )

    by_seed: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in out_rows:
        by_seed[str(row["seed"])].append(row)
    for seed_rows in by_seed.values():
        keep_weight = balanced_weight(seed_rows, 1.0)
        suppress_weight = balanced_weight(seed_rows, 0.0)
        for row in seed_rows:
            base = keep_weight if float(row["householder_reliability_keep"]) == 1.0 else suppress_weight
            row["householder_reliability_weight_balanced"] = base
            row["householder_reliability_weight_margin_balanced"] = base * float(row["householder_reliability_weight_margin"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    aggregate = {
        "rows": len(out_rows),
        "keep_fraction": mean([float(row["householder_reliability_keep"]) for row in out_rows]),
        "mean_hcg_minus_hcs": mean([float(row["hcg_minus_hcs"]) for row in out_rows]),
        "oracle_minus_hcs": mean([min(float(row["hcs_rd"]), float(row["hcg_rd"])) for row in out_rows])
        - mean([float(row["hcs_rd"]) for row in out_rows]),
        "hcs_rd": mean([float(row["hcs_rd"]) for row in out_rows]),
        "hcg_rd": mean([float(row["hcg_rd"]) for row in out_rows]),
    }
    seed_summary = {}
    for seed, rows in sorted(by_seed.items()):
        seed_summary[seed] = {
            "rows": len(rows),
            "keep_fraction": mean([float(row["householder_reliability_keep"]) for row in rows]),
            "mean_hcg_minus_hcs": mean([float(row["hcg_minus_hcs"]) for row in rows]),
            "oracle_minus_hcs": mean([min(float(row["hcs_rd"]), float(row["hcg_rd"])) for row in rows])
            - mean([float(row["hcs_rd"]) for row in rows]),
            "hcs_rd": mean([float(row["hcs_rd"]) for row in rows]),
            "hcg_rd": mean([float(row["hcg_rd"]) for row in rows]),
            "mean_weight_margin_balanced": mean([
                float(row["householder_reliability_weight_margin_balanced"]) for row in rows
            ]),
        }

    payload = {
        "input": str(INPUT.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "aggregate": aggregate,
        "by_seed": seed_summary,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E146 Low-Rate Bias010 Reliability Teacher Labels",
        "",
        "Labels are exported from the independent E144 transfer split. `householder_reliability_keep=1` means the fixed HCG bias010 row beats the matched HCS row for the same seed and image; `0` means suppress the geometry path for the reliability-head pilot.",
        "",
        "| split | rows | keep frac | HCG-HCS | oracle-HCS | HCS RD | HCG RD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| aggregate | {aggregate['rows']} | {fmt(aggregate['keep_fraction'])} | {fmt(aggregate['mean_hcg_minus_hcs'])} | {fmt(aggregate['oracle_minus_hcs'])} | {fmt(aggregate['hcs_rd'])} | {fmt(aggregate['hcg_rd'])} |",
    ]
    for seed, row in seed_summary.items():
        lines.append(
            f"| seed{seed} | {row['rows']} | {fmt(row['keep_fraction'])} | {fmt(row['mean_hcg_minus_hcs'])} | {fmt(row['oracle_minus_hcs'])} | {fmt(row['hcs_rd'])} | {fmt(row['hcg_rd'])} |"
        )
    lines.extend(["", f"CSV: `{OUTPUT.relative_to(ROOT)}`", ""])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
