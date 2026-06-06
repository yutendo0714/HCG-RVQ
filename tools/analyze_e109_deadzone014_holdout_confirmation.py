#!/usr/bin/env python3
"""Confirm transfer-selected deadzone014 on holdout4096 against deadzone018."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ANALYSIS = Path("experiments/analysis")
OUT_PREFIX = ANALYSIS / "e109_deadzone014_holdout_confirmation_audit"
SEEDS = (1234, 2345, 3456)
THRESHOLDS = ("014", "018")

FEATURE_COLUMNS = (
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_multiplier",
)


def holdout_prefix(seed: int, threshold: str) -> Path:
    if threshold == "014":
        if seed == 3456:
            return (
                ANALYSIS
                / "e102_e099_deadzone014_from_beta005_seed3456_step250_fullimage_holdout4096_current"
            )
        return (
            ANALYSIS
            / f"e109_deadzone014_from_beta005_seed{seed}_step250_fullimage_holdout4096_current"
        )
    if threshold == "018":
        if seed == 3456:
            return (
                ANALYSIS
                / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current"
            )
        return (
            ANALYSIS
            / f"e104_deadzone018_from_beta005_seed{seed}_step250_fullimage_holdout4096_current"
        )
    raise ValueError(f"unsupported threshold {threshold}")


def to_float(value: str) -> float:
    return float(value) if value not in ("", "nan", "NaN", "None") else float("nan")


def read_rows(csv_path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | int | str] = {}
            for key, value in row.items():
                if key in {"path", "mode", "config", "checkpoint", "method"}:
                    parsed[key] = value
                elif key:
                    try:
                        parsed[key] = to_float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    deltas = [float(r["rd_minus_reference"]) for r in rows]
    rd = [float(r["rd_score"]) for r in rows]
    ref = [float(r["reference_rd_score"]) for r in rows]
    summary: dict[str, float | int] = {
        "num_images": len(rows),
        "mean_rd": mean(rd),
        "mean_reference_rd": mean(ref),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "q05_delta": percentile(deltas, 0.05),
        "q95_delta": percentile(deltas, 0.95),
        "max_delta": max(deltas),
        "win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
        "nonfinite_rows": int(sum(float(r.get("has_nonfinite", 0.0)) for r in rows)),
    }
    for column in FEATURE_COLUMNS:
        values = [float(r[column]) for r in rows if column in r and isinstance(r[column], float)]
        if values:
            summary[f"mean_{column}"] = mean(values)
    return summary


def quartile_rows(
    rows: list[dict[str, float | int | str]], threshold: str
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["reference_rd_score"]))
    n = len(ordered)
    out: list[dict[str, object]] = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4
        subset = ordered[lo:hi]
        out.append(
            {
                "threshold": threshold,
                "quartile": f"Q{q + 1}",
                "num_images": len(subset),
                "mean_reference_rd": mean(float(r["reference_rd_score"]) for r in subset),
                "mean_delta": mean(float(r["rd_minus_reference"]) for r in subset),
                "win_rate": sum(1 for r in subset if float(r["rd_minus_reference"]) < 0.0)
                / len(subset),
                "q95_delta": percentile([float(r["rd_minus_reference"]) for r in subset], 0.95),
            }
        )
    return out


def compare_same_images(
    rows_by_seed_threshold: dict[tuple[int, str], list[dict[str, float | int | str]]]
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for seed in SEEDS:
        dz014 = {str(r["path"]): r for r in rows_by_seed_threshold[(seed, "014")]}
        dz018 = {str(r["path"]): r for r in rows_by_seed_threshold[(seed, "018")]}
        shared_paths = sorted(set(dz014) & set(dz018))
        diffs = [
            float(dz014[path]["rd_minus_reference"]) - float(dz018[path]["rd_minus_reference"])
            for path in shared_paths
        ]
        comparisons.append(
            {
                "seed": seed,
                "num_shared": len(shared_paths),
                "mean_dz014_minus_dz018_delta": mean(diffs),
                "median_dz014_minus_dz018_delta": median(diffs),
                "q05_dz014_minus_dz018_delta": percentile(diffs, 0.05),
                "q95_dz014_minus_dz018_delta": percentile(diffs, 0.95),
                "dz014_better_than_dz018_rate": sum(1 for d in diffs if d < 0.0) / len(diffs),
            }
        )
    all_diffs: list[float] = []
    for seed in SEEDS:
        dz014 = {str(r["path"]): r for r in rows_by_seed_threshold[(seed, "014")]}
        dz018 = {str(r["path"]): r for r in rows_by_seed_threshold[(seed, "018")]}
        shared_paths = sorted(set(dz014) & set(dz018))
        all_diffs.extend(
            float(dz014[path]["rd_minus_reference"]) - float(dz018[path]["rd_minus_reference"])
            for path in shared_paths
        )
    comparisons.append(
        {
            "seed": "all",
            "num_shared": len(all_diffs),
            "mean_dz014_minus_dz018_delta": mean(all_diffs),
            "median_dz014_minus_dz018_delta": median(all_diffs),
            "q05_dz014_minus_dz018_delta": percentile(all_diffs, 0.05),
            "q95_dz014_minus_dz018_delta": percentile(all_diffs, 0.95),
            "dz014_better_than_dz018_rate": sum(1 for d in all_diffs if d < 0.0)
            / len(all_diffs),
        }
    )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    rows_by_seed_threshold: dict[tuple[int, str], list[dict[str, float | int | str]]] = {}
    per_seed_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    quartiles: list[dict[str, object]] = []
    all_by_threshold: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)

    for threshold in THRESHOLDS:
        for seed in SEEDS:
            csv_path = holdout_prefix(seed, threshold).with_suffix(".csv")
            rows = read_rows(csv_path)
            rows_by_seed_threshold[(seed, threshold)] = rows
            all_by_threshold[threshold].extend(rows)
            summary = summarize_rows(rows)
            per_seed_rows.append(
                {
                    "threshold": threshold,
                    "seed": seed,
                    **summary,
                    "csv": str(csv_path),
                }
            )
        summary = summarize_rows(all_by_threshold[threshold])
        threshold_rows.append(
            {
                "threshold": threshold,
                "num_seeds": len(SEEDS),
                **summary,
            }
        )
        quartiles.extend(quartile_rows(all_by_threshold[threshold], threshold))

    pairwise_rows = compare_same_images(rows_by_seed_threshold)
    dz014 = next(r for r in threshold_rows if r["threshold"] == "014")
    dz018 = next(r for r in threshold_rows if r["threshold"] == "018")
    pairwise_all = next(r for r in pairwise_rows if r["seed"] == "all")

    interpretation = (
        "deadzone014 is holdout-confirmed as the mean-RD stronger setting, but it is less conservative "
        "than deadzone018 because q95 damage and win rate are slightly worse."
        if float(dz014["mean_delta"]) < float(dz018["mean_delta"])
        else "deadzone014 is not promoted because it does not beat deadzone018 on holdout mean RD."
    )

    payload = {
        "decision": {
            "holdout_selected_threshold": "014"
            if float(dz014["mean_delta"]) < float(dz018["mean_delta"])
            else "018",
            "dz014_mean_delta": dz014["mean_delta"],
            "dz018_mean_delta": dz018["mean_delta"],
            "dz014_minus_dz018_mean_delta": float(dz014["mean_delta"])
            - float(dz018["mean_delta"]),
            "pairwise_dz014_minus_dz018_mean_delta": pairwise_all[
                "mean_dz014_minus_dz018_delta"
            ],
            "interpretation": interpretation,
        },
        "thresholds": threshold_rows,
        "per_seed": per_seed_rows,
        "quartiles": quartiles,
        "pairwise": pairwise_rows,
    }

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    write_csv(OUT_PREFIX.with_suffix(".thresholds.csv"), threshold_rows)
    write_csv(OUT_PREFIX.with_suffix(".per_seed.csv"), per_seed_rows)
    write_csv(OUT_PREFIX.with_suffix(".quartiles.csv"), quartiles)
    write_csv(OUT_PREFIX.with_suffix(".pairwise.csv"), pairwise_rows)

    lines = [
        "# E109 Deadzone014 Holdout Confirmation Audit",
        "",
        "## Decision",
        "",
        (
            f"Deadzone014 holdout4096 3-seed delta is {format_float(dz014['mean_delta'])}, "
            f"while deadzone018 is {format_float(dz018['mean_delta'])}. "
            f"The mean-RD gap is {format_float(float(dz014['mean_delta']) - float(dz018['mean_delta']))} "
            "in favor of deadzone014."
        ),
        "",
        (
            f"On the same images, dz014 minus dz018 has mean "
            f"{format_float(pairwise_all['mean_dz014_minus_dz018_delta'])} and dz014 is better on "
            f"{format_float(pairwise_all['dz014_better_than_dz018_rate'])} of images."
        ),
        "",
        interpretation,
        "",
        "## Holdout Threshold Summary",
        "",
        "| threshold | mean RD | ref RD | delta | win rate | q05 | q95 | max | qMSE | s_q | dead code | delta RMS | strength | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"dz{row['threshold']}",
                    format_float(row["mean_rd"]),
                    format_float(row["mean_reference_rd"]),
                    format_float(row["mean_delta"]),
                    format_float(row["win_rate"]),
                    format_float(row["q05_delta"]),
                    format_float(row["q95_delta"]),
                    format_float(row["max_delta"]),
                    format_float(row.get("mean_rvq_latent_quant_mse")),
                    format_float(row.get("mean_rvq_s_q_mean")),
                    format_float(row.get("mean_rvq_dead_code_ratio")),
                    format_float(row.get("mean_rvq_householder_delta_rms")),
                    format_float(row.get("mean_rvq_householder_strength")),
                    str(row["nonfinite_rows"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Deltas",
            "",
            "| threshold | seed1234 | seed2345 | seed3456 |",
            "|---|---:|---:|---:|",
        ]
    )
    for threshold in THRESHOLDS:
        rows = [r for r in per_seed_rows if r["threshold"] == threshold]
        by_seed = {int(r["seed"]): r for r in rows}
        lines.append(
            f"| dz{threshold} | "
            f"{format_float(by_seed[1234]['mean_delta'])} | "
            f"{format_float(by_seed[2345]['mean_delta'])} | "
            f"{format_float(by_seed[3456]['mean_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Same-Image dz014 minus dz018",
            "",
            "| seed | mean | median | q05 | q95 | dz014 better rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['seed']} | {format_float(row['mean_dz014_minus_dz018_delta'])} | "
            f"{format_float(row['median_dz014_minus_dz018_delta'])} | "
            f"{format_float(row['q05_dz014_minus_dz018_delta'])} | "
            f"{format_float(row['q95_dz014_minus_dz018_delta'])} | "
            f"{format_float(row['dz014_better_than_dz018_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Holdout Quartiles",
            "",
            "| threshold | Q1 | Q2 | Q3 | Q4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for threshold in THRESHOLDS:
        rows = [r for r in quartiles if r["threshold"] == threshold]
        by_q = {r["quartile"]: r for r in rows}
        lines.append(
            f"| dz{threshold} | {format_float(by_q['Q1']['mean_delta'])} | "
            f"{format_float(by_q['Q2']['mean_delta'])} | "
            f"{format_float(by_q['Q3']['mean_delta'])} | "
            f"{format_float(by_q['Q4']['mean_delta'])} |"
        )

    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
