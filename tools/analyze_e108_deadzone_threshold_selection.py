#!/usr/bin/env python3
"""Audit start8192 dead-zone threshold selection for the E099/E104 branch."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ANALYSIS = Path("experiments/analysis")
OUT_PREFIX = ANALYSIS / "e108_deadzone_transfer_threshold_selection_audit"
SEEDS = (1234, 2345, 3456)
TRANSFER_THRESHOLDS = ("014", "016", "018", "020")

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


def transfer_prefix(seed: int, threshold: str) -> Path:
    if seed == 3456:
        e_id = {"014": "e102", "016": "e103", "018": "e104", "020": "e105"}[threshold]
        return (
            ANALYSIS
            / f"{e_id}_e099_deadzone{threshold}_from_beta005_seed3456_step250_fullimage_start8192_current"
        )
    if threshold == "018":
        return ANALYSIS / f"e104_deadzone018_from_beta005_seed{seed}_step250_fullimage_start8192_current"
    return ANALYSIS / f"e108_deadzone{threshold}_from_beta005_seed{seed}_step250_fullimage_start8192_current"


def holdout_prefix(seed: int, threshold: str) -> Path:
    if seed == 3456:
        e_id = {"010": "e101", "012": "e100", "014": "e102", "016": "e103", "018": "e104"}[
            threshold
        ]
        return (
            ANALYSIS
            / f"{e_id}_e099_deadzone{threshold}_from_beta005_seed3456_step250_fullimage_holdout4096_current"
        )
    if threshold != "018":
        raise ValueError(f"holdout threshold {threshold} is unavailable for seed {seed}")
    return ANALYSIS / f"e104_deadzone018_from_beta005_seed{seed}_step250_fullimage_holdout4096_current"


def to_float(value: str) -> float:
    return float(value) if value not in ("", "nan", "NaN", "None") else float("nan")


def read_rows(csv_path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | int | str] = {
                "path": row.get("path", ""),
                "seed": row.get("seed", ""),
            }
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
        "win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
        "nonfinite_rows": int(sum(float(r.get("has_nonfinite", 0.0)) for r in rows)),
    }
    for column in FEATURE_COLUMNS:
        values = [float(r[column]) for r in rows if column in r and isinstance(r[column], float)]
        if values:
            summary[f"mean_{column}"] = mean(values)
    return summary


def quartile_rows(rows: list[dict[str, float | int | str]], split: str, threshold: str) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["reference_rd_score"]))
    n = len(ordered)
    out: list[dict[str, object]] = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4
        subset = ordered[lo:hi]
        out.append(
            {
                "split": split,
                "threshold": threshold,
                "quartile": f"Q{q + 1}",
                "num_images": len(subset),
                "mean_reference_rd": mean(float(r["reference_rd_score"]) for r in subset),
                "mean_delta": mean(float(r["rd_minus_reference"]) for r in subset),
                "win_rate": sum(1 for r in subset if float(r["rd_minus_reference"]) < 0.0)
                / len(subset),
            }
        )
    return out


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
    per_seed_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    quartiles: list[dict[str, object]] = []
    transfer_all: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)

    for threshold in TRANSFER_THRESHOLDS:
        for seed in SEEDS:
            csv_path = transfer_prefix(seed, threshold).with_suffix(".csv")
            rows = read_rows(csv_path)
            transfer_all[threshold].extend(rows)
            summary = summarize_rows(rows)
            per_seed_rows.append(
                {
                    "split": "transfer_start8192",
                    "threshold": threshold,
                    "seed": seed,
                    **summary,
                    "csv": str(csv_path),
                }
            )
        summary = summarize_rows(transfer_all[threshold])
        threshold_rows.append(
            {
                "split": "transfer_start8192",
                "threshold": threshold,
                "num_seeds": len(SEEDS),
                **summary,
            }
        )
        quartiles.extend(quartile_rows(transfer_all[threshold], "transfer_start8192", threshold))

    holdout_dz018_rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        rows = read_rows(holdout_prefix(seed, "018").with_suffix(".csv"))
        holdout_dz018_rows.extend(rows)
        summary = summarize_rows(rows)
        per_seed_rows.append(
            {
                "split": "holdout4096",
                "threshold": "018",
                "seed": seed,
                **summary,
                "csv": str(holdout_prefix(seed, "018").with_suffix(".csv")),
            }
        )
    threshold_rows.append(
        {
            "split": "holdout4096",
            "threshold": "018",
            "num_seeds": len(SEEDS),
            **summarize_rows(holdout_dz018_rows),
        }
    )
    quartiles.extend(quartile_rows(holdout_dz018_rows, "holdout4096", "018"))

    seed3456_holdout_rows: list[dict[str, object]] = []
    for threshold in ("010", "012", "014", "016", "018"):
        rows = read_rows(holdout_prefix(3456, threshold).with_suffix(".csv"))
        summary = summarize_rows(rows)
        seed3456_holdout_rows.append({"threshold": threshold, **summary})

    best_transfer = min(
        (r for r in threshold_rows if r["split"] == "transfer_start8192"),
        key=lambda row: float(row["mean_delta"]),
    )
    dz018_transfer = next(
        r for r in threshold_rows if r["split"] == "transfer_start8192" and r["threshold"] == "018"
    )
    dz018_holdout = next(
        r for r in threshold_rows if r["split"] == "holdout4096" and r["threshold"] == "018"
    )

    payload = {
        "decision": {
            "transfer_selected_threshold": best_transfer["threshold"],
            "transfer_selected_mean_delta": best_transfer["mean_delta"],
            "transfer_dz018_mean_delta": dz018_transfer["mean_delta"],
            "transfer_selected_minus_dz018": float(best_transfer["mean_delta"])
            - float(dz018_transfer["mean_delta"]),
            "holdout_dz018_mean_delta": dz018_holdout["mean_delta"],
            "interpretation": (
                "start8192 selects dz014, while dz018 remains the already-confirmed holdout-safe "
                "candidate; dz014 needs all-seed holdout confirmation before promotion."
            ),
        },
        "thresholds": threshold_rows,
        "per_seed": per_seed_rows,
        "quartiles": quartiles,
        "seed3456_holdout_sweep": seed3456_holdout_rows,
    }

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    write_csv(OUT_PREFIX.with_suffix(".per_seed.csv"), per_seed_rows)
    write_csv(OUT_PREFIX.with_suffix(".thresholds.csv"), threshold_rows)
    write_csv(OUT_PREFIX.with_suffix(".quartiles.csv"), quartiles)
    write_csv(OUT_PREFIX.with_suffix(".seed3456_holdout.csv"), seed3456_holdout_rows)

    lines = [
        "# E108 Dead-Zone Threshold Selection Audit",
        "",
        "## Decision",
        "",
        (
            f"The independent start8192 calibration split selects deadzone{best_transfer['threshold']} "
            f"with mean delta {format_float(best_transfer['mean_delta'])}. "
            f"Deadzone018 remains close at {format_float(dz018_transfer['mean_delta'])}, "
            f"a gap of {format_float(float(best_transfer['mean_delta']) - float(dz018_transfer['mean_delta']))} RD."
        ),
        "",
        (
            "This updates the protocol status: deadzone018 is the already-confirmed "
            "holdout-safe manuscript candidate, but a pre-declared transfer-split rule would "
            "choose deadzone014. Therefore deadzone014 should be confirmed on holdout for "
            "seed1234/2345 before replacing deadzone018."
        ),
        "",
        "## Transfer Thresholds",
        "",
        "| threshold | mean RD | ref RD | delta | win rate | q05 | q95 | qMSE | s_q | dead code | delta RMS | strength | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in threshold_rows if r["split"] == "transfer_start8192"]:
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
            "## Per-Seed Transfer Deltas",
            "",
            "| threshold | seed1234 | seed2345 | seed3456 |",
            "|---|---:|---:|---:|",
        ]
    )
    for threshold in TRANSFER_THRESHOLDS:
        rows = [
            r
            for r in per_seed_rows
            if r["split"] == "transfer_start8192" and r["threshold"] == threshold
        ]
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
            "## Holdout Anchors",
            "",
            (
                f"Deadzone018 holdout4096 3-seed delta is {format_float(dz018_holdout['mean_delta'])} "
                f"with win rate {format_float(dz018_holdout['win_rate'])} and nonfinite "
                f"{dz018_holdout['nonfinite_rows']}."
            ),
            "",
            "Seed3456 holdout sweep, retained as historical anchor:",
            "",
            "| threshold | delta | win rate | qMSE | s_q | delta RMS | strength |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed3456_holdout_rows:
        lines.append(
            f"| dz{row['threshold']} | {format_float(row['mean_delta'])} | "
            f"{format_float(row['win_rate'])} | {format_float(row.get('mean_rvq_latent_quant_mse'))} | "
            f"{format_float(row.get('mean_rvq_s_q_mean'))} | "
            f"{format_float(row.get('mean_rvq_householder_delta_rms'))} | "
            f"{format_float(row.get('mean_rvq_householder_strength'))} |"
        )
    lines.extend(
        [
            "",
            "## Transfer Quartiles",
            "",
            "| threshold | Q1 | Q2 | Q3 | Q4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for threshold in TRANSFER_THRESHOLDS:
        rows = [
            r
            for r in quartiles
            if r["split"] == "transfer_start8192" and r["threshold"] == threshold
        ]
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
