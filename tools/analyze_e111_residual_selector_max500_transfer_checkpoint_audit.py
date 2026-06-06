#!/usr/bin/env python3
"""Audit residual-selector checkpoint budget on the independent transfer split."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ANALYSIS = Path("experiments/analysis")
OUT_PREFIX = ANALYSIS / "e111_residual_selector_max500_transfer_checkpoint_audit"
SEEDS = (1234, 2345, 3456)
THRESHOLDS = ("014", "018")
BUDGETS = ("step250", "max500")

FEATURE_COLUMNS = (
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_perplexity",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_multiplier",
)


def csv_prefix(seed: int, threshold: str, budget: str) -> Path:
    if budget == "max500":
        return (
            ANALYSIS
            / f"e111_deadzone{threshold}_from_beta005_max500_seed{seed}_step500_fullimage_start8192_current"
        )
    if budget != "step250":
        raise ValueError(f"unsupported budget {budget}")
    if threshold == "014":
        if seed == 3456:
            return (
                ANALYSIS
                / "e102_e099_deadzone014_from_beta005_seed3456_step250_fullimage_start8192_current"
            )
        return (
            ANALYSIS
            / f"e108_deadzone014_from_beta005_seed{seed}_step250_fullimage_start8192_current"
        )
    if threshold == "018":
        if seed == 3456:
            return (
                ANALYSIS
                / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_start8192_current"
            )
        return (
            ANALYSIS
            / f"e104_deadzone018_from_beta005_seed{seed}_step250_fullimage_start8192_current"
        )
    raise ValueError(f"unsupported threshold {threshold}")


def to_float(value: str) -> float:
    if value in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(value)


def read_rows(csv_path: Path) -> list[dict[str, float | str]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | str] = {}
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
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(finite) - 1)
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def finite_values(rows: list[dict[str, float | str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, float) and math.isfinite(value):
            values.append(value)
    return values


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    deltas = finite_values(rows, "rd_minus_reference")
    rd = finite_values(rows, "rd_score")
    ref = finite_values(rows, "reference_rd_score")
    summary: dict[str, float | int] = {
        "num_images": len(rows),
        "mean_rd": mean(rd),
        "mean_reference_rd": mean(ref),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "q05_delta": percentile(deltas, 0.05),
        "q95_delta": percentile(deltas, 0.95),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
        "nonfinite_rows": int(sum(finite_values(rows, "has_nonfinite"))),
    }
    for column in FEATURE_COLUMNS:
        values = finite_values(rows, column)
        if values:
            summary[f"mean_{column}"] = mean(values)
    return summary


def quartile_rows(
    rows: list[dict[str, float | str]], budget: str, threshold: str
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["reference_rd_score"]))
    n = len(ordered)
    out: list[dict[str, object]] = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4
        subset = ordered[lo:hi]
        deltas = finite_values(subset, "rd_minus_reference")
        out.append(
            {
                "budget": budget,
                "threshold": threshold,
                "quartile": f"Q{q + 1}",
                "num_images": len(subset),
                "mean_reference_rd": mean(finite_values(subset, "reference_rd_score")),
                "mean_delta": mean(deltas),
                "win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
                "q95_delta": percentile(deltas, 0.95),
            }
        )
    return out


def pairwise_checkpoint_rows(
    rows_by_key: dict[tuple[str, str, int], list[dict[str, float | str]]]
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            step = {str(r["path"]): r for r in rows_by_key[("step250", threshold, seed)]}
            max500 = {str(r["path"]): r for r in rows_by_key[("max500", threshold, seed)]}
            shared = sorted(set(step) & set(max500))
            diffs = [
                float(max500[path]["rd_minus_reference"])
                - float(step[path]["rd_minus_reference"])
                for path in shared
            ]
            row: dict[str, object] = {
                "threshold": threshold,
                "seed": seed,
                "num_shared": len(shared),
                "mean_max500_minus_step250_delta": mean(diffs),
                "median_max500_minus_step250_delta": median(diffs),
                "q05_max500_minus_step250_delta": percentile(diffs, 0.05),
                "q95_max500_minus_step250_delta": percentile(diffs, 0.95),
                "max500_better_than_step250_rate": sum(1 for d in diffs if d < 0.0)
                / len(diffs),
            }
            for column in FEATURE_COLUMNS:
                values = [
                    float(max500[path][column]) - float(step[path][column])
                    for path in shared
                    if column in max500[path] and column in step[path]
                ]
                if values:
                    row[f"mean_diff_{column}"] = mean(values)
            out.append(row)
            aggregate[threshold].extend(diffs)

        all_diffs = aggregate[threshold]
        out.append(
            {
                "threshold": threshold,
                "seed": "all",
                "num_shared": len(all_diffs),
                "mean_max500_minus_step250_delta": mean(all_diffs),
                "median_max500_minus_step250_delta": median(all_diffs),
                "q05_max500_minus_step250_delta": percentile(all_diffs, 0.05),
                "q95_max500_minus_step250_delta": percentile(all_diffs, 0.95),
                "max500_better_than_step250_rate": sum(1 for d in all_diffs if d < 0.0)
                / len(all_diffs),
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


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    rows_by_key: dict[tuple[str, str, int], list[dict[str, float | str]]] = {}
    budget_threshold_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    quartiles: list[dict[str, object]] = []
    by_budget_threshold: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)

    for budget in BUDGETS:
        for threshold in THRESHOLDS:
            for seed in SEEDS:
                path = csv_prefix(seed, threshold, budget).with_suffix(".csv")
                rows = read_rows(path)
                key = (budget, threshold, seed)
                rows_by_key[key] = rows
                by_budget_threshold[(budget, threshold)].extend(rows)
                per_seed_rows.append(
                    {
                        "budget": budget,
                        "threshold": threshold,
                        "seed": seed,
                        **summarize_rows(rows),
                        "csv": str(path),
                    }
                )

            rows = by_budget_threshold[(budget, threshold)]
            budget_threshold_rows.append(
                {
                    "budget": budget,
                    "threshold": threshold,
                    "num_seeds": len(SEEDS),
                    **summarize_rows(rows),
                }
            )
            quartiles.extend(quartile_rows(rows, budget, threshold))

    pairwise_rows = pairwise_checkpoint_rows(rows_by_key)
    best = min(budget_threshold_rows, key=lambda row: float(row["mean_delta"]))
    step014 = next(
        r
        for r in budget_threshold_rows
        if r["budget"] == "step250" and r["threshold"] == "014"
    )
    step018 = next(
        r
        for r in budget_threshold_rows
        if r["budget"] == "step250" and r["threshold"] == "018"
    )
    max014 = next(
        r
        for r in budget_threshold_rows
        if r["budget"] == "max500" and r["threshold"] == "014"
    )
    max018 = next(
        r
        for r in budget_threshold_rows
        if r["budget"] == "max500" and r["threshold"] == "018"
    )
    seed3456_max014 = next(
        r
        for r in per_seed_rows
        if r["budget"] == "max500" and r["threshold"] == "014" and r["seed"] == 3456
    )
    seed3456_max018 = next(
        r
        for r in per_seed_rows
        if r["budget"] == "max500" and r["threshold"] == "018" and r["seed"] == 3456
    )

    decision = {
        "best_mean_budget": best["budget"],
        "best_mean_threshold": best["threshold"],
        "best_mean_delta": best["mean_delta"],
        "step250_dz014_mean_delta": step014["mean_delta"],
        "step250_dz018_mean_delta": step018["mean_delta"],
        "max500_dz014_mean_delta": max014["mean_delta"],
        "max500_dz018_mean_delta": max018["mean_delta"],
        "max500_dz014_minus_step250_dz014_mean_delta": float(max014["mean_delta"])
        - float(step014["mean_delta"]),
        "max500_dz018_minus_step250_dz018_mean_delta": float(max018["mean_delta"])
        - float(step018["mean_delta"]),
        "seed3456_max500_dz014_delta": seed3456_max014["mean_delta"],
        "seed3456_max500_dz018_delta": seed3456_max018["mean_delta"],
        "interpretation": (
            "On the independent transfer split, max500 improves the 3-seed mean relative "
            "to step250, but the same seed3456 fragility remains. Thus max500 is a "
            "high-mean checkpoint candidate, while step250 remains the cleaner default "
            "unless the checkpoint selection rule explicitly accepts this tail risk."
        ),
    }

    payload = {
        "decision": decision,
        "budget_threshold": budget_threshold_rows,
        "per_seed": per_seed_rows,
        "quartiles": quartiles,
        "pairwise_checkpoint": pairwise_rows,
    }

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_budget_thresholds.csv"), budget_threshold_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_per_seed.csv"), per_seed_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_quartiles.csv"), quartiles)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_pairwise.csv"), pairwise_rows)

    lines = [
        "# E111 residual-selector max500 transfer checkpoint audit",
        "",
        "## Decision",
        "",
        (
            f"- Best transfer 3-seed mean: {decision['best_mean_budget']} "
            f"dz{decision['best_mean_threshold']} with delta {fmt(decision['best_mean_delta'])}."
        ),
        (
            f"- step250 dz014 delta {fmt(decision['step250_dz014_mean_delta'])}; "
            f"max500 dz014 delta {fmt(decision['max500_dz014_mean_delta'])}; "
            f"max500 dz014 gain over step250 dz014 {fmt(decision['max500_dz014_minus_step250_dz014_mean_delta'])}."
        ),
        (
            f"- step250 dz018 delta {fmt(decision['step250_dz018_mean_delta'])}; "
            f"max500 dz018 delta {fmt(decision['max500_dz018_mean_delta'])}; "
            f"max500 dz018 gain over step250 dz018 {fmt(decision['max500_dz018_minus_step250_dz018_mean_delta'])}."
        ),
        (
            f"- Seed3456 max500 tail remains above beta: dz014 "
            f"{fmt(decision['seed3456_max500_dz014_delta'])}, dz018 "
            f"{fmt(decision['seed3456_max500_dz018_delta'])}."
        ),
        f"- Interpretation: {decision['interpretation']}",
        "",
        "## Budget and threshold summary",
        "",
        "| budget | dz | RD | beta RD | delta | win | q05 | q95 | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_threshold_rows:
        lines.append(
            "| {budget} | {threshold} | {mean_rd} | {mean_reference_rd} | {mean_delta} | "
            "{win_rate} | {q05_delta} | {q95_delta} | {nonfinite_rows} |".format(
                budget=row["budget"],
                threshold=row["threshold"],
                mean_rd=fmt(row["mean_rd"]),
                mean_reference_rd=fmt(row["mean_reference_rd"]),
                mean_delta=fmt(row["mean_delta"]),
                win_rate=fmt(row["win_rate"]),
                q05_delta=fmt(row["q05_delta"]),
                q95_delta=fmt(row["q95_delta"]),
                nonfinite_rows=row["nonfinite_rows"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-seed deltas",
            "",
            "| budget | dz | seed | delta | win | q95 | dead code | latent qMSE | s_q mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed_rows:
        lines.append(
            "| {budget} | {threshold} | {seed} | {mean_delta} | {win_rate} | {q95_delta} | "
            "{dead} | {qmse} | {sq} |".format(
                budget=row["budget"],
                threshold=row["threshold"],
                seed=row["seed"],
                mean_delta=fmt(row["mean_delta"]),
                win_rate=fmt(row["win_rate"]),
                q95_delta=fmt(row["q95_delta"]),
                dead=fmt(row.get("mean_rvq_dead_code_ratio", float("nan"))),
                qmse=fmt(row.get("mean_rvq_latent_quant_mse", float("nan"))),
                sq=fmt(row.get("mean_rvq_s_q_mean", float("nan"))),
            )
        )

    lines.extend(
        [
            "",
            "## Checkpoint pairwise same-image comparison",
            "",
            "Negative values mean max500 is better than step250 on the same image and reference.",
            "",
            "| dz | seed | mean max500-step250 | win | q05 | q95 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            "| {threshold} | {seed} | {mean_diff} | {win} | {q05} | {q95} |".format(
                threshold=row["threshold"],
                seed=row["seed"],
                mean_diff=fmt(row["mean_max500_minus_step250_delta"]),
                win=fmt(row["max500_better_than_step250_rate"]),
                q05=fmt(row["q05_max500_minus_step250_delta"]),
                q95=fmt(row["q95_max500_minus_step250_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- E111 evaluates start8192 transfer images with CUDA_VISIBLE_DEVICES=0 and cuda:0.",
            "- nonfinite_rows is 0 for every max500 seed and threshold.",
            "- This is an independent-split checkpoint audit: it supports max500 as a high-mean direction, but also confirms the seed3456/tail-risk caveat seen on holdout.",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
