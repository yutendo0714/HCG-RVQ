#!/usr/bin/env python3
"""In-sample feature-risk audit for GLC q-curve replacement rows.

This is a diagnostic, not a paper claim.  It checks whether decoder-side feature
signals can explain the positive tail seen in E371 when q indexes are pooled.
PSNR columns are ignored completely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FEATURES = [
    "active_mse_ratio",
    "active_scalar_mse",
    "active_rvq_mse",
    "active_replacement_delta_bpp",
    "index_entropy_mean",
    "index_dead_frac_mean",
    "index_used_frac_mean",
    "gate_mean",
    "base_bpp",
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


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def read_rows(paths: list[Path], label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("label") != label:
                    continue
                score = fval(raw, "delta_dists") + 3.0 * fval(raw, "delta_lpips") + fval(raw, "delta_bpp")
                fixed_score = (
                    fval(raw, "delta_dists")
                    + 3.0 * fval(raw, "delta_lpips")
                    + fval(raw, "active_rvq_fixed_bpp")
                    - fval(raw, "active_scalar_bpp")
                )
                row = {
                    "q_index": int(fval(raw, "q_index", -1)),
                    "image": raw.get("image", ""),
                    "source": str(path),
                    "score": score,
                    "fixed_score": fixed_score,
                }
                for feature in FEATURES:
                    row[feature] = fval(raw, feature)
                rows.append(row)
    return rows


def evaluate_policy(rows: list[dict[str, object]], name: str, selected: list[dict[str, object]]) -> dict[str, object] | None:
    if not selected:
        return None
    scores = [float(row["score"]) for row in selected]
    fixed_scores = [float(row["fixed_score"]) for row in selected]
    return {
        "policy": name,
        "selected_rows": len(selected),
        "selected_frac": len(selected) / len(rows),
        "score_all": sum(scores) / len(rows),
        "fixed_score_all": sum(fixed_scores) / len(rows),
        "selected_mean_score": mean(scores),
        "selected_mean_fixed_score": mean(fixed_scores),
        "selected_win_frac": mean([1.0 if score < 0.0 else 0.0 for score in scores]),
        "selected_fixed_win_frac": mean([1.0 if score < 0.0 else 0.0 for score in fixed_scores]),
        "selected_worst_score": max(scores),
        "selected_worst_fixed_score": max(fixed_scores),
    }


def scan(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    policies: list[dict[str, object]] = []
    for qt in range(4):
        policy = evaluate_policy(rows, f"q_index <= {qt}", [row for row in rows if int(row["q_index"]) <= qt])
        if policy:
            policies.append(policy)
    for feature in FEATURES:
        values = sorted(float(row[feature]) for row in rows if math.isfinite(float(row[feature])))
        if not values:
            continue
        quantiles = [values[int((len(values) - 1) * pct)] for pct in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]
        for threshold in quantiles:
            for op in ["<=", ">="]:
                for qt in [None, 0, 1, 2, 3]:
                    if op == "<=":
                        selected = [row for row in rows if float(row[feature]) <= threshold and (qt is None or int(row["q_index"]) <= qt)]
                    else:
                        selected = [row for row in rows if float(row[feature]) >= threshold and (qt is None or int(row["q_index"]) <= qt)]
                    if len(selected) < 10:
                        continue
                    suffix = "" if qt is None else f" & q_index <= {qt}"
                    policy = evaluate_policy(rows, f"{feature} {op} {threshold:.6g}{suffix}", selected)
                    if policy:
                        policies.append(policy)
    policies.sort(key=lambda row: (float(row["selected_worst_score"]) > 0.0, float(row["score_all"])))
    return policies


def fmt(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if abs(value) < 1 and value != 0:
            return f"{value:+.6f}"
        return f"{value:.6f}"
    return str(value)


def table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Legacy single input CSV.")
    parser.add_argument("--inputs", type=Path, nargs="*", default=[], help="One or more input CSVs to pool.")
    parser.add_argument("--label", default="trained_replacement_soft")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.inputs)
    if args.input:
        paths.append(args.input)
    if not paths:
        raise SystemExit("at least one --input or --inputs path is required")
    rows = read_rows(paths, args.label)
    policies = scan(rows)
    q_summary = []
    for q in sorted({int(row["q_index"]) for row in rows}):
        subset = [row for row in rows if int(row["q_index"]) == q]
        q_summary.append(evaluate_policy(rows, f"all rows at q_index {q}", subset))
    oracle = evaluate_policy(rows, "oracle score < 0", [row for row in rows if float(row["score"]) < 0.0])

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"inputs": [str(path) for path in paths], "label": args.label, "q_summary": q_summary, "oracle": oracle, "policies": policies}
    with args.output_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    fields = [
        "policy",
        "selected_rows",
        "selected_frac",
        "score_all",
        "fixed_score_all",
        "selected_mean_score",
        "selected_win_frac",
        "selected_fixed_win_frac",
        "selected_worst_score",
        "selected_worst_fixed_score",
    ]
    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC Q-Curve Feature-Risk Audit\n\n")
        handle.write("Diagnostic only. PSNR is excluded. Unselected rows contribute zero to score_all.\n\n")
        handle.write("## q-index summary\n\n")
        handle.write("\n".join(table([row for row in q_summary if row], fields)))
        handle.write("\n\n")
        handle.write("## oracle\n\n")
        handle.write("\n".join(table([oracle] if oracle else [], fields)))
        handle.write("\n\n")
        handle.write("## top feature policies\n\n")
        handle.write("\n".join(table(policies[:25], fields)))
        handle.write("\n")


if __name__ == "__main__":
    main()
