#!/usr/bin/env python3
"""Cross-split audit for GLC replacement-rate controllers.

E281 showed that a selected replacement branch is still useful under stricter
fixed-index accounting, but the cap choice can be optimistic if selected on the
same rows it evaluates. This script selects simple replacement-rate caps on one
split/domain and evaluates them on another. It is an analysis artifact, not a
new codec implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from analyze_e281_glc_replacement_accounting import CAPS, mean, num, read_replacement_rows


POLICIES = [
    "empirical_best",
    "fixed_best",
    "safe_empirical_best",
    "safe_fixed_best",
    "fixed_win095_best",
    "constant_0p0035",
    "constant_0p0040",
]


def row_key(row: dict[str, float | str], key: str) -> str:
    return str(row.get(key, ""))


def fmean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def selected_scores(rows: list[dict[str, float | str]], cap: float, score_key: str) -> list[float]:
    return [
        num(row, score_key) if num(row, "active_replacement_delta_bpp") <= cap else 0.0
        for row in rows
    ]


def selected_rows(rows: list[dict[str, float | str]], cap: float) -> list[dict[str, float | str]]:
    return [row for row in rows if num(row, "active_replacement_delta_bpp") <= cap]


def eval_cap(rows: list[dict[str, float | str]], cap: float) -> dict[str, float | int]:
    selected = selected_rows(rows, cap)
    emp_scores = selected_scores(rows, cap, "score")
    fixed_scores = selected_scores(rows, cap, "fixed_replacement_score")
    emp_dbpp = [
        num(row, "active_replacement_delta_bpp") if row in selected else 0.0
        for row in rows
    ]
    fixed_dbpp = [
        num(row, "fixed_replacement_delta_bpp") if row in selected else 0.0
        for row in rows
    ]
    return {
        "images": len(rows),
        "cap": cap,
        "selected_frac": len(selected) / len(rows) if rows else math.nan,
        "empirical_score": fmean(emp_scores),
        "fixed_score": fmean(fixed_scores),
        "empirical_delta_bpp": fmean(emp_dbpp),
        "fixed_delta_bpp": fmean(fixed_dbpp),
        "empirical_win_frac": fmean(1.0 if value < 0.0 else 0.0 for value in emp_scores),
        "fixed_win_frac": fmean(1.0 if value < 0.0 else 0.0 for value in fixed_scores),
        "selected_empirical_win_frac": fmean(1.0 if num(row, "score") < 0.0 else 0.0 for row in selected),
        "selected_fixed_win_frac": fmean(
            1.0 if num(row, "fixed_replacement_score") < 0.0 else 0.0 for row in selected
        ),
        "selected_mean_empirical_score": fmean(num(row, "score") for row in selected),
        "selected_mean_fixed_score": fmean(num(row, "fixed_replacement_score") for row in selected),
    }


def choose_cap(rows: list[dict[str, float | str]], policy: str) -> tuple[float, str]:
    if policy == "constant_0p0035":
        return 0.0035, "constant"
    if policy == "constant_0p0040":
        return 0.0040, "constant"

    evaluations = [(cap, eval_cap(rows, cap)) for cap in CAPS]
    valid: list[tuple[float, dict[str, float | int]]] = []
    objective = "empirical_score"
    note = policy

    if policy == "empirical_best":
        valid = evaluations
        objective = "empirical_score"
    elif policy == "fixed_best":
        valid = evaluations
        objective = "fixed_score"
    elif policy == "safe_empirical_best":
        valid = [
            item for item in evaluations if float(item[1]["selected_empirical_win_frac"]) >= 1.0
        ]
        objective = "empirical_score"
        note = "selected_empirical_win_frac>=1"
    elif policy == "safe_fixed_best":
        valid = [item for item in evaluations if float(item[1]["selected_fixed_win_frac"]) >= 1.0]
        objective = "fixed_score"
        note = "selected_fixed_win_frac>=1"
    elif policy == "fixed_win095_best":
        valid = [
            item for item in evaluations if float(item[1]["selected_fixed_win_frac"]) >= 0.95
        ]
        objective = "fixed_score"
        note = "selected_fixed_win_frac>=0.95"
    else:
        raise ValueError(f"unknown policy: {policy}")

    if not valid:
        valid = evaluations
        note += ";fallback_all_caps"

    # Prefer lower score, then larger selected fraction, then lower cap for ties.
    best = min(
        valid,
        key=lambda item: (
            float(item[1][objective]),
            -float(item[1]["selected_frac"]),
            item[0],
        ),
    )
    return best[0], note


def split_sets(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    splits: dict[str, list[dict[str, float | str]]] = {"all": rows}
    for key in ["domain", "source"]:
        for value in sorted({row_key(row, key) for row in rows}):
            splits[f"{key}:{value}"] = [row for row in rows if row_key(row, key) == value]
    return splits


def transfer_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    splits = split_sets(rows)
    eval_split_names = [name for name in splits if name != "all"]
    train_split_names = ["all"] + eval_split_names
    out: list[dict[str, float | int | str]] = []

    for train_name in train_split_names:
        train_rows = splits[train_name]
        if not train_rows:
            continue
        for policy in POLICIES:
            cap, note = choose_cap(train_rows, policy)
            train_eval = eval_cap(train_rows, cap)
            for test_name in ["all"] + eval_split_names:
                test_rows = splits[test_name]
                if not test_rows:
                    continue
                metrics = eval_cap(test_rows, cap)
                out.append(
                    {
                        "train_split": train_name,
                        "test_split": test_name,
                        "policy": policy,
                        "chosen_cap": cap,
                        "selection_note": note,
                        "train_images": len(train_rows),
                        "test_images": len(test_rows),
                        "train_empirical_score": train_eval["empirical_score"],
                        "train_fixed_score": train_eval["fixed_score"],
                        "test_selected_frac": metrics["selected_frac"],
                        "test_empirical_score": metrics["empirical_score"],
                        "test_fixed_score": metrics["fixed_score"],
                        "test_empirical_delta_bpp": metrics["empirical_delta_bpp"],
                        "test_fixed_delta_bpp": metrics["fixed_delta_bpp"],
                        "test_empirical_win_frac": metrics["empirical_win_frac"],
                        "test_fixed_win_frac": metrics["fixed_win_frac"],
                        "test_selected_empirical_win_frac": metrics["selected_empirical_win_frac"],
                        "test_selected_fixed_win_frac": metrics["selected_fixed_win_frac"],
                    }
                )
    return out


def leave_one_source(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    sources = sorted({row_key(row, "source") for row in rows})
    out: list[dict[str, float | int | str]] = []
    for held in sources:
        train_rows = [row for row in rows if row_key(row, "source") != held]
        test_rows = [row for row in rows if row_key(row, "source") == held]
        for policy in POLICIES:
            cap, note = choose_cap(train_rows, policy)
            metrics = eval_cap(test_rows, cap)
            train_metrics = eval_cap(train_rows, cap)
            out.append(
                {
                    "heldout_source": held,
                    "policy": policy,
                    "chosen_cap": cap,
                    "selection_note": note,
                    "train_images": len(train_rows),
                    "test_images": len(test_rows),
                    "train_empirical_score": train_metrics["empirical_score"],
                    "train_fixed_score": train_metrics["fixed_score"],
                    "test_selected_frac": metrics["selected_frac"],
                    "test_empirical_score": metrics["empirical_score"],
                    "test_fixed_score": metrics["fixed_score"],
                    "test_empirical_win_frac": metrics["empirical_win_frac"],
                    "test_fixed_win_frac": metrics["fixed_win_frac"],
                    "test_selected_empirical_win_frac": metrics["selected_empirical_win_frac"],
                    "test_selected_fixed_win_frac": metrics["selected_fixed_win_frac"],
                }
            )
    return out


def domain_transfer_focus(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    splits = split_sets(rows)
    pairs = [
        ("domain:kodak", "domain:clic"),
        ("domain:clic", "domain:kodak"),
        ("domain:clic", "all"),
        ("domain:kodak", "all"),
        ("all", "domain:clic"),
        ("all", "domain:kodak"),
    ]
    out: list[dict[str, float | int | str]] = []
    for train_name, test_name in pairs:
        train_rows = splits[train_name]
        test_rows = splits[test_name]
        for policy in POLICIES:
            cap, note = choose_cap(train_rows, policy)
            metrics = eval_cap(test_rows, cap)
            out.append(
                {
                    "train_split": train_name,
                    "test_split": test_name,
                    "policy": policy,
                    "chosen_cap": cap,
                    "selection_note": note,
                    "test_images": len(test_rows),
                    "test_selected_frac": metrics["selected_frac"],
                    "test_empirical_score": metrics["empirical_score"],
                    "test_fixed_score": metrics["fixed_score"],
                    "test_empirical_win_frac": metrics["empirical_win_frac"],
                    "test_fixed_win_frac": metrics["fixed_win_frac"],
                    "test_selected_empirical_win_frac": metrics["selected_empirical_win_frac"],
                    "test_selected_fixed_win_frac": metrics["selected_fixed_win_frac"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def md_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_replacement_rows(args.inputs)
    if not rows:
        raise SystemExit("no trained_replacement_soft rows found")

    transfer = transfer_rows(rows)
    loso = leave_one_source(rows)
    domain_focus = domain_transfer_focus(rows)

    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".transfer.csv"), transfer)
    write_csv(prefix.with_suffix(".leave_one_source.csv"), loso)
    write_csv(prefix.with_suffix(".domain_focus.csv"), domain_focus)

    payload = {
        "inputs": [str(path) for path in args.inputs],
        "rows": len(rows),
        "policies": POLICIES,
        "transfer": transfer,
        "leave_one_source": loso,
        "domain_focus": domain_focus,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    focus_keep = [
        row
        for row in domain_focus
        if row["policy"]
        in {
            "empirical_best",
            "fixed_best",
            "safe_empirical_best",
            "safe_fixed_best",
            "fixed_win095_best",
            "constant_0p0035",
            "constant_0p0040",
        }
    ]
    loso_keep = [
        row
        for row in loso
        if row["policy"] in {"fixed_win095_best", "constant_0p0035", "constant_0p0040"}
    ]

    md: list[str] = []
    md.append("# GLC Replacement Controller Transfer Audit")
    md.append("")
    md.append(
        "This audit selects simple replacement-rate caps on one split/domain and "
        "evaluates them on another. It checks whether the current cap choice is "
        "a robust controller candidate or a same-split artifact."
    )
    md.append("")
    md.append("## Domain Transfer Focus")
    md.extend(
        md_table(
            focus_keep,
            [
                "train_split",
                "test_split",
                "policy",
                "chosen_cap",
                "test_selected_frac",
                "test_empirical_score",
                "test_fixed_score",
                "test_selected_empirical_win_frac",
                "test_selected_fixed_win_frac",
            ],
        )
    )
    md.append("")
    md.append("## Leave-One-Source Focus")
    md.extend(
        md_table(
            loso_keep,
            [
                "heldout_source",
                "policy",
                "chosen_cap",
                "test_selected_frac",
                "test_empirical_score",
                "test_fixed_score",
                "test_selected_empirical_win_frac",
                "test_selected_fixed_win_frac",
            ],
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "If a cap chosen on Kodak transfers poorly to CLIC, the controller should "
        "not be trained or tuned only on the forgiving domain. If a conservative "
        "fixed-win policy chooses a smaller cap and preserves negative fixed "
        "score on held-out CLIC sources, it is the safer paper-facing controller. "
        "Aggressive constants can remain performance-search candidates, but they "
        "need failure-case reporting."
    )
    md.append("")
    prefix.with_suffix(".md").write_text("\n".join(md).rstrip() + "\n")


if __name__ == "__main__":
    main()
