#!/usr/bin/env python3
"""E261 domain-robust controller calibration audit.

This audit follows E260.  The E260 controller module is healthy, but the small
MLP overfits the existing GLC rows under LOOCV/leave-domain evaluation.  E261
asks a narrower question before any full-training promotion: can simple,
interpretable branch diagnostics select a non-harmful local HCG/RVQ branch
under domain-robust calibration?

The script intentionally avoids score/delta columns as predictors.  It uses
active residual, index, and rate diagnostics that can be measured by a candidate
branch, then evaluates threshold rules with resubstitution, LOOCV,
leave-domain, and leave-variant protocols.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]

BASE_FEATURES = [
    "active_mse_ratio",
    "active_scalar_mse",
    "active_rvq_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "empirical_bpp_delta",
    "fixed_bpp_delta",
    "base_bpp",
    "base_psnr",
    "base_ms_ssim",
    "base_lpips",
    "base_dists",
    "area",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "analysis"
        / "e257_glc_domain_mixed_with_cliccalib_gate_readiness.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "experiments"
        / "analysis"
        / "e261_domain_robust_controller_calibration",
    )
    p.add_argument("--margins", type=float, nargs="*", default=[0.0, 5e-4, 1e-3, 2e-3])
    return p.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            row["score_with_side"] = finite(raw.get("score_with_side"))
            row["oracle_select"] = as_bool(raw.get("oracle_select"))
            for key in BASE_FEATURES:
                row[key] = finite(raw.get(key))
            row["active_mse_gain"] = row["active_scalar_mse"] - row["active_rvq_mse"]
            row["active_mse_log_ratio"] = (
                math.log(max(row["active_mse_ratio"], 1e-12))
                if math.isfinite(row["active_mse_ratio"])
                else float("nan")
            )
            row["bpp_delta_gap"] = row["empirical_bpp_delta"] - row["fixed_bpp_delta"]
            row["index_entropy_per_used"] = (
                row["index_entropy_mean"] / max(row["index_used_frac_mean"], 1e-12)
                if math.isfinite(row["index_entropy_mean"]) and math.isfinite(row["index_used_frac_mean"])
                else float("nan")
            )
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows loaded from {path}")
    return rows


FEATURES = BASE_FEATURES + [
    "active_mse_gain",
    "active_mse_log_ratio",
    "bpp_delta_gap",
    "index_entropy_per_used",
]


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str | None = None
    direction: str | None = None
    threshold: float | None = None

    def apply(self, row: dict[str, Any]) -> bool:
        if self.name == "no_branch":
            return False
        if self.name == "all_on":
            return True
        if self.name == "oracle":
            return bool(row["oracle_select"])
        value = finite(row.get(self.feature))
        if not math.isfinite(value):
            return False
        if self.direction == "le":
            return value <= float(self.threshold)
        if self.direction == "ge":
            return value >= float(self.threshold)
        raise ValueError(f"unknown rule direction: {self.direction}")

    def label(self) -> str:
        if self.feature is None:
            return self.name
        return f"{self.feature}_{self.direction}_{self.threshold:.8g}"


@dataclass(frozen=True)
class Guard:
    name: str
    margin: float = 0.0
    group_key: str | None = None
    group_margin: float = 0.0


def selected_scores(rows: list[dict[str, Any]], selected: list[bool]) -> list[float]:
    return [finite(row["score_with_side"]) if use else 0.0 for row, use in zip(rows, selected)]


def summarize_selection(rows: list[dict[str, Any]], selected: list[bool], policy: str, rule_label: str = "") -> dict[str, Any]:
    scores = selected_scores(rows, selected)
    selected_rows = [row for row, use in zip(rows, selected) if use]
    selected_raw_scores = [finite(row["score_with_side"]) for row in selected_rows]
    return {
        "policy": policy,
        "rule": rule_label,
        "selected": sum(1 for use in selected if use),
        "total": len(rows),
        "selected_frac": sum(1 for use in selected if use) / len(rows) if rows else 0.0,
        "mean_score": mean(scores),
        "selected_mean_score": mean(selected_raw_scores),
        "selected_win_rate": mean([1.0 if score < 0.0 else 0.0 for score in selected_raw_scores]),
    }


def rule_selected(rows: list[dict[str, Any]], rule: Rule) -> list[bool]:
    return [rule.apply(row) for row in rows]


def thresholds(values: list[float]) -> list[float]:
    vals = sorted({v for v in values if math.isfinite(v)})
    if not vals:
        return []
    points = [vals[0] - 1e-12, vals[-1] + 1e-12]
    points.extend(vals)
    points.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(points))


def candidate_rules(rows: list[dict[str, Any]]) -> list[Rule]:
    rules = [Rule("no_branch")]
    for feature in FEATURES:
        vals = [finite(row.get(feature)) for row in rows]
        for threshold in thresholds(vals):
            rules.append(Rule("threshold", feature=feature, direction="le", threshold=threshold))
            rules.append(Rule("threshold", feature=feature, direction="ge", threshold=threshold))
    return rules


def guard_accepts(rows: list[dict[str, Any]], selected: list[bool], guard: Guard) -> bool:
    summary = summarize_selection(rows, selected, "train")
    if summary["selected"] == 0:
        return True
    if summary["mean_score"] > -guard.margin:
        return False
    if guard.group_key is None:
        return True
    for group in sorted({str(row.get(guard.group_key)) for row in rows}):
        idx = [i for i, row in enumerate(rows) if str(row.get(guard.group_key)) == group]
        group_rows = [rows[i] for i in idx]
        group_selected = [selected[i] for i in idx]
        group_summary = summarize_selection(group_rows, group_selected, "group")
        if group_summary["mean_score"] > -guard.group_margin:
            return False
    return True


def choose_rule(rows: list[dict[str, Any]], guard: Guard) -> Rule:
    best = Rule("no_branch")
    best_summary = summarize_selection(rows, rule_selected(rows, best), "train", best.label())
    for rule in candidate_rules(rows):
        selected = rule_selected(rows, rule)
        if not guard_accepts(rows, selected, guard):
            continue
        summary = summarize_selection(rows, selected, "train", rule.label())
        rank = 0 if rule.name == "no_branch" else 1
        best_rank = 0 if best.name == "no_branch" else 1
        key = (summary["mean_score"], -summary["selected"], rank, rule.label())
        best_key = (best_summary["mean_score"], -best_summary["selected"], best_rank, best.label())
        if key < best_key:
            best = rule
            best_summary = summary
    return best


def evaluate_protocol(
    rows: list[dict[str, Any]],
    policy: str,
    guard: Guard,
    folds: list[tuple[str, list[int], list[int]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [False] * len(rows)
    fold_rules: list[str] = []
    per_row: list[dict[str, Any]] = []
    for fold_name, train_idx, eval_idx in folds:
        train_rows = [rows[i] for i in train_idx]
        rule = choose_rule(train_rows, guard)
        fold_rules.append(rule.label())
        for idx in eval_idx:
            use = rule.apply(rows[idx])
            selected[idx] = use
            per_row.append(
                {
                    "policy": policy,
                    "fold": fold_name,
                    "guard": guard.name,
                    "rule": rule.label(),
                    "domain": rows[idx].get("domain", ""),
                    "variant": rows[idx].get("variant", ""),
                    "image": rows[idx].get("image", ""),
                    "selected": use,
                    "score_if_selected": finite(rows[idx]["score_with_side"]),
                    "realized_score": finite(rows[idx]["score_with_side"]) if use else 0.0,
                }
            )
    summary = summarize_selection(rows, selected, f"{policy}_{guard.name}", "")
    counts = Counter(fold_rules)
    summary["unique_rules"] = len(counts)
    summary["most_common_rule"] = counts.most_common(1)[0][0] if counts else ""
    return summary, per_row


def folds_resub(rows: list[dict[str, Any]]) -> list[tuple[str, list[int], list[int]]]:
    idx = list(range(len(rows)))
    return [("all", idx, idx)]


def folds_loocv(rows: list[dict[str, Any]]) -> list[tuple[str, list[int], list[int]]]:
    idx = list(range(len(rows)))
    return [(str(i), [j for j in idx if j != i], [i]) for i in idx]


def folds_leave_group(rows: list[dict[str, Any]], group_key: str) -> list[tuple[str, list[int], list[int]]]:
    folds = []
    all_idx = list(range(len(rows)))
    for group in sorted({str(row.get(group_key)) for row in rows}):
        eval_idx = [i for i, row in enumerate(rows) if str(row.get(group_key)) == group]
        train_idx = [i for i in all_idx if i not in set(eval_idx)]
        folds.append((group, train_idx, eval_idx))
    return folds


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    guards: list[Guard],
    summaries: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    fieldnames = [
        "policy",
        "rule",
        "selected",
        "total",
        "selected_frac",
        "mean_score",
        "selected_mean_score",
        "selected_win_rate",
        "unique_rules",
        "most_common_rule",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    selection_path = args.output_prefix.with_suffix(".selected.csv")
    selection_fields = [
        "policy",
        "fold",
        "guard",
        "rule",
        "domain",
        "variant",
        "image",
        "selected",
        "score_if_selected",
        "realized_score",
    ]
    with selection_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selection_fields)
        writer.writeheader()
        for row in selections:
            writer.writerow({key: row.get(key, "") for key in selection_fields})

    payload = {
        "rows": str(args.rows),
        "n_rows": len(rows),
        "features": FEATURES,
        "guards": [guard.__dict__ for guard in guards],
        "summaries": summaries,
        "note": "Threshold calibration audit only; held-out positive scores block full-training claims.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(value: Any) -> str:
        val = finite(value)
        return "NA" if not math.isfinite(val) else f"{val:.6f}"

    heldout = [
        row
        for row in summaries
        if row["policy"].startswith(("loocv_", "leave_domain_", "leave_variant_"))
    ]
    robust_nonzero = [row for row in heldout if int(row["selected"]) > 0 and finite(row["mean_score"]) <= 0.0]
    destructive = [row for row in heldout if int(row["selected"]) > 0 and finite(row["mean_score"]) > 0.0]

    lines = [
        "# E261 Domain-Robust Controller Calibration",
        "",
        "E261 tests whether simple, interpretable diagnostics can safely select the "
        "GLC local branch before promoting the design to codec-loop/full training.",
        "Score columns and reconstruction deltas are used only as labels, not predictors.",
        "",
        "## Baselines",
        "",
        "| policy | selected | mean score | selected mean | win rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if row["policy"] in {"no_branch", "all_on", "oracle"}:
            lines.append(
                f"| {row['policy']} | {int(row['selected'])}/{int(row['total'])} | "
                f"{fmt(row['mean_score'])} | {fmt(row['selected_mean_score'])} | {fmt(row['selected_win_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Calibration Protocols",
            "",
            "| policy | selected | mean score | selected mean | win rate | unique rules | most common rule |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summaries:
        if row["policy"] in {"no_branch", "all_on", "oracle"}:
            continue
        lines.append(
            f"| {row['policy']} | {int(row['selected'])}/{int(row['total'])} | "
            f"{fmt(row['mean_score'])} | {fmt(row['selected_mean_score'])} | "
            f"{fmt(row['selected_win_rate'])} | {int(row.get('unique_rules', 0) or 0)} | "
            f"`{row.get('most_common_rule', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Held-out nonzero non-harmful policies: `{len(robust_nonzero)}`.",
            f"- Held-out nonzero harmful policies: `{len(destructive)}`.",
        ]
    )
    if robust_nonzero:
        best = min(robust_nonzero, key=lambda row: finite(row["mean_score"]))
        lines.append(
            f"- Best held-out nonzero policy: `{best['policy']}` with mean score `{fmt(best['mean_score'])}`."
        )
    else:
        lines.append(
            "- No held-out nonzero threshold policy is safe enough yet; keep no-branch fallback and move "
            "the controller into codec-loop training or collect stronger calibration labels."
        )
    lines.append("")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.rows)
    guards = [
        Guard("free", margin=0.0),
        *[Guard(f"margin{margin:g}", margin=margin) for margin in args.margins if margin > 0.0],
        Guard("domain_nonharmful", margin=0.0, group_key="domain", group_margin=0.0),
        Guard("variant_nonharmful", margin=0.0, group_key="variant", group_margin=0.0),
        Guard("domain_margin5e-4", margin=5e-4, group_key="domain", group_margin=0.0),
        Guard("variant_margin5e-4", margin=5e-4, group_key="variant", group_margin=0.0),
    ]

    summaries = [
        summarize_selection(rows, [False] * len(rows), "no_branch", "no_branch"),
        summarize_selection(rows, [True] * len(rows), "all_on", "all_on"),
        summarize_selection(rows, [bool(row["oracle_select"]) for row in rows], "oracle", "oracle"),
    ]
    selections: list[dict[str, Any]] = []
    protocols: list[tuple[str, Callable[[list[dict[str, Any]]], list[tuple[str, list[int], list[int]]]]]] = [
        ("resub", folds_resub),
        ("loocv", folds_loocv),
        ("leave_domain", lambda rs: folds_leave_group(rs, "domain")),
        ("leave_variant", lambda rs: folds_leave_group(rs, "variant")),
    ]
    for protocol_name, fold_builder in protocols:
        folds = fold_builder(rows)
        for guard in guards:
            summary, per_row = evaluate_protocol(rows, protocol_name, guard, folds)
            summaries.append(summary)
            selections.extend(per_row)

    write_outputs(args, rows, guards, summaries, selections)
    print(
        json.dumps(
            {
                "output_prefix": str(args.output_prefix),
                "n_rows": len(rows),
                "summary_csv": str(args.output_prefix.with_suffix(".summary.csv")),
                "md": str(args.output_prefix.with_suffix(".md")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
