#!/usr/bin/env python3
"""E238 teacher-label margin audit for EF-LIC local HCG controller training.

E236 showed a strong local-policy oracle. E237 showed that image-level shallow
selectors do not generalize. This script turns the E236 oracle into training
signals for the next in-codec local family/strength head:

* how much improvement is available when we fall back to zero on ambiguous cases,
* how confident the coarse family labels are,
* how costly each wrong family activation is,
* which labels should be used as high-confidence curriculum examples.

The script is intentionally post-hoc and CPU-only. It is not a performance
method; it defines supervision and loss priorities for the learned controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

E236_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e236_eflic_kodak24_local_controller_map.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e236_eflic_clicpro41_local_controller_map.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e238_eflic_teacher_label_margins",
    )
    return p.parse_args()


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def score_from_row(row: dict[str, str]) -> float:
    score = to_float(row.get("score_dists_3lpips"))
    if math.isfinite(score):
        return score
    return to_float(row.get("delta_dists"), 0.0) + 3.0 * to_float(row.get("delta_lpips"), 0.0)


def valid_contract(row: dict[str, str]) -> bool:
    return (
        abs(to_float(row.get("delta_bpp"), 1.0)) <= 1e-12
        and to_float(row.get("max_decode_diff"), 1.0) <= 1e-10
        and int(to_float(row.get("nonfinite"), 1.0)) == 0
        and int(to_float(row.get("payload_len_equal"), 0.0)) == 1
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_e236(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            for raw in csv.DictReader(fobj):
                if not valid_contract(raw):
                    continue
                item = (dataset, raw["image"])
                policy = raw["policy"]
                rows_by_item[item][policy] = {
                    "dataset": dataset,
                    "image": raw["image"],
                    "policy": policy,
                    "family": raw["family"],
                    "score": score_from_row(raw),
                    "delta_dists": to_float(raw.get("delta_dists"), 0.0),
                    "delta_lpips": to_float(raw.get("delta_lpips"), 0.0),
                    "delta_psnr": to_float(raw.get("delta_psnr"), 0.0),
                    "alpha_mean": to_float(raw.get("y_alpha_mean"), 0.0),
                    "alpha_active_frac": to_float(raw.get("y_alpha_active_frac"), 0.0),
                    "geometry_delta_rms": to_float(raw.get("y_avg_geometry_delta_rms"), 0.0),
                    "y_mismatch_frac": to_float(raw.get("y_mismatch"), 0.0) / max(1.0, to_float(raw.get("y_total"), 0.0)),
                    "y_index_entropy": to_float(raw.get("y_avg_index_entropy"), 0.0),
                    "y_index_used_frac": to_float(raw.get("y_avg_index_used_frac"), 0.0),
                }
    return dict(rows_by_item)


def common_policies(rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> list[str]:
    items = sorted(rows_by_item)
    if not items:
        raise SystemExit("no E236 rows found")
    common = set(rows_by_item[items[0]])
    for item in items[1:]:
        common &= set(rows_by_item[item])
    if "zero" not in common:
        raise SystemExit("zero fallback policy is missing")
    return sorted(common)


def build_labels(
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    policies: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    label_rows: list[dict[str, object]] = []
    family_score_rows: list[dict[str, object]] = []
    for item in sorted(rows_by_item):
        by_policy = rows_by_item[item]
        best_policy = min(policies, key=lambda p: float(by_policy[p]["score"]))
        zero_score = float(by_policy["zero"]["score"])
        sorted_policies = sorted(policies, key=lambda p: float(by_policy[p]["score"]))
        second_policy = sorted_policies[1] if len(sorted_policies) > 1 else best_policy

        family_best: dict[str, dict[str, Any]] = {}
        for policy in policies:
            row = by_policy[policy]
            family = str(row["family"])
            if family not in family_best or float(row["score"]) < float(family_best[family]["score"]):
                family_best[family] = row
        sorted_families = sorted(family_best, key=lambda f: float(family_best[f]["score"]))
        best_family = sorted_families[0]
        second_family = sorted_families[1] if len(sorted_families) > 1 else best_family
        best_row = by_policy[best_policy]

        for family in sorted(family_best):
            family_row = family_best[family]
            family_score_rows.append(
                {
                    "dataset": item[0],
                    "image": item[1],
                    "oracle_family": best_family,
                    "candidate_family": family,
                    "candidate_policy": family_row["policy"],
                    "candidate_score": family_row["score"],
                    "oracle_score": best_row["score"],
                    "regret_vs_oracle": float(family_row["score"]) - float(best_row["score"]),
                }
            )

        label_rows.append(
            {
                "dataset": item[0],
                "image": item[1],
                "oracle_policy": best_policy,
                "oracle_family": best_family,
                "oracle_score": best_row["score"],
                "zero_score": zero_score,
                "improvement_vs_zero": zero_score - float(best_row["score"]),
                "second_policy": second_policy,
                "policy_margin": float(by_policy[second_policy]["score"]) - float(best_row["score"]),
                "second_family": second_family,
                "family_margin": float(family_best[second_family]["score"]) - float(best_row["score"]),
                "active_label": int(float(best_row["score"]) < 0.0),
                "high_gain_5e4": int(zero_score - float(best_row["score"]) >= 5e-4),
                "high_margin_5e4": int(float(family_best[second_family]["score"]) - float(best_row["score"]) >= 5e-4),
                "alpha_mean": best_row["alpha_mean"],
                "alpha_active_frac": best_row["alpha_active_frac"],
                "geometry_delta_rms": best_row["geometry_delta_rms"],
                "y_mismatch_frac": best_row["y_mismatch_frac"],
                "y_index_entropy": best_row["y_index_entropy"],
                "y_index_used_frac": best_row["y_index_used_frac"],
            }
        )
    return label_rows, family_score_rows


def summarize_labels(label_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    datasets = sorted({str(r["dataset"]) for r in label_rows}) + ["pooled"]
    for dataset in datasets:
        rows = label_rows if dataset == "pooled" else [r for r in label_rows if r["dataset"] == dataset]
        if not rows:
            continue
        scores = [float(r["oracle_score"]) for r in rows]
        improvements = [float(r["improvement_vs_zero"]) for r in rows]
        margins = [float(r["family_margin"]) for r in rows]
        out.append(
            {
                "dataset": dataset,
                "images": len(rows),
                "oracle_score": mean(scores),
                "active_frac": mean(float(r["active_label"]) for r in rows),
                "family_counts": json.dumps(dict(Counter(str(r["oracle_family"]) for r in rows)), sort_keys=True),
                "policy_counts": json.dumps(dict(Counter(str(r["oracle_policy"]) for r in rows)), sort_keys=True),
                "mean_improvement_vs_zero": mean(improvements),
                "median_improvement_vs_zero": float(np.median(improvements)),
                "mean_family_margin": mean(margins),
                "median_family_margin": float(np.median(margins)),
                "ambiguous_family_margin_lt_1e4_frac": mean(float(v < 1e-4) for v in margins),
                "ambiguous_family_margin_lt_5e4_frac": mean(float(v < 5e-4) for v in margins),
                "gain_ge_5e4_frac": mean(float(v >= 5e-4) for v in improvements),
                "gain_ge_1e3_frac": mean(float(v >= 1e-3) for v in improvements),
            }
        )
    return out


def curriculum_summaries(label_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    gain_thresholds = [0.0, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
    margin_thresholds = [0.0, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3]
    out: list[dict[str, object]] = []
    datasets = sorted({str(r["dataset"]) for r in label_rows}) + ["pooled"]
    for dataset in datasets:
        rows = label_rows if dataset == "pooled" else [r for r in label_rows if r["dataset"] == dataset]
        if not rows:
            continue
        oracle_score = mean(float(r["oracle_score"]) for r in rows)
        for gain_t in gain_thresholds:
            for margin_t in margin_thresholds:
                chosen_scores = []
                activated = 0
                for row in rows:
                    use_oracle = (
                        float(row["improvement_vs_zero"]) >= gain_t
                        and float(row["family_margin"]) >= margin_t
                    )
                    chosen_scores.append(float(row["oracle_score"]) if use_oracle else 0.0)
                    activated += int(use_oracle)
                score = mean(chosen_scores)
                retained = (-score / -oracle_score) if oracle_score < 0.0 else float("nan")
                out.append(
                    {
                        "dataset": dataset,
                        "gain_threshold": gain_t,
                        "family_margin_threshold": margin_t,
                        "activated_frac": activated / len(rows),
                        "score": score,
                        "oracle_score": oracle_score,
                        "oracle_headroom_retained_frac": retained,
                        "score_win_frac": mean(float(v < 0.0) for v in chosen_scores),
                    }
                )
    return out


def family_cost_matrix(family_score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in family_score_rows:
        grouped[(str(row["dataset"]), str(row["oracle_family"]), str(row["candidate_family"]))].append(row)
        grouped[("pooled", str(row["oracle_family"]), str(row["candidate_family"]))].append(row)
    out: list[dict[str, object]] = []
    for (dataset, oracle_family, candidate_family), rows in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "oracle_family": oracle_family,
                "candidate_family": candidate_family,
                "images": len(rows),
                "candidate_score": mean(float(r["candidate_score"]) for r in rows),
                "oracle_score": mean(float(r["oracle_score"]) for r in rows),
                "regret_vs_oracle": mean(float(r["regret_vs_oracle"]) for r in rows),
                "positive_score_frac": mean(float(float(r["candidate_score"]) > 0.0) for r in rows),
            }
        )
    return out


def markdown_report(
    path: Path,
    *,
    label_summary: list[dict[str, object]],
    curriculum: list[dict[str, object]],
    family_costs: list[dict[str, object]],
) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if abs(value) < 10:
                return f"{value:+.6f}"
            return f"{value:.3f}"
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fobj:
        fobj.write("# E238 EF-LIC Teacher Label Margin Audit\n\n")
        fobj.write(
            "E238 converts the E236 local-policy oracle into supervision design "
            "for a learned EF-LIC local HCG controller. It is not a deployable "
            "selector and uses no GPU.\n\n"
        )
        fobj.write("## Oracle Label Summary\n\n")
        keys = [
            "dataset",
            "images",
            "oracle_score",
            "active_frac",
            "mean_improvement_vs_zero",
            "mean_family_margin",
            "ambiguous_family_margin_lt_5e4_frac",
            "gain_ge_5e4_frac",
        ]
        fobj.write("| " + " | ".join(keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for row in label_summary:
            fobj.write("| " + " | ".join(fmt(row.get(k, "")) for k in keys) + " |\n")
        fobj.write("\n## Best Curriculum Rows\n\n")
        fobj.write(
            "Rows below show how much oracle headroom is retained if ambiguous or "
            "small-gain labels are forced to zero fallback.\n\n"
        )
        for dataset in ["pooled", "kodak24", "clicpro41"]:
            subset = [r for r in curriculum if r["dataset"] == dataset and float(r["activated_frac"]) <= 0.75]
            subset = sorted(subset, key=lambda r: (float(r["oracle_headroom_retained_frac"]), -float(r["activated_frac"])), reverse=True)[:8]
            fobj.write(f"### {dataset}\n\n")
            keys2 = [
                "gain_threshold",
                "family_margin_threshold",
                "activated_frac",
                "score",
                "oracle_headroom_retained_frac",
            ]
            fobj.write("| " + " | ".join(keys2) + " |\n")
            fobj.write("|" + "|".join(["---"] * len(keys2)) + "|\n")
            for row in subset:
                fobj.write("| " + " | ".join(fmt(row.get(k, "")) for k in keys2) + " |\n")
            fobj.write("\n")
        fobj.write("## Largest Pooled Wrong-Family Costs\n\n")
        pooled_wrong = [
            r
            for r in family_costs
            if r["dataset"] == "pooled" and r["oracle_family"] != r["candidate_family"]
        ]
        pooled_wrong = sorted(pooled_wrong, key=lambda r: float(r["regret_vs_oracle"]), reverse=True)[:12]
        keys3 = ["oracle_family", "candidate_family", "images", "candidate_score", "regret_vs_oracle", "positive_score_frac"]
        fobj.write("| " + " | ".join(keys3) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys3)) + "|\n")
        for row in pooled_wrong:
            fobj.write("| " + " | ".join(fmt(row.get(k, "")) for k in keys3) + " |\n")
        fobj.write(
            "\nConclusion: the next controller should use high-confidence family "
            "labels first, train a zero/fallback logit explicitly, and use an "
            "asymmetric loss that penalizes wrong nonzero families more strongly "
            "than missed small gains.\n"
        )


def main() -> None:
    args = parse_args()
    rows_by_item = read_e236(E236_INPUTS)
    policies = common_policies(rows_by_item)
    labels, family_scores = build_labels(rows_by_item, policies)
    label_summary = summarize_labels(labels)
    curriculum = curriculum_summaries(labels)
    family_costs = family_cost_matrix(family_scores)

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".labels.csv"), labels)
    write_csv(prefix.with_suffix(".curriculum.csv"), curriculum)
    write_csv(prefix.with_suffix(".family_costs.csv"), family_costs)
    write_csv(prefix.with_suffix(".summary.csv"), label_summary)
    payload = {
        "experiment": "E238 EF-LIC teacher label margin audit",
        "inputs": {key: str(value) for key, value in E236_INPUTS.items()},
        "policies": policies,
        "items": len(labels),
        "label_summary": label_summary,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_report(
        prefix.with_suffix(".md"),
        label_summary=label_summary,
        curriculum=curriculum,
        family_costs=family_costs,
    )
    print(f"wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
