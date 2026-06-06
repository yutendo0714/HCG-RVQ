#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402
import analyze_e187_eflic_selector_splitfit as e187  # noqa: E402
import analyze_e190_eflic_multiobjective_selector as e190  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005.csv"),
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005_feature_manifest.csv"),
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e192_eflic_force0_two_stage_selector"),
    )
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--primary-feature", default="slice0_mean_min")
    p.add_argument("--primary-op", choices=[">=", "<="], default=">=")
    p.add_argument("--primary-threshold", type=float, default=-10.7447786331)
    p.add_argument("--feature-set", default="global_predecision_context")
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--positive-penalty", type=float, default=20.0)
    return p.parse_args()


def primary_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[bool]:
    return e184.threshold_decisions(rows, args.primary_feature, args.primary_threshold, args.primary_op)


def score(rows: list[dict[str, Any]], decisions: list[bool], args: argparse.Namespace) -> float:
    return e190.multiobjective_score(
        rows,
        decisions,
        args.dists_weight,
        args.lpips_weight,
        args.psnr_weight,
        args.positive_penalty,
    )


def best_secondary(
    rows: list[dict[str, Any]],
    features: list[str],
    args: argparse.Namespace,
) -> tuple[str, str, float, list[bool], float]:
    primary = primary_decisions(rows, args)
    best: tuple[str, str, float, list[bool], float] = ("none", ">=", math.nan, primary, score(rows, primary, args))
    for feature in e184.valid_features(rows, features):
        values = [float(r[feature]) for r in rows]
        for threshold in e184.candidate_thresholds(values):
            for direction in (">=", "<="):
                secondary = e184.threshold_decisions(rows, feature, threshold, direction)
                decisions = [a and b for a, b in zip(primary, secondary)]
                value = score(rows, decisions, args)
                if value < best[4]:
                    best = (feature, direction, threshold, decisions, value)
    return best


def summarize(
    group: str,
    selector: str,
    rows: list[dict[str, Any]],
    decisions: list[bool],
    args: argparse.Namespace,
    feature_set: str = "",
    rule: str = "",
) -> dict[str, Any]:
    out = e184.summarize_policy(group, selector, rows, decisions, 0.0, "lpips", feature_set, rule)
    out["multiobjective_score"] = score(rows, decisions, args)
    out["selected_win_lpips"] = sum(
        (float(r["active_lpips"]) - float(r["base_lpips"])) < 0 for r, d in zip(rows, decisions) if d
    )
    out["dists_weight"] = args.dists_weight
    out["lpips_weight"] = args.lpips_weight
    out["psnr_weight"] = args.psnr_weight
    out["positive_penalty"] = args.positive_penalty
    return out


def loocv(rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> tuple[list[bool], str]:
    decisions = []
    rules = []
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1 :]
        feature, direction, threshold, _, _ = best_secondary(train, features, args)
        if feature == "none":
            decision = primary_decisions([row], args)[0]
            rule = "primary_only"
        else:
            p = primary_decisions([row], args)[0]
            s = e184.threshold_decisions([row], feature, threshold, direction)[0]
            decision = p and s
            rule = f"{feature} {direction} {threshold:.6g}"
        decisions.append(decision)
        rules.append(rule)
    unique = sorted(set(rules))
    return decisions, "; ".join(unique) if len(unique) <= 6 else f"{len(unique)} fold-specific rules"


def split_rows(split: str, rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    train, eval_rows = e187.split_rows(rows, split)
    feature, direction, threshold, train_decisions, _ = best_secondary(train, features, args)
    if feature == "none":
        eval_decisions = primary_decisions(eval_rows, args)
        rule = "primary_only"
    else:
        primary_eval = primary_decisions(eval_rows, args)
        secondary_eval = e184.threshold_decisions(eval_rows, feature, threshold, direction)
        eval_decisions = [a and b for a, b in zip(primary_eval, secondary_eval)]
        rule = f"{feature} {direction} {threshold:.9g}"
    return [
        summarize(split, "train_primary", train, primary_decisions(train, args), args, args.feature_set, primary_rule(args)),
        summarize(split, "train_two_stage", train, train_decisions, args, args.feature_set, rule),
        summarize(split, "eval_primary", eval_rows, primary_decisions(eval_rows, args), args, args.feature_set, primary_rule(args)),
        summarize(split, "eval_apply_two_stage", eval_rows, eval_decisions, args, args.feature_set, rule),
    ]


def primary_rule(args: argparse.Namespace) -> str:
    return f"{args.primary_feature} {args.primary_op} {args.primary_threshold:.9g}"


def write_outputs(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E192 EF-LIC Two-Stage Global Selector Audit",
        "",
        "This diagnostic tests whether the E190 rule can be strengthened by a second no-side-bit global-predecision scalar condition. It is still a Kodak diagnostic unless run on an independent fit split.",
        "",
        f"Primary rule: `{primary_rule(args)}`",
        f"Weights: DISTS `{args.dists_weight}`, LPIPS `{args.lpips_weight}`, PSNR `{args.psnr_weight}`, positive penalty `{args.positive_penalty}`",
        "",
        "| group | selector | branch share | dDISTS | dLPIPS | dPSNR | score | DISTS wins | LPIPS wins | rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row['branch_share']:.3f} | "
            f"{row['selected_delta_dists']:+.6f} | {row['selected_delta_lpips']:+.6f} | "
            f"{row['selected_delta_psnr']:+.6f} | {row['multiobjective_score']:+.6f} | "
            f"{row['selected_win_dists']}/{row['images']} | {row['selected_win_lpips']}/{row['images']} | {row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- The secondary condition is conjunctive: active branch is used only when both primary and secondary rules pass.",
            "- `loocv_two_stage` and `eval_apply_two_stage` are anti-overfit checks; same-table `best_two_stage` is only a design headroom row.",
            "- If the secondary condition is unstable across splits, prefer E190 primary-only for paper-facing held-out evaluation.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(prefix.with_suffix(".md"))


def main() -> None:
    args = parse_args()
    rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in rows if int(float(r["force_ind"])) == args.force]
    if not rows:
        raise SystemExit(f"no finite rows for force{args.force}")
    manifest = e184.read_manifest(args.manifest_csv)
    features = e185.feature_sets(manifest)[args.feature_set][0]

    results: list[dict[str, Any]] = []
    primary = primary_decisions(rows, args)
    feature, direction, threshold, decisions, _ = best_secondary(rows, features, args)
    rule = "primary_only" if feature == "none" else f"{primary_rule(args)} AND {feature} {direction} {threshold:.9g}"
    loo_decisions, loo_rule = loocv(rows, features, args)
    results.append(summarize(f"force{args.force}", "baseline", rows, [False] * len(rows), args))
    results.append(summarize(f"force{args.force}", "always_active", rows, [True] * len(rows), args))
    results.append(summarize(f"force{args.force}", "primary", rows, primary, args, args.feature_set, primary_rule(args)))
    results.append(summarize(f"force{args.force}", "best_two_stage", rows, decisions, args, args.feature_set, rule))
    results.append(summarize(f"force{args.force}", "loocv_two_stage", rows, loo_decisions, args, args.feature_set, loo_rule))
    for split in ("first12_eval_last12", "last12_eval_first12", "odd_eval_even", "even_eval_odd"):
        results.extend(split_rows(split, rows, features, args))
    write_outputs(args.output_prefix, results, args)


if __name__ == "__main__":
    main()
