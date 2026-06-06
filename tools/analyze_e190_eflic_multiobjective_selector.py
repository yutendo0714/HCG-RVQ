#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402
import analyze_e187_eflic_selector_splitfit as e187  # noqa: E402


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
        default=Path("experiments/analysis/e190_eflic_force0_global_selector_multiobj"),
    )
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=1.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument(
        "--positive-penalty",
        type=float,
        default=20.0,
        help="Penalty applied to mean positive DISTS/LPIPS deltas to prefer rules improving both metrics.",
    )
    p.add_argument(
        "--feature-set",
        choices=["global_predecision_context", "sequential_context", "legacy_decoder_safe_context"],
        default="global_predecision_context",
    )
    return p.parse_args()


def metric_delta(row: dict[str, Any], use_active: bool, name: str) -> float:
    if not use_active:
        return 0.0
    if name == "dists":
        return float(row["active_dists"]) - float(row["base_dists"])
    if name == "lpips":
        return float(row["active_lpips"]) - float(row["base_lpips"])
    if name == "psnr":
        return float(row["active_psnr"]) - float(row["base_psnr"])
    raise ValueError(name)


def multiobjective_score(
    rows: list[dict[str, Any]],
    decisions: list[bool],
    dists_weight: float,
    lpips_weight: float,
    psnr_weight: float,
    positive_penalty: float,
) -> float:
    n = max(1, len(rows))
    dd = sum(metric_delta(r, d, "dists") for r, d in zip(rows, decisions)) / n
    dl = sum(metric_delta(r, d, "lpips") for r, d in zip(rows, decisions)) / n
    dp = sum(metric_delta(r, d, "psnr") for r, d in zip(rows, decisions)) / n
    score = dists_weight * dd + lpips_weight * dl - psnr_weight * dp
    score += positive_penalty * max(dd, 0.0)
    score += positive_penalty * max(dl, 0.0)
    return score


def combined_oracle(
    rows: list[dict[str, Any]],
    dists_weight: float,
    lpips_weight: float,
    psnr_weight: float,
) -> list[bool]:
    decisions = []
    for row in rows:
        score = (
            dists_weight * metric_delta(row, True, "dists")
            + lpips_weight * metric_delta(row, True, "lpips")
            - psnr_weight * metric_delta(row, True, "psnr")
        )
        decisions.append(score < 0.0)
    return decisions


def best_multiobjective_threshold(
    rows: list[dict[str, Any]],
    features: list[str],
    args: argparse.Namespace,
) -> tuple[str, str, float, list[bool], float]:
    best: tuple[str, str, float, list[bool], float] | None = None
    for feature in e184.valid_features(rows, features):
        values = [float(r[feature]) for r in rows]
        for threshold in e184.candidate_thresholds(values):
            for direction in (">=", "<="):
                decisions = e184.threshold_decisions(rows, feature, threshold, direction)
                score = multiobjective_score(
                    rows,
                    decisions,
                    args.dists_weight,
                    args.lpips_weight,
                    args.psnr_weight,
                    args.positive_penalty,
                )
                if best is None or score < best[4]:
                    best = (feature, direction, threshold, decisions, score)
    if best is None:
        decisions = [False] * len(rows)
        return "", ">=", float("nan"), decisions, multiobjective_score(
            rows, decisions, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty
        )
    return best


def loocv_multiobjective(rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> tuple[list[bool], str]:
    decisions = []
    rules = []
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1 :]
        feature, direction, threshold, _, _ = best_multiobjective_threshold(train, features, args)
        if not feature:
            decisions.append(False)
            rules.append("baseline")
            continue
        decision = e184.threshold_decisions([row], feature, threshold, direction)[0]
        decisions.append(decision)
        rules.append(f"{feature} {direction} {threshold:.6g}")
    unique = sorted(set(rules))
    if len(unique) <= 6:
        return decisions, "; ".join(unique)
    return decisions, f"{len(unique)} fold-specific rules"


def add_score(row: dict[str, Any], rows: list[dict[str, Any]], decisions: list[bool], args: argparse.Namespace) -> None:
    row["multiobjective_score"] = multiobjective_score(
        rows, decisions, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty
    )
    row["dists_weight"] = args.dists_weight
    row["lpips_weight"] = args.lpips_weight
    row["psnr_weight"] = args.psnr_weight
    row["positive_penalty"] = args.positive_penalty


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
    add_score(out, rows, decisions, args)
    return out


def analyze_split(split: str, rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    train, eval_rows = e187.split_rows(rows, split)
    if not train or not eval_rows:
        return []
    feature, direction, threshold, train_decisions, _ = best_multiobjective_threshold(train, features, args)
    eval_decisions = e184.threshold_decisions(eval_rows, feature, threshold, direction) if feature else [False] * len(eval_rows)
    rule = f"{feature} {direction} {threshold:.9g}" if feature else "baseline"
    return [
        summarize(split, "train_always_active", train, [True] * len(train), args),
        summarize(split, "train_combined_oracle", train, combined_oracle(train, args.dists_weight, args.lpips_weight, args.psnr_weight), args, "metric_oracle"),
        summarize(split, "train_fit_multiobj_threshold", train, train_decisions, args, args.feature_set, rule),
        summarize(split, "eval_always_active", eval_rows, [True] * len(eval_rows), args),
        summarize(split, "eval_combined_oracle", eval_rows, combined_oracle(eval_rows, args.dists_weight, args.lpips_weight, args.psnr_weight), args, "metric_oracle"),
        summarize(split, "eval_apply_multiobj_threshold", eval_rows, eval_decisions, args, args.feature_set, rule),
    ]


def write_outputs(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E190 EF-LIC Multi-Objective Global Selector Audit",
        "",
        "This audit searches scalar decoder-side global-predecision rules that jointly optimize DISTS and LPIPS. It is still a Kodak diagnostic unless the input CSV comes from an independent fit split.",
        "",
        f"Force index: `{args.force}`",
        f"Feature set: `{args.feature_set}`",
        f"Weights: DISTS `{args.dists_weight}`, LPIPS `{args.lpips_weight}`, PSNR `{args.psnr_weight}`, positive penalty `{args.positive_penalty}`",
        "",
        "| group | selector | branch share | dDISTS | dLPIPS | dPSNR | score | DISTS wins | rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row['branch_share']:.3f} | "
            f"{row['selected_delta_dists']:+.6f} | {row['selected_delta_lpips']:+.6f} | "
            f"{row['selected_delta_psnr']:+.6f} | {row['multiobjective_score']:+.6f} | "
            f"{row['selected_win_dists']}/{row['images']} | {row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- Uses no-side-bit decoder-side features when `global_predecision_context` is selected.",
            "- `combined_oracle` is a diagnostic upper bound using measured metric deltas.",
            "- Split rows are anti-overfit checks, not substitutes for external validation.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in rows if int(r["force_ind"]) == args.force]
    if not rows:
        raise SystemExit(f"no finite rows for force{args.force}")
    manifest = e184.read_manifest(args.manifest_csv)
    features = e185.feature_sets(manifest)[args.feature_set][0]

    results: list[dict[str, Any]] = []
    results.append(summarize(f"force{args.force}", "baseline", rows, [False] * len(rows), args))
    results.append(summarize(f"force{args.force}", "always_active", rows, [True] * len(rows), args))
    results.append(
        summarize(
            f"force{args.force}",
            "combined_oracle",
            rows,
            combined_oracle(rows, args.dists_weight, args.lpips_weight, args.psnr_weight),
            args,
            "metric_oracle",
        )
    )
    feature, direction, threshold, decisions, _ = best_multiobjective_threshold(rows, features, args)
    rule = f"{feature} {direction} {threshold:.9g}" if feature else "baseline"
    results.append(summarize(f"force{args.force}", "best_multiobj_threshold", rows, decisions, args, args.feature_set, rule))

    loocv_decisions, loocv_rule = loocv_multiobjective(rows, features, args)
    results.append(
        summarize(f"force{args.force}", "loocv_multiobj_threshold", rows, loocv_decisions, args, args.feature_set, loocv_rule)
    )

    for split in ("first12_eval_last12", "last12_eval_first12", "odd_eval_even", "even_eval_odd"):
        results.extend(analyze_split(split, rows, features, args))

    write_outputs(args.output_prefix, results, args)


if __name__ == "__main__":
    main()
