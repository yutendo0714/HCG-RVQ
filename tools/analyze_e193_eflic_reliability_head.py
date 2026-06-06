#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
        default=Path("experiments/analysis/e193_eflic_force0_global_reliability_head"),
    )
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--feature-set", default="global_predecision_context")
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--positive-penalty", type=float, default=20.0)
    p.add_argument("--l2", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--primary-feature", default="slice0_mean_min")
    p.add_argument("--primary-op", choices=[">=", "<="], default=">=")
    p.add_argument("--primary-threshold", type=float, default=-10.7447786331)
    p.add_argument("--top-coefficients", type=int, default=16)
    return p.parse_args()


def active_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        args.dists_weight * float(row["delta_dists"])
        + args.lpips_weight * float(row["delta_lpips"])
        - args.psnr_weight * float(row["delta_psnr"])
    )


def primary_rule(args: argparse.Namespace) -> str:
    return f"{args.primary_feature} {args.primary_op} {args.primary_threshold:.9g}"


def primary_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[bool]:
    return e184.threshold_decisions(rows, args.primary_feature, args.primary_threshold, args.primary_op)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def feature_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)


def labels(rows: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    return np.asarray([1.0 if active_score(row, args) < 0.0 else 0.0 for row in rows], dtype=np.float64)


def tune_threshold(rows: list[dict[str, Any]], probs: np.ndarray, args: argparse.Namespace) -> tuple[float, list[bool], float]:
    vals = sorted(set(float(v) for v in probs if math.isfinite(float(v))))
    if not vals:
        decisions = [False] * len(rows)
        return 1.0, decisions, e190.multiobjective_score(
            rows, decisions, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty
        )
    mids = [(a + b) * 0.5 for a, b in zip(vals, vals[1:])]
    candidates = [max(0.0, vals[0] - 1e-9), *mids, min(1.0, vals[-1] + 1e-9), 0.5]
    best: tuple[float, list[bool], float] | None = None
    for threshold in candidates:
        decisions = [float(p) >= threshold for p in probs]
        score = e190.multiobjective_score(
            rows, decisions, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty
        )
        if best is None or score < best[2]:
            best = (threshold, decisions, score)
    assert best is not None
    return best


def fit_head(train: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> dict[str, Any]:
    x_raw = feature_matrix(train, features)
    y = labels(train, args)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (x_raw - mean) / std
    pos = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    bias = math.log(pos / (1.0 - pos))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    if y.min() != y.max():
        for _ in range(args.epochs):
            prob = sigmoid(x @ weights + bias)
            grad_w = (x.T @ (prob - y)) / max(1, len(y)) + args.l2 * weights
            grad_b = float(np.mean(prob - y))
            weights -= args.lr * grad_w
            bias -= args.lr * grad_b
    train_probs = sigmoid(x @ weights + bias)
    threshold, train_decisions, train_score = tune_threshold(train, train_probs, args)
    return {
        "features": features,
        "mean": mean,
        "std": std,
        "weights": weights,
        "bias": bias,
        "threshold": threshold,
        "train_decisions": train_decisions,
        "train_score": train_score,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[bool]]:
    x_raw = feature_matrix(rows, model["features"])
    x = (x_raw - model["mean"]) / model["std"]
    probs = sigmoid(x @ model["weights"] + float(model["bias"]))
    decisions = [float(p) >= float(model["threshold"]) for p in probs]
    return probs, decisions


def combined_oracle(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[bool]:
    return e190.combined_oracle(rows, args.dists_weight, args.lpips_weight, args.psnr_weight)


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
    out["multiobjective_score"] = e190.multiobjective_score(
        rows, decisions, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty
    )
    out["selected_win_lpips"] = sum(
        (float(r["active_lpips"]) - float(r["base_lpips"])) < 0 for r, d in zip(rows, decisions) if d
    )
    out["selected_good"] = sum(active_score(r, args) < 0 and d for r, d in zip(rows, decisions))
    out["selected_bad"] = sum(active_score(r, args) >= 0 and d for r, d in zip(rows, decisions))
    out["missed_good"] = sum(active_score(r, args) < 0 and not d for r, d in zip(rows, decisions))
    out["dists_weight"] = args.dists_weight
    out["lpips_weight"] = args.lpips_weight
    out["psnr_weight"] = args.psnr_weight
    out["positive_penalty"] = args.positive_penalty
    return out


def loocv_head(rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> tuple[list[bool], str]:
    decisions: list[bool] = []
    thresholds: list[float] = []
    active_counts: list[int] = []
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1 :]
        model = fit_head(train, features, args)
        _, one_decision = predict(model, [row])
        decisions.extend(one_decision)
        thresholds.append(float(model["threshold"]))
        active_counts.append(sum(model["train_decisions"]))
    rule = (
        f"logistic_head LOOCV, threshold mean={np.mean(thresholds):.4f}, "
        f"train active mean={np.mean(active_counts):.2f}"
    )
    return decisions, rule


def split_head(split: str, rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    train, eval_rows = e187.split_rows(rows, split)
    model = fit_head(train, features, args)
    _, train_decisions = predict(model, train)
    _, eval_decisions = predict(model, eval_rows)
    rule = f"logistic_head threshold={float(model['threshold']):.6f}"
    return [
        summarize(split, "train_primary", train, primary_decisions(train, args), args, args.feature_set, primary_rule(args)),
        summarize(split, "train_reliability_head", train, train_decisions, args, args.feature_set, rule),
        summarize(split, "eval_primary", eval_rows, primary_decisions(eval_rows, args), args, args.feature_set, primary_rule(args)),
        summarize(split, "eval_apply_reliability_head", eval_rows, eval_decisions, args, args.feature_set, rule),
    ]


def top_coefficients(model: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    rows = []
    for feature, weight in zip(model["features"], model["weights"]):
        rows.append({"feature": feature, "coefficient": float(weight), "abs_coefficient": abs(float(weight))})
    rows.sort(key=lambda r: float(r["abs_coefficient"]), reverse=True)
    return rows[:topk]


def write_outputs(
    prefix: Path,
    rows: list[dict[str, Any]],
    coefficients: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    coeff_path = prefix.with_name(prefix.name + "_coefficients.csv")
    with coeff_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coefficient", "abs_coefficient"])
        writer.writeheader()
        writer.writerows(coefficients)
    prefix.with_suffix(".json").write_text(
        json.dumps({"args": vars(args), "rows": rows, "coefficients": coefficients}, indent=2, sort_keys=True, default=str)
        + "\n"
    )

    lines = [
        "# E193 EF-LIC Learned Reliability Head Audit",
        "",
        "This diagnostic trains a tiny logistic decoder-side reliability head using only global-predecision features. It is a controller-headroom audit, not a final codec row unless the input label table comes from an independent fit split.",
        "",
        f"Force index: `{args.force}`",
        f"Feature set: `{args.feature_set}`",
        f"Primary reference rule: `{primary_rule(args)}`",
        f"Weights: DISTS `{args.dists_weight}`, LPIPS `{args.lpips_weight}`, PSNR `{args.psnr_weight}`, positive penalty `{args.positive_penalty}`",
        "",
        "| group | selector | branch share | dDISTS | dLPIPS | dPSNR | score | good/bad/missed | rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row['branch_share']:.3f} | "
            f"{row['selected_delta_dists']:+.6f} | {row['selected_delta_lpips']:+.6f} | "
            f"{row['selected_delta_psnr']:+.6f} | {row['multiobjective_score']:+.6f} | "
            f"{row['selected_good']}/{row['selected_bad']}/{row['missed_good']} | {row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Top standardized coefficients:",
            "",
            "| feature | coefficient |",
            "|---|---:|",
        ]
    )
    for row in coefficients[: args.top_coefficients]:
        lines.append(f"| {row['feature']} | {row['coefficient']:+.6f} |")
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- Uses only decoder-known global-predecision features, so no side bit is assumed.",
            "- Same-table `reliability_head` is design headroom; LOOCV and split rows are the anti-overfit checks.",
            "- If the head improves same-table but not split/LOOCV, keep E190 primary as the paper-facing default and use the head only as a target for independent-fit training.",
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
    feature_sets = e185.feature_sets(manifest)
    if args.feature_set not in feature_sets:
        raise SystemExit(f"unknown feature set: {args.feature_set}")
    features = e184.valid_features(rows, feature_sets[args.feature_set][0])
    if not features:
        raise SystemExit(f"no finite candidate features in {args.feature_set}")

    results: list[dict[str, Any]] = []
    model = fit_head(rows, features, args)
    _, head_decisions = predict(model, rows)
    loo_decisions, loo_rule = loocv_head(rows, features, args)

    results.append(summarize(f"force{args.force}", "baseline", rows, [False] * len(rows), args))
    results.append(summarize(f"force{args.force}", "always_active", rows, [True] * len(rows), args))
    results.append(summarize(f"force{args.force}", "primary", rows, primary_decisions(rows, args), args, args.feature_set, primary_rule(args)))
    results.append(summarize(f"force{args.force}", "combined_oracle", rows, combined_oracle(rows, args), args, "metric_oracle"))
    results.append(
        summarize(
            f"force{args.force}",
            "reliability_head",
            rows,
            head_decisions,
            args,
            args.feature_set,
            f"logistic_head threshold={float(model['threshold']):.6f}",
        )
    )
    results.append(summarize(f"force{args.force}", "loocv_reliability_head", rows, loo_decisions, args, args.feature_set, loo_rule))
    for split in ("first12_eval_last12", "last12_eval_first12", "odd_eval_even", "even_eval_odd"):
        results.extend(split_head(split, rows, features, args))

    coefficients = top_coefficients(model, args.top_coefficients)
    write_outputs(args.output_prefix, results, coefficients, args)


if __name__ == "__main__":
    main()
