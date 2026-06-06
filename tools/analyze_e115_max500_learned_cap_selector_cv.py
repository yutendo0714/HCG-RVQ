#!/usr/bin/env python3
"""Leave-one-seed CV for learned max500 per-image selector-cap policies."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import analyze_e114_max500_per_image_cap_headroom as e114  # noqa: E402


ANALYSIS = Path("experiments/analysis")
OUT_PREFIX = ANALYSIS / "e115_max500_learned_cap_selector_cv"
CAPS = e114.CAPS
LOWER_CAPS = e114.LOWER_CAPS
SEEDS = e114.SEEDS
ALPHAS = (1e-4, 1e-2, 1.0, 10.0, 100.0)
FEATURE_SETS = {
    "selector": (
        "rvq_householder_residual_selector_multiplier",
        "rvq_householder_residual_selector_multiplier_min",
        "rvq_householder_residual_selector_multiplier_std",
        "rvq_householder_residual_selector_prob",
        "rvq_householder_residual_selector_prob_max",
        "rvq_householder_residual_selector_prob_std",
    ),
    "selector_delta": (
        "rvq_householder_residual_selector_multiplier",
        "rvq_householder_residual_selector_multiplier_min",
        "rvq_householder_residual_selector_multiplier_std",
        "rvq_householder_residual_selector_prob",
        "rvq_householder_residual_selector_prob_max",
        "rvq_householder_residual_selector_prob_std",
        "rvq_householder_delta_rms",
        "rvq_householder_delta_rms_local_max",
        "rvq_householder_delta_rms_local_mean",
        "rvq_householder_delta_rms_local_std",
    ),
    "diagnostic": e114.POLICY_FEATURES,
    "all": e114.FEATURES,
}


def item_delta(item: dict[str, object], cap: float) -> float:
    return float(item["rd_by_cap"][cap]) - float(item["reference_rd"])  # type: ignore[index]


def summarize_caps(items: list[dict[str, object]], caps: list[float]) -> dict[str, object]:
    deltas = [item_delta(item, cap) for item, cap in zip(items, caps)]
    by_seed: dict[int, list[float]] = defaultdict(list)
    cap_counts: Counter[float] = Counter()
    for item, cap, delta in zip(items, caps, deltas):
        by_seed[int(item["seed"])].append(delta)
        cap_counts[cap] += 1
    return {
        "mean_delta": mean(deltas),
        "win_rate_vs_reference": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "num_images": len(deltas),
        "per_seed_delta": {str(seed): mean(vals) for seed, vals in sorted(by_seed.items())},
        "cap_counts": {f"{cap:.2f}": count for cap, count in sorted(cap_counts.items())},
    }


def matrix(items: list[dict[str, object]], features: tuple[str, ...]) -> np.ndarray:
    rows = []
    for item in items:
        values = []
        for feature in features:
            value = float(item["features"][feature])  # type: ignore[index]
            values.append(value if math.isfinite(value) else 0.0)
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ train_y)


def predict(design_x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((design_x.shape[0], 1)), design_x], axis=1)
    return design @ weights


def train_models(
    train_items: list[dict[str, object]],
    test_items: list[dict[str, object]],
    features: tuple[str, ...],
    alpha: float,
) -> list[float]:
    train_x = matrix(train_items, features)
    test_x = matrix(test_items, features)
    train_x, test_x = standardize(train_x, test_x)

    predictions = {0.50: np.zeros(test_x.shape[0], dtype=np.float64)}
    for cap in LOWER_CAPS:
        target = np.asarray(
            [float(item["rd_by_cap"][cap]) - float(item["rd_by_cap"][0.50]) for item in train_items],  # type: ignore[index]
            dtype=np.float64,
        )
        weights = fit_ridge(train_x, target, alpha)
        predictions[cap] = predict(test_x, weights)

    pred_stack = np.stack([predictions[cap] for cap in CAPS], axis=1)
    selected = pred_stack.argmin(axis=1)
    return [float(CAPS[index]) for index in selected]


def evaluate_candidate(
    train_items: list[dict[str, object]],
    test_items: list[dict[str, object]],
    features: tuple[str, ...],
    alpha: float,
) -> dict[str, object]:
    caps = train_models(train_items, test_items, features, alpha)
    summary = summarize_caps(test_items, caps)
    summary.update({"alpha": alpha, "num_features": len(features), "features": ",".join(features)})
    return summary


def inner_cv_select(train_items: list[dict[str, object]]) -> dict[str, object]:
    train_seeds = sorted({int(item["seed"]) for item in train_items})
    candidates = []
    for feature_set, features in FEATURE_SETS.items():
        for alpha in ALPHAS:
            held_summaries = []
            for held_seed in train_seeds:
                inner_train = [item for item in train_items if int(item["seed"]) != held_seed]
                inner_test = [item for item in train_items if int(item["seed"]) == held_seed]
                held_summaries.append(evaluate_candidate(inner_train, inner_test, tuple(features), alpha))
            mean_delta = mean(float(row["mean_delta"]) for row in held_summaries)
            candidates.append(
                {
                    "feature_set": feature_set,
                    "features": tuple(features),
                    "alpha": alpha,
                    "inner_cv_delta": mean_delta,
                    "inner_rows": held_summaries,
                }
            )
    return min(candidates, key=lambda row: float(row["inner_cv_delta"]))


def leave_one_seed_cv(items: list[dict[str, object]]) -> dict[str, object]:
    rows = []
    all_test_items = []
    all_test_caps = []
    baseline = summarize_caps(items, [0.50] * len(items))
    oracle = summarize_caps(
        items,
        [min(CAPS, key=lambda cap: float(item["rd_by_cap"][cap])) for item in items],  # type: ignore[index]
    )
    for held_seed in SEEDS:
        train_items = [item for item in items if int(item["seed"]) != held_seed]
        test_items = [item for item in items if int(item["seed"]) == held_seed]
        selected = inner_cv_select(train_items)
        features = tuple(selected["features"])  # type: ignore[arg-type]
        caps = train_models(train_items, test_items, features, float(selected["alpha"]))
        summary = summarize_caps(test_items, caps)
        all_test_items.extend(test_items)
        all_test_caps.extend(caps)
        rows.append(
            {
                "held_seed": held_seed,
                "feature_set": selected["feature_set"],
                "alpha": selected["alpha"],
                "inner_cv_delta": selected["inner_cv_delta"],
                "test_delta": summary["mean_delta"],
                "test_gain_vs_cap050": float(summary["mean_delta"]) - float(baseline["per_seed_delta"][str(held_seed)]),  # type: ignore[index]
                "test_win_rate": summary["win_rate_vs_reference"],
                "test_cap_counts": summary["cap_counts"],
            }
        )
    cv_summary = summarize_caps(all_test_items, all_test_caps)
    return {
        "baseline_cap050": baseline,
        "oracle": oracle,
        "learned_cv": cv_summary,
        "learned_cv_gain_vs_cap050": float(cv_summary["mean_delta"]) - float(baseline["mean_delta"]),
        "oracle_gain_vs_cap050": float(oracle["mean_delta"]) - float(baseline["mean_delta"]),
        "outer_rows": rows,
    }


def in_sample_candidates(items: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for feature_set, features in FEATURE_SETS.items():
        for alpha in ALPHAS:
            caps = train_models(items, items, tuple(features), alpha)
            summary = summarize_caps(items, caps)
            rows.append(
                {
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "mean_delta": summary["mean_delta"],
                    "gain_vs_cap050": float(summary["mean_delta"])
                    - float(summarize_caps(items, [0.50] * len(items))["mean_delta"]),
                    "win_rate_vs_reference": summary["win_rate_vs_reference"],
                    "cap_counts": summary["cap_counts"],
                }
            )
    return sorted(rows, key=lambda row: float(row["mean_delta"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    items = e114.read_rows()
    cv = leave_one_seed_cv(items)
    candidates = in_sample_candidates(items)
    payload = {
        **cv,
        "best_in_sample_candidate": candidates[0],
        "interpretation": (
            "This is a transfer-only learned-policy audit. The outer leave-one-seed result "
            "estimates whether a multi-feature cap selector generalizes well enough to justify "
            "a deployable reliability controller."
        ),
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".outer_cv.csv"), cv["outer_rows"])  # type: ignore[arg-type]
    write_csv(OUT_PREFIX.with_suffix(".in_sample.csv"), candidates)

    baseline = cv["baseline_cap050"]
    oracle = cv["oracle"]
    learned = cv["learned_cv"]
    md = [
        "# E115 max500 learned cap-selector CV",
        "",
        "## Summary",
        "",
        f"- Baseline cap0.50 transfer delta: {float(baseline['mean_delta']):.6f}",
        f"- Per-image oracle transfer delta: {float(oracle['mean_delta']):.6f}",
        f"- Learned leave-one-seed CV transfer delta: {float(learned['mean_delta']):.6f}",
        f"- Learned CV gain vs cap0.50: {float(cv['learned_cv_gain_vs_cap050']):.6f}",
        f"- Oracle gain vs cap0.50: {float(cv['oracle_gain_vs_cap050']):.6f}",
        f"- Learned CV cap counts: {learned['cap_counts']}",
        "",
        "## Outer CV Rows",
        "",
        "| held seed | feature set | alpha | inner CV delta | test delta | gain vs cap0.50 | win rate | cap counts |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in cv["outer_rows"]:  # type: ignore[index]
        md.append(
            f"| {row['held_seed']} | {row['feature_set']} | {float(row['alpha']):.4g} | "
            f"{float(row['inner_cv_delta']):.6f} | {float(row['test_delta']):.6f} | "
            f"{float(row['test_gain_vs_cap050']):.6f} | {float(row['test_win_rate']):.6f} | "
            f"{row['test_cap_counts']} |"
        )
    best = candidates[0]
    md.extend(
        [
            "",
            "## Best In-Sample Candidate",
            "",
            f"- Feature set: {best['feature_set']}",
            f"- Alpha: {float(best['alpha']):.4g}",
            f"- Transfer delta: {float(best['mean_delta']):.6f}",
            f"- Gain vs cap0.50: {float(best['gain_vs_cap050']):.6f}",
            f"- Cap counts: {best['cap_counts']}",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
