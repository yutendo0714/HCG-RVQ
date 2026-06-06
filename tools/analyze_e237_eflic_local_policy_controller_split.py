#!/usr/bin/env python3
"""E237 split-safe controller audit for E236 local EF-LIC policies.

E236 produced a decoder-reproducible local-policy vocabulary with a strong
per-image oracle but weak fixed-policy behavior. This script asks whether that
oracle can be harvested by a small controller using only information available
before the y-slice index decision: E233 decoder-safe global features plus
E236 alpha-map design statistics. Outcome fields such as DISTS/LPIPS deltas,
index mismatch, and residual errors are never used as selector inputs.

This is a design gate for the eventual in-codec learned local controller, not
the final paper method.
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

FEATURE_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e233_eflic_kodak24_decoder_safe_branch_features.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e233_eflic_clicpro41_decoder_safe_branch_features.csv",
}

E236_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e236_eflic_kodak24_local_controller_map.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e236_eflic_clicpro41_local_controller_map.csv",
}

DROP_FEATURE_COLUMNS = {
    "dataset",
    "image",
    "force_ind",
    "height",
    "width",
    "pixels",
    "nonfinite_features",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e237_eflic_local_policy_controller_split",
    )
    p.add_argument("--top-k", default="16,64,0")
    p.add_argument("--ridge-alpha", type=float, default=25.0)
    p.add_argument("--max-md-rows", type=int, default=80)
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


def score_from_row(row: dict[str, str]) -> float:
    score = to_float(row.get("score_dists_3lpips"))
    if math.isfinite(score):
        return score
    return to_float(row.get("delta_dists"), 0.0) + 3.0 * to_float(row.get("delta_lpips"), 0.0)


def load_feature_rows(paths: dict[str, Path]) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    cols: set[str] = set()
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            for row in csv.DictReader(fobj):
                item = (dataset, row["image"])
                rows[item] = row
                for col, value in row.items():
                    if col in DROP_FEATURE_COLUMNS:
                        continue
                    if math.isfinite(to_float(value)):
                        cols.add(col)
    return rows, sorted(cols)


def is_valid_policy_row(row: dict[str, str]) -> bool:
    return (
        abs(to_float(row.get("delta_bpp"), 1.0)) <= 1e-12
        and to_float(row.get("max_decode_diff"), 1.0) <= 1e-10
        and int(to_float(row.get("nonfinite"), 1.0)) == 0
        and int(to_float(row.get("payload_len_equal"), 0.0)) == 1
    )


def load_e236_rows(paths: dict[str, Path]) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], list[str]]:
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    alpha_cols: set[str] = set()
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            reader = csv.DictReader(fobj)
            for col in reader.fieldnames or []:
                if "_alpha_" in col or col.endswith("_alpha_mean") or col.startswith("y_alpha"):
                    alpha_cols.add(col)
            fobj.seek(0)
            for raw in csv.DictReader(fobj):
                if not is_valid_policy_row(raw):
                    continue
                item = (dataset, raw["image"])
                policy = raw["policy"]
                alpha = {}
                for col in alpha_cols:
                    value = to_float(raw.get(col), 0.0)
                    alpha[col] = value if math.isfinite(value) else 0.0
                rows_by_item[item][policy] = {
                    "dataset": dataset,
                    "image": raw["image"],
                    "policy": policy,
                    "family": raw["family"],
                    "score": score_from_row(raw),
                    "delta_dists": to_float(raw.get("delta_dists"), 0.0),
                    "delta_lpips": to_float(raw.get("delta_lpips"), 0.0),
                    "delta_psnr": to_float(raw.get("delta_psnr"), 0.0),
                    "bpp": to_float(raw.get("bpp"), 0.0),
                    "alpha": alpha,
                    "alpha_mean": to_float(raw.get("y_alpha_mean"), 0.0),
                    "alpha_active_frac": to_float(raw.get("y_alpha_active_frac"), 0.0),
                }
    return dict(rows_by_item), sorted(alpha_cols)


def common_items_and_policies(
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    items = sorted(item for item in rows_by_item if item in feature_rows)
    if not items:
        raise SystemExit("no overlapping E233 feature rows and E236 policy rows")
    common = set(rows_by_item[items[0]])
    for item in items[1:]:
        common &= set(rows_by_item[item])
    if "zero" not in common:
        raise SystemExit("E236 rows must include zero fallback")
    policies = sorted(common)
    families = sorted({str(rows_by_item[items[0]][policy]["family"]) for policy in policies})
    return items, policies, families


def base_feature_matrix(
    items: list[tuple[str, str]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
) -> np.ndarray:
    x = np.zeros((len(items), len(cols)), dtype=np.float64)
    for i, item in enumerate(items):
        row = feature_rows[item]
        for j, col in enumerate(cols):
            value = to_float(row.get(col), 0.0)
            x[i, j] = value if math.isfinite(value) else 0.0
    return x


def oracle_labels(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    policies: list[str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], list[dict[str, object]]]:
    oracle_policy: dict[tuple[str, str], str] = {}
    oracle_family: dict[tuple[str, str], str] = {}
    rows: list[dict[str, object]] = []
    for item in items:
        choice = min(policies, key=lambda policy: float(rows_by_item[item][policy]["score"]))
        row = rows_by_item[item][choice]
        oracle_policy[item] = choice
        oracle_family[item] = str(row["family"])
        rows.append(
            {
                "dataset": item[0],
                "image": item[1],
                "oracle_policy": choice,
                "oracle_family": row["family"],
                "oracle_score": row["score"],
                "oracle_delta_dists": row["delta_dists"],
                "oracle_delta_lpips": row["delta_lpips"],
                "oracle_delta_psnr": row["delta_psnr"],
            }
        )
    return oracle_policy, oracle_family, rows


def anova_feature_scores(
    items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
) -> list[dict[str, object]]:
    x = base_feature_matrix(items, feature_rows, cols)
    y = np.asarray([labels[item] for item in items], dtype=object)
    out: list[dict[str, object]] = []
    for j, col in enumerate(cols):
        values = x[:, j]
        total = float(np.sum((values - values.mean()) ** 2))
        between = 0.0
        if total > 1e-18:
            for label in sorted(set(y)):
                subset = values[y == label]
                between += float(len(subset) * (subset.mean() - values.mean()) ** 2)
        out.append(
            {
                "feature": col,
                "oracle_family_anova_r2": between / total if total > 1e-18 else 0.0,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return sorted(out, key=lambda row: to_float(row["oracle_family_anova_r2"]), reverse=True)


def select_base_cols(
    train_items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
    top_k: int,
) -> list[str]:
    if top_k <= 0 or top_k >= len(cols):
        return cols
    ranked = anova_feature_scores(train_items, labels, feature_rows, cols)
    return [str(row["feature"]) for row in ranked[:top_k]]


def nearest_centroid_family(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
) -> dict[tuple[str, str], str]:
    if not cols:
        majority = Counter(labels[item] for item in train_items).most_common(1)[0][0]
        return {item: majority for item in test_items}
    x_train = base_feature_matrix(train_items, feature_rows, cols)
    x_test = base_feature_matrix(test_items, feature_rows, cols)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_train = np.asarray([labels[item] for item in train_items], dtype=object)
    centroids = {label: x_train[y_train == label].mean(axis=0) for label in sorted(set(y_train))}
    return {
        item: min(centroids, key=lambda label: float(np.sum((vec - centroids[label]) ** 2)))
        for item, vec in zip(test_items, x_test)
    }


def vector_for_item_policy(
    item: tuple[str, str],
    policy: str,
    feature_rows: dict[tuple[str, str], dict[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    base_cols: list[str],
    alpha_cols: list[str],
    policies: list[str],
    families: list[str],
) -> list[float]:
    base_row = feature_rows[item]
    policy_row = rows_by_item[item][policy]
    out: list[float] = []
    for col in base_cols:
        value = to_float(base_row.get(col), 0.0)
        out.append(value if math.isfinite(value) else 0.0)
    alpha = policy_row["alpha"]
    for col in alpha_cols:
        out.append(float(alpha.get(col, 0.0)))
    out.extend(1.0 if policy == candidate else 0.0 for candidate in policies)
    family = str(policy_row["family"])
    out.extend(1.0 if family == candidate else 0.0 for candidate in families)
    return out


def design_matrix(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    base_cols: list[str],
    alpha_cols: list[str],
    policies: list[str],
    families: list[str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[str, str], str]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    keys: list[tuple[tuple[str, str], str]] = []
    for item in items:
        for policy in policies:
            xs.append(vector_for_item_policy(item, policy, feature_rows, rows_by_item, base_cols, alpha_cols, policies, families))
            ys.append(float(rows_by_item[item][policy]["score"]))
            keys.append((item, policy))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), keys


def fit_long_ridge(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    base_cols: list[str],
    alpha_cols: list[str],
    policies: list[str],
    families: list[str],
    ridge_alpha: float,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], dict[str, float]], dict[str, object]]:
    x_train, y_train, train_keys = design_matrix(train_items, rows_by_item, feature_rows, base_cols, alpha_cols, policies, families)
    x_test, _, test_keys = design_matrix(test_items, rows_by_item, feature_rows, base_cols, alpha_cols, policies, families)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    reg = np.eye(x_train.shape[1], dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0
    lhs = x_train.T @ x_train + reg
    beta = np.linalg.solve(lhs, x_train.T @ y_train)
    pred_train_vec = x_train @ beta
    pred_test_vec = x_test @ beta
    pred_train: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    pred_test: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (item, policy), pred in zip(train_keys, pred_train_vec):
        pred_train[item][policy] = float(pred)
    for (item, policy), pred in zip(test_keys, pred_test_vec):
        pred_test[item][policy] = float(pred)
    meta = {
        "base_feature_count": len(base_cols),
        "alpha_feature_count": len(alpha_cols),
        "total_feature_count": int(x_train.shape[1] - 1),
        "ridge_rank": int(np.linalg.matrix_rank(x_train)),
        "condition": float(np.linalg.cond(lhs)),
    }
    return dict(pred_train), dict(pred_test), meta


def tune_fallback_threshold(
    train_items: list[tuple[str, str]],
    pred_scores: dict[tuple[str, str], dict[str, float]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    policies: list[str],
) -> float:
    margins = []
    for item in train_items:
        best = min(policies, key=lambda policy: pred_scores[item][policy])
        margins.append(float(pred_scores[item]["zero"] - pred_scores[item][best]))
    values = sorted(set(margins + [0.0]))
    candidates = [values[0] - 1e-9, values[-1] + 1e-9]
    candidates.extend((a + b) * 0.5 for a, b in zip(values, values[1:]))
    candidates.append(0.0)
    best_threshold = 0.0
    best_score = float("inf")
    for threshold in candidates:
        scores = []
        for item in train_items:
            best = min(policies, key=lambda policy: pred_scores[item][policy])
            margin = float(pred_scores[item]["zero"] - pred_scores[item][best])
            chosen = best if best != "zero" and margin >= threshold else "zero"
            scores.append(float(rows_by_item[item][chosen]["score"]))
        value = mean(scores)
        if value < best_score:
            best_score = value
            best_threshold = float(threshold)
    return best_threshold


def best_fixed_policy(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    policies: list[str],
) -> str:
    return min(policies, key=lambda policy: mean(rows_by_item[item][policy]["score"] for item in train_items))


def best_policy_by_family(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    policies: list[str],
) -> dict[str, str]:
    family_to_policies: dict[str, list[str]] = defaultdict(list)
    ref_item = train_items[0]
    for policy in policies:
        family_to_policies[str(rows_by_item[ref_item][policy]["family"])].append(policy)
    return {
        family: best_fixed_policy(train_items, rows_by_item, family_policies)
        for family, family_policies in sorted(family_to_policies.items())
    }


def summarize_predictions(
    *,
    split: str,
    method: str,
    train_count: int,
    predictions: dict[tuple[str, str], str],
    oracle_policy: dict[tuple[str, str], str],
    oracle_family: dict[tuple[str, str], str],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    scores = []
    oracle_scores = []
    delta_dists = []
    delta_lpips = []
    delta_psnr = []
    for item, pred in sorted(predictions.items()):
        pred_row = rows_by_item[item][pred]
        oracle_row = rows_by_item[item][oracle_policy[item]]
        score = float(pred_row["score"])
        oracle_score = float(oracle_row["score"])
        scores.append(score)
        oracle_scores.append(oracle_score)
        delta_dists.append(float(pred_row["delta_dists"]))
        delta_lpips.append(float(pred_row["delta_lpips"]))
        delta_psnr.append(float(pred_row["delta_psnr"]))
        rows.append(
            {
                "split": split,
                "method": method,
                "dataset": item[0],
                "image": item[1],
                "predicted_policy": pred,
                "predicted_family": pred_row["family"],
                "oracle_policy": oracle_policy[item],
                "oracle_family": oracle_family[item],
                "score": score,
                "oracle_score": oracle_score,
                "regret_vs_oracle": score - oracle_score,
                "delta_dists": pred_row["delta_dists"],
                "delta_lpips": pred_row["delta_lpips"],
                "delta_psnr": pred_row["delta_psnr"],
                "alpha_mean": pred_row["alpha_mean"],
                "alpha_active_frac": pred_row["alpha_active_frac"],
            }
        )
    summary = {
        "split": split,
        "method": method,
        "train_images": train_count,
        "test_images": len(predictions),
        "score": mean(scores),
        "oracle_score": mean(oracle_scores),
        "regret_vs_oracle": mean(scores) - mean(oracle_scores),
        "score_win_frac": mean(1.0 if value < 0.0 else 0.0 for value in scores),
        "positive_score_frac": mean(1.0 if value > 0.0 else 0.0 for value in scores),
        "delta_dists": mean(delta_dists),
        "delta_lpips": mean(delta_lpips),
        "delta_psnr": mean(delta_psnr),
        "policy_accuracy": mean(1.0 if r["predicted_policy"] == r["oracle_policy"] else 0.0 for r in rows),
        "family_accuracy": mean(1.0 if r["predicted_family"] == r["oracle_family"] else 0.0 for r in rows),
        "predicted_policy_counts": json.dumps(dict(Counter(str(r["predicted_policy"]) for r in rows)), sort_keys=True),
        "predicted_family_counts": json.dumps(dict(Counter(str(r["predicted_family"]) for r in rows)), sort_keys=True),
    }
    if extra:
        summary.update(extra)
    return summary, rows


def evaluate_split(
    split: str,
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    alpha_cols: list[str],
    policies: list[str],
    families: list[str],
    oracle_policy: dict[tuple[str, str], str],
    oracle_family: dict[tuple[str, str], str],
    top_ks: list[int],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    predictions_all: list[dict[str, object]] = []

    fixed = best_fixed_policy(train_items, rows_by_item, policies)
    family_best = best_policy_by_family(train_items, rows_by_item, policies)
    baselines = [
        ("oracle_all", {item: oracle_policy[item] for item in test_items}, None),
        ("zero", {item: "zero" for item in test_items}, None),
        ("best_fixed_train", {item: fixed for item in test_items}, {"fixed_policy": fixed}),
        (
            "true_family_train_best_policy",
            {item: family_best.get(oracle_family[item], fixed) for item in test_items},
            {"family_best_policy": json.dumps(family_best, sort_keys=True)},
        ),
    ]
    for name, preds, extra in baselines:
        summary, rows = summarize_predictions(
            split=split,
            method=name,
            train_count=len(train_items),
            predictions=preds,
            oracle_policy=oracle_policy,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra=extra,
        )
        summaries.append(summary)
        predictions_all.extend(rows)

    for top_k in top_ks:
        base_cols = select_base_cols(train_items, oracle_family, feature_rows, feature_cols, top_k)
        label_suffix = top_k if top_k > 0 else "all"
        nearest_family = nearest_centroid_family(train_items, test_items, oracle_family, feature_rows, base_cols)
        nearest_preds = {item: family_best.get(nearest_family[item], fixed) for item in test_items}
        summary, rows = summarize_predictions(
            split=split,
            method=f"nearest_family_top{label_suffix}",
            train_count=len(train_items),
            predictions=nearest_preds,
            oracle_policy=oracle_policy,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={"base_feature_count": len(base_cols), "selected_base_features": json.dumps(base_cols[:12])},
        )
        summaries.append(summary)
        predictions_all.extend(rows)

        pred_train, pred_test, meta = fit_long_ridge(
            train_items,
            test_items,
            rows_by_item,
            feature_rows,
            base_cols,
            alpha_cols,
            policies,
            families,
            ridge_alpha,
        )
        ridge_preds = {item: min(policies, key=lambda policy: pred_test[item][policy]) for item in test_items}
        summary, rows = summarize_predictions(
            split=split,
            method=f"long_ridge_score_top{label_suffix}_l2{ridge_alpha:g}",
            train_count=len(train_items),
            predictions=ridge_preds,
            oracle_policy=oracle_policy,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={**meta, "selected_base_features": json.dumps(base_cols[:12])},
        )
        summaries.append(summary)
        predictions_all.extend(rows)

        threshold = tune_fallback_threshold(train_items, pred_train, rows_by_item, policies)
        gated_preds = {}
        for item in test_items:
            best = min(policies, key=lambda policy: pred_test[item][policy])
            margin = float(pred_test[item]["zero"] - pred_test[item][best])
            gated_preds[item] = best if best != "zero" and margin >= threshold else "zero"
        summary, rows = summarize_predictions(
            split=split,
            method=f"long_ridge_fallback_top{label_suffix}_l2{ridge_alpha:g}",
            train_count=len(train_items),
            predictions=gated_preds,
            oracle_policy=oracle_policy,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={**meta, "fallback_threshold": threshold, "selected_base_features": json.dumps(base_cols[:12])},
        )
        summaries.append(summary)
        predictions_all.extend(rows)
    return summaries, predictions_all


def evaluate_leave_one_image_out(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    alpha_cols: list[str],
    policies: list[str],
    families: list[str],
    oracle_policy: dict[tuple[str, str], str],
    oracle_family: dict[tuple[str, str], str],
    top_ks: list[int],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        train_items = [other for other in items if other != item]
        _, preds = evaluate_split(
            "pooled_leave_one_image_out",
            train_items,
            [item],
            rows_by_item,
            feature_rows,
            feature_cols,
            alpha_cols,
            policies,
            families,
            oracle_policy,
            oracle_family,
            top_ks,
            ridge_alpha,
        )
        for row in preds:
            grouped[str(row["method"])].append(row)
    summaries = []
    predictions = []
    for method, rows in sorted(grouped.items()):
        preds = {(str(r["dataset"]), str(r["image"])): str(r["predicted_policy"]) for r in rows}
        summary, collapsed = summarize_predictions(
            split="pooled_leave_one_image_out",
            method=method,
            train_count=len(items) - 1,
            predictions=preds,
            oracle_policy=oracle_policy,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
        )
        summaries.append(summary)
        predictions.extend(collapsed)
    return summaries, predictions


def split_defs(items: list[tuple[str, str]]) -> list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]]:
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        by_dataset[item[0]].append(item)
    out: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = [("pooled_resub", items, items)]
    for dataset, ds_items in sorted(by_dataset.items()):
        ds_items = sorted(ds_items)
        out.append((f"{dataset}_resub", ds_items, ds_items))
        if len(ds_items) >= 8:
            mid = len(ds_items) // 2
            out.append((f"{dataset}_first_half_test_second_half", ds_items[:mid], ds_items[mid:]))
            out.append((f"{dataset}_second_half_test_first_half", ds_items[mid:], ds_items[:mid]))
    if "clicpro41" in by_dataset and "kodak24" in by_dataset:
        out.append(("train_clicpro41_test_kodak24", sorted(by_dataset["clicpro41"]), sorted(by_dataset["kodak24"])))
        out.append(("train_kodak24_test_clicpro41", sorted(by_dataset["kodak24"]), sorted(by_dataset["clicpro41"])))
    return out


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:+.6f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, object]],
    labels: list[dict[str, object]],
    feature_sep: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    key_splits = {
        "pooled_resub",
        "pooled_leave_one_image_out",
        "train_clicpro41_test_kodak24",
        "train_kodak24_test_clicpro41",
    }
    key_methods = {
        "oracle_all",
        "zero",
        "best_fixed_train",
        "true_family_train_best_policy",
        "nearest_family_topall",
        "nearest_family_top64",
        f"long_ridge_score_topall_l2{args.ridge_alpha:g}",
        f"long_ridge_fallback_topall_l2{args.ridge_alpha:g}",
        f"long_ridge_score_top64_l2{args.ridge_alpha:g}",
        f"long_ridge_fallback_top64_l2{args.ridge_alpha:g}",
    }
    key_rows = [row for row in summary_rows if row["split"] in key_splits and row["method"] in key_methods]
    key_rows = sorted(key_rows, key=lambda r: (str(r["split"]), str(r["method"])))
    lines = [
        "# E237 EF-LIC Local Policy Controller Split Audit",
        "",
        "E237 tests whether the E236 local-policy oracle can be harvested by a split-safe controller using only decoder-safe E233 features and E236 alpha-map design statistics. DISTS/LPIPS deltas, mismatch, index outcomes, and residual-error outcomes are targets only, not selector inputs.",
        "",
        f"Ridge L2: `{args.ridge_alpha}`",
        f"Top-k base feature settings: `{args.top_k}`",
        "",
        "## Key Rows",
        "",
        md_table(
            key_rows[: args.max_md_rows],
            [
                "split",
                "method",
                "test_images",
                "score",
                "oracle_score",
                "regret_vs_oracle",
                "score_win_frac",
                "positive_score_frac",
                "policy_accuracy",
                "family_accuracy",
                "predicted_policy_counts",
            ],
        ),
        "",
        "## Oracle Policy Counts",
        "",
    ]
    lines.append(
        md_table(
            [
                {
                    "group": group,
                    "count": count,
                }
                for group, count in sorted(Counter(str(row["oracle_policy"]) for row in labels).items())
            ],
            ["group", "count"],
        )
    )
    lines.extend(
        [
            "",
            "## Top Family-Separating Decoder-Safe Features",
            "",
            md_table(feature_sep[:16], ["feature", "oracle_family_anova_r2", "mean", "std"]),
            "",
            "## Interpretation Guardrails",
            "",
            "- `oracle_all` is an upper bound over already executable E236 policies, not a deployable method.",
            "- `best_fixed_train` is the strongest fixed-policy baseline chosen only from the training side of each split.",
            "- `long_ridge_*` predicts policy scores from decoder-safe global features plus alpha-map design statistics; it still operates at image-policy level and is therefore a readiness audit, not the final local neural head.",
            "- If resubstitution is strong but leave-one-image-out or cross-dataset transfer is weak, the next paper-facing step remains an in-codec local controller trained on independent fit/calibration data with false-positive/fallback regularization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    top_ks = [int(part.strip()) for part in args.top_k.split(",") if part.strip()]
    feature_rows, feature_cols = load_feature_rows(FEATURE_INPUTS)
    rows_by_item, alpha_cols = load_e236_rows(E236_INPUTS)
    items, policies, families = common_items_and_policies(rows_by_item, feature_rows)
    oracle_policy, oracle_family, label_rows = oracle_labels(items, rows_by_item, policies)
    feature_sep = anova_feature_scores(items, oracle_family, feature_rows, feature_cols)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split, train_items, test_items in split_defs(items):
        rows, preds = evaluate_split(
            split,
            train_items,
            test_items,
            rows_by_item,
            feature_rows,
            feature_cols,
            alpha_cols,
            policies,
            families,
            oracle_policy,
            oracle_family,
            top_ks,
            args.ridge_alpha,
        )
        summary_rows.extend(rows)
        prediction_rows.extend(preds)

    rows, preds = evaluate_leave_one_image_out(
        items,
        rows_by_item,
        feature_rows,
        feature_cols,
        alpha_cols,
        policies,
        families,
        oracle_policy,
        oracle_family,
        top_ks,
        args.ridge_alpha,
    )
    summary_rows.extend(rows)
    prediction_rows.extend(preds)

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".summary.csv"), summary_rows)
    write_csv(prefix.with_suffix(".predictions.csv"), prediction_rows)
    write_csv(prefix.with_suffix(".labels.csv"), label_rows)
    write_csv(prefix.with_suffix(".feature_separation.csv"), feature_sep)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "experiment": "E237 EF-LIC local policy controller split audit",
                "args": vars(args),
                "inputs": {
                    "features": {k: str(v) for k, v in FEATURE_INPUTS.items()},
                    "e236": {k: str(v) for k, v in E236_INPUTS.items()},
                },
                "items": len(items),
                "policies": policies,
                "families": families,
                "alpha_features": alpha_cols,
                "summary_rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    write_markdown(prefix.with_suffix(".md"), summary_rows, label_rows, feature_sep, args)
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
