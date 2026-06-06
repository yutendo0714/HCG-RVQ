#!/usr/bin/env python3
"""E235 compact EF-LIC controller-readiness audit.

E234 established a small executable no-sidebit branch vocabulary. This script
asks the next question before adding a trainable codec-path head: can the E234
compact branch choices be predicted from decoder-safe context features, and can
a simple fallback rule reduce false positives?

This is not a final paper method. It is a design gate for the learned
branch/strength controller that should live after EF-LIC `_mean_scale`.
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

E234_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e234_eflic_kodak24_branch_controller_scaffold.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e234_eflic_clicpro41_branch_controller_scaffold.csv",
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


def to_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
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
    if row.get("score_dists_3lpips", "") != "":
        return to_float(row["score_dists_3lpips"], 0.0)
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


def load_e234_rows(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            for raw in csv.DictReader(fobj):
                valid = (
                    abs(to_float(raw.get("delta_bpp"), 1.0)) < 1e-12
                    and to_float(raw.get("max_decode_diff"), 1.0) == 0.0
                    and int(to_float(raw.get("nonfinite"), 1.0)) == 0
                )
                if not valid:
                    continue
                item = (dataset, raw["image"])
                preset = raw["preset"]
                rows_by_item[item][preset] = {
                    "dataset": dataset,
                    "image": raw["image"],
                    "preset": preset,
                    "family": raw["family"],
                    "score": score_from_row(raw),
                    "delta_dists": to_float(raw.get("delta_dists"), 0.0),
                    "delta_lpips": to_float(raw.get("delta_lpips"), 0.0),
                    "delta_psnr": to_float(raw.get("delta_psnr"), 0.0),
                    "y_mismatch_frac": to_float(raw.get("y_mismatch"), 0.0) / max(1.0, to_float(raw.get("y_total"), 1.0)),
                    "geometry_delta_rms": to_float(raw.get("y_avg_geometry_delta_rms"), 0.0),
                }
    return dict(rows_by_item)


def common_items_and_presets(
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    items = sorted(item for item in rows_by_item if item in feature_rows)
    if not items:
        raise SystemExit("no overlapping E233 feature rows and E234 preset rows")
    common = set(rows_by_item[items[0]])
    for item in items[1:]:
        common &= set(rows_by_item[item])
    if "zero" not in common:
        raise SystemExit("E234 rows must include zero fallback")
    return items, sorted(common)


def feature_matrix(items: list[tuple[str, str]], feature_rows: dict[tuple[str, str], dict[str, str]], cols: list[str]) -> np.ndarray:
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
    presets: list[str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], list[dict[str, object]]]:
    best_preset: dict[tuple[str, str], str] = {}
    best_family: dict[tuple[str, str], str] = {}
    rows: list[dict[str, object]] = []
    for item in items:
        choice = min(presets, key=lambda preset: rows_by_item[item][preset]["score"])
        row = rows_by_item[item][choice]
        best_preset[item] = choice
        best_family[item] = str(row["family"])
        rows.append(
            {
                "dataset": item[0],
                "image": item[1],
                "oracle_preset": choice,
                "oracle_family": row["family"],
                "oracle_score": row["score"],
                "oracle_delta_dists": row["delta_dists"],
                "oracle_delta_lpips": row["delta_lpips"],
                "oracle_delta_psnr": row["delta_psnr"],
            }
        )
    return best_preset, best_family, rows


def anova_feature_scores(
    items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
) -> list[dict[str, object]]:
    x = feature_matrix(items, feature_rows, cols)
    y = np.array([labels[item] for item in items], dtype=object)
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
                "family_anova_r2": between / total if total > 1e-18 else 0.0,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return sorted(out, key=lambda row: to_float(row["family_anova_r2"]), reverse=True)


def select_cols(
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


def standardize(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = feature_matrix(train_items, feature_rows, cols)
    x_test = feature_matrix(test_items, feature_rows, cols)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


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
    x_train, x_test = standardize(train_items, test_items, feature_rows, cols)
    y_train = np.array([labels[item] for item in train_items], dtype=object)
    centroids = {lab: x_train[y_train == lab].mean(axis=0) for lab in sorted(set(y_train))}
    return {
        item: min(centroids, key=lambda lab: float(np.sum((vec - centroids[lab]) ** 2)))
        for item, vec in zip(test_items, x_test)
    }


def ridge_predicted_scores(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    cols: list[str],
    presets: list[str],
    ridge_alpha: float,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, object]]:
    if not cols:
        train_mean = {preset: mean(rows_by_item[item][preset]["score"] for item in train_items) for preset in presets}
        return {item: dict(train_mean) for item in test_items}, {"ridge_rank": 0, "condition": 0.0}
    x_train, x_test = standardize(train_items, test_items, feature_rows, cols)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    y = np.array([[rows_by_item[item][preset]["score"] for preset in presets] for item in train_items], dtype=np.float64)
    reg = np.eye(x_train.shape[1], dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0
    lhs = x_train.T @ x_train + reg
    beta = np.linalg.solve(lhs, x_train.T @ y)
    pred = x_test @ beta
    return (
        {item: {preset: float(val) for preset, val in zip(presets, row)} for item, row in zip(test_items, pred)},
        {"ridge_rank": int(np.linalg.matrix_rank(x_train)), "condition": float(np.linalg.cond(lhs))},
    )


def tune_fallback_margin(
    train_items: list[tuple[str, str]],
    pred_scores: dict[tuple[str, str], dict[str, float]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    presets: list[str],
) -> float:
    margins = np.linspace(-0.006, 0.001, 71)
    best_margin = 0.0
    best_score = float("inf")
    for margin in margins:
        vals = []
        for item in train_items:
            best = min(presets, key=lambda preset: pred_scores[item][preset])
            chosen = best if pred_scores[item][best] < margin else "zero"
            vals.append(float(rows_by_item[item][chosen]["score"]))
        value = mean(vals)
        if value < best_score:
            best_score = value
            best_margin = float(margin)
    return best_margin


def best_fixed_key(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    presets: list[str],
) -> str:
    return min(presets, key=lambda preset: mean(rows_by_item[item][preset]["score"] for item in train_items))


def best_key_by_family(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    presets: list[str],
) -> dict[str, str]:
    families = sorted({str(rows_by_item[train_items[0]][preset]["family"]) for preset in presets})
    out = {}
    for family in families:
        family_presets = [preset for preset in presets if str(rows_by_item[train_items[0]][preset]["family"]) == family]
        out[family] = best_fixed_key(train_items, rows_by_item, family_presets)
    return out


def summarize_predictions(
    *,
    split: str,
    method: str,
    train_count: int,
    predictions: dict[tuple[str, str], str],
    oracle_preset: dict[tuple[str, str], str],
    oracle_family: dict[tuple[str, str], str],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    scores = []
    oracle_scores = []
    for item, pred in predictions.items():
        pred_row = rows_by_item[item][pred]
        oracle_row = rows_by_item[item][oracle_preset[item]]
        score = float(pred_row["score"])
        oracle_score = float(oracle_row["score"])
        scores.append(score)
        oracle_scores.append(oracle_score)
        rows.append(
            {
                "split": split,
                "method": method,
                "dataset": item[0],
                "image": item[1],
                "predicted_preset": pred,
                "predicted_family": pred_row["family"],
                "oracle_preset": oracle_preset[item],
                "oracle_family": oracle_family[item],
                "score": score,
                "oracle_score": oracle_score,
                "regret_vs_oracle": score - oracle_score,
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
        "score_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in scores),
        "preset_accuracy": mean(1.0 if r["predicted_preset"] == r["oracle_preset"] else 0.0 for r in rows),
        "family_accuracy": mean(1.0 if r["predicted_family"] == r["oracle_family"] else 0.0 for r in rows),
        "predicted_preset_counts": json.dumps(dict(Counter(str(r["predicted_preset"]) for r in rows)), sort_keys=True),
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
    presets: list[str],
    oracle_preset: dict[tuple[str, str], str],
    oracle_family: dict[tuple[str, str], str],
    top_ks: list[int],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    predictions_all: list[dict[str, object]] = []

    policies: list[tuple[str, dict[tuple[str, str], str], dict[str, object] | None]] = []
    policies.append(("oracle_all", {item: oracle_preset[item] for item in test_items}, None))
    policies.append(("zero", {item: "zero" for item in test_items}, None))
    fixed = best_fixed_key(train_items, rows_by_item, presets)
    policies.append(("best_fixed_train", {item: fixed for item in test_items}, {"fixed_preset": fixed}))
    family_best = best_key_by_family(train_items, rows_by_item, presets)
    policies.append(
        (
            "true_family_train_best_preset",
            {item: family_best[oracle_family[item]] for item in test_items},
            {"family_best": json.dumps(family_best, sort_keys=True)},
        )
    )

    for name, preds, extra in policies:
        summary, rows = summarize_predictions(
            split=split,
            method=name,
            train_count=len(train_items),
            predictions=preds,
            oracle_preset=oracle_preset,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra=extra,
        )
        summaries.append(summary)
        predictions_all.extend(rows)

    for top_k in top_ks:
        cols = select_cols(train_items, oracle_family, feature_rows, feature_cols, top_k)
        nearest_family = nearest_centroid_family(train_items, test_items, oracle_family, feature_rows, cols)
        nearest_preds = {item: family_best[nearest_family[item]] for item in test_items}
        label = f"nearest_family_top{top_k if top_k > 0 else 'all'}"
        summary, rows = summarize_predictions(
            split=split,
            method=label,
            train_count=len(train_items),
            predictions=nearest_preds,
            oracle_preset=oracle_preset,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={"feature_count": len(cols), "selected_features": json.dumps(cols[:12])},
        )
        summaries.append(summary)
        predictions_all.extend(rows)

        pred_train, ridge_meta_train = ridge_predicted_scores(
            train_items, train_items, rows_by_item, feature_rows, cols, presets, ridge_alpha
        )
        pred_test, ridge_meta = ridge_predicted_scores(
            train_items, test_items, rows_by_item, feature_rows, cols, presets, ridge_alpha
        )
        ridge_preds = {item: min(presets, key=lambda preset: pred_test[item][preset]) for item in test_items}
        label = f"ridge_score_top{top_k if top_k > 0 else 'all'}_l2{ridge_alpha:g}"
        summary, rows = summarize_predictions(
            split=split,
            method=label,
            train_count=len(train_items),
            predictions=ridge_preds,
            oracle_preset=oracle_preset,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={"feature_count": len(cols), "selected_features": json.dumps(cols[:12]), **ridge_meta},
        )
        summaries.append(summary)
        predictions_all.extend(rows)

        margin = tune_fallback_margin(train_items, pred_train, rows_by_item, presets)
        gated_preds = {}
        for item in test_items:
            best = min(presets, key=lambda preset: pred_test[item][preset])
            gated_preds[item] = best if pred_test[item][best] < margin else "zero"
        gated_label = f"ridge_fallback_top{top_k if top_k > 0 else 'all'}_l2{ridge_alpha:g}"
        summary, rows = summarize_predictions(
            split=split,
            method=gated_label,
            train_count=len(train_items),
            predictions=gated_preds,
            oracle_preset=oracle_preset,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
            extra={
                "feature_count": len(cols),
                "selected_features": json.dumps(cols[:12]),
                "fallback_margin": margin,
                **ridge_meta_train,
            },
        )
        summaries.append(summary)
        predictions_all.extend(rows)

    return summaries, predictions_all


def evaluate_leave_one_image_out(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, Any]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    presets: list[str],
    oracle_preset: dict[tuple[str, str], str],
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
            presets,
            oracle_preset,
            oracle_family,
            top_ks,
            ridge_alpha,
        )
        for row in preds:
            grouped[str(row["method"])].append(row)
    summaries = []
    predictions = []
    for method, rows in grouped.items():
        preds = {tuple([str(r["dataset"]), str(r["image"])]): str(r["predicted_preset"]) for r in rows}
        summary, collapsed = summarize_predictions(
            split="pooled_leave_one_image_out",
            method=method,
            train_count=len(items) - 1,
            predictions=preds,
            oracle_preset=oracle_preset,
            oracle_family=oracle_family,
            rows_by_item=rows_by_item,
        )
        summaries.append(summary)
        predictions.extend(collapsed)
    return summaries, predictions


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            vals.append(f"{val:+.8f}" if isinstance(val, float) else str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e235_eflic_compact_controller_readiness")
    p.add_argument("--top-k", default="8,16,32,64,0")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    top_ks = [int(part.strip()) for part in args.top_k.split(",") if part.strip()]
    feature_rows, feature_cols = load_feature_rows(FEATURE_INPUTS)
    rows_by_item = load_e234_rows(E234_INPUTS)
    items, presets = common_items_and_presets(rows_by_item, feature_rows)
    oracle_preset, oracle_family, label_rows = oracle_labels(items, rows_by_item, presets)
    feature_sep = anova_feature_scores(items, oracle_family, feature_rows, feature_cols)

    items_by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        items_by_dataset[item[0]].append(item)

    split_defs: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = [("pooled_resub", items, items)]
    for dataset, ds_items in sorted(items_by_dataset.items()):
        split_defs.append((f"{dataset}_resub", ds_items, ds_items))
    if "clicpro41" in items_by_dataset and "kodak24" in items_by_dataset:
        split_defs.append(("train_clicpro41_test_kodak24", items_by_dataset["clicpro41"], items_by_dataset["kodak24"]))
        split_defs.append(("train_kodak24_test_clicpro41", items_by_dataset["kodak24"], items_by_dataset["clicpro41"]))

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split, train_items, test_items in split_defs:
        rows, preds = evaluate_split(
            split,
            train_items,
            test_items,
            rows_by_item,
            feature_rows,
            feature_cols,
            presets,
            oracle_preset,
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
        presets,
        oracle_preset,
        oracle_family,
        top_ks,
        args.ridge_alpha,
    )
    summary_rows.extend(rows)
    prediction_rows.extend(preds)

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".labels.csv"), label_rows)
    write_csv(prefix.with_suffix(".feature_separation.csv"), feature_sep)
    write_csv(prefix.with_suffix(".summary.csv"), summary_rows)
    write_csv(prefix.with_suffix(".predictions.csv"), prediction_rows)

    payload = {
        "experiment": "E235 EF-LIC compact controller readiness",
        "items": len(items),
        "datasets": dict(Counter(item[0] for item in items)),
        "presets": presets,
        "feature_count": len(feature_cols),
        "top_k": top_ks,
        "ridge_alpha": args.ridge_alpha,
        "oracle_family_counts": dict(Counter(oracle_family.values())),
        "oracle_preset_counts": dict(Counter(oracle_preset.values())),
        "feature_inputs": {key: str(path) for key, path in FEATURE_INPUTS.items()},
        "e234_inputs": {key: str(path) for key, path in E234_INPUTS.items()},
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    focus_splits = {
        "pooled_resub",
        "pooled_leave_one_image_out",
        "train_clicpro41_test_kodak24",
        "train_kodak24_test_clicpro41",
    }
    focus_methods = {
        "oracle_all",
        "zero",
        "best_fixed_train",
        "true_family_train_best_preset",
        "ridge_score_top16_l210",
        "ridge_fallback_top16_l210",
        "ridge_score_top32_l210",
        "ridge_fallback_top32_l210",
        "ridge_score_topall_l210",
        "ridge_fallback_topall_l210",
    }
    focus_rows = [
        row
        for row in summary_rows
        if row["split"] in focus_splits and (row["method"] in focus_methods or str(row["method"]).startswith("nearest_family_top16"))
    ]
    method_rank = {
        "oracle_all": 0,
        "zero": 1,
        "best_fixed_train": 2,
        "true_family_train_best_preset": 3,
    }
    focus_rows.sort(key=lambda row: (str(row["split"]), method_rank.get(str(row["method"]), 10), str(row["method"])))

    with prefix.with_suffix(".md").open("w") as fobj:
        fobj.write("# E235 EF-LIC Compact Controller Readiness\n\n")
        fobj.write(
            "This diagnostic uses only the E234 executable no-sidebit preset vocabulary and E233 decoder-safe features. "
            "Lower `score = delta_dists + 3 * delta_lpips` is better.\n\n"
        )
        for key, value in payload.items():
            fobj.write(f"- `{key}`: `{value}`\n")
        fobj.write("\n## Focus Summary\n\n")
        fobj.write(
            md_table(
                focus_rows,
                [
                    "split",
                    "method",
                    "test_images",
                    "score",
                    "oracle_score",
                    "regret_vs_oracle",
                    "score_win_frac",
                    "preset_accuracy",
                    "family_accuracy",
                    "predicted_preset_counts",
                ],
            )
        )
        fobj.write("\n## Top Family-Separating Decoder-Safe Features\n\n")
        fobj.write(md_table(feature_sep[:30], ["feature", "family_anova_r2", "mean", "std", "min", "max"]))
        fobj.write(
            "\nInterpretation: strong resubstitution with weak held-out transfer means the compact vocabulary has readable signal, "
            "but a paper-main method still needs an in-codec trained local controller with fallback/false-positive regularization.\n"
        )
    print(f"wrote {prefix.with_suffix('.summary.csv')}, {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
