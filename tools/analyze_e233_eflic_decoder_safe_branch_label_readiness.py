#!/usr/bin/env python3
"""Check whether E232 branch choices are readable from decoder-safe features.

E233 dumps context features that are available to both encoder and decoder
before each EF-LIC y-slice quantization decision. This script joins those
features to the E232 codec-valid branch library and asks a narrower question:
can the per-image oracle branch family be predicted without using post-decision
residual/index features?

This is a controller-readiness diagnostic, not a final paper method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_e232_eflic_branch_library_audit as e232  # noqa: E402


FEATURE_INPUTS = {
    "clicpro41": Path("experiments/analysis/e233_eflic_clicpro41_decoder_safe_branch_features.csv"),
    "kodak24": Path("experiments/analysis/e233_eflic_kodak24_decoder_safe_branch_features.csv"),
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
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 8) -> str:
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, object]], columns: list[str], digits: int = 8) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col, ""), digits) for col in columns) + " |")
    return "\n".join(out) + "\n"


def load_feature_rows(paths: dict[str, Path]) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    feature_cols: set[str] = set()
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            reader = csv.DictReader(fobj)
            for row in reader:
                item = (dataset, row["image"])
                rows[item] = row
                for col, value in row.items():
                    if col in DROP_FEATURE_COLUMNS:
                        continue
                    if math.isfinite(to_float(value)):
                        feature_cols.add(col)
    return rows, sorted(feature_cols)


def feature_matrix(
    items: list[tuple[str, str]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
) -> np.ndarray:
    arr = np.zeros((len(items), len(feature_cols)), dtype=np.float64)
    for i, item in enumerate(items):
        row = feature_rows[item]
        for j, col in enumerate(feature_cols):
            value = to_float(row.get(col), 0.0)
            arr[i, j] = value if math.isfinite(value) else 0.0
    return arr


def oracle_by_item(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    candidate_keys: set[str],
) -> dict[tuple[str, str], dict[str, object]]:
    records = e232.oracle_records(items, rows_by_item, candidate_keys)
    return {(str(r["dataset"]), str(r["image"])): r for r in records}


def candidate_score(
    item: tuple[str, str],
    key: str,
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
) -> float:
    return e232.score(rows_by_item[item][key])


def best_key(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    keys: set[str],
) -> str:
    return min(keys, key=lambda key: mean(candidate_score(item, key, rows_by_item) for item in train_items))


def best_key_by_family(
    train_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    keys: set[str],
) -> dict[str, str]:
    out = {}
    families = sorted({e232.family_for_key(key) for key in keys})
    for family in families:
        family_keys = {key for key in keys if e232.family_for_key(key) == family}
        out[family] = best_key(train_items, rows_by_item, family_keys)
    return out


def anova_feature_scores(
    items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
) -> list[dict[str, object]]:
    x = feature_matrix(items, feature_rows, feature_cols)
    y = np.array([labels[item] for item in items], dtype=object)
    out: list[dict[str, object]] = []
    for j, col in enumerate(feature_cols):
        values = x[:, j]
        total = float(np.sum((values - values.mean()) ** 2))
        if total <= 1e-18:
            r2 = 0.0
        else:
            between = 0.0
            for family in sorted(set(y)):
                subset = values[y == family]
                between += float(len(subset) * (subset.mean() - values.mean()) ** 2)
            r2 = between / total
        out.append(
            {
                "feature": col,
                "family_anova_r2": r2,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return sorted(out, key=lambda row: to_float(row["family_anova_r2"]), reverse=True)


def selected_features(
    train_items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    top_k: int,
) -> list[str]:
    if top_k <= 0 or top_k >= len(feature_cols):
        return feature_cols
    ranked = anova_feature_scores(train_items, labels, feature_rows, feature_cols)
    return [str(row["feature"]) for row in ranked[:top_k]]


def nearest_centroid_predict(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
) -> dict[tuple[str, str], str]:
    if not feature_cols:
        common = Counter(labels[item] for item in train_items).most_common(1)[0][0]
        return {item: common for item in test_items}
    x_train = feature_matrix(train_items, feature_rows, feature_cols)
    x_test = feature_matrix(test_items, feature_rows, feature_cols)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    train_y = np.array([labels[item] for item in train_items], dtype=object)
    centroids = {}
    for family in sorted(set(train_y)):
        centroids[family] = x_train[train_y == family].mean(axis=0)
    predictions = {}
    for item, vec in zip(test_items, x_test):
        best_family = min(centroids, key=lambda fam: float(np.sum((vec - centroids[fam]) ** 2)))
        predictions[item] = best_family
    return predictions


def ridge_score_predict(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    keys: set[str],
    ridge_alpha: float,
) -> dict[tuple[str, str], str]:
    key_list = sorted(keys)
    if not feature_cols:
        train_best = best_key(train_items, rows_by_item, keys)
        return {item: train_best for item in test_items}
    x_train = feature_matrix(train_items, feature_rows, feature_cols)
    x_test = feature_matrix(test_items, feature_rows, feature_cols)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    y = np.array(
        [[candidate_score(item, key, rows_by_item) for key in key_list] for item in train_items],
        dtype=np.float64,
    )
    reg = np.eye(x_train.shape[1], dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y)
    pred_scores = x_test @ beta
    out = {}
    for item, row in zip(test_items, pred_scores):
        out[item] = key_list[int(np.argmin(row))]
    return out


def summarize_policy(
    *,
    split: str,
    method: str,
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    keys: set[str],
    oracle: dict[tuple[str, str], dict[str, object]],
    predicted_families: dict[tuple[str, str], str] | None = None,
    predicted_keys: dict[tuple[str, str], str] | None = None,
    fixed_key_override: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    family_best = best_key_by_family(train_items, rows_by_item, keys)
    if fixed_key_override is None and predicted_families is None and predicted_keys is None:
        fixed_key = best_key(train_items, rows_by_item, keys)
    else:
        fixed_key = fixed_key_override

    pred_rows: list[dict[str, object]] = []
    scores = []
    oracle_scores = []
    family_hits = []
    key_hits = []
    for item in test_items:
        oracle_row = oracle[item]
        oracle_key = str(oracle_row["choice"])
        oracle_family = str(oracle_row["family"])
        if predicted_keys is not None:
            pred_key = predicted_keys[item]
            pred_family = e232.family_for_key(pred_key)
        elif fixed_key is not None:
            pred_family = e232.family_for_key(fixed_key)
            pred_key = fixed_key
        else:
            pred_family = str(predicted_families[item])
            pred_key = family_best[pred_family]
        value = candidate_score(item, pred_key, rows_by_item)
        oracle_value = to_float(oracle_row["score"], 0.0)
        scores.append(value)
        oracle_scores.append(oracle_value)
        family_hits.append(1.0 if pred_family == oracle_family else 0.0)
        key_hits.append(1.0 if pred_key == oracle_key else 0.0)
        pred_rows.append(
            {
                "split": split,
                "method": method,
                "dataset": item[0],
                "image": item[1],
                "predicted_family": pred_family,
                "predicted_key": pred_key,
                "oracle_family": oracle_family,
                "oracle_key": oracle_key,
                "score": value,
                "oracle_score": oracle_value,
                "regret_vs_oracle": value - oracle_value,
            }
        )
    summary = {
        "split": split,
        "method": method,
        "train_images": len(train_items),
        "test_images": len(test_items),
        "score": mean(scores),
        "oracle_score": mean(oracle_scores),
        "regret_vs_oracle": mean(scores) - mean(oracle_scores),
        "score_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in scores),
        "family_accuracy": mean(family_hits),
        "candidate_accuracy": mean(key_hits),
        "predicted_family_counts": json.dumps(dict(Counter(r["predicted_family"] for r in pred_rows)), sort_keys=True),
        "predicted_key_counts": json.dumps(dict(Counter(r["predicted_key"] for r in pred_rows)), sort_keys=True),
    }
    return summary, pred_rows


def oracle_summary(
    split: str,
    test_items: list[tuple[str, str]],
    oracle: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    vals = [to_float(oracle[item]["score"], 0.0) for item in test_items]
    return {
        "split": split,
        "method": "oracle_all",
        "train_images": len(test_items),
        "test_images": len(test_items),
        "score": mean(vals),
        "oracle_score": mean(vals),
        "regret_vs_oracle": 0.0,
        "score_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in vals),
        "family_accuracy": 1.0,
        "candidate_accuracy": 1.0,
        "predicted_family_counts": json.dumps(dict(Counter(str(oracle[item]["family"]) for item in test_items)), sort_keys=True),
        "predicted_key_counts": json.dumps(dict(Counter(str(oracle[item]["choice"]) for item in test_items)), sort_keys=True),
    }


def evaluate_split(
    split: str,
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    keys: set[str],
    oracle: dict[tuple[str, str], dict[str, object]],
    labels: dict[tuple[str, str], str],
    top_ks: list[int],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows = [oracle_summary(split, test_items, oracle)]
    prediction_rows: list[dict[str, object]] = []

    zero_summary, zero_preds = summarize_policy(
        split=split,
        method="zero",
        train_items=train_items,
        test_items=test_items,
        rows_by_item=rows_by_item,
        keys=keys,
        oracle=oracle,
        fixed_key_override="zero",
    )
    summary_rows.append(zero_summary)
    prediction_rows.extend(zero_preds)

    fixed_summary, fixed_preds = summarize_policy(
        split=split,
        method="best_fixed_train",
        train_items=train_items,
        test_items=test_items,
        rows_by_item=rows_by_item,
        keys=keys,
        oracle=oracle,
    )
    summary_rows.append(fixed_summary)
    prediction_rows.extend(fixed_preds)

    true_family = {item: labels[item] for item in test_items}
    tf_summary, tf_preds = summarize_policy(
        split=split,
        method="true_family_train_best_candidate",
        train_items=train_items,
        test_items=test_items,
        rows_by_item=rows_by_item,
        keys=keys,
        oracle=oracle,
        predicted_families=true_family,
    )
    summary_rows.append(tf_summary)
    prediction_rows.extend(tf_preds)

    for top_k in top_ks:
        cols = selected_features(train_items, labels, feature_rows, feature_cols, top_k)
        preds = nearest_centroid_predict(train_items, test_items, labels, feature_rows, cols)
        label = f"nearest_centroid_top{top_k if top_k > 0 else 'all'}"
        centroid_summary, centroid_preds = summarize_policy(
            split=split,
            method=label,
            train_items=train_items,
            test_items=test_items,
            rows_by_item=rows_by_item,
            keys=keys,
            oracle=oracle,
            predicted_families=preds,
        )
        centroid_summary["feature_count"] = len(cols)
        centroid_summary["selected_features"] = json.dumps(cols[:12])
        summary_rows.append(centroid_summary)
        prediction_rows.extend(centroid_preds)

        ridge_keys = ridge_score_predict(
            train_items,
            test_items,
            rows_by_item,
            feature_rows,
            cols,
            keys,
            ridge_alpha,
        )
        ridge_label = f"ridge_score_top{top_k if top_k > 0 else 'all'}_l2{ridge_alpha:g}"
        ridge_summary, ridge_preds = summarize_policy(
            split=split,
            method=ridge_label,
            train_items=train_items,
            test_items=test_items,
            rows_by_item=rows_by_item,
            keys=keys,
            oracle=oracle,
            predicted_keys=ridge_keys,
        )
        ridge_summary["feature_count"] = len(cols)
        ridge_summary["selected_features"] = json.dumps(cols[:12])
        summary_rows.append(ridge_summary)
        prediction_rows.extend(ridge_preds)
    return summary_rows, prediction_rows


def evaluate_leave_one_image_out(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    feature_cols: list[str],
    keys: set[str],
    oracle: dict[tuple[str, str], dict[str, object]],
    labels: dict[tuple[str, str], str],
    top_ks: list[int],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows = [oracle_summary("pooled_leave_one_image_out", items, oracle)]
    prediction_rows: list[dict[str, object]] = []
    all_methods: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        train_items = [other for other in items if other != item]
        _, zero_preds = summarize_policy(
            split="pooled_leave_one_image_out",
            method="zero",
            train_items=train_items,
            test_items=[item],
            rows_by_item=rows_by_item,
            keys=keys,
            oracle=oracle,
            fixed_key_override="zero",
        )
        all_methods["zero"].extend(zero_preds)
        _, fixed_preds = summarize_policy(
            split="pooled_leave_one_image_out",
            method="best_fixed_train",
            train_items=train_items,
            test_items=[item],
            rows_by_item=rows_by_item,
            keys=keys,
            oracle=oracle,
        )
        all_methods["best_fixed_train"].extend(fixed_preds)
        for top_k in top_ks:
            cols = selected_features(train_items, labels, feature_rows, feature_cols, top_k)
            preds = nearest_centroid_predict(train_items, [item], labels, feature_rows, cols)
            label = f"nearest_centroid_top{top_k if top_k > 0 else 'all'}"
            _, centroid_preds = summarize_policy(
                split="pooled_leave_one_image_out",
                method=label,
                train_items=train_items,
                test_items=[item],
                rows_by_item=rows_by_item,
                keys=keys,
                oracle=oracle,
                predicted_families=preds,
            )
            all_methods[label].extend(centroid_preds)

            ridge_keys = ridge_score_predict(
                train_items,
                [item],
                rows_by_item,
                feature_rows,
                cols,
                keys,
                ridge_alpha,
            )
            ridge_label = f"ridge_score_top{top_k if top_k > 0 else 'all'}_l2{ridge_alpha:g}"
            _, ridge_preds = summarize_policy(
                split="pooled_leave_one_image_out",
                method=ridge_label,
                train_items=train_items,
                test_items=[item],
                rows_by_item=rows_by_item,
                keys=keys,
                oracle=oracle,
                predicted_keys=ridge_keys,
            )
            all_methods[ridge_label].extend(ridge_preds)

    for method, preds in all_methods.items():
        scores = [to_float(r["score"], 0.0) for r in preds]
        oracle_scores = [to_float(r["oracle_score"], 0.0) for r in preds]
        summary_rows.append(
            {
                "split": "pooled_leave_one_image_out",
                "method": method,
                "train_images": len(items) - 1,
                "test_images": len(items),
                "score": mean(scores),
                "oracle_score": mean(oracle_scores),
                "regret_vs_oracle": mean(scores) - mean(oracle_scores),
                "score_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in scores),
                "family_accuracy": mean(1.0 if r["predicted_family"] == r["oracle_family"] else 0.0 for r in preds),
                "candidate_accuracy": mean(1.0 if r["predicted_key"] == r["oracle_key"] else 0.0 for r in preds),
                "predicted_family_counts": json.dumps(dict(Counter(r["predicted_family"] for r in preds)), sort_keys=True),
                "predicted_key_counts": json.dumps(dict(Counter(r["predicted_key"] for r in preds)), sort_keys=True),
            }
        )
        prediction_rows.extend(preds)
    return summary_rows, prediction_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e233_eflic_decoder_safe_branch_label_readiness"))
    p.add_argument("--top-k", type=str, default="8,16,32,64,0", help="comma-separated feature counts; 0 means all")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    top_ks = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]

    rows_by_item, _candidate_meta, items_by_dataset = e232.load_rows()
    feature_rows, feature_cols = load_feature_rows(FEATURE_INPUTS)
    items = [item for item in e232.all_items(items_by_dataset) if item in feature_rows]
    if not items:
        raise SystemExit("no overlapping E232/E233 items")
    keys = e232.item_keys(items, rows_by_item)
    oracle = oracle_by_item(items, rows_by_item, keys)
    labels = {item: str(oracle[item]["family"]) for item in items}

    label_rows = []
    for item in items:
        rec = oracle[item]
        label_rows.append(
            {
                "dataset": item[0],
                "image": item[1],
                "oracle_choice": rec["choice"],
                "oracle_family": rec["family"],
                "oracle_score": rec["score"],
                "oracle_delta_dists": rec["delta_dists"],
                "oracle_delta_lpips": rec["delta_lpips"],
            }
        )

    feature_sep = anova_feature_scores(items, labels, feature_rows, feature_cols)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_defs: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = [
        ("pooled_resub", items, items),
    ]
    for dataset, ds_items in sorted(items_by_dataset.items()):
        ds_items = [item for item in ds_items if item in feature_rows]
        if ds_items:
            split_defs.append((f"{dataset}_resub", ds_items, ds_items))
    if "clicpro41" in items_by_dataset and "kodak24" in items_by_dataset:
        clic = [item for item in items_by_dataset["clicpro41"] if item in feature_rows]
        kodak = [item for item in items_by_dataset["kodak24"] if item in feature_rows]
        split_defs.extend(
            [
                ("train_clicpro41_test_kodak24", clic, kodak),
                ("train_kodak24_test_clicpro41", kodak, clic),
            ]
        )

    for split, train_items, test_items in split_defs:
        if not train_items or not test_items:
            continue
        rows, preds = evaluate_split(
            split,
            train_items,
            test_items,
            rows_by_item,
            feature_rows,
            feature_cols,
            keys,
            oracle,
            labels,
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
        keys,
        oracle,
        labels,
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
    summary_json = {
        "experiment": "E233 EF-LIC decoder-safe branch label readiness",
        "items": len(items),
        "datasets": dict(Counter(item[0] for item in items)),
        "candidate_count": len(keys),
        "feature_count": len(feature_cols),
        "top_k": top_ks,
        "ridge_alpha": args.ridge_alpha,
        "label_counts": dict(Counter(labels.values())),
        "feature_inputs": {k: str(v) for k, v in FEATURE_INPUTS.items()},
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary_json, indent=2, sort_keys=True))

    top_summary = [
        row
        for row in summary_rows
        if row["split"]
        in {
            "pooled_resub",
            "pooled_leave_one_image_out",
            "train_clicpro41_test_kodak24",
            "train_kodak24_test_clicpro41",
        }
    ]
    method_order = {
        "oracle_all": 0,
        "zero": 1,
        "best_fixed_train": 2,
        "true_family_train_best_candidate": 3,
    }
    top_summary = sorted(
        top_summary,
        key=lambda row: (
            str(row["split"]),
            method_order.get(str(row["method"]), 10),
            str(row["method"]),
        ),
    )
    top_features = feature_sep[:30]

    with prefix.with_suffix(".md").open("w") as fobj:
        fobj.write("# E233 EF-LIC Decoder-Safe Branch Label Readiness\n\n")
        fobj.write("Lower `score = delta_dists + 3 * delta_lpips` is better. This is a diagnostic over E232 branch candidates joined with E233 decoder-safe features.\n\n")
        fobj.write("## Dataset and Label Counts\n\n")
        for key, value in summary_json.items():
            fobj.write(f"- `{key}`: `{value}`\n")
        fobj.write("\n## Controller-Readiness Summary\n\n")
        fobj.write(md_table(top_summary, [
            "split",
            "method",
            "test_images",
            "score",
            "oracle_score",
            "regret_vs_oracle",
            "score_win_frac",
            "family_accuracy",
            "candidate_accuracy",
            "predicted_family_counts",
        ], digits=8))
        fobj.write("\n## Top Decoder-Safe Feature Separations\n\n")
        fobj.write(md_table(top_features, ["feature", "family_anova_r2", "mean", "std"], digits=6))
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- These predictors use only E233 predecision context features; image shape columns are excluded to avoid dataset-proxy leakage.\n")
        fobj.write("- Same-table rows estimate capacity; leave-one-image-out and leave-dataset-out rows estimate whether a simple hand-built controller is stable enough.\n")
        fobj.write("- If leave-dataset-out remains weak, the result supports a trained codec-path controller rather than another fixed handcrafted selector.\n")

    print(f"wrote {prefix}.*")


if __name__ == "__main__":
    main()
