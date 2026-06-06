#!/usr/bin/env python3
"""E246 decoder-safe feature-group audit for EF-LIC HCG activation.

E244/E245 showed that local mean/scale/support maps alone do not provide a
reliable activation controller. This diagnostic joins the E242 spatial-teacher
manifests with E233 decoder-safe image/slice summaries and asks which feature
groups are worth promoting before full EF-LIC training.

The script intentionally stays at the audit level: teacher alpha maps and E236
outcome deltas are labels/targets only, not model inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FEATURE_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e233_eflic_kodak24_decoder_safe_branch_features.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e233_eflic_clicpro41_decoder_safe_branch_features.csv",
}

MANIFEST_INPUTS = {
    "kodak24": ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts_kodak24" / "manifest_kodak24_n24.csv",
    "clicpro41": ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts_clicpro41" / "manifest_clicpro41_n41.csv",
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
        default=ROOT / "experiments" / "analysis" / "e246_eflic_decoder_safe_feature_groups",
    )
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--fp-weight", type=float, default=4.0)
    p.add_argument("--fn-weight", type=float, default=2.0)
    p.add_argument("--max-md-rows", type=int, default=80)
    return p.parse_args()


def to_float(value: object, default: float = math.nan) -> float:
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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_feature_rows() -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    cols: set[str] = set()
    for dataset, path in FEATURE_INPUTS.items():
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


def load_manifest_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for dataset, path in MANIFEST_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            for row in csv.DictReader(fobj):
                rows[(dataset, row["image"])] = row
    return rows


def manifest_numeric_cols(rows: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    blocked = {
        "active_frac",
        "alpha_max",
        "alpha_mean",
        "alpha_shape",
        "confident_nonzero",
        "dataset",
        "finite_alpha",
        "finite_context",
        "image",
        "nonfinite_alpha",
        "nonfinite_context",
        "pixel_family_counts",
        "sample_weight",
        "target_family",
        "target_index",
        "target_map_shape",
        "teacher_policy",
        "tensor_path",
    }
    cols: set[str] = set()
    for row in rows.values():
        for col, value in row.items():
            if col in blocked:
                continue
            if math.isfinite(to_float(value)):
                cols.add(col)
    return sorted(cols)


def feature_groups(feature_cols: list[str], manifest_cols: list[str]) -> dict[str, list[tuple[str, str]]]:
    def e233(names: Iterable[str]) -> list[tuple[str, str]]:
        return [("e233", name) for name in names]

    def manifest(names: Iterable[str]) -> list[tuple[str, str]]:
        return [("manifest", name) for name in names]

    z_cols = [c for c in feature_cols if c.startswith("z_")]
    local_cols = [c for c in feature_cols if c.startswith("slice")]
    mean_scale_cols = [
        c
        for c in local_cols
        if "_mean_" in c or "_scale_" in c or c.endswith("_mean_abs_mean") or c.endswith("_scale_rms_mean")
    ]
    support_cols = [c for c in local_cols if "_support_" in c or "_prev_" in c]
    shape_cols = [c for c in manifest_cols if c in {"height", "width", "latent_height", "latent_width", "context_abs_mean", "context_rms"}]

    return {
        "manifest_context_shape": manifest(shape_cols),
        "e233_z_prior": e233(z_cols),
        "e233_mean_scale": e233(mean_scale_cols),
        "e233_support_state": e233(support_cols),
        "e233_local_all": e233(local_cols),
        "e233_z_plus_local": e233(z_cols + local_cols),
        "manifest_plus_e233_all": manifest(shape_cols) + e233(z_cols + local_cols),
    }


def value_for_item(
    item: tuple[str, str],
    spec: tuple[str, str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    manifest_rows: dict[tuple[str, str], dict[str, str]],
) -> float:
    source, col = spec
    row = feature_rows[item] if source == "e233" else manifest_rows[item]
    value = to_float(row.get(col), 0.0)
    return value if math.isfinite(value) else 0.0


def matrix(
    items: list[tuple[str, str]],
    specs: list[tuple[str, str]],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    manifest_rows: dict[tuple[str, str], dict[str, str]],
) -> np.ndarray:
    x = np.zeros((len(items), len(specs)), dtype=np.float64)
    for i, item in enumerate(items):
        for j, spec in enumerate(specs):
            x[i, j] = value_for_item(item, spec, feature_rows, manifest_rows)
    return x


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[1] == 0:
        return x_train, x_test
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def fit_ridge_binary(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    if x_train.shape[1] == 0:
        return np.full((x_test.shape[0],), float(y_train.mean()), dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train, x_test)
    ones_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
    ones_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
    xt = np.concatenate([ones_train, x_train], axis=1)
    xv = np.concatenate([ones_test, x_test], axis=1)
    reg = np.eye(xt.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(xt.T @ xt + reg) @ xt.T @ y_train
    return xv @ coef


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    greater = 0.0
    total = 0
    for p in pos:
        greater += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return greater / total if total else 0.5


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = 0
    fp = 0
    area = 0.0
    prev_recall = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return area


def threshold_candidates(scores: np.ndarray) -> list[float]:
    vals = sorted({float(v) for v in scores if math.isfinite(float(v))})
    if not vals:
        return [0.0]
    mids = [(a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-9] + mids + [vals[-1] + 1e-9]


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, fp_weight: float, fn_weight: float) -> dict[str, float]:
    labels = labels.astype(np.int64)
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum((~pred) & (labels == 1)))
    tn = int(np.sum((~pred) & (labels == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    risk = (fp_weight * fp + fn_weight * fn) / max(1, len(labels))
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "weighted_risk": risk,
        "active_pred_frac": float(np.mean(pred)) if len(pred) else 0.0,
    }


def choose_threshold(scores: np.ndarray, labels: np.ndarray, mode: str, fp_weight: float, fn_weight: float) -> float:
    rows = [threshold_metrics(scores, labels, t, fp_weight, fn_weight) for t in threshold_candidates(scores)]
    if mode == "best_f1":
        return float(max(rows, key=lambda r: (r["f1"], -r["weighted_risk"]))["threshold"])
    if mode == "min_weighted_risk":
        return float(min(rows, key=lambda r: (r["weighted_risk"], -r["f1"]))["threshold"])
    if mode == "fpr_le_010":
        ok = [r for r in rows if r["fpr"] <= 0.10]
        if ok:
            return float(max(ok, key=lambda r: (r["recall"], r["f1"]))["threshold"])
        return float(min(rows, key=lambda r: (r["fpr"], -r["recall"]))["threshold"])
    raise ValueError(mode)


def split_defs(items: list[tuple[str, str]]) -> list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]]:
    kodak = [item for item in items if item[0] == "kodak24"]
    clic = [item for item in items if item[0] == "clicpro41"]
    splits: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = [
        ("pooled_resub", items, items),
        ("train_clicpro41_test_kodak24", clic, kodak),
        ("train_kodak24_test_clicpro41", kodak, clic),
    ]
    for heldout in items:
        train = [item for item in items if item != heldout]
        splits.append((f"loio__{heldout[0]}__{heldout[1]}", train, [heldout]))
    return splits


def evaluate_active(
    items: list[tuple[str, str]],
    specs: list[tuple[str, str]],
    labels: dict[tuple[str, str], int],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    manifest_rows: dict[tuple[str, str], dict[str, str]],
    ridge_alpha: float,
    fp_weight: float,
    fn_weight: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pred_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    modes = ["best_f1", "fpr_le_010", "min_weighted_risk"]
    for split, train_items, test_items in split_defs(items):
        if not train_items or not test_items:
            continue
        x_train = matrix(train_items, specs, feature_rows, manifest_rows)
        x_test = matrix(test_items, specs, feature_rows, manifest_rows)
        y_train = np.asarray([labels[item] for item in train_items], dtype=np.float64)
        y_test = np.asarray([labels[item] for item in test_items], dtype=np.int64)
        scores_train = fit_ridge_binary(x_train, y_train, x_train, ridge_alpha)
        scores_test = fit_ridge_binary(x_train, y_train, x_test, ridge_alpha)
        for mode in modes:
            threshold = choose_threshold(scores_train, y_train.astype(np.int64), mode, fp_weight, fn_weight)
            metrics = threshold_metrics(scores_test, y_test, threshold, fp_weight, fn_weight)
            summary_rows.append(
                {
                    "split": split,
                    "threshold_mode": mode,
                    "test_images": len(test_items),
                    "active_frac": float(np.mean(y_test)) if len(y_test) else 0.0,
                    "auroc": auroc(scores_test, y_test),
                    "auprc": auprc(scores_test, y_test),
                    **metrics,
                }
            )
            for item, score, label in zip(test_items, scores_test, y_test):
                pred_rows.append(
                    {
                        "split": split,
                        "threshold_mode": mode,
                        "dataset": item[0],
                        "image": item[1],
                        "score": float(score),
                        "threshold": threshold,
                        "label_active": int(label),
                        "pred_active": int(score >= threshold),
                        "active_frac": to_float(manifest_rows[item].get("active_frac"), 0.0),
                        "target_family": manifest_rows[item].get("target_family", ""),
                        "teacher_policy": manifest_rows[item].get("teacher_policy", ""),
                    }
                )
    return summary_rows, pred_rows


def summarize_loio_predictions(rows: list[dict[str, object]], fp_weight: float, fn_weight: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        split = str(row["split"])
        if not split.startswith("loio__"):
            continue
        buckets.setdefault((str(row.get("feature_group", "")), str(row["threshold_mode"])), []).append(row)
    for (feature_group, mode), bucket in sorted(buckets.items()):
        labels = np.asarray([int(r["label_active"]) for r in bucket], dtype=np.int64)
        preds = np.asarray([int(r["pred_active"]) for r in bucket], dtype=bool)
        scores = np.asarray([float(r["score"]) for r in bucket], dtype=np.float64)
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        fn = int(np.sum((~preds) & (labels == 1)))
        tn = int(np.sum((~preds) & (labels == 0)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out.append(
            {
                "split": "pooled_leave_one_image_out",
                "feature_group": feature_group,
                "threshold_mode": mode,
                "test_images": len(bucket),
                "active_frac": float(np.mean(labels)) if len(labels) else 0.0,
                "auroc": auroc(scores, labels),
                "auprc": auprc(scores, labels),
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "f1": f1,
                "weighted_risk": (fp_weight * fp + fn_weight * fn) / max(1, len(labels)),
                "active_pred_frac": float(np.mean(preds)) if len(preds) else 0.0,
                "threshold": float("nan"),
            }
        )
    return out


def family_nearest_centroid(
    train_items: list[tuple[str, str]],
    test_items: list[tuple[str, str]],
    specs: list[tuple[str, str]],
    family_labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    manifest_rows: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    x_train = matrix(train_items, specs, feature_rows, manifest_rows)
    x_test = matrix(test_items, specs, feature_rows, manifest_rows)
    x_train, x_test = standardize_train_test(x_train, x_test)
    y_train = np.asarray([family_labels[item] for item in train_items], dtype=object)
    families = sorted(set(y_train))
    centroids = {
        fam: x_train[y_train == fam].mean(axis=0) if x_train.shape[1] else np.zeros((0,), dtype=np.float64)
        for fam in families
    }
    out: list[dict[str, object]] = []
    for item, vec in zip(test_items, x_test):
        pred = max(Counter(y_train).items(), key=lambda kv: kv[1])[0]
        if x_train.shape[1]:
            pred = min(families, key=lambda fam: float(np.sum((vec - centroids[fam]) ** 2)))
        out.append(
            {
                "dataset": item[0],
                "image": item[1],
                "pred_family": pred,
                "target_family": family_labels[item],
                "correct": int(pred == family_labels[item]),
            }
        )
    return out


def evaluate_family(
    items: list[tuple[str, str]],
    specs: list[tuple[str, str]],
    family_labels: dict[tuple[str, str], str],
    feature_rows: dict[tuple[str, str], dict[str, str]],
    manifest_rows: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for split, train_items, test_items in split_defs(items):
        if not train_items or not test_items:
            continue
        preds = family_nearest_centroid(train_items, test_items, specs, family_labels, feature_rows, manifest_rows)
        for row in preds:
            row["split"] = split
            pred_rows.append(row)
        summary_rows.append(
            {
                "split": split,
                "test_images": len(test_items),
                "family_accuracy": mean(int(r["correct"]) for r in preds),
                "active_family_accuracy": mean(
                    int(r["correct"]) for r in preds if str(r["target_family"]) != "zero"
                ),
                "target_family_counts": json.dumps(dict(Counter(str(r["target_family"]) for r in preds)), sort_keys=True),
                "pred_family_counts": json.dumps(dict(Counter(str(r["pred_family"]) for r in preds)), sort_keys=True),
            }
        )
    return summary_rows, pred_rows


def main() -> None:
    args = parse_args()
    feature_rows, feature_cols = load_feature_rows()
    manifest_rows = load_manifest_rows()
    manifest_cols = manifest_numeric_cols(manifest_rows)
    items = sorted(item for item in manifest_rows if item in feature_rows)
    if not items:
        raise SystemExit("no overlapping E242 manifest rows and E233 feature rows")

    active_labels = {
        item: int(to_float(manifest_rows[item].get("active_frac"), 0.0) > 1e-9)
        for item in items
    }
    family_labels = {item: str(manifest_rows[item].get("target_family", "zero")) for item in items}
    groups = feature_groups(feature_cols, manifest_cols)

    active_summary: list[dict[str, object]] = []
    active_preds: list[dict[str, object]] = []
    family_summary: list[dict[str, object]] = []
    family_preds: list[dict[str, object]] = []

    for name, specs in groups.items():
        group_active, group_preds = evaluate_active(
            items,
            specs,
            active_labels,
            feature_rows,
            manifest_rows,
            args.ridge_alpha,
            args.fp_weight,
            args.fn_weight,
        )
        for row in group_active:
            row["feature_group"] = name
            row["feature_count"] = len(specs)
            active_summary.append(row)
        for row in group_preds:
            row["feature_group"] = name
            active_preds.append(row)

        fam_summary, fam_preds = evaluate_family(items, specs, family_labels, feature_rows, manifest_rows)
        for row in fam_summary:
            row["feature_group"] = name
            row["feature_count"] = len(specs)
            family_summary.append(row)
        for row in fam_preds:
            row["feature_group"] = name
            family_preds.append(row)

    active_summary.extend(summarize_loio_predictions(active_preds, args.fp_weight, args.fn_weight))

    key_rows = [
        row
        for row in active_summary
        if row["split"] in {"pooled_resub", "pooled_leave_one_image_out", "train_clicpro41_test_kodak24", "train_kodak24_test_clicpro41"}
        and row["threshold_mode"] in {"fpr_le_010", "min_weighted_risk", "best_f1"}
    ]
    key_rows.sort(
        key=lambda r: (
            str(r["split"]),
            str(r["threshold_mode"]),
            float(r.get("weighted_risk", 0.0)),
            -float(r.get("f1", 0.0)),
        )
    )

    family_key_rows = [
        row
        for row in family_summary
        if row["split"] in {"pooled_resub", "pooled_leave_one_image_out", "train_clicpro41_test_kodak24", "train_kodak24_test_clicpro41"}
    ]
    loio_family: list[dict[str, object]] = []
    for name in groups:
        rows = [r for r in family_summary if r["feature_group"] == name and str(r["split"]).startswith("loio__")]
        if rows:
            loio_family.append(
                {
                    "split": "pooled_leave_one_image_out",
                    "feature_group": name,
                    "feature_count": next((r["feature_count"] for r in rows), 0),
                    "test_images": int(sum(int(r["test_images"]) for r in rows)),
                    "family_accuracy": mean(float(r["family_accuracy"]) for r in rows),
                    "active_family_accuracy": mean(float(r["active_family_accuracy"]) for r in rows),
                    "target_family_counts": "",
                    "pred_family_counts": "",
                }
            )
    family_key_rows.extend(loio_family)
    family_key_rows.sort(key=lambda r: (str(r["split"]), -float(r.get("family_accuracy", 0.0)), str(r["feature_group"])))

    out = {
        "experiment": "E246 EF-LIC decoder-safe feature-group audit",
        "items": len(items),
        "dataset_counts": dict(Counter(dataset for dataset, _ in items)),
        "active_count": int(sum(active_labels.values())),
        "active_fraction": mean(active_labels.values()),
        "family_counts": dict(Counter(family_labels.values())),
        "ridge_alpha": args.ridge_alpha,
        "fp_weight": args.fp_weight,
        "fn_weight": args.fn_weight,
        "feature_groups": {name: len(specs) for name, specs in groups.items()},
        "key_active_rows": key_rows[: args.max_md_rows],
        "key_family_rows": family_key_rows[: args.max_md_rows],
        "decision": (
            "Use this audit to decide whether z-prior/index summaries and richer "
            "decoder-safe state are strong enough to promote before full EF-LIC training. "
            "Teacher alpha and outcome deltas remain labels only."
        ),
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".json")).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_suffix(".active_summary.csv"), active_summary)
    write_csv(prefix.with_suffix(".active_predictions.csv"), active_preds)
    write_csv(prefix.with_suffix(".family_summary.csv"), family_summary + loio_family)
    write_csv(prefix.with_suffix(".family_predictions.csv"), family_preds)

    md: list[str] = []
    md.append("# E246 EF-LIC Decoder-Safe Feature-Group Audit\n")
    md.append(
        "This audit joins E242 spatial-teacher manifests with E233 decoder-safe "
        "summaries. It checks whether adding z-prior/index and richer local state "
        "is justified before full EF-LIC training. Teacher alpha maps are labels, "
        "not inputs.\n"
    )
    md.append(f"Images: `{len(items)}`; active fraction: `{out['active_fraction']:.6f}`.\n")
    md.append("## Feature Groups\n")
    md.append("| group | features |\n|---|---:|\n")
    for name, count in out["feature_groups"].items():
        md.append(f"| {name} | {count} |\n")
    md.append("\n## Key Activation Rows\n")
    md.append(
        "| split | threshold | group | n | active | AUROC | AUPRC | precision | recall | FPR | F1 | risk | pred active |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for row in key_rows[: args.max_md_rows]:
        md.append(
            f"| {row['split']} | {row['threshold_mode']} | {row['feature_group']} | {int(row['test_images'])} | "
            f"{float(row['active_frac']):.3f} | {float(row['auroc']):+.3f} | {float(row['auprc']):+.3f} | "
            f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['fpr']):.3f} | "
            f"{float(row['f1']):.3f} | {float(row['weighted_risk']):.3f} | {float(row['active_pred_frac']):.3f} |\n"
        )
    md.append("\n## Key Family Rows\n")
    md.append("| split | group | n | family acc | active-family acc | predicted families |\n|---|---|---:|---:|---:|---|\n")
    for row in family_key_rows[: args.max_md_rows]:
        md.append(
            f"| {row['split']} | {row['feature_group']} | {int(row['test_images'])} | "
            f"{float(row['family_accuracy']):.3f} | {float(row['active_family_accuracy']):.3f} | "
            f"{row.get('pred_family_counts', '')} |\n"
        )
    md.append(
        "\n## Interpretation\n\n"
        "- This is a promotion gate, not a codec row: it does not alter EF-LIC and does not claim final RD performance.\n"
        "- If z/index-rich groups improve held-out activation without high FPR collapse, they should be added to the next local controller export.\n"
        "- If all feature groups still fail leave-one-image-out or cross-dataset transfer, the next full-training run should keep the original EF-LIC loss dominant and use activation only as weak initialization/regularization.\n"
    )
    (prefix.with_suffix(".md")).write_text("".join(md))


if __name__ == "__main__":
    main()
