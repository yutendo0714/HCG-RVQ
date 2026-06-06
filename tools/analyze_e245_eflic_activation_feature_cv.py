#!/usr/bin/env python3
"""Cross-validate simple EF-LIC decoder-safe activation features.

E243/E244 showed that learned activation heads can overfit the tiny Kodak24
teacher split. E245 asks a stricter question: do any individual decoder-safe
context channels separate E242 active-vs-zero regions under image-held-out
cross-validation? If not, the next controller needs richer signals or
independent fit data before codec-loop training.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import FAMILY_TO_INDEX  # noqa: E402

FEATURE_NAMES = (
    "mean_abs",
    "mean_rms",
    "scale_abs",
    "scale_rms",
    "support_rms",
    "support_over_scale",
    "prev_rms",
    "prev_over_scale",
    "slice_bit0",
    "slice_bit1",
    "slice_bit2",
)


@dataclass(frozen=True)
class Sample:
    index: int
    dataset: str
    image: str
    tensor_path: Path
    target_family: str
    teacher_policy: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context-manifest", type=Path, default=ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts_kodak24" / "manifest_kodak24_n24.csv")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e245_eflic_activation_feature_cv_kodak24")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=2.0)
    p.add_argument("--max-quantiles", type=int, default=201)
    return p.parse_args()


def read_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="") as fobj:
        for idx, row in enumerate(csv.DictReader(fobj)):
            if int(row.get("finite_context", 1)) != 1 or int(row.get("finite_alpha", 1)) != 1:
                continue
            samples.append(
                Sample(
                    index=idx,
                    dataset=row["dataset"],
                    image=row["image"],
                    tensor_path=Path(row["tensor_path"]),
                    target_family=row["target_family"],
                    teacher_policy=row["teacher_policy"],
                )
            )
    if not samples:
        raise SystemExit(f"no finite samples found in {path}")
    return samples


def load_feature_arrays(samples: list[Sample]) -> tuple[dict[int, dict[str, Any]], np.ndarray, np.ndarray]:
    metadata: dict[int, dict[str, Any]] = {}
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sample in samples:
        obj = torch.load(sample.tensor_path, map_location="cpu")
        maps = obj["context_maps"].float()
        target_map = obj["target_map"].long()
        if maps.ndim != 4 or maps.shape[0] != 4 or maps.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError(f"bad context shape for {sample.tensor_path}: {tuple(maps.shape)}")
        if target_map.shape != (maps.shape[0], maps.shape[2], maps.shape[3]):
            raise RuntimeError(f"bad target shape for {sample.tensor_path}: {tuple(target_map.shape)}")
        if not torch.isfinite(maps).all().item():
            raise RuntimeError(f"nonfinite context in {sample.tensor_path}")
        active = (target_map != FAMILY_TO_INDEX["zero"]).numpy().astype(bool).reshape(-1)
        feats = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1]).numpy()
        metadata[sample.index] = {
            "dataset": sample.dataset,
            "image": sample.image,
            "target_family": sample.target_family,
            "teacher_policy": sample.teacher_policy,
            "n": int(active.size),
            "active_frac": float(active.mean()),
        }
        features.append(feats)
        labels.append(active)
    return metadata, np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def load_by_split(samples: list[Sample]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    loaded = {}
    for sample in samples:
        obj = torch.load(sample.tensor_path, map_location="cpu")
        maps = obj["context_maps"].float()
        target_map = obj["target_map"].long()
        feats = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1]).numpy()
        active = (target_map != FAMILY_TO_INDEX["zero"]).numpy().astype(bool).reshape(-1)
        loaded[sample.index] = (feats, active)
    return loaded


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, direction: str) -> dict[str, float]:
    if direction == "ge":
        pred = scores >= threshold
    elif direction == "le":
        pred = scores <= threshold
    else:
        raise ValueError(direction)
    pos = labels.astype(bool)
    neg = ~pos
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    missed = fn / max(1, tp + fn)
    pred_active = int(pred.sum()) / max(1, pred.size)
    accuracy = (tp + tn) / max(1, pred.size)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "direction": direction,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "missed_active_rate": float(missed),
        "pred_active_frac": float(pred_active),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def auc_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool)
    pos_n = int(labels.sum())
    neg_n = int((~labels).sum())
    if pos_n == 0 or neg_n == 0:
        return {"auroc": float("nan"), "auprc": float("nan")}
    if np.unique(scores).size <= 1:
        base = pos_n / max(1, pos_n + neg_n)
        return {"auroc": 0.5, "auprc": float(base)}

    order_asc = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order_asc]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order_asc[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[labels].sum())
    auroc = (rank_sum_pos - pos_n * (pos_n + 1) / 2.0) / max(1.0, pos_n * neg_n)

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / pos_n
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    auprc = float(np.trapz(precision, recall))
    return {"auroc": float(auroc), "auprc": auprc}


def candidate_thresholds(values: np.ndarray, max_quantiles: int) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, max(2, int(max_quantiles)))
    return np.unique(np.quantile(values.astype(np.float64), qs))


def select_rules(train_x: np.ndarray, train_y: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for feature_idx, name in enumerate(FEATURE_NAMES):
        scores = train_x[:, feature_idx]
        thresholds = candidate_thresholds(scores, args.max_quantiles)
        for direction in ("ge", "le"):
            for thr in thresholds:
                row = threshold_metrics(scores, train_y, float(thr), direction)
                row.update({"feature_idx": feature_idx, "feature": name})
                row["weighted_risk"] = args.false_positive_weight * row["false_positive_rate"] + args.missed_active_weight * row["missed_active_rate"]
                candidates.append(row)
    selected: list[dict[str, Any]] = []

    def add(name: str, row: dict[str, Any]) -> None:
        out = dict(row)
        out["selector"] = name
        selected.append(out)

    add("min_weighted_risk", min(candidates, key=lambda r: (r["weighted_risk"], -r["f1"])))
    add("best_f1", max(candidates, key=lambda r: (r["f1"], r["recall"], -r["false_positive_rate"])))
    for cap in (0.01, 0.05, 0.10, 0.20):
        feasible = [r for r in candidates if r["false_positive_rate"] <= cap]
        if feasible:
            chosen = max(feasible, key=lambda r: (r["recall"], r["precision"], -r["weighted_risk"]))
        else:
            chosen = min(candidates, key=lambda r: (r["false_positive_rate"], -r["recall"]))
        add(f"fpr_le_{cap:.2f}", chosen)
    top_train = sorted(candidates, key=lambda r: (r["weighted_risk"], -r["f1"]))[:20]
    return selected, top_train


def apply_rule(x: np.ndarray, y: np.ndarray, rule: dict[str, Any]) -> dict[str, Any]:
    scores = x[:, int(rule["feature_idx"])]
    row = threshold_metrics(scores, y, float(rule["threshold"]), str(rule["direction"]))
    row.update(
        {
            "feature_idx": int(rule["feature_idx"]),
            "feature": str(rule["feature"]),
            "direction": str(rule["direction"]),
            "threshold": float(rule["threshold"]),
            "selector": str(rule["selector"]),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    samples = read_samples(args.context_manifest)
    loaded = load_by_split(samples)
    fold_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []

    for fold in range(args.folds):
        train_ids = [s.index for s in samples if s.index % args.folds != fold]
        val_ids = [s.index for s in samples if s.index % args.folds == fold]
        train_x = np.concatenate([loaded[idx][0] for idx in train_ids], axis=0)
        train_y = np.concatenate([loaded[idx][1] for idx in train_ids], axis=0)
        val_x = np.concatenate([loaded[idx][0] for idx in val_ids], axis=0)
        val_y = np.concatenate([loaded[idx][1] for idx in val_ids], axis=0)
        selected, top_train = select_rules(train_x, train_y, args)
        for rank, row in enumerate(top_train, start=1):
            out = dict(row)
            out.update({"fold": fold, "rank": rank, "split": "train"})
            top_rows.append(out)
        for rule in selected:
            train_row = apply_rule(train_x, train_y, rule)
            val_row = apply_rule(val_x, val_y, rule)
            train_row.update({"fold": fold, "split": "train"})
            val_row.update({"fold": fold, "split": "val"})
            fold_rows.extend([train_row, val_row])
        for feature_idx, name in enumerate(FEATURE_NAMES):
            train_auc = auc_metrics(train_x[:, feature_idx], train_y)
            val_auc = auc_metrics(val_x[:, feature_idx], val_y)
            auc_rows.append(
                {
                    "fold": fold,
                    "feature_idx": feature_idx,
                    "feature": name,
                    "train_auroc": train_auc["auroc"],
                    "train_auprc": train_auc["auprc"],
                    "val_auroc": val_auc["auroc"],
                    "val_auprc": val_auc["auprc"],
                }
            )

    summary_rows: list[dict[str, Any]] = []
    numeric = ["precision", "recall", "f1", "false_positive_rate", "missed_active_rate", "pred_active_frac", "accuracy"]
    for selector in sorted({r["selector"] for r in fold_rows}):
        for split in ("train", "val"):
            subset = [r for r in fold_rows if r["selector"] == selector and r["split"] == split]
            out: dict[str, Any] = {"selector": selector, "split": split, "n_folds": len(subset)}
            features = [str(r["feature"]) + ":" + str(r["direction"]) for r in subset]
            out["selected_rules"] = "; ".join(features)
            for field in numeric:
                vals = [float(r[field]) for r in subset]
                out[f"{field}_mean"] = mean(vals)
                out[f"{field}_min"] = min(vals)
                out[f"{field}_max"] = max(vals)
            summary_rows.append(out)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".folds.csv").open("w", newline="") as fobj:
        fields = sorted({key for row in fold_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fold_rows)
    with args.output_prefix.with_suffix(".summary.csv").open("w", newline="") as fobj:
        fields = sorted({key for row in summary_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.output_prefix.with_suffix(".top_train.csv").open("w", newline="") as fobj:
        fields = sorted({key for row in top_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(top_rows)
    with args.output_prefix.with_suffix(".auc.csv").open("w", newline="") as fobj:
        fields = sorted({key for row in auc_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(auc_rows)

    payload = {
        "experiment": "E245 EF-LIC decoder-safe activation feature CV",
        "context_manifest": str(args.context_manifest),
        "fold_rows": fold_rows,
        "summary_rows": summary_rows,
        "auc_rows": auc_rows,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    val_summary = [r for r in summary_rows if r["split"] == "val"]
    best_val = max(val_summary, key=lambda r: (float(r["f1_mean"]), -float(r["false_positive_rate_mean"])))
    min_risk = next(r for r in val_summary if r["selector"] == "min_weighted_risk")
    with args.output_prefix.with_suffix(".md").open("w") as fobj:
        fobj.write("# E245 EF-LIC Activation Feature CV\n\n")
        fobj.write("This tests whether single decoder-safe EF-LIC context channels can separate E242 active-vs-zero targets under image-held-out cross-validation. It is a signal audit, not codec R-D evidence.\n\n")
        fobj.write(f"- Context manifest: `{args.context_manifest}`\n")
        fobj.write(f"- Folds: `{args.folds}`\n")
        fobj.write(f"- Best validation selector by mean F1: `{best_val['selector']}` F1 `{float(best_val['f1_mean']):.6f}` FPR `{float(best_val['false_positive_rate_mean']):.6f}` recall `{float(best_val['recall_mean']):.6f}`\n")
        fobj.write(f"- Validation min-weighted-risk: F1 `{float(min_risk['f1_mean']):.6f}` FPR `{float(min_risk['false_positive_rate_mean']):.6f}` recall `{float(min_risk['recall_mean']):.6f}` rules `{min_risk['selected_rules']}`\n\n")
        fobj.write("| selector | split | precision | recall | fpr | missed | pred active | f1 | rules |\n")
        fobj.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            fobj.write(
                f"| {row['selector']} | {row['split']} | {float(row['precision_mean']):.6f} | {float(row['recall_mean']):.6f} | "
                f"{float(row['false_positive_rate_mean']):.6f} | {float(row['missed_active_rate_mean']):.6f} | "
                f"{float(row['pred_active_frac_mean']):.6f} | {float(row['f1_mean']):.6f} | {row['selected_rules']} |\n"
            )
        fobj.write("\nDecision:\n\n")
        fobj.write("If these single-feature selectors do not beat the learned-head audits on held-out folds, the bottleneck is not just head capacity. EF-LIC insertion should then move toward richer decoder-safe features or independent fit data rather than larger heads on Kodak24 teachers.\n")

    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
