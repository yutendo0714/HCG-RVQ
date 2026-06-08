#!/usr/bin/env python3
"""LOOCV learned EF-LIC HCG slice policy from E317/E318 teacher labels.

E319/E320 showed that one-feature threshold rules do not recover the
powerset-oracle headroom. This diagnostic tests the next-smallest step: a
regularized logistic slice gate trained only on the leave-one-image-out training
fold, followed by a threshold selected on train-fold RD outcomes.

This is still an offline controller audit, not final paper evidence. It uses
the E317 powerset rows as a lookup table to score predicted slice subsets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ALL_SLICES = (0, 1, 2, 3)
OUTCOME_COLUMNS = {
    "all_delta_psnr",
    "best_delta_psnr",
    "best_gain_over_all",
    "best_slice_set",
    "contextual_margin_psnr",
    "contextual_positive",
    "leave_one_out_delta_psnr",
    "oracle_active",
    "oracle_vs_context_agree",
    "oracle_vs_single_agree",
    "single_delta_psnr",
    "single_positive",
    "single_vs_context_agree",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.rows.csv",
    )
    p.add_argument(
        "--slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e321_eflic_slice_policy_logistic_cv_kodak24",
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--objective-tail-weight", type=float, default=0.25)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def label_for_slices(slices: set[int]) -> str:
    if not slices:
        return "none"
    if slices == set(ALL_SLICES):
        return "all"
    return ",".join(str(s) for s in sorted(slices))


def build_delta_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        lookup[(row["image"], row["active_slices"])] = fnum(row, "delta_psnr")
    return lookup


def candidate_features(label_rows: list[dict[str, str]]) -> list[str]:
    if not label_rows:
        return []
    out: list[str] = []
    for key in sorted(label_rows[0]):
        if key in {"image", "slice"} or key in OUTCOME_COLUMNS:
            continue
        if not (key.startswith("allctx_") or key.startswith("single_")):
            continue
        vals = [fnum(row, key) for row in label_rows]
        finite = [v for v in vals if math.isfinite(v)]
        if len(finite) >= 8 and float(np.std(finite)) > 0.0:
            out.append(key)
    # The slice id is decoder-available and important for EF-LIC's sequential
    # support-buffer dynamics, so include it as explicit low-dimensional cues.
    out.extend(["slice_norm", "slice_is_0", "slice_is_1", "slice_is_2", "slice_is_3"])
    return out


def feature_value(row: dict[str, str], feature: str) -> float:
    slice_id = int(row["slice"])
    if feature == "slice_norm":
        return float(slice_id) / 3.0
    if feature.startswith("slice_is_"):
        return 1.0 if slice_id == int(feature.rsplit("_", 1)[1]) else 0.0
    return fnum(row, feature)


def matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[feature_value(row, f) for f in features] for row in rows], dtype=np.float32)


def standardize(
    x_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    train_filled = np.where(np.isfinite(x_train), x_train, mu)
    eval_filled = np.where(np.isfinite(x_eval), x_eval, mu)
    sigma = train_filled.std(axis=0)
    sigma = np.where(sigma > 1e-6, sigma, 1.0)
    return (
        torch.from_numpy((train_filled - mu) / sigma).float(),
        torch.from_numpy((eval_filled - mu) / sigma).float(),
        mu,
        sigma,
    )


def train_logistic(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    lr: float,
    steps: int,
    l2: float,
    false_positive_weight: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    weight = torch.zeros((x_train.shape[1], 1), dtype=torch.float32, requires_grad=True)
    bias_prior = float(torch.logit(y_train.mean().clamp(0.05, 0.95)))
    bias = torch.tensor([bias_prior], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    sample_weight = torch.where(y_train > 0.5, torch.ones_like(y_train), torch.full_like(y_train, false_positive_weight))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = (x_train @ weight).squeeze(1) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train, weight=sample_weight)
        loss = loss + float(l2) * weight.square().mean()
        loss.backward()
        opt.step()
    return weight.detach(), bias.detach()


def predict_probs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid((x @ weight).squeeze(1) + bias).cpu().numpy()


def rows_by_image(label_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        out.setdefault(row["image"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["slice"]))
    return out


def label_from_probs(image_rows: list[dict[str, str]], probs: np.ndarray, threshold: float) -> str:
    active = {int(row["slice"]) for row, prob in zip(image_rows, probs, strict=True) if float(prob) >= threshold}
    return label_for_slices(active)


def score_predictions(
    images: list[str],
    probs_by_image: dict[str, np.ndarray],
    labels_by_image: dict[str, list[dict[str, str]]],
    deltas: dict[tuple[str, str], float],
    threshold: float,
) -> dict[str, float]:
    vals: list[float] = []
    gains: list[float] = []
    active_fracs: list[float] = []
    for image in images:
        label = label_from_probs(labels_by_image[image], probs_by_image[image], threshold)
        delta = deltas[(image, label)]
        vals.append(delta)
        gains.append(delta - deltas[(image, "all")])
        active_fracs.append(0.0 if label == "none" else 1.0 if label == "all" else len(label.split(",")) / 4.0)
    return {
        "mean_delta_psnr": mean(vals),
        "worst_delta_psnr": min(vals),
        "mean_gain_over_all": mean(gains),
        "mean_active_frac": mean(active_fracs),
    }


def choose_threshold(
    train_images: list[str],
    probs_by_image: dict[str, np.ndarray],
    labels_by_image: dict[str, list[dict[str, str]]],
    deltas: dict[tuple[str, str], float],
    *,
    objective_tail_weight: float,
) -> dict[str, Any]:
    raw = np.concatenate([probs_by_image[image] for image in train_images])
    thresholds = sorted({0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99})
    thresholds.extend(float(v) for v in np.quantile(raw, np.linspace(0.05, 0.95, 19)))
    best: dict[str, Any] | None = None
    for threshold in sorted(set(thresholds)):
        scored = score_predictions(train_images, probs_by_image, labels_by_image, deltas, threshold)
        row: dict[str, Any] = {"threshold": float(threshold), **scored}
        objective = row["mean_delta_psnr"] + float(objective_tail_weight) * row["worst_delta_psnr"]
        row["objective"] = objective
        key = (objective, row["mean_delta_psnr"], row["worst_delta_psnr"])
        if best is None or key > (best["objective"], best["mean_delta_psnr"], best["worst_delta_psnr"]):
            best = row
    if best is None:
        raise RuntimeError("no threshold selected")
    return best


def train_and_select(
    train_rows: list[dict[str, str]],
    train_images: list[str],
    all_rows_by_image: dict[str, list[dict[str, str]]],
    features: list[str],
    deltas: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    y = torch.tensor([fnum(row, "oracle_active") for row in train_rows], dtype=torch.float32)
    x_train_np = matrix(train_rows, features)
    x_all_np = matrix([row for image in train_images for row in all_rows_by_image[image]], features)
    x_train, x_all, _, _ = standardize(x_train_np, x_all_np)

    best: dict[str, Any] | None = None
    # Small grid: tune calibration pressure on train-fold RD, not held-out RD.
    for l2 in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        for fp_weight in (1.0, 2.0, 4.0, 8.0):
            weight, bias = train_logistic(
                x_train,
                y,
                lr=args.lr,
                steps=args.steps,
                l2=l2,
                false_positive_weight=fp_weight,
                seed=args.seed,
            )
            probs_all = predict_probs(x_all, weight, bias)
            probs_by_image: dict[str, np.ndarray] = {}
            offset = 0
            for image in train_images:
                n = len(all_rows_by_image[image])
                probs_by_image[image] = probs_all[offset : offset + n]
                offset += n
            threshold = choose_threshold(
                train_images,
                probs_by_image,
                all_rows_by_image,
                deltas,
                objective_tail_weight=args.objective_tail_weight,
            )
            row: dict[str, Any] = {
                "l2": l2,
                "false_positive_weight": fp_weight,
                "threshold": threshold["threshold"],
                "train_mean_delta_psnr": threshold["mean_delta_psnr"],
                "train_worst_delta_psnr": threshold["worst_delta_psnr"],
                "train_mean_gain_over_all": threshold["mean_gain_over_all"],
                "train_mean_active_frac": threshold["mean_active_frac"],
                "train_objective": threshold["objective"],
                "weight": weight,
                "bias": bias,
            }
            key = (row["train_objective"], row["train_mean_delta_psnr"], row["train_worst_delta_psnr"])
            if best is None or key > (best["train_objective"], best["train_mean_delta_psnr"], best["train_worst_delta_psnr"]):
                best = row
    if best is None:
        raise RuntimeError("no logistic policy selected")
    return best


def main() -> None:
    args = parse_args()
    sweep_rows = read_csv(args.rows)
    label_rows = read_csv(args.slice_labels)
    deltas = build_delta_lookup(sweep_rows)
    labels_by_image = rows_by_image(label_rows)
    images = sorted(labels_by_image)
    features = candidate_features(label_rows)

    folds: list[dict[str, Any]] = []
    prob_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for fold_idx, heldout in enumerate(images):
        train_images = [image for image in images if image != heldout]
        train_rows = [row for image in train_images for row in labels_by_image[image]]
        policy = train_and_select(train_rows, train_images, labels_by_image, features, deltas, args)

        x_train_np = matrix(train_rows, features)
        x_eval_np = matrix(labels_by_image[heldout], features)
        _, x_eval, mu, sigma = standardize(x_train_np, x_eval_np)
        probs = predict_probs(x_eval, policy["weight"], policy["bias"])
        predicted_label = label_from_probs(labels_by_image[heldout], probs, policy["threshold"])
        predicted_delta = deltas[(heldout, predicted_label)]
        oracle_label, oracle_delta = max(
            ((label, delta) for (image, label), delta in deltas.items() if image == heldout),
            key=lambda item: item[1],
        )
        best_all_none = max(deltas[(heldout, "all")], deltas[(heldout, "none")])
        folds.append(
            {
                "image": heldout,
                "predicted_slice_set": predicted_label,
                "predicted_delta_psnr": predicted_delta,
                "all_delta_psnr": deltas[(heldout, "all")],
                "none_delta_psnr": deltas[(heldout, "none")],
                "best_all_none_delta_psnr": best_all_none,
                "full_oracle_slice_set": oracle_label,
                "full_oracle_delta_psnr": oracle_delta,
                "gain_over_all": predicted_delta - deltas[(heldout, "all")],
                "gap_to_best_all_none": best_all_none - predicted_delta,
                "gap_to_full_oracle": oracle_delta - predicted_delta,
                "threshold": policy["threshold"],
                "l2": policy["l2"],
                "false_positive_weight": policy["false_positive_weight"],
                "train_mean_delta_psnr": policy["train_mean_delta_psnr"],
                "train_worst_delta_psnr": policy["train_worst_delta_psnr"],
                "train_mean_active_frac": policy["train_mean_active_frac"],
            }
        )
        for row, prob in zip(labels_by_image[heldout], probs, strict=True):
            prob_rows.append(
                {
                    "image": heldout,
                    "slice": row["slice"],
                    "prob_active": float(prob),
                    "threshold": policy["threshold"],
                    "predicted_active": int(float(prob) >= policy["threshold"]),
                    "oracle_active": int(fnum(row, "oracle_active")),
                }
            )
        weights_np = policy["weight"].squeeze(1).cpu().numpy()
        for feature, w, m, s in zip(features, weights_np, mu, sigma, strict=True):
            coef_rows.append(
                {
                    "fold": fold_idx,
                    "heldout": heldout,
                    "feature": feature,
                    "coef_standardized": float(w),
                    "train_mean": float(m),
                    "train_std": float(s),
                }
            )

    full_oracle = {
        image: max(delta for (img, _), delta in deltas.items() if img == image)
        for image in images
    }
    fixed_labels = sorted({label for _, label in deltas})
    fixed_scores = []
    for label in fixed_labels:
        vals = [deltas[(image, label)] for image in images]
        fixed_scores.append((mean(vals), min(vals), label))
    best_fixed_mean, best_fixed_worst, best_fixed_label = max(fixed_scores)
    summary = [
        {
            "policy": "all",
            "images": len(images),
            "mean_delta_psnr": mean([deltas[(image, "all")] for image in images]),
            "worst_delta_psnr": min(deltas[(image, "all")] for image in images),
            "mean_gain_over_all": 0.0,
            "mean_gap_to_full_oracle": mean([full_oracle[image] - deltas[(image, "all")] for image in images]),
        },
        {
            "policy": "best_fixed_subset",
            "slice_set": best_fixed_label,
            "images": len(images),
            "mean_delta_psnr": best_fixed_mean,
            "worst_delta_psnr": best_fixed_worst,
            "mean_gain_over_all": mean([deltas[(image, best_fixed_label)] - deltas[(image, "all")] for image in images]),
            "mean_gap_to_full_oracle": mean([full_oracle[image] - deltas[(image, best_fixed_label)] for image in images]),
        },
        {
            "policy": "best_all_none_oracle",
            "images": len(images),
            "mean_delta_psnr": mean([max(deltas[(image, "all")], deltas[(image, "none")]) for image in images]),
            "worst_delta_psnr": min(max(deltas[(image, "all")], deltas[(image, "none")]) for image in images),
            "mean_gain_over_all": mean([max(deltas[(image, "all")], deltas[(image, "none")]) - deltas[(image, "all")] for image in images]),
            "mean_gap_to_full_oracle": mean([full_oracle[image] - max(deltas[(image, "all")], deltas[(image, "none")]) for image in images]),
        },
        {
            "policy": "loocv_logistic_slice_gate",
            "images": len(images),
            "mean_delta_psnr": mean([fnum(row, "predicted_delta_psnr") for row in folds]),
            "worst_delta_psnr": min(fnum(row, "predicted_delta_psnr") for row in folds),
            "mean_gain_over_all": mean([fnum(row, "gain_over_all") for row in folds]),
            "mean_gap_to_full_oracle": mean([fnum(row, "gap_to_full_oracle") for row in folds]),
        },
        {
            "policy": "full_subset_oracle",
            "images": len(images),
            "mean_delta_psnr": mean([full_oracle[image] for image in images]),
            "worst_delta_psnr": min(full_oracle[image] for image in images),
            "mean_gain_over_all": mean([full_oracle[image] - deltas[(image, "all")] for image in images]),
            "mean_gap_to_full_oracle": 0.0,
        },
    ]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.output_prefix.with_suffix(".folds.csv"): folds,
        args.output_prefix.with_suffix(".slice_probs.csv"): prob_rows,
        args.output_prefix.with_suffix(".coefs.csv"): coef_rows,
        args.output_prefix.with_suffix(".summary.csv"): summary,
    }
    for path, rows in outputs.items():
        with path.open("w", newline="") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)

    pred_counts = Counter(row["predicted_slice_set"] for row in folds)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "features": features,
                "summary": summary,
                "folds": folds,
                "predicted_slice_set_counts": dict(pred_counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# E321 EF-LIC Logistic Slice-Gate LOOCV",
        "",
        "Offline diagnostic trained on E318 oracle-active labels and scored with E317 powerset rows.",
        "",
        "| policy | images | mean_delta_psnr | worst_delta_psnr | mean_gain_over_all | mean_gap_to_full_oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['images']} | {row['mean_delta_psnr']:.6f} | "
            f"{row['worst_delta_psnr']:.6f} | {row['mean_gain_over_all']:.6f} | "
            f"{row['mean_gap_to_full_oracle']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Predicted slice-set counts:",
            "",
            *[f"- `{k}`: {v}" for k, v in sorted(pred_counts.items())],
            "",
            "Notes:",
            "",
            "- Outcome columns such as delta PSNR, best subset, and oracle agreement are excluded from features.",
            "- Hyperparameters and probability threshold are selected using only the training images in each LOOCV fold.",
            "- This is not final RD evidence; it tests whether a learned decoder-available controller has more signal than hand thresholds.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
