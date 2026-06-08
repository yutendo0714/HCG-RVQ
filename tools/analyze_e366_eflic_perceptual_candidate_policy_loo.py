#!/usr/bin/env python3
"""Leave-one-image-out EF-LIC perceptual candidate-policy audit.

This script treats E313/E364 slice-isolation rows as candidate actions and asks
whether simple decoder-visible statistics can recover the perceptual oracle
better than fixed slice policies.  PSNR is kept only as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TARGET_COLUMNS = {
    "active_dists",
    "active_lpips",
    "active_ms_ssim",
    "active_psnr",
    "base_dists",
    "base_lpips",
    "base_ms_ssim",
    "base_psnr",
    "bpp",
    "contract_ok",
    "delta_bpp",
    "delta_dists",
    "delta_lpips",
    "delta_ms_ssim",
    "delta_psnr",
    "max_baseline_diff",
    "max_decode_diff",
    "mean_baseline_diff",
    "mean_decode_diff",
    "nonfinite",
    "payload_equal",
    "payload_len_equal",
    "perceptual_score",
    "perceptual_score_win",
    "triple_perceptual_win",
}


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_rows(path: Path, include_noop: bool) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    clean: list[dict[str, Any]] = []
    for row in rows:
        if parse_float(row.get("contract_ok")) < 1.0:
            continue
        if parse_float(row.get("nonfinite")) != 0.0:
            continue
        if abs(parse_float(row.get("delta_bpp"))) > 1e-12:
            continue
        if abs(parse_float(row.get("max_decode_diff"))) > 1e-12:
            continue
        row = dict(row)
        row["candidate"] = row.get("active_slices", "")
        clean.append(row)

    if not include_noop:
        return clean

    by_image: dict[str, dict[str, Any]] = {}
    for row in clean:
        by_image.setdefault(row["image"], row)

    augmented = list(clean)
    for image, proto in sorted(by_image.items()):
        noop = {key: "" for key in proto.keys()}
        noop["image"] = image
        noop["candidate"] = "noop"
        noop["active_slices"] = "noop"
        noop["slice_count"] = "0"
        noop["perceptual_score"] = "0"
        noop["delta_psnr"] = "0"
        noop["delta_lpips"] = "0"
        noop["delta_dists"] = "0"
        noop["delta_ms_ssim"] = "0"
        noop["delta_bpp"] = "0"
        noop["contract_ok"] = "1"
        noop["nonfinite"] = "0"
        noop["max_decode_diff"] = "0"
        noop["payload_equal"] = "1"
        noop["payload_len_equal"] = "1"

        for key, value in proto.items():
            if key.startswith("y_") or key.startswith("z_"):
                noop[key] = value
            elif key.startswith("slice") and key not in {"slice_count"}:
                noop[key] = "0"
        augmented.append(noop)

    return augmented


def candidate_set(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["candidate"]) for row in rows})


def feature_columns(rows: list[dict[str, Any]], candidates: list[str]) -> list[str]:
    cols: list[str] = []
    for key in rows[0].keys():
        if key in TARGET_COLUMNS or key in {"image", "candidate", "active_slices"}:
            continue
        if key.startswith("slice") or key.startswith("y_") or key.startswith("z_"):
            vals = [parse_float(row.get(key)) for row in rows]
            finite = [v for v in vals if math.isfinite(v)]
            if finite and max(finite) != min(finite):
                cols.append(key)
    return cols


def make_matrix(
    rows: list[dict[str, Any]], cols: list[str], candidates: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x_parts = [np.ones((len(rows), 1), dtype=np.float64)]
    for cand in candidates:
        x_parts.append(
            np.array([[1.0 if row["candidate"] == cand else 0.0] for row in rows], dtype=np.float64)
        )
    numeric = np.array(
        [[parse_float(row.get(col)) for col in cols] for row in rows],
        dtype=np.float64,
    )
    numeric[~np.isfinite(numeric)] = 0.0
    x_parts.append(numeric)
    x = np.concatenate(x_parts, axis=1)
    y = np.array([parse_float(row.get("perceptual_score")) for row in rows], dtype=np.float64)
    return x, y


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * eye, x.T @ y)


def summarize_choices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [parse_float(row["perceptual_score"]) for row in rows]
    dpsnr = [parse_float(row.get("delta_psnr")) for row in rows]
    return {
        "mean_score": float(np.mean(scores)) if scores else math.nan,
        "worst_score": float(np.max(scores)) if scores else math.nan,
        "score_wins": int(sum(1 for v in scores if v < 0.0)),
        "mean_delta_psnr": float(np.mean(dpsnr)) if dpsnr else math.nan,
        "choices": dict(Counter(row["candidate"] for row in rows)),
    }


def oracle_by_image(groups: dict[str, list[dict[str, Any]]], key: str, lower: bool = True) -> list[dict[str, Any]]:
    out = []
    for image in sorted(groups):
        rows = groups[image]
        if lower:
            out.append(min(rows, key=lambda r: parse_float(r[key])))
        else:
            out.append(max(rows, key=lambda r: parse_float(r[key])))
    return out


def policy_cv(
    rows: list[dict[str, Any]],
    cols: list[str],
    candidates: list[str],
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image"]].append(row)

    chosen: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for held_image in sorted(groups):
        train = [row for row in rows if row["image"] != held_image]
        test = groups[held_image]
        x_train, y_train = make_matrix(train, cols, candidates)
        x_test, _ = make_matrix(test, cols, candidates)

        mu = x_train[:, len(candidates) + 1 :].mean(axis=0)
        sigma = x_train[:, len(candidates) + 1 :].std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, len(candidates) + 1 :] = (x_train[:, len(candidates) + 1 :] - mu) / sigma
        x_test[:, len(candidates) + 1 :] = (x_test[:, len(candidates) + 1 :] - mu) / sigma

        beta = fit_ridge(x_train, y_train, alpha)
        pred = x_test @ beta
        idx = int(np.argmin(pred))
        row = dict(test[idx])
        row["predicted_score"] = float(pred[idx])
        chosen.append(row)

        best_score = min(test, key=lambda r: parse_float(r["perceptual_score"]))
        best_psnr = max(test, key=lambda r: parse_float(r["delta_psnr"]))
        detail.append(
            {
                "image": held_image,
                "chosen": row["candidate"],
                "chosen_score": parse_float(row["perceptual_score"]),
                "chosen_delta_psnr": parse_float(row["delta_psnr"]),
                "predicted_score": float(pred[idx]),
                "oracle": best_score["candidate"],
                "oracle_score": parse_float(best_score["perceptual_score"]),
                "psnr_oracle": best_psnr["candidate"],
                "psnr_oracle_score": parse_float(best_psnr["perceptual_score"]),
            }
        )
    return chosen, detail


def cv_best_fixed(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    chosen = []
    images = sorted(groups)
    candidates = sorted({row["candidate"] for rows in groups.values() for row in rows})
    for held_image in images:
        train = [row for image in images if image != held_image for row in groups[image]]
        means = {}
        for cand in candidates:
            vals = [parse_float(row["perceptual_score"]) for row in train if row["candidate"] == cand]
            if vals:
                means[cand] = float(np.mean(vals))
        best_cand = min(means, key=means.get)
        held_rows = [row for row in groups[held_image] if row["candidate"] == best_cand]
        if held_rows:
            chosen.append(held_rows[0])
    return chosen


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--alpha", nargs="*", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--no-noop", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.rows, include_noop=not args.no_noop)
    candidates = candidate_set(rows)
    cols = feature_columns(rows, candidates)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image"]].append(row)

    baselines = {
        "oracle_score": summarize_choices(oracle_by_image(groups, "perceptual_score", lower=True)),
        "oracle_psnr": summarize_choices(oracle_by_image(groups, "delta_psnr", lower=False)),
        "cv_best_fixed": summarize_choices(cv_best_fixed(groups)),
    }
    for cand in candidates:
        cand_rows = [next(r for r in groups[image] if r["candidate"] == cand) for image in sorted(groups) if any(r["candidate"] == cand for r in groups[image])]
        baselines[f"fixed_{cand}"] = summarize_choices(cand_rows)

    policies = {}
    details_by_alpha = {}
    for alpha in args.alpha:
        chosen, detail = policy_cv(rows, cols, candidates, alpha)
        key = f"ridge_alpha_{alpha:g}"
        policies[key] = summarize_choices(chosen)
        details_by_alpha[key] = detail

    best_policy_key = min(policies, key=lambda k: policies[k]["mean_score"])

    result = {
        "rows": len(rows),
        "images": len(groups),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "feature_count": len(cols),
        "features": cols,
        "baselines": baselines,
        "policies": policies,
        "best_policy": best_policy_key,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True))
    write_csv(args.output_prefix.with_suffix(".per_image.csv"), details_by_alpha[best_policy_key])

    lines = [
        "# E366 EF-LIC Perceptual Candidate Policy LOO",
        "",
        f"Rows: {len(rows)}",
        f"Images: {len(groups)}",
        f"Candidates: {', '.join(candidates)}",
        f"Feature count: {len(cols)}",
        "",
        "## Baselines",
        "",
        "| policy | mean score | worst score | score wins | mean dPSNR | choices |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, val in baselines.items():
        lines.append(
            f"| {key} | {val['mean_score']:+.6f} | {val['worst_score']:+.6f} | "
            f"{val['score_wins']}/{len(groups)} | {val['mean_delta_psnr']:+.6f} | {val['choices']} |"
        )
    lines.extend(["", "## Learned Policies", "", "| policy | mean score | worst score | score wins | mean dPSNR | choices |", "|---|---:|---:|---:|---:|---|"])
    for key, val in policies.items():
        lines.append(
            f"| {key} | {val['mean_score']:+.6f} | {val['worst_score']:+.6f} | "
            f"{val['score_wins']}/{len(groups)} | {val['mean_delta_psnr']:+.6f} | {val['choices']} |"
        )
    lines.extend(
        [
            "",
            f"Best learned policy: `{best_policy_key}`.",
            "",
            "Interpretation note: lower score is better. PSNR is diagnostic only.",
        ]
    )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
