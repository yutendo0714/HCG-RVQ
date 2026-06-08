#!/usr/bin/env python3
"""Audit whether E236 slice/candidate stats can select useful EF-LIC HCG policies.

This is a diagnostic before building a two-stage local controller. It uses
already measured E236 codec-loop rows and asks whether decoder-side-ish candidate
statistics explain per-image policy success. It is not final codec evidence.
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

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kodak-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e236_eflic_kodak24_local_controller_map.csv",
    )
    parser.add_argument(
        "--clic-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e236_eflic_clicpro41_local_controller_map.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e311_eflic_slice_policy_signal",
    )
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = [(v, y) for v, y in zip(values, labels) if math.isfinite(v)]
    pos = [v for v, y in pairs if y == 1]
    neg = [v for v, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


BASE_FEATURES = [
    "y_alpha_mean",
    "y_alpha_active_frac",
    "y_avg_geometry_delta_rms",
    "y_avg_index_entropy",
    "y_avg_index_used_frac",
    "y_avg_residual_error_rms",
    "y_mismatch_frac",
]

SLICE_FEATURES = [
    "alpha_mean",
    "alpha_active_frac",
    "avg_geometry_delta_rms",
    "avg_index_entropy",
    "avg_index_used_frac",
    "avg_residual_error_rms",
    "stage0_geometry_delta_rms",
    "stage0_index_entropy",
    "stage0_index_perplexity",
    "stage0_index_used_frac",
    "stage0_residual_error_rms",
]


def read_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        reader = csv.DictReader(fobj)
        for raw in reader:
            score = to_float(raw.get("score_dists_3lpips"), 0.0)
            if not math.isfinite(score):
                score = to_float(raw.get("delta_dists"), 0.0) + 3.0 * to_float(raw.get("delta_lpips"), 0.0)
            nonfinite = int(to_float(raw.get("nonfinite"), 0.0))
            valid_contract = int(
                abs(to_float(raw.get("delta_bpp"), 0.0)) <= 1e-12
                and to_float(raw.get("max_decode_diff"), 0.0) <= 1e-10
                and nonfinite == 0
                and int(to_float(raw.get("payload_len_equal"), 0.0)) == 1
            )
            if not valid_contract:
                continue
            item: dict[str, Any] = {
                "dataset": dataset,
                "image": raw["image"],
                "policy": raw["policy"],
                "family": raw["family"],
                "score": score,
                "delta_dists": to_float(raw.get("delta_dists"), 0.0),
                "delta_lpips": to_float(raw.get("delta_lpips"), 0.0),
                "delta_psnr": to_float(raw.get("delta_psnr"), 0.0),
                "beneficial": int(score < 0.0),
                "strong_beneficial": int(score < -1e-3),
                "harmful": int(score > 0.0),
            }
            y_total = max(1.0, to_float(raw.get("y_total"), 0.0))
            y_mismatch = to_float(raw.get("y_mismatch"), 0.0)
            item["y_mismatch_frac"] = y_mismatch / y_total
            for key in BASE_FEATURES:
                if key != "y_mismatch_frac":
                    item[key] = to_float(raw.get(key), 0.0)
            for name in SLICE_FEATURES:
                vals = []
                for sid in range(4):
                    key = f"slice{sid}_{name}"
                    value = to_float(raw.get(key), float("nan"))
                    item[key] = value
                    vals.append(value)
                finite = [v for v in vals if math.isfinite(v)]
                if finite:
                    item[f"{name}_mean4"] = float(np.mean(finite))
                    item[f"{name}_std4"] = float(np.std(finite))
                    item[f"{name}_min4"] = float(np.min(finite))
                    item[f"{name}_max4"] = float(np.max(finite))
                    item[f"{name}_range4"] = float(np.max(finite) - np.min(finite))
                    item[f"{name}_late_minus_early"] = float(np.mean(finite[2:]) - np.mean(finite[:2])) if len(finite) == 4 else float("nan")
            rows.append(item)
    return rows


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    skip = {
        "dataset",
        "image",
        "policy",
        "family",
        "score",
        "delta_dists",
        "delta_lpips",
        "delta_psnr",
        "beneficial",
        "strong_beneficial",
        "harmful",
    }
    names = sorted(key for key in rows[0] if key not in skip and isinstance(rows[0][key], float))
    return names


def separation(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}) + ["pooled"]:
        subset = rows if dataset == "pooled" else [row for row in rows if row["dataset"] == dataset]
        for family in sorted({row["family"] for row in subset if row["policy"] != "zero"}) + ["nonzero"]:
            fam_subset = [row for row in subset if row["policy"] != "zero"] if family == "nonzero" else [row for row in subset if row["family"] == family]
            if len(fam_subset) < 8:
                continue
            labels = [int(row["score"] < 0.0) for row in fam_subset]
            if len(set(labels)) < 2:
                continue
            for feature in features:
                values = [float(row[feature]) for row in fam_subset]
                auc = auc_score(values, labels)
                if not math.isfinite(auc):
                    continue
                oriented_auc = max(auc, 1.0 - auc)
                out.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "feature": feature,
                        "rows": len(fam_subset),
                        "positive_frac": mean([float(v) for v in labels]),
                        "auc_high_is_good": auc,
                        "oriented_auc": oriented_auc,
                        "orientation": "high_good" if auc >= 0.5 else "low_good",
                        "corr_with_score": corr(values, [row["score"] for row in fam_subset]),
                    }
                )
    out.sort(key=lambda row: (-row["oriented_auc"], row["dataset"], row["family"], row["feature"]))
    return out


def policy_selector_score(rows: list[dict[str, Any]], feature: str, threshold: float, high_good: bool) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["policy"] != "zero":
            grouped[(row["dataset"], row["image"])].append(row)
    selected_scores = []
    selected_policies = []
    for _, cand in grouped.items():
        eligible = []
        for row in cand:
            value = float(row[feature])
            if not math.isfinite(value):
                continue
            if (value >= threshold) if high_good else (value <= threshold):
                eligible.append(row)
        if not eligible:
            selected_scores.append(0.0)
            selected_policies.append("zero")
            continue
        chooser = max if high_good else min
        best = chooser(eligible, key=lambda row: float(row[feature]))
        selected_scores.append(float(best["score"]))
        selected_policies.append(str(best["policy"]))
    return {
        "images": len(selected_scores),
        "mean_score": mean(selected_scores),
        "worst_score": max(selected_scores) if selected_scores else float("nan"),
        "win_frac": mean([float(v < 0.0) for v in selected_scores]),
        "nonpositive_frac": mean([float(v <= 0.0) for v in selected_scores]),
        "selected_frac": mean([float(p != "zero") for p in selected_policies]),
        "policy_counts": dict(Counter(selected_policies)),
    }


def train_test_rules(rows: list[dict[str, Any]], features: list[str], top_feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    splits = [("kodak24", "clicpro41"), ("clicpro41", "kodak24")]
    candidate_features = []
    for item in top_feature_rows:
        if item["feature"] not in candidate_features:
            candidate_features.append(item["feature"])
        if len(candidate_features) >= 40:
            break
    if not candidate_features:
        candidate_features = features[:40]
    for train_ds, test_ds in splits:
        train = [row for row in rows if row["dataset"] == train_ds]
        test = [row for row in rows if row["dataset"] == test_ds]
        for objective in ["best_mean", "safe_worst_nonpositive"]:
            best: dict[str, Any] | None = None
            for feature in candidate_features:
                vals = [float(row[feature]) for row in train if row["policy"] != "zero" and math.isfinite(float(row[feature]))]
                if not vals:
                    continue
                thresholds = sorted(set(float(x) for x in np.quantile(vals, np.linspace(0.0, 1.0, 41))))
                for high_good in [True, False]:
                    for th in thresholds:
                        train_score = policy_selector_score(train, feature, th, high_good)
                        if objective == "safe_worst_nonpositive" and train_score["worst_score"] > 0.0:
                            continue
                        key = (train_score["mean_score"], -train_score["win_frac"], -train_score["selected_frac"])
                        if best is None or key < best["_key"]:
                            best = {
                                "_key": key,
                                "feature": feature,
                                "threshold": th,
                                "orientation": "high_good" if high_good else "low_good",
                                "train": train_score,
                            }
            if best is None:
                continue
            test_score = policy_selector_score(test, best["feature"], best["threshold"], best["orientation"] == "high_good")
            out.append(
                {
                    "train_dataset": train_ds,
                    "test_dataset": test_ds,
                    "objective": objective,
                    "feature": best["feature"],
                    "threshold": best["threshold"],
                    "orientation": best["orientation"],
                    "train_mean_score": best["train"]["mean_score"],
                    "train_worst_score": best["train"]["worst_score"],
                    "train_win_frac": best["train"]["win_frac"],
                    "train_selected_frac": best["train"]["selected_frac"],
                    "test_mean_score": test_score["mean_score"],
                    "test_worst_score": test_score["worst_score"],
                    "test_win_frac": test_score["win_frac"],
                    "test_selected_frac": test_score["selected_frac"],
                    "test_policy_counts": json.dumps(test_score["policy_counts"], sort_keys=True),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.kodak_csv, "kodak24") + read_rows(args.clic_csv, "clicpro41")
    features = feature_names(rows)
    sep = separation(rows, features)
    rules = train_test_rules(rows, features, sep[: args.top_k])

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sep_path = args.output_prefix.with_suffix(".feature_separation.csv")
    rule_path = args.output_prefix.with_suffix(".rules.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(sep_path, sep)
    write_csv(rule_path, rules)
    json_path.write_text(
        json.dumps(
            {
                "experiment": "E311 EF-LIC slice/candidate policy signal audit",
                "rows": len(rows),
                "features": features,
                "top_feature_separation": sep[: args.top_k],
                "rules": rules,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E311 EF-LIC Slice/Candidate Policy Signal Audit\n\n")
        fobj.write(
            "This audits whether existing E236 slice/candidate statistics are strong enough to drive a policy selector. "
            "It is diagnostic only: policy candidate statistics are not automatically a decoder-side decision rule.\n\n"
        )
        fobj.write(f"- Rows: `{len(rows)}` valid codec-loop rows\n")
        fobj.write(f"- Features: `{len(features)}` scalar global/slice summary features\n\n")
        fobj.write("## Top Feature Separations\n\n")
        fobj.write("| dataset | family | feature | rows | pos frac | oriented AUC | orientation | corr(score) |\n")
        fobj.write("|---|---|---|---:|---:|---:|---|---:|\n")
        for item in sep[: args.top_k]:
            fobj.write(
                f"| {item['dataset']} | {item['family']} | {item['feature']} | {item['rows']} | "
                f"{item['positive_frac']:.3f} | {item['oriented_auc']:.3f} | {item['orientation']} | "
                f"{item['corr_with_score']:+.3f} |\n"
            )
        fobj.write("\n## Cross-Dataset Single-Feature Rules\n\n")
        fobj.write("| train | test | objective | feature | orient | threshold | train score | train worst | test score | test worst | test win | selected | policies |\n")
        fobj.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for item in rules:
            fobj.write(
                f"| {item['train_dataset']} | {item['test_dataset']} | {item['objective']} | {item['feature']} | "
                f"{item['orientation']} | {item['threshold']:.6g} | {item['train_mean_score']:+.6f} | "
                f"{item['train_worst_score']:+.6f} | {item['test_mean_score']:+.6f} | {item['test_worst_score']:+.6f} | "
                f"{item['test_win_frac']:.3f} | {item['test_selected_frac']:.3f} | `{item['test_policy_counts']}` |\n"
            )
        fobj.write("\nInterpretation:\n\n")
        fobj.write(
            "- Strong same-dataset separability but weak cross-dataset rules means E236 stats are useful diagnostics, not a deployable controller.\n"
        )
        fobj.write(
            "- If safe rules fall back or transfer poorly, the next step is true local residual/headroom labels and a two-stage controller.\n"
        )
    print(f"wrote {sep_path}, {rule_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
