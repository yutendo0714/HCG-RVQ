#!/usr/bin/env python3
"""Build teacher labels and feature diagnostics for a usage-aware HCG controller."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
INPUT_CSV = ANALYSIS_DIR / "e129_staged_geometry_kodak24_audit_per_image.csv"
OUT_PREFIX = ANALYSIS_DIR / "e132_usage_controller_teacher_labels"
BASELINE_CASE = "hcs_warmup_step30"
TARGET_CASE = "staged_gate001_step30"
DEAD_CAPS = [0.05, 0.075, 0.10]

FEATURES = [
    "base_dead_code_ratio",
    "base_perplexity",
    "base_stage_entropy",
    "base_latent_quant_mse",
    "base_rd_score",
    "base_mse",
    "base_psnr",
    "hcg_dead_code_ratio",
    "hcg_perplexity",
    "hcg_stage_entropy",
    "hcg_latent_quant_mse",
    "hcg_s_q_mean",
    "hcg_s_q_std",
    "hcg_mu_q_abs_mean",
    "hcg_householder_delta_rms",
    "hcg_householder_v_abs_mean",
]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = finite(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_records() -> list[dict[str, object]]:
    raw = read_csv(INPUT_CSV)
    by_image: dict[int, dict[str, dict[str, str]]] = {}
    for row in raw:
        by_image.setdefault(int(row["image_index"]), {})[row["case"]] = row

    records: list[dict[str, object]] = []
    for image_index, rows in sorted(by_image.items()):
        base = rows[BASELINE_CASE]
        hcg = rows[TARGET_CASE]
        record: dict[str, object] = {
            "image_index": image_index,
            "path": hcg["path"],
            "delta_rd_score": as_float(hcg["delta_rd_score"]),
            "delta_dead_code_ratio": as_float(hcg["delta_dead_code_ratio"]),
            "delta_perplexity": as_float(hcg["delta_perplexity"]),
            "delta_stage_entropy": as_float(hcg["delta_stage_entropy"]),
            "delta_latent_quant_mse": as_float(hcg["delta_latent_quant_mse"]),
            "delta_s_q_mean": as_float(hcg.get("delta_s_q_mean")),
            "delta_s_q_std": as_float(hcg.get("delta_s_q_std")),
            "rd_win": int(as_float(hcg["delta_rd_score"]) < 0.0),
        }
        for cap in DEAD_CAPS:
            key = f"safe_win_dead_le_{cap:.3f}"
            record[key] = int(record["rd_win"] and as_float(hcg["delta_dead_code_ratio"]) <= cap)
        for key in ("dead_code_ratio", "perplexity", "stage_entropy", "latent_quant_mse", "rd_score", "mse", "psnr"):
            record[f"base_{key}"] = as_float(base.get(key))
        for key in (
            "dead_code_ratio",
            "perplexity",
            "stage_entropy",
            "latent_quant_mse",
            "s_q_mean",
            "s_q_std",
            "mu_q_abs_mean",
            "householder_delta_rms",
            "householder_v_abs_mean",
        ):
            record[f"hcg_{key}"] = as_float(hcg.get(key))
        records.append(record)
    return records


def label_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = ["rd_win"] + [f"safe_win_dead_le_{cap:.3f}" for cap in DEAD_CAPS]
    rows: list[dict[str, object]] = []
    for label in labels:
        positives = [r for r in records if int(r[label]) == 1]
        negatives = [r for r in records if int(r[label]) == 0]
        rows.append(
            {
                "label": label,
                "positive": len(positives),
                "negative": len(negatives),
                "positive_rate": len(positives) / len(records) if records else float("nan"),
                "positive_mean_delta_rd": mean([as_float(r["delta_rd_score"]) for r in positives]),
                "positive_mean_delta_dead": mean([as_float(r["delta_dead_code_ratio"]) for r in positives]),
                "positive_mean_delta_qmse": mean([as_float(r["delta_latent_quant_mse"]) for r in positives]),
                "negative_mean_delta_rd": mean([as_float(r["delta_rd_score"]) for r in negatives]),
                "negative_mean_delta_dead": mean([as_float(r["delta_dead_code_ratio"]) for r in negatives]),
            }
        )
    return rows


def feature_separation(records: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = ["rd_win"] + [f"safe_win_dead_le_{cap:.3f}" for cap in DEAD_CAPS]
    rows: list[dict[str, object]] = []
    for label in labels:
        positives = [r for r in records if int(r[label]) == 1]
        negatives = [r for r in records if int(r[label]) == 0]
        for feature in FEATURES:
            pos_vals = [as_float(r[feature]) for r in positives]
            neg_vals = [as_float(r[feature]) for r in negatives]
            pos_mean = mean(pos_vals)
            neg_mean = mean(neg_vals)
            pooled = math.sqrt((std(pos_vals) ** 2 + std(neg_vals) ** 2) / 2.0)
            effect = (pos_mean - neg_mean) / pooled if pooled > 0.0 and math.isfinite(pooled) else float("nan")
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "positive_mean": pos_mean,
                    "negative_mean": neg_mean,
                    "mean_diff": pos_mean - neg_mean,
                    "abs_effect": abs(effect) if math.isfinite(effect) else float("nan"),
                    "signed_effect": effect,
                }
            )
    return rows


def write_markdown(records: list[dict[str, object]], summaries: list[dict[str, object]], features: list[dict[str, object]]) -> None:
    lines = [
        "# E132 Usage Controller Teacher Labels",
        "",
        "This package converts the E129 full-Kodak staged-geometry audit into teacher labels for a future usage-aware controller. It is analysis-only and does not add new GPU evaluation.",
        "",
        f"- Input: `{INPUT_CSV}`",
        f"- Target case: `{TARGET_CASE}`",
        f"- Images: `{len(records)}`",
        "",
        "## Label Summary",
        "",
        "| label | positives | rate | pos delta RD | pos delta dead | neg delta RD | neg delta dead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {positive} | {rate} | {pos_rd} | {pos_dead} | {neg_rd} | {neg_dead} |".format(
                label=row["label"],
                positive=row["positive"],
                rate=fmt(float(row["positive_rate"])),
                pos_rd=fmt(float(row["positive_mean_delta_rd"])),
                pos_dead=fmt(float(row["positive_mean_delta_dead"])),
                neg_rd=fmt(float(row["negative_mean_delta_rd"])),
                neg_dead=fmt(float(row["negative_mean_delta_dead"])),
            )
        )

    lines.extend(
        [
            "",
            "## Top Feature Separation",
            "",
            "| label | feature | abs effect | positive mean | negative mean | diff |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label in ["rd_win", "safe_win_dead_le_0.050", "safe_win_dead_le_0.075", "safe_win_dead_le_0.100"]:
        label_rows = [r for r in features if r["label"] == label and math.isfinite(float(r["abs_effect"]))]
        for row in sorted(label_rows, key=lambda r: float(r["abs_effect"]), reverse=True)[:8]:
            lines.append(
                "| {label} | {feature} | {effect} | {pos} | {neg} | {diff} |".format(
                    label=row["label"],
                    feature=row["feature"],
                    effect=fmt(float(row["abs_effect"])),
                    pos=fmt(float(row["positive_mean"])),
                    neg=fmt(float(row["negative_mean"])),
                    diff=fmt(float(row["mean_diff"])),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `rd_win` is the broad mean-RD teacher and is positive on most images.",
            "- `safe_win_dead_le_0.075` and `safe_win_dead_le_0.100` are better near-term safety labels than the very strict `0.050` label, which has fewer positives.",
            "- Candidate-forward features such as `hcg_latent_quant_mse`, `hcg_householder_delta_rms`, `hcg_dead_code_ratio`, and `hcg_perplexity` should be the first controller inputs. Baseline-only features can be kept as auxiliary inputs.",
            "- The next trainable controller should be selected on a separate split and evaluated with the E129/E130 protocol rather than tuned on the reported images.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_labels.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_summary.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_feature_separation.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    records = build_records()
    summaries = label_summary(records)
    features = feature_separation(records)
    payload = {
        "experiment": "E132 usage controller teacher labels",
        "input": str(INPUT_CSV),
        "target_case": TARGET_CASE,
        "labels": records,
        "summary": summaries,
        "feature_separation": features,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_labels.csv"), records)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), summaries)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_feature_separation.csv"), features)
    write_markdown(records, summaries, features)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
