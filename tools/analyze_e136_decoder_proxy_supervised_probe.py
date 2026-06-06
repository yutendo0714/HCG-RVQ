#!/usr/bin/env python3
"""Split-protocol supervised probes for decoder-known HCG reliability proxies."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
INPUT_CSV = ANALYSIS_DIR / "e132_usage_controller_teacher_labels_labels.csv"
OUT_PREFIX = ANALYSIS_DIR / "e136_decoder_proxy_supervised_probe"

PROTOCOLS = {
    "first_half_to_second_half": (lambda i: i < 12, lambda i: i >= 12),
    "second_half_to_first_half": (lambda i: i >= 12, lambda i: i < 12),
    "even_to_odd": (lambda i: i % 2 == 0, lambda i: i % 2 == 1),
    "odd_to_even": (lambda i: i % 2 == 1, lambda i: i % 2 == 0),
}

LABELS = [
    "rd_win",
    "safe_win_dead_le_0.050",
    "safe_win_dead_le_0.075",
    "safe_win_dead_le_0.100",
]

FEATURE_SETS: dict[str, dict[str, object]] = {
    "hyper_preindex": {
        "deployability": "decoder_preindex_no_side_bit_candidate",
        "features": [
            "hcg_s_q_mean",
            "hcg_s_q_std",
            "hcg_mu_q_abs_mean",
            "hcg_householder_v_abs_mean",
        ],
    },
    "baseline_diagnostic": {
        "deployability": "diagnostic_baseline_forward_or_encoder_metric",
        "features": [
            "base_dead_code_ratio",
            "base_perplexity",
            "base_stage_entropy",
            "base_latent_quant_mse",
            "base_rd_score",
        ],
    },
    "hyper_plus_baseline_diagnostic": {
        "deployability": "diagnostic_mixed_not_pure_decoder_preindex",
        "features": [
            "hcg_s_q_mean",
            "hcg_s_q_std",
            "hcg_mu_q_abs_mean",
            "hcg_householder_v_abs_mean",
            "base_dead_code_ratio",
            "base_perplexity",
            "base_stage_entropy",
            "base_latent_quant_mse",
            "base_rd_score",
        ],
    },
    "candidate_reference": {
        "deployability": "teacher_reference_requires_candidate_forward",
        "features": [
            "hcg_dead_code_ratio",
            "hcg_perplexity",
            "hcg_stage_entropy",
            "hcg_latent_quant_mse",
            "hcg_householder_delta_rms",
        ],
    },
}

OBJECTIVES = [
    ("mean_dead_budget", 0.05),
    ("mean_dead_budget", 0.075),
    ("strict_selected_dead_cap", 0.075),
    ("strict_selected_dead_cap", 0.10),
]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def read_rows() -> list[dict[str, object]]:
    with INPUT_CSV.open(newline="") as file:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(file):
            parsed: dict[str, object] = {"path": row["path"], "image_index": int(row["image_index"])}
            for key, value in row.items():
                if key not in parsed and key != "path":
                    parsed[key] = as_float(value)
            rows.append(parsed)
    return sorted(rows, key=lambda r: int(r["image_index"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standardize(
    train_rows: list[dict[str, object]], test_rows: list[dict[str, object]], features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray([[as_float(row[f]) for f in features] for row in train_rows], dtype=np.float64)
    test = np.asarray([[as_float(row[f]) for f in features] for row in test_rows], dtype=np.float64)
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale[scale < 1e-12] = 1.0
    return np.nan_to_num((train - center) / scale, nan=0.0), np.nan_to_num((test - center) / scale, nan=0.0)


def train_logistic(x: np.ndarray, y: np.ndarray, steps: int = 800, lr: float = 0.08, l2: float = 0.02) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(steps):
        logits = x_aug @ weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        grad = x_aug.T @ ((probs - y) * sample_weight) / len(y)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    logits = x_aug @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def policy_metrics(rows: list[dict[str, object]], selected: np.ndarray) -> dict[str, float]:
    if len(rows) == 0:
        return {
            "selected": 0.0,
            "delta_rd": float("nan"),
            "delta_dead": float("nan"),
            "delta_qmse": float("nan"),
            "q95_damage_rd": float("nan"),
            "max_damage_rd": float("nan"),
            "max_selected_dead": float("nan"),
            "selected_win_rate": float("nan"),
        }
    selected = selected.astype(bool)
    delta_rd = np.asarray([as_float(row["delta_rd_score"]) for row in rows], dtype=np.float64)
    delta_dead = np.asarray([as_float(row["delta_dead_code_ratio"]) for row in rows], dtype=np.float64)
    delta_qmse = np.asarray([as_float(row["delta_latent_quant_mse"]) for row in rows], dtype=np.float64)
    chosen_rd = np.where(selected, delta_rd, 0.0)
    chosen_dead = np.where(selected, delta_dead, 0.0)
    chosen_qmse = np.where(selected, delta_qmse, 0.0)
    damage = np.maximum(chosen_rd, 0.0)
    return {
        "selected": float(selected.sum()),
        "delta_rd": float(chosen_rd.mean()),
        "delta_dead": float(chosen_dead.mean()),
        "delta_qmse": float(chosen_qmse.mean()),
        "q95_damage_rd": float(np.quantile(damage, 0.95)),
        "max_damage_rd": float(damage.max()),
        "max_selected_dead": float(delta_dead[selected].max()) if selected.any() else 0.0,
        "selected_win_rate": float(np.mean(delta_rd[selected] < 0.0)) if selected.any() else float("nan"),
    }


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    vals = sorted({float(v) for v in scores}, reverse=True)
    return [float("inf")] + [v - 1e-12 for v in vals]


def choose_threshold(
    train_rows: list[dict[str, object]], train_scores: np.ndarray, objective: str, budget: float
) -> tuple[float, dict[str, float]]:
    best_threshold = float("inf")
    best_metrics = policy_metrics(train_rows, np.zeros(len(train_rows), dtype=bool))
    best_key = (best_metrics["delta_rd"], best_metrics["q95_damage_rd"], -best_metrics["selected"])
    for threshold in candidate_thresholds(train_scores):
        selected = train_scores >= threshold
        metrics = policy_metrics(train_rows, selected)
        if objective == "mean_dead_budget":
            ok = metrics["delta_dead"] <= budget + 1e-12
        elif objective == "strict_selected_dead_cap":
            ok = metrics["max_selected_dead"] <= budget + 1e-12
        else:
            raise ValueError(objective)
        if not ok:
            continue
        key = (metrics["delta_rd"], metrics["q95_damage_rd"], -metrics["selected"])
        if key < best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def summarize_metric(rows: list[dict[str, object]], key: str) -> tuple[float, float, float]:
    vals = [as_float(row[key]) for row in rows]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows = read_rows()
    detail_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for protocol, (train_pred, test_pred) in PROTOCOLS.items():
        train_rows = [r for r in rows if train_pred(int(r["image_index"]))]
        test_rows = [r for r in rows if test_pred(int(r["image_index"]))]
        for label in LABELS:
            y_train = np.asarray([as_float(r[label]) for r in train_rows], dtype=np.float64)
            for feature_set, meta in FEATURE_SETS.items():
                features = list(meta["features"])
                x_train, x_test = standardize(train_rows, test_rows, features)
                weights = train_logistic(x_train, y_train)
                train_scores = predict_logistic(x_train, weights)
                test_scores = predict_logistic(x_test, weights)
                for feature, weight in zip(["bias"] + features, weights):
                    weight_rows.append(
                        {
                            "protocol": protocol,
                            "label": label,
                            "feature_set": feature_set,
                            "deployability": meta["deployability"],
                            "feature": feature,
                            "weight": float(weight),
                        }
                    )
                for objective, budget in OBJECTIVES:
                    threshold, train_metrics = choose_threshold(train_rows, train_scores, objective, budget)
                    test_metrics = policy_metrics(test_rows, test_scores >= threshold)
                    row = {
                        "protocol": protocol,
                        "label": label,
                        "feature_set": feature_set,
                        "deployability": meta["deployability"],
                        "objective": objective,
                        "budget": budget,
                        "threshold": threshold,
                        "positive_protocol": int(test_metrics["delta_rd"] < 0.0),
                    }
                    for prefix, metrics in (("train", train_metrics), ("test", test_metrics)):
                        for key, value in metrics.items():
                            row[f"{prefix}_{key}"] = value
                    detail_rows.append(row)

    summary_rows: list[dict[str, object]] = []
    group_keys = ["label", "feature_set", "deployability", "objective", "budget"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in detail_rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    metric_keys = [
        "test_selected",
        "test_delta_rd",
        "test_delta_dead",
        "test_delta_qmse",
        "test_q95_damage_rd",
        "test_max_damage_rd",
        "test_max_selected_dead",
        "test_selected_win_rate",
    ]
    for group_key, members in sorted(groups.items()):
        summary: dict[str, object] = dict(zip(group_keys, group_key))
        summary["num_protocols"] = len(members)
        summary["positive_protocols"] = int(sum(int(row["positive_protocol"]) for row in members))
        summary["worst_delta_rd"] = max(as_float(row["test_delta_rd"]) for row in members)
        summary["worst_q95_damage_rd"] = max(as_float(row["test_q95_damage_rd"]) for row in members)
        for metric in metric_keys:
            avg, low, high = summarize_metric(members, metric)
            summary[f"mean_{metric}"] = avg
            summary[f"min_{metric}"] = low
            summary[f"max_{metric}"] = high
        summary_rows.append(summary)

    return detail_rows, summary_rows, weight_rows


def top_rows(summary_rows: list[dict[str, object]], feature_sets: set[str]) -> list[dict[str, object]]:
    rows = [
        row
        for row in summary_rows
        if row["feature_set"] in feature_sets and int(row["positive_protocols"]) == int(row["num_protocols"])
    ]
    return sorted(
        rows,
        key=lambda row: (
            as_float(row["mean_test_delta_rd"]),
            as_float(row["mean_test_delta_dead"]),
            as_float(row["mean_test_q95_damage_rd"]),
        ),
    )[:12]


def write_markdown(summary_rows: list[dict[str, object]], deployable: list[dict[str, object]], reference: list[dict[str, object]]) -> None:
    main_rows = [
        row
        for row in summary_rows
        if row["objective"] == "mean_dead_budget" and abs(as_float(row["budget"]) - 0.05) < 1e-12
    ]
    lines = [
        "# E136 Decoder Proxy Supervised Probe",
        "",
        "This analysis trains tiny split-protocol logistic probes on E132 teacher labels, with a focus on whether decoder-known hyperprior summaries can act as a deployable reliability proxy.",
        "",
        f"- Input: `{INPUT_CSV}`",
        "- Protocols: first/second half and even/odd cross splits",
        "- Thresholds are selected on train splits using RD/dead-code objectives, then evaluated on held-out splits.",
        "",
        "## Main Budget 0.05 Result",
        "",
        "| feature set | label | deployability | protocols won | selected | delta RD | delta dead | q95 damage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(main_rows, key=lambda r: (str(r["feature_set"]), str(r["label"]))):
        lines.append(
            "| {feature_set} | {label} | {deployability} | {positive}/{total} | {selected} | {delta_rd} | {delta_dead} | {q95} |".format(
                feature_set=row["feature_set"],
                label=row["label"],
                deployability=row["deployability"],
                positive=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(as_float(row["mean_test_selected"]), 2),
                delta_rd=fmt(as_float(row["mean_test_delta_rd"])),
                delta_dead=fmt(as_float(row["mean_test_delta_dead"])),
                q95=fmt(as_float(row["mean_test_q95_damage_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Best Decoder-Known Hyper Probes",
            "",
            "| label | objective | budget | protocols won | selected | delta RD | delta dead | q95 damage | worst delta RD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deployable:
        lines.append(
            "| {label} | {objective} | {budget} | {positive}/{total} | {selected} | {delta_rd} | {delta_dead} | {q95} | {worst_rd} |".format(
                label=row["label"],
                objective=row["objective"],
                budget=fmt(as_float(row["budget"]), 3),
                positive=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(as_float(row["mean_test_selected"]), 2),
                delta_rd=fmt(as_float(row["mean_test_delta_rd"])),
                delta_dead=fmt(as_float(row["mean_test_delta_dead"])),
                q95=fmt(as_float(row["mean_test_q95_damage_rd"])),
                worst_rd=fmt(as_float(row["worst_delta_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Best Reference Probes",
            "",
            "| feature set | label | objective | budget | protocols won | selected | delta RD | delta dead | q95 damage |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reference:
        lines.append(
            "| {feature_set} | {label} | {objective} | {budget} | {positive}/{total} | {selected} | {delta_rd} | {delta_dead} | {q95} |".format(
                feature_set=row["feature_set"],
                label=row["label"],
                objective=row["objective"],
                budget=fmt(as_float(row["budget"]), 3),
                positive=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(as_float(row["mean_test_selected"]), 2),
                delta_rd=fmt(as_float(row["mean_test_delta_rd"])),
                delta_dead=fmt(as_float(row["mean_test_delta_dead"])),
                q95=fmt(as_float(row["mean_test_q95_damage_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A `hyper_preindex` logistic proxy is deployable in the same sense as the E135 hyper-preindex threshold, but it must beat that deterministic baseline before becoming the main no-side-bit route.",
            "- `baseline_diagnostic`, `hyper_plus_baseline_diagnostic`, and `candidate_reference` are retained as reference probes. They can guide teacher/proxy design but are not pure decoder-preindex claims.",
            "- Any promoted controller still needs a held-out checkpoint evaluation with RD, qMSE, codebook usage, q95 damage, and nonfinite checks.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_detail.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_summary.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_weights.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    detail_rows, summary_rows, weight_rows = run()
    deployable = top_rows(summary_rows, {"hyper_preindex"})
    reference = top_rows(summary_rows, {"baseline_diagnostic", "hyper_plus_baseline_diagnostic", "candidate_reference"})
    payload = {
        "experiment": "E136 decoder proxy supervised probe",
        "input": str(INPUT_CSV),
        "feature_sets": FEATURE_SETS,
        "objectives": OBJECTIVES,
        "top_deployable": deployable,
        "top_reference": reference,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_detail.csv"), detail_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), summary_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_weights.csv"), weight_rows)
    write_markdown(summary_rows, deployable, reference)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
