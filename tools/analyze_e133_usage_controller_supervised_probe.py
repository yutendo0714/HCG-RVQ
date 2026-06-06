#!/usr/bin/env python3
"""Split-protocol supervised probes for the usage-aware HCG controller."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
INPUT_CSV = ANALYSIS_DIR / "e132_usage_controller_teacher_labels_labels.csv"
OUT_PREFIX = ANALYSIS_DIR / "e133_usage_controller_supervised_probe"

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

FEATURE_SETS = {
    "baseline": [
        "base_dead_code_ratio",
        "base_perplexity",
        "base_stage_entropy",
        "base_latent_quant_mse",
        "base_rd_score",
    ],
    "candidate": [
        "hcg_dead_code_ratio",
        "hcg_perplexity",
        "hcg_stage_entropy",
        "hcg_latent_quant_mse",
        "hcg_householder_delta_rms",
    ],
    "candidate_compact": [
        "hcg_latent_quant_mse",
        "hcg_householder_delta_rms",
        "hcg_dead_code_ratio",
    ],
    "combined": [
        "base_dead_code_ratio",
        "base_perplexity",
        "base_stage_entropy",
        "base_latent_quant_mse",
        "base_rd_score",
        "hcg_dead_code_ratio",
        "hcg_perplexity",
        "hcg_stage_entropy",
        "hcg_latent_quant_mse",
        "hcg_householder_delta_rms",
    ],
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
        rows = []
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
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[std < 1e-12] = 1.0
    train = np.nan_to_num((train - mean) / std, nan=0.0)
    test = np.nan_to_num((test - mean) / std, nan=0.0)
    return train, test


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
            "q95_damage_rd": float("nan"),
            "max_damage_rd": float("nan"),
            "max_selected_dead": float("nan"),
            "label_positive_rate": float("nan"),
        }
    delta_rd = np.asarray([as_float(row["delta_rd_score"]) for row in rows], dtype=np.float64)
    delta_dead = np.asarray([as_float(row["delta_dead_code_ratio"]) for row in rows], dtype=np.float64)
    selected = selected.astype(bool)
    chosen_rd = np.where(selected, delta_rd, 0.0)
    chosen_dead = np.where(selected, delta_dead, 0.0)
    damage = np.maximum(chosen_rd, 0.0)
    if selected.any():
        max_selected_dead = float(delta_dead[selected].max())
    else:
        max_selected_dead = 0.0
    return {
        "selected": float(selected.sum()),
        "delta_rd": float(chosen_rd.mean()),
        "delta_dead": float(chosen_dead.mean()),
        "q95_damage_rd": float(np.quantile(damage, 0.95)),
        "max_damage_rd": float(damage.max()),
        "max_selected_dead": max_selected_dead,
    }


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    vals = sorted({float(v) for v in scores}, reverse=True)
    eps = 1e-12
    return [float("inf")] + [v - eps for v in vals]


def choose_threshold(
    train_rows: list[dict[str, object]], train_scores: np.ndarray, objective: str, budget: float
) -> tuple[float, dict[str, float]]:
    best_threshold = float("inf")
    best_metrics = policy_metrics(train_rows, np.zeros(len(train_rows), dtype=bool))
    best_key = (best_metrics["delta_rd"], -best_metrics["selected"])
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
        key = (metrics["delta_rd"], -metrics["selected"])
        if key < best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def summarize(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = [as_float(row[key]) for row in rows]
        out[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
        out[f"min_{key}"] = float(np.min(values)) if values else float("nan")
        out[f"max_{key}"] = float(np.max(values)) if values else float("nan")
    return out


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows = read_rows()
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for protocol, (train_pred, test_pred) in PROTOCOLS.items():
        train_rows = [r for r in rows if train_pred(int(r["image_index"]))]
        test_rows = [r for r in rows if test_pred(int(r["image_index"]))]
        for label in LABELS:
            y_train = np.asarray([as_float(r[label]) for r in train_rows], dtype=np.float64)
            for feature_set, features in FEATURE_SETS.items():
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
                            "feature": feature,
                            "weight": float(weight),
                        }
                    )

                for objective, budget in OBJECTIVES:
                    threshold, train_metrics = choose_threshold(train_rows, train_scores, objective, budget)
                    test_selected = test_scores >= threshold
                    test_metrics = policy_metrics(test_rows, test_selected)
                    row = {
                        "protocol": protocol,
                        "label": label,
                        "feature_set": feature_set,
                        "objective": objective,
                        "budget": budget,
                        "threshold": threshold,
                        "positive_protocol": int(test_metrics["delta_rd"] < 0.0),
                    }
                    for prefix, metrics in (("train", train_metrics), ("test", test_metrics)):
                        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
                    detail_rows.append(row)

    group_keys = ["label", "feature_set", "objective", "budget"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in detail_rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    metric_keys = [
        "test_selected",
        "test_delta_rd",
        "test_delta_dead",
        "test_q95_damage_rd",
        "test_max_damage_rd",
        "test_max_selected_dead",
    ]
    for key, members in sorted(groups.items()):
        base = dict(zip(group_keys, key))
        summary = {
            **base,
            "num_protocols": len(members),
            "positive_protocols": int(sum(int(r["positive_protocol"]) for r in members)),
            "worst_delta_rd": max(as_float(r["test_delta_rd"]) for r in members),
            "worst_q95_damage_rd": max(as_float(r["test_q95_damage_rd"]) for r in members),
            **summarize(members, metric_keys),
        }
        summary_rows.append(summary)

    return detail_rows, summary_rows, weight_rows


def top_recommendations(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = [
        r
        for r in summary_rows
        if r["label"] in {"safe_win_dead_le_0.075", "safe_win_dead_le_0.100", "rd_win"}
        and r["feature_set"] in {"candidate", "candidate_compact", "combined"}
        and int(r["positive_protocols"]) == int(r["num_protocols"])
    ]
    return sorted(
        filtered,
        key=lambda r: (
            as_float(r["mean_test_delta_rd"]),
            as_float(r["mean_test_delta_dead"]),
            -as_float(r["mean_test_selected"]),
        ),
    )[:12]


def write_markdown(summary_rows: list[dict[str, object]], recommendations: list[dict[str, object]]) -> None:
    lines = [
        "# E133 Usage Controller Supervised Probe",
        "",
        "This analysis trains tiny split-protocol logistic probes on the E132 teacher labels. It is a feasibility check for a learned usage/reliability controller; it does not add new GPU evaluation.",
        "",
        f"- Input: `{INPUT_CSV}`",
        "- Protocols: first/second half and even/odd cross splits",
        "- Thresholds are selected on the train split using the RD/dead-code objective, then evaluated on the held-out split.",
        "",
        "## Top Held-Out Controller Probes",
        "",
        "| label | feature set | objective | budget | protocols won | selected | delta RD | delta dead | q95 damage | worst delta RD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in recommendations:
        lines.append(
            "| {label} | {feature_set} | {objective} | {budget} | {won}/{total} | {selected} | {rd} | {dead} | {q95} | {worst} |".format(
                label=row["label"],
                feature_set=row["feature_set"],
                objective=row["objective"],
                budget=fmt(float(row["budget"])),
                won=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(float(row["mean_test_selected"])),
                rd=fmt(float(row["mean_test_delta_rd"])),
                dead=fmt(float(row["mean_test_delta_dead"])),
                q95=fmt(float(row["mean_test_q95_damage_rd"])),
                worst=fmt(float(row["worst_delta_rd"])),
            )
        )

    lines.extend(
        [
            "",
            "## Best Summary Rows By Objective",
            "",
            "| objective | budget | label | feature set | protocols won | selected | delta RD | delta dead | q95 damage | max selected dead |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for objective, budget in OBJECTIVES:
        candidates = [
            r
            for r in summary_rows
            if r["objective"] == objective
            and abs(as_float(r["budget"]) - budget) < 1e-12
            and r["feature_set"] in {"candidate", "candidate_compact", "combined"}
        ]
        if not candidates:
            continue
        best = sorted(candidates, key=lambda r: (as_float(r["mean_test_delta_rd"]), -int(r["positive_protocols"])))[0]
        lines.append(
            "| {objective} | {budget} | {label} | {feature_set} | {won}/{total} | {selected} | {rd} | {dead} | {q95} | {maxdead} |".format(
                objective=best["objective"],
                budget=fmt(float(best["budget"])),
                label=best["label"],
                feature_set=best["feature_set"],
                won=best["positive_protocols"],
                total=best["num_protocols"],
                selected=fmt(float(best["mean_test_selected"])),
                rd=fmt(float(best["mean_test_delta_rd"])),
                dead=fmt(float(best["mean_test_delta_dead"])),
                q95=fmt(float(best["mean_test_q95_damage_rd"])),
                maxdead=fmt(float(best["mean_test_max_selected_dead"])),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is intentionally a small-data probe, not a final learned controller result.",
            "- If supervised probes beat or match the E131 one-feature split protocol, the next code target is a tiny reliability head trained on independent teacher labels.",
            "- If they overfit or select unsafe tails, E131's candidate-forward deterministic guard remains the safer implementation target.",
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
    recommendations = top_recommendations(summary_rows)
    payload = {
        "experiment": "E133 usage controller supervised probe",
        "input": str(INPUT_CSV),
        "detail": detail_rows,
        "summary": summary_rows,
        "recommendations": recommendations,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_detail.csv"), detail_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), summary_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_weights.csv"), weight_rows)
    write_markdown(summary_rows, recommendations)
    print(json.dumps({"recommendations": recommendations[:6]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
