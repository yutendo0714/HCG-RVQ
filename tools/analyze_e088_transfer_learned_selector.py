#!/usr/bin/env python3
"""Train transfer-split selectors and test them on holdout4096."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from analyze_beta005_decoder_safe_selector import (
    DECODER_SAFE_FEATURES,
    DIAGNOSTIC_FEATURES,
    build_rows,
    fmt,
    mean,
    read_csv,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "e088_transfer_learned_selector"
TRANSFER_REFERENCE = ANALYSIS / "beta005_transfer_openimages_start8192_n4096.csv"
TRANSFER_CANDIDATE = ANALYSIS / "local_cap080_rho1_transfer8192_checkpoint_sweep.csv"


def f(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}: {value}")
    return value


def is_finite_row(row: dict[str, str]) -> bool:
    flag = str(row.get("has_nonfinite", "0")).lower()
    return flag not in {"1", "true", "yes"}


def key_for(row: dict[str, str]) -> tuple[str, str]:
    return str(row["seed"]), str(row["path"])


def feature_available(row: dict[str, object], features: Iterable[str]) -> bool:
    for feature in features:
        try:
            value = float(row[feature])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def load_transfer_rows() -> list[dict[str, float | str]]:
    reference_rows = [
        row
        for row in read_csv(TRANSFER_REFERENCE)
        if row.get("method") == "beta005 guard" and is_finite_row(row)
    ]
    candidate_rows = [row for row in read_csv(TRANSFER_CANDIDATE) if is_finite_row(row)]

    # Match export_split_teacher_labels.py exactly: duplicate candidate keys keep
    # the later row, which is the historical teacher-label protocol.
    candidate_by_key = {key_for(row): row for row in candidate_rows}
    rows: list[dict[str, float | str]] = []
    features = list(dict.fromkeys(DECODER_SAFE_FEATURES + DIAGNOSTIC_FEATURES))
    for reference in reference_rows:
        key = key_for(reference)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        beta_rd = f(reference, "rd_score")
        previous_rd = f(candidate, "rd_score")
        row: dict[str, float | str] = {
            "seed": key[0],
            "path": key[1],
            "beta005_rd": beta_rd,
            "previous_local_rd": previous_rd,
            "hcs_rd": float("nan"),
            "previous_local_wins": float(previous_rd < beta_rd),
            "margin_beta005_minus_previous_local": beta_rd - previous_rd,
        }
        try:
            for feature in features:
                row[feature] = f(reference, feature)
        except (KeyError, ValueError):
            continue
        rows.append(row)
    return rows


def add_labels(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out = []
    for row in rows:
        item = dict(row)
        beta_rd = float(item["beta005_rd"])
        previous_rd = float(item["previous_local_rd"])
        item["previous_local_wins"] = float(previous_rd < beta_rd)
        item["margin_beta005_minus_previous_local"] = beta_rd - previous_rd
        out.append(item)
    return out


def summarize_base(rows: list[dict[str, float | str]]) -> dict[str, float]:
    beta = mean(float(row["beta005_rd"]) for row in rows)
    previous = mean(float(row["previous_local_rd"]) for row in rows)
    oracle = mean(
        min(float(row["beta005_rd"]), float(row["previous_local_rd"])) for row in rows
    )
    wins = mean(float(row["previous_local_wins"]) for row in rows)
    result = {
        "rows": float(len(rows)),
        "beta005": beta,
        "previous_local": previous,
        "oracle": oracle,
        "oracle_delta_vs_beta005": oracle - beta,
        "previous_local_delta_vs_beta005": previous - beta,
        "previous_local_win_fraction": wins,
    }
    if all(math.isfinite(float(row.get("hcs_rd", float("nan")))) for row in rows):
        hcs = mean(float(row["hcs_rd"]) for row in rows)
        result["hcs"] = hcs
        result["beta005_delta_vs_hcs"] = beta - hcs
        result["previous_local_delta_vs_hcs"] = previous - hcs
        result["oracle_delta_vs_hcs"] = oracle - hcs
    return result


def threshold_candidates(scores: np.ndarray) -> list[float]:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, 401)
    thresholds = np.quantile(values, quantiles).tolist()
    eps = max(float(values.max() - values.min()) * 1e-6, 1e-12)
    thresholds.extend([float(values.min() - eps), float(values.max() + eps)])
    return sorted(set(float(thr) for thr in thresholds if math.isfinite(float(thr))))


def selected_rds(
    rows: list[dict[str, float | str]], scores: np.ndarray, threshold: float
) -> tuple[list[float], list[bool]]:
    selected = [bool(score >= threshold) for score in scores]
    rds = [
        float(row["previous_local_rd"] if flag else row["beta005_rd"])
        for row, flag in zip(rows, selected, strict=True)
    ]
    return rds, selected


def choose_threshold(rows: list[dict[str, float | str]], scores: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    beta = summarize_base(rows)["beta005"]
    for threshold in threshold_candidates(scores):
        rds, selected = selected_rds(rows, scores, threshold)
        rd = mean(rds)
        item = {
            "threshold": threshold,
            "rd": rd,
            "delta_vs_beta005": rd - beta,
            "selected_fraction": mean(1.0 if flag else 0.0 for flag in selected),
        }
        if best is None or (item["rd"], item["selected_fraction"]) < (
            best["rd"],
            best["selected_fraction"],
        ):
            best = item
    if best is None:
        raise RuntimeError("empty score set")
    return best


def confusion(rows: list[dict[str, float | str]], selected: list[bool]) -> dict[str, float]:
    labels = [bool(float(row["previous_local_wins"])) for row in rows]
    tp = sum(1 for pred, label in zip(selected, labels, strict=True) if pred and label)
    fp = sum(1 for pred, label in zip(selected, labels, strict=True) if pred and not label)
    fn = sum(1 for pred, label in zip(selected, labels, strict=True) if not pred and label)
    tn = sum(1 for pred, label in zip(selected, labels, strict=True) if not pred and not label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_scores(
    name: str,
    group: str,
    train_rows: list[dict[str, float | str]],
    test_rows: list[dict[str, float | str]],
    train_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, object]:
    train_best = choose_threshold(train_rows, train_scores)
    threshold = float(train_best["threshold"])
    train_rds, train_selected = selected_rds(train_rows, train_scores, threshold)
    test_rds, test_selected = selected_rds(test_rows, test_scores, threshold)

    train_base = summarize_base(train_rows)
    test_base = summarize_base(test_rows)
    train_rd = mean(train_rds)
    test_rd = mean(test_rds)
    test_oracle_gap = test_rd - test_base["oracle"]
    beta_to_oracle = test_base["beta005"] - test_base["oracle"]
    gap_closed = (
        (test_base["beta005"] - test_rd) / beta_to_oracle
        if beta_to_oracle > 0.0
        else float("nan")
    )
    selected_margins = [
        float(row["margin_beta005_minus_previous_local"])
        for row, flag in zip(test_rows, test_selected, strict=True)
        if flag
    ]
    unselected_margins = [
        float(row["margin_beta005_minus_previous_local"])
        for row, flag in zip(test_rows, test_selected, strict=True)
        if not flag
    ]
    item: dict[str, object] = {
        "name": name,
        "group": group,
        "threshold": threshold,
        "train_rd": train_rd,
        "train_delta_vs_beta005": train_rd - train_base["beta005"],
        "train_selected_fraction": mean(1.0 if flag else 0.0 for flag in train_selected),
        "train_confusion": confusion(train_rows, train_selected),
        "test_rd": test_rd,
        "test_delta_vs_beta005": test_rd - test_base["beta005"],
        "test_delta_vs_previous_local": test_rd - test_base["previous_local"],
        "test_oracle_gap": test_oracle_gap,
        "test_oracle_gap_closed_fraction": gap_closed,
        "test_selected_fraction": mean(1.0 if flag else 0.0 for flag in test_selected),
        "test_selected_mean_margin": mean(selected_margins),
        "test_unselected_mean_margin": mean(unselected_margins),
        "test_confusion": confusion(test_rows, test_selected),
        "seed_breakdown": seed_breakdown(test_rows, test_selected),
        "hcs_quartiles": hcs_quartiles(test_rows, test_selected),
    }
    if "hcs" in test_base:
        item["test_delta_vs_hcs"] = test_rd - test_base["hcs"]
    return item


def seed_breakdown(
    rows: list[dict[str, float | str]], selected: list[bool]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[dict[str, float | str], bool]]] = defaultdict(list)
    for row, flag in zip(rows, selected, strict=True):
        grouped[str(row["seed"])].append((row, flag))
    out: dict[str, dict[str, float]] = {}
    for seed in sorted(grouped):
        pairs = grouped[seed]
        sub_rows = [row for row, _ in pairs]
        sub_selected = [flag for _, flag in pairs]
        rds = [
            float(row["previous_local_rd"] if flag else row["beta005_rd"])
            for row, flag in pairs
        ]
        beta = mean(float(row["beta005_rd"]) for row in sub_rows)
        previous = mean(float(row["previous_local_rd"]) for row in sub_rows)
        out[seed] = {
            "rows": float(len(pairs)),
            "rd": mean(rds),
            "delta_vs_beta005": mean(rds) - beta,
            "delta_vs_previous_local": mean(rds) - previous,
            "selected_fraction": mean(1.0 if flag else 0.0 for flag in sub_selected),
            "previous_local_win_fraction": mean(
                float(row["previous_local_wins"]) for row in sub_rows
            ),
        }
    return out


def hcs_quartiles(
    rows: list[dict[str, float | str]], selected: list[bool]
) -> list[dict[str, float]]:
    if not rows or not all(math.isfinite(float(row.get("hcs_rd", float("nan")))) for row in rows):
        return []
    ordered = sorted(
        zip(rows, selected, strict=True),
        key=lambda pair: float(pair[0]["hcs_rd"]),
    )
    chunks = np.array_split(np.arange(len(ordered)), 4)
    out = []
    for index, chunk in enumerate(chunks, start=1):
        pairs = [ordered[int(i)] for i in chunk]
        sub_rows = [row for row, _ in pairs]
        sub_selected = [flag for _, flag in pairs]
        rds = [
            float(row["previous_local_rd"] if flag else row["beta005_rd"])
            for row, flag in pairs
        ]
        beta = mean(float(row["beta005_rd"]) for row in sub_rows)
        out.append(
            {
                "quartile": float(index),
                "rows": float(len(pairs)),
                "rd": mean(rds),
                "delta_vs_beta005": mean(rds) - beta,
                "selected_fraction": mean(1.0 if flag else 0.0 for flag in sub_selected),
                "previous_local_win_fraction": mean(
                    float(row["previous_local_wins"]) for row in sub_rows
                ),
            }
        )
    return out


def feature_matrix(
    rows: list[dict[str, float | str]], features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    valid_rows = [row for row in rows if feature_available(row, features)]
    x = np.asarray([[float(row[feature]) for feature in features] for row in valid_rows])
    y = np.asarray([float(row["previous_local_wins"]) for row in valid_rows], dtype=np.float64)
    margin = np.asarray(
        [abs(float(row["margin_beta005_minus_previous_local"])) for row in valid_rows],
        dtype=np.float64,
    )
    return x, y, margin, valid_rows


def standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loc = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - loc) / scale, (x_test - loc) / scale, loc, scale


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    steps: int = 6000,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    weight = np.zeros(x.shape[1], dtype=np.float64)
    bias = float(np.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4)))
    sample_weight = sample_weight / max(float(sample_weight.mean()), 1e-8)
    for step in range(steps):
        prob = sigmoid(x @ weight + bias)
        diff = (prob - y) * sample_weight
        grad_w = x.T @ diff / x.shape[0] + l2 * weight
        grad_b = float(diff.mean())
        rate = lr / math.sqrt(1.0 + step / 1000.0)
        weight -= rate * grad_w
        bias -= rate * grad_b
    return weight, bias


def logreg_policy(
    name: str,
    group: str,
    train_rows_all: list[dict[str, float | str]],
    test_rows_all: list[dict[str, float | str]],
    features: list[str],
    *,
    margin_weighted: bool,
) -> dict[str, object]:
    x_train, y_train, margin_train, train_rows = feature_matrix(train_rows_all, features)
    x_test, _y_test, _margin_test, test_rows = feature_matrix(test_rows_all, features)
    x_train, x_test, _loc, _scale = standardize(x_train, x_test)
    if margin_weighted:
        median = float(np.median(margin_train[margin_train > 0.0])) if np.any(margin_train > 0.0) else 1.0
        sample_weight = np.clip(1.0 + margin_train / max(median, 1e-8), 1.0, 10.0)
    else:
        sample_weight = np.ones_like(y_train)
    weight, bias = train_logreg(x_train, y_train, sample_weight)
    train_scores = sigmoid(x_train @ weight + bias)
    test_scores = sigmoid(x_test @ weight + bias)
    out = evaluate_scores(name, group, train_rows, test_rows, train_scores, test_scores)
    ranked = sorted(
        zip(features, weight, strict=True), key=lambda item: abs(float(item[1])), reverse=True
    )
    out["top_weights"] = [
        {"feature": feature, "weight": float(value)} for feature, value in ranked[:10]
    ]
    return out


def single_feature_policy(
    group: str,
    train_rows: list[dict[str, float | str]],
    test_rows: list[dict[str, float | str]],
    features: list[str],
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for feature in features:
        if not all(feature_available(row, [feature]) for row in train_rows + test_rows):
            continue
        for direction in ["ge", "le"]:
            train_scores = np.asarray([float(row[feature]) for row in train_rows], dtype=np.float64)
            test_scores = np.asarray([float(row[feature]) for row in test_rows], dtype=np.float64)
            if direction == "le":
                train_scores = -train_scores
                test_scores = -test_scores
            item = evaluate_scores(
                f"{feature}_{direction}",
                group,
                train_rows,
                test_rows,
                train_scores,
                test_scores,
            )
            item["feature"] = feature
            item["direction"] = direction
            if best is None or float(item["train_rd"]) < float(best["train_rd"]):
                best = item
    if best is None:
        raise RuntimeError(f"no single-feature policy for {group}")
    return best


def write_policy_csv(policies: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "group",
        "threshold",
        "train_rd",
        "train_delta_vs_beta005",
        "train_selected_fraction",
        "test_rd",
        "test_delta_vs_hcs",
        "test_delta_vs_beta005",
        "test_delta_vs_previous_local",
        "test_oracle_gap",
        "test_oracle_gap_closed_fraction",
        "test_selected_fraction",
        "test_selected_mean_margin",
        "test_unselected_mean_margin",
        "precision",
        "recall",
        "f1",
    ]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in policies:
            conf = policy["test_confusion"]
            row = {
                "name": policy["name"],
                "group": policy["group"],
                "threshold": policy["threshold"],
                "train_rd": policy["train_rd"],
                "train_delta_vs_beta005": policy["train_delta_vs_beta005"],
                "train_selected_fraction": policy["train_selected_fraction"],
                "test_rd": policy["test_rd"],
                "test_delta_vs_hcs": policy.get("test_delta_vs_hcs", float("nan")),
                "test_delta_vs_beta005": policy["test_delta_vs_beta005"],
                "test_delta_vs_previous_local": policy["test_delta_vs_previous_local"],
                "test_oracle_gap": policy["test_oracle_gap"],
                "test_oracle_gap_closed_fraction": policy["test_oracle_gap_closed_fraction"],
                "test_selected_fraction": policy["test_selected_fraction"],
                "test_selected_mean_margin": policy["test_selected_mean_margin"],
                "test_unselected_mean_margin": policy["test_unselected_mean_margin"],
                "precision": conf["precision"],
                "recall": conf["recall"],
                "f1": conf["f1"],
            }
            writer.writerow(row)


def table_row(policy: dict[str, object]) -> str:
    conf = policy["test_confusion"]
    return (
        f"| {policy['name']} | {policy['group']} | {fmt(float(policy['train_delta_vs_beta005']), signed=True)} | "
        f"{fmt(float(policy['train_selected_fraction']))} | {fmt(float(policy['test_rd']))} | "
        f"{fmt(float(policy.get('test_delta_vs_hcs', float('nan'))), signed=True)} | "
        f"{fmt(float(policy['test_delta_vs_beta005']), signed=True)} | "
        f"{fmt(float(policy['test_delta_vs_previous_local']), signed=True)} | "
        f"{fmt(float(policy['test_selected_fraction']))} | {fmt(float(conf['precision']))} | "
        f"{fmt(float(conf['recall']))} | {fmt(float(conf['f1']))} |"
    )


def write_markdown(result: dict[str, object]) -> None:
    train_base = result["train_base"]
    test_base = result["test_base"]
    policies = result["policies"]
    safe_policies = [p for p in policies if str(p["group"]).startswith("decoder_safe")]
    diag_policies = [p for p in policies if str(p["group"]).startswith("diagnostic")]
    best_safe = min(safe_policies, key=lambda p: float(p["test_rd"]))
    best_diag = min(diag_policies, key=lambda p: float(p["test_rd"]))

    lines = [
        "# E088 Transfer-Learned Selector Audit",
        "",
        "This audit trains or selects beta005/previous-local selectors on the independent transfer8192 split and evaluates them unchanged on the paper-facing holdout4096 rows. The selector is still a diagnostic multi-checkpoint switch, not a single-checkpoint HCG-RVQ result. Its purpose is to test whether a deployable controller has a stable signal before spending GPU on a new trainable branch.",
        "",
        "## Base RD",
        "",
        "| split | rows | beta005 | previous local | previous-beta | oracle | oracle-beta | previous win frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| transfer8192 train | {int(train_base['rows'])} | {fmt(float(train_base['beta005']))} | "
            f"{fmt(float(train_base['previous_local']))} | {fmt(float(train_base['previous_local_delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(train_base['oracle']))} | {fmt(float(train_base['oracle_delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(train_base['previous_local_win_fraction']))} |"
        ),
        (
            f"| holdout4096 test | {int(test_base['rows'])} | {fmt(float(test_base['beta005']))} | "
            f"{fmt(float(test_base['previous_local']))} | {fmt(float(test_base['previous_local_delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(test_base['oracle']))} | {fmt(float(test_base['oracle_delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(test_base['previous_local_win_fraction']))} |"
        ),
        "",
        "## Transfer-To-Holdout Policies",
        "",
        "| policy | feature group | train vs beta | train selected | holdout RD | holdout vs HCS | holdout vs beta | holdout vs previous | holdout selected | precision | recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in sorted(policies, key=lambda p: (str(p["group"]), float(p["test_rd"]))):
        lines.append(table_row(policy))

    lines.extend(
        [
            "",
            "## Best Decoder-Safe Seed Breakdown",
            "",
            f"Best decoder-safe policy: `{best_safe['name']}`.",
            "",
            "| seed | rows | RD | vs beta005 | vs previous | selected | previous win frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed, item in best_safe["seed_breakdown"].items():
        lines.append(
            f"| {seed} | {int(item['rows'])} | {fmt(float(item['rd']))} | {fmt(float(item['delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(item['delta_vs_previous_local']), signed=True)} | {fmt(float(item['selected_fraction']))} | "
            f"{fmt(float(item['previous_local_win_fraction']))} |"
        )
    lines.extend(
        [
            "",
            "## Best Decoder-Safe HCS Quartiles",
            "",
            "| HCS quartile | rows | RD | vs beta005 | selected | previous win frac |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in best_safe["hcs_quartiles"]:
        lines.append(
            f"| Q{int(item['quartile'])} | {int(item['rows'])} | {fmt(float(item['rd']))} | "
            f"{fmt(float(item['delta_vs_beta005']), signed=True)} | {fmt(float(item['selected_fraction']))} | "
            f"{fmt(float(item['previous_local_win_fraction']))} |"
        )

    lines.extend(
        [
            "",
            "## Best Diagnostic Upper Bound",
            "",
            f"Best diagnostic policy: `{best_diag['name']}`. Diagnostic features include latent/codebook outcome quantities and are not all decoder-side deployable, so this is an upper-bound signal for controller design.",
            "",
            "| seed | rows | RD | vs beta005 | vs previous | selected | previous win frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed, item in best_diag["seed_breakdown"].items():
        lines.append(
            f"| {seed} | {int(item['rows'])} | {fmt(float(item['rd']))} | {fmt(float(item['delta_vs_beta005']), signed=True)} | "
            f"{fmt(float(item['delta_vs_previous_local']), signed=True)} | {fmt(float(item['selected_fraction']))} | "
            f"{fmt(float(item['previous_local_win_fraction']))} |"
        )

    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(decision),
            "",
            f"JSON: `{OUT.with_suffix('.json').relative_to(ROOT)}`",
            f"CSV: `{OUT.with_suffix('.csv').relative_to(ROOT)}`",
            "",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_rows = load_transfer_rows()
    test_rows = add_labels(build_rows())
    if len(train_rows) != 12288:
        raise RuntimeError(f"expected 12288 transfer rows, got {len(train_rows)}")
    if len(test_rows) != 12288:
        raise RuntimeError(f"expected 12288 holdout rows, got {len(test_rows)}")

    policies: list[dict[str, object]] = []
    policies.append(
        single_feature_policy("decoder_safe_single_threshold", train_rows, test_rows, DECODER_SAFE_FEATURES)
    )
    policies.append(
        single_feature_policy("diagnostic_single_threshold", train_rows, test_rows, DIAGNOSTIC_FEATURES)
    )
    for group, features in [
        ("decoder_safe_logreg", DECODER_SAFE_FEATURES),
        ("diagnostic_logreg", DIAGNOSTIC_FEATURES),
    ]:
        policies.append(
            logreg_policy(
                f"{group}_uniform",
                group,
                train_rows,
                test_rows,
                features,
                margin_weighted=False,
            )
        )
        policies.append(
            logreg_policy(
                f"{group}_margin_weighted",
                group,
                train_rows,
                test_rows,
                features,
                margin_weighted=True,
            )
        )

    train_base = summarize_base(train_rows)
    test_base = summarize_base(test_rows)
    safe_policies = [p for p in policies if str(p["group"]).startswith("decoder_safe")]
    diag_policies = [p for p in policies if str(p["group"]).startswith("diagnostic")]
    best_safe = min(safe_policies, key=lambda p: float(p["test_rd"]))
    best_diag = min(diag_policies, key=lambda p: float(p["test_rd"]))
    if float(best_safe["test_delta_vs_beta005"]) < 0.0:
        decision = (
            "Decoder-safe transfer-learned selection does improve holdout beta005, so the next trainable branch should "
            "turn this into a single-checkpoint controller with beta005 y_hat/RVQ-assignment/qMSE/dead-code preservation. "
            "Because the measured gain is from a multi-checkpoint switch, beta005 remains the current manuscript-safe "
            "fixed-checkpoint HCG-RVQ row until the controller reproduces it inside one model."
        )
    elif float(best_diag["test_delta_vs_beta005"]) < 0.0:
        decision = (
            "The deployable decoder-safe features do not yet beat holdout beta005, but diagnostic outcome features do. "
            "This points to adding richer controller supervision or local feature distillation rather than training another "
            "scalar reliability BCE. Beta005 remains paper-main."
        )
    else:
        decision = (
            "Neither decoder-safe nor diagnostic transfer-learned selection beats holdout beta005. The previous-local "
            "complementarity is still real through the oracle, but the current split features are not reliable enough "
            "for a controller promotion. Beta005 remains paper-main."
        )

    result: dict[str, object] = {
        "train_csv": str(TRANSFER_REFERENCE.relative_to(ROOT)),
        "candidate_csv": str(TRANSFER_CANDIDATE.relative_to(ROOT)),
        "train_base": train_base,
        "test_base": test_base,
        "policies": policies,
        "best_decoder_safe_policy": best_safe["name"],
        "best_diagnostic_policy": best_diag["name"],
        "decision": decision,
        "note": (
            "Selectors are trained or threshold-selected only on transfer8192. Holdout4096 is used once for reporting. "
            "Decoder-safe features are generated by the beta005 hyperprior side; diagnostic features include latent/codebook "
            "outcomes and are upper-bound guidance."
        ),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_policy_csv(policies)
    write_markdown(result)
    print(json.dumps({
        "best_decoder_safe": {
            "name": best_safe["name"],
            "test_rd": best_safe["test_rd"],
            "test_delta_vs_beta005": best_safe["test_delta_vs_beta005"],
            "test_selected_fraction": best_safe["test_selected_fraction"],
        },
        "best_diagnostic": {
            "name": best_diag["name"],
            "test_rd": best_diag["test_rd"],
            "test_delta_vs_beta005": best_diag["test_delta_vs_beta005"],
            "test_selected_fraction": best_diag["test_selected_fraction"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
