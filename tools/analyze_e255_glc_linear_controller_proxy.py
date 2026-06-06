#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FEATURE_SETS = {
    "branch_internal": [
        "active_mse_ratio",
        "active_scalar_mse",
        "active_rvq_mse",
        "index_entropy_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
        "base_bpp",
    ],
    "rate_proxy": [
        "empirical_bpp_delta",
        "fixed_bpp_delta",
        "base_bpp",
    ],
    "branch_plus_rate": [
        "active_mse_ratio",
        "active_scalar_mse",
        "active_rvq_mse",
        "index_entropy_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
        "empirical_bpp_delta",
        "fixed_bpp_delta",
        "base_bpp",
    ],
    "analysis_upper": [
        "active_mse_ratio",
        "active_scalar_mse",
        "active_rvq_mse",
        "index_entropy_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
        "empirical_bpp_delta",
        "fixed_bpp_delta",
        "base_bpp",
        "base_psnr",
        "base_ms_ssim",
        "base_lpips",
        "base_dists",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e254_glc_domain_mixed_gate_readiness.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e255_glc_linear_controller_proxy",
    )
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=0.001)
    return p.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["score_with_side"] = finite(row.get("score_with_side"))
            row["score"] = finite(row.get("score"))
            row["oracle_select"] = str(row.get("oracle_select", "")).lower() == "true"
            for features in FEATURE_SETS.values():
                for feature in features:
                    row[feature] = finite(row.get(feature))
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def standardize(train: list[dict[str, Any]], rows: list[dict[str, Any]], features: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    means = [mean([finite(row.get(feature), 0.0) for row in train]) for feature in features]
    stds = []
    for idx, feature in enumerate(features):
        vals = [finite(row.get(feature), means[idx]) for row in train]
        var = mean([(v - means[idx]) ** 2 for v in vals])
        stds.append(math.sqrt(var) if math.isfinite(var) and var > 1e-12 else 1.0)

    def transform(items: list[dict[str, Any]]) -> list[list[float]]:
        matrix = []
        for row in items:
            matrix.append([(finite(row.get(feature), means[i]) - means[i]) / stds[i] for i, feature in enumerate(features)])
        return matrix

    return transform(train), transform(rows)


def train_logistic(
    x: list[list[float]],
    y: list[float],
    steps: int,
    lr: float,
    l2: float,
) -> tuple[list[float], float]:
    if not x:
        raise ValueError("empty train matrix")
    dim = len(x[0])
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        bias = 8.0 if pos > 0 else -8.0
        return [0.0] * dim, bias

    w = [0.0] * dim
    b = math.log((pos + 0.5) / (neg + 0.5))
    pos_weight = neg / pos
    for _ in range(steps):
        grad_w = [l2 * wi for wi in w]
        grad_b = 0.0
        total_weight = 0.0
        for xi, yi in zip(x, y):
            pred = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b)
            weight = pos_weight if yi > 0.5 else 1.0
            err = (pred - yi) * weight
            total_weight += weight
            for j, xj in enumerate(xi):
                grad_w[j] += err * xj
            grad_b += err
        scale = 1.0 / max(total_weight, 1.0)
        for j in range(dim):
            w[j] -= lr * grad_w[j] * scale
        b -= lr * grad_b * scale
    return w, b


def train_linear_score(
    x: list[list[float]],
    y: list[float],
    steps: int,
    lr: float,
    l2: float,
) -> tuple[list[float], float]:
    if not x:
        raise ValueError("empty train matrix")
    dim = len(x[0])
    w = [0.0] * dim
    b = mean(y)
    for _ in range(steps):
        grad_w = [l2 * wi for wi in w]
        grad_b = 0.0
        for xi, yi in zip(x, y):
            pred = sum(wj * xj for wj, xj in zip(w, xi)) + b
            err = pred - yi
            for j, xj in enumerate(xi):
                grad_w[j] += err * xj
            grad_b += err
        scale = 1.0 / len(x)
        for j in range(dim):
            w[j] -= lr * grad_w[j] * scale
        b -= lr * grad_b * scale
    return w, b


def predict_linear(x: list[list[float]], w: list[float], b: float) -> list[float]:
    return [sum(wj * xj for wj, xj in zip(w, xi)) + b for xi in x]


def threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted({v for v in values if math.isfinite(v)})
    if not unique:
        return [0.0]
    points = [unique[0] - 1e-12, unique[-1] + 1e-12]
    points.extend(unique)
    points.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    return sorted(set(points))


def summarize_policy(rows: list[dict[str, Any]], selected: list[bool], name: str) -> dict[str, Any]:
    return {
        "policy": name,
        "selected": sum(1 for v in selected if v),
        "total": len(rows),
        "selected_frac": sum(1 for v in selected if v) / len(rows) if rows else 0.0,
        "mean_score": mean([finite(row["score_with_side"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_delta_psnr": mean([finite(row["delta_psnr"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_delta_ms_ssim": mean([finite(row["delta_ms_ssim"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_delta_lpips": mean([finite(row["delta_lpips"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_delta_dists": mean([finite(row["delta_dists"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_delta_bpp": mean([finite(row["delta_bpp"]) if use else 0.0 for row, use in zip(rows, selected)]),
    }


def choose_threshold(rows: list[dict[str, Any]], values: list[float], select_high: bool) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in threshold_candidates(values):
        selected = [value >= threshold if select_high else value <= threshold for value in values]
        summary = summarize_policy(rows, selected, "train_threshold")
        candidate = {
            "threshold": threshold,
            "select_high": select_high,
            "train_score": summary["mean_score"],
            "train_selected": summary["selected"],
        }
        if best is None or candidate["train_score"] < best["train_score"]:
            best = candidate
    if best is None:
        raise ValueError("no threshold")
    return best


def fit_predict(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    features: list[str],
    model_name: str,
    steps: int,
    lr: float,
    l2: float,
) -> tuple[list[bool], dict[str, Any]]:
    train_x, test_x = standardize(train, test, features)
    if model_name == "logistic":
        y = [1.0 if row["oracle_select"] else 0.0 for row in train]
        w, b = train_logistic(train_x, y, steps, lr, l2)
        train_values = [sigmoid(v) for v in predict_linear(train_x, w, b)]
        test_values = [sigmoid(v) for v in predict_linear(test_x, w, b)]
        threshold = choose_threshold(train, train_values, select_high=True)
    elif model_name == "score_regressor":
        y = [finite(row["score_with_side"], 0.0) for row in train]
        w, b = train_linear_score(train_x, y, steps, lr, l2)
        train_values = predict_linear(train_x, w, b)
        test_values = predict_linear(test_x, w, b)
        threshold = choose_threshold(train, train_values, select_high=False)
    else:
        raise ValueError(model_name)
    selected = [
        value >= threshold["threshold"] if threshold["select_high"] else value <= threshold["threshold"]
        for value in test_values
    ]
    model_info = {
        "model": model_name,
        "features": features,
        "threshold": threshold["threshold"],
        "select_high": threshold["select_high"],
        "train_score": threshold["train_score"],
        "train_selected": threshold["train_selected"],
        "weights": w,
        "bias": b,
    }
    return selected, model_info


def eval_resub(rows: list[dict[str, Any]], feature_set: str, model_name: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected, info = fit_predict(rows, rows, FEATURE_SETS[feature_set], model_name, args.steps, args.lr, args.l2)
    summary = summarize_policy(rows, selected, f"resub_{feature_set}_{model_name}")
    summary.update({"feature_set": feature_set, "model": model_name, "protocol": "resub"})
    return summary, [dict(info, protocol="resub", heldout="all")]


def eval_leave_one_out(rows: list[dict[str, Any]], feature_set: str, model_name: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_all = [False] * len(rows)
    infos: list[dict[str, Any]] = []
    for idx in range(len(rows)):
        train = [row for j, row in enumerate(rows) if j != idx]
        test = [rows[idx]]
        selected, info = fit_predict(train, test, FEATURE_SETS[feature_set], model_name, args.steps, args.lr, args.l2)
        selected_all[idx] = selected[0]
        infos.append(dict(info, protocol="loocv", heldout=rows[idx]["image"]))
    summary = summarize_policy(rows, selected_all, f"loocv_{feature_set}_{model_name}")
    summary.update({"feature_set": feature_set, "model": model_name, "protocol": "loocv"})
    return summary, infos


def eval_leave_group_out(rows: list[dict[str, Any]], group_key: str, feature_set: str, model_name: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_all = [False] * len(rows)
    infos: list[dict[str, Any]] = []
    for group in sorted({str(row[group_key]) for row in rows}):
        train = [row for row in rows if str(row[group_key]) != group]
        test_indexes = [idx for idx, row in enumerate(rows) if str(row[group_key]) == group]
        test = [rows[idx] for idx in test_indexes]
        selected, info = fit_predict(train, test, FEATURE_SETS[feature_set], model_name, args.steps, args.lr, args.l2)
        for idx, use in zip(test_indexes, selected):
            selected_all[idx] = use
        fold_summary = summarize_policy(test, selected, f"heldout_{group}_{feature_set}_{model_name}")
        infos.append(dict(info, protocol=f"leave_{group_key}_out", heldout=group, fold_score=fold_summary["mean_score"], fold_selected=fold_summary["selected"], fold_total=fold_summary["total"]))
    summary = summarize_policy(rows, selected_all, f"leave_{group_key}_out_{feature_set}_{model_name}")
    summary.update({"feature_set": feature_set, "model": model_name, "protocol": f"leave_{group_key}_out"})
    return summary, infos


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = [
        summarize_policy(rows, [False] * len(rows), "no_branch"),
        summarize_policy(rows, [True] * len(rows), "all_on"),
        summarize_policy(rows, [bool(row["oracle_select"]) for row in rows], "oracle"),
    ]
    for summary in summaries:
        summary.setdefault("feature_set", "none")
        summary.setdefault("model", "none")
        summary.setdefault("protocol", "reference")

    model_infos: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for model_name in ("logistic", "score_regressor"):
            for evaluator in (eval_resub, eval_leave_one_out):
                summary, infos = evaluator(rows, feature_set, model_name, args)
                summaries.append(summary)
                model_infos.extend(infos)
            summary, infos = eval_leave_group_out(rows, "domain", feature_set, model_name, args)
            summaries.append(summary)
            model_infos.extend(infos)
            summary, infos = eval_leave_group_out(rows, "variant", feature_set, model_name, args)
            summaries.append(summary)
            model_infos.extend(infos)

    write_csv(args.output_prefix.with_suffix(".summary.csv"), summaries)
    write_csv(args.output_prefix.with_suffix(".model_info.csv"), model_infos)

    payload = {
        "experiment": "E255 GLC linear controller proxy",
        "input_rows": str(args.rows),
        "feature_sets": FEATURE_SETS,
        "summaries": summaries,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best = sorted(summaries, key=lambda row: finite(row["mean_score"]))[:10]
    leave_domain = [row for row in summaries if row.get("protocol") == "leave_domain_out"]
    lines = [
        "# E255 GLC Linear Controller Proxy",
        "",
        "This audit tests whether the E254 local-RVQ activation decision can be approximated by a tiny linear/logistic controller before changing the GLC codec loop.",
        "",
        "## Best Policies",
        "",
        "| policy | protocol | feature set | model | selected | mean score | dPSNR | dLPIPS | dDISTS | dbpp |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['policy']} | {row.get('protocol', '')} | {row.get('feature_set', '')} | {row.get('model', '')} | "
            f"{row['selected']}/{row['total']} | {row['mean_score']:+.6f} | {row['mean_delta_psnr']:+.6f} | "
            f"{row['mean_delta_lpips']:+.6f} | {row['mean_delta_dists']:+.6f} | {row['mean_delta_bpp']:+.6f} |"
        )
    lines += [
        "",
        "## Leave-Domain-Out Policies",
        "",
        "| policy | feature set | model | selected | mean score | dPSNR | dLPIPS | dDISTS | dbpp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(leave_domain, key=lambda item: (item["feature_set"], item["model"])):
        lines.append(
            f"| {row['policy']} | {row['feature_set']} | {row['model']} | {row['selected']}/{row['total']} | "
            f"{row['mean_score']:+.6f} | {row['mean_delta_psnr']:+.6f} | {row['mean_delta_lpips']:+.6f} | "
            f"{row['mean_delta_dists']:+.6f} | {row['mean_delta_bpp']:+.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A tiny controller is useful only if it beats no-branch under held-out protocols, not only under resubstitution. "
        "If resubstitution or LOOCV improves but leave-domain-out fails, the next implementation should gather domain-mixed calibration labels or train the controller inside the codec loop with a DISTS/bpp guard before any expensive full-training claim.",
        "",
    ]
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_prefix": str(args.output_prefix), "best": best[:3]}, indent=2))


if __name__ == "__main__":
    main()
