#!/usr/bin/env python3
"""Probe learned image-level no-op fallback from EF-LIC HCG controller features.

E343/E344 showed that decoder-visible summary features can sometimes predict
when the EF-LIC HCG controller should fall back to the original RVQ path.  This
script asks a colder question: does a tiny learned classifier trained only on
the first Kodak split generalize better than single-feature thresholds?

The probe is deliberately simple and diagnostic.  It does not use oracle gains
or controller deltas as input features, and it selects the fallback threshold on
the training split before evaluating on held-out images.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


EXCLUDE_FEATURES = {
    "controller_delta_psnr",
    "controller_tail_unsafe",
    "controller_unsafe",
    "image",
    "run",
    "source_csv",
    "active_slices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e343_eflic_none_oracle_feature_audit_kodak24.joined.csv",
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--tail-floor", type=float, default=-0.02)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--pos-weight", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--tail-weight", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--label-mode",
        nargs="+",
        choices=["negative", "tail"],
        default=["negative", "tail"],
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e345_eflic_noop_linear_probe_kodak16_8",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for raw in csv.DictReader(fobj):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                fval = safe_float(value)
                row[key] = fval if math.isfinite(fval) else value
            rows.append(row)
    return rows


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for key in rows[0]:
        if key in EXCLUDE_FEATURES or key.startswith("oracle_"):
            continue
        values = [row.get(key, float("nan")) for row in rows]
        if all(isinstance(value, float) and math.isfinite(value) for value in values):
            names.append(key)
    return sorted(names)


def split_rows(rows: list[dict[str, Any]], train_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: str(row["image"]))
    return ordered[:train_count], ordered[train_count:]


def labels(rows: list[dict[str, Any]], *, mode: str, tail_floor: float) -> list[int]:
    if mode == "negative":
        return [int(float(row["controller_delta_psnr"]) < 0.0) for row in rows]
    return [int(float(row["controller_delta_psnr"]) < tail_floor) for row in rows]


def matrix(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    return [[float(row[name]) for name in names] for row in rows]


def standardize(
    x_train: list[list[float]], x_eval: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    cols = len(x_train[0])
    means: list[float] = []
    scales: list[float] = []
    for idx in range(cols):
        vals = [row[idx] for row in x_train]
        mval = mean(vals)
        var = mean([(value - mval) ** 2 for value in vals])
        scale = math.sqrt(max(var, 1e-12))
        means.append(mval)
        scales.append(scale)

    def apply(rows: list[list[float]]) -> list[list[float]]:
        return [[(value - means[idx]) / scales[idx] for idx, value in enumerate(row)] for row in rows]

    return apply(x_train), apply(x_eval), means, scales


def sigmoid(value: float) -> float:
    if value >= 0:
        zval = math.exp(-value)
        return 1.0 / (1.0 + zval)
    zval = math.exp(value)
    return zval / (1.0 + zval)


def train_logistic(
    x_train: list[list[float]],
    y_train: list[int],
    *,
    steps: int,
    lr: float,
    l2: float,
    pos_weight: float,
) -> tuple[list[float], float]:
    dims = len(x_train[0])
    weights = [0.0] * dims
    bias = math.log((sum(y_train) + 0.5) / (len(y_train) - sum(y_train) + 0.5))
    for _ in range(steps):
        grad_w = [0.0] * dims
        grad_b = 0.0
        denom = 0.0
        for row, label in zip(x_train, y_train):
            score = bias + sum(w * x for w, x in zip(weights, row))
            prob = sigmoid(score)
            sample_weight = pos_weight if label else 1.0
            diff = (prob - label) * sample_weight
            denom += sample_weight
            grad_b += diff
            for idx, value in enumerate(row):
                grad_w[idx] += diff * value
        denom = max(denom, 1e-12)
        bias -= lr * grad_b / denom
        for idx in range(dims):
            grad = grad_w[idx] / denom + l2 * weights[idx] / max(1, len(y_train))
            weights[idx] -= lr * grad
    return weights, bias


def predict_proba(x_rows: list[list[float]], weights: list[float], bias: float) -> list[float]:
    return [sigmoid(bias + sum(w * x for w, x in zip(weights, row))) for row in x_rows]


def threshold_candidates(values: list[float]) -> list[float]:
    vals = sorted({value for value in values if math.isfinite(value)})
    if not vals:
        return []
    candidates = [vals[0] - 1e-9, vals[-1] + 1e-9]
    candidates.extend(vals)
    candidates.extend((aval + bval) / 2.0 for aval, bval in zip(vals, vals[1:]))
    return sorted(set(candidates))


def evaluate_probs(rows: list[dict[str, Any]], probs: list[float], threshold: float) -> dict[str, Any]:
    deltas: list[float] = []
    suppressed = 0
    suppressed_negative = 0
    suppressed_positive = 0
    for row, prob in zip(rows, probs):
        original = float(row["controller_delta_psnr"])
        if prob >= threshold:
            suppressed += 1
            if original < 0.0:
                suppressed_negative += 1
            elif original > 0.0:
                suppressed_positive += 1
            deltas.append(0.0)
        else:
            deltas.append(original)
    return {
        "threshold": threshold,
        "mean_delta_psnr": mean(deltas),
        "worst_delta_psnr": min(deltas) if deltas else float("nan"),
        "negative_count": sum(1 for value in deltas if value < 0.0),
        "positive_count": sum(1 for value in deltas if value > 0.0),
        "suppressed_count": suppressed,
        "suppressed_negative_count": suppressed_negative,
        "suppressed_positive_count": suppressed_positive,
    }


def raw_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["controller_delta_psnr"]) for row in rows]
    return {
        "records": len(deltas),
        "mean_delta_psnr": mean(deltas),
        "worst_delta_psnr": min(deltas) if deltas else float("nan"),
        "negative_count": sum(1 for value in deltas if value < 0.0),
        "positive_count": sum(1 for value in deltas if value > 0.0),
    }


def score_policy(policy: dict[str, Any], tail_weight: float) -> float:
    return float(policy["mean_delta_psnr"] + tail_weight * min(0.0, policy["worst_delta_psnr"]))


def top_weights(feature_names_: list[str], weights: list[float], count: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(zip(feature_names_, weights), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": feature, "weight": weight} for feature, weight in ranked[:count]]


def probe_run(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    run: str,
) -> dict[str, Any]:
    train_rows, eval_rows = split_rows(rows, args.train_count)
    names = feature_names(rows)
    x_train_raw = matrix(train_rows, names)
    x_eval_raw = matrix(eval_rows, names)
    x_train, x_eval, _, _ = standardize(x_train_raw, x_eval_raw)

    selected_rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for label_mode in args.label_mode:
        y_train = labels(train_rows, mode=label_mode, tail_floor=args.tail_floor)
        y_eval = labels(eval_rows, mode=label_mode, tail_floor=args.tail_floor)
        if not any(y_train) or all(y_train):
            continue
        for l2_value in args.l2:
            for pos_weight in args.pos_weight:
                weights, bias = train_logistic(
                    x_train,
                    y_train,
                    steps=args.steps,
                    lr=args.lr,
                    l2=float(l2_value),
                    pos_weight=float(pos_weight),
                )
                p_train = predict_proba(x_train, weights, bias)
                p_eval = predict_proba(x_eval, weights, bias)
                train_candidates = [
                    evaluate_probs(train_rows, p_train, threshold)
                    for threshold in threshold_candidates(p_train)
                ]
                model_meta = {
                    "run": run,
                    "label_mode": label_mode,
                    "l2": float(l2_value),
                    "pos_weight": float(pos_weight),
                    "train_positive_labels": sum(y_train),
                    "eval_positive_labels": sum(y_eval),
                    "top_weights": top_weights(names, weights),
                }
                models.append(model_meta)
                for tail_weight in args.tail_weight:
                    best = max(train_candidates, key=lambda policy: score_policy(policy, float(tail_weight)))
                    eval_policy = evaluate_probs(eval_rows, p_eval, float(best["threshold"]))
                    selected_rows.append(
                        {
                            **model_meta,
                            "selection": f"tail_weight_{float(tail_weight):g}",
                            "tail_weight": float(tail_weight),
                            **{f"train_{key}": value for key, value in best.items()},
                            **{f"eval_{key}": value for key, value in eval_policy.items()},
                        }
                    )
                safe = [p for p in train_candidates if p["worst_delta_psnr"] >= args.tail_floor]
                if safe:
                    best = max(safe, key=lambda policy: policy["mean_delta_psnr"])
                    eval_policy = evaluate_probs(eval_rows, p_eval, float(best["threshold"]))
                    selected_rows.append(
                        {
                            **model_meta,
                            "selection": f"train_worst_ge_{args.tail_floor:g}",
                            "tail_weight": float("nan"),
                            **{f"train_{key}": value for key, value in best.items()},
                            **{f"eval_{key}": value for key, value in eval_policy.items()},
                        }
                    )

    return {
        "run": run,
        "features": names,
        "raw_train": raw_summary(train_rows),
        "raw_eval": raw_summary(eval_rows),
        "selected": selected_rows,
        "models": models,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if key != "top_weights"})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def best_by_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(row["run"], row["selection"]) for row in rows})
    for run, selection in groups:
        candidates = [row for row in rows if row["run"] == run and row["selection"] == selection]
        if selection.startswith("tail_weight_"):
            best = max(
                candidates,
                key=lambda row: score_policy(
                    {
                        "mean_delta_psnr": float(row["train_mean_delta_psnr"]),
                        "worst_delta_psnr": float(row["train_worst_delta_psnr"]),
                    },
                    float(row["tail_weight"]),
                ),
            )
        else:
            best = max(candidates, key=lambda row: float(row["train_mean_delta_psnr"]))
        out.append(best)
    return out


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_rows = read_rows(args.joined_csv)
    payload: dict[str, Any] = {
        "experiment": "E345 EF-LIC learned no-op linear probe",
        "purpose": (
            "Train tiny image-level no-op classifiers on decoder-visible controller features "
            "and evaluate split generalization."
        ),
        "args": {
            "joined_csv": str(args.joined_csv),
            "train_count": args.train_count,
            "tail_floor": args.tail_floor,
            "steps": args.steps,
            "lr": args.lr,
            "l2": [float(v) for v in args.l2],
            "pos_weight": [float(v) for v in args.pos_weight],
            "tail_weight": [float(v) for v in args.tail_weight],
            "label_mode": list(args.label_mode),
        },
        "runs": {},
    }
    all_selected: list[dict[str, Any]] = []
    for run in sorted({str(row["run"]) for row in all_rows}):
        rows = [row for row in all_rows if str(row["run"]) == run]
        result = probe_run(rows, args=args, run=run)
        payload["runs"][run] = result
        all_selected.extend(result["selected"])

    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(csv_path, all_selected)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows = best_by_selection(all_selected)
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E345 EF-LIC Learned No-op Linear Probe\n\n")
        fobj.write(
            "A small logistic fallback classifier is trained on the first Kodak split "
            "from decoder-visible controller features only.  The fallback threshold is "
            "selected on the same train split and evaluated on the held-out Kodak split.\n\n"
        )
        fobj.write("## Raw Eval Baselines\n\n")
        fobj.write("| run | train mean | train worst | eval mean | eval worst | eval neg |\n")
        fobj.write("|---|---:|---:|---:|---:|---:|\n")
        for run, result in payload["runs"].items():
            train = result["raw_train"]
            eval_ = result["raw_eval"]
            fobj.write(
                f"| {run} | {train['mean_delta_psnr']:+.6f} | {train['worst_delta_psnr']:+.6f} | "
                f"{eval_['mean_delta_psnr']:+.6f} | {eval_['worst_delta_psnr']:+.6f} | "
                f"{eval_['negative_count']} |\n"
            )
        fobj.write("\n## Best Train-Selected Linear Policies\n\n")
        fobj.write(
            "| run | selection | label | l2 | pos_w | train mean | train worst | "
            "eval mean | eval worst | eval neg | eval suppressed |\n"
        )
        fobj.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            fobj.write(
                f"| {row['run']} | {row['selection']} | {row['label_mode']} | "
                f"{float(row['l2']):g} | {float(row['pos_weight']):g} | "
                f"{float(row['train_mean_delta_psnr']):+.6f} | "
                f"{float(row['train_worst_delta_psnr']):+.6f} | "
                f"{float(row['eval_mean_delta_psnr']):+.6f} | "
                f"{float(row['eval_worst_delta_psnr']):+.6f} | "
                f"{int(row['eval_negative_count'])} | {int(row['eval_suppressed_count'])} |\n"
            )
        fobj.write("\nInterpretation:\n\n")
        fobj.write(
            "- If this probe cannot beat the raw controller or E344's scalar threshold on the held-out split, "
            "a naive global image-level no-op head is not worth promoting.\n"
        )
        fobj.write(
            "- A useful next controller should instead train a local/sequential no-op head from independent "
            "codec-gain labels, while preserving EF-LIC's original RVQ path as exact fallback.\n"
        )
    print(f"wrote {csv_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
