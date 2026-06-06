#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


DEFAULT_INPUTS = [
    ("clic_professional24", "experiments/analysis/e212_eflic_clic_professional24_mean_alpha_sweep_active.csv"),
    ("tecnick24", "experiments/analysis/e212_eflic_tecnick_b01r01_24_mean_alpha_sweep_active.csv"),
    ("clic_mobile24", "experiments/analysis/e215_eflic_clic_mobile24_mean_alpha_sweep_active.csv"),
    ("openimages24576_eval32", "experiments/analysis/e215_eflic_openimages24576_eval32_mean_alpha_sweep_active.csv"),
    ("div2k_valid24", "experiments/analysis/e215_eflic_div2k_valid24_mean_alpha_sweep_active.csv"),
]

FEATURE_PREFIXES = ("z_hat_", "z_index_", "slice0_mean_", "slice0_scale_")
DROP_FEATURE_SUFFIXES = ("_finite_frac",)
ALPHA_ZERO = 0.0
N_METHODS = 5
MAX_STUMP_THRESHOLDS = 15


@dataclass(frozen=True)
class Candidate:
    dataset: str
    image: str
    alpha: float
    delta_dists: float
    delta_lpips: float
    delta_psnr: float
    score: float
    features: dict[str, float]


def parse_float(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if value == "":
        return float("nan")
    return float(value)


def is_feature(name: str) -> bool:
    if not name.startswith(FEATURE_PREFIXES):
        return False
    if name.endswith(DROP_FEATURE_SUFFIXES):
        return False
    return True


def read_candidates(dataset: str, path: Path, dists_weight: float, lpips_weight: float) -> dict[str, list[Candidate]]:
    rows = list(csv.DictReader(path.open()))
    by_image: dict[str, list[Candidate]] = {}
    for row in rows:
        if int(parse_float(row.get("nonfinite", 0.0))) != 0:
            continue
        if abs(parse_float(row.get("max_decode_diff", 0.0))) > 1e-12:
            continue
        features = {k: parse_float(v) for k, v in row.items() if is_feature(k)}
        delta_dists = parse_float(row["delta_dists"])
        delta_lpips = parse_float(row["delta_lpips"])
        delta_psnr = parse_float(row["delta_psnr"])
        score = dists_weight * delta_dists + lpips_weight * delta_lpips
        c = Candidate(
            dataset=dataset,
            image=row["image"],
            alpha=parse_float(row["alpha"]),
            delta_dists=delta_dists,
            delta_lpips=delta_lpips,
            delta_psnr=delta_psnr,
            score=score,
            features=features,
        )
        by_image.setdefault(row["image"], []).append(c)

    for image, candidates in list(by_image.items()):
        if not candidates:
            del by_image[image]
            continue
        base = candidates[0]
        by_image[image].append(
            Candidate(
                dataset=dataset,
                image=image,
                alpha=ALPHA_ZERO,
                delta_dists=0.0,
                delta_lpips=0.0,
                delta_psnr=0.0,
                score=0.0,
                features=base.features,
            )
        )
        by_image[image] = sorted(by_image[image], key=lambda c: c.alpha)
    return by_image


def parse_input_specs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_INPUTS
    specs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--input must be name=path, got: {value}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit(f"--input must be name=path, got: {value}")
        specs.append((name, path))
    return specs


def common_features(groups: Iterable[dict[str, list[Candidate]]]) -> list[str]:
    names: set[str] | None = None
    for group in groups:
        for candidates in group.values():
            finite = {k for k, v in candidates[0].features.items() if math.isfinite(v)}
            names = finite if names is None else names & finite
    return sorted(names or [])


def best_candidate(candidates: list[Candidate]) -> Candidate:
    return min(candidates, key=lambda c: (c.score, c.delta_dists, c.delta_lpips, c.alpha))


def get_candidate(candidates: list[Candidate], alpha: float) -> Candidate:
    for c in candidates:
        if abs(c.alpha - alpha) < 1e-12:
            return c
    raise KeyError(alpha)


def common_alphas(items: list[tuple[str, list[Candidate]]]) -> list[float]:
    alphas: set[float] | None = None
    for _, candidates in items:
        values = {c.alpha for c in candidates}
        alphas = values if alphas is None else alphas & values
    return sorted(alphas or {ALPHA_ZERO})


def best_fixed_alpha_for_items(items: list[tuple[str, list[Candidate]]], alphas: list[float]) -> float:
    if not items:
        return ALPHA_ZERO
    return min(alphas, key=lambda a: mean(get_candidate(candidates, a).score for _, candidates in items))


def matrix(items: list[tuple[str, list[Candidate]]], features: list[str]) -> np.ndarray:
    return np.array([[candidates[0].features[f] for f in features] for _, candidates in items], dtype=np.float64)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def predict_fixed_alpha(train: list[tuple[str, list[Candidate]]], test: list[tuple[str, list[Candidate]]]) -> list[float]:
    alphas = common_alphas(train + test)
    best_alpha = best_fixed_alpha_for_items(train, alphas)
    return [best_alpha for _ in test]


def predict_centroid(
    train: list[tuple[str, list[Candidate]]],
    test: list[tuple[str, list[Candidate]]],
    features: list[str],
) -> list[float]:
    if not train:
        return [ALPHA_ZERO for _ in test]
    train_x = matrix(train, features)
    test_x = matrix(test, features)
    train_x, test_x = standardize(train_x, test_x)
    labels = np.array([best_candidate(candidates).alpha for _, candidates in train], dtype=np.float64)
    centroids: dict[float, np.ndarray] = {}
    for alpha in sorted(set(labels.tolist())):
        centroids[alpha] = train_x[labels == alpha].mean(axis=0)
    preds = []
    for x in test_x:
        alpha = min(centroids, key=lambda a: float(np.sum((x - centroids[a]) ** 2)))
        preds.append(float(alpha))
    return preds


def predict_stump_score(
    train: list[tuple[str, list[Candidate]]],
    test: list[tuple[str, list[Candidate]]],
    features: list[str],
) -> list[float]:
    if len(train) < 2:
        return predict_fixed_alpha(train, test)
    train_x = matrix(train, features)
    test_x = matrix(test, features)
    train_x, test_x = standardize(train_x, test_x)
    alphas = common_alphas(train + test)
    global_alpha = best_fixed_alpha_for_items(train, alphas)
    best_loss = mean(get_candidate(candidates, global_alpha).score for _, candidates in train)
    best_rule: tuple[int, float, float, float] | None = None

    for feature_idx in range(train_x.shape[1]):
        values = sorted(set(float(v) for v in train_x[:, feature_idx]))
        if len(values) > MAX_STUMP_THRESHOLDS + 1:
            raw_thresholds = np.quantile(train_x[:, feature_idx], np.linspace(0.05, 0.95, MAX_STUMP_THRESHOLDS))
            thresholds = sorted(set(float(v) for v in raw_thresholds))
        else:
            thresholds = [(a + b) * 0.5 for a, b in zip(values, values[1:])]
        for threshold in thresholds:
            left = [(key, candidates) for (key, candidates), x in zip(train, train_x) if x[feature_idx] <= threshold]
            right = [(key, candidates) for (key, candidates), x in zip(train, train_x) if x[feature_idx] > threshold]
            if not left or not right:
                continue
            left_alpha = best_fixed_alpha_for_items(left, alphas)
            right_alpha = best_fixed_alpha_for_items(right, alphas)
            loss = mean(
                get_candidate(candidates, left_alpha if x[feature_idx] <= threshold else right_alpha).score
                for (_, candidates), x in zip(train, train_x)
            )
            if loss < best_loss:
                best_loss = loss
                best_rule = (feature_idx, threshold, left_alpha, right_alpha)

    if best_rule is None:
        return [global_alpha for _ in test]
    feature_idx, threshold, left_alpha, right_alpha = best_rule
    return [left_alpha if x[feature_idx] <= threshold else right_alpha for x in test_x]


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    reg = np.eye(x_aug.shape[1]) * ridge
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x_aug @ w


def predict_ridge_score(
    train: list[tuple[str, list[Candidate]]],
    test: list[tuple[str, list[Candidate]]],
    features: list[str],
    ridge: float,
) -> list[float]:
    if not train:
        return [ALPHA_ZERO for _ in test]
    train_x = matrix(train, features)
    test_x = matrix(test, features)
    train_x, test_x = standardize(train_x, test_x)
    alphas = common_alphas(train + test)
    weights: dict[float, np.ndarray] = {}
    for alpha in alphas:
        y = np.array([get_candidate(candidates, alpha).score for _, candidates in train], dtype=np.float64)
        weights[alpha] = ridge_fit(train_x, y, ridge)
    pred_scores = {alpha: ridge_predict(test_x, weights[alpha]) for alpha in alphas}
    preds = []
    for i in range(len(test)):
        preds.append(min(alphas, key=lambda a: float(pred_scores[a][i])))
    return preds


def summarize_selection(
    protocol: str,
    method: str,
    items: list[tuple[str, list[Candidate]]],
    preds: list[float],
) -> dict[str, float | str]:
    selected = [get_candidate(candidates, alpha) for (_, candidates), alpha in zip(items, preds)]
    oracle = [best_candidate(candidates) for _, candidates in items]
    return {
        "protocol": protocol,
        "method": method,
        "n": len(items),
        "delta_dists_mean": mean(c.delta_dists for c in selected),
        "delta_lpips_mean": mean(c.delta_lpips for c in selected),
        "delta_psnr_mean": mean(c.delta_psnr for c in selected),
        "score_mean": mean(c.score for c in selected),
        "active_frac": mean(float(c.alpha > 0.0) for c in selected),
        "mean_alpha": mean(c.alpha for c in selected),
        "dists_wins": sum(c.delta_dists < 0 for c in selected),
        "lpips_wins": sum(c.delta_lpips < 0 for c in selected),
        "both_wins": sum(c.delta_dists < 0 and c.delta_lpips < 0 for c in selected),
        "oracle_score_mean": mean(c.score for c in oracle),
        "oracle_gap_score": mean(c.score for c in selected) - mean(c.score for c in oracle),
    }


def make_items(group: dict[str, list[Candidate]]) -> list[tuple[str, list[Candidate]]]:
    return sorted(group.items())


def merge_groups(named_groups: Iterable[tuple[str, dict[str, list[Candidate]]]]) -> dict[str, list[Candidate]]:
    merged: dict[str, list[Candidate]] = {}
    for name, group in named_groups:
        for image, candidates in group.items():
            merged[f"{name}/{image}"] = candidates
    return merged


def loocv(
    group: dict[str, list[Candidate]],
    features: list[str],
    ridge: float,
    protocol: str = "loocv",
) -> list[dict[str, float | str]]:
    items = make_items(group)
    protocols: dict[str, list[float]] = {"fixed_alpha": [], "centroid": [], "stump_score": [], "ridge_score": [], "oracle": []}
    for i, item in enumerate(items):
        train = items[:i] + items[i + 1 :]
        test = [item]
        protocols["fixed_alpha"].extend(predict_fixed_alpha(train, test))
        protocols["centroid"].extend(predict_centroid(train, test, features))
        protocols["stump_score"].extend(predict_stump_score(train, test, features))
        protocols["ridge_score"].extend(predict_ridge_score(train, test, features, ridge))
        protocols["oracle"].append(best_candidate(item[1]).alpha)
    return [summarize_selection(protocol, method, items, preds) for method, preds in protocols.items()]


def train_eval(
    train_group: dict[str, list[Candidate]],
    test_group: dict[str, list[Candidate]],
    train_name: str,
    test_name: str,
    features: list[str],
    ridge: float,
) -> list[dict[str, float | str]]:
    train = make_items(train_group)
    test = make_items(test_group)
    protocol = f"train={train_name}->eval={test_name}"
    return [
        summarize_selection(protocol, "fixed_alpha", test, predict_fixed_alpha(train, test)),
        summarize_selection(protocol, "centroid", test, predict_centroid(train, test, features)),
        summarize_selection(protocol, "stump_score", test, predict_stump_score(train, test, features)),
        summarize_selection(protocol, "ridge_score", test, predict_ridge_score(train, test, features, ridge)),
        summarize_selection(protocol, "oracle", test, [best_candidate(candidates).alpha for _, candidates in test]),
    ]


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "nan"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e214_eflic_strength_controller_probe"))
    parser.add_argument("--dists-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=3.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--input",
        action="append",
        help="Optional dataset=csv override. May be passed multiple times. Defaults to E212+E215 mixed-domain alpha sweeps.",
    )
    args = parser.parse_args()

    input_specs = parse_input_specs(args.input)
    groups = {
        name: read_candidates(name, Path(path), args.dists_weight, args.lpips_weight) for name, path in input_specs
    }
    features = common_features(groups.values())
    if not features:
        raise SystemExit("no common global predecision features")

    rows: list[dict[str, float | str]] = []
    for name, group in groups.items():
        rows.extend(loocv(group, features, args.ridge))
        for row in rows[-N_METHODS:]:
            row["dataset"] = name

    names = list(groups)
    if len(names) >= 2:
        pooled = merge_groups(groups.items())
        rows.extend(loocv(pooled, features, args.ridge, protocol="pooled_loocv"))
        for row in rows[-N_METHODS:]:
            row["dataset"] = "pooled"

        for eval_name in names:
            train_group = merge_groups((name, group) for name, group in groups.items() if name != eval_name)
            rows.extend(train_eval(train_group, groups[eval_name], f"all_except_{eval_name}", eval_name, features, args.ridge))
            for row in rows[-N_METHODS:]:
                row["dataset"] = eval_name

        rows.extend(train_eval(groups[names[0]], groups[names[1]], names[0], names[1], features, args.ridge))
        rows.extend(train_eval(groups[names[1]], groups[names[0]], names[1], names[0], features, args.ridge))
        for row in rows[-2 * N_METHODS:]:
            row["dataset"] = row["protocol"].split("eval=")[-1]

    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    args.output_prefix.with_suffix(".features.json").write_text(json.dumps({"features": features}, indent=2) + "\n")

    lines = [
        f"# {args.output_prefix.stem} EF-LIC Strength-Controller Probe",
        "",
        "This diagnostic uses only global predecision features (`z_hat_*`, `z_index_*`, `slice0_mean_*`, `slice0_scale_*`) and includes `alpha=0` fallback.",
        "It is a controller-design probe, not a final paper-facing trained policy.",
        "",
        "| dataset | protocol | method | n | dDISTS | dLPIPS | dPSNR | score | active frac | mean alpha | both wins | oracle gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {protocol} | {method} | {n} | {delta_dists_mean} | {delta_lpips_mean} | {delta_psnr_mean} | {score_mean} | {active_frac} | {mean_alpha} | {both_wins} | {oracle_gap_score} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Fixed-alpha baselines test whether strength tuning alone is enough.",
        "- Centroid, decision-stump, and ridge-score policies test whether decoder-safe predecision features contain usable strength-control signal.",
        "- A large oracle gap means the alpha headroom exists but the current feature/controller family is still insufficient.",
    ]
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"args": vars(args), "features": features, "rows": rows}, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")
    print(f"features={len(features)}")


if __name__ == "__main__":
    main()
