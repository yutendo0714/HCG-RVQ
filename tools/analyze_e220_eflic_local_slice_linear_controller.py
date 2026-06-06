#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analyze_e218_eflic_local_slice_predictability import (
    build_samples,
    read_rows,
    summarize_policy,
    to_float,
)


@dataclass(frozen=True)
class RidgePolicy:
    l2: float
    threshold: float
    train_score: float
    weights: list[float]
    feature_mean: list[float]
    feature_std: list[float]


def make_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    x = np.array([[to_float(row.get(f)) for f in feature_names] for row in rows], dtype=float)
    return x


def standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(x), x, mean[None, :])
    std = filled.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (filled - mean[None, :]) / std[None, :], mean, std


def standardize_eval(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, mean[None, :])
    return (filled - mean[None, :]) / std[None, :]


def fit_ridge_weights(x_std: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=float), x_std], axis=1)
    reg = np.eye(design.shape[1], dtype=float) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def predict_with_weights(x_std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=float), x_std], axis=1)
    return design @ weights


def best_threshold_from_predictions(rows: list[dict[str, Any]], pred: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(pred)
    if not finite.any():
        return -float('inf'), 0.0, 0.0
    vals = np.unique(pred[finite])
    if vals.size == 1:
        thresholds = np.array([vals[0] - 1e-9, vals[0] + 1e-9], dtype=float)
    else:
        mids = (vals[:-1] + vals[1:]) * 0.5
        thresholds = np.concatenate(([vals[0] - 1e-9], mids, [vals[-1] + 1e-9]))
    best_t = -float('inf')
    best_score = 0.0
    best_active = 0.0
    for t in thresholds:
        active = (pred <= t) & finite
        summary = summarize_policy(rows, active)
        score = float(summary['score'])
        if score < best_score - 1e-15:
            best_score = score
            best_t = float(t)
            best_active = float(active.mean())
    return best_t, best_score, best_active


def fit_policy(rows: list[dict[str, Any]], feature_names: list[str], l2_values: list[float]) -> RidgePolicy:
    x = make_matrix(rows, feature_names)
    y = np.array([to_float(row['score']) for row in rows], dtype=float)
    x_std, mean, std = standardize_train(x)
    best: RidgePolicy | None = None
    for l2 in l2_values:
        weights = fit_ridge_weights(x_std, y, l2)
        pred = predict_with_weights(x_std, weights)
        threshold, train_score, _ = best_threshold_from_predictions(rows, pred)
        if best is None or train_score < best.train_score - 1e-15:
            best = RidgePolicy(
                l2=float(l2),
                threshold=float(threshold),
                train_score=float(train_score),
                weights=[float(v) for v in weights],
                feature_mean=[float(v) for v in mean],
                feature_std=[float(v) for v in std],
            )
    assert best is not None
    return best


def eval_policy(rows: list[dict[str, Any]], feature_names: list[str], policy: RidgePolicy, threshold: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    x = make_matrix(rows, feature_names)
    mean = np.array(policy.feature_mean, dtype=float)
    std = np.array(policy.feature_std, dtype=float)
    weights = np.array(policy.weights, dtype=float)
    pred = predict_with_weights(standardize_eval(x, mean, std), weights)
    t = policy.threshold if threshold is None else threshold
    active = np.isfinite(pred) & (pred <= t)
    summary = summarize_policy(rows, active)
    summary['l2'] = policy.l2
    summary['threshold'] = float(t)
    summary['pred_mean'] = float(np.nanmean(pred))
    summary['pred_std'] = float(np.nanstd(pred))
    return active, summary


def top_weights(policy: RidgePolicy, feature_names: list[str], k: int = 10) -> list[dict[str, Any]]:
    weights = np.array(policy.weights[1:], dtype=float)
    order = np.argsort(np.abs(weights))[::-1]
    return [{'feature': feature_names[i], 'weight': float(weights[i])} for i in order[:k]]


def add_summary(summaries: list[dict[str, Any]], dataset: str, policy: str, rows: list[dict[str, Any]], active: np.ndarray, **extra: Any) -> None:
    summary = summarize_policy(rows, active)
    summary.update({'dataset': dataset, 'policy': policy})
    summary.update(extra)
    summaries.append(summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', action='append', required=True, help='dataset=csv')
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--lpips-weight', type=float, default=3.0)
    p.add_argument('--l2-grid', default='0.001,0.01,0.1,1,10,100,1000')
    args = p.parse_args()

    l2_values = [float(x) for x in args.l2_grid.split(',') if x]
    raw_rows = read_rows(args.input)
    samples, feature_names = build_samples(raw_rows, args.lpips_weight)
    if not samples:
        raise SystemExit('no single-slice samples found')

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_suffix('.summary.csv')
    json_path = args.output_prefix.with_suffix('.json')
    md_path = args.output_prefix.with_suffix('.md')

    summaries: list[dict[str, Any]] = []
    datasets = sorted({row['dataset'] for row in samples})
    groups = [('pooled', samples)] + [(d, [r for r in samples if r['dataset'] == d]) for d in datasets]

    for name, rows in groups:
        scores = np.array([to_float(r['score']) for r in rows], dtype=float)
        add_summary(summaries, name, 'all_off', rows, np.zeros(len(rows), dtype=bool))
        add_summary(summaries, name, 'all_on_single_slice', rows, np.ones(len(rows), dtype=bool))
        add_summary(summaries, name, 'oracle_single_slice', rows, scores < 0.0)
        policy = fit_policy(rows, feature_names, l2_values)
        _, train_t_summary = eval_policy(rows, feature_names, policy)
        train_t_summary.update({'dataset': name, 'policy': 'same_table_ridge_train_threshold'})
        summaries.append(train_t_summary)
        _, zero_summary = eval_policy(rows, feature_names, policy, threshold=0.0)
        zero_summary.update({'dataset': name, 'policy': 'same_table_ridge_zero_threshold'})
        summaries.append(zero_summary)

    active = []
    chosen_l2: dict[str, int] = {}
    for i, row in enumerate(samples):
        train = samples[:i] + samples[i + 1:]
        policy = fit_policy(train, feature_names, l2_values)
        chosen_l2[str(policy.l2)] = chosen_l2.get(str(policy.l2), 0) + 1
        act, _ = eval_policy([row], feature_names, policy)
        active.append(bool(act[0]))
    loocv_summary = summarize_policy(samples, np.array(active, dtype=bool))
    loocv_summary.update({'dataset': 'pooled', 'policy': 'sample_loocv_ridge_train_threshold', 'chosen_l2': chosen_l2})
    summaries.append(loocv_summary)

    lodo_weights: dict[str, list[dict[str, Any]]] = {}
    for held in datasets:
        train = [r for r in samples if r['dataset'] != held]
        eval_rows = [r for r in samples if r['dataset'] == held]
        policy = fit_policy(train, feature_names, l2_values)
        lodo_weights[held] = top_weights(policy, feature_names)
        _, train_t_summary = eval_policy(eval_rows, feature_names, policy)
        train_t_summary.update(
            {
                'dataset': held,
                'policy': 'leave_dataset_out_ridge_train_threshold',
                'train_dataset': '+'.join(sorted({r['dataset'] for r in train})),
            }
        )
        summaries.append(train_t_summary)
        _, zero_summary = eval_policy(eval_rows, feature_names, policy, threshold=0.0)
        zero_summary.update(
            {
                'dataset': held,
                'policy': 'leave_dataset_out_ridge_zero_threshold',
                'train_dataset': '+'.join(sorted({r['dataset'] for r in train})),
            }
        )
        summaries.append(zero_summary)

    fields = sorted({key for row in summaries for key in row.keys()})
    preferred = [
        'dataset', 'policy', 'samples', 'score', 'delta_dists', 'delta_lpips', 'delta_psnr',
        'active_frac', 'helpful_frac', 'precision', 'recall', 'both_win_frac_when_active',
        'l2', 'threshold', 'pred_mean', 'pred_std', 'train_dataset', 'chosen_l2',
    ]
    fields = preferred + [f for f in fields if f not in preferred]
    with summary_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        'lpips_weight': args.lpips_weight,
        'l2_grid': l2_values,
        'inputs': args.input,
        'samples': len(samples),
        'features': feature_names,
        'summaries': summaries,
        'leave_dataset_out_top_weights': lodo_weights,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

    lines = [
        '# E220 EF-LIC Local Slice Linear Controller Probe',
        '',
        f'Objective: `score = dDISTS + {args.lpips_weight:g}*dLPIPS`; negative is better.',
        'This is still a diagnostic controller probe. It uses only decoder-safe predecision features and learns a ridge score predictor plus a train-selected active threshold.',
        '',
        '| dataset | policy | samples | score | dDISTS | dLPIPS | active | helpful | precision | recall | l2 | threshold | train |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in summaries:
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('policy', '')} | {int(row.get('samples', 0))} | "
            f"{float(row.get('score', 0.0)):+.6f} | {float(row.get('delta_dists', 0.0)):+.6f} | "
            f"{float(row.get('delta_lpips', 0.0)):+.6f} | {float(row.get('active_frac', 0.0)):.3f} | "
            f"{float(row.get('helpful_frac', 0.0)):.3f} | {float(row.get('precision', 0.0)):.3f} | "
            f"{float(row.get('recall', 0.0)):.3f} | {row.get('l2', '')} | {row.get('threshold', '')} | "
            f"{row.get('train_dataset', '')} |"
        )
    lines.extend(['', 'Leave-dataset-out top absolute ridge weights:', ''])
    for held, rows in lodo_weights.items():
        joined = ', '.join(f"{r['feature']}={r['weight']:+.3g}" for r in rows[:8])
        lines.append(f'- {held}: {joined}')
    lines.extend(
        [
            '',
            'Interpretation:',
            '',
            '- `same_table_*` rows are capacity/headroom diagnostics and should not be used as final claims.',
            '- `leave_dataset_out_*` rows test whether the decoder-safe local feature family transfers across domains without touching the held dataset.',
            '- If ridge improves same-table but fails leave-dataset-out, the next method needs more independent labels, stronger local features, or a trained HCG geometry head rather than a hand-built controller.',
        ]
    )
    md_path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {md_path}')


if __name__ == '__main__':
    main()
