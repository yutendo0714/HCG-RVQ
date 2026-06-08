#!/usr/bin/env python3
"""Weighted mixed-domain EF-LIC perceptual selector audit.

This extends the E353 ridge selector by weighting calibration candidates by
source dataset. It is a design probe, not a final paper claim: the purpose is to
see whether Kodak teacher rows should regularize or dominate a CLIC-targeted
selector before moving to a local/sequential controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e353_eflic_perceptual_learned_selector_split import (  # noqa: E402
    FEATURES,
    GLOBAL_FEATURES,
    Model,
    candidate_rows,
    choose_oracle,
    fmt,
    model_feature_vector,
    parse_run,
    predict_score,
    read_rows,
    risk_list,
    select_fixed,
    select_with_model,
    summarize,
    write_csv,
)


def parse_weights(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in spec.split(','):
        if not part:
            continue
        if '=' not in part:
            raise argparse.ArgumentTypeError('source weights must be name=value')
        name, value = part.split('=', 1)
        out[name] = float(value)
    return out


def image_source_map(runs: Dict[str, Dict[str, dict]]) -> Dict[str, str]:
    first = next(iter(runs.values()))
    out = {}
    for image, row in first.items():
        out[image] = str(row.get('source_dataset') or row.get('source') or 'unknown')
    return out


def weighted_matrix(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    risk_names: List[str],
    source_weights: Dict[str, float],
    source_by_image: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[List[float]] = []
    ys: List[float] = []
    ws: List[float] = []
    for image in images:
        source = source_by_image.get(image, 'unknown')
        weight = float(source_weights.get(source, source_weights.get('default', 1.0)))
        for row in candidate_rows(image, runs):
            xs.append(model_feature_vector(row, risk_names, feature_names))
            ys.append(float(row['score']))
            ws.append(weight)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(ws, dtype=np.float64)


def fit_weighted_ridge(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    lambda_value: float,
    source_weights: Dict[str, float],
    source_by_image: Dict[str, str],
) -> Tuple[Model, dict]:
    risk_names = risk_list(runs)
    x, y, w = weighted_matrix(images, runs, feature_names, risk_names, source_weights, source_by_image)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    mu[0] = 0.0
    sigma[0] = 1.0
    sigma[sigma < 1e-9] = 1.0
    xz = (x - mu) / sigma
    sw = np.sqrt(np.maximum(w, 1e-9))[:, None]
    xw = xz * sw
    yw = y * sw.reshape(-1)
    reg = np.eye(xw.shape[1], dtype=np.float64) * float(lambda_value)
    reg[0, 0] = 0.0
    lhs = xw.T @ xw + reg
    rhs = xw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    model = Model(
        name=f'weighted_ridge_score_lam{lambda_value:g}',
        risk_names=risk_names,
        feature_names=feature_names,
        mean=mu.tolist(),
        scale=sigma.tolist(),
        weights=beta.reshape(-1, 1).tolist(),
        lambda_value=float(lambda_value),
        train_mode='weighted_ridge_candidate_score',
    )
    return model, {
        'rank': int(np.linalg.matrix_rank(xw)),
        'condition': float(np.linalg.cond(lhs)),
        'n_train_candidates': int(xw.shape[0]),
        'n_features': int(xw.shape[1]),
        'weight_min': float(w.min()),
        'weight_max': float(w.max()),
    }


def selected_rows(images: Sequence[str], runs: Dict[str, Dict[str, dict]], model: Model, margin: float) -> List[dict]:
    return [select_with_model(img, runs, model, margin) for img in images]


def loocv(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    lambda_value: float,
    margin: float,
    source_weights: Dict[str, float],
    source_by_image: Dict[str, str],
) -> dict:
    rows: List[dict] = []
    infos: List[dict] = []
    for held in images:
        train = [img for img in images if img != held]
        model, info = fit_weighted_ridge(train, runs, feature_names, lambda_value, source_weights, source_by_image)
        infos.append(info)
        rows.append(select_with_model(held, runs, model, margin))
    s = summarize(rows, f'loocv_lam{lambda_value:g}_margin{margin:g}')
    s['lambda'] = float(lambda_value)
    s['margin'] = float(margin)
    s['mean_condition'] = mean(i['condition'] for i in infos)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', type=parse_run, required=True)
    ap.add_argument('--cal-count', type=int, required=True)
    ap.add_argument('--lpips-weight', type=float, default=3.0)
    ap.add_argument('--lambda-grid', default='0,1e-4,1e-3,1e-2,1e-1,1,10,100')
    ap.add_argument('--margin-grid', default='0,0.0001,0.00025,0.0005,0.001')
    ap.add_argument('--feature-set', choices=['global', 'global_slice'], default='global_slice')
    ap.add_argument('--source-weights', type=parse_weights, default=parse_weights('default=1'))
    ap.add_argument('--output-prefix', required=True)
    args = ap.parse_args()

    runs = {name: read_rows(path, name, args.lpips_weight) for name, path in args.run}
    images = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    cal_images = images[:args.cal_count]
    eval_images = images[args.cal_count:]
    if len(cal_images) < 4 or not eval_images:
        raise SystemExit('Need calibration and eval images')
    source_by_image = image_source_map(runs)
    feature_names = GLOBAL_FEATURES if args.feature_set == 'global' else FEATURES
    lambdas = [float(x) for x in args.lambda_grid.split(',') if x]
    margins = [float(x) for x in args.margin_grid.split(',') if x]

    cv_rows: List[dict] = []
    for lam in lambdas:
        for margin in margins:
            cv_rows.append(loocv(cal_images, runs, feature_names, lam, margin, args.source_weights, source_by_image))
    cv_rows.sort(key=lambda r: (r['mean_score'], r['worst_score'], -r['mean_delta_ms_ssim']))
    selected_cv = cv_rows[0]

    model, fit_info = fit_weighted_ridge(
        cal_images, runs, feature_names, selected_cv['lambda'], args.source_weights, source_by_image
    )
    cal_summary = summarize(selected_rows(cal_images, runs, model, selected_cv['margin']), 'weighted_selected_cal')
    eval_rows = selected_rows(eval_images, runs, model, selected_cv['margin'])
    eval_summary = summarize(eval_rows, 'weighted_selected_eval')
    cal_oracle = summarize([choose_oracle(img, runs) for img in cal_images], 'oracle_cal')
    eval_oracle = summarize([choose_oracle(img, runs) for img in eval_images], 'oracle_eval')

    fixed_summaries = []
    for risk in ['noop'] + risk_list(runs):
        fixed_summaries.append({'split': 'cal', **summarize(select_fixed(cal_images, runs, risk), f'fixed_{risk}')})
        fixed_summaries.append({'split': 'eval', **summarize(select_fixed(eval_images, runs, risk), f'fixed_{risk}')})

    per_image = []
    for split, split_images in [('cal', cal_images), ('eval', eval_images)]:
        for image in split_images:
            chosen = select_with_model(image, runs, model, selected_cv['margin'])
            oracle = choose_oracle(image, runs)
            row = {
                'split': split,
                'image': image,
                'source_dataset': source_by_image.get(image, 'unknown'),
                'chosen_risk': chosen['risk'],
                'chosen_predicted_score': chosen.get('predicted_score', ''),
                'chosen_score': chosen['score'],
                'chosen_delta_psnr': chosen['delta_psnr'],
                'chosen_delta_ms_ssim': chosen['delta_ms_ssim'],
                'chosen_delta_lpips': chosen['delta_lpips'],
                'chosen_delta_dists': chosen['delta_dists'],
                'oracle_risk': oracle['risk'],
                'oracle_score': oracle['score'],
            }
            per_image.append(row)

    prefix = Path(args.output_prefix)
    payload = {
        'score': f'delta_DISTS + {args.lpips_weight:g} * delta_LPIPS',
        'psnr_role': 'diagnostic_tail_metric_not_selector',
        'feature_set': args.feature_set,
        'source_weights': args.source_weights,
        'cal_count': len(cal_images),
        'eval_count': len(eval_images),
        'selected_cv': selected_cv,
        'fit_info': fit_info,
        'cal_selected': cal_summary,
        'eval_selected': eval_summary,
        'cal_oracle': cal_oracle,
        'eval_oracle': eval_oracle,
        'fixed_summaries': fixed_summaries,
        'model': {
            'lambda_value': model.lambda_value,
            'margin': selected_cv['margin'],
            'risk_names': model.risk_names,
            'feature_names': model.feature_names,
            'mean': model.mean,
            'scale': model.scale,
            'weights': model.weights,
        },
    }
    prefix.with_suffix('.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    write_csv(prefix.with_name(prefix.name + '_cv_grid.csv'), cv_rows, cv_rows[0].keys())
    write_csv(prefix.with_name(prefix.name + '_fixed_summaries.csv'), fixed_summaries, fixed_summaries[0].keys())
    write_csv(prefix.with_name(prefix.name + '_per_image.csv'), per_image, per_image[0].keys())

    lines = [
        '# E357 EF-LIC Weighted Mixed Perceptual Selector Audit',
        '',
        f"Score: `delta_DISTS + {args.lpips_weight:g} * delta_LPIPS` (lower is better). PSNR is diagnostic only.",
        f'Feature set: `{args.feature_set}` with {len(feature_names)} decoder-visible candidate features.',
        f'Source weights: `{args.source_weights}`.',
        f'Calibration images: {len(cal_images)}; eval images: {len(eval_images)}.',
        '',
        '## Selected Weighted Ridge Policy',
        '',
        f"LOOCV selected lambda `{selected_cv['lambda']}` and active margin `{selected_cv['margin']}`.",
        f"LOOCV calibration score `{fmt(selected_cv['mean_score'])}`, worst `{fmt(selected_cv['worst_score'])}`.",
        '',
        '| split | score | worst score | dPSNR | worst dPSNR | dMS-SSIM | dLPIPS | dDISTS | score wins | choices | max dBPP | decode max | nonfinite |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|',
    ]
    for label, s in [('cal selected', cal_summary), ('eval selected', eval_summary), ('cal oracle', cal_oracle), ('eval oracle', eval_oracle)]:
        lines.append(
            f"| {label} | {fmt(s['mean_score'])} | {fmt(s['worst_score'])} | {fmt(s['mean_delta_psnr'])} | {fmt(s['worst_delta_psnr'])} | {fmt(s['mean_delta_ms_ssim'])} | {fmt(s['mean_delta_lpips'])} | {fmt(s['mean_delta_dists'])} | {s['score_win_count']}/{s['n']} | `{s['choices']}` | {fmt(s['max_abs_delta_bpp'])} | {s['max_decode_diff']:.3e} | {s['nonfinite_rows']} |"
        )
    lines += ['', '## Fixed-Risk Baselines', '', '| split | policy | score | worst score | dPSNR | worst dPSNR | score wins | choices |', '|---|---|---:|---:|---:|---:|---:|---|']
    for s in fixed_summaries:
        lines.append(
            f"| {s['split']} | {s['policy']} | {fmt(s['mean_score'])} | {fmt(s['worst_score'])} | {fmt(s['mean_delta_psnr'])} | {fmt(s['worst_delta_psnr'])} | {s['score_win_count']}/{s['n']} | `{s['choices']}` |"
        )
    lines += ['', 'Interpretation:', '', '- Weighted source fitting is a selector-design audit, not final paper evidence.', '- A useful weight should improve held perceptual score without hiding bpp/decode or nonfinite issues.', '- PSNR is listed only to catch codec-health tails.']
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f"wrote {prefix.with_suffix('.md')}, {prefix.with_suffix('.json')}")


if __name__ == '__main__':
    main()
