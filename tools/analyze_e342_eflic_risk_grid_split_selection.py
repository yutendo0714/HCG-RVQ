#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--controller-csv', nargs='+', required=True)
    p.add_argument('--train-count', type=int, default=16)
    p.add_argument('--eval-count', type=int, default=8)
    p.add_argument('--tail-weight', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument('--output-prefix', default='experiments/analysis/e342_eflic_e329_risk_grid_split_selection')
    return p.parse_args()


def label_from_path(path):
    name = path.stem
    m = re.search('risk(none|m[0-9]+)', name)
    if not m:
        return name
    token = m.group(1)
    if token == 'none':
        return 'risk_none'
    return 'risk_-0.' + token[1:]


def load_rows(path):
    rows = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mode') != 'trained_hard':
                continue
            rows.append({
                'image': row.get('image', ''),
                'delta_psnr': float(row['delta_psnr']),
                'delta_bpp': float(row.get('delta_bpp', 0.0)),
                'nonfinite_rows': int(float(row.get('nonfinite_rows', row.get('nonfinite', 0)))),
                'max_decode_diff': float(row.get('max_decode_diff', 0.0)),
                'alpha_mean': float(row.get('alpha_mean', row.get('y_alpha_mean', 0.0))),
                'gate_mean': float(row.get('gate_mean', row.get('y_gate_mean', 0.0))),
                'y_mismatch_frac': float(row.get('y_mismatch_frac', row.get('y_mismatch', 0.0))),
            })
    rows.sort(key=lambda r: r['image'])
    return rows


def summarize(rows):
    vals = [r['delta_psnr'] for r in rows]
    if not vals:
        return {
            'images': 0,
            'mean_delta_psnr': math.nan,
            'worst_delta_psnr': math.nan,
            'positive': 0,
            'negative': 0,
            'mean_delta_bpp': math.nan,
            'nonfinite_rows': 0,
            'max_decode_diff': math.nan,
            'alpha_mean': math.nan,
            'gate_mean': math.nan,
            'y_mismatch_frac': math.nan,
        }
    return {
        'images': len(rows),
        'mean_delta_psnr': mean(vals),
        'worst_delta_psnr': min(vals),
        'positive': sum(v > 0 for v in vals),
        'negative': sum(v < 0 for v in vals),
        'mean_delta_bpp': mean(r['delta_bpp'] for r in rows),
        'nonfinite_rows': sum(r['nonfinite_rows'] for r in rows),
        'max_decode_diff': max(r['max_decode_diff'] for r in rows),
        'alpha_mean': mean(r['alpha_mean'] for r in rows),
        'gate_mean': mean(r['gate_mean'] for r in rows),
        'y_mismatch_frac': mean(r['y_mismatch_frac'] for r in rows),
    }


def objective(summary, tail_weight):
    return summary['mean_delta_psnr'] + tail_weight * min(0.0, summary['worst_delta_psnr'])


def main():
    args = parse_args()
    entries = []
    for csv_path in args.controller_csv:
        path = Path(csv_path)
        rows = load_rows(path)
        if len(rows) < args.train_count + args.eval_count:
            raise SystemExit('not enough rows in {}: {}'.format(path, len(rows)))
        train_rows = rows[:args.train_count]
        eval_rows = rows[args.train_count:args.train_count + args.eval_count]
        entries.append({
            'label': label_from_path(path),
            'path': str(path),
            'train': summarize(train_rows),
            'eval': summarize(eval_rows),
            'all': summarize(rows),
        })

    selections = []
    for tw in args.tail_weight:
        ranked = sorted(entries, key=lambda e: (objective(e['train'], tw), e['train']['mean_delta_psnr']), reverse=True)
        best = ranked[0]
        selections.append({
            'tail_weight': tw,
            'selected_label': best['label'],
            'train_objective': objective(best['train'], tw),
            'train': best['train'],
            'eval': best['eval'],
            'all': best['all'],
        })

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix('.json')
    csv_path = out_prefix.with_suffix('.summary.csv')
    md_path = out_prefix.with_suffix('.md')
    json_path.write_text(json.dumps({'entries': entries, 'selections': selections}, indent=2))

    with csv_path.open('w', newline='') as f:
        fieldnames = [
            'tail_weight', 'selected_label', 'train_objective',
            'train_mean_delta_psnr', 'train_worst_delta_psnr',
            'eval_mean_delta_psnr', 'eval_worst_delta_psnr',
            'eval_positive', 'eval_negative', 'eval_mean_delta_bpp',
            'eval_nonfinite_rows', 'eval_max_decode_diff',
            'eval_alpha_mean', 'eval_gate_mean',
            'all_mean_delta_psnr', 'all_worst_delta_psnr',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in selections:
            writer.writerow({
                'tail_weight': s['tail_weight'],
                'selected_label': s['selected_label'],
                'train_objective': s['train_objective'],
                'train_mean_delta_psnr': s['train']['mean_delta_psnr'],
                'train_worst_delta_psnr': s['train']['worst_delta_psnr'],
                'eval_mean_delta_psnr': s['eval']['mean_delta_psnr'],
                'eval_worst_delta_psnr': s['eval']['worst_delta_psnr'],
                'eval_positive': s['eval']['positive'],
                'eval_negative': s['eval']['negative'],
                'eval_mean_delta_bpp': s['eval']['mean_delta_bpp'],
                'eval_nonfinite_rows': s['eval']['nonfinite_rows'],
                'eval_max_decode_diff': s['eval']['max_decode_diff'],
                'eval_alpha_mean': s['eval']['alpha_mean'],
                'eval_gate_mean': s['eval']['gate_mean'],
                'all_mean_delta_psnr': s['all']['mean_delta_psnr'],
                'all_worst_delta_psnr': s['all']['worst_delta_psnr'],
            })

    lines = []
    lines.append('# E342 EF-LIC Risk Grid Split-Selection Audit')
    lines.append('')
    lines.append('This audit selects the max-risk setting on the first 16 Kodak images and reports the held-out last 8 images.')
    lines.append('It checks threshold-selection protocol only; the controller itself was trained on the first 16 images.')
    lines.append('')
    lines.append('## Candidate Full24 Rows')
    lines.append('')
    lines.append('| label | train mean | train worst | eval mean | eval worst | all mean | all worst | eval nonfinite | eval decode max |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for e in entries:
        lines.append('| {} | {:+.8f} | {:+.8f} | {:+.8f} | {:+.8f} | {:+.8f} | {:+.8f} | {} | {:.8f} |'.format(
            e['label'], e['train']['mean_delta_psnr'], e['train']['worst_delta_psnr'],
            e['eval']['mean_delta_psnr'], e['eval']['worst_delta_psnr'],
            e['all']['mean_delta_psnr'], e['all']['worst_delta_psnr'],
            e['eval']['nonfinite_rows'], e['eval']['max_decode_diff']))
    lines.append('')
    lines.append('## Train-Selected Policies')
    lines.append('')
    lines.append('| tail weight | selected | train objective | train mean | train worst | eval mean | eval worst | eval pos/neg | all mean | all worst |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in selections:
        lines.append('| {:.2f} | {} | {:+.8f} | {:+.8f} | {:+.8f} | {:+.8f} | {:+.8f} | {}/{} | {:+.8f} | {:+.8f} |'.format(
            s['tail_weight'], s['selected_label'], s['train_objective'],
            s['train']['mean_delta_psnr'], s['train']['worst_delta_psnr'],
            s['eval']['mean_delta_psnr'], s['eval']['worst_delta_psnr'],
            s['eval']['positive'], s['eval']['negative'],
            s['all']['mean_delta_psnr'], s['all']['worst_delta_psnr']))
    lines.append('')
    lines.append('Interpretation:')
    lines.append('')
    lines.append('- Mean-only selection prefers the aggressive setting if its train mean is high, but this keeps a dangerous eval tail.')
    lines.append('- Tail-aware selection can choose a stricter risk setting, but too much strictness gives up mean improvement.')
    lines.append('- Promotion needs independent validation and probably a learned no-op classifier rather than post-hoc threshold tuning.')
    md_path.write_text('\n'.join(lines) + '\n')
    print('wrote {}, {}, {}'.format(md_path, json_path, csv_path))


if __name__ == '__main__':
    main()
