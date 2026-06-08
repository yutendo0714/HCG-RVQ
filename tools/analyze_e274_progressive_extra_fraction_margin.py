#!/usr/bin/env python3
"""E274 margin audit for base-plus-active-RVQ progressive enhancement.

E273 tests a conservative progressive enhancement model: keep the base bitstream
and send active RVQ indices as an extra enhancement.  This script asks what
fraction of those active RVQ bits can be afforded before the soft/progressive
quality gain becomes non-negative.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--label', default='trained_progressive_extra_soft')
    return p.parse_args()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(vals: list[float]) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    return float(np.mean(xs)) if xs else float('nan')


def percentile(vals: list[float], q: float) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    return float(np.percentile(xs, q)) if xs else float('nan')


def source_name(path: Path) -> str:
    stem = path.stem.lower()
    if 'clic' in stem:
        return 'clic'
    if 'kodak' in stem:
        return 'kodak'
    return path.stem


def read_rows(paths: list[Path], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('label') != label:
                    continue
                score = finite(row.get('score'), float('nan'))
                delta_bpp = finite(row.get('delta_bpp'), float('nan'))
                active_extra = finite(row.get('active_rvq_extra_bpp'), delta_bpp)
                quality_score = score - delta_bpp
                affordable = -quality_score / active_extra if active_extra > 0 and quality_score < 0 else 0.0
                rows.append({
                    'source': source_name(path),
                    'image': row.get('image', ''),
                    'score_full_extra': score,
                    'score_no_extra': quality_score,
                    'delta_bpp_full_extra': delta_bpp,
                    'active_rvq_extra_bpp': active_extra,
                    'active_scalar_bpp': finite(row.get('active_scalar_bpp'), float('nan')),
                    'replacement_delta_bpp': active_extra - finite(row.get('active_scalar_bpp'), 0.0),
                    'affordable_extra_fraction': affordable,
                    'full_extra_win': int(score < 0.0),
                    'half_extra_score': quality_score + 0.5 * active_extra,
                    'half_extra_win': int((quality_score + 0.5 * active_extra) < 0.0),
                    'three_quarter_extra_score': quality_score + 0.75 * active_extra,
                    'three_quarter_extra_win': int((quality_score + 0.75 * active_extra) < 0.0),
                    'delta_psnr': finite(row.get('delta_psnr'), float('nan')),
                    'delta_ms_ssim': finite(row.get('delta_ms_ssim'), float('nan')),
                    'delta_lpips': finite(row.get('delta_lpips'), float('nan')),
                    'delta_dists': finite(row.get('delta_dists'), float('nan')),
                    'gate_mean': finite(row.get('gate_mean'), float('nan')),
                    'nonfinite': int(finite(row.get('nonfinite'), 0.0)),
                })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({r['source'] for r in rows}) + ['all']:
        subset = rows if source == 'all' else [r for r in rows if r['source'] == source]
        if not subset:
            continue
        out.append({
            'source': source,
            'images': len(subset),
            'score_no_extra': mean([r['score_no_extra'] for r in subset]),
            'score_full_extra': mean([r['score_full_extra'] for r in subset]),
            'active_rvq_extra_bpp': mean([r['active_rvq_extra_bpp'] for r in subset]),
            'active_scalar_bpp': mean([r['active_scalar_bpp'] for r in subset]),
            'replacement_delta_bpp': mean([r['replacement_delta_bpp'] for r in subset]),
            'affordable_extra_fraction_mean': mean([r['affordable_extra_fraction'] for r in subset]),
            'affordable_extra_fraction_p10': percentile([r['affordable_extra_fraction'] for r in subset], 10),
            'affordable_extra_fraction_min': min(r['affordable_extra_fraction'] for r in subset),
            'full_extra_win_frac': mean([float(r['full_extra_win']) for r in subset]),
            'half_extra_score': mean([r['half_extra_score'] for r in subset]),
            'half_extra_win_frac': mean([float(r['half_extra_win']) for r in subset]),
            'three_quarter_extra_score': mean([r['three_quarter_extra_score'] for r in subset]),
            'three_quarter_extra_win_frac': mean([float(r['three_quarter_extra_win']) for r in subset]),
            'nonfinite_rows': sum(int(r['nonfinite']) for r in subset),
        })
    return out


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'experiment': 'E274 progressive extra fraction margin audit',
        'note': 'Accounting audit over E273 rows. Not a final entropy-coded codec.',
        'inputs': [str(p) for p in args.inputs],
        'label': args.label,
        'summary': summary,
        'rows': rows,
    }
    args.output_prefix.with_suffix('.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = [
        '# E274 Progressive Extra Fraction Margin Audit',
        '',
        'Audits how much active RVQ enhancement bpp can be afforded for E273 base-plus-enhancement rows.',
        '',
        '| source | images | no-extra score | full-extra score | extra bpp | active scalar bpp | replacement dbpp | afford frac mean | afford frac p10 | afford frac min | full win | half score | half win | 0.75 score | 0.75 win | nonfinite |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in summary:
        lines.append(
            f"| {r['source']} | {r['images']} | {r['score_no_extra']:+.6f} | {r['score_full_extra']:+.6f} | "
            f"{r['active_rvq_extra_bpp']:.6f} | {r['active_scalar_bpp']:.6f} | {r['replacement_delta_bpp']:+.6f} | "
            f"{r['affordable_extra_fraction_mean']:.3f} | {r['affordable_extra_fraction_p10']:.3f} | {r['affordable_extra_fraction_min']:.3f} | "
            f"{r['full_extra_win_frac']:.3f} | {r['half_extra_score']:+.6f} | {r['half_extra_win_frac']:.3f} | "
            f"{r['three_quarter_extra_score']:+.6f} | {r['three_quarter_extra_win_frac']:.3f} | {r['nonfinite_rows']} |"
        )
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs, args.label)
    if not rows:
        raise SystemExit('no matching rows')
    summary = summarize(rows)
    write_outputs(args, rows, summary)


if __name__ == '__main__':
    main()
