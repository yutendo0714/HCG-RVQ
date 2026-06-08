#!/usr/bin/env python3
"""E275 replacement-rate audit for GLC active RVQ rows.

E273/E274 show that base-plus-full-active-RVQ enhancement can overpay rate,
especially on CLIC.  This audit asks whether the same soft branch would remain
useful if active scalar residual bits were replaced by active RVQ bits instead
of transmitted in addition to them.
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
    p.add_argument('--cap-dbpp', type=float, nargs='*', default=[0.0015, 0.0020, 0.0025, 0.0030])
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
                score_full_extra = finite(row.get('score'), float('nan'))
                full_delta_bpp = finite(row.get('delta_bpp'), float('nan'))
                active_rvq = finite(row.get('active_rvq_extra_bpp'), full_delta_bpp)
                active_scalar = finite(row.get('active_scalar_bpp'), 0.0)
                quality_score = score_full_extra - full_delta_bpp
                replacement_dbpp = active_rvq - active_scalar
                replacement_score = quality_score + replacement_dbpp
                scalar_saved_fraction = active_scalar / active_rvq if active_rvq > 0 else 0.0
                rows.append({
                    'source': source_name(path),
                    'image': row.get('image', ''),
                    'score_no_rate': quality_score,
                    'score_full_extra': score_full_extra,
                    'score_replacement': replacement_score,
                    'full_extra_dbpp': active_rvq,
                    'active_scalar_bpp': active_scalar,
                    'replacement_dbpp': replacement_dbpp,
                    'scalar_saved_fraction': scalar_saved_fraction,
                    'full_extra_win': int(score_full_extra < 0.0),
                    'replacement_win': int(replacement_score < 0.0),
                    'delta_psnr': finite(row.get('delta_psnr'), float('nan')),
                    'delta_ms_ssim': finite(row.get('delta_ms_ssim'), float('nan')),
                    'delta_lpips': finite(row.get('delta_lpips'), float('nan')),
                    'delta_dists': finite(row.get('delta_dists'), float('nan')),
                    'gate_mean': finite(row.get('gate_mean'), float('nan')),
                    'nonfinite': int(finite(row.get('nonfinite'), 0.0)),
                })
    return rows


def summarize_group(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    subset = rows if source == 'all' else [r for r in rows if r['source'] == source]
    return {
        'source': source,
        'images': len(subset),
        'score_no_rate': mean([r['score_no_rate'] for r in subset]),
        'score_full_extra': mean([r['score_full_extra'] for r in subset]),
        'score_replacement': mean([r['score_replacement'] for r in subset]),
        'full_extra_dbpp': mean([r['full_extra_dbpp'] for r in subset]),
        'active_scalar_bpp': mean([r['active_scalar_bpp'] for r in subset]),
        'replacement_dbpp': mean([r['replacement_dbpp'] for r in subset]),
        'scalar_saved_fraction': mean([r['scalar_saved_fraction'] for r in subset]),
        'full_extra_win_frac': mean([float(r['full_extra_win']) for r in subset]),
        'replacement_win_frac': mean([float(r['replacement_win']) for r in subset]),
        'nonfinite_rows': sum(int(r['nonfinite']) for r in subset),
    }


def summarize_caps(rows: list[dict[str, Any]], caps: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sources = sorted({r['source'] for r in rows}) + ['all']
    for source in sources:
        subset = rows if source == 'all' else [r for r in rows if r['source'] == source]
        if not subset:
            continue
        for cap in caps:
            selected = [r for r in subset if r['replacement_dbpp'] <= cap]
            scores = [r['score_replacement'] if r['replacement_dbpp'] <= cap else 0.0 for r in subset]
            out.append({
                'source': source,
                'cap_dbpp': cap,
                'images': len(subset),
                'selected_frac': len(selected) / len(subset),
                'selected_win_frac': mean([float(r['replacement_win']) for r in selected]) if selected else 0.0,
                'score': mean(scores),
            })
    return out


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: list[dict[str, Any]], caps: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    with args.output_prefix.with_name(args.output_prefix.name + '_caps').with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'cap_dbpp', 'images', 'selected_frac', 'selected_win_frac', 'score'])
        writer.writeheader()
        writer.writerows(caps)
    payload = {
        'experiment': 'E275 replacement-rate margin audit',
        'note': 'Accounting audit over E273 rows. Not a final entropy-coded codec.',
        'inputs': [str(p) for p in args.inputs],
        'label': args.label,
        'summary': summary,
        'caps': caps,
        'rows': rows,
    }
    args.output_prefix.with_suffix('.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = [
        '# E275 Replacement-Rate Margin Audit',
        '',
        'Audits the same E273 rows under a replacement accounting model: active RVQ bits replace active scalar bits instead of being sent in addition to the base stream.',
        '',
        '| source | images | no-rate score | full-extra score | replacement score | full extra dbpp | active scalar bpp | replacement dbpp | scalar saved frac | full win | replacement win | nonfinite |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in summary:
        lines.append(
            f"| {r['source']} | {r['images']} | {r['score_no_rate']:+.6f} | {r['score_full_extra']:+.6f} | "
            f"{r['score_replacement']:+.6f} | {r['full_extra_dbpp']:.6f} | {r['active_scalar_bpp']:.6f} | "
            f"{r['replacement_dbpp']:+.6f} | {r['scalar_saved_fraction']:.3f} | {r['full_extra_win_frac']:.3f} | "
            f"{r['replacement_win_frac']:.3f} | {r['nonfinite_rows']} |"
        )
    lines += [
        '',
        '## Replacement cap sweep',
        '',
        '| source | cap dbpp | images | selected frac | selected win | score |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in caps:
        lines.append(
            f"| {r['source']} | {r['cap_dbpp']:.6f} | {r['images']} | {r['selected_frac']:.3f} | "
            f"{r['selected_win_frac']:.3f} | {r['score']:+.6f} |"
        )
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs, args.label)
    if not rows:
        raise SystemExit('no matching rows')
    sources = sorted({r['source'] for r in rows}) + ['all']
    summary = [summarize_group(rows, source) for source in sources]
    caps = summarize_caps(rows, sorted(args.cap_dbpp))
    write_outputs(args, rows, summary, caps)


if __name__ == '__main__':
    main()
