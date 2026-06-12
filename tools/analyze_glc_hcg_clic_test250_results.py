#!/usr/bin/env python3
"""Aggregate official CLIC test250 metrics for GLC + HCG-RVQ runs.

The script compares seed-wise exported GLC base reconstructions and HCG-RVQ
replacement reconstructions against the official GLC test-image evaluation.
It intentionally focuses on perceptual metrics for paper-claim analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


LOWER_IS_BETTER = ("bpp", "fid", "kid", "dists", "lpips")
HIGHER_IS_BETTER = ("ms_ssim",)
METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER
PERCEPTUAL_METRICS = ("fid", "kid", "dists", "lpips")


@dataclass(frozen=True)
class Row:
    source: str
    seed: str
    label: str
    q: int
    bpp: float
    fid: float
    kid: float
    dists: float
    lpips: float
    ms_ssim: float

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "seed": self.seed,
            "label": self.label,
            "q": self.q,
            "bpp": self.bpp,
            "fid": self.fid,
            "kid": self.kid,
            "dists": self.dists,
            "lpips": self.lpips,
            "ms_ssim": self.ms_ssim,
        }


def parse_res(path: Path) -> dict[str, float]:
    text = path.read_text()
    pairs = dict(re.findall(r"([a-zA-Z0-9-]+)\s*=\s*([-+0-9.eE]+)", text))
    required = ("bpp", "ms-ssim", "lpips", "dists", "fid", "kid")
    missing = [key for key in required if key not in pairs]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return {
        "bpp": float(pairs["bpp"]),
        "ms_ssim": float(pairs["ms-ssim"]),
        "lpips": float(pairs["lpips"]),
        "dists": float(pairs["dists"]),
        "fid": float(pairs["fid"]),
        "kid": float(pairs["kid"]),
    }


def seed_from_dir(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot parse seed from {path}")
    return match.group(1)


def load_rows(analysis_root: Path) -> list[Row]:
    rows: list[Row] = []
    official_root = analysis_root / "glc_official_clicpro_test_eval"
    if official_root.exists():
        for res_path in sorted(official_root.glob("q*/res.txt")):
            q = int(res_path.parent.name[1:])
            metrics = parse_res(res_path)
            rows.append(Row("official", "official", "glc_official", q, **metrics))

    for seed_dir in sorted(analysis_root.glob("e393_glc_hcg_seed*_clic_test250_export")):
        seed = seed_from_dir(seed_dir)
        for label in ("base", "replacement_soft"):
            for res_path in sorted((seed_dir / label).glob("q*/res.txt")):
                q = int(res_path.parent.name[1:])
                metrics = parse_res(res_path)
                rows.append(Row("export", seed, label, q, **metrics))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct_improvement(metric: str, candidate: float, reference: float) -> float:
    if metric in LOWER_IS_BETTER:
        return (reference - candidate) / reference * 100.0
    if metric in HIGHER_IS_BETTER:
        return (candidate - reference) / reference * 100.0
    raise KeyError(metric)


def summarize_by_q(rows: list[Row], label: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    qs = sorted({row.q for row in rows if row.label == label})
    for q in qs:
        subset = [row for row in rows if row.label == label and row.q == q]
        if not subset:
            continue
        item: dict[str, object] = {"label": label, "q": q, "n": len(subset)}
        for metric in METRICS:
            values = [getattr(row, metric) for row in subset]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def summarize_pairwise(rows: list[Row]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seeds = sorted({row.seed for row in rows if row.label == "replacement_soft"})
    qs = sorted({row.q for row in rows if row.label == "replacement_soft"})
    by_key = {(row.seed, row.label, row.q): row for row in rows}
    for seed in seeds:
        for q in qs:
            ref = by_key.get((seed, "base", q))
            cand = by_key.get((seed, "replacement_soft", q))
            if ref is None or cand is None:
                continue
            item: dict[str, object] = {"seed": seed, "q": q}
            for metric in METRICS:
                item[f"{metric}_base"] = getattr(ref, metric)
                item[f"{metric}_hcg"] = getattr(cand, metric)
                item[f"{metric}_delta"] = getattr(cand, metric) - getattr(ref, metric)
                item[f"{metric}_improvement_pct"] = pct_improvement(
                    metric, getattr(cand, metric), getattr(ref, metric)
                )
            out.append(item)
    return out


def mean_pairwise_by_q(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    qs = sorted({int(row["q"]) for row in pair_rows})
    for q in qs:
        subset = [row for row in pair_rows if int(row["q"]) == q]
        item: dict[str, object] = {"q": q, "n": len(subset)}
        for metric in METRICS:
            for suffix in ("base", "hcg", "delta", "improvement_pct"):
                key = f"{metric}_{suffix}"
                values = [float(row[key]) for row in subset]
                item[f"{key}_mean"] = mean(values)
                item[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def pooled_pairwise(pair_rows: list[dict[str, object]]) -> dict[str, object]:
    item: dict[str, object] = {"n": len(pair_rows)}
    for metric in METRICS:
        for suffix in ("base", "hcg", "delta", "improvement_pct"):
            key = f"{metric}_{suffix}"
            values = [float(row[key]) for row in pair_rows]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return item


def mean_curve(rows: list[Row], label: str) -> list[dict[str, float]]:
    by_q = summarize_by_q(rows, label)
    curve: list[dict[str, float]] = []
    for row in by_q:
        point = {"q": float(row["q"])}
        for metric in METRICS:
            point[metric] = float(row[f"{metric}_mean"])
        curve.append(point)
    return sorted(curve, key=lambda item: item["q"])


def interp_x_at_y(points: list[tuple[float, float]], y: float) -> float:
    pts = sorted(points, key=lambda pair: pair[1])
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            if y1 == y0:
                return (x0 + x1) / 2.0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    raise ValueError(f"y={y} outside interpolation range")


def bd_rate_like(ref_curve: list[dict[str, float]], cand_curve: list[dict[str, float]], metric: str) -> float | None:
    ref_points = [(math.log(point["bpp"]), point[metric]) for point in ref_curve]
    cand_points = [(math.log(point["bpp"]), point[metric]) for point in cand_curve]
    ref_ys = [point[1] for point in ref_points]
    cand_ys = [point[1] for point in cand_points]
    lo = max(min(ref_ys), min(cand_ys))
    hi = min(max(ref_ys), max(cand_ys))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    sample_count = 200
    ys = [lo + (hi - lo) * i / (sample_count - 1) for i in range(sample_count)]
    deltas = []
    for y in ys:
        try:
            deltas.append(interp_x_at_y(cand_points, y) - interp_x_at_y(ref_points, y))
        except ValueError:
            continue
    if not deltas:
        return None
    return (math.exp(mean(deltas)) - 1.0) * 100.0


def equal_quality_savings(ref_curve: list[dict[str, float]], cand_curve: list[dict[str, float]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for metric in PERCEPTUAL_METRICS:
        ref_points = [(math.log(point["bpp"]), point[metric]) for point in ref_curve]
        cand_points = [(math.log(point["bpp"]), point[metric]) for point in cand_curve]
        for ref_point in ref_curve:
            quality = ref_point[metric]
            try:
                cand_log_bpp = interp_x_at_y(cand_points, quality)
            except ValueError:
                continue
            cand_bpp = math.exp(cand_log_bpp)
            out.append(
                {
                    "metric": metric,
                    "reference_q": int(ref_point["q"]),
                    "reference_quality": quality,
                    "reference_bpp": ref_point["bpp"],
                    "candidate_bpp_at_equal_quality": cand_bpp,
                    "bpp_saving_pct": (ref_point["bpp"] - cand_bpp) / ref_point["bpp"] * 100.0,
                }
            )
    return out


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    path: Path,
    pair_by_q: list[dict[str, object]],
    pooled: dict[str, object],
    bd_rows: list[dict[str, object]],
    equal_rows: list[dict[str, object]],
    official_deltas: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# GLC + HCG-RVQ CLIC Test250 Summary")
    lines.append("")
    lines.append("Metrics are official CLIC Professional test split exports/evaluations. PSNR is intentionally excluded from this paper-claim summary.")
    lines.append("")
    lines.append("## 3-Seed Mean vs Exported GLC Base")
    lines.append("")
    lines.append("| q | bpp imp % | FID imp % | KID imp % | DISTS imp % | LPIPS imp % | MS-SSIM imp % |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in pair_by_q:
        lines.append(
            "| {q} | {bpp} | {fid} | {kid} | {dists} | {lpips} | {msssim} |".format(
                q=row["q"],
                bpp=fmt(row["bpp_improvement_pct_mean"], 3),
                fid=fmt(row["fid_improvement_pct_mean"], 3),
                kid=fmt(row["kid_improvement_pct_mean"], 3),
                dists=fmt(row["dists_improvement_pct_mean"], 3),
                lpips=fmt(row["lpips_improvement_pct_mean"], 3),
                msssim=fmt(row["ms_ssim_improvement_pct_mean"], 3),
            )
        )
    lines.append("")
    lines.append("## Pooled 3-Seed x 4-Quality Mean")
    lines.append("")
    lines.append("| metric | improvement % mean | improvement % std | base mean | HCG mean | delta mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        lines.append(
            "| {metric} | {imp} | {imp_std} | {base} | {hcg} | {delta} |".format(
                metric=metric,
                imp=fmt(pooled[f"{metric}_improvement_pct_mean"], 3),
                imp_std=fmt(pooled[f"{metric}_improvement_pct_std"], 3),
                base=fmt(pooled[f"{metric}_base_mean"], 6),
                hcg=fmt(pooled[f"{metric}_hcg_mean"], 6),
                delta=fmt(pooled[f"{metric}_delta_mean"], 6),
            )
        )
    lines.append("")
    lines.append("## BD-Rate-Like Bpp Change")
    lines.append("")
    lines.append("| reference | candidate | metric | bd_rate_like_bpp_change_pct |")
    lines.append("|---|---|---|---:|")
    for row in bd_rows:
        lines.append(
            f"| {row['reference']} | {row['candidate']} | {row['metric']} | {fmt(row['bd_rate_like_bpp_change_pct'], 3)} |"
        )
    lines.append("")
    lines.append("## Equal-Quality Bpp Savings")
    lines.append("")
    lines.append("| metric | reference q | reference bpp | candidate bpp | saving % |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in equal_rows:
        lines.append(
            "| {metric} | {q} | {ref_bpp} | {cand_bpp} | {saving} |".format(
                metric=row["metric"],
                q=row["reference_q"],
                ref_bpp=fmt(row["reference_bpp"], 6),
                cand_bpp=fmt(row["candidate_bpp_at_equal_quality"], 6),
                saving=fmt(row["bpp_saving_pct"], 3),
            )
        )
    lines.append("")
    lines.append("## Exported Base vs Official GLC Test-Image Eval")
    lines.append("")
    lines.append("| q | bpp delta % | FID delta % | KID delta % | DISTS delta % | LPIPS delta % |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in official_deltas:
        lines.append(
            "| {q} | {bpp} | {fid} | {kid} | {dists} | {lpips} |".format(
                q=row["q"],
                bpp=fmt(row["bpp_delta_pct"], 3),
                fid=fmt(row["fid_delta_pct"], 3),
                kid=fmt(row["kid_delta_pct"], 3),
                dists=fmt(row["dists_delta_pct"], 3),
                lpips=fmt(row["lpips_delta_pct"], 3),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def official_base_deltas(rows: list[Row]) -> list[dict[str, object]]:
    official = {(row.q): row for row in rows if row.label == "glc_official"}
    base_curve = mean_curve(rows, "base")
    out: list[dict[str, object]] = []
    for point in base_curve:
        q = int(point["q"])
        off = official.get(q)
        if off is None:
            continue
        item: dict[str, object] = {"q": q}
        for metric in LOWER_IS_BETTER:
            item[f"{metric}_official"] = getattr(off, metric)
            item[f"{metric}_base_mean"] = point[metric]
            item[f"{metric}_delta_pct"] = (point[metric] - getattr(off, metric)) / getattr(off, metric) * 100.0
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("experiments/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/analysis/e394_glc_hcg_clic_test250_summary"))
    args = parser.parse_args()

    rows = load_rows(args.analysis_root)
    if not rows:
        raise SystemExit("no rows found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "raw_metrics.csv", [row.as_dict() for row in rows], list(rows[0].as_dict().keys()))

    by_q_rows: list[dict[str, object]] = []
    for label in ("glc_official", "base", "replacement_soft"):
        by_q_rows.extend(summarize_by_q(rows, label))
    by_q_fields = sorted({key for row in by_q_rows for key in row.keys()})
    write_csv(args.output_dir / "mean_by_q.csv", by_q_rows, by_q_fields)

    pair_rows = summarize_pairwise(rows)
    pair_fields = sorted({key for row in pair_rows for key in row.keys()})
    write_csv(args.output_dir / "pairwise_seed_q.csv", pair_rows, pair_fields)

    pair_by_q = mean_pairwise_by_q(pair_rows)
    pair_by_q_fields = sorted({key for row in pair_by_q for key in row.keys()})
    write_csv(args.output_dir / "pairwise_mean_by_q.csv", pair_by_q, pair_by_q_fields)

    pooled = pooled_pairwise(pair_rows)
    write_csv(args.output_dir / "pooled_pairwise_summary.csv", [pooled], sorted(pooled.keys()))

    ref_curve = mean_curve(rows, "base")
    cand_curve = mean_curve(rows, "replacement_soft")
    official_curve = mean_curve(rows, "glc_official")

    bd_rows: list[dict[str, object]] = []
    for metric in PERCEPTUAL_METRICS:
        bd_rows.append(
            {
                "reference": "export_base_3seed_mean",
                "candidate": "hcg_replacement_soft_3seed_mean",
                "metric": metric,
                "bd_rate_like_bpp_change_pct": bd_rate_like(ref_curve, cand_curve, metric),
            }
        )
        if official_curve:
            bd_rows.append(
                {
                    "reference": "official_glc_test_image",
                    "candidate": "hcg_replacement_soft_3seed_mean",
                    "metric": metric,
                    "bd_rate_like_bpp_change_pct": bd_rate_like(official_curve, cand_curve, metric),
                }
            )
    write_csv(args.output_dir / "bd_rate_like.csv", bd_rows, list(bd_rows[0].keys()))

    equal_rows = equal_quality_savings(ref_curve, cand_curve)
    write_csv(args.output_dir / "equal_quality_bpp_savings.csv", equal_rows, list(equal_rows[0].keys()) if equal_rows else [])

    official_deltas = official_base_deltas(rows)
    write_csv(args.output_dir / "official_vs_export_base.csv", official_deltas, sorted({key for row in official_deltas for key in row.keys()}))

    write_summary(args.output_dir / "summary.md", pair_by_q, pooled, bd_rows, equal_rows, official_deltas)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
