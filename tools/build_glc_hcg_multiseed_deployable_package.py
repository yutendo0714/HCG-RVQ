#!/usr/bin/env python3
"""Aggregate deployable GLC + HCG-RVQ eval packages over multiple seeds."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_glc_hcg_deployable_eval_package import (
    METRICS,
    PERCEPTUAL_METRICS,
    bd_rate_like,
    build_improvements,
    curve_for,
    equal_quality_savings,
    improvement_pct,
    load_eval_curves,
    load_paper_points,
    paper_glc_same_bpp_delta,
    plot_rd,
    write_csv,
)


def seed_name(root: Path) -> str:
    for part in root.name.split("_"):
        if part.startswith("seed"):
            return part.replace("seed", "")
    return root.name


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


def aggregate_same_q(per_seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    qs = sorted({int(row["q"]) for row in per_seed_rows})
    for q in qs:
        subset = [row for row in per_seed_rows if int(row["q"]) == q]
        item: dict[str, object] = {"q": q, "n_seeds": len(subset)}
        for metric in METRICS:
            for suffix in ("base", "candidate", "delta", "improvement_pct"):
                key = f"{metric}_{suffix}"
                values = [float(row[key]) for row in subset if key in row]
                avg, std = mean_std(values)
                item[f"{key}_mean"] = avg
                item[f"{key}_std"] = std
        out.append(item)
    return out


def aggregate_bd(per_seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    metrics = sorted({str(row["metric"]) for row in per_seed_rows})
    for metric in metrics:
        subset = [row for row in per_seed_rows if str(row["metric"]) == metric]
        values = [float(row["bd_rate_like_bpp_change_pct"]) for row in subset if row.get("bd_rate_like_bpp_change_pct") is not None]
        avg, std = mean_std(values)
        out.append({"metric": metric, "n_seeds": len(values), "bd_rate_like_bpp_change_pct_mean": avg, "bd_rate_like_bpp_change_pct_std": std})
    return out


def aggregate_export_rows(eval_roots: list[Path]) -> list[dict[str, object]]:
    import json

    rows = []
    for root in eval_roots:
        path = root / "export_rows.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        seed = seed_name(root)
        for row in payload.get("rows", payload if isinstance(payload, list) else []):
            out = {"seed": seed, "q": int(row.get("q_index", row.get("q")))}
            for key in (
                "base_bpp",
                "branch_bpp",
                "replacement_bpp",
                "soft_gate_mean",
                "quantized_soft_gate_mean",
                "hard_gate_mean",
                "active_mse_ratio",
                "index_entropy_mean",
                "index_used_frac_mean",
                "index_dead_frac_mean",
            ):
                if row.get(key) is not None:
                    out[key] = float(row[key])
            rows.append(out)
    summary = []
    for q in sorted({int(row["q"]) for row in rows}):
        subset = [row for row in rows if int(row["q"]) == q]
        item: dict[str, object] = {"q": q, "n_images_times_seeds": len(subset)}
        for key in sorted({k for row in subset for k in row.keys()} - {"seed", "q"}):
            values = [float(row[key]) for row in subset if key in row]
            avg, std = mean_std(values)
            item[f"{key}_mean"] = avg
            item[f"{key}_std"] = std
        summary.append(item)
    return summary


def average_curve(seed_curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    points = []
    qs = sorted({int(point["q"]) for curve in seed_curves for point in curve})
    for q in qs:
        subset = [point for curve in seed_curves for point in curve if int(point["q"]) == q]
        item = {"q": float(q)}
        for metric in METRICS:
            values = [float(point[metric]) for point in subset if metric in point]
            item[metric] = mean(values)
        points.append(item)
    return points


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    path: Path,
    candidate_label: str,
    same_q: list[dict[str, object]],
    bd_rows: list[dict[str, object]],
    eq_rows: list[dict[str, object]],
    paper_rows: list[dict[str, object]],
    export_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# 3-Seed Deployable GLC + HCG-RVQ CLIC250 Summary",
        "",
        f"Candidate label: `{candidate_label}`.",
        "Positive improvement means better. PSNR is excluded from this claim summary.",
        "",
        "## Same-q Mean Improvement vs Exported GLC Base",
        "",
        "| q | bpp imp % | FID imp % | KID imp % | DISTS imp % | LPIPS imp % | MS-SSIM imp % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in same_q:
        lines.append(
            "| {q} | {bpp} | {fid} | {kid} | {dists} | {lpips} | {ms} |".format(
                q=row["q"],
                bpp=fmt(row["bpp_improvement_pct_mean"]),
                fid=fmt(row["fid_improvement_pct_mean"]),
                kid=fmt(row["kid_improvement_pct_mean"]),
                dists=fmt(row["dists_improvement_pct_mean"]),
                lpips=fmt(row["lpips_improvement_pct_mean"]),
                ms=fmt(row["ms_ssim_improvement_pct_mean"]),
            )
        )
    lines.extend(["", "## Mean BD-Rate-Like Bpp Change vs Exported GLC Base", ""])
    lines.extend(["| metric | mean % | std % |", "|---|---:|---:|"])
    for row in bd_rows:
        lines.append(
            f"| {row['metric']} | {fmt(row['bd_rate_like_bpp_change_pct_mean'])} | {fmt(row['bd_rate_like_bpp_change_pct_std'])} |"
        )
    lines.extend(["", "## Equal-Quality Bpp Savings on 3-Seed Mean Curves", ""])
    lines.extend(["| metric | reference q | reference bpp | candidate bpp | saving % |", "|---|---:|---:|---:|---:|"])
    for row in eq_rows:
        lines.append(
            "| {metric} | {q} | {ref_bpp} | {cand_bpp} | {saving} |".format(
                metric=row["metric"],
                q=row["reference_q"],
                ref_bpp=fmt(row["reference_bpp"], 6),
                cand_bpp=fmt(row["candidate_bpp_at_equal_quality"], 6),
                saving=fmt(row["bpp_saving_pct"]),
            )
        )
    if paper_rows:
        lines.extend(["", "## Orientation vs Digitized Paper GLC at Same Bpp", ""])
        lines.extend(["| metric | q | bpp | paper GLC | candidate | improvement % |", "|---|---:|---:|---:|---:|---:|"])
        for row in paper_rows:
            lines.append(
                "| {metric} | {q} | {bpp} | {paper} | {candidate} | {imp} |".format(
                    metric=row["metric"],
                    q=row["q"],
                    bpp=fmt(row["candidate_bpp"], 6),
                    paper=fmt(row["paper_glc_value_at_same_bpp"], 6),
                    candidate=fmt(row["candidate_value"], 6),
                    imp=fmt(row["improvement_pct_vs_paper_glc_same_bpp"]),
                )
            )
    lines.extend(["", "## Mechanism Signals", ""])
    lines.extend(["| q | replacement bpp | soft gate | index entropy | active mse ratio |", "|---:|---:|---:|---:|---:|"])
    for row in export_rows:
        lines.append(
            "| {q} | {rbpp} | {gate} | {entropy} | {mse} |".format(
                q=row["q"],
                rbpp=fmt(row.get("replacement_bpp_mean"), 6),
                gate=fmt(row.get("soft_gate_mean_mean"), 4),
                entropy=fmt(row.get("index_entropy_mean_mean"), 4),
                mse=fmt(row.get("active_mse_ratio_mean"), 4),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-label", default="replacement_soft_qgate8b_sig8b")
    parser.add_argument("--paper-json", type=Path, default=Path("third_party/GLC/rate_distortion_perceptual_metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    per_seed_imp: list[dict[str, object]] = []
    per_seed_bd: list[dict[str, object]] = []
    base_curves = []
    cand_curves = []
    for root in args.eval_roots:
        seed = seed_name(root)
        rows = load_eval_curves(root)
        for row in rows:
            raw_rows.append({"seed": seed, **row})
        for row in build_improvements(rows, args.candidate_label):
            per_seed_imp.append({"seed": seed, **row})
        base_curve = curve_for(rows, "base")
        cand_curve = curve_for(rows, args.candidate_label)
        base_curves.append(base_curve)
        cand_curves.append(cand_curve)
        for metric in PERCEPTUAL_METRICS:
            per_seed_bd.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "bd_rate_like_bpp_change_pct": bd_rate_like(base_curve, cand_curve, metric),
                }
            )

    same_q = aggregate_same_q(per_seed_imp)
    bd_rows = aggregate_bd(per_seed_bd)
    base_mean_curve = average_curve(base_curves)
    cand_mean_curve = average_curve(cand_curves)
    eq_rows = equal_quality_savings(base_mean_curve, cand_mean_curve)
    paper_points = load_paper_points(args.paper_json) if args.paper_json.exists() else []
    paper_rows = paper_glc_same_bpp_delta(paper_points, cand_mean_curve) if paper_points else []
    export_rows = aggregate_export_rows(args.eval_roots)

    # Mean-curve same-bpp paper orientation is useful, but local-base claims use
    # paired per-seed rows above.
    write_csv(args.output_dir / "raw_metrics_by_seed.csv", raw_rows)
    write_csv(args.output_dir / "same_q_improvement_by_seed.csv", per_seed_imp)
    write_csv(args.output_dir / "same_q_improvement_3seed_mean.csv", same_q)
    write_csv(args.output_dir / "bd_rate_like_by_seed.csv", per_seed_bd)
    write_csv(args.output_dir / "bd_rate_like_3seed_mean.csv", bd_rows)
    write_csv(args.output_dir / "equal_quality_bpp_savings_3seed_mean.csv", eq_rows)
    write_csv(args.output_dir / "paper_glc_same_bpp_orientation_3seed_mean.csv", paper_rows)
    write_csv(args.output_dir / "mechanism_export_rows_3seed_mean.csv", export_rows)
    plot_rd(args.output_dir / "rd_curves_paper_style_3seed_mean.png", paper_points, base_mean_curve, cand_mean_curve, "HCG-RVQ+GLC deployable 3seed")
    write_summary(args.output_dir / "summary.md", args.candidate_label, same_q, bd_rows, eq_rows, paper_rows, export_rows)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
