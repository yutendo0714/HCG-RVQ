#!/usr/bin/env python3
"""Build paper-facing summaries for deployable GLC + HCG-RVQ eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


LOWER_IS_BETTER = {"bpp", "fid", "kid", "dists", "lpips"}
HIGHER_IS_BETTER = {"ms_ssim"}
METRICS = ("bpp", "fid", "kid", "dists", "lpips", "ms_ssim")
PERCEPTUAL_METRICS = ("fid", "kid", "dists", "lpips")
PAPER_MODELS = ("FCC", "HiFiC", "MS-ILLM", "GLC")

RES_RE = re.compile(
    r"bpp\s*=\s*([0-9.eE+-]+).*?"
    r"ms-ssim\s*=\s*([0-9.eE+-]+).*?"
    r"lpips\s*=\s*([0-9.eE+-]+).*?"
    r"dists\s*=\s*([0-9.eE+-]+)"
    r"(?:.*?fid\s*=\s*([0-9.eE+-]+))?"
    r"(?:.*?kid\s*=\s*([0-9.eE+-]+))?",
    re.S,
)


def read_res(path: Path) -> dict[str, float]:
    text = path.read_text()
    match = RES_RE.search(text)
    if match is None:
        raise ValueError(f"could not parse metrics from {path}")
    bpp, ms_ssim, lpips, dists, fid, kid = match.groups()
    out = {
        "bpp": float(bpp),
        "ms_ssim": float(ms_ssim),
        "lpips": float(lpips),
        "dists": float(dists),
    }
    if fid is not None:
        out["fid"] = float(fid)
    if kid is not None:
        out["kid"] = float(kid)
    return out


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_eval_curves(eval_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        for res_path in sorted(label_dir.glob("q*/res.txt")):
            q = int(res_path.parent.name[1:])
            vals = read_res(res_path)
            rows.append({"label": label_dir.name, "q": q, **vals})
    if not rows:
        raise ValueError(f"no res.txt files found under {eval_root}")
    return rows


def improvement_pct(metric: str, base: float, candidate: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    if metric in HIGHER_IS_BETTER:
        return (candidate - base) / base * 100.0
    return (base - candidate) / base * 100.0


def curve_for(rows: list[dict[str, object]], label: str) -> list[dict[str, float]]:
    curve = []
    for row in rows:
        if row["label"] != label:
            continue
        curve.append({key: float(row[key]) for key in ("q", *METRICS) if key in row})
    return sorted(curve, key=lambda row: row["q"])


def build_improvements(rows: list[dict[str, object]], candidate_label: str) -> list[dict[str, object]]:
    by_key = {(str(row["label"]), int(row["q"])): row for row in rows}
    out: list[dict[str, object]] = []
    for q in sorted(int(row["q"]) for row in rows if row["label"] == "base"):
        base = by_key.get(("base", q))
        cand = by_key.get((candidate_label, q))
        if base is None or cand is None:
            continue
        item: dict[str, object] = {"candidate_label": candidate_label, "q": q}
        for metric in METRICS:
            if metric not in base or metric not in cand:
                continue
            item[f"{metric}_base"] = float(base[metric])
            item[f"{metric}_candidate"] = float(cand[metric])
            item[f"{metric}_delta"] = float(cand[metric]) - float(base[metric])
            item[f"{metric}_improvement_pct"] = improvement_pct(metric, float(base[metric]), float(cand[metric]))
        out.append(item)
    return out


def interp_x_at_y(points: list[tuple[float, float]], y: float) -> float:
    points = sorted(points, key=lambda pair: pair[1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            if abs(y1 - y0) < 1e-12:
                return (x0 + x1) / 2.0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    raise ValueError(f"target={y} outside curve range")


def bd_rate_like(ref_curve: list[dict[str, float]], cand_curve: list[dict[str, float]], metric: str) -> float | None:
    ref_points = [(math.log(point["bpp"]), point[metric]) for point in ref_curve]
    cand_points = [(math.log(point["bpp"]), point[metric]) for point in cand_curve]
    ref_ys = [point[1] for point in ref_points]
    cand_ys = [point[1] for point in cand_points]
    lo = max(min(ref_ys), min(cand_ys))
    hi = min(max(ref_ys), max(cand_ys))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    samples = [lo + (hi - lo) * i / 199.0 for i in range(200)]
    deltas = []
    for y in samples:
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
        cand_points = [(math.log(point["bpp"]), point[metric]) for point in cand_curve]
        for ref in ref_curve:
            try:
                cand_log_bpp = interp_x_at_y(cand_points, ref[metric])
            except ValueError:
                continue
            cand_bpp = math.exp(cand_log_bpp)
            out.append(
                {
                    "metric": metric,
                    "reference_q": int(ref["q"]),
                    "reference_bpp": ref["bpp"],
                    "reference_quality": ref[metric],
                    "candidate_bpp_at_equal_quality": cand_bpp,
                    "bpp_saving_pct": (ref["bpp"] - cand_bpp) / ref["bpp"] * 100.0,
                }
            )
    return out


def load_paper_points(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    rows = []
    for item in data:
        for metric, metric_item in item["metrics"].items():
            rows.append(
                {
                    "model": item["model"],
                    "quality": item["quality"],
                    "metric": metric.lower(),
                    "bpp": float(metric_item["bpp"]),
                    "value": float(metric_item["value"]),
                }
            )
    return rows


def paper_glc_same_bpp_delta(
    paper_points: list[dict[str, object]],
    cand_curve: list[dict[str, float]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in paper_points:
        if row["model"] == "GLC" and str(row["metric"]) in PERCEPTUAL_METRICS:
            grouped[str(row["metric"])].append((float(row["bpp"]), float(row["value"])))
    for metric, points in grouped.items():
        points = sorted(points)
        for cand in cand_curve:
            bpp = cand["bpp"]
            if bpp < points[0][0] or bpp > points[-1][0]:
                continue
            paper_value = interp_x_at_y([(value, bpp_) for bpp_, value in points], bpp)
            cand_value = cand[metric]
            out.append(
                {
                    "metric": metric,
                    "q": int(cand["q"]),
                    "candidate_bpp": bpp,
                    "paper_glc_value_at_same_bpp": paper_value,
                    "candidate_value": cand_value,
                    "improvement_pct_vs_paper_glc_same_bpp": improvement_pct(metric, paper_value, cand_value),
                }
            )
    return out


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
    improvements: list[dict[str, object]],
    bd_rows: list[dict[str, object]],
    eq_rows: list[dict[str, object]],
    paper_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Deployable GLC + HCG-RVQ Evaluation Package",
        "",
        f"Candidate label: `{candidate_label}`.",
        "Positive improvement means better. PSNR is intentionally excluded from the claim summary.",
        "",
        "## Same-q Improvement vs Exported GLC Base",
        "",
        "| q | bpp imp % | FID imp % | KID imp % | DISTS imp % | LPIPS imp % | MS-SSIM imp % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in improvements:
        lines.append(
            "| {q} | {bpp} | {fid} | {kid} | {dists} | {lpips} | {ms} |".format(
                q=row["q"],
                bpp=fmt(row.get("bpp_improvement_pct")),
                fid=fmt(row.get("fid_improvement_pct")),
                kid=fmt(row.get("kid_improvement_pct")),
                dists=fmt(row.get("dists_improvement_pct")),
                lpips=fmt(row.get("lpips_improvement_pct")),
                ms=fmt(row.get("ms_ssim_improvement_pct")),
            )
        )
    lines.extend(["", "## BD-Rate-Like Bpp Change vs Exported GLC Base", ""])
    lines.extend(["| metric | bd_rate_like_bpp_change_pct |", "|---|---:|"])
    for row in bd_rows:
        lines.append(f"| {row['metric']} | {fmt(row['bd_rate_like_bpp_change_pct'])} |")
    lines.extend(["", "## Equal-Quality Bpp Savings vs Exported GLC Base", ""])
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
    path.write_text("\n".join(lines) + "\n")


def plot_rd(
    path: Path,
    paper_points: list[dict[str, object]],
    base_curve: list[dict[str, float]],
    cand_curve: list[dict[str, float]],
    candidate_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable: {exc}")
        return

    paper_by_metric: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in paper_points:
        if row["model"] in PAPER_MODELS and row["metric"] in PERCEPTUAL_METRICS:
            paper_by_metric[(str(row["model"]), str(row["metric"]))].append((float(row["bpp"]), float(row["value"])))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), dpi=180)
    colors = {"FCC": "#d8a11d", "HiFiC": "#ff8c00", "MS-ILLM": "#1f40ff", "GLC": "#ff0000"}
    for ax, metric in zip(axes, PERCEPTUAL_METRICS):
        for model in PAPER_MODELS:
            pts = sorted(paper_by_metric.get((model, metric), []))
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", linewidth=1.2, markersize=3.2, label=f"{model} paper", color=colors[model])
        ax.plot([p["bpp"] for p in base_curve], [p[metric] for p in base_curve], marker="s", linewidth=1.4, linestyle="--", label="exported GLC base", color="#2ca02c")
        ax.plot([p["bpp"] for p in cand_curve], [p[metric] for p in cand_curve], marker="*", linewidth=2.0, markersize=8, label=candidate_name, color="#d62728")
        ax.set_title(metric.upper())
        ax.set_xlabel("bpp")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.28)
        if metric in {"fid", "kid"}:
            ax.set_yscale("log", base=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--candidate-label", default="replacement_soft_qgate8b_sig8b")
    parser.add_argument("--paper-json", type=Path, default=Path("third_party/GLC/rate_distortion_perceptual_metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_eval_curves(args.eval_root)
    base_curve = curve_for(rows, "base")
    cand_curve = curve_for(rows, args.candidate_label)
    if not base_curve:
        raise ValueError("base curve is missing")
    if not cand_curve:
        raise ValueError(f"candidate curve is missing: {args.candidate_label}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mean_by_q.csv", rows)
    improvements = build_improvements(rows, args.candidate_label)
    write_csv(args.output_dir / "same_q_improvement_vs_base.csv", improvements)

    bd_rows = [
        {"metric": metric, "bd_rate_like_bpp_change_pct": bd_rate_like(base_curve, cand_curve, metric)}
        for metric in PERCEPTUAL_METRICS
    ]
    write_csv(args.output_dir / "bd_rate_like_vs_base.csv", bd_rows)

    eq_rows = equal_quality_savings(base_curve, cand_curve)
    write_csv(args.output_dir / "equal_quality_bpp_savings.csv", eq_rows)

    paper_points = load_paper_points(args.paper_json) if args.paper_json.exists() else []
    write_csv(args.output_dir / "paper_points_long.csv", paper_points)
    paper_delta = paper_glc_same_bpp_delta(paper_points, cand_curve) if paper_points else []
    write_csv(args.output_dir / "paper_glc_same_bpp_orientation.csv", paper_delta)
    plot_rd(args.output_dir / "rd_curves_paper_style.png", paper_points, base_curve, cand_curve, "HCG-RVQ+GLC deployable")
    write_summary(args.output_dir / "summary.md", args.candidate_label, improvements, bd_rows, eq_rows, paper_delta)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
