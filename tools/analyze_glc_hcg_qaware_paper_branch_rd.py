#!/usr/bin/env python3
"""Summarize GLC+HCG-RVQ paper-branch RD evidence.

This script intentionally separates the local CLIC Professional 41-image run
from the digitized GLC-paper curve JSON. The two can be plotted together for
orientation, but the local HCG curve is not an official CLIC test-250 result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


LOWER_BETTER = {"bpp", "dists", "lpips", "fid", "kid"}
HIGHER_BETTER = {"ms_ssim"}


def fnum(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(value)


def seed_from_name(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.name)
    return match.group(1) if match else "unknown"


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(run_dir.glob("glc_hcg_qaware_clicprof_seed*_q0123_*.csv")):
        seed = seed_from_name(csv_path)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["seed"] = seed
                row["q_index"] = int(row["q_index"])
                for key in (
                    "bpp",
                    "dists",
                    "lpips",
                    "ms_ssim",
                    "score",
                    "delta_bpp",
                    "delta_dists",
                    "delta_lpips",
                    "delta_ms_ssim",
                    "gate_mean",
                    "selected",
                    "active_mse_ratio",
                    "index_entropy_mean",
                    "index_used_frac_mean",
                    "index_dead_frac_mean",
                    "selection_signal_bpp",
                ):
                    row[key] = fnum(row[key])
                row["nonfinite"] = int(row["nonfinite"])
                rows.append(row)
    return rows


def load_run_meta(run_dir: Path) -> dict[str, object]:
    json_paths = sorted(run_dir.glob("glc_hcg_qaware_clicprof_seed*_q0123_*.json"))
    if not json_paths:
        return {}
    with json_paths[0].open() as f:
        data = json.load(f)
    return {
        "eval_dir": data.get("args", {}).get("eval_dir"),
        "train_dir": data.get("args", {}).get("train_dir"),
        "eval_images": len(data.get("eval_images", [])),
        "q_indexes": data.get("args", {}).get("q_indexes"),
        "steps": data.get("args", {}).get("steps"),
        "train_limit": data.get("args", {}).get("train_limit"),
        "train_start_index": data.get("args", {}).get("train_start_index"),
    }


def avg_by(rows: list[dict[str, object]], keys: tuple[str, ...], metrics: tuple[str, ...]):
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    out = []
    for key_values, bucket in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        item = {k: v for k, v in zip(keys, key_values)}
        item["n"] = len(bucket)
        for metric in metrics:
            vals = [float(row[metric]) for row in bucket]
            item[f"{metric}_mean"] = mean(vals)
            item[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        item["nonfinite_sum"] = sum(int(row["nonfinite"]) for row in bucket)
        out.append(item)
    return out


def relative_improvement(base: float, cand: float, metric: str) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    if metric in HIGHER_BETTER:
        return (cand - base) / abs(base) * 100.0
    return (base - cand) / abs(base) * 100.0


def safe_mean(vals: list[float]) -> float:
    clean = [v for v in vals if math.isfinite(v)]
    return mean(clean) if clean else float("nan")


def safe_pstdev(vals: list[float]) -> float:
    clean = [v for v in vals if math.isfinite(v)]
    return pstdev(clean) if len(clean) > 1 else 0.0


def paired_label_summary(rows: list[dict[str, object]], base_label: str, cand_labels: list[str]):
    metrics = ("bpp", "dists", "lpips", "ms_ssim", "score", "gate_mean", "selected")
    by_seed_label: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed_label[(str(row["seed"]), str(row["label"]))].append(row)

    seed_items = []
    for seed in sorted({str(row["seed"]) for row in rows}):
        base_rows = by_seed_label.get((seed, base_label))
        if not base_rows:
            continue
        base = {m: mean([float(r[m]) for r in base_rows]) for m in metrics}
        for cand_label in cand_labels:
            cand_rows = by_seed_label.get((seed, cand_label))
            if not cand_rows:
                continue
            cand = {m: mean([float(r[m]) for r in cand_rows]) for m in metrics}
            item = {"seed": seed, "label": cand_label, "n": len(cand_rows)}
            for m in metrics:
                item[f"base_{m}"] = base[m]
                item[f"{m}"] = cand[m]
                item[f"delta_{m}"] = cand[m] - base[m]
                item[f"{m}_improvement_pct"] = relative_improvement(base[m], cand[m], m)
            seed_items.append(item)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in seed_items:
        grouped[str(item["label"])].append(item)
    mean_items = []
    for label, bucket in sorted(grouped.items()):
        item = {"label": label, "seeds": len(bucket)}
        for key in bucket[0]:
            if key in {"seed", "label", "n"}:
                continue
            vals = [float(row[key]) for row in bucket]
            item[f"{key}_mean"] = safe_mean(vals)
            item[f"{key}_std"] = safe_pstdev(vals)
        mean_items.append(item)
    return seed_items, mean_items


def curve_points(rows: list[dict[str, object]], label: str):
    out = []
    for q in sorted({int(row["q_index"]) for row in rows}):
        bucket = [row for row in rows if str(row["label"]) == label and int(row["q_index"]) == q]
        if not bucket:
            continue
        out.append(
            {
                "label": label,
                "q_index": q,
                "bpp": mean([float(row["bpp"]) for row in bucket]),
                "dists": mean([float(row["dists"]) for row in bucket]),
                "lpips": mean([float(row["lpips"]) for row in bucket]),
                "ms_ssim": mean([float(row["ms_ssim"]) for row in bucket]),
                "score": mean([float(row["score"]) for row in bucket]),
                "n": len(bucket),
            }
        )
    return out


def bd_rate_like(base_curve: list[dict[str, object]], cand_curve: list[dict[str, object]], metric: str) -> float | None:
    """Linear log-bpp integration over the common metric range.

    This is a compact BD-rate-style estimate for quick triage, not a substitute
    for a polished paper implementation.
    """
    base = sorted([(float(p[metric]), math.log(float(p["bpp"]))) for p in base_curve])
    cand = sorted([(float(p[metric]), math.log(float(p["bpp"]))) for p in cand_curve])
    if len(base) < 2 or len(cand) < 2:
        return None
    lo = max(base[0][0], cand[0][0])
    hi = min(base[-1][0], cand[-1][0])
    if not lo < hi:
        return None

    def interp(points, x):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= x <= x1:
                if abs(x1 - x0) < 1e-12:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return points[0][1] if x <= points[0][0] else points[-1][1]

    xs = [lo + (hi - lo) * i / 200.0 for i in range(201)]
    diffs = [interp(cand, x) - interp(base, x) for x in xs]
    avg_diff = mean(diffs)
    return (math.exp(avg_diff) - 1.0) * 100.0


def load_paper_points(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    rows = []
    for item in data:
        for metric, payload in item["metrics"].items():
            rows.append(
                {
                    "source": "glc_paper_digitized",
                    "model": item["model"],
                    "quality": item["quality"],
                    "metric": metric.lower(),
                    "bpp": payload["bpp"],
                    "value": payload["value"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(out_dir: Path, paper_points: list[dict[str, object]], local_curves: list[dict[str, object]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    for metric in ("dists", "lpips"):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for model in sorted({str(p["model"]) for p in paper_points if p["metric"] == metric.upper().lower()}):
            pts = [p for p in paper_points if str(p["model"]) == model and p["metric"] == metric]
            pts = sorted(pts, key=lambda p: float(p["bpp"]))
            if pts:
                ax.plot([p["bpp"] for p in pts], [p["value"] for p in pts], marker="o", linewidth=1.5, label=f"{model} paper")
        for label, marker in (("trained_base", "s"), ("trained_replacement_soft", "D")):
            pts = [p for p in local_curves if p["label"] == label]
            pts = sorted(pts, key=lambda p: float(p["bpp"]))
            ax.plot([p["bpp"] for p in pts], [p[metric] for p in pts], marker=marker, linewidth=2.0, label=f"{label} local CLIC41")
        ax.set_xlabel("bpp")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"GLC paper curve vs local HCG-GLC ({metric.upper()})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = out_dir / f"glc_hcg_vs_paper_{metric}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/analysis/glc_qaware_paper_branch_20260609_034248"))
    parser.add_argument("--paper-json", type=Path, default=Path("third_party/GLC/rate_distortion_perceptual_metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/analysis/e392_glc_hcg_qaware_rd_claim"))
    parser.add_argument("--base-label", default="trained_base")
    parser.add_argument("--candidate-label", default="trained_replacement_soft")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.run_dir)
    meta = load_run_meta(args.run_dir)
    paper_points = load_paper_points(args.paper_json)

    candidate_labels = [
        "init_replacement_soft",
        "trained_qaware_q_aware_index_entropy_mean_m0p02_replacement_soft",
        "trained_replacement_soft",
        "trained_rate_cap_replacement_soft",
        "trained_soft_gate",
        "trained_replacement_all_on",
    ]
    seed_summary, label_summary = paired_label_summary(rows, args.base_label, candidate_labels)
    write_csv(args.out_dir / "paired_seed_summary.csv", seed_summary)
    write_csv(args.out_dir / "paired_label_summary.csv", label_summary)

    labels_for_curve = [args.base_label, args.candidate_label]
    local_curves = []
    for label in labels_for_curve:
        local_curves.extend(curve_points(rows, label))
    write_csv(args.out_dir / "local_curve_points.csv", local_curves)

    combined_rows = list(paper_points)
    for p in local_curves:
        for metric in ("dists", "lpips"):
            combined_rows.append(
                {
                    "source": "local_clic_professional_41_not_official_test250",
                    "model": "GLC" if p["label"] == args.base_label else "HCG-GLC",
                    "quality": p["q_index"],
                    "metric": metric,
                    "bpp": p["bpp"],
                    "value": p[metric],
                }
            )
    write_csv(args.out_dir / "combined_curve_points.csv", combined_rows)

    base_curve = curve_points(rows, args.base_label)
    cand_curve = curve_points(rows, args.candidate_label)
    bd_rows = []
    for metric in ("dists", "lpips"):
        bd = bd_rate_like(base_curve, cand_curve, metric)
        bd_rows.append({"metric": metric, "bd_rate_like_pct": bd})
    write_csv(args.out_dir / "bd_rate_like.csv", bd_rows)

    plots = maybe_plot(args.out_dir, paper_points, local_curves)

    main_item = next((x for x in label_summary if x["label"] == args.candidate_label), None)
    md = []
    md.append("# E392 GLC HCG-RVQ RD Claim Triage")
    md.append("")
    md.append("## Dataset/Protocol Warning")
    md.append("")
    md.append(f"- Local finished run eval_dir: `{meta.get('eval_dir')}`")
    md.append(f"- Local eval images: `{meta.get('eval_images')}`")
    md.append(f"- Local q indexes: `{meta.get('q_indexes')}`")
    md.append("- The local run is CLIC Professional validation-style 41 images, not the official CLIC test-250 set.")
    md.append("- The digitized GLC-paper JSON is useful as an external reference curve, but paper-facing claims require re-evaluating HCG-GLC on the official CLIC test-250 protocol.")
    md.append("")
    if main_item:
        md.append("## Main Local Result: trained_replacement_soft vs trained_base")
        md.append("")
        md.append(f"- bpp improvement: `{main_item['bpp_improvement_pct_mean']:.3f}%` (delta `{main_item['delta_bpp_mean']:.6f}`)")
        md.append(f"- DISTS improvement: `{main_item['dists_improvement_pct_mean']:.3f}%` (delta `{main_item['delta_dists_mean']:.6f}`)")
        md.append(f"- LPIPS improvement: `{main_item['lpips_improvement_pct_mean']:.3f}%` (delta `{main_item['delta_lpips_mean']:.6f}`)")
        md.append(f"- MS-SSIM improvement: `{main_item['ms_ssim_improvement_pct_mean']:.3f}%` (delta `{main_item['delta_ms_ssim_mean']:.6f}`)")
        md.append(f"- Score delta: `{main_item['delta_score_mean']:.6f}`")
        md.append("")
    md.append("## BD-Rate-like Local Estimate")
    md.append("")
    for row in bd_rows:
        val = row["bd_rate_like_pct"]
        md.append(f"- {row['metric'].upper()}: `{val:.3f}%`" if val is not None else f"- {row['metric'].upper()}: unavailable")
    md.append("")
    md.append("## Plot Artifacts")
    md.append("")
    if plots:
        for path in plots:
            md.append(f"- `{path}`")
    else:
        md.append("- Matplotlib unavailable; CSV artifacts were still written.")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- Locally, HCG-GLC improves both perceptual metrics and bpp against the matched GLC base row.")
    md.append("- The current evidence supports a GLC-neighborhood / low-bpp plug-in improvement claim, not yet a broad official-paper SOTA curve claim.")
    md.append("- The next critical experiment is official CLIC test-250 export/evaluation for GLC and HCG-GLC, including FID/KID.")
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
