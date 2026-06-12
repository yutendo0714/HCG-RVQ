#!/usr/bin/env python3
"""Publication-style RD curve plots for GLC + HCG-RVQ.

This intentionally excludes FCC and the local official-GLC re-evaluation curve.
The comparison is kept to paper SOTA baselines plus the official CLIC test250
HCG-RVQ+GLC 3-seed mean.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

METRICS = ["FID", "KID", "DISTS", "LPIPS"]
PAPER_MODELS = ["HiFiC", "MS-ILLM", "GLC"]
MODEL_LABELS = {"HiFiC": "HiFiC", "MS-ILLM": "MS-ILLM", "GLC": "GLC (paper)", "HCG": "HCG-RVQ + GLC"}
COLORS = {"HiFiC": "#4C78A8", "MS-ILLM": "#F58518", "GLC": "#54A24B", "HCG": "#D62728"}
MARKERS = {"HiFiC": "o", "MS-ILLM": "s", "GLC": "^", "HCG": "*"}


def load_paper(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    data = json.loads(path.read_text())
    curves: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for item in data:
        model = item["model"]
        if model not in PAPER_MODELS:
            continue
        curves.setdefault(model, {})
        for metric in METRICS:
            if metric not in item["metrics"]:
                continue
            m = item["metrics"][metric]
            curves[model].setdefault(metric, []).append((float(m["bpp"]), float(m["value"])))
    for model in curves.values():
        for pts in model.values():
            pts.sort(key=lambda x: x[0])
    return curves


def load_hcg(path: Path) -> dict[str, list[tuple[float, float]]]:
    rows = list(csv.DictReader(path.open()))
    selected = sorted([r for r in rows if r["label"] == "replacement_soft"], key=lambda r: int(r["q"]))
    curves = {}
    for metric in METRICS:
        key = metric.lower() + "_mean"
        curves[metric] = [(float(r["bpp_mean"]), float(r[key])) for r in selected]
    return curves


def format_pow2(value: float, _pos=None) -> str:
    if value <= 0:
        return ""
    import math
    exp = round(math.log(value, 2))
    if abs(value - 2**exp) / value < 1e-6:
        return rf"$2^{{{exp}}}$"
    return f"{value:g}"


def setup_axis(ax, metric: str, variant: str) -> None:
    ax.set_xlabel("bpp")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_xlim(0.019, 0.045 if variant == "paper_axes" else 0.035)
    ax.set_xticks([0.02, 0.025, 0.03, 0.035] + ([] if variant == "low_bpp_zoom" else [0.04, 0.045]))
    if metric == "FID":
        ax.set_yscale("log", base=2)
        ax.set_ylim(4.0, 32.0)
        ax.yaxis.set_major_locator(FixedLocator([4, 8, 16, 32]))
        ax.yaxis.set_major_formatter(FuncFormatter(format_pow2))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif metric == "KID":
        ax.set_yscale("log", base=2)
        ax.set_ylim(2 ** -11.3, 2 ** -7.0)
        ax.yaxis.set_major_locator(FixedLocator([2**-11, 2**-10, 2**-9, 2**-8, 2**-7]))
        ax.yaxis.set_major_formatter(FuncFormatter(format_pow2))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif metric == "DISTS":
        ax.set_ylim(0.055, 0.16)
        ax.set_yticks([0.05, 0.075, 0.10, 0.125, 0.15])
    elif metric == "LPIPS":
        ax.set_ylim(0.095, 0.17)
        ax.set_yticks([0.10, 0.125, 0.15, 0.175])


def plot_curves(paper_curves, hcg_curves, output: Path, variant: str) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.35), dpi=220)
    for ax, metric in zip(axes.flatten(), METRICS):
        for model in PAPER_MODELS:
            pts = paper_curves.get(model, {}).get(metric, [])
            if not pts:
                continue
            if variant == "low_bpp_zoom":
                pts = [p for p in pts if p[0] <= 0.035]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=4.2,
                linewidth=1.45,
                label=MODEL_LABELS[model],
            )
        hpts = hcg_curves[metric]
        if variant == "low_bpp_zoom":
            hpts = [p for p in hpts if p[0] <= 0.035]
        ax.plot(
            [p[0] for p in hpts],
            [p[1] for p in hpts],
            color=COLORS["HCG"],
            marker=MARKERS["HCG"],
            markersize=8.0,
            linewidth=2.35,
            label=MODEL_LABELS["HCG"],
            zorder=5,
        )
        ax.set_title(metric)
        setup_axis(ax, metric, variant)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-json", type=Path, default=Path("third_party/GLC/rate_distortion_perceptual_metrics.json"))
    parser.add_argument("--mean-by-q", type=Path, default=Path("experiments/analysis/e394_glc_hcg_clic_test250_summary/mean_by_q.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/analysis/e394_glc_hcg_clic_test250_summary"))
    args = parser.parse_args()
    paper = load_paper(args.paper_json)
    hcg = load_hcg(args.mean_by_q)
    plot_curves(paper, hcg, args.output_dir / "rd_curves_sota_only_paper_axes", "paper_axes")
    plot_curves(paper, hcg, args.output_dir / "rd_curves_sota_only_low_bpp_zoom", "low_bpp_zoom")
    print(args.output_dir / "rd_curves_sota_only_paper_axes.png")
    print(args.output_dir / "rd_curves_sota_only_low_bpp_zoom.png")


if __name__ == "__main__":
    main()
