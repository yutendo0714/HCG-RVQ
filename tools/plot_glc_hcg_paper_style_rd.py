#!/usr/bin/env python3
"""Plot GLC/HCG-RVQ RD curves with GLC-paper-like perceptual axes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("fid", "kid", "dists", "lpips")
PAPER_MODELS = ("FCC", "HiFiC", "MS-ILLM", "GLC")
COLORS = {
    "FCC": "#d8a11d",
    "HiFiC": "#ff8c00",
    "MS-ILLM": "#1f40ff",
    "GLC": "#ff0000",
    "Official GLC": "#2ca02c",
    "HCG-RVQ+GLC": "#d62728",
}


def load_mean_points(path: Path, label: str) -> list[dict[str, float]]:
    rows = csv.DictReader(path.open())
    points = []
    for row in rows:
        if row["label"] != label:
            continue
        points.append({
            "q": float(row["q"]),
            "bpp": float(row["bpp_mean"]),
            "fid": float(row["fid_mean"]),
            "kid": float(row["kid_mean"]),
            "dists": float(row["dists_mean"]),
            "lpips": float(row["lpips_mean"]),
        })
    return sorted(points, key=lambda item: item["q"])


def load_paper_points(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for item in data:
        model = item["model"]
        out.setdefault(model, {})
        for metric, metric_item in item["metrics"].items():
            out[model].setdefault(metric.lower(), []).append(
                (float(metric_item["bpp"]), float(metric_item["value"]))
            )
    for metric_map in out.values():
        for points in metric_map.values():
            points.sort()
    return out


def set_paper_style_axis(ax, metric: str) -> None:
    ax.set_xlim(0.020, 0.045)
    ax.set_xticks([0.020, 0.025, 0.030, 0.035, 0.040, 0.045])
    if metric == "fid":
        ticks = [2**p for p in range(2, 8)]
        ax.set_yscale("log", base=2)
        ax.set_ylim(2**2, 2**7)
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$2^{p}$" for p in range(2, 8)])
    elif metric == "kid":
        ticks = [2**p for p in range(-11, -3)]
        ax.set_yscale("log", base=2)
        ax.set_ylim(2**-11, 2**-4)
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$2^{{{p}}}$" for p in range(-11, -3)])
    elif metric == "dists":
        ax.set_ylim(0.05, 0.30)
        ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    elif metric == "lpips":
        ax.set_ylim(0.10, 0.45)
        ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])


def plot(args: argparse.Namespace) -> None:
    paper = load_paper_points(args.paper_json)
    official = load_mean_points(args.mean_by_q, "glc_official")
    hcg = load_mean_points(args.mean_by_q, "replacement_soft")

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), dpi=180)
    for ax, metric in zip(axes, METRICS):
        for model in PAPER_MODELS:
            points = paper.get(model, {}).get(metric, [])
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.5, label=model, color=COLORS[model])
        if official:
            ax.plot(
                [point["bpp"] for point in official],
                [point[metric] for point in official],
                marker="s",
                linewidth=1.4,
                markersize=3.5,
                linestyle="--",
                label="Official GLC",
                color=COLORS["Official GLC"],
            )
        ax.plot(
            [point["bpp"] for point in hcg],
            [point[metric] for point in hcg],
            marker="*",
            linewidth=1.9,
            markersize=8,
            label="HCG-RVQ+GLC",
            color=COLORS["HCG-RVQ+GLC"],
        )
        set_paper_style_axis(ax, metric)
        ax.set_title(metric.upper())
        ax.set_xlabel("bpp")
        ax.set_ylabel(metric.upper())
        ax.grid(True, which="major", alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-by-q", type=Path, default=Path("experiments/analysis/e394_glc_hcg_clic_test250_summary/mean_by_q.csv"))
    parser.add_argument("--paper-json", type=Path, default=Path("third_party/GLC/rate_distortion_perceptual_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/analysis/e394_glc_hcg_clic_test250_summary/rd_curves_paper_style.png"))
    args = parser.parse_args()
    plot(args)
    print(args.output)


if __name__ == "__main__":
    main()
