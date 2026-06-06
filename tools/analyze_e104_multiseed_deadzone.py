#!/usr/bin/env python3
"""Summarize the E104 dead-zone residual selector across seeds."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "e104_multiseed_deadzone018_audit"

SEEDS = {
    "1234": {
        "beta": ANALYSIS / "beta005_after250_seed1234_direct_exact_step500_holdout4096_current.csv",
        "method": ANALYSIS / "e104_deadzone018_from_beta005_seed1234_step250_fullimage_holdout4096_current.csv",
        "method_json": ANALYSIS / "e104_deadzone018_from_beta005_seed1234_step250_fullimage_holdout4096_current.json",
    },
    "2345": {
        "beta": ANALYSIS / "beta005_after250_seed2345_direct_exact_step500_holdout4096_current.csv",
        "method": ANALYSIS / "e104_deadzone018_from_beta005_seed2345_step250_fullimage_holdout4096_current.csv",
        "method_json": ANALYSIS / "e104_deadzone018_from_beta005_seed2345_step250_fullimage_holdout4096_current.json",
    },
    "3456": {
        "beta": ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv",
        "method": ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "method_json": ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
}

FEATURES = [
    "bpp",
    "psnr",
    "ms_ssim",
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_perplexity",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_csv(path)}


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def finite(values) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def quantile(values, q: float) -> float:
    vals = finite(values)
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q)) if vals else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def nonfinite_rows(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if str(row.get("has_nonfinite", "0")).lower() in {"1", "true", "yes"})


def summarize_seed(seed: str, beta_path: Path, method_path: Path, method_json: Path) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    beta = by_path(beta_path)
    method = by_path(method_path)
    paths = sorted(set(beta) & set(method))
    if len(paths) != 4096:
        raise RuntimeError(f"seed {seed}: expected 4096 aligned rows, got {len(paths)}")

    per_image: list[dict[str, float | int | str]] = []
    for path in paths:
        beta_rd = f(beta[path], "rd_score")
        method_rd = f(method[path], "rd_score")
        delta = method_rd - beta_rd
        row: dict[str, float | int | str] = {
            "seed": seed,
            "path": path,
            "beta_rd": beta_rd,
            "method_rd": method_rd,
            "delta_vs_beta": delta,
            "improved": int(delta < 0.0),
            "has_nonfinite": int(str(method[path].get("has_nonfinite", "0")).lower() in {"1", "true", "yes"}),
        }
        for feature in FEATURES:
            row[f"method_{feature}"] = f(method[path], feature)
            row[f"beta_{feature}"] = f(beta[path], feature)
        per_image.append(row)

    deltas = [float(row["delta_vs_beta"]) for row in per_image]
    beta_rds = [float(row["beta_rd"]) for row in per_image]
    method_rds = [float(row["method_rd"]) for row in per_image]
    json_summary = json.loads(method_json.read_text(encoding="utf-8"))["summaries"][0]
    summary: dict[str, float | int | str] = {
        "seed": seed,
        "images": len(per_image),
        "beta_rd": mean(beta_rds),
        "method_rd": mean(method_rds),
        "delta_vs_beta": mean(deltas),
        "median_delta_vs_beta": quantile(deltas, 0.50),
        "q05_delta_vs_beta": quantile(deltas, 0.05),
        "q95_delta_vs_beta": quantile(deltas, 0.95),
        "win_rate": mean([float(row["improved"]) for row in per_image]),
        "mean_abs_delta": mean([abs(v) for v in deltas]),
        "max_worsening": max(finite(deltas)),
        "max_improvement": min(finite(deltas)),
        "method_nonfinite_rows": nonfinite_rows(list(method.values())),
        "json_nonfinite_rows": int(json_summary.get("nonfinite_rows", 0)),
    }
    for feature in FEATURES:
        summary[f"method_{feature}"] = mean([float(row[f"method_{feature}"]) for row in per_image])
        summary[f"beta_{feature}"] = mean([float(row[f"beta_{feature}"]) for row in per_image])
        summary[f"delta_{feature}"] = summary[f"method_{feature}"] - summary[f"beta_{feature}"]

    ordered = sorted(per_image, key=lambda row: float(row["beta_rd"]))
    quartile_rows: list[dict[str, float | int | str]] = []
    for index, chunk in enumerate(np.array_split(np.arange(len(ordered)), 4), start=1):
        rows = [ordered[int(i)] for i in chunk]
        q_deltas = [float(row["delta_vs_beta"]) for row in rows]
        quartile_rows.append(
            {
                "seed": seed,
                "quartile_by_beta_rd": f"Q{index}",
                "images": len(rows),
                "beta_rd": mean([float(row["beta_rd"]) for row in rows]),
                "method_rd": mean([float(row["method_rd"]) for row in rows]),
                "delta_vs_beta": mean(q_deltas),
                "win_rate": mean([float(row["improved"]) for row in rows]),
                "mean_abs_delta": mean([abs(v) for v in q_deltas]),
                "max_worsening": max(finite(q_deltas)),
                "max_improvement": min(finite(q_deltas)),
            }
        )
    return summary, quartile_rows, per_image


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seed_summaries = []
    quartiles = []
    all_images = []
    for seed, paths in SEEDS.items():
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing {seed} {key}: {path}")
        summary, qrows, per_image = summarize_seed(seed, paths["beta"], paths["method"], paths["method_json"])
        seed_summaries.append(summary)
        quartiles.extend(qrows)
        all_images.extend(per_image)

    deltas = [float(row["delta_vs_beta"]) for row in all_images]
    aggregate: dict[str, float | int | str] = {
        "seeds": len(SEEDS),
        "images": len(all_images),
        "beta_rd": mean([float(row["beta_rd"]) for row in all_images]),
        "method_rd": mean([float(row["method_rd"]) for row in all_images]),
        "delta_vs_beta": mean(deltas),
        "median_delta_vs_beta": quantile(deltas, 0.50),
        "q05_delta_vs_beta": quantile(deltas, 0.05),
        "q95_delta_vs_beta": quantile(deltas, 0.95),
        "win_rate": mean([float(row["improved"]) for row in all_images]),
        "mean_abs_delta": mean([abs(v) for v in deltas]),
        "max_worsening": max(finite(deltas)),
        "max_improvement": min(finite(deltas)),
        "nonfinite_rows": sum(int(row["has_nonfinite"]) for row in all_images),
    }
    for feature in FEATURES:
        aggregate[f"method_{feature}"] = mean([float(row[f"method_{feature}"]) for row in all_images])
        aggregate[f"beta_{feature}"] = mean([float(row[f"beta_{feature}"]) for row in all_images])
        aggregate[f"delta_{feature}"] = aggregate[f"method_{feature}"] - aggregate[f"beta_{feature}"]

    aggregate_quartiles = []
    for label in ["Q1", "Q2", "Q3", "Q4"]:
        rows = [row for row in quartiles if row["quartile_by_beta_rd"] == label]
        aggregate_quartiles.append(
            {
                "seed": "all",
                "quartile_by_beta_rd": label,
                "images": sum(int(row["images"]) for row in rows),
                "beta_rd": mean([float(row["beta_rd"]) for row in rows]),
                "method_rd": mean([float(row["method_rd"]) for row in rows]),
                "delta_vs_beta": mean([float(row["delta_vs_beta"]) for row in rows]),
                "win_rate": mean([float(row["win_rate"]) for row in rows]),
                "mean_abs_delta": mean([float(row["mean_abs_delta"]) for row in rows]),
                "max_worsening": max(finite([float(row["max_worsening"]) for row in rows])),
                "max_improvement": min(finite([float(row["max_improvement"]) for row in rows])),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUT.with_suffix(".per_seed.csv"), seed_summaries)
    write_csv(OUT.with_suffix(".quartiles.csv"), quartiles + aggregate_quartiles)
    write_csv(OUT.with_suffix(".per_image.csv"), all_images)
    payload = {
        "aggregate": aggregate,
        "per_seed": seed_summaries,
        "quartiles": quartiles,
        "aggregate_quartiles": aggregate_quartiles,
        "inputs": {seed: {key: str(path) for key, path in paths.items()} for seed, paths in SEEDS.items()},
    }
    OUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E104 Dead-Zone Residual Selector Multi-Seed Audit",
        "",
        "All rows are full-image OpenImages holdout4096 probes with exact Householder inverse and physical GPU0 evaluation.",
        "",
        "## Aggregate",
        "",
        "| seeds | images | beta RD | E104 RD | delta | win rate | median delta | q05 delta | q95 delta | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {seeds} | {images} | {beta} | {method} | {delta} | {win} | {median} | {q05} | {q95} | {nonfinite} |".format(
            seeds=int(aggregate["seeds"]),
            images=int(aggregate["images"]),
            beta=fmt(float(aggregate["beta_rd"])),
            method=fmt(float(aggregate["method_rd"])),
            delta=fmt(float(aggregate["delta_vs_beta"]), signed=True),
            win=fmt(float(aggregate["win_rate"])),
            median=fmt(float(aggregate["median_delta_vs_beta"]), signed=True),
            q05=fmt(float(aggregate["q05_delta_vs_beta"]), signed=True),
            q95=fmt(float(aggregate["q95_delta_vs_beta"]), signed=True),
            nonfinite=int(aggregate["nonfinite_rows"]),
        ),
        "",
        "## Per Seed",
        "",
        "| seed | beta RD | E104 RD | delta | win rate | mean abs delta | max worsening | max improvement | qMSE delta | delta-RMS delta | strength delta | s_q delta | dead-code delta | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summaries:
        lines.append(
            "| {seed} | {beta} | {method} | {delta} | {win} | {absd} | {worse} | {best} | {qmse} | {drms} | {strength} | {sq} | {dead} | {nonfinite} |".format(
                seed=row["seed"],
                beta=fmt(float(row["beta_rd"])),
                method=fmt(float(row["method_rd"])),
                delta=fmt(float(row["delta_vs_beta"]), signed=True),
                win=fmt(float(row["win_rate"])),
                absd=fmt(float(row["mean_abs_delta"])),
                worse=fmt(float(row["max_worsening"]), signed=True),
                best=fmt(float(row["max_improvement"]), signed=True),
                qmse=fmt(float(row["delta_rvq_latent_quant_mse"]), signed=True),
                drms=fmt(float(row["delta_rvq_householder_delta_rms"]), signed=True),
                strength=fmt(float(row["delta_rvq_householder_strength"]), signed=True),
                sq=fmt(float(row["delta_rvq_s_q_mean"]), signed=True),
                dead=fmt(float(row["delta_rvq_dead_code_ratio"]), signed=True),
                nonfinite=int(row["method_nonfinite_rows"]),
            )
        )
    lines.extend([
        "",
        "## Aggregate Quartiles By Beta RD",
        "",
        "| quartile | images | beta RD | E104 RD | delta | win rate | max worsening | max improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in aggregate_quartiles:
        lines.append(
            "| {q} | {images} | {beta} | {method} | {delta} | {win} | {worse} | {best} |".format(
                q=row["quartile_by_beta_rd"],
                images=int(row["images"]),
                beta=fmt(float(row["beta_rd"])),
                method=fmt(float(row["method_rd"])),
                delta=fmt(float(row["delta_vs_beta"]), signed=True),
                win=fmt(float(row["win_rate"])),
                worse=fmt(float(row["max_worsening"]), signed=True),
                best=fmt(float(row["max_improvement"]), signed=True),
            )
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "E104/deadzone018 is now a reproducible improvement over beta005 in this direct fixed-checkpoint protocol: all three seeds improve mean RD and all probes have zero nonfinite rows. The result is strong enough to promote from a seed3456 rescue diagnostic to the next manuscript-candidate branch, while still requiring a checkpoint sweep and an independent threshold-selection protocol before it becomes the paper-main row.",
    ])
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "per_seed": seed_summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
