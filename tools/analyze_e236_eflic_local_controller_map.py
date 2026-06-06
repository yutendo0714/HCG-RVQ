#!/usr/bin/env python3
"""Summarize E236 EF-LIC local controller-map runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kodak-csv",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e236_eflic_kodak24_local_controller_map.csv",
    )
    p.add_argument(
        "--clic-csv",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e236_eflic_clicpro41_local_controller_map.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e236_eflic_local_controller_map_summary",
    )
    return p.parse_args()


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            item: dict[str, Any] = dict(row)
            item["dataset"] = dataset
            item["score_dists_3lpips"] = to_float(item.get("score_dists_3lpips"))
            if not math.isfinite(item["score_dists_3lpips"]):
                item["score_dists_3lpips"] = to_float(item.get("delta_dists"), 0.0) + 3.0 * to_float(item.get("delta_lpips"), 0.0)
            for key in [
                "bpp",
                "delta_bpp",
                "delta_dists",
                "delta_lpips",
                "delta_psnr",
                "max_decode_diff",
                "nonfinite",
                "payload_len_equal",
                "payload_equal",
                "y_mismatch",
                "y_total",
                "y_alpha_mean",
                "y_alpha_std",
                "y_alpha_max",
                "y_alpha_active_frac",
                "y_avg_geometry_delta_rms",
                "y_avg_index_entropy",
                "y_avg_index_used_frac",
            ]:
                item[key] = to_float(item.get(key), 0.0)
            item["valid_contract"] = int(
                abs(item["delta_bpp"]) <= 1e-12
                and item["max_decode_diff"] <= 1e-10
                and int(item["nonfinite"]) == 0
                and int(item["payload_len_equal"]) == 1
            )
            rows.append(item)
    return rows


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_fixed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["policy"])].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, policy), subset in sorted(groups.items()):
        valid = [row for row in subset if row["valid_contract"]]
        family = valid[0]["family"] if valid else subset[0]["family"]
        out.append(
            {
                "dataset": dataset,
                "policy": policy,
                "family": family,
                "images": len(subset),
                "valid_rows": len(valid),
                "score_dists_3lpips": mean([row["score_dists_3lpips"] for row in valid]),
                "score_win_frac": mean([float(row["score_dists_3lpips"] < 0.0) for row in valid]),
                "delta_dists": mean([row["delta_dists"] for row in valid]),
                "delta_lpips": mean([row["delta_lpips"] for row in valid]),
                "delta_psnr": mean([row["delta_psnr"] for row in valid]),
                "y_mismatch_frac": float(sum(row["y_mismatch"] for row in valid) / max(1.0, sum(row["y_total"] for row in valid))),
                "alpha_mean": mean([row["y_alpha_mean"] for row in valid]),
                "alpha_active_frac": mean([row["y_alpha_active_frac"] for row in valid]),
                "geometry_delta_rms": mean([row["y_avg_geometry_delta_rms"] for row in valid]),
                "max_decode_diff": max([row["max_decode_diff"] for row in valid], default=float("nan")),
                "nonfinite_rows": int(sum(int(row["nonfinite"]) for row in subset)),
            }
        )
    pooled_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pooled_groups[row["policy"]].append(row)
    for policy, subset in sorted(pooled_groups.items()):
        valid = [row for row in subset if row["valid_contract"]]
        family = valid[0]["family"] if valid else subset[0]["family"]
        out.append(
            {
                "dataset": "pooled",
                "policy": policy,
                "family": family,
                "images": len(subset),
                "valid_rows": len(valid),
                "score_dists_3lpips": mean([row["score_dists_3lpips"] for row in valid]),
                "score_win_frac": mean([float(row["score_dists_3lpips"] < 0.0) for row in valid]),
                "delta_dists": mean([row["delta_dists"] for row in valid]),
                "delta_lpips": mean([row["delta_lpips"] for row in valid]),
                "delta_psnr": mean([row["delta_psnr"] for row in valid]),
                "y_mismatch_frac": float(sum(row["y_mismatch"] for row in valid) / max(1.0, sum(row["y_total"] for row in valid))),
                "alpha_mean": mean([row["y_alpha_mean"] for row in valid]),
                "alpha_active_frac": mean([row["y_alpha_active_frac"] for row in valid]),
                "geometry_delta_rms": mean([row["y_avg_geometry_delta_rms"] for row in valid]),
                "max_decode_diff": max([row["max_decode_diff"] for row in valid], default=float("nan")),
                "nonfinite_rows": int(sum(int(row["nonfinite"]) for row in subset)),
            }
        )
    return out


def summarize_oracle(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    choices: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["valid_contract"]:
            grouped[(row["dataset"], row["image"])].append(row)
    for (dataset, image), subset in sorted(grouped.items()):
        best = min(subset, key=lambda row: row["score_dists_3lpips"])
        choices.append(
            {
                "dataset": dataset,
                "image": image,
                "policy": best["policy"],
                "family": best["family"],
                "score_dists_3lpips": best["score_dists_3lpips"],
                "delta_dists": best["delta_dists"],
                "delta_lpips": best["delta_lpips"],
                "delta_psnr": best["delta_psnr"],
                "alpha_mean": best["y_alpha_mean"],
                "alpha_active_frac": best["y_alpha_active_frac"],
                "y_mismatch_frac": best["y_mismatch"] / max(1.0, best["y_total"]),
            }
        )
    summary: list[dict[str, Any]] = []
    for dataset in sorted({choice["dataset"] for choice in choices}) + ["pooled"]:
        subset = choices if dataset == "pooled" else [choice for choice in choices if choice["dataset"] == dataset]
        if not subset:
            continue
        policy_counts = Counter(choice["policy"] for choice in subset)
        family_counts = Counter(choice["family"] for choice in subset)
        summary.append(
            {
                "dataset": dataset,
                "images": len(subset),
                "score_dists_3lpips": mean([choice["score_dists_3lpips"] for choice in subset]),
                "score_win_frac": mean([float(choice["score_dists_3lpips"] < 0.0) for choice in subset]),
                "delta_dists": mean([choice["delta_dists"] for choice in subset]),
                "delta_lpips": mean([choice["delta_lpips"] for choice in subset]),
                "delta_psnr": mean([choice["delta_psnr"] for choice in subset]),
                "policy_counts": dict(policy_counts),
                "family_counts": dict(family_counts),
            }
        )
    return summary, choices


def risk_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "y_mismatch_frac",
        "y_alpha_mean",
        "y_alpha_active_frac",
        "y_avg_geometry_delta_rms",
        "y_avg_index_entropy",
        "y_avg_index_used_frac",
    ]
    out: list[dict[str, Any]] = []
    enriched = []
    for row in rows:
        if not row["valid_contract"] or row["policy"] == "zero":
            continue
        copy = dict(row)
        copy["y_mismatch_frac"] = row["y_mismatch"] / max(1.0, row["y_total"])
        enriched.append(copy)
    for dataset in sorted({row["dataset"] for row in enriched}) + ["pooled"]:
        subset = enriched if dataset == "pooled" else [row for row in enriched if row["dataset"] == dataset]
        for feature in features:
            out.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "corr_with_score": corr([row[feature] for row in subset], [row["score_dists_3lpips"] for row in subset]),
                    "rows": len(subset),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}" if abs(value) < 10 else f"{value:.3f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.kodak_csv, "kodak24") + read_rows(args.clic_csv, "clicpro41")
    fixed = summarize_fixed(rows)
    oracle_summary, oracle_choices = summarize_oracle(rows)
    risks = risk_correlations(rows)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fixed_path = args.output_prefix.with_name(args.output_prefix.name + "_fixed.csv")
    oracle_path = args.output_prefix.with_name(args.output_prefix.name + "_oracle.csv")
    choices_path = args.output_prefix.with_name(args.output_prefix.name + "_oracle_choices.csv")
    risk_path = args.output_prefix.with_name(args.output_prefix.name + "_risk_correlations.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    write_csv(fixed_path, fixed)
    write_csv(oracle_path, oracle_summary)
    write_csv(choices_path, oracle_choices)
    write_csv(risk_path, risks)
    json_path.write_text(
        json.dumps(
            {
                "experiment": "E236 EF-LIC local controller-map summary",
                "inputs": {"kodak": str(args.kodak_csv), "clic": str(args.clic_csv)},
                "fixed": fixed,
                "oracle": oracle_summary,
                "risk_correlations": risks,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fixed_keys = [
        "dataset",
        "policy",
        "images",
        "valid_rows",
        "score_dists_3lpips",
        "score_win_frac",
        "delta_dists",
        "delta_lpips",
        "delta_psnr",
        "y_mismatch_frac",
        "alpha_mean",
        "alpha_active_frac",
        "geometry_delta_rms",
        "nonfinite_rows",
    ]
    oracle_keys = ["dataset", "images", "score_dists_3lpips", "score_win_frac", "delta_dists", "delta_lpips", "delta_psnr", "policy_counts"]
    with md_path.open("w") as fobj:
        fobj.write("# E236 EF-LIC Local Controller-Map Summary\n\n")
        fobj.write(
            "This summarizes decoder-safe local alpha-map composition policies on Kodak24 and CLIC professional 41. "
            "All rows are expected to preserve bpp and exact decoder reproduction; violations remain visible in the fixed table.\n\n"
        )
        fobj.write("## Fixed Policies\n\n")
        fobj.write("| " + " | ".join(fixed_keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(fixed_keys)) + "|\n")
        for item in fixed:
            fobj.write("| " + " | ".join(fmt(item.get(key, "")) for key in fixed_keys) + " |\n")
        fobj.write("\n## Per-Image Oracle Over E236 Policies\n\n")
        fobj.write("| " + " | ".join(oracle_keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(oracle_keys)) + "|\n")
        for item in oracle_summary:
            fobj.write("| " + " | ".join(fmt(item.get(key, "")) for key in oracle_keys) + " |\n")
        fobj.write("\n## Risk Correlations\n\n")
        fobj.write("| dataset | feature | corr_with_score | rows |\n")
        fobj.write("|---|---|---|---|\n")
        for item in risks:
            fobj.write(f"| {item['dataset']} | {item['feature']} | {fmt(item['corr_with_score'])} | {item['rows']} |\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- Negative score is better under `DISTS + 3*LPIPS` at unchanged bpp.\n")
        fobj.write("- Fixed local compositions are diagnostics; the oracle is an upper bound, not a paper row.\n")
        fobj.write("- Correlations identify false-positive signals that the learned controller should regularize.\n")

    print(f"wrote {fixed_path}, {oracle_path}, {choices_path}, {risk_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
