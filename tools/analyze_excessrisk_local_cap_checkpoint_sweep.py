#!/usr/bin/env python3
"""Checkpoint sweep for the excess-risk local-cap HCG-RVQ variant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
VARIANT = "excessrisk090_local_cap080_rho1"
RUN_PREFIX = "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090"
RUN_SUFFIX = "frozen_g64_l1_k128_lambda0035"
FEATURE_KEYS = [
    "rvq_s_q_mean",
    "rvq_householder_strength",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
    "rvq_householder_risk_multiplier",
]


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def maybe_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def split_defaults(split: str) -> tuple[str, int, int]:
    if split == "kodak":
        return "/dpl/kodak", 0, 24
    if split == "holdout4096":
        return "/dpl/openimages/open-images-v6/train/data", 4096, 4096
    if split == "transfer8192":
        return "/dpl/openimages/open-images-v6/train/data", 8192, 4096
    raise ValueError(f"unknown split: {split}")


def run_spec(seed: str, step: str, run_prefix: str, run_suffix: str) -> dict[str, str]:
    return {
        "seed": seed,
        "step": step,
        "config": f"configs/{run_prefix}_frozen_seed{seed}.yaml",
        "checkpoint": (
            f"experiments/{run_prefix}_{run_suffix}_seed{seed}/"
            f"checkpoint_step_{step}.pth.tar"
        ),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "seed": str(rows[0]["seed"]),
        "step": str(rows[0]["step"]),
        "num_images": len(rows),
        "mean_rd": mean([float(row["rd_score"]) for row in rows]),
        "mean_bpp": mean([float(row["bpp"]) for row in rows]),
        "mean_psnr": mean([float(row["psnr"]) for row in rows]),
        "mean_ms_ssim": mean([float(row["ms_ssim"]) for row in rows]),
        "nonfinite_rows": sum(int(row.get("has_nonfinite", 0)) for row in rows),
    }
    for key in FEATURE_KEYS:
        values = [maybe_float(row, key) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[f"mean_{key}"] = mean(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["kodak", "holdout4096", "transfer8192"], default="kodak")
    parser.add_argument("--seeds", nargs="+", default=["1234", "2345", "3456"])
    parser.add_argument("--steps", nargs="+", default=["250", "500"])
    parser.add_argument("--variant-name", default=VARIANT)
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--run-suffix", default=RUN_SUFFIX)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data_root, start_index, max_images = split_defaults(args.split)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, float | int | str]] = []

    for seed in args.seeds:
        for step in args.steps:
            spec = run_spec(seed, step, args.run_prefix, args.run_suffix)
            for key in ("config", "checkpoint"):
                path = ROOT / spec[key]
                if not path.exists():
                    raise FileNotFoundError(path)
            rows, _ = evaluate_mode(
                mode="exact",
                config_path=str(ROOT / spec["config"]),
                checkpoint_path=str(ROOT / spec["checkpoint"]),
                data_root=data_root,
                device=device,
                max_images=max_images,
                start_index=start_index,
                patch_size=None,
                reference={},
            )
            for row in rows:
                row.update(spec)
            all_rows.extend(rows)
            summaries.append(summarize(rows))

    stem = f"{args.variant_name}_{args.split}_checkpoint_sweep"
    out_csv = ANALYSIS / f"{stem}.csv"
    out_json = ANALYSIS / f"{stem}.json"
    out_md = ANALYSIS / f"{stem}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in all_rows for key in row})
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    by_seed_step = {(str(row["seed"]), str(row["step"])): row for row in summaries}
    per_seed = []
    for seed in args.seeds:
        seed_rows = [by_seed_step[(seed, step)] for step in args.steps]
        best = min(seed_rows, key=lambda row: float(row["mean_rd"]))
        item: dict[str, float | str] = {
            "seed": seed,
            "best_step": str(best["step"]),
            "best_rd": float(best["mean_rd"]),
        }
        for row in seed_rows:
            step = str(row["step"])
            item[f"step{step}_rd"] = float(row["mean_rd"])
            item[f"step{step}_s_q"] = float(row.get("mean_rvq_s_q_mean", float("nan")))
            item[f"step{step}_qmse"] = float(row.get("mean_rvq_latent_quant_mse", float("nan")))
        if "step250_rd" in item and "step500_rd" in item:
            item["step500_minus_step250"] = float(item["step500_rd"]) - float(item["step250_rd"])
        per_seed.append(item)

    aggregate = {
        "best_checkpoint_rd": mean([float(row["best_rd"]) for row in per_seed]),
        "num_images": len(all_rows),
        "nonfinite_rows": sum(int(row.get("has_nonfinite", 0)) for row in all_rows),
    }
    for step in args.steps:
        key = f"step{step}_rd"
        values = [float(row[key]) for row in per_seed if key in row]
        if values:
            aggregate[f"step{step}_rd"] = mean(values)

    result = {
        "variant": args.variant_name,
        "split": args.split,
        "data_root": data_root,
        "start_index": start_index,
        "max_images": max_images,
        "device": str(device),
        "summaries": summaries,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {args.variant_name} {args.split} Checkpoint Sweep",
        "",
        f"Split: `{data_root}`, start_index={start_index}, max_images={max_images}, device=`{device}`.",
        "",
        "| seed | step250 RD | step500 RD | step500-step250 | best | step250 s_q | step500 s_q | step250 qMSE | step500 qMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {rd250} | {rd500} | {delta} | {best} | {sq250} | {sq500} | {qmse250} | {qmse500} |".format(
                seed=row["seed"],
                rd250=fmt(float(row.get("step250_rd", float("nan")))),
                rd500=fmt(float(row.get("step500_rd", float("nan")))),
                delta=fmt(float(row.get("step500_minus_step250", float("nan"))), signed=True),
                best=row["best_step"],
                sq250=fmt(float(row.get("step250_s_q", float("nan")))),
                sq500=fmt(float(row.get("step500_s_q", float("nan")))),
                qmse250=fmt(float(row.get("step250_qmse", float("nan")))),
                qmse500=fmt(float(row.get("step500_qmse", float("nan")))),
            )
        )
    lines.extend(
        [
            "",
            "| aggregate best-checkpoint RD | step250 RD | step500 RD | nonfinite rows |",
            "|---:|---:|---:|---:|",
            "| {best} | {rd250} | {rd500} | {nonfinite} |".format(
                best=fmt(float(aggregate["best_checkpoint_rd"])),
                rd250=fmt(float(aggregate.get("step250_rd", float("nan")))),
                rd500=fmt(float(aggregate.get("step500_rd", float("nan")))),
                nonfinite=int(aggregate["nonfinite_rows"]),
            ),
            "",
            "Artifacts:",
            "",
            f"- `{out_csv.relative_to(ROOT)}`",
            f"- `{out_json.relative_to(ROOT)}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(out_md), "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
