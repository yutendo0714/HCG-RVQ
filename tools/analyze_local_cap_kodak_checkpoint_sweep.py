#!/usr/bin/env python3
"""Checkpoint sweep for local-cap HCG-RVQ on Kodak."""

from __future__ import annotations

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
OUT_CSV = ANALYSIS / "local_cap080_rho1_kodak_checkpoint_sweep.csv"
OUT_JSON = ANALYSIS / "local_cap080_rho1_kodak_checkpoint_sweep.json"
OUT_MD = ANALYSIS / "local_cap080_rho1_kodak_checkpoint_sweep.md"

SEEDS = ["1234", "2345", "3456"]
STEPS = ["250", "500"]
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


def run_spec(seed: str, step: str) -> dict[str, str]:
    return {
        "seed": seed,
        "step": step,
        "config": f"configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_seed{seed}.yaml",
        "checkpoint": (
            "experiments/"
            "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
            f"frozen_g64_l1_k128_lambda0035_seed{seed}/checkpoint_step_{step}.pth.tar"
        ),
    }


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, float | int | str]] = []

    for seed in SEEDS:
        for step in STEPS:
            spec = run_spec(seed, step)
            for key in ("config", "checkpoint"):
                path = ROOT / spec[key]
                if not path.exists():
                    raise FileNotFoundError(path)
            rows, _ = evaluate_mode(
                mode="exact",
                config_path=str(ROOT / spec["config"]),
                checkpoint_path=str(ROOT / spec["checkpoint"]),
                data_root="/dpl/kodak",
                device=device,
                max_images=24,
                start_index=0,
                patch_size=None,
                reference={},
            )
            for row in rows:
                row.update(spec)
            all_rows.extend(rows)
            summaries.append(summarize(rows))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in all_rows for key in row})
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    by_seed_step = {(str(row["seed"]), str(row["step"])): row for row in summaries}
    per_seed = []
    for seed in SEEDS:
        s250 = by_seed_step[(seed, "250")]
        s500 = by_seed_step[(seed, "500")]
        per_seed.append(
            {
                "seed": seed,
                "step250_rd": float(s250["mean_rd"]),
                "step500_rd": float(s500["mean_rd"]),
                "step500_minus_step250": float(s500["mean_rd"]) - float(s250["mean_rd"]),
                "step250_qmse": float(s250.get("mean_rvq_latent_quant_mse", float("nan"))),
                "step500_qmse": float(s500.get("mean_rvq_latent_quant_mse", float("nan"))),
                "step250_s_q": float(s250.get("mean_rvq_s_q_mean", float("nan"))),
                "step500_s_q": float(s500.get("mean_rvq_s_q_mean", float("nan"))),
            }
        )

    aggregate = {
        "step250_rd": mean([row["step250_rd"] for row in per_seed]),
        "step500_rd": mean([row["step500_rd"] for row in per_seed]),
    }
    aggregate["step500_minus_step250"] = aggregate["step500_rd"] - aggregate["step250_rd"]

    result = {
        "device": str(device),
        "summaries": summaries,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Local Cap080/rho1 Kodak Checkpoint Sweep",
        "",
        "Kodak 24-image direct exact-inverse probe for local cap080/rho1 step250 vs step500.",
        "",
        "| seed | step250 RD | step500 RD | step500-step250 | step250 s_q | step500 s_q | step250 qMSE | step500 qMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {rd250} | {rd500} | {delta} | {sq250} | {sq500} | {qmse250} | {qmse500} |".format(
                seed=row["seed"],
                rd250=fmt(row["step250_rd"]),
                rd500=fmt(row["step500_rd"]),
                delta=fmt(row["step500_minus_step250"], signed=True),
                sq250=fmt(row["step250_s_q"]),
                sq500=fmt(row["step500_s_q"]),
                qmse250=fmt(row["step250_qmse"]),
                qmse500=fmt(row["step500_qmse"]),
            )
        )
    lines.extend(
        [
            "",
            "| aggregate step250 RD | aggregate step500 RD | step500-step250 |",
            "|---:|---:|---:|",
            "| {rd250} | {rd500} | {delta} |".format(
                rd250=fmt(aggregate["step250_rd"]),
                rd500=fmt(aggregate["step500_rd"]),
                delta=fmt(aggregate["step500_minus_step250"], signed=True),
            ),
            "",
            "Conclusion:",
            "",
            "- Step500 is a diagnostic checkpoint, not a rescue path, if its RD remains above step250 under this split.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT_MD), "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
