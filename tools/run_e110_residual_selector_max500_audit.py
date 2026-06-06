#!/usr/bin/env python3
"""Run E110 500-step residual-selector budget audit on GPU0."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ANALYSIS = Path("experiments/analysis")
CONFIG_PREFIX = (
    "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50"
)
RUN_PREFIX = (
    "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_g64_l1_k128_lambda0035"
)
SEEDS = (1234, 2345, 3456)
THRESHOLDS = ("014", "018")


def train_config(seed: int) -> str:
    return f"{CONFIG_PREFIX}_from_beta005_max500_seed{seed}.yaml"


def run_dir(seed: int) -> Path:
    return Path("experiments") / f"{RUN_PREFIX}_seed{seed}_max500"


def checkpoint(seed: int) -> Path:
    return run_dir(seed) / "checkpoint_step_500.pth.tar"


def eval_config(seed: int, threshold: str) -> str:
    return f"{CONFIG_PREFIX}_deadzone{threshold}_from_beta005_seed{seed}.yaml"


def reference_csv(seed: int) -> str:
    if seed == 3456:
        return (
            "experiments/analysis/"
            "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"
        )
    return f"experiments/analysis/beta005_after250_seed{seed}_direct_exact_step500_holdout4096_current.csv"


def result_prefix(seed: int, threshold: str) -> Path:
    return (
        ANALYSIS
        / f"e110_deadzone{threshold}_from_beta005_max500_seed{seed}_step500_fullimage_holdout4096_current"
    )


def run_training(seed: int, env: dict[str, str]) -> None:
    ckpt = checkpoint(seed)
    if ckpt.exists():
        print(f"skip train seed={seed}: {ckpt}")
        return
    cmd = [sys.executable, "train.py", "--config", train_config(seed), "--device", "cuda:0"]
    print(f"train seed={seed} max500")
    subprocess.run(cmd, check=True, env=env)


def run_eval(seed: int, threshold: str, env: dict[str, str]) -> dict[str, object]:
    prefix = result_prefix(seed, threshold)
    json_path = prefix.with_suffix(".json")
    if not json_path.exists():
        cmd = [
            sys.executable,
            "tools/probe_householder_inverse_modes.py",
            "--config",
            eval_config(seed, threshold),
            "--checkpoint",
            str(checkpoint(seed)),
            "--data-root",
            "/dpl/openimages/open-images-v6/train/data",
            "--device",
            "cuda:0",
            "--max-images",
            "4096",
            "--start-index",
            "4096",
            "--modes",
            "exact",
            "--reference-csv",
            reference_csv(seed),
            "--reference-column",
            "rd_score",
            "--output-csv",
            str(prefix.with_suffix(".csv")),
            "--output-json",
            str(json_path),
            "--output-md",
            str(prefix.with_suffix(".md")),
        ]
        print(f"eval seed={seed} dz{threshold} max500 holdout4096")
        subprocess.run(cmd, check=True, env=env)
    data = json.loads(json_path.read_text())
    summary = data["summaries"][0]
    return {
        "seed": seed,
        "threshold": threshold,
        "mean_rd": summary["mean_rd"],
        "mean_delta": summary.get("mean_rd_minus_reference"),
        "nonfinite_rows": summary["nonfinite_rows"],
        "json": str(json_path),
    }


def main() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    summaries: list[dict[str, object]] = []
    for seed in SEEDS:
        run_training(seed, env)
        for threshold in THRESHOLDS:
            summary = run_eval(seed, threshold, env)
            summaries.append(summary)
            print(
                f"done seed={seed} dz{threshold}: delta={summary['mean_delta']} "
                f"nonfinite={summary['nonfinite_rows']}"
            )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
