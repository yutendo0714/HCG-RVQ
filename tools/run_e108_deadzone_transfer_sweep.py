#!/usr/bin/env python3
"""Run the E108 start8192 dead-zone calibration sweep on GPU0."""

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
CHECKPOINT_PREFIX = (
    "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_g64_l1_k128_lambda0035"
)

SEEDS = (1234, 2345)
THRESHOLDS = ("014", "016", "020")


def result_prefix(seed: int, threshold: str) -> Path:
    return ANALYSIS / f"e108_deadzone{threshold}_from_beta005_seed{seed}_step250_fullimage_start8192_current"


def main() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    summaries: list[dict[str, object]] = []

    for seed in SEEDS:
        for threshold in THRESHOLDS:
            prefix = result_prefix(seed, threshold)
            json_path = prefix.with_suffix(".json")
            if json_path.exists():
                data = json.loads(json_path.read_text())
                summary = data["summaries"][0]
                summaries.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "mean_rd": summary["mean_rd"],
                        "mean_rd_minus_reference": summary.get("mean_rd_minus_reference"),
                        "nonfinite_rows": summary["nonfinite_rows"],
                        "skipped": True,
                    }
                )
                print(f"skip seed={seed} dz={threshold}: {summary.get('mean_rd_minus_reference')}")
                continue

            config = f"{CONFIG_PREFIX}_deadzone{threshold}_from_beta005_seed{seed}.yaml"
            checkpoint = f"{CHECKPOINT_PREFIX}_seed{seed}/checkpoint_step_250.pth.tar"
            reference = f"experiments/analysis/beta005_seed{seed}_transfer_start8192_reference.csv"
            cmd = [
                sys.executable,
                "tools/probe_householder_inverse_modes.py",
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--data-root",
                "/dpl/openimages/open-images-v6/train/data",
                "--device",
                "cuda:0",
                "--max-images",
                "4096",
                "--start-index",
                "8192",
                "--modes",
                "exact",
                "--reference-csv",
                reference,
                "--reference-column",
                "rd_score",
                "--output-csv",
                str(prefix.with_suffix(".csv")),
                "--output-json",
                str(json_path),
                "--output-md",
                str(prefix.with_suffix(".md")),
            ]
            print(f"run seed={seed} dz={threshold}")
            subprocess.run(cmd, check=True, env=env)
            data = json.loads(json_path.read_text())
            summary = data["summaries"][0]
            summaries.append(
                {
                    "seed": seed,
                    "threshold": threshold,
                    "mean_rd": summary["mean_rd"],
                    "mean_rd_minus_reference": summary.get("mean_rd_minus_reference"),
                    "nonfinite_rows": summary["nonfinite_rows"],
                    "skipped": False,
                }
            )
            print(
                f"done seed={seed} dz={threshold}: "
                f"delta={summary.get('mean_rd_minus_reference')} nonfinite={summary['nonfinite_rows']}"
            )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
