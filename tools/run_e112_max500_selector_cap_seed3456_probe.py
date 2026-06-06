#!/usr/bin/env python3
"""Probe deploy-time selector cap for the max500 seed3456 fragile tail."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ANALYSIS = Path("experiments/analysis")
TMP_CONFIG_DIR = ANALYSIS / "e112_tmp_configs"
SEED = 3456
THRESHOLD = "018"
CAPS = (0.25, 0.35, 0.45)
BASELINE_CAP = 0.50

BASE_CONFIG = Path(
    "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_deadzone018_from_beta005_seed3456.yaml"
)
CHECKPOINT = Path(
    "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_g64_l1_k128_lambda0035_seed3456_max500/"
    "checkpoint_step_500.pth.tar"
)

TRANSFER_REFERENCE = "experiments/analysis/beta005_seed3456_transfer_start8192_reference.csv"
HOLDOUT_REFERENCE = (
    "experiments/analysis/"
    "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"
)
BASELINE_TRANSFER_JSON = (
    ANALYSIS
    / "e111_deadzone018_from_beta005_max500_seed3456_step500_fullimage_start8192_current.json"
)
BASELINE_HOLDOUT_JSON = (
    ANALYSIS
    / "e110_deadzone018_from_beta005_max500_seed3456_step500_fullimage_holdout4096_current.json"
)
OUT_PREFIX = ANALYSIS / "e112_max500_selector_cap_seed3456_probe"


def cap_tag(cap: float) -> str:
    return f"cap{int(round(cap * 100)):03d}"


def write_config(cap: float) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(BASE_CONFIG.read_text())
    tag = cap_tag(cap)
    config["run_name"] = f"{config['run_name']}_e112_{tag}"
    config["quantizer"]["householder_gate_residual_selector_max"] = float(cap)
    config["wandb"]["enabled"] = False
    config["wandb"]["name"] = config["run_name"]
    path = TMP_CONFIG_DIR / f"{BASE_CONFIG.stem}_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_probe(
    cap: float,
    split: str,
    config_path: Path,
    env: dict[str, str],
) -> dict[str, object]:
    start_index = "8192" if split == "transfer" else "4096"
    reference = TRANSFER_REFERENCE if split == "transfer" else HOLDOUT_REFERENCE
    prefix = (
        ANALYSIS
        / f"e112_deadzone{THRESHOLD}_{cap_tag(cap)}_max500_seed{SEED}_step500_fullimage_{split}_current"
    )
    json_path = prefix.with_suffix(".json")
    if not json_path.exists():
        cmd = [
            sys.executable,
            "tools/probe_householder_inverse_modes.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(CHECKPOINT),
            "--data-root",
            "/dpl/openimages/open-images-v6/train/data",
            "--device",
            "cuda:0",
            "--max-images",
            "4096",
            "--start-index",
            start_index,
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
        print(f"eval seed={SEED} dz{THRESHOLD} max500 {split} {cap_tag(cap)}")
        subprocess.run(cmd, check=True, env=env)
    return summary_row(cap, split, json_path)


def summary_row(cap: float, split: str, json_path: Path) -> dict[str, object]:
    data = json.loads(json_path.read_text())
    summary = data["summaries"][0]
    return {
        "split": split,
        "seed": SEED,
        "threshold": THRESHOLD,
        "selector_cap": cap,
        "mean_rd": summary["mean_rd"],
        "mean_delta": summary["mean_rd_minus_reference"],
        "win_rate": summary.get("win_rate", ""),
        "nonfinite_rows": summary["nonfinite_rows"],
        "latent_qmse": summary.get("mean_rvq_latent_quant_mse"),
        "s_q_mean": summary.get("mean_rvq_s_q_mean"),
        "dead_code": summary.get("mean_rvq_dead_code_ratio"),
        "selector_prob": summary.get("mean_rvq_householder_residual_selector_prob"),
        "selector_multiplier": summary.get("mean_rvq_householder_residual_selector_multiplier"),
        "json": str(json_path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    rows: list[dict[str, object]] = []
    configs = {cap: write_config(cap) for cap in CAPS}

    rows.append(summary_row(BASELINE_CAP, "transfer", BASELINE_TRANSFER_JSON))
    for cap in CAPS:
        rows.append(run_probe(cap, "transfer", configs[cap], env))

    transfer_rows = [row for row in rows if row["split"] == "transfer"]
    best_transfer = min(transfer_rows, key=lambda row: float(row["mean_delta"]))
    best_cap = float(best_transfer["selector_cap"])

    if best_cap == BASELINE_CAP:
        rows.append(summary_row(BASELINE_CAP, "holdout", BASELINE_HOLDOUT_JSON))
    else:
        rows.append(run_probe(best_cap, "holdout", configs[best_cap], env))

    decision = {
        "best_transfer_cap": best_cap,
        "best_transfer_delta": best_transfer["mean_delta"],
        "baseline_transfer_delta": next(
            row["mean_delta"]
            for row in rows
            if row["split"] == "transfer" and row["selector_cap"] == BASELINE_CAP
        ),
        "holdout_delta_at_selected_cap": next(
            row["mean_delta"]
            for row in rows
            if row["split"] == "holdout" and row["selector_cap"] == best_cap
        ),
        "interpretation": (
            "This is a seed3456-only deploy-time selector-cap probe. If a lower cap "
            "improves transfer and holds on holdout, the next step is a 3-seed "
            "independent cap-selection audit."
        ),
    }
    payload = {"decision": decision, "rows": rows}
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".csv"), rows)

    lines = [
        "# E112 max500 selector-cap seed3456 probe",
        "",
        "## Decision",
        "",
        f"- Best transfer cap: {fmt(best_cap)} with delta {fmt(best_transfer['mean_delta'])}.",
        (
            f"- Baseline cap {fmt(BASELINE_CAP)} transfer delta "
            f"{fmt(decision['baseline_transfer_delta'])}."
        ),
        (
            f"- Holdout delta at selected cap: "
            f"{fmt(decision['holdout_delta_at_selected_cap'])}."
        ),
        f"- Interpretation: {decision['interpretation']}",
        "",
        "## Rows",
        "",
        "| split | cap | delta | RD | nonfinite | qMSE | s_q | dead code | selector prob |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {cap} | {delta} | {rd} | {nonfinite} | {qmse} | {sq} | {dead} | {prob} |".format(
                split=row["split"],
                cap=fmt(row["selector_cap"]),
                delta=fmt(row["mean_delta"]),
                rd=fmt(row["mean_rd"]),
                nonfinite=row["nonfinite_rows"],
                qmse=fmt(row.get("latent_qmse")),
                sq=fmt(row.get("s_q_mean")),
                dead=fmt(row.get("dead_code")),
                prob=fmt(row.get("selector_prob")),
            )
        )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
