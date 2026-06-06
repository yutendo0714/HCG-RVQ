#!/usr/bin/env python3
"""Select a deploy-time selector cap for max500 on transfer, then audit holdout."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ANALYSIS = Path("experiments/analysis")
TMP_CONFIG_DIR = ANALYSIS / "e113_tmp_configs"
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
OUT_PREFIX = ANALYSIS / "e113_max500_selector_cap_multiseed_audit"
SEEDS = (1234, 2345, 3456)
THRESHOLD = "018"
CAPS = (0.25, 0.35, 0.45)
BASELINE_CAP = 0.50


def cap_tag(cap: float) -> str:
    return f"cap{int(round(cap * 100)):03d}"


def checkpoint(seed: int) -> Path:
    return Path("experiments") / f"{RUN_PREFIX}_seed{seed}_max500" / "checkpoint_step_500.pth.tar"


def base_config(seed: int) -> Path:
    return Path(f"{CONFIG_PREFIX}_deadzone{THRESHOLD}_from_beta005_seed{seed}.yaml")


def transfer_reference(seed: int) -> str:
    return f"experiments/analysis/beta005_seed{seed}_transfer_start8192_reference.csv"


def holdout_reference(seed: int) -> str:
    if seed == 3456:
        return (
            "experiments/analysis/"
            "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"
        )
    return f"experiments/analysis/beta005_after250_seed{seed}_direct_exact_step500_holdout4096_current.csv"


def baseline_json(seed: int, split: str) -> Path:
    exp = "e111" if split == "transfer" else "e110"
    split_name = "start8192" if split == "transfer" else "holdout4096"
    return (
        ANALYSIS
        / f"{exp}_deadzone{THRESHOLD}_from_beta005_max500_seed{seed}_step500_fullimage_{split_name}_current.json"
    )


def e112_json(seed: int, cap: float, split: str) -> Path | None:
    if seed != 3456:
        return None
    split_name = "transfer" if split == "transfer" else "holdout"
    path = (
        ANALYSIS
        / f"e112_deadzone{THRESHOLD}_{cap_tag(cap)}_max500_seed{seed}_step500_fullimage_{split_name}_current.json"
    )
    return path if path.exists() else None


def result_prefix(seed: int, cap: float, split: str) -> Path:
    return (
        ANALYSIS
        / f"e113_deadzone{THRESHOLD}_{cap_tag(cap)}_max500_seed{seed}_step500_fullimage_{split}_current"
    )


def write_config(seed: int, cap: float) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = base_config(seed)
    config = yaml.safe_load(config_path.read_text())
    tag = cap_tag(cap)
    config["run_name"] = f"{config['run_name']}_e113_{tag}"
    config["quantizer"]["householder_gate_residual_selector_max"] = float(cap)
    config["wandb"]["enabled"] = False
    config["wandb"]["name"] = config["run_name"]
    path = TMP_CONFIG_DIR / f"{config_path.stem}_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def summary_row(seed: int, cap: float, split: str, json_path: Path) -> dict[str, object]:
    data = json.loads(json_path.read_text())
    summary = data["summaries"][0]
    return {
        "split": split,
        "seed": seed,
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


def run_probe(
    seed: int,
    cap: float,
    split: str,
    config_path: Path,
    env: dict[str, str],
) -> dict[str, object]:
    reused = e112_json(seed, cap, split)
    if reused is not None:
        return summary_row(seed, cap, split, reused)

    prefix = result_prefix(seed, cap, split)
    json_path = prefix.with_suffix(".json")
    if not json_path.exists():
        cmd = [
            sys.executable,
            "tools/probe_householder_inverse_modes.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint(seed)),
            "--data-root",
            "/dpl/openimages/open-images-v6/train/data",
            "--device",
            "cuda:0",
            "--max-images",
            "4096",
            "--start-index",
            "8192" if split == "transfer" else "4096",
            "--modes",
            "exact",
            "--reference-csv",
            transfer_reference(seed) if split == "transfer" else holdout_reference(seed),
            "--reference-column",
            "rd_score",
            "--output-csv",
            str(prefix.with_suffix(".csv")),
            "--output-json",
            str(json_path),
            "--output-md",
            str(prefix.with_suffix(".md")),
        ]
        print(f"eval seed={seed} dz{THRESHOLD} max500 {split} {cap_tag(cap)}")
        subprocess.run(cmd, check=True, env=env)
    return summary_row(seed, cap, split, json_path)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["split"]), float(row["selector_cap"])), []).append(row)

    out = []
    for (split, cap), group in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "selector_cap": cap,
                "num_seeds": len(group),
                "mean_rd": sum(float(row["mean_rd"]) for row in group) / len(group),
                "mean_delta": sum(float(row["mean_delta"]) for row in group) / len(group),
                "nonfinite_rows": sum(int(row["nonfinite_rows"]) for row in group),
                "latent_qmse": sum(float(row["latent_qmse"]) for row in group) / len(group),
                "s_q_mean": sum(float(row["s_q_mean"]) for row in group) / len(group),
                "dead_code": sum(float(row["dead_code"]) for row in group) / len(group),
            }
        )
    return out


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
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for seed in SEEDS:
        if not checkpoint(seed).exists():
            raise FileNotFoundError(checkpoint(seed))

    configs = {(seed, cap): write_config(seed, cap) for seed in SEEDS for cap in CAPS}
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        rows.append(summary_row(seed, BASELINE_CAP, "transfer", baseline_json(seed, "transfer")))
        for cap in CAPS:
            rows.append(run_probe(seed, cap, "transfer", configs[(seed, cap)], env))

    aggregate_transfer = [
        row for row in aggregate(rows) if row["split"] == "transfer" and row["num_seeds"] == len(SEEDS)
    ]
    best = min(aggregate_transfer, key=lambda row: float(row["mean_delta"]))
    best_cap = float(best["selector_cap"])

    for seed in SEEDS:
        if best_cap == BASELINE_CAP:
            rows.append(summary_row(seed, BASELINE_CAP, "holdout", baseline_json(seed, "holdout")))
        else:
            rows.append(run_probe(seed, best_cap, "holdout", configs[(seed, best_cap)], env))

    aggregates = aggregate(rows)
    selected_holdout = next(
        row for row in aggregates if row["split"] == "holdout" and row["selector_cap"] == best_cap
    )
    baseline_transfer = next(
        row for row in aggregates if row["split"] == "transfer" and row["selector_cap"] == BASELINE_CAP
    )

    decision = {
        "best_transfer_cap": best_cap,
        "best_transfer_delta": best["mean_delta"],
        "baseline_transfer_delta": baseline_transfer["mean_delta"],
        "holdout_delta_at_selected_cap": selected_holdout["mean_delta"],
        "nonfinite_rows_at_selected_holdout": selected_holdout["nonfinite_rows"],
        "interpretation": (
            "This audit selects the max500 selector cap on the independent transfer split "
            "before checking holdout. It is meant to decide whether deploy-time cap "
            "calibration can make max500 paper-clean."
        ),
    }

    payload = {"decision": decision, "aggregates": aggregates, "rows": rows}
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".per_seed.csv"), rows)
    write_csv(OUT_PREFIX.with_suffix(".aggregates.csv"), aggregates)

    lines = [
        "# E113 max500 selector-cap multiseed audit",
        "",
        "## Decision",
        "",
        f"- Best transfer cap: {fmt(best_cap)} with aggregate delta {fmt(best['mean_delta'])}.",
        f"- Baseline cap {fmt(BASELINE_CAP)} transfer delta: {fmt(baseline_transfer['mean_delta'])}.",
        f"- Holdout delta at selected cap: {fmt(selected_holdout['mean_delta'])}.",
        f"- Selected holdout nonfinite rows: {selected_holdout['nonfinite_rows']}.",
        f"- Interpretation: {decision['interpretation']}",
        "",
        "## Aggregate Rows",
        "",
        "| split | cap | seeds | delta | RD | nonfinite | qMSE | s_q | dead code |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {split} | {cap} | {seeds} | {delta} | {rd} | {nonfinite} | {qmse} | {sq} | {dead} |".format(
                split=row["split"],
                cap=fmt(row["selector_cap"]),
                seeds=row["num_seeds"],
                delta=fmt(row["mean_delta"]),
                rd=fmt(row["mean_rd"]),
                nonfinite=row["nonfinite_rows"],
                qmse=fmt(row["latent_qmse"]),
                sq=fmt(row["s_q_mean"]),
                dead=fmt(row["dead_code"]),
            )
        )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
