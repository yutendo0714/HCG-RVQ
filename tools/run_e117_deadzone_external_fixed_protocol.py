#!/usr/bin/env python3
"""Evaluate dead-zone HCG-RVQ checkpoints on external fixed-protocol splits.

This script compares the transfer-selected step250 dead-zone branches against
the existing beta005 guard rows on Kodak and CLIC validation splits.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


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
SEEDS = (1234, 2345, 3456)
THRESHOLDS = ("014", "018")
SPLITS = {
    "kodak": {
        "data_root": "/dpl/kodak",
        "max_images": 24,
        "start_index": 0,
        "reference_source": ANALYSIS / "beta005_external_kodak_fixed_protocol.csv",
    },
    "clic_mobile_valid": {
        "data_root": "/dpl/clic/mobile/valid",
        "max_images": 1000,
        "start_index": 0,
        "reference_source": ANALYSIS / "beta005_external_clic_mobile_valid.csv",
    },
    "clic_professional_valid": {
        "data_root": "/dpl/clic/professional/valid",
        "max_images": 1000,
        "start_index": 0,
        "reference_source": ANALYSIS / "beta005_external_clic_professional_valid.csv",
    },
}
FEATURE_COLUMNS = (
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_multiplier",
)
SUMMARY_PREFIX = ANALYSIS / "e117_deadzone_external_fixed_protocol_audit"


def config_path(seed: int, threshold: str) -> Path:
    return Path(f"{CONFIG_PREFIX}_deadzone{threshold}_from_beta005_seed{seed}.yaml")


def checkpoint_path(seed: int) -> Path:
    return Path(f"{CHECKPOINT_PREFIX}_seed{seed}/checkpoint_step_250.pth.tar")


def reference_path(split: str, seed: int) -> Path:
    return ANALYSIS / f"e117_beta005_reference_{split}_seed{seed}.csv"


def result_prefix(split: str, threshold: str, seed: int) -> Path:
    return (
        ANALYSIS
        / f"e117_deadzone{threshold}_from_beta005_seed{seed}_step250_fullimage_{split}_current"
    )


def to_float(value: str) -> float:
    if value in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(value)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in {"path", "mode", "config", "checkpoint", "method"}:
                    parsed[key] = value
                elif key:
                    try:
                        parsed[key] = to_float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export_beta_reference(split: str, seed: int) -> Path:
    meta = SPLITS[split]
    source = Path(meta["reference_source"])
    out = reference_path(split, seed)
    if out.exists():
        return out
    rows: list[dict[str, object]] = []
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") == "beta005 guard" and int(float(row.get("seed", "nan"))) == seed:
                rows.append({"path": row["path"], "rd_score": row["rd_score"]})
    if not rows:
        raise RuntimeError(f"no beta005 guard rows found in {source} for seed {seed}")
    write_csv(out, rows)
    return out


def run_probe(split: str, threshold: str, seed: int) -> None:
    prefix = result_prefix(split, threshold, seed)
    json_path = prefix.with_suffix(".json")
    if json_path.exists():
        print(f"skip split={split} dz{threshold} seed={seed}: existing")
        return

    cfg = config_path(seed, threshold)
    checkpoint = checkpoint_path(seed)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    ref = export_beta_reference(split, seed)
    meta = SPLITS[split]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        sys.executable,
        "tools/probe_householder_inverse_modes.py",
        "--config",
        str(cfg),
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(meta["data_root"]),
        "--device",
        "cuda:0",
        "--max-images",
        str(meta["max_images"]),
        "--start-index",
        str(meta["start_index"]),
        "--modes",
        "exact",
        "--reference-csv",
        str(ref),
        "--reference-column",
        "rd_score",
        "--output-csv",
        str(prefix.with_suffix(".csv")),
        "--output-json",
        str(json_path),
        "--output-md",
        str(prefix.with_suffix(".md")),
    ]
    print(f"run split={split} dz{threshold} seed={seed}")
    subprocess.run(cmd, check=True, env=env)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [float(row["rd_minus_reference"]) for row in rows]
    rd = [float(row["rd_score"]) for row in rows]
    ref = [float(row["reference_rd_score"]) for row in rows]
    summary: dict[str, object] = {
        "num_images": len(rows),
        "mean_rd": mean(rd),
        "mean_reference_rd": mean(ref),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "q05_delta": percentile(deltas, 0.05),
        "q95_delta": percentile(deltas, 0.95),
        "max_delta": max(deltas),
        "win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
        "nonfinite_rows": int(sum(float(row.get("has_nonfinite", 0.0)) for row in rows)),
    }
    for column in FEATURE_COLUMNS:
        values = [
            float(row[column])
            for row in rows
            if column in row and isinstance(row[column], (float, int)) and math.isfinite(float(row[column]))
        ]
        if values:
            summary[f"mean_{column}"] = mean(values)
    return summary


def format_float(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def aggregate() -> None:
    all_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    per_seed_rows: list[dict[str, object]] = []
    for split in SPLITS:
        for threshold in THRESHOLDS:
            for seed in SEEDS:
                csv_path = result_prefix(split, threshold, seed).with_suffix(".csv")
                rows = read_rows(csv_path)
                all_rows[(split, threshold)].extend(rows)
                per_seed_rows.append(
                    {
                        "split": split,
                        "threshold": threshold,
                        "seed": seed,
                        **summarize(rows),
                        "csv": str(csv_path),
                    }
                )

    split_threshold_rows: list[dict[str, object]] = []
    threshold_external_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    quartile_rows: list[dict[str, object]] = []
    for split in SPLITS:
        for threshold in THRESHOLDS:
            rows = all_rows[(split, threshold)]
            summary = summarize(rows)
            split_threshold_rows.append(
                {
                    "split": split,
                    "threshold": threshold,
                    "num_seeds": len(SEEDS),
                    **summary,
                }
            )
            threshold_external_rows[threshold].extend(rows)

            ordered = sorted(rows, key=lambda row: float(row["reference_rd_score"]))
            n = len(ordered)
            for q in range(4):
                subset = ordered[q * n // 4 : (q + 1) * n // 4]
                quartile_rows.append(
                    {
                        "split": split,
                        "threshold": threshold,
                        "quartile": f"Q{q + 1}",
                        "num_images": len(subset),
                        "mean_reference_rd": mean(float(r["reference_rd_score"]) for r in subset),
                        "mean_delta": mean(float(r["rd_minus_reference"]) for r in subset),
                        "win_rate": sum(1 for r in subset if float(r["rd_minus_reference"]) < 0.0)
                        / len(subset),
                        "q95_delta": percentile([float(r["rd_minus_reference"]) for r in subset], 0.95),
                    }
                )

    threshold_rows = [
        {
            "threshold": threshold,
            "scope": "all_external_splits_weighted_by_image_count",
            **summarize(rows),
        }
        for threshold, rows in threshold_external_rows.items()
    ]

    pairwise_rows: list[dict[str, object]] = []
    for split in SPLITS:
        dz014 = {str(row["path"]): row for row in all_rows[(split, "014")]}
        dz018 = {str(row["path"]): row for row in all_rows[(split, "018")]}
        shared = sorted(set(dz014) & set(dz018))
        diffs = [
            float(dz014[path]["rd_minus_reference"]) - float(dz018[path]["rd_minus_reference"])
            for path in shared
        ]
        pairwise_rows.append(
            {
                "split": split,
                "num_shared": len(shared),
                "mean_dz014_minus_dz018_delta": mean(diffs),
                "median_dz014_minus_dz018_delta": median(diffs),
                "q05_dz014_minus_dz018_delta": percentile(diffs, 0.05),
                "q95_dz014_minus_dz018_delta": percentile(diffs, 0.95),
                "dz014_better_than_dz018_rate": sum(1 for d in diffs if d < 0.0) / len(diffs),
            }
        )

    dz014 = next(row for row in threshold_rows if row["threshold"] == "014")
    dz018 = next(row for row in threshold_rows if row["threshold"] == "018")
    selected = "014" if float(dz014["mean_delta"]) < float(dz018["mean_delta"]) else "018"
    all_negative = {
        threshold: all(
            float(row["mean_delta"]) < 0.0
            for row in split_threshold_rows
            if row["threshold"] == threshold
        )
        for threshold in THRESHOLDS
    }
    decision = {
        "selected_external_threshold_by_mean_delta": selected,
        "dz014_all_external_mean_delta": dz014["mean_delta"],
        "dz018_all_external_mean_delta": dz018["mean_delta"],
        "dz014_minus_dz018_all_external_mean_delta": float(dz014["mean_delta"]) - float(dz018["mean_delta"]),
        "all_splits_negative_vs_beta005": all_negative,
        "interpretation": (
            "external splits support promoting dz014/dz018 beyond OpenImages"
            if all_negative[selected]
            else "external splits do not yet support a broad promotion beyond beta005"
        ),
    }

    payload = {
        "decision": decision,
        "threshold_rows": threshold_rows,
        "split_threshold_rows": split_threshold_rows,
        "per_seed_rows": per_seed_rows,
        "pairwise_rows": pairwise_rows,
        "quartile_rows": quartile_rows,
    }
    SUMMARY_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(SUMMARY_PREFIX.with_suffix(".per_seed.csv"), per_seed_rows)
    write_csv(SUMMARY_PREFIX.with_suffix(".split_threshold.csv"), split_threshold_rows)
    write_csv(SUMMARY_PREFIX.with_suffix(".threshold.csv"), threshold_rows)
    write_csv(SUMMARY_PREFIX.with_suffix(".pairwise.csv"), pairwise_rows)
    write_csv(SUMMARY_PREFIX.with_suffix(".quartiles.csv"), quartile_rows)

    lines = [
        "# E117 Dead-Zone External Fixed-Protocol Audit",
        "",
        "This audit evaluates the step250 dead-zone HCG-RVQ branches on Kodak and CLIC validation splits against seed-matched beta005 guard references.",
        "",
        "## Decision",
        "",
        f"- Selected external threshold by weighted mean delta: `{selected}`",
        f"- dz014 all-external delta: `{format_float(dz014['mean_delta'])}`",
        f"- dz018 all-external delta: `{format_float(dz018['mean_delta'])}`",
        f"- dz014 - dz018 delta: `{format_float(decision['dz014_minus_dz018_all_external_mean_delta'])}`",
        f"- Interpretation: {decision['interpretation']}",
        "",
        "## Split Results",
        "",
        "| split | threshold | mean RD | beta005 ref | delta | win | q95 | nonfinite | qMSE | s_q | dead-code |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_threshold_rows:
        lines.append(
            "| {split} | dz{threshold} | {rd} | {ref} | {delta} | {win} | {q95} | {nonfinite} | {qmse} | {sq} | {dead} |".format(
                split=row["split"],
                threshold=row["threshold"],
                rd=format_float(row["mean_rd"]),
                ref=format_float(row["mean_reference_rd"]),
                delta=format_float(row["mean_delta"]),
                win=format_float(row["win_rate"]),
                q95=format_float(row["q95_delta"]),
                nonfinite=int(row["nonfinite_rows"]),
                qmse=format_float(row.get("mean_rvq_latent_quant_mse", "n/a")),
                sq=format_float(row.get("mean_rvq_s_q_mean", "n/a")),
                dead=format_float(row.get("mean_rvq_dead_code_ratio", "n/a")),
            )
        )
    lines.extend(
        [
            "",
            "## All External Splits",
            "",
            "| threshold | images | mean RD | beta005 ref | delta | win | q95 | nonfinite |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            "| dz{threshold} | {images} | {rd} | {ref} | {delta} | {win} | {q95} | {nonfinite} |".format(
                threshold=row["threshold"],
                images=int(row["num_images"]),
                rd=format_float(row["mean_rd"]),
                ref=format_float(row["mean_reference_rd"]),
                delta=format_float(row["mean_delta"]),
                win=format_float(row["win_rate"]),
                q95=format_float(row["q95_delta"]),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## Pairwise dz014 vs dz018",
            "",
            "| split | shared | dz014-dz018 delta | dz014 better rate | q95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            "| {split} | {shared} | {delta} | {rate} | {q95} |".format(
                split=row["split"],
                shared=int(row["num_shared"]),
                delta=format_float(row["mean_dz014_minus_dz018_delta"]),
                rate=format_float(row["dz014_better_than_dz018_rate"]),
                q95=format_float(row["q95_dz014_minus_dz018_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{SUMMARY_PREFIX.with_suffix('.json')}`",
            f"- `{SUMMARY_PREFIX.with_suffix('.split_threshold.csv')}`",
            f"- `{SUMMARY_PREFIX.with_suffix('.per_seed.csv')}`",
            f"- `{SUMMARY_PREFIX.with_suffix('.quartiles.csv')}`",
        ]
    )
    SUMMARY_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for seed in SEEDS:
            export_beta_reference(split, seed)
    for split in SPLITS:
        for threshold in THRESHOLDS:
            for seed in SEEDS:
                run_probe(split, threshold, seed)
    aggregate()


if __name__ == "__main__":
    main()
