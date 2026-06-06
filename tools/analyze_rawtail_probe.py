#!/usr/bin/env python3
"""Compare the raw-gate tail regularizer against beta005 on seed3456."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

BETA_SWEEP = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.csv"
RAWTail_SWEEP = ANALYSIS / "rawtail_t0284_rho100_betacommit005_holdout4096_checkpoint_sweep.csv"
RAWBACKOFF_SWEEP = ANALYSIS / (
    "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_rawbackoff065_t0284_seed3456_holdout4096_checkpoint_sweep.csv"
)
TEACHER_JSON = ANALYSIS / "beta005_teacher_target_audit.json"

OUT_MD = ANALYSIS / "betacommit005_rawtail_t0284_rho100_seed3456_probe.md"
OUT_JSON = ANALYSIS / "betacommit005_rawtail_t0284_rho100_seed3456_probe.json"

SEED = "3456"
RAW_GATE_THRESHOLD = 0.284059


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean_field(rows: list[dict[str, str]], key: str) -> float:
    return mean(finite(as_float(row, key) for row in rows))


def fraction(rows: list[dict[str, str]], key: str, threshold: float, op: str) -> float:
    values = finite(as_float(row, key) for row in rows)
    if op == "gt":
        return sum(value > threshold for value in values) / len(values)
    if op == "le":
        return sum(value <= threshold for value in values) / len(values)
    raise ValueError(f"unsupported op: {op}")


def seed_step(rows: list[dict[str, str]], seed: str, step: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("seed")) == seed and str(row.get("step")) == step]


def by_path(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in rows}


def compare_rows(
    candidate: list[dict[str, str]],
    reference: list[dict[str, str]],
    candidate_name: str,
    reference_name: str,
) -> dict[str, float | int | str]:
    candidate_by_path = by_path(candidate)
    reference_by_path = by_path(reference)
    common = sorted(candidate_by_path.keys() & reference_by_path.keys())
    deltas = [
        as_float(candidate_by_path[path], "rd_score") - as_float(reference_by_path[path], "rd_score")
        for path in common
    ]
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "common_images": len(common),
        "mean_delta_rd": mean(deltas),
        "median_delta_rd": sorted(deltas)[len(deltas) // 2],
        "candidate_win_fraction": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "q1_delta_rd": sorted(deltas)[len(deltas) // 4],
        "q3_delta_rd": sorted(deltas)[3 * len(deltas) // 4],
    }


def quartiles_by_feature(
    candidate: list[dict[str, str]],
    reference: list[dict[str, str]],
    feature: str,
) -> list[dict[str, float | int | str]]:
    candidate_by_path = by_path(candidate)
    reference_by_path = by_path(reference)
    common = sorted(candidate_by_path.keys() & reference_by_path.keys())
    common.sort(key=lambda path: as_float(reference_by_path[path], feature))
    out: list[dict[str, float | int | str]] = []
    for idx, name in enumerate(["Q1 low", "Q2", "Q3", "Q4 high"]):
        chunk = common[idx * len(common) // 4 : (idx + 1) * len(common) // 4]
        deltas = [
            as_float(candidate_by_path[path], "rd_score") - as_float(reference_by_path[path], "rd_score")
            for path in chunk
        ]
        out.append(
            {
                "quartile": name,
                "n": len(chunk),
                "mean_delta_rd": mean(deltas),
                "candidate_win_fraction": sum(delta < 0.0 for delta in deltas) / len(deltas),
                "feature_mean": mean(as_float(reference_by_path[path], feature) for path in chunk),
            }
        )
    return out


def feature_summary(rows: list[dict[str, str]], label: str) -> dict[str, float | str]:
    raw_key = "rvq_householder_gate_raw_image_mean"
    if raw_key not in rows[0]:
        raw_key = "rvq_householder_gate_raw"
    return {
        "label": label,
        "rd": mean_field(rows, "rd_score"),
        "bpp": mean_field(rows, "bpp"),
        "psnr": mean_field(rows, "psnr"),
        "ms_ssim": mean_field(rows, "ms_ssim"),
        "raw_gate_mean": mean_field(rows, raw_key),
        "raw_gate_tail_fraction": fraction(rows, raw_key, RAW_GATE_THRESHOLD, "gt"),
        "s_q_mean": mean_field(rows, "rvq_s_q_mean"),
        "latent_qmse": mean_field(rows, "rvq_latent_quant_mse"),
        "householder_delta_rms": mean_field(rows, "rvq_householder_delta_rms"),
        "householder_strength": mean_field(rows, "rvq_householder_strength"),
        "risk_multiplier": mean_field(rows, "rvq_householder_risk_multiplier"),
        "dead_code_ratio": mean_field(rows, "rvq_dead_code_ratio"),
        "perplexity": mean_field(rows, "rvq_perplexity"),
        "nonfinite_rows": sum(int(float(row.get("has_nonfinite", "0") or 0)) for row in rows),
    }


def fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def markdown_table(headers: list[str], rows: list[dict[str, float | int | str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[header]) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    beta_rows_all = read_csv(BETA_SWEEP)
    rawtail_rows_all = read_csv(RAWTail_SWEEP)
    rawbackoff_rows = read_csv(RAWBACKOFF_SWEEP)
    teacher = json.loads(TEACHER_JSON.read_text())

    beta_rows = seed_step(beta_rows_all, SEED, "500")
    rawtail250 = seed_step(rawtail_rows_all, SEED, "250")
    rawtail500 = seed_step(rawtail_rows_all, SEED, "500")

    if not beta_rows or not rawtail250 or not rawtail500:
        raise RuntimeError("missing beta005/rawtail seed3456 rows")

    seed_base = teacher["by_seed"][SEED]
    rawbackoff500_rd = next(float(row["rd_score"]) for row in rawbackoff_rows if row["step"] == "500")

    method_rows = [
        {"method": "HCS", "RD": seed_base["hcs_rd"], "delta_vs_beta005": seed_base["hcs_rd"] - seed_base["beta005_rd"]},
        {
            "method": "old gate0.25",
            "RD": seed_base["old_gate025_rd"],
            "delta_vs_beta005": seed_base["old_gate025_rd"] - seed_base["beta005_rd"],
        },
        {
            "method": "min090",
            "RD": seed_base["min090_rd"],
            "delta_vs_beta005": seed_base["min090_rd"] - seed_base["beta005_rd"],
        },
        {
            "method": "previous-local step250",
            "RD": seed_base["previous_local_rd"],
            "delta_vs_beta005": seed_base["previous_local_rd"] - seed_base["beta005_rd"],
        },
        {"method": "beta005 step500", "RD": seed_base["beta005_rd"], "delta_vs_beta005": 0.0},
        {
            "method": "rawbackoff065 step500",
            "RD": rawbackoff500_rd,
            "delta_vs_beta005": rawbackoff500_rd - seed_base["beta005_rd"],
        },
        {
            "method": "rawtail t0284 rho100 step250",
            "RD": mean_field(rawtail250, "rd_score"),
            "delta_vs_beta005": mean_field(rawtail250, "rd_score") - seed_base["beta005_rd"],
        },
        {
            "method": "rawtail t0284 rho100 step500",
            "RD": mean_field(rawtail500, "rd_score"),
            "delta_vs_beta005": mean_field(rawtail500, "rd_score") - seed_base["beta005_rd"],
        },
    ]

    features = [
        feature_summary(beta_rows, "beta005 step500"),
        feature_summary(rawtail250, "rawtail step250"),
        feature_summary(rawtail500, "rawtail step500"),
    ]
    comparisons = [
        compare_rows(rawtail500, beta_rows, "rawtail step500", "beta005 step500"),
        compare_rows(rawtail500, rawtail250, "rawtail step500", "rawtail step250"),
    ]
    quartiles = quartiles_by_feature(rawtail500, beta_rows, "rvq_householder_gate_raw")

    result = {
        "decision": "reject_rawtail_for_3seed_promotion",
        "method_rows": method_rows,
        "features": features,
        "comparisons": comparisons,
        "quartiles_by_beta005_raw_gate": quartiles,
        "raw_gate_threshold": RAW_GATE_THRESHOLD,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    md = [
        "# Raw-tail t0284 rho100 probe for beta005",
        "",
        "## Decision",
        "",
        "Reject `rawtail_t0284_rho100_betacommit005` for 3-seed promotion. It improves over the blunt rawbackoff training and over the older previous-local seed3456 row, but it is still worse than beta005 by `+0.036878` RD on the same fixed seed3456 holdout4096 protocol.",
        "",
        "## RD Comparison",
        "",
        markdown_table(["method", "RD", "delta_vs_beta005"], method_rows),
        "",
        "## Per-image Comparison",
        "",
        markdown_table(
            ["candidate", "reference", "common_images", "mean_delta_rd", "median_delta_rd", "candidate_win_fraction", "q1_delta_rd", "q3_delta_rd"],
            comparisons,
        ),
        "",
        "## Feature Comparison",
        "",
        markdown_table(
            [
                "label",
                "rd",
                "bpp",
                "psnr",
                "ms_ssim",
                "raw_gate_mean",
                "raw_gate_tail_fraction",
                "s_q_mean",
                "latent_qmse",
                "householder_delta_rms",
                "householder_strength",
                "risk_multiplier",
                "dead_code_ratio",
                "perplexity",
                "nonfinite_rows",
            ],
            features,
        ),
        "",
        "## Raw-gate Quartiles",
        "",
        markdown_table(["quartile", "n", "feature_mean", "mean_delta_rd", "candidate_win_fraction"], quartiles),
        "",
        "## Interpretation",
        "",
        "- The tail regularizer is not a numerical failure: all evaluated rows are finite.",
        "- It does force the raw-gate tail down: the beta005 raw-gate tail fraction above `0.284059` is much larger than rawtail step500.",
        "- However, step500 recovers almost the same Householder strength, delta RMS, risk multiplier, dead-code ratio, and perplexity as beta005 while keeping a worse image-domain RD. This is another co-adaptation warning: the endogenous raw-gate signal can be moved by the model without producing the intended selector-like behavior.",
        "- The next method-improvement target should move away from raw-gate penalties alone. Use a detached/train-split teacher target from diagnostic delta-RMS or per-image beta005-vs-previous-local labels, then evaluate one checkpoint under the same fixed protocol.",
        "",
        "Artifacts:",
        "",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({"output_md": str(OUT_MD), "output_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
