#!/usr/bin/env python3
"""Audit E085 local-delta weighted reliability teacher results."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

REFERENCE_CSV = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv"
BETA_CSV = ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"

LOCAL_DELTA_THRESHOLD = 0.052714
RAW_GATE_THRESHOLD = 0.285817

VARIANTS = [
    {
        "name": "E078_marginw_step250",
        "family": "margin_weighted",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E078_marginw_step500",
        "family": "margin_weighted",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
    {
        "name": "E080_yhat_anchor_step250",
        "family": "yhat_anchor",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E080_yhat_anchor_step500",
        "family": "yhat_anchor",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
    {
        "name": "E085_localdelta_step250",
        "family": "localdelta_weighted",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E085_localdelta_step500",
        "family": "localdelta_weighted",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
]

FEATURES = [
    "rvq_householder_reliability_multiplier",
    "rvq_householder_reliability_multiplier_std",
    "rvq_householder_risk_multiplier",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else default


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return mean(values) if values else math.nan


def safe_fraction(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else math.nan


def load_reference() -> dict[str, dict[str, float]]:
    refs = {}
    for row in read_csv(REFERENCE_CSV):
        if row["seed"] != "3456":
            continue
        refs[row["path"]] = {
            "hcs": float(row["hcs_rd"]),
            "old": float(row["old_rd"]),
            "min090": float(row["min090_rd"]),
            "previous_local": float(row["previous_local_rd"]),
            "beta005": float(row["variant500_rd"]),
        }
    return refs


def quartile_means(items: list[tuple[float, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[0])
    n = len(ordered)
    out = {}
    for qi in range(4):
        chunk = ordered[qi * n // 4 : (qi + 1) * n // 4]
        out[f"Q{qi + 1}"] = safe_mean([value for _, value in chunk])
    return out


def subset_summary(rows: list[dict], selected: list[bool]) -> dict:
    selected_deltas = [row["delta_vs_beta"] for row, flag in zip(rows, selected) if flag]
    unselected_deltas = [row["delta_vs_beta"] for row, flag in zip(rows, selected) if not flag]
    mixed = [row["delta_vs_beta"] if flag else 0.0 for row, flag in zip(rows, selected)]
    return {
        "selected_fraction": safe_fraction(selected),
        "selected_mean_delta_vs_beta005": safe_mean(selected_deltas),
        "unselected_mean_delta_vs_beta005": safe_mean(unselected_deltas),
        "mixed_apply_only_selected_delta_vs_beta005": safe_mean(mixed),
        "mixed_quartile_delta_vs_beta005": quartile_means(
            [(row["hcs"], row["delta_vs_beta"] if flag else 0.0) for row, flag in zip(rows, selected)]
        ),
    }


def feature_stats(rows: list[dict], selected: list[bool]) -> dict[str, dict[str, float]]:
    out = {}
    for feature in FEATURES:
        values = [row[feature] for row in rows if math.isfinite(row[feature])]
        high = [row[feature] for row, flag in zip(rows, selected) if flag and math.isfinite(row[feature])]
        low = [row[feature] for row, flag in zip(rows, selected) if not flag and math.isfinite(row[feature])]
        out[feature] = {
            "mean": safe_mean(values),
            "selected_mean": safe_mean(high),
            "unselected_mean": safe_mean(low),
            "selected_minus_unselected": safe_mean(high) - safe_mean(low),
        }
    return out


def summarize_variant(
    variant: dict,
    refs: dict[str, dict[str, float]],
    beta_rows: dict[str, dict[str, str]],
    e080_same_step_rows: dict[str, dict[str, str]],
) -> dict:
    raw_rows = read_csv(variant["csv"])
    rows = []
    for row in raw_rows:
        path = row["path"]
        if path not in refs or path not in beta_rows:
            continue
        beta = beta_rows[path]
        e080 = e080_same_step_rows.get(path)
        item = {
            "path": path,
            "hcs": refs[path]["hcs"],
            "rd": f(row, "rd_score"),
            "beta_rd": refs[path]["beta005"],
            "delta_vs_beta": f(row, "rd_score") - refs[path]["beta005"],
            "delta_vs_e080": f(row, "rd_score") - f(e080, "rd_score") if e080 else math.nan,
            "beta_delta_rms": f(beta, "rvq_householder_delta_rms"),
            "beta_gate_raw": f(beta, "rvq_householder_gate_raw"),
            "has_nonfinite": int(f(row, "has_nonfinite", 0.0)),
        }
        for feature in FEATURES:
            item[feature] = f(row, feature)
        rows.append(item)

    high_delta = [row["beta_delta_rms"] >= LOCAL_DELTA_THRESHOLD for row in rows]
    high_gate = [row["beta_gate_raw"] >= RAW_GATE_THRESHOLD for row in rows]
    summary = read_json(variant["json"])["summaries"][0]
    return {
        "name": variant["name"],
        "family": variant["family"],
        "step": variant["step"],
        "csv": str(variant["csv"].relative_to(ROOT)),
        "json": str(variant["json"].relative_to(ROOT)),
        "rows": len(rows),
        "mean_rd": safe_mean([row["rd"] for row in rows]),
        "mean_delta_vs_beta005": safe_mean([row["delta_vs_beta"] for row in rows]),
        "mean_delta_vs_e080_same_step": safe_mean([row["delta_vs_e080"] for row in rows]),
        "win_fraction_vs_beta005": safe_fraction([row["delta_vs_beta"] < 0.0 for row in rows]),
        "nonfinite_rows": sum(row["has_nonfinite"] for row in rows),
        "json_mean_rd": summary.get("mean_rd"),
        "json_mean_delta_vs_reference": summary.get("mean_rd_minus_reference"),
        "quartile_delta_vs_beta005": quartile_means([(row["hcs"], row["delta_vs_beta"]) for row in rows]),
        "high_beta_delta_rms_selector": subset_summary(rows, high_delta),
        "high_beta_raw_gate_selector": subset_summary(rows, high_gate),
        "feature_stats_by_high_beta_delta_rms": feature_stats(rows, high_delta),
    }


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# E085 Local-Delta Weighted Teacher Audit",
        "",
        "This audit checks whether the E084 posthoc high-delta-RMS headroom becomes a learned controller when the image-level teacher is weighted on local delta-RMS maps.",
        "",
        "## Mean Results",
        "",
        "| variant | mean RD | beta005 delta | E080 same-step delta | win vs beta005 | nonfinite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {row['name']} | {fmt(row['mean_rd'])} | {fmt(row['mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['mean_delta_vs_e080_same_step'], signed=True)} | {fmt(row['win_fraction_vs_beta005'])} | {row['nonfinite_rows']} |"
        )

    lines += [
        "",
        "## HCS Difficulty Quartiles",
        "",
        "| variant | Q1 easy | Q2 | Q3 | Q4 hard |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        q = row["quartile_delta_vs_beta005"]
        lines.append(
            f"| {row['name']} | {fmt(q['Q1'], signed=True)} | {fmt(q['Q2'], signed=True)} | "
            f"{fmt(q['Q3'], signed=True)} | {fmt(q['Q4'], signed=True)} |"
        )

    lines += [
        "",
        "## Beta-Side High Delta-RMS Selector",
        "",
        f"Threshold: beta005 `rvq_householder_delta_rms >= {LOCAL_DELTA_THRESHOLD}`.",
        "",
        "| variant | selected | selected delta | unselected delta | mixed apply-selected-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        s = row["high_beta_delta_rms_selector"]
        lines.append(
            f"| {row['name']} | {fmt(s['selected_fraction'])} | "
            f"{fmt(s['selected_mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(s['unselected_mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(s['mixed_apply_only_selected_delta_vs_beta005'], signed=True)} |"
        )

    lines += [
        "",
        "## Intermediate Feature Split",
        "",
        "High/low groups use the same beta005 delta-RMS selector above.",
        "",
        "| variant | reliability mean | reliability high-low | strength mean | qMSE mean | dead-code mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        stats = row["feature_stats_by_high_beta_delta_rms"]
        lines.append(
            f"| {row['name']} | {fmt(stats['rvq_householder_reliability_multiplier']['mean'])} | "
            f"{fmt(stats['rvq_householder_reliability_multiplier']['selected_minus_unselected'], signed=True)} | "
            f"{fmt(stats['rvq_householder_strength']['mean'])} | "
            f"{fmt(stats['rvq_latent_quant_mse']['mean'])} | "
            f"{fmt(stats['rvq_dead_code_ratio']['mean'])} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Next Action",
        "",
        payload["next_action"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    refs = load_reference()
    beta_rows = {row["path"]: row for row in read_csv(BETA_CSV)}
    e080_by_step = {
        250: {
            row["path"]: row
            for row in read_csv(
                ANALYSIS
                / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"
            )
        },
        500: {
            row["path"]: row
            for row in read_csv(
                ANALYSIS
                / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv"
            )
        },
    }
    variants = [summarize_variant(v, refs, beta_rows, e080_by_step[v["step"]]) for v in VARIANTS]
    e085_best = min(
        [v for v in variants if v["family"] == "localdelta_weighted"],
        key=lambda row: row["mean_delta_vs_beta005"],
    )
    e080_best = min(
        [v for v in variants if v["family"] == "yhat_anchor"],
        key=lambda row: row["mean_delta_vs_beta005"],
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "split": "OpenImages holdout4096 seed3456 full-image",
            "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
            "beta_csv": str(BETA_CSV.relative_to(ROOT)),
            "local_delta_threshold": LOCAL_DELTA_THRESHOLD,
            "raw_gate_threshold": RAW_GATE_THRESHOLD,
        },
        "variants": variants,
        "decision": (
            "E085 is numerically stable, but local map weighting of the scalar reliability teacher does not yet convert "
            "the E084 posthoc high-risk headroom into a paper-main improvement. Treat it as an informative negative: "
            "the controller must learn a deployable selection/keep policy, not merely a stronger local weighting of the same image label."
        ),
        "next_action": (
            f"Do not promote E085. Use {e085_best['name']} only as evidence that qMSE/dead-code remain safe, then move to an E086 "
            "selector-style objective: preserve beta005 on low-risk images/locations and supervise suppression only on independently "
            "defined high-risk regions, with threshold/teacher chosen on transfer8192 rather than holdout4096."
        ),
        "best_e085": e085_best["name"],
        "best_e080": e080_best["name"],
    }
    out_json = ANALYSIS / "e085_localdelta_weighted_teacher_audit.json"
    out_md = ANALYSIS / "e085_localdelta_weighted_teacher_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, out_md)
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "best_e085": payload["best_e085"]}, indent=2))


if __name__ == "__main__":
    main()
