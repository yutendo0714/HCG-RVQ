#!/usr/bin/env python3
"""Audit E087 RD-only head-only reliability controller results."""

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

HOLDOUT_DIAGNOSTIC_DELTA_THRESHOLD = 0.052714
TRANSFER_DELTA_THRESHOLD = 0.045151

VARIANTS = [
    {
        "name": "E080_yhat_anchor_step250",
        "family": "yhat_anchor_bce",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E080_yhat_anchor_step500",
        "family": "yhat_anchor_bce",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
    {
        "name": "E085_localdelta_step250",
        "family": "localdelta_weighted_bce",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E085_localdelta_step500",
        "family": "localdelta_weighted_bce",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
    {
        "name": "E086_selector_keep_step250",
        "family": "selector_keep_target_bce",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E086_selector_keep_step500",
        "family": "selector_keep_target_bce",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
    },
    {
        "name": "E087_rdonly_step250",
        "family": "rdonly_yhat_anchor",
        "step": 250,
        "csv": ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E087_rdonly_step500",
        "family": "rdonly_yhat_anchor",
        "step": 500,
        "csv": ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
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


def f(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
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
    e085_same_step_rows: dict[str, dict[str, str]],
    e086_same_step_rows: dict[str, dict[str, str]],
) -> dict:
    raw_rows = read_csv(variant["csv"])
    rows = []
    for row in raw_rows:
        path = row["path"]
        if path not in refs or path not in beta_rows:
            continue
        beta = beta_rows[path]
        item = {
            "path": path,
            "hcs": refs[path]["hcs"],
            "rd": f(row, "rd_score"),
            "beta_rd": refs[path]["beta005"],
            "delta_vs_beta": f(row, "rd_score") - refs[path]["beta005"],
            "delta_vs_e080": f(row, "rd_score") - f(e080_same_step_rows.get(path), "rd_score"),
            "delta_vs_e085": f(row, "rd_score") - f(e085_same_step_rows.get(path), "rd_score"),
            "delta_vs_e086": f(row, "rd_score") - f(e086_same_step_rows.get(path), "rd_score"),
            "beta_delta_rms": f(beta, "rvq_householder_delta_rms"),
            "has_nonfinite": int(f(row, "has_nonfinite", 0.0)),
        }
        for feature in FEATURES:
            item[feature] = f(row, feature)
        rows.append(item)

    holdout_high_delta = [row["beta_delta_rms"] >= HOLDOUT_DIAGNOSTIC_DELTA_THRESHOLD for row in rows]
    transfer_high_delta = [row["beta_delta_rms"] >= TRANSFER_DELTA_THRESHOLD for row in rows]
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
        "mean_abs_delta_vs_beta005": safe_mean([abs(row["delta_vs_beta"]) for row in rows]),
        "mean_delta_vs_e080_same_step": safe_mean([row["delta_vs_e080"] for row in rows]),
        "mean_delta_vs_e085_same_step": safe_mean([row["delta_vs_e085"] for row in rows]),
        "mean_delta_vs_e086_same_step": safe_mean([row["delta_vs_e086"] for row in rows]),
        "win_fraction_vs_beta005": safe_fraction([row["delta_vs_beta"] < 0.0 for row in rows]),
        "nonfinite_rows": sum(row["has_nonfinite"] for row in rows),
        "json_mean_rd": summary.get("mean_rd"),
        "json_mean_delta_vs_reference": summary.get("mean_rd_minus_reference"),
        "quartile_delta_vs_beta005": quartile_means([(row["hcs"], row["delta_vs_beta"]) for row in rows]),
        "holdout_diagnostic_delta_selector": subset_summary(rows, holdout_high_delta),
        "transfer_delta_selector": subset_summary(rows, transfer_high_delta),
        "feature_stats_by_transfer_delta_selector": feature_stats(rows, transfer_high_delta),
    }


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def write_selector_table(lines: list[str], variants: list[dict], title: str, key: str, threshold: float) -> None:
    lines += [
        "",
        f"## {title}",
        "",
        f"Threshold: beta005 `rvq_householder_delta_rms >= {threshold}`.",
        "",
        "| variant | selected | selected delta | unselected delta | mixed apply-selected-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in variants:
        s = row[key]
        lines.append(
            f"| {row['name']} | {fmt(s['selected_fraction'])} | "
            f"{fmt(s['selected_mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(s['unselected_mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(s['mixed_apply_only_selected_delta_vs_beta005'], signed=True)} |"
        )


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# E087 RD-Only Head Probe Audit",
        "",
        "This audit checks whether removing the BCE-style teacher and training only the reliability head through the RD objective plus a y-hat anchor gives a better beta005-preserving controller.",
        "",
        "## Mean Results",
        "",
        "| variant | mean RD | beta005 delta | vs E080 | vs E085 | vs E086 | win vs beta005 | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {row['name']} | {fmt(row['mean_rd'])} | {fmt(row['mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['mean_delta_vs_e080_same_step'], signed=True)} | {fmt(row['mean_delta_vs_e085_same_step'], signed=True)} | "
            f"{fmt(row['mean_delta_vs_e086_same_step'], signed=True)} | {fmt(row['win_fraction_vs_beta005'])} | {row['nonfinite_rows']} |"
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

    write_selector_table(
        lines,
        payload["variants"],
        "Holdout Diagnostic High Delta-RMS Selector",
        "holdout_diagnostic_delta_selector",
        HOLDOUT_DIAGNOSTIC_DELTA_THRESHOLD,
    )
    write_selector_table(
        lines,
        payload["variants"],
        "Transfer-Derived High Delta-RMS Selector",
        "transfer_delta_selector",
        TRANSFER_DELTA_THRESHOLD,
    )

    lines += [
        "",
        "## Intermediate Feature Split",
        "",
        "High/low groups use the transfer-derived delta-RMS selector.",
        "",
        "| variant | reliability mean | reliability high-low | strength mean | qMSE mean | s_q mean | dead-code mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        stats = row["feature_stats_by_transfer_delta_selector"]
        lines.append(
            f"| {row['name']} | {fmt(stats['rvq_householder_reliability_multiplier']['mean'])} | "
            f"{fmt(stats['rvq_householder_reliability_multiplier']['selected_minus_unselected'], signed=True)} | "
            f"{fmt(stats['rvq_householder_strength']['mean'])} | "
            f"{fmt(stats['rvq_latent_quant_mse']['mean'])} | "
            f"{fmt(stats['rvq_s_q_mean']['mean'])} | "
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


def rows_by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_csv(path)}


def main() -> None:
    refs = load_reference()
    beta_rows = rows_by_path(BETA_CSV)
    e080_by_step = {
        250: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"),
        500: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv"),
    }
    e085_by_step = {
        250: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"),
        500: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv"),
    }
    e086_by_step = {
        250: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"),
        500: rows_by_path(ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv"),
    }
    variants = [
        summarize_variant(
            variant,
            refs,
            beta_rows,
            e080_by_step[variant["step"]],
            e085_by_step[variant["step"]],
            e086_by_step[variant["step"]],
        )
        for variant in VARIANTS
    ]
    e087_rows = [v for v in variants if v["family"] == "rdonly_yhat_anchor"]
    best_e087 = min(e087_rows, key=lambda row: row["mean_delta_vs_beta005"])
    best_overall = min(variants, key=lambda row: row["mean_delta_vs_beta005"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "split": "OpenImages holdout4096 seed3456 full-image",
            "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
            "beta_csv": str(BETA_CSV.relative_to(ROOT)),
            "holdout_diagnostic_delta_threshold": HOLDOUT_DIAGNOSTIC_DELTA_THRESHOLD,
            "transfer_delta_threshold": TRANSFER_DELTA_THRESHOLD,
        },
        "variants": variants,
        "best_e087": best_e087["name"],
        "best_overall_in_audit": best_overall["name"],
        "decision": (
            "E087 is the best beta005-preserving controller in this local branch and confirms that outcome-level RD training is a better direction "
            "than BCE teacher targets. However, the best checkpoint still remains slightly worse than beta005 on mean RD, so it is not a paper-main "
            "promotion. Its value is narrowing the controller design: keep the RD/y-hat anchored shell, but add a deployable selective or ranking "
            "objective instead of treating reliability as a supervised target."
        ),
        "next_action": (
            "Use E087 as the new controller baseline for the improvement track. The next experiment should add explicit selection or pairwise "
            "ranking against beta005/local-cap outcomes while preserving beta005 reconstruction, RVQ assignment usage, qMSE, s_q, and dead-code "
            "statistics on low-risk regions."
        ),
    }
    out_json = ANALYSIS / "e087_rdonly_head_probe_audit.json"
    out_md = ANALYSIS / "e087_rdonly_head_probe_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, out_md)
    print(
        json.dumps(
            {
                "json": str(out_json),
                "markdown": str(out_md),
                "best_e087": best_e087["name"],
                "best_overall": best_overall["name"],
                "best_e087_delta": best_e087["mean_delta_vs_beta005"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
