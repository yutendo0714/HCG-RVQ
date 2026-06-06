#!/usr/bin/env python3
"""Audit E086 selector keep-target reliability controller results."""

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
    {
        "name": "E086_selector_keep_step250",
        "family": "selector_keep_target",
        "step": 250,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
    {
        "name": "E086_selector_keep_step500",
        "family": "selector_keep_target",
        "step": 500,
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
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
    e085_same_step_rows: dict[str, dict[str, str]],
    e080_same_step_rows: dict[str, dict[str, str]],
) -> dict:
    raw_rows = read_csv(variant["csv"])
    rows = []
    for row in raw_rows:
        path = row["path"]
        if path not in refs or path not in beta_rows:
            continue
        beta = beta_rows[path]
        e085 = e085_same_step_rows.get(path)
        e080 = e080_same_step_rows.get(path)
        item = {
            "path": path,
            "hcs": refs[path]["hcs"],
            "rd": f(row, "rd_score"),
            "beta_rd": refs[path]["beta005"],
            "delta_vs_beta": f(row, "rd_score") - refs[path]["beta005"],
            "delta_vs_e085": f(row, "rd_score") - f(e085, "rd_score") if e085 else math.nan,
            "delta_vs_e080": f(row, "rd_score") - f(e080, "rd_score") if e080 else math.nan,
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
        "mean_delta_vs_e085_same_step": safe_mean([row["delta_vs_e085"] for row in rows]),
        "mean_delta_vs_e080_same_step": safe_mean([row["delta_vs_e080"] for row in rows]),
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
        "# E086 Selector Keep-Target Audit",
        "",
        "This audit checks whether an explicit transfer-derived keep/selector target improves the E085 local-weight-only controller.",
        "",
        "## Mean Results",
        "",
        "| variant | mean RD | beta005 delta | E085 same-step delta | E080 same-step delta | win vs beta005 | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {row['name']} | {fmt(row['mean_rd'])} | {fmt(row['mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['mean_delta_vs_e085_same_step'], signed=True)} | {fmt(row['mean_delta_vs_e080_same_step'], signed=True)} | "
            f"{fmt(row['win_fraction_vs_beta005'])} | {row['nonfinite_rows']} |"
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


def main() -> None:
    refs = load_reference()
    beta_rows = {row["path"]: row for row in read_csv(BETA_CSV)}
    e085_by_step = {
        250: {
            row["path"]: row
            for row in read_csv(
                ANALYSIS
                / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"
            )
        },
        500: {
            row["path"]: row
            for row in read_csv(
                ANALYSIS
                / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv"
            )
        },
    }
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
    variants = [summarize_variant(v, refs, beta_rows, e085_by_step[v["step"]], e080_by_step[v["step"]]) for v in VARIANTS]
    e086_rows = [v for v in variants if v["family"] == "selector_keep_target"]
    e086_best = min(e086_rows, key=lambda row: row["mean_delta_vs_beta005"])
    best_overall = min(variants, key=lambda row: row["mean_delta_vs_beta005"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "split": "OpenImages holdout4096 seed3456 full-image",
            "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
            "beta_csv": str(BETA_CSV.relative_to(ROOT)),
            "holdout_diagnostic_delta_threshold": HOLDOUT_DIAGNOSTIC_DELTA_THRESHOLD,
            "transfer_delta_threshold": TRANSFER_DELTA_THRESHOLD,
            "transfer_threshold_source": "experiments/analysis/e086_transfer_selector_thresholds.md",
        },
        "variants": variants,
        "decision": (
            "E086 is stable and its transfer-derived local keep-target does no numerical harm. It marginally improves the global "
            "mean over E085/E080, but it still does not improve beta005 and it weakens the high-risk selector/tail gain. The explicit "
            "keep target therefore closes the local-weight-only loophole as another safe negative rather than becoming a promotion candidate."
        ),
        "next_action": (
            "Do not spend more GPU on BCE-style reliability labels in this branch. Move the improvement track to a direct deployable "
            "selector or mixture objective: keep beta005 reconstruction/indices on low-risk locations, and train the selected high-risk "
            "path with an RD or ranking signal rather than only a reliability target."
        ),
        "best_e086": e086_best["name"],
        "best_overall_in_audit": best_overall["name"],
    }
    out_json = ANALYSIS / "e086_selector_keep_target_audit.json"
    out_md = ANALYSIS / "e086_selector_keep_target_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, out_md)
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "best_e086": e086_best["name"]}, indent=2))


if __name__ == "__main__":
    main()
