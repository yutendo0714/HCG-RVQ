#!/usr/bin/env python3
"""Build E259 full-training readiness audit after GLC CLIC calibration.

This report consolidates the current EF-LIC and GLC short-cycle evidence into
paper-safe promotion decisions.  The purpose is not to claim final performance;
it decides which HCG-RVQ variants are ready for codec-loop/full-training
experiments and which still need reliability/index control.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e259_full_training_readiness_after_glc_calib"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str] | None, key: str, default: int = 0) -> int:
    value = as_float(row, key)
    if not math.isfinite(value):
        return default
    return int(value)


def finite(value: float) -> bool:
    return math.isfinite(value)


def safe_num(value: float, digits: int = 6) -> str:
    if not finite(value):
        return "NA"
    return f"{value:.{digits}f}"


def first_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in matches.items()):
            return row
    return None


def min_by(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    candidates = [r for r in rows if finite(as_float(r, key))]
    if not candidates:
        return None
    return min(candidates, key=lambda r: as_float(r, key))


def max_by(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    candidates = [r for r in rows if finite(as_float(r, key))]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r, key))


def summarize_e246() -> dict[str, Any]:
    active = read_csv(ANALYSIS / "e246_eflic_decoder_safe_feature_groups.active_summary.csv")
    family = read_csv(ANALYSIS / "e246_eflic_decoder_safe_feature_groups.family_summary.csv")

    pooled_resub = [r for r in active if r.get("split") == "pooled_resub"]
    cross_ck = [r for r in active if r.get("split") == "train_clicpro41_test_kodak24"]
    cross_kc = [r for r in active if r.get("split") == "train_kodak24_test_clicpro41"]
    loio = [r for r in active if r.get("split", "").startswith("loio__")]

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in loio:
        grouped[(row.get("feature_group", ""), row.get("threshold_mode", ""))].append(row)

    loio_agg: list[dict[str, str]] = []
    for (feature_group, threshold_mode), rows in grouped.items():
        loio_agg.append(
            {
                "feature_group": feature_group,
                "threshold_mode": threshold_mode,
                "test_images": str(len(rows)),
                "recall": safe_num(mean(as_float(r, "recall") for r in rows)),
                "fpr": safe_num(mean(as_float(r, "fpr") for r in rows)),
                "f1": safe_num(mean(as_float(r, "f1") for r in rows)),
                "weighted_risk": safe_num(mean(as_float(r, "weighted_risk") for r in rows)),
                "active_pred_frac": safe_num(mean(as_float(r, "active_pred_frac") for r in rows)),
            }
        )

    family_resub = [r for r in family if r.get("split") == "pooled_resub"]
    family_ck = [r for r in family if r.get("split") == "train_clicpro41_test_kodak24"]
    family_kc = [r for r in family if r.get("split") == "train_kodak24_test_clicpro41"]

    representative = pooled_resub[0] if pooled_resub else {}
    return {
        "images": as_int(representative, "test_images"),
        "active_frac": as_float(representative, "active_frac"),
        "pooled_resub_best_f1": max_by(
            [r for r in pooled_resub if r.get("threshold_mode") == "best_f1"], "f1"
        ),
        "pooled_resub_min_risk": min_by(
            [r for r in pooled_resub if r.get("threshold_mode") == "min_weighted_risk"],
            "weighted_risk",
        ),
        "cross_clic_to_kodak_min_risk": min_by(
            [r for r in cross_ck if r.get("threshold_mode") == "min_weighted_risk"],
            "weighted_risk",
        ),
        "cross_kodak_to_clic_min_risk": min_by(
            [r for r in cross_kc if r.get("threshold_mode") == "min_weighted_risk"],
            "weighted_risk",
        ),
        "loio_min_risk": min_by(
            [r for r in loio_agg if r.get("threshold_mode") == "min_weighted_risk"],
            "weighted_risk",
        ),
        "family_resub_best": max_by(family_resub, "family_accuracy"),
        "family_cross_clic_to_kodak_best": max_by(family_ck, "family_accuracy"),
        "family_cross_kodak_to_clic_best": max_by(family_kc, "family_accuracy"),
    }


def summarize_e247() -> dict[str, Any]:
    rows = read_csv(ANALYSIS / "e247_loss_objective_audit.csv")
    return {
        "configs": len(rows),
        "rd_commit_only": sum(1 for r in rows if as_int(r, "noncore_count") == 0),
        "teacher_selector_or_anchor": sum(
            1
            for r in rows
            if as_int(r, "teacher_selector_count") > 0 or as_int(r, "anchor_count") > 0
        ),
        "geometry_gate_regularizer": sum(1 for r in rows if as_int(r, "regularizer_count") > 0),
    }


def summarize_glc() -> dict[str, Any]:
    policies = read_csv(ANALYSIS / "e257_glc_domain_mixed_with_cliccalib_gate_readiness.policies.csv")
    groups = read_csv(ANALYSIS / "e257_glc_domain_mixed_with_cliccalib_gate_readiness.groups.csv")
    controller = read_csv(ANALYSIS / "e258_glc_linear_controller_proxy_with_cliccalib.summary.csv")

    def policy(name: str) -> dict[str, str] | None:
        return first_row(policies, policy=name)

    def ctl(name: str) -> dict[str, str] | None:
        return first_row(controller, policy=name)

    return {
        "e257": {
            "all_on": policy("primary_all_on"),
            "oracle": policy("primary_oracle_with_side"),
            "best_internal_threshold": policy("primary_best_internal_active_rvq_mse_>="),
            "leave_domain_internal_threshold": policy("primary_leave_domain_active_rvq_mse_>="),
            "groups": groups,
        },
        "e258": {
            "best_deployable_loocv": ctl("loocv_rate_proxy_score_regressor"),
            "branch_internal_loocv": ctl("loocv_branch_internal_score_regressor"),
            "branch_plus_rate_loocv": ctl("loocv_branch_plus_rate_score_regressor"),
            "branch_internal_leave_domain": ctl("leave_domain_out_branch_internal_score_regressor"),
            "analysis_upper_loocv": ctl("loocv_analysis_upper_score_regressor"),
        },
    }


def metric(row: dict[str, str] | None, key: str, digits: int = 6) -> str:
    return safe_num(as_float(row, key), digits)


def row_label(row: dict[str, str] | None, keys: list[str]) -> str:
    if row is None:
        return "missing"
    parts = []
    for key in keys:
        value = row.get(key, "")
        if value != "":
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "available"


def build_decision_rows(eflic: dict[str, Any], glc: dict[str, Any], loss: dict[str, Any]) -> list[dict[str, str]]:
    e257 = glc["e257"]
    e258 = glc["e258"]
    return [
        {
            "track": "EF-LIC",
            "candidate": "frozen decoder-safe reliability controller",
            "status": "blocked_for_paper_main",
            "headline_metric": (
                f"pooled min-risk={metric(eflic['pooled_resub_min_risk'], 'weighted_risk')}; "
                f"LOIO min-risk={metric(eflic['loio_min_risk'], 'weighted_risk')}"
            ),
            "blocker": "resub/pooled features do not transfer reliably across held-out images/domains",
            "next_action": (
                "use E246 features as diagnostics/warmup only; train a compact codec-loop HCG branch "
                "with conservative fallback under the original objective"
            ),
        },
        {
            "track": "GLC",
            "candidate": "E250 all-on tail RVQ branch",
            "status": "blocked_for_full_training_claim",
            "headline_metric": (
                f"all-on score={metric(e257['all_on'], 'mean_combined')}; "
                f"oracle score={metric(e257['oracle'], 'mean_combined')}"
            ),
            "blocker": "CLIC slices show positive rate/perceptual overhead; branch helps only selected images",
            "next_action": (
                "learn reliability/index-prior control before codec-loop promotion; keep DISTS+bpp guard"
            ),
        },
        {
            "track": "GLC",
            "candidate": "linear reliability controller proxy",
            "status": "diagnostic_positive_but_not_domain_stable",
            "headline_metric": (
                f"deployable LOOCV score={metric(e258['best_deployable_loocv'], 'mean_score')}; "
                f"leave-domain score={metric(e258['branch_internal_leave_domain'], 'mean_score')}"
            ),
            "blocker": "LOOCV gain is tiny and leave-domain behavior remains fragile/silent",
            "next_action": (
                "expand calibration labels and replace hand gate with a small learned controller tied to index cost"
            ),
        },
        {
            "track": "Objective",
            "candidate": "loss policy for full training",
            "status": "guardrail",
            "headline_metric": (
                f"{loss['rd_commit_only']}/{loss['configs']} configs are RD/commit-only; "
                f"{loss['teacher_selector_or_anchor']} use teacher/selector/anchor terms"
            ),
            "blocker": "auxiliary-heavy objectives can improve diagnostics while diluting RD optimization",
            "next_action": (
                "keep paper-main loss simple: original RD/perceptual terms plus direct VQ/index/commitment terms"
            ),
        },
    ]


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row.get(c, "") for c in columns) + " |")
    return "\n".join(out)


def write_outputs(report: dict[str, Any], rows: list[dict[str, str]]) -> None:
    csv_path = OUT_PREFIX.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track", "candidate", "status", "headline_metric", "blocker", "next_action"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_PREFIX.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    eflic = report["eflic_e246"]
    glc = report["glc_e257_e258"]
    loss = report["loss_e247"]
    e257 = glc["e257"]
    e258 = glc["e258"]

    md = [
        "# E259 Full-Training Readiness After GLC Calibration",
        "",
        "## Purpose",
        "",
        "This audit converts the current EF-LIC and GLC short-cycle results into a "
        "full-training promotion decision. It is a gate, not a final performance claim.",
        "",
        "The prompt-level thesis remains: hyperprior/context should generate local "
        "quantizer geometry, not only entropy parameters. The current evidence says this "
        "should be pursued through compact reliability/index-aware HCG branches, not "
        "through dense all-on VQ insertion.",
        "",
        "## Decision Matrix",
        "",
        markdown_table(rows, ["track", "candidate", "status", "headline_metric", "next_action"]),
        "",
        "## EF-LIC Evidence",
        "",
        f"- E246 images: {eflic['images']}, active fraction: {safe_num(eflic['active_frac'])}.",
        f"- Pooled resub min-risk: {metric(eflic['pooled_resub_min_risk'], 'weighted_risk')} "
        f"with {row_label(eflic['pooled_resub_min_risk'], ['feature_group', 'threshold_mode', 'fpr', 'recall'])}.",
        f"- LOIO min-risk aggregate: {metric(eflic['loio_min_risk'], 'weighted_risk')} "
        f"with {row_label(eflic['loio_min_risk'], ['feature_group', 'threshold_mode', 'fpr', 'recall'])}.",
        f"- CLIC->Kodak min-risk: {metric(eflic['cross_clic_to_kodak_min_risk'], 'weighted_risk')} "
        f"with {row_label(eflic['cross_clic_to_kodak_min_risk'], ['feature_group', 'threshold_mode', 'fpr', 'recall'])}.",
        f"- Kodak->CLIC min-risk: {metric(eflic['cross_kodak_to_clic_min_risk'], 'weighted_risk')} "
        f"with {row_label(eflic['cross_kodak_to_clic_min_risk'], ['feature_group', 'threshold_mode', 'fpr', 'recall'])}.",
        "",
        "Interpretation: EF-LIC has enough local signal to justify a learned HCG branch, "
        "but the frozen decoder-safe controller is not reliable enough for paper-main "
        "selection. The next EF-LIC experiment should be codec-loop training with the "
        "original EF-LIC objective dominant and only light VQ/index terms.",
        "",
        "## GLC Evidence",
        "",
        f"- E257 all-on score: {metric(e257['all_on'], 'mean_combined')} "
        f"({metric(e257['all_on'], 'selected')}/{metric(e257['all_on'], 'total')} selected).",
        f"- E257 oracle score: {metric(e257['oracle'], 'mean_combined')} "
        f"({metric(e257['oracle'], 'selected')}/{metric(e257['oracle'], 'total')} selected).",
        f"- E258 deployable LOOCV rate-proxy regressor: score {metric(e258['best_deployable_loocv'], 'mean_score')}, "
        f"selected {metric(e258['best_deployable_loocv'], 'selected')}/{metric(e258['best_deployable_loocv'], 'total')}.",
        f"- E258 analysis-upper LOOCV: score {metric(e258['analysis_upper_loocv'], 'mean_score')} "
        "(diagnostic only, because it uses non-deployable information).",
        "",
        "Interpretation: GLC tail RVQ can help selected images, but all-on insertion is "
        "still harmful under the DISTS + 3*LPIPS + bpp guard. The next GLC experiment "
        "should attach a small reliability/index prior to decide when the branch earns "
        "its rate cost.",
        "",
        "## Loss Policy",
        "",
        f"- Audited configs: {loss['configs']}.",
        f"- RD/commit-only configs: {loss['rd_commit_only']}.",
        f"- Teacher/selector/anchor configs: {loss['teacher_selector_or_anchor']}.",
        f"- Geometry/gate regularizer configs: {loss['geometry_gate_regularizer']}.",
        "",
        "The paper-main objective should remain simple: keep the original codec RD or "
        "perceptual-RD loss dominant, then add only direct VQ commitment/index terms "
        "that are needed to make the quantizer trainable and rate-aware. Teacher and "
        "selector losses stay diagnostic unless ablated as auxiliary training aids.",
        "",
        "## Promotion Rule",
        "",
        "Do not promote dense all-on EF-LIC/GLC HCG-RVQ to full training yet. Promote "
        "a candidate only after a mid-scale codec-loop run shows finite training, no "
        "dead-code collapse, non-worsening DISTS+bpp guarded score, and held-out "
        "activation that beats the no-branch fallback.",
        "",
        "The next paper-safe implementation target is therefore: compact local HCG-RVQ "
        "branch + learned reliability/index control + conservative fallback, trained "
        "with the original codec objective dominant.",
        "",
    ]
    md_path = OUT_PREFIX.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    eflic = summarize_e246()
    glc = summarize_glc()
    loss = summarize_e247()
    rows = build_decision_rows(eflic, glc, loss)
    report = {
        "prompt_thesis": "Hyperprior/context should generate local quantizer geometry, not only entropy parameters.",
        "eflic_e246": eflic,
        "glc_e257_e258": glc,
        "loss_e247": loss,
        "decision_rows": rows,
    }
    write_outputs(report, rows)
    print(f"wrote {OUT_PREFIX.with_suffix('.md')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.json')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
