#!/usr/bin/env python3
"""Summarize E083 full-image controller closure and teacher headroom."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

BETA_JSON = ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.json"
REFERENCE_CSV = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv"
TRANSFER_TEACHER_JSON = ANALYSIS / "beta005_previous_local_teacher_labels_transfer8192.json"

VARIANTS = [
    {
        "name": "beta005_fullimage",
        "family": "baseline",
        "step": 500,
        "json": BETA_JSON,
        "csv": ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv",
        "note": "paper-main fixed checkpoint",
    },
    {
        "name": "E076_rel075_rho005_step250",
        "family": "binary_teacher",
        "step": 250,
        "json": ANALYSIS / "teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.csv",
        "note": "head-only reliability, rho=0.05",
    },
    {
        "name": "E076_rel075_rho005_step500",
        "family": "binary_teacher",
        "step": 500,
        "json": ANALYSIS / "teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
        "note": "head-only reliability, rho=0.05",
    },
    {
        "name": "E077_rel075_rho050_step250",
        "family": "binary_teacher",
        "step": 250,
        "json": ANALYSIS / "teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.csv",
        "note": "head-only reliability, rho=0.50",
    },
    {
        "name": "E077_rel075_rho050_step500",
        "family": "binary_teacher",
        "step": 500,
        "json": ANALYSIS / "teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
        "note": "head-only reliability, rho=0.50",
    },
    {
        "name": "E078_marginw_rho050_step250",
        "family": "margin_weighted",
        "step": 250,
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "note": "margin-weighted teacher",
    },
    {
        "name": "E078_marginw_rho050_step500",
        "family": "margin_weighted",
        "step": 500,
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "note": "margin-weighted teacher",
    },
    {
        "name": "E079_rel095_marginw_lrm025_step250",
        "family": "low_amplitude_margin_weighted",
        "step": 250,
        "json": ANALYSIS / "teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "note": "reliability floor 0.95, low LR",
    },
    {
        "name": "E079_rel095_marginw_lrm025_step500",
        "family": "low_amplitude_margin_weighted",
        "step": 500,
        "json": ANALYSIS / "teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "note": "reliability floor 0.95, low LR",
    },
    {
        "name": "E080_yhat_anchor_step250",
        "family": "yhat_anchor",
        "step": 250,
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
        "note": "margin-weighted teacher with y_hat anchor",
    },
    {
        "name": "E080_yhat_anchor_step500",
        "family": "yhat_anchor",
        "step": 500,
        "json": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.json",
        "csv": ANALYSIS / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
        "note": "margin-weighted teacher with y_hat anchor",
    },
]


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def read_rows(path: Path) -> list[dict[str, str]]:
    actual = path
    if not actual.exists():
        actual = next(v["fallback_csv"] for v in VARIANTS if v.get("csv") == path and v.get("fallback_csv"))
    with actual.open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return mean(values) if values else math.nan


def auc_score(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_holdout_reference() -> tuple[dict[str, dict[str, float]], dict]:
    refs = {}
    with REFERENCE_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["seed"] != "3456":
                continue
            refs[row["path"]] = {
                "hcs": float(row["hcs_rd"]),
                "old": float(row["old_rd"]),
                "min090": float(row["min090_rd"]),
                "previous_local": float(row["previous_local_rd"]),
                "beta005": float(row["variant500_rd"]),
            }
    beta = [r["beta005"] for r in refs.values()]
    previous = [r["previous_local"] for r in refs.values()]
    hcs = [r["hcs"] for r in refs.values()]
    oracle = [min(r["beta005"], r["previous_local"]) for r in refs.values()]
    fallback = [r["previous_local"] < r["beta005"] for r in refs.values()]
    headroom = {
        "rows": len(refs),
        "beta005_mean_rd": safe_mean(beta),
        "previous_local_mean_rd": safe_mean(previous),
        "hcs_mean_rd": safe_mean(hcs),
        "oracle_beta005_previous_local_mean_rd": safe_mean(oracle),
        "previous_local_win_fraction": sum(fallback) / len(fallback),
        "oracle_minus_beta005": safe_mean(oracle) - safe_mean(beta),
        "previous_local_minus_beta005": safe_mean(previous) - safe_mean(beta),
    }
    return refs, headroom


def quartile_deltas(rows: list[dict[str, str]], refs: dict[str, dict[str, float]]) -> dict[str, float]:
    joined = []
    for row in rows:
        ref = refs.get(row["path"])
        if ref is None:
            continue
        joined.append((ref["hcs"], f(row, "rd_score") - ref["beta005"]))
    joined.sort(key=lambda x: x[0])
    out = {}
    for qi in range(4):
        chunk = joined[qi * len(joined) // 4 : (qi + 1) * len(joined) // 4]
        out[f"Q{qi + 1}"] = safe_mean([delta for _, delta in chunk])
    return out


def summarize_variant(variant: dict, refs: dict[str, dict[str, float]], beta_features: dict) -> dict:
    data = read_json(variant["json"])
    summary = data["summaries"][0]
    rows = read_rows(variant["csv"])
    joined = [(row, refs[row["path"]]) for row in rows if row.get("path") in refs]
    deltas = [f(row, "rd_score") - ref["beta005"] for row, ref in joined]
    fallback = [ref["previous_local"] < ref["beta005"] for _, ref in joined]
    fallback_deltas = [delta for delta, is_fallback in zip(deltas, fallback) if is_fallback]
    keep_deltas = [delta for delta, is_fallback in zip(deltas, fallback) if not is_fallback]
    reliability = [f(row, "rvq_householder_reliability_multiplier") for row, _ in joined if row.get("rvq_householder_reliability_multiplier", "") != ""]
    reliability_by_path = {
        row["path"]: f(row, "rvq_householder_reliability_multiplier")
        for row, _ in joined
        if row.get("rvq_householder_reliability_multiplier", "") != ""
    }
    rel_fallback = [reliability_by_path[row["path"]] for row, ref in joined if row["path"] in reliability_by_path and ref["previous_local"] < ref["beta005"]]
    rel_keep = [reliability_by_path[row["path"]] for row, ref in joined if row["path"] in reliability_by_path and not ref["previous_local"] < ref["beta005"]]
    auc = math.nan
    if reliability:
        scores = [-reliability_by_path[row["path"]] for row, _ in joined if row["path"] in reliability_by_path]
        labels = [1 if ref["previous_local"] < ref["beta005"] else 0 for row, ref in joined if row["path"] in reliability_by_path]
        auc = auc_score(scores, labels)

    row = {
        "variant": variant["name"],
        "family": variant["family"],
        "step": variant["step"],
        "rd": summary["mean_rd"],
        "delta_vs_beta005": summary["mean_rd"] - beta_features["rd"],
        "win_fraction_vs_beta005": sum(delta < 0 for delta in deltas) / len(deltas),
        "fallback_delta_vs_beta005": safe_mean(fallback_deltas),
        "keep_delta_vs_beta005": safe_mean(keep_deltas),
        "hcs_difficulty_quartile_delta_vs_beta005": quartile_deltas(rows, refs),
        "nonfinite": summary.get("nonfinite_rows", 0),
        "bpp": summary.get("mean_bpp"),
        "psnr": summary.get("mean_psnr"),
        "ms_ssim": summary.get("mean_ms_ssim"),
        "qMSE": summary.get("mean_rvq_latent_quant_mse"),
        "qMSE_delta_vs_beta005": summary.get("mean_rvq_latent_quant_mse", math.nan) - beta_features["qMSE"],
        "s_q": summary.get("mean_rvq_s_q_mean"),
        "s_q_delta_vs_beta005": summary.get("mean_rvq_s_q_mean", math.nan) - beta_features["s_q"],
        "delta_rms": summary.get("mean_rvq_householder_delta_rms"),
        "delta_rms_delta_vs_beta005": summary.get("mean_rvq_householder_delta_rms", math.nan) - beta_features["delta_rms"],
        "strength": summary.get("mean_rvq_householder_strength"),
        "strength_delta_vs_beta005": summary.get("mean_rvq_householder_strength", math.nan) - beta_features["strength"],
        "dead_code": summary.get("mean_rvq_dead_code_ratio"),
        "dead_code_delta_vs_beta005": summary.get("mean_rvq_dead_code_ratio", math.nan) - beta_features["dead_code"],
        "reliability_mean": safe_mean(reliability),
        "reliability_fallback_mean": safe_mean(rel_fallback),
        "reliability_keep_mean": safe_mean(rel_keep),
        "reliability_gap_fallback_minus_keep": safe_mean(rel_fallback) - safe_mean(rel_keep),
        "auc_low_reliability_for_previous_local_win": auc,
        "patch_size": data.get("patch_size"),
        "csv": str(variant["csv"].relative_to(ROOT)),
        "json": str(variant["json"].relative_to(ROOT)),
        "note": variant["note"],
    }
    if variant["name"] == "beta005_fullimage":
        row["win_fraction_vs_beta005"] = None
    if variant.get("fallback_csv"):
        row["csv"] = str(variant["fallback_csv"].relative_to(ROOT))
    return row


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def write_markdown(output: dict, path: Path) -> None:
    rows = output["controller_rows"]
    lines = [
        "# E083 Full-Image Controller Family Closure",
        "",
        "This audit closes the beta005-preserving controller branch under one paper-facing full-image protocol.",
        f"Protocol: start_index={output['protocol']['start_index']}, max_images={output['protocol']['max_images']}, patch_size=None, physical GPU0/cuda:0.",
        "",
        "## Teacher Headroom",
        "",
        "| split | beta005 RD | previous-local RD | oracle RD | oracle-beta005 | previous-local win |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    h = output["holdout_seed3456_headroom"]
    lines.append(
        "| holdout4096 seed3456 | "
        f"{fmt(h['beta005_mean_rd'])} | {fmt(h['previous_local_mean_rd'])} | "
        f"{fmt(h['oracle_beta005_previous_local_mean_rd'])} | {fmt(h['oracle_minus_beta005'], signed=True)} | "
        f"{fmt(h['previous_local_win_fraction'])} |"
    )
    t = output["transfer8192_teacher_headroom"]
    lines.append(
        "| transfer8192 3-seed | "
        f"{fmt(t['reference_mean_rd'])} | {fmt(t['candidate_mean_rd'])} | "
        f"{fmt(t['oracle_mean_rd'])} | {fmt(t['oracle_minus_reference'], signed=True)} | "
        f"{fmt(t['candidate_win_fraction'])} |"
    )
    lines += [
        "",
        "## Controller Rows",
        "",
        "| variant | RD | delta | win | fallback delta | keep delta | qMSE delta | dead delta | rel gap | AUC | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['rd'])} | {fmt(row['delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['win_fraction_vs_beta005'])} | {fmt(row['fallback_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['keep_delta_vs_beta005'], signed=True)} | {fmt(row['qMSE_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['dead_code_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['reliability_gap_fallback_minus_keep'], signed=True)} | "
            f"{fmt(row['auc_low_reliability_for_previous_local_win'])} | {row['nonfinite']} |"
        )
    lines += [
        "",
        "## HCS-Difficulty Quartile Delta vs Beta005",
        "",
        "| variant | Q1 easy | Q2 | Q3 | Q4 hard |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        q = row["hcs_difficulty_quartile_delta_vs_beta005"]
        lines.append(
            f"| {row['variant']} | {fmt(q['Q1'], signed=True)} | {fmt(q['Q2'], signed=True)} | "
            f"{fmt(q['Q3'], signed=True)} | {fmt(q['Q4'], signed=True)} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        output["decision"],
        "",
        "## Next Action",
        "",
        output["next_action"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    refs, holdout_headroom = load_holdout_reference()
    beta_summary = read_json(BETA_JSON)["summaries"][0]
    beta_features = {
        "rd": beta_summary["mean_rd"],
        "qMSE": beta_summary["mean_rvq_latent_quant_mse"],
        "s_q": beta_summary["mean_rvq_s_q_mean"],
        "delta_rms": beta_summary["mean_rvq_householder_delta_rms"],
        "strength": beta_summary["mean_rvq_householder_strength"],
        "dead_code": beta_summary["mean_rvq_dead_code_ratio"],
    }
    rows = [summarize_variant(variant, refs, beta_features) for variant in VARIANTS]
    transfer = read_json(TRANSFER_TEACHER_JSON)["aggregate"]
    transfer["oracle_minus_reference"] = transfer["oracle_mean_rd"] - transfer["reference_mean_rd"]
    best_controller = min((r for r in rows if r["variant"] != "beta005_fullimage"), key=lambda r: r["rd"])
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Close full-image beta005-preserving reliability-controller experiments and separate teacher headroom from controller usefulness.",
        "protocol": {
            "data_root": "/dpl/openimages/open-images-v6/train/data",
            "start_index": 4096,
            "max_images": 4096,
            "patch_size": None,
            "device": "cuda:0 with CUDA_VISIBLE_DEVICES=0",
            "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
            "reference_column": "variant500_rd",
            "reference_seed": 3456,
        },
        "holdout_seed3456_headroom": holdout_headroom,
        "transfer8192_teacher_headroom": transfer,
        "controller_rows": rows,
        "best_controller_by_rd": best_controller,
        "decision": (
            "All controller variants are numerically stable under full-image evaluation, but none beats beta005. "
            f"The best controller is {best_controller['variant']} at delta {best_controller['delta_vs_beta005']:+.6f} RD. "
            "The teacher/local-cap oracle still has clear headroom, and all trained controllers improve the Q4 hard quartile slightly while degrading easier quartiles. "
            "The current failure mode is therefore not lack of signal or instability; it is insufficiently selective image-mean head-only reliability supervision."
        ),
        "next_action": (
            "Stop plain BCE/margin-weight scaling for this branch. The next serious method-improvement step should preserve the Q4 gain while removing Q1-Q3 damage by using "
            "beta005-preserving selective control with a stronger locality-aware target: map-level or region-level reliability, "
            "ranking/distillation against beta005, and explicit qMSE/dead-code/RVQ-assignment preservation."
        ),
    }
    json_path = ANALYSIS / "e083_fullimage_controller_family_closure.json"
    md_path = ANALYSIS / "e083_fullimage_controller_family_closure.md"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    write_markdown(output, md_path)
    print(json_path)
    print(md_path)
    print(json.dumps({"best_controller": best_controller["variant"], "delta": best_controller["delta_vs_beta005"]}, indent=2))


if __name__ == "__main__":
    main()
