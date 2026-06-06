#!/usr/bin/env python3
"""Audit E089, the first E088 selector distillation checkpoint."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "e089_e088_selector_distill_audit"

BETA = ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"
HCS_OLD = ANALYSIS / "per_image_seed3456_hcs250_vs_hcgh_gate025_step500_val4096_holdout4096_current.csv"
PREVIOUS_LOCAL = ANALYSIS / "direct_local_cap080_rho1_seed3456_step250_val4096_holdout4096_current.csv"
E088_TEACHER = ANALYSIS / "e088_decoder_safe_selector_teacher_labels_transfer8192.json"

VARIANTS = [
    (
        "e080_yhat_step500",
        ANALYSIS
        / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    ),
    (
        "e085_localdelta_step500",
        ANALYSIS
        / "teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    ),
    (
        "e086_keep_target_step500",
        ANALYSIS
        / "teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    ),
    (
        "e087_rdonly_step250",
        ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e087_rdonly_step500",
        ANALYSIS / "rdonly_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    ),
    (
        "e089_e088sel_rho025_step250",
        ANALYSIS / "e088sel_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e089_e088sel_rho025_step500",
        ANALYSIS / "e088sel_rho025_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    ),
    (
        "e090_e088sel_step250_rdpolish_step250",
        ANALYSIS / "e090_e088sel_step250_rdpolish_yhatanchor100_headonly_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e092_e088sel_distmargin_yhat25_step250",
        ANALYSIS / "e092_e088sel_distmargin_yhatanchor25_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e093_e088sel_distmargin_yhat10_condheads_step250",
        ANALYSIS / "e093_e088sel_distmargin_yhatanchor10_condheads_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e094_e088sel_relmin05_rho025_yhat100_step250",
        ANALYSIS / "e094_e088sel_relmin05_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e095_e088sel_relmin05_localtarget_t0451_step250",
        ANALYSIS / "e095_e088sel_relmin05_localtarget_t0451_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e096_e088sel_residualselector_localdelta_t0451_step250",
        ANALYSIS
        / "e096_e088sel_residualselector_localdelta_t0451_rho025_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e097_e088sel_residualselector_localdelta_t0527_rho100_max075_lmin000_step250",
        ANALYSIS
        / "e097_e088sel_residualselector_localdelta_t0527_rho100_max075_lmin000_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e098_e088sel_residualselector_localdelta_t0527_rho100_max075_nooplow_t0451_rho100_step250",
        ANALYSIS
        / "e098_e088sel_residualselector_localdelta_t0527_rho100_max075_nooplow_t0451_rho100_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e099_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_yhat50_step250",
        ANALYSIS
        / "e099_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e100_e099_deadzone012_step250",
        ANALYSIS / "e100_e099_deadzone012_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e101_e099_deadzone010_step250",
        ANALYSIS / "e101_e099_deadzone010_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e102_e099_deadzone014_step250",
        ANALYSIS / "e102_e099_deadzone014_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e103_e099_deadzone016_step250",
        ANALYSIS / "e103_e099_deadzone016_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
    (
        "e104_e099_deadzone018_step250",
        ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv",
    ),
]

FEATURES = [
    "bpp",
    "psnr",
    "ms_ssim",
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_perplexity",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_gate_raw",
    "rvq_householder_strength",
    "rvq_householder_v_abs_mean",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_path(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["path"]: row for row in rows}


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    if key not in row or row[key] == "":
        return default
    value = float(row[key])
    return value if math.isfinite(value) else default


def finite(values) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def pearson(a: list[float], b: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(a, b, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def nonfinite_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if str(row.get("has_nonfinite", "0")).lower() in {"1", "true", "yes"})


def load_aligned() -> dict[str, object]:
    beta = by_path(BETA)
    hcs_old = by_path(HCS_OLD)
    previous = by_path(PREVIOUS_LOCAL)
    variants = {name: by_path(path) for name, path in VARIANTS if path.exists()}
    paths = sorted(set(beta) & set(hcs_old) & set(previous))
    for name, rows in variants.items():
        paths = sorted(set(paths) & set(rows))
    if len(paths) != 4096:
        raise RuntimeError(f"expected 4096 aligned seed3456 rows, got {len(paths)}")
    return {
        "paths": paths,
        "beta": beta,
        "hcs_old": hcs_old,
        "previous": previous,
        "variants": variants,
    }


def score_e088_selector(beta_rows: dict[str, dict[str, str]], paths: list[str]) -> tuple[dict[str, float], dict[str, bool]]:
    teacher = json.loads(E088_TEACHER.read_text(encoding="utf-8"))
    model = teacher["model"]
    features = list(model["features"])
    loc = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    bias = float(model["bias"])
    threshold = float(teacher["threshold"])
    scores: dict[str, float] = {}
    selected: dict[str, bool] = {}
    for path in paths:
        row = beta_rows[path]
        x = np.asarray([f(row, feature) for feature in features], dtype=np.float64)
        z = float(((x - loc) / scale) @ weight + bias)
        score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
        scores[path] = score
        selected[path] = score >= threshold
    return scores, selected


def summarize_mask(
    *,
    paths: list[str],
    mask: dict[str, bool],
    beta: dict[str, dict[str, str]],
    method: dict[str, dict[str, str]],
    label: str,
) -> dict[str, float | str]:
    selected = [path for path in paths if mask[path]]
    if not selected:
        return {"label": label, "rows": 0.0, "fraction": 0.0}
    beta_rd = [f(beta[path], "rd_score") for path in selected]
    method_rd = [f(method[path], "rd_score") for path in selected]
    delta = [m - b for m, b in zip(method_rd, beta_rd, strict=True)]
    mixed_delta = [
        (f(method[path], "rd_score") - f(beta[path], "rd_score")) if mask[path] else 0.0
        for path in paths
    ]
    return {
        "label": label,
        "rows": float(len(selected)),
        "fraction": len(selected) / len(paths),
        "method_rd": mean(method_rd),
        "beta_rd": mean(beta_rd),
        "delta_vs_beta": mean(delta),
        "mixed_apply_selected_delta": mean(mixed_delta),
    }


def hcs_quartile_masks(paths: list[str], hcs_old: dict[str, dict[str, str]]) -> dict[str, dict[str, bool]]:
    ordered = sorted(paths, key=lambda path: f(hcs_old[path], "HCS_rd_score"))
    chunks = np.array_split(np.arange(len(ordered)), 4)
    out = {}
    for i, chunk in enumerate(chunks, start=1):
        members = {ordered[int(index)] for index in chunk}
        out[f"Q{i}"] = {path: path in members for path in paths}
    return out


def summarize_variant(
    name: str,
    rows: dict[str, dict[str, str]],
    *,
    paths: list[str],
    beta: dict[str, dict[str, str]],
    hcs_old: dict[str, dict[str, str]],
    previous: dict[str, dict[str, str]],
    e088_scores: dict[str, float],
    e088_selected: dict[str, bool],
) -> dict[str, object]:
    deltas = [f(rows[path], "rd_score") - f(beta[path], "rd_score") for path in paths]
    base: dict[str, object] = {
        "name": name,
        "rows": len(paths),
        "rd": mean(f(rows[path], "rd_score") for path in paths),
        "delta_vs_beta005": mean(deltas),
        "delta_vs_hcs": mean(f(rows[path], "rd_score") - f(hcs_old[path], "HCS_rd_score") for path in paths),
        "mean_abs_delta_vs_beta005": mean(abs(value) for value in deltas),
        "max_abs_delta_vs_beta005": max(abs(value) for value in deltas),
        "nonfinite_rows": nonfinite_count([rows[path] for path in paths]),
        "previous_local_wins_fraction": mean(
            f(previous[path], "rd_score") < f(beta[path], "rd_score") for path in paths
        ),
    }
    feature_means = {}
    for feature in FEATURES:
        method_mean = mean(f(rows[path], feature) for path in paths)
        beta_mean = mean(f(beta[path], feature) for path in paths)
        feature_means[feature] = {
            "method": method_mean,
            "beta005": beta_mean,
            "delta_vs_beta005": method_mean - beta_mean,
        }
    base["feature_means"] = feature_means

    masks: dict[str, dict[str, bool]] = {
        "e088_decoder_safe_selected": e088_selected,
        "beta_delta_rms_ge_045151": {
            path: f(beta[path], "rvq_householder_delta_rms") >= 0.045151 for path in paths
        },
        "beta_delta_rms_ge_052714": {
            path: f(beta[path], "rvq_householder_delta_rms") >= 0.052714 for path in paths
        },
        "previous_local_wins": {
            path: f(previous[path], "rd_score") < f(beta[path], "rd_score") for path in paths
        },
    }
    masks.update(hcs_quartile_masks(paths, hcs_old))
    base["subsets"] = [
        summarize_mask(paths=paths, mask=mask, beta=beta, method=rows, label=label)
        for label, mask in masks.items()
    ]

    if "rvq_householder_reliability_multiplier" in next(iter(rows.values())):
        rel = [f(rows[path], "rvq_householder_reliability_multiplier") for path in paths]
        selected_rel = [f(rows[path], "rvq_householder_reliability_multiplier") for path in paths if e088_selected[path]]
        unselected_rel = [f(rows[path], "rvq_householder_reliability_multiplier") for path in paths if not e088_selected[path]]
        base["reliability_alignment"] = {
            "mean": mean(rel),
            "selected_mean": mean(selected_rel),
            "unselected_mean": mean(unselected_rel),
            "selected_minus_unselected": mean(selected_rel) - mean(unselected_rel),
            "corr_e088_score_with_reliability_suppression": pearson(
                [e088_scores[path] for path in paths],
                [1.0 - value for value in rel],
            ),
            "min": mean(f(rows[path], "rvq_householder_reliability_multiplier_min") for path in paths),
            "max": mean(f(rows[path], "rvq_householder_reliability_multiplier_max") for path in paths),
            "std": mean(f(rows[path], "rvq_householder_reliability_multiplier_std") for path in paths),
        }
    return base


def summarize_baselines(
    paths: list[str],
    beta: dict[str, dict[str, str]],
    hcs_old: dict[str, dict[str, str]],
    previous: dict[str, dict[str, str]],
    e088_selected: dict[str, bool],
) -> dict[str, float]:
    beta_rd = [f(beta[path], "rd_score") for path in paths]
    hcs_rd = [f(hcs_old[path], "HCS_rd_score") for path in paths]
    previous_rd = [f(previous[path], "rd_score") for path in paths]
    e088_rd = [
        f(previous[path], "rd_score") if e088_selected[path] else f(beta[path], "rd_score")
        for path in paths
    ]
    oracle_rd = [min(f(beta[path], "rd_score"), f(previous[path], "rd_score")) for path in paths]
    return {
        "hcs": mean(hcs_rd),
        "beta005": mean(beta_rd),
        "previous_local": mean(previous_rd),
        "previous_local_delta_vs_beta005": mean(p - b for p, b in zip(previous_rd, beta_rd, strict=True)),
        "previous_local_win_fraction": mean(p < b for p, b in zip(previous_rd, beta_rd, strict=True)),
        "e088_decoder_safe_selector": mean(e088_rd),
        "e088_decoder_safe_selector_delta_vs_beta005": mean(e - b for e, b in zip(e088_rd, beta_rd, strict=True)),
        "e088_selected_fraction": mean(e088_selected[path] for path in paths),
        "oracle": mean(oracle_rd),
        "oracle_delta_vs_beta005": mean(o - b for o, b in zip(oracle_rd, beta_rd, strict=True)),
    }


def write_csv(result: dict[str, object]) -> None:
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "rd",
                "delta_vs_beta005",
                "delta_vs_hcs",
                "mean_abs_delta_vs_beta005",
                "nonfinite_rows",
                "latent_qmse",
                "s_q_mean",
                "dead_code",
                "perplexity",
                "delta_rms",
                "strength",
                "reliability_mean",
                "reliability_selected_minus_unselected",
                "corr_score_with_suppression",
                "e088_selected_delta_vs_beta",
                "e088_selected_mixed_delta",
                "q4_delta_vs_beta",
            ],
        )
        writer.writeheader()
        for item in result["variants"]:
            subsets = {row["label"]: row for row in item["subsets"]}
            features = item["feature_means"]
            rel = item.get("reliability_alignment", {})
            writer.writerow(
                {
                    "name": item["name"],
                    "rd": item["rd"],
                    "delta_vs_beta005": item["delta_vs_beta005"],
                    "delta_vs_hcs": item["delta_vs_hcs"],
                    "mean_abs_delta_vs_beta005": item["mean_abs_delta_vs_beta005"],
                    "nonfinite_rows": item["nonfinite_rows"],
                    "latent_qmse": features["rvq_latent_quant_mse"]["method"],
                    "s_q_mean": features["rvq_s_q_mean"]["method"],
                    "dead_code": features["rvq_dead_code_ratio"]["method"],
                    "perplexity": features["rvq_perplexity"]["method"],
                    "delta_rms": features["rvq_householder_delta_rms"]["method"],
                    "strength": features["rvq_householder_strength"]["method"],
                    "reliability_mean": rel.get("mean", float("nan")),
                    "reliability_selected_minus_unselected": rel.get("selected_minus_unselected", float("nan")),
                    "corr_score_with_suppression": rel.get("corr_e088_score_with_reliability_suppression", float("nan")),
                    "e088_selected_delta_vs_beta": subsets["e088_decoder_safe_selected"].get("delta_vs_beta", float("nan")),
                    "e088_selected_mixed_delta": subsets["e088_decoder_safe_selected"].get("mixed_apply_selected_delta", float("nan")),
                    "q4_delta_vs_beta": subsets["Q4"].get("delta_vs_beta", float("nan")),
                }
            )


def write_markdown(result: dict[str, object]) -> None:
    base = result["baselines"]
    variants = result["variants"]
    lines = [
        "# E089 E088-Selector Distillation Audit",
        "",
        "This audit checks whether the E088 transfer-learned decoder-safe selector can be distilled into one beta005-initialized checkpoint. It uses seed3456 full-image holdout4096 rows, exact inverse mode, and compares checkpoint means, E088-selected subsets, HCS difficulty quartiles, and intermediate feature preservation.",
        "",
        "## Baselines",
        "",
        "| row | RD | vs beta005 | selected/win frac |",
        "|---|---:|---:|---:|",
        f"| HCS | {fmt(base['hcs'])} | {fmt(base['hcs'] - base['beta005'], signed=True)} | - |",
        f"| beta005 | {fmt(base['beta005'])} | {fmt(0.0, signed=True)} | - |",
        f"| previous-local | {fmt(base['previous_local'])} | {fmt(base['previous_local_delta_vs_beta005'], signed=True)} | {fmt(base['previous_local_win_fraction'])} |",
        f"| E088 decoder-safe switch | {fmt(base['e088_decoder_safe_selector'])} | {fmt(base['e088_decoder_safe_selector_delta_vs_beta005'], signed=True)} | {fmt(base['e088_selected_fraction'])} |",
        f"| oracle min(beta005, previous-local) | {fmt(base['oracle'])} | {fmt(base['oracle_delta_vs_beta005'], signed=True)} | - |",
        "",
        "## Checkpoint Summary",
        "",
        "| variant | RD | vs beta005 | vs HCS | mean abs vs beta | nonfinite | qMSE | s_q | dead | delta RMS | strength | rel mean | rel selected-unselected | score/suppression corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in variants:
        features = item["feature_means"]
        rel = item.get("reliability_alignment", {})
        lines.append(
            f"| {item['name']} | {fmt(item['rd'])} | {fmt(item['delta_vs_beta005'], signed=True)} | "
            f"{fmt(item['delta_vs_hcs'], signed=True)} | {fmt(item['mean_abs_delta_vs_beta005'])} | "
            f"{int(item['nonfinite_rows'])} | {fmt(features['rvq_latent_quant_mse']['method'])} | "
            f"{fmt(features['rvq_s_q_mean']['method'])} | {fmt(features['rvq_dead_code_ratio']['method'])} | "
            f"{fmt(features['rvq_householder_delta_rms']['method'])} | {fmt(features['rvq_householder_strength']['method'])} | "
            f"{fmt(float(rel.get('mean', float('nan'))))} | {fmt(float(rel.get('selected_minus_unselected', float('nan'))), signed=True)} | "
            f"{fmt(float(rel.get('corr_e088_score_with_reliability_suppression', float('nan'))), signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## E088-Selected Subset",
            "",
            "| variant | selected frac | selected vs beta | mixed apply-selected delta | Q4 vs beta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in variants:
        subsets = {row["label"]: row for row in item["subsets"]}
        selected = subsets["e088_decoder_safe_selected"]
        q4 = subsets["Q4"]
        lines.append(
            f"| {item['name']} | {fmt(selected['fraction'])} | {fmt(selected.get('delta_vs_beta', float('nan')), signed=True)} | "
            f"{fmt(selected.get('mixed_apply_selected_delta', float('nan')), signed=True)} | {fmt(q4.get('delta_vs_beta', float('nan')), signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## E089/E092/E093/E094/E095/E096/E097/E098/E099/E100-E104 Quartiles",
            "",
            "| variant | Q1 vs beta | Q2 vs beta | Q3 vs beta | Q4 vs beta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in variants:
        if not (
            str(item["name"]).startswith("e089")
            or str(item["name"]).startswith("e090")
            or str(item["name"]).startswith("e092")
            or str(item["name"]).startswith("e093")
            or str(item["name"]).startswith("e094")
            or str(item["name"]).startswith("e095")
            or str(item["name"]).startswith("e096")
            or str(item["name"]).startswith("e097")
            or str(item["name"]).startswith("e098")
            or str(item["name"]).startswith("e099")
            or str(item["name"]).startswith("e100")
            or str(item["name"]).startswith("e101")
            or str(item["name"]).startswith("e102")
            or str(item["name"]).startswith("e103")
            or str(item["name"]).startswith("e104")
        ):
            continue
        subsets = {row["label"]: row for row in item["subsets"]}
        lines.append(
            f"| {item['name']} | {fmt(subsets['Q1'].get('delta_vs_beta', float('nan')), signed=True)} | "
            f"{fmt(subsets['Q2'].get('delta_vs_beta', float('nan')), signed=True)} | "
            f"{fmt(subsets['Q3'].get('delta_vs_beta', float('nan')), signed=True)} | "
            f"{fmt(subsets['Q4'].get('delta_vs_beta', float('nan')), signed=True)} |"
        )

    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision,
            "",
            f"JSON: `{OUT.with_suffix('.json').relative_to(ROOT)}`",
            f"CSV: `{OUT.with_suffix('.csv').relative_to(ROOT)}`",
            "",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    aligned = load_aligned()
    paths = aligned["paths"]
    beta = aligned["beta"]
    hcs_old = aligned["hcs_old"]
    previous = aligned["previous"]
    variants = aligned["variants"]
    e088_scores, e088_selected = score_e088_selector(beta, paths)

    result: dict[str, object] = {
        "rows": len(paths),
        "inputs": {
            "beta005": str(BETA.relative_to(ROOT)),
            "hcs_old": str(HCS_OLD.relative_to(ROOT)),
            "previous_local": str(PREVIOUS_LOCAL.relative_to(ROOT)),
            "e088_teacher": str(E088_TEACHER.relative_to(ROOT)),
        },
        "baselines": summarize_baselines(paths, beta, hcs_old, previous, e088_selected),
        "variants": [
            summarize_variant(
                name,
                rows,
                paths=paths,
                beta=beta,
                hcs_old=hcs_old,
                previous=previous,
                e088_scores=e088_scores,
                e088_selected=e088_selected,
            )
            for name, rows in variants.items()
        ],
    }
    result["decision"] = (
        "E089/E099 close the first single-checkpoint selector-distillation family: the runs are numerically stable and preserve beta005-like intermediate features, but they do not recover enough of the E088 transfer-selector headroom to beat beta005. "
        "E096 is the best pre-dead-zone exact-default residual-selector row before E099, with RD +0.000271 vs beta005, E088-selected images -0.000874, and Q4 -0.000486, while Q1-Q3 damage leaves the mean worse. "
        "E099 reduces the mean damage to +0.000267 vs beta005 by using selected-image distortion-margin supervision, but it weakens the selected/Q4 gain slightly and still does not beat beta005. "
        "E100-E104 test a deploy-time dead-zone on the same E099 checkpoint, leaving low-confidence selector probabilities as exact no-op. This directly targets the precision failure. The holdout sweep is monotonic over the tested range: deadzone010 gives +0.000126 vs beta005, deadzone012 gives -0.000062, deadzone014 gives -0.000253, deadzone016 gives -0.000373, and deadzone018 gives -0.000423. "
        "E104 is therefore the best current single-checkpoint diagnostic on seed3456 holdout4096, with RD 2.155993, mean abs delta 0.000802, Q4/selected behavior audited below, beta005-like qMSE/s_q/dead-code, and nonfinite_rows=0. The same threshold also improves the start8192 transfer slice by -0.000425 vs the seed3456 beta005 reference, close to the holdout improvement and slightly stronger than deadzone016 on that split. "
        "The interpretation is promising but must stay disciplined: the dead-zone threshold family was explored through diagnostic sweeps, so E104 is not yet a paper-main row. The next paper-safe step is to lock a calibration rule or calibration split for the dead-zone threshold, then rerun the chosen threshold path-matched across seeds and checkpoints. If that holds, the exact-default residual selector plus calibrated dead-zone becomes a stronger HCG-RVQ variant; until then beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ row, and E100-E104 are evidence that the remaining bottleneck is policy calibration/precision rather than instability, VQ collapse, or lack of local geometry signal."
    )


    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(result)
    write_markdown(result)
    print(json.dumps({"rows": result["rows"], "out": str(OUT.with_suffix(".md").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
