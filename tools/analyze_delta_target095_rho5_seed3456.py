import json
import math
from pathlib import Path

import analyze_delta_reg010_seed3456 as base


ANALYSIS_DIR = Path("experiments/analysis")
OUTPUT_PREFIX = "delta_target095_rho5_seed3456_val4096_holdout4096_current"


base.METHODS = {
    "old gate0.25": {
        "kind": "trusted",
        "rd_field": "old_rd",
        "feature_file": "old_features",
    },
    "trained min090 risk": {
        "kind": "trusted",
        "rd_field": "min090_rd",
        "feature_file": "risk_features",
    },
    "trained min095 risk": {
        "kind": "feature",
        "feature_path": ANALYSIS_DIR
        / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min095_seed3456_step500_val4096_holdout4096_current.csv",
    },
    "delta_reg010": {
        "kind": "feature",
        "feature_path": ANALYSIS_DIR
        / "per_image_features_hcg_h_gate025_delta_reg010_seed3456_step500_val4096_holdout4096_current.csv",
    },
    "delta_target095_rho5": {
        "kind": "feature",
        "feature_path": ANALYSIS_DIR
        / "per_image_features_hcg_h_gate025_delta_target095_rho5_seed3456_step500_val4096_holdout4096_current.csv",
    },
}


def target_quartiles(rows):
    rows = sorted(rows, key=lambda row: row["hcs_rd"])
    chunks = []
    n = len(rows)
    for idx in range(4):
        lo = idx * n // 4
        hi = (idx + 1) * n // 4
        chunk = rows[lo:hi]
        row = {
            "quartile": idx + 1,
            "n": len(chunk),
            "hcs_rd": base.mean(item["hcs_rd"] for item in chunk),
        }
        for method in base.METHODS:
            row[f"{method}_delta_vs_hcs"] = base.mean(
                item[f"{method}_delta_vs_hcs"] for item in chunk
            )
        for method in ("delta_reg010", "delta_target095_rho5"):
            for key in (
                "s_q_mean",
                "householder_strength_mean",
                "householder_delta_rms",
                "rvq_latent_quant_mse",
                "index_empirical_bpp",
                "rvq_dead_code_ratio",
            ):
                row[f"{method}_{key}"] = base.mean(item[f"{method}_{key}"] for item in chunk)
        return_row = row
        chunks.append(return_row)
    return chunks


def main():
    rows = base.joined_rows()
    summaries = {method: base.summarize_method(rows, method) for method in base.METHODS}
    hcs_summary = {"n": len(rows), "rd": base.mean(row["hcs_rd"] for row in rows)}
    qs = target_quartiles(rows)
    correlations = {
        key: base.corr(
            [row["delta_target095_rho5_delta_vs_hcs"] for row in rows],
            [row[f"delta_target095_rho5_{key}"] for row in rows],
        )
        for key in (
            "s_q_mean",
            "householder_strength_mean",
            "householder_delta_rms",
            "rvq_latent_quant_mse",
            "index_empirical_bpp",
            "rvq_dead_code_ratio",
        )
    }
    summary = {
        "hcs": hcs_summary,
        "methods": summaries,
        "hcs_difficulty_quartiles": qs,
        "delta_target095_rho5_correlations_with_delta_vs_hcs": correlations,
        "artifacts": {
            "config": "configs/pilot_hcg_rvq_h_gate025_delta_target095_rho5_frozen_seed3456.yaml",
            "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_delta_target095_rho5_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar",
            "features": str(base.METHODS["delta_target095_rho5"]["feature_path"]),
        },
    }

    json_path = ANALYSIS_DIR / f"{OUTPUT_PREFIX}.json"
    json_path.write_text(json.dumps(base.json_safe(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Delta-target095-rho5 Householder geometry audit",
        "",
        "This audit evaluates seed3456 with target regularization `rho_householder_delta_target=5.0` and `householder_delta_target=0.095`. RD values are path-matched against the trusted current-holdout HCS/old gate0.25/min090 protocol.",
        "",
        "## Overall RD",
        "",
        "| method | mean RD | delta vs HCS | delta vs old | delta vs min090 | win vs HCS |",
        "|---|---:|---:|---:|---:|---:|",
        f"| HCS | {base.fmt(hcs_summary['rd'])} | {base.fmt(0.0, True)} | n/a | n/a | n/a |",
    ]
    for method, stats in summaries.items():
        lines.append(
            f"| {method} | {base.fmt(stats['rd'])} | {base.fmt(stats['delta_vs_hcs'], True)} | "
            f"{base.fmt(stats['delta_vs_old'], True)} | {base.fmt(stats['delta_vs_min090'], True)} | "
            f"{base.fmt(stats['win_rate_vs_hcs'])} |"
        )

    lines.extend(
        [
            "",
            "## Intermediate Features",
            "",
            "| method | s_q | raw gate | risk mult | strength | delta RMS | latent qMSE | index bpp | dead code |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, stats in summaries.items():
        lines.append(
            f"| {method} | {base.fmt(stats['s_q_mean'])} | "
            f"{base.fmt(stats['householder_gate_raw_mean'])} | "
            f"{base.fmt(stats['householder_risk_multiplier_mean'])} | "
            f"{base.fmt(stats['householder_strength_mean'])} | "
            f"{base.fmt(stats['householder_delta_rms'])} | "
            f"{base.fmt(stats['rvq_latent_quant_mse'])} | "
            f"{base.fmt(stats['index_empirical_bpp'])} | "
            f"{base.fmt(stats['rvq_dead_code_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## HCS-Difficulty Quartiles",
            "",
            "| Q | HCS RD | old-HCS | min090-HCS | min095-HCS | delta010-HCS | target095-HCS | target s_q | target strength | target delta RMS | target qMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in qs:
        lines.append(
            f"| {row['quartile']} | {base.fmt(row['hcs_rd'])} | "
            f"{base.fmt(row['old gate0.25_delta_vs_hcs'], True)} | "
            f"{base.fmt(row['trained min090 risk_delta_vs_hcs'], True)} | "
            f"{base.fmt(row['trained min095 risk_delta_vs_hcs'], True)} | "
            f"{base.fmt(row['delta_reg010_delta_vs_hcs'], True)} | "
            f"{base.fmt(row['delta_target095_rho5_delta_vs_hcs'], True)} | "
            f"{base.fmt(row['delta_target095_rho5_s_q_mean'])} | "
            f"{base.fmt(row['delta_target095_rho5_householder_strength_mean'])} | "
            f"{base.fmt(row['delta_target095_rho5_householder_delta_rms'])} | "
            f"{base.fmt(row['delta_target095_rho5_rvq_latent_quant_mse'])} |"
        )

    lines.extend(
        [
            "",
            "## Target Correlations",
            "",
            "| feature | corr with target095-HCS |",
            "|---|---:|",
        ]
    )
    for key, value in correlations.items():
        lines.append(f"| {key} | {base.fmt(value, True)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The target loss succeeded at matching the intended Householder displacement scale: delta RMS is `0.092294`, close to the old/min090 regime.",
            "- Matching delta RMS is not sufficient. RD worsens to `3.332857`, and latent quantization MSE rises to `0.293593` with a dead-code ratio of `0.261705`.",
            "- The model appears to satisfy the geometry-scale target while moving `s_q`, codebook use, and latent quantization into a worse regime.",
            "- The next controller should anchor the full conditioning/quantization state, not only the Householder displacement magnitude.",
        ]
    )

    md_path = ANALYSIS_DIR / f"{OUTPUT_PREFIX}.md"
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
