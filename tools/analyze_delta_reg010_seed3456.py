import csv
import json
import math
from pathlib import Path

from analyze_gate_selector import current_holdout_paths, load_rows


SEED = 3456
ANALYSIS_DIR = Path("experiments/analysis")
OUTPUT_PREFIX = "delta_reg010_seed3456_val4096_holdout4096_current"


METHODS = {
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
}


FEATURE_KEYS = [
    "bpp",
    "bpp_y",
    "bpp_z",
    "psnr",
    "ms_ssim",
    "commit_loss",
    "s_q_mean",
    "householder_gate_raw_mean",
    "householder_risk_multiplier_mean",
    "householder_strength_mean",
    "householder_delta_rms",
    "rvq_latent_quant_mse",
    "y_error_rms",
    "index_empirical_bpp",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_float(row, key):
    raw = row.get(key, "")
    if raw == "":
        return float("nan")
    return float(raw)


def mean(values):
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return float("nan")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(
        x_var * y_var
    )


def fmt(value, signed=False):
    if not math.isfinite(value):
        return "n/a"
    return ("{:+.6f}" if signed else "{:.6f}").format(value)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_method(rows, method):
    rd_key = f"{method}_rd"
    delta_key = f"{method}_delta_vs_hcs"
    old_key = "old gate0.25_rd"
    min090_key = "trained min090 risk_rd"
    summary = {
        "n": len(rows),
        "rd": mean(row[rd_key] for row in rows),
        "delta_vs_hcs": mean(row[delta_key] for row in rows),
        "delta_vs_old": mean(row[rd_key] - row[old_key] for row in rows),
        "delta_vs_min090": mean(row[rd_key] - row[min090_key] for row in rows),
        "win_rate_vs_hcs": mean(1.0 if row[delta_key] < 0.0 else 0.0 for row in rows),
        "win_rate_vs_old": mean(1.0 if row[rd_key] < row[old_key] else 0.0 for row in rows),
        "win_rate_vs_min090": mean(
            1.0 if row[rd_key] < row[min090_key] else 0.0 for row in rows
        ),
    }
    for key in FEATURE_KEYS:
        summary[key] = mean(row[f"{method}_{key}"] for row in rows)
    return summary


def quartiles(rows):
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
            "hcs_rd": mean(item["hcs_rd"] for item in chunk),
        }
        for method in METHODS:
            row[f"{method}_delta_vs_hcs"] = mean(
                item[f"{method}_delta_vs_hcs"] for item in chunk
            )
        for key in (
            "s_q_mean",
            "householder_strength_mean",
            "householder_delta_rms",
            "rvq_latent_quant_mse",
            "index_empirical_bpp",
            "rvq_dead_code_ratio",
        ):
            row[f"delta_reg010_{key}"] = mean(
                item[f"delta_reg010_{key}"] for item in chunk
            )
        chunks.append(row)
    return chunks


def load_feature_maps():
    paths = current_holdout_paths(ANALYSIS_DIR, SEED)
    maps = {}
    maps["old_features"] = {
        row["path"]: row for row in read_csv(paths["old_features"])
    }
    maps["risk_features"] = {
        row["path"]: row for row in read_csv(paths["risk_features"])
    }
    for method, spec in METHODS.items():
        if spec["kind"] == "feature":
            maps[method] = {row["path"]: row for row in read_csv(spec["feature_path"])}
    return maps


def joined_rows():
    trusted = [row for row in load_rows(ANALYSIS_DIR, "current_holdout") if row["seed"] == SEED]
    feature_maps = load_feature_maps()
    rows = []
    for row in trusted:
        path = row["path"]
        joined = {
            "seed": SEED,
            "index": row["index"],
            "path": path,
            "hcs_rd": row["hcs_rd"],
            "old gate0.25_rd": row["hcs_rd"] + row["old_delta_rd"],
            "trained min090 risk_rd": row["hcs_rd"] + row["risk_delta_rd"],
        }
        joined["old gate0.25_delta_vs_hcs"] = joined["old gate0.25_rd"] - row["hcs_rd"]
        joined["trained min090 risk_delta_vs_hcs"] = (
            joined["trained min090 risk_rd"] - row["hcs_rd"]
        )

        for method, spec in METHODS.items():
            if spec["kind"] == "trusted":
                feature = feature_maps[spec["feature_file"]][path]
            else:
                feature = feature_maps[method][path]
                joined[f"{method}_rd"] = read_float(feature, "rd_score")
                joined[f"{method}_delta_vs_hcs"] = joined[f"{method}_rd"] - row["hcs_rd"]
            for key in FEATURE_KEYS:
                joined[f"{method}_{key}"] = read_float(feature, key)
        rows.append(joined)
    return rows


def main():
    rows = joined_rows()
    summaries = {method: summarize_method(rows, method) for method in METHODS}
    hcs_summary = {"n": len(rows), "rd": mean(row["hcs_rd"] for row in rows)}
    qs = quartiles(rows)
    correlations = {
        key: corr(
            [row["delta_reg010_delta_vs_hcs"] for row in rows],
            [row[f"delta_reg010_{key}"] for row in rows],
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
        "delta_reg010_correlations_with_delta_vs_hcs": correlations,
        "artifacts": {
            "config": "configs/pilot_hcg_rvq_h_gate025_delta_reg010_frozen_seed3456.yaml",
            "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_delta_reg010_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar",
            "delta_features": str(METHODS["delta_reg010"]["feature_path"]),
            "min095_features": str(METHODS["trained min095 risk"]["feature_path"]),
        },
    }

    json_path = ANALYSIS_DIR / f"{OUTPUT_PREFIX}.json"
    json_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Delta-reg010 Householder geometry audit",
        "",
        "This audit evaluates seed3456 with direct `rho_householder_delta=0.10` regularization. RD values are path-matched against the trusted current-holdout HCS/old gate0.25/min090 protocol.",
        "",
        "## Overall RD",
        "",
        "| method | mean RD | delta vs HCS | delta vs old | delta vs min090 | win vs HCS |",
        "|---|---:|---:|---:|---:|---:|",
        f"| HCS | {fmt(hcs_summary['rd'])} | {fmt(0.0, True)} | n/a | n/a | n/a |",
    ]
    for method, stats in summaries.items():
        lines.append(
            f"| {method} | {fmt(stats['rd'])} | {fmt(stats['delta_vs_hcs'], True)} | "
            f"{fmt(stats['delta_vs_old'], True)} | {fmt(stats['delta_vs_min090'], True)} | "
            f"{fmt(stats['win_rate_vs_hcs'])} |"
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
            f"| {method} | {fmt(stats['s_q_mean'])} | "
            f"{fmt(stats['householder_gate_raw_mean'])} | "
            f"{fmt(stats['householder_risk_multiplier_mean'])} | "
            f"{fmt(stats['householder_strength_mean'])} | "
            f"{fmt(stats['householder_delta_rms'])} | "
            f"{fmt(stats['rvq_latent_quant_mse'])} | "
            f"{fmt(stats['index_empirical_bpp'])} | "
            f"{fmt(stats['rvq_dead_code_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## HCS-Difficulty Quartiles",
            "",
            "| Q | HCS RD | old-HCS | min090-HCS | min095-HCS | delta_reg010-HCS | delta s_q | delta strength | delta RMS | delta qMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in qs:
        lines.append(
            f"| {row['quartile']} | {fmt(row['hcs_rd'])} | "
            f"{fmt(row['old gate0.25_delta_vs_hcs'], True)} | "
            f"{fmt(row['trained min090 risk_delta_vs_hcs'], True)} | "
            f"{fmt(row['trained min095 risk_delta_vs_hcs'], True)} | "
            f"{fmt(row['delta_reg010_delta_vs_hcs'], True)} | "
            f"{fmt(row['delta_reg010_s_q_mean'])} | "
            f"{fmt(row['delta_reg010_householder_strength_mean'])} | "
            f"{fmt(row['delta_reg010_householder_delta_rms'])} | "
            f"{fmt(row['delta_reg010_rvq_latent_quant_mse'])} |"
        )

    lines.extend(
        [
            "",
            "## Delta-Reg010 Correlations",
            "",
            "| feature | corr with delta_reg010-HCS |",
            "|---|---:|",
        ]
    )
    for key, value in correlations.items():
        lines.append(f"| {key} | {fmt(value, True)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Direct `rho_householder_delta=0.10` works mechanically: it reduces Householder delta RMS far below old gate0.25/min090.",
            "- It is not a useful controller at this strength. RD becomes worse than HCS, old gate0.25, min090, and even the failed min095 risk floor.",
            "- The failure is consistent with over-constraining the geometry: lower delta/strength is accompanied by larger latent quantization MSE and worse reconstruction RD.",
            "- Therefore the next controller should not shrink geometry toward zero. It should constrain geometry toward a validated regime, or use a much weaker/staged target combined with checkpoint and feature-distribution monitoring.",
        ]
    )

    md_path = ANALYSIS_DIR / f"{OUTPUT_PREFIX}.md"
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
