import csv
import json
import math
from pathlib import Path

from analyze_gate_selector import SEEDS, load_rows


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_float(row, key):
    raw = row.get(key, "")
    return float(raw) if raw != "" else float("nan")


def mean(values):
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value, signed=False):
    if not math.isfinite(value):
        return "nan"
    return ("{:+.6f}" if signed else "{:.6f}").format(value)


def feature_path(analysis_dir, seed):
    return (
        analysis_dir
        / f"per_image_features_hcg_h_gate025_risk_inv_detach_s044_min095_seed{seed}_step500_val4096_holdout4096_current.csv"
    )


def summarize(rows, method_key):
    return {
        "n": len(rows),
        "hcs_rd": mean(row["hcs_rd"] for row in rows),
        "old_gate025_rd": mean(row["old_rd"] for row in rows),
        "trained_min090_rd": mean(row["trained_min090_rd"] for row in rows),
        "trained_min095_rd": mean(row[method_key] for row in rows),
        "old_gate025_delta_vs_hcs": mean(row["old_rd"] - row["hcs_rd"] for row in rows),
        "trained_min090_delta_vs_hcs": mean(
            row["trained_min090_rd"] - row["hcs_rd"] for row in rows
        ),
        "trained_min095_delta_vs_hcs": mean(row[method_key] - row["hcs_rd"] for row in rows),
        "min095_minus_old_gate025": mean(row[method_key] - row["old_rd"] for row in rows),
        "min095_minus_min090": mean(row[method_key] - row["trained_min090_rd"] for row in rows),
        "min095_better_than_hcs_rate": mean(
            1.0 if row[method_key] < row["hcs_rd"] else 0.0 for row in rows
        ),
        "min095_better_than_old_rate": mean(
            1.0 if row[method_key] < row["old_rd"] else 0.0 for row in rows
        ),
        "s_q_mean": mean(row["s_q_mean"] for row in rows),
        "raw_gate_mean": mean(row["raw_gate_mean"] for row in rows),
        "risk_multiplier_mean": mean(row["risk_multiplier_mean"] for row in rows),
        "effective_strength_mean": mean(row["strength_mean"] for row in rows),
        "householder_delta_rms": mean(row["delta_rms"] for row in rows),
        "rvq_latent_quant_mse": mean(row["latent_quant_mse"] for row in rows),
        "index_empirical_bpp": mean(row["index_empirical_bpp"] for row in rows),
        "rvq_perplexity": mean(row["rvq_perplexity"] for row in rows),
        "rvq_dead_code_ratio": mean(row["rvq_dead_code_ratio"] for row in rows),
    }


def quartiles_for_seed(rows, method_key):
    rows = sorted(rows, key=lambda row: row["hcs_rd"])
    chunks = []
    n = len(rows)
    for idx in range(4):
        lo = idx * n // 4
        hi = (idx + 1) * n // 4
        chunk = rows[lo:hi]
        chunks.append(
            {
                "quartile": idx + 1,
                "n": len(chunk),
                "hcs_rd": mean(row["hcs_rd"] for row in chunk),
                "old_delta": mean(row["old_rd"] - row["hcs_rd"] for row in chunk),
                "trained_min090_delta": mean(
                    row["trained_min090_rd"] - row["hcs_rd"] for row in chunk
                ),
                "trained_min095_delta": mean(row[method_key] - row["hcs_rd"] for row in chunk),
                "s_q_mean": mean(row["s_q_mean"] for row in chunk),
                "risk_multiplier_mean": mean(row["risk_multiplier_mean"] for row in chunk),
                "effective_strength_mean": mean(row["strength_mean"] for row in chunk),
                "latent_quant_mse": mean(row["latent_quant_mse"] for row in chunk),
            }
        )
    return chunks


def main():
    analysis_dir = Path("experiments/analysis")
    method_key = "trained_min095_rd"
    trusted_rows = load_rows(analysis_dir, "current_holdout")
    trusted_by_seed = {
        seed: [row for row in trusted_rows if row["seed"] == seed] for seed in SEEDS
    }
    joined_by_seed = {}

    for seed in SEEDS:
        trusted_by_path = {row["path"]: row for row in trusted_by_seed[seed]}
        min095_rows = read_csv(feature_path(analysis_dir, seed))
        joined = []
        for min095 in min095_rows:
            trusted = trusted_by_path[min095["path"]]
            hcs_rd = trusted["hcs_rd"]
            old_rd = hcs_rd + trusted["old_delta_rd"]
            min090_rd = hcs_rd + trusted["risk_delta_rd"]
            joined.append(
                {
                    "seed": seed,
                    "path": min095["path"],
                    "hcs_rd": hcs_rd,
                    "old_rd": old_rd,
                    "trained_min090_rd": min090_rd,
                    method_key: read_float(min095, "rd_score"),
                    "s_q_mean": read_float(min095, "s_q_mean"),
                    "raw_gate_mean": read_float(min095, "householder_gate_raw_mean"),
                    "risk_multiplier_mean": read_float(
                        min095, "householder_risk_multiplier_mean"
                    ),
                    "strength_mean": read_float(min095, "householder_strength_mean"),
                    "delta_rms": read_float(min095, "householder_delta_rms"),
                    "latent_quant_mse": read_float(min095, "rvq_latent_quant_mse"),
                    "index_empirical_bpp": read_float(min095, "index_empirical_bpp"),
                    "rvq_perplexity": read_float(min095, "rvq_perplexity"),
                    "rvq_dead_code_ratio": read_float(min095, "rvq_dead_code_ratio"),
                }
            )
        joined_by_seed[str(seed)] = joined

    all_joined = [row for rows in joined_by_seed.values() for row in rows]
    per_seed = {seed: summarize(rows, method_key) for seed, rows in joined_by_seed.items()}
    overall = summarize(all_joined, method_key)
    quartiles = {
        seed: quartiles_for_seed(rows, method_key) for seed, rows in joined_by_seed.items()
    }

    summary = {
        "overall": overall,
        "per_seed": per_seed,
        "hcs_difficulty_quartiles": quartiles,
        "artifacts": {
            "min095_feature_csvs": [str(feature_path(analysis_dir, seed)) for seed in SEEDS],
            "trusted_selector_protocol": "experiments/analysis/gate025_min090_selector_reporting_protocol.json",
        },
    }

    json_path = analysis_dir / "risk_floor_min095_val4096_holdout4096_current.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Risk floor min095 single-checkpoint audit",
        "",
        "This audit evaluates the trained inverse/detached risk controller with `householder_gate_risk_min=0.95`. RD values are matched by image path against the trusted current-holdout HCS/old gate0.25/min090 artifacts.",
        "",
        "## Overall",
        "",
        "| method | mean RD | delta vs HCS |",
        "|---|---:|---:|",
        f"| HCS | {fmt(overall['hcs_rd'])} | {fmt(0.0, True)} |",
        f"| old gate0.25 | {fmt(overall['old_gate025_rd'])} | {fmt(overall['old_gate025_delta_vs_hcs'], True)} |",
        f"| trained min090 risk | {fmt(overall['trained_min090_rd'])} | {fmt(overall['trained_min090_delta_vs_hcs'], True)} |",
        f"| trained min095 risk | {fmt(overall['trained_min095_rd'])} | {fmt(overall['trained_min095_delta_vs_hcs'], True)} |",
        "",
        "## Per Seed",
        "",
        "| seed | HCS | old gate0.25 | min090 | min095 | min095-HCS | min095-old | min095-min090 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in map(str, SEEDS):
        row = per_seed[seed]
        lines.append(
            f"| {seed} | {fmt(row['hcs_rd'])} | {fmt(row['old_gate025_rd'])} | "
            f"{fmt(row['trained_min090_rd'])} | {fmt(row['trained_min095_rd'])} | "
            f"{fmt(row['trained_min095_delta_vs_hcs'], True)} | "
            f"{fmt(row['min095_minus_old_gate025'], True)} | "
            f"{fmt(row['min095_minus_min090'], True)} |"
        )

    lines.extend(
        [
            "",
            "## Intermediate Features",
            "",
            "| seed | s_q | raw gate | risk mult | eff. strength | delta RMS | latent qMSE | index bpp | dead code |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in map(str, SEEDS):
        row = per_seed[seed]
        lines.append(
            f"| {seed} | {fmt(row['s_q_mean'])} | {fmt(row['raw_gate_mean'])} | "
            f"{fmt(row['risk_multiplier_mean'])} | {fmt(row['effective_strength_mean'])} | "
            f"{fmt(row['householder_delta_rms'])} | {fmt(row['rvq_latent_quant_mse'])} | "
            f"{fmt(row['index_empirical_bpp'])} | {fmt(row['rvq_dead_code_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## HCS-Difficulty Quartiles",
            "",
            "| seed | Q | HCS RD | old-HCS | min090-HCS | min095-HCS | s_q | risk mult | eff. strength | latent qMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in map(str, SEEDS):
        for row in quartiles[seed]:
            lines.append(
                f"| {seed} | {row['quartile']} | {fmt(row['hcs_rd'])} | "
                f"{fmt(row['old_delta'], True)} | {fmt(row['trained_min090_delta'], True)} | "
                f"{fmt(row['trained_min095_delta'], True)} | {fmt(row['s_q_mean'])} | "
                f"{fmt(row['risk_multiplier_mean'])} | {fmt(row['effective_strength_mean'])} | "
                f"{fmt(row['latent_quant_mse'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `min095` is rejected as a main single-checkpoint controller in this training setting. It is much worse than HCS, old gate0.25, and trained min090 on every seed.",
            "- Raising the risk floor from 0.90 to 0.95 does not solve the stability issue. The risk multiplier is indeed conservative, but the trained model still moves to a high-strength geometry regime with high latent quantization error.",
            "- The failure is therefore not just over-suppressing the Householder gate. It points to unconstrained geometry/scale co-adaptation under this controller and training schedule.",
            "- The next controller should constrain the geometry directly, for example with lower learning rate, explicit delta/strength regularization, or a staged schedule, while keeping the old gate0.25 row as the safest current single-checkpoint HCG-H baseline.",
        ]
    )

    md_path = analysis_dir / "risk_floor_min095_val4096_holdout4096_current.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
