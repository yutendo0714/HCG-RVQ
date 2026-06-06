import csv
import json
import math
from pathlib import Path

from analyze_gate_selector import SEEDS, load_rows


POSTHOC_STEPS = {
    1234: 250,
    2345: 250,
    3456: 500,
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_float(row, key):
    raw = row.get(key, "")
    return float(raw) if raw != "" else float("nan")


def mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value, signed=False):
    if not math.isfinite(value):
        return "nan"
    return ("{:+.6f}" if signed else "{:.6f}").format(value)


def feature_path(analysis_dir, seed):
    step = POSTHOC_STEPS[seed]
    return (
        analysis_dir
        / f"per_image_features_hcg_h_gate025_posthoc_min090_oldw_seed{seed}_step{step}_val4096_holdout4096_current.csv"
    )


def compare_path(analysis_dir, seed):
    hcs_step = 500 if seed == 1234 else 250
    step = POSTHOC_STEPS[seed]
    return (
        analysis_dir
        / f"per_image_seed{seed}_hcs{hcs_step}_vs_hcgh_gate025_posthoc_min090_oldw_step{step}_val4096_holdout4096_current.csv"
    )


def summarize_joined(rows):
    return {
        "n": len(rows),
        "hcs_rd": mean(row["hcs_rd"] for row in rows),
        "old_gate025_rd": mean(row["old_rd"] for row in rows),
        "trained_min090_rd": mean(row["trained_min090_rd"] for row in rows),
        "posthoc_min090_oldw_rd": mean(row["posthoc_rd"] for row in rows),
        "old_gate025_delta_vs_hcs": mean(row["old_rd"] - row["hcs_rd"] for row in rows),
        "trained_min090_delta_vs_hcs": mean(
            row["trained_min090_rd"] - row["hcs_rd"] for row in rows
        ),
        "posthoc_delta_vs_hcs": mean(row["posthoc_rd"] - row["hcs_rd"] for row in rows),
        "posthoc_minus_old_gate025": mean(row["posthoc_rd"] - row["old_rd"] for row in rows),
        "posthoc_minus_trained_min090": mean(
            row["posthoc_rd"] - row["trained_min090_rd"] for row in rows
        ),
        "posthoc_better_than_hcs_rate": mean(
            1.0 if row["posthoc_rd"] < row["hcs_rd"] else 0.0 for row in rows
        ),
        "posthoc_better_than_old_rate": mean(
            1.0 if row["posthoc_rd"] < row["old_rd"] else 0.0 for row in rows
        ),
        "posthoc_better_than_trained_min090_rate": mean(
            1.0 if row["posthoc_rd"] < row["trained_min090_rd"] else 0.0 for row in rows
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


def quartiles_for_seed(rows):
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
                "posthoc_delta": mean(row["posthoc_rd"] - row["hcs_rd"] for row in chunk),
                "s_q_mean": mean(row["s_q_mean"] for row in chunk),
                "risk_multiplier_mean": mean(row["risk_multiplier_mean"] for row in chunk),
                "effective_strength_mean": mean(row["strength_mean"] for row in chunk),
                "latent_quant_mse": mean(row["latent_quant_mse"] for row in chunk),
            }
        )
    return chunks


def compare_audit(analysis_dir, trusted_rows_by_seed, posthoc_by_seed):
    audits = {}
    for seed in SEEDS:
        path = compare_path(analysis_dir, seed)
        if not path.exists():
            continue
        rows = read_csv(path)
        trusted = trusted_rows_by_seed[seed]
        posthoc = posthoc_by_seed[seed]
        hcs_keys = [key for key in rows[0] if key.lower().endswith("rd_score") and key.startswith("HCS")]
        posthoc_keys = [
            key
            for key in rows[0]
            if key.lower().endswith("rd_score")
            and not key.startswith("HCS")
            and not key.startswith("delta")
        ]
        hcs_key = hcs_keys[0]
        posthoc_key = posthoc_keys[0]
        audits[str(seed)] = {
            "compare_csv": str(path),
            "compare_hcs_mean": mean(read_float(row, hcs_key) for row in rows),
            "trusted_hcs_mean": mean(row["hcs_rd"] for row in trusted),
            "compare_minus_trusted_hcs_mean": mean(read_float(row, hcs_key) for row in rows)
            - mean(row["hcs_rd"] for row in trusted),
            "compare_posthoc_mean": mean(read_float(row, posthoc_key) for row in rows),
            "feature_posthoc_mean": mean(row["posthoc_rd"] for row in posthoc),
            "compare_delta_mean": mean(read_float(row, "delta_rd_score") for row in rows),
            "trusted_delta_from_features": mean(
                row["posthoc_rd"] - row["hcs_rd"] for row in posthoc
            ),
        }
    return audits


def main():
    analysis_dir = Path("experiments/analysis")
    trusted_rows = load_rows(analysis_dir, "current_holdout")
    trusted_by_seed = {seed: [row for row in trusted_rows if row["seed"] == seed] for seed in SEEDS}
    posthoc_by_seed = {}
    joined_by_seed = {}

    for seed in SEEDS:
        trusted_by_path = {row["path"]: row for row in trusted_by_seed[seed]}
        posthoc_rows = read_csv(feature_path(analysis_dir, seed))
        joined = []
        for posthoc in posthoc_rows:
            trusted = trusted_by_path[posthoc["path"]]
            hcs_rd = trusted["hcs_rd"]
            old_rd = hcs_rd + trusted["old_delta_rd"]
            trained_min090_rd = hcs_rd + trusted["risk_delta_rd"]
            joined.append(
                {
                    "seed": seed,
                    "path": posthoc["path"],
                    "hcs_rd": hcs_rd,
                    "old_rd": old_rd,
                    "trained_min090_rd": trained_min090_rd,
                    "posthoc_rd": read_float(posthoc, "rd_score"),
                    "s_q_mean": read_float(posthoc, "s_q_mean"),
                    "raw_gate_mean": read_float(posthoc, "householder_gate_raw_mean"),
                    "risk_multiplier_mean": read_float(
                        posthoc, "householder_risk_multiplier_mean"
                    ),
                    "strength_mean": read_float(posthoc, "householder_strength_mean"),
                    "delta_rms": read_float(posthoc, "householder_delta_rms"),
                    "latent_quant_mse": read_float(posthoc, "rvq_latent_quant_mse"),
                    "index_empirical_bpp": read_float(posthoc, "index_empirical_bpp"),
                    "rvq_perplexity": read_float(posthoc, "rvq_perplexity"),
                    "rvq_dead_code_ratio": read_float(posthoc, "rvq_dead_code_ratio"),
                }
            )
        posthoc_by_seed[seed] = joined
        joined_by_seed[str(seed)] = joined

    all_joined = [row for rows in joined_by_seed.values() for row in rows]
    per_seed = {seed: summarize_joined(joined_by_seed[str(seed)]) for seed in map(str, SEEDS)}
    overall = summarize_joined(all_joined)
    quartiles = {seed: quartiles_for_seed(joined_by_seed[str(seed)]) for seed in map(str, SEEDS)}
    audits = compare_audit(analysis_dir, trusted_by_seed, posthoc_by_seed)

    summary = {
        "overall": overall,
        "per_seed": per_seed,
        "hcs_difficulty_quartiles": quartiles,
        "compare_artifact_audit": audits,
        "artifacts": {
            "posthoc_feature_csvs": [str(feature_path(analysis_dir, seed)) for seed in SEEDS],
            "trusted_selector_protocol": "experiments/analysis/gate025_min090_selector_reporting_protocol.json",
        },
    }

    json_path = analysis_dir / "posthoc_single_checkpoint_controller_val4096_holdout4096_current.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Posthoc single-checkpoint controller audit",
        "",
        "This audit evaluates the deterministic min090 risk controller when it is applied to the already-trained old gate0.25 weights, without training a separate min090 checkpoint. RD values are matched by image path against the trusted current-holdout HCS/old/min090 artifacts.",
        "",
        "## Overall",
        "",
        "| method | mean RD | delta vs HCS |",
        "|---|---:|---:|",
        f"| HCS | {fmt(overall['hcs_rd'])} | {fmt(0.0, True)} |",
        f"| old gate0.25 | {fmt(overall['old_gate025_rd'])} | {fmt(overall['old_gate025_delta_vs_hcs'], True)} |",
        f"| trained min090 risk | {fmt(overall['trained_min090_rd'])} | {fmt(overall['trained_min090_delta_vs_hcs'], True)} |",
        f"| posthoc min090 on old weights | {fmt(overall['posthoc_min090_oldw_rd'])} | {fmt(overall['posthoc_delta_vs_hcs'], True)} |",
        "",
        "## Per Seed",
        "",
        "| seed | HCS | old gate0.25 | trained min090 | posthoc oldw/min090 | posthoc-HCS | posthoc-old |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in map(str, SEEDS):
        row = per_seed[seed]
        lines.append(
            f"| {seed} | {fmt(row['hcs_rd'])} | {fmt(row['old_gate025_rd'])} | "
            f"{fmt(row['trained_min090_rd'])} | {fmt(row['posthoc_min090_oldw_rd'])} | "
            f"{fmt(row['posthoc_delta_vs_hcs'], True)} | {fmt(row['posthoc_minus_old_gate025'], True)} |"
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
            "| seed | Q | HCS RD | old delta | trained min090 delta | posthoc delta | s_q | risk mult | latent qMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in map(str, SEEDS):
        for row in quartiles[seed]:
            lines.append(
                f"| {seed} | {row['quartile']} | {fmt(row['hcs_rd'])} | "
                f"{fmt(row['old_delta'], True)} | {fmt(row['trained_min090_delta'], True)} | "
                f"{fmt(row['posthoc_delta'], True)} | {fmt(row['s_q_mean'])} | "
                f"{fmt(row['risk_multiplier_mean'])} | {fmt(row['latent_quant_mse'])} |"
            )

    lines.extend(
        [
            "",
            "## Artifact Audit",
            "",
            "| seed | compare HCS | trusted HCS | HCS gap | compare delta | trusted delta from features |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in map(str, SEEDS):
        row = audits[seed]
        lines.append(
            f"| {seed} | {fmt(row['compare_hcs_mean'])} | {fmt(row['trusted_hcs_mean'])} | "
            f"{fmt(row['compare_minus_trusted_hcs_mean'], True)} | "
            f"{fmt(row['compare_delta_mean'], True)} | {fmt(row['trusted_delta_from_features'], True)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The earlier posthoc comparison deltas are not paper-facing because their HCS side used a much worse baseline checkpoint. The posthoc model RD itself matches the feature diagnostics, but after matching it to the trusted current-holdout HCS rows, it is far worse than HCS, old gate0.25, and the separately trained min090 checkpoints.",
            "",
            "This means a pure inference-time min090 risk multiplier applied to old gate0.25 weights is not the right main path. The reliable result remains the trained old/min090 geometry variants plus image-level reliability/oracle analysis; the next method should train the controller jointly or constrain it with validation-selected safeguards rather than changing Householder strength posthoc.",
        ]
    )
    md_path = analysis_dir / "posthoc_single_checkpoint_controller_val4096_holdout4096_current.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
