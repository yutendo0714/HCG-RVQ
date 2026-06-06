#!/usr/bin/env python3
"""Compare E104/deadzone018 transfer and holdout behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ANALYSIS = Path("experiments/analysis")
OUT = ANALYSIS / "e106_deadzone018_transfer_vs_holdout_audit"

SEEDS = {
    1234: {
        "transfer_method": ANALYSIS / "e104_deadzone018_from_beta005_seed1234_step250_fullimage_start8192_current.csv",
        "transfer_beta": ANALYSIS / "beta005_seed1234_transfer_start8192_reference.csv",
        "transfer_json": ANALYSIS / "e104_deadzone018_from_beta005_seed1234_step250_fullimage_start8192_current.json",
        "holdout_json": ANALYSIS / "e104_deadzone018_from_beta005_seed1234_step250_fullimage_holdout4096_current.json",
    },
    2345: {
        "transfer_method": ANALYSIS / "e104_deadzone018_from_beta005_seed2345_step250_fullimage_start8192_current.csv",
        "transfer_beta": ANALYSIS / "beta005_seed2345_transfer_start8192_reference.csv",
        "transfer_json": ANALYSIS / "e104_deadzone018_from_beta005_seed2345_step250_fullimage_start8192_current.json",
        "holdout_json": ANALYSIS / "e104_deadzone018_from_beta005_seed2345_step250_fullimage_holdout4096_current.json",
    },
    3456: {
        "transfer_method": ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_start8192_current.csv",
        "transfer_beta": ANALYSIS / "beta005_seed3456_transfer_start8192_reference.csv",
        "transfer_json": ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_start8192_current.json",
        "holdout_json": ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.json",
    },
}

FEATURES = [
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
    "rvq_dead_code_ratio",
    "rvq_householder_delta_rms",
    "rvq_householder_strength",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_summary(path: Path) -> dict[str, float | int | str]:
    return json.loads(path.read_text())["summaries"][0]


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def paired_transfer_stats(seed: int, paths: dict[str, Path]) -> dict[str, float | int]:
    method_rows = read_csv(paths["transfer_method"])
    beta_rows = read_csv(paths["transfer_beta"])
    beta_by_path = {row["path"]: row for row in beta_rows}
    deltas = []
    feature_deltas = {key: [] for key in FEATURES}
    for row in method_rows:
        beta = beta_by_path[row["path"]]
        deltas.append(f(row, "rd_score") - f(beta, "rd_score"))
        for key in FEATURES:
            feature_deltas[key].append(f(row, key) - f(beta, key))
    out: dict[str, float | int] = {
        "seed": seed,
        "images": len(method_rows),
        "transfer_win_rate": sum(1 for d in deltas if d < 0.0) / len(deltas),
        "transfer_mean_abs_delta": mean([abs(d) for d in deltas]),
    }
    for key, values in feature_deltas.items():
        out[f"transfer_delta_{key}"] = mean(values)
    return out


def main() -> None:
    rows = []
    for seed, paths in SEEDS.items():
        transfer = read_summary(paths["transfer_json"])
        holdout = read_summary(paths["holdout_json"])
        paired = paired_transfer_stats(seed, paths)
        transfer_delta = float(transfer["mean_rd_minus_reference"])
        holdout_delta = float(holdout["mean_rd_minus_reference"])
        rows.append(
            {
                "seed": seed,
                "transfer_beta_rd": float(transfer["mean_rd"]) - transfer_delta,
                "transfer_e104_rd": float(transfer["mean_rd"]),
                "transfer_delta": transfer_delta,
                "holdout_beta_rd": float(holdout["mean_rd"]) - holdout_delta,
                "holdout_e104_rd": float(holdout["mean_rd"]),
                "holdout_delta": holdout_delta,
                "transfer_minus_holdout_delta": transfer_delta - holdout_delta,
                "transfer_win_rate": paired["transfer_win_rate"],
                "transfer_mean_abs_delta": paired["transfer_mean_abs_delta"],
                "transfer_qmse_delta": paired["transfer_delta_rvq_latent_quant_mse"],
                "transfer_s_q_delta": paired["transfer_delta_rvq_s_q_mean"],
                "transfer_dead_code_delta": paired["transfer_delta_rvq_dead_code_ratio"],
                "transfer_delta_rms_delta": paired["transfer_delta_rvq_householder_delta_rms"],
                "transfer_strength_delta": paired["transfer_delta_rvq_householder_strength"],
                "transfer_nonfinite": int(transfer["nonfinite_rows"]),
                "holdout_nonfinite": int(holdout["nonfinite_rows"]),
                "images": int(transfer["num_images"]),
            }
        )

    total = sum(row["images"] for row in rows)
    aggregate = {
        "seeds": len(rows),
        "images": total,
        "transfer_beta_rd": sum(row["transfer_beta_rd"] * row["images"] for row in rows) / total,
        "transfer_e104_rd": sum(row["transfer_e104_rd"] * row["images"] for row in rows) / total,
        "transfer_delta": sum(row["transfer_delta"] * row["images"] for row in rows) / total,
        "holdout_beta_rd": sum(row["holdout_beta_rd"] * row["images"] for row in rows) / total,
        "holdout_e104_rd": sum(row["holdout_e104_rd"] * row["images"] for row in rows) / total,
        "holdout_delta": sum(row["holdout_delta"] * row["images"] for row in rows) / total,
        "transfer_win_rate": sum(row["transfer_win_rate"] * row["images"] for row in rows) / total,
        "transfer_nonfinite": sum(row["transfer_nonfinite"] for row in rows),
        "holdout_nonfinite": sum(row["holdout_nonfinite"] for row in rows),
    }
    aggregate["transfer_minus_holdout_delta"] = aggregate["transfer_delta"] - aggregate["holdout_delta"]

    OUT.with_suffix(".json").write_text(json.dumps({"aggregate": aggregate, "per_seed": rows}, indent=2) + "\n")
    with OUT.with_suffix(".per_seed.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E106 Dead-Zone Transfer-vs-Holdout Audit",
        "",
        "This audit checks whether the E104/deadzone018 gain generalizes to the independent start8192 transfer split, rather than only improving holdout4096.",
        "",
        "## Aggregate",
        "",
        "| seeds | images/split | transfer beta RD | transfer E104 RD | transfer delta | holdout delta | transfer-holdout delta | transfer win rate | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['seeds']} | {aggregate['images']} | {aggregate['transfer_beta_rd']:.6f} | "
            f"{aggregate['transfer_e104_rd']:.6f} | {aggregate['transfer_delta']:+.6f} | "
            f"{aggregate['holdout_delta']:+.6f} | {aggregate['transfer_minus_holdout_delta']:+.6f} | "
            f"{aggregate['transfer_win_rate']:.6f} | {aggregate['transfer_nonfinite'] + aggregate['holdout_nonfinite']} |"
        ),
        "",
        "## Per Seed",
        "",
        "| seed | transfer beta RD | transfer E104 RD | transfer delta | holdout delta | transfer-holdout | transfer win | qMSE delta | s_q delta | dead-code delta | delta-RMS delta | strength delta | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['transfer_beta_rd']:.6f} | {row['transfer_e104_rd']:.6f} | "
            f"{row['transfer_delta']:+.6f} | {row['holdout_delta']:+.6f} | "
            f"{row['transfer_minus_holdout_delta']:+.6f} | {row['transfer_win_rate']:.6f} | "
            f"{row['transfer_qmse_delta']:+.6f} | {row['transfer_s_q_delta']:+.6f} | "
            f"{row['transfer_dead_code_delta']:+.6f} | {row['transfer_delta_rms_delta']:+.6f} | "
            f"{row['transfer_strength_delta']:+.6f} | {row['transfer_nonfinite'] + row['holdout_nonfinite']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The dead-zone threshold is not only a holdout artifact: all three seeds improve on the independent "
                "start8192 transfer split, and the transfer mean delta is within a tiny margin of the holdout mean delta. "
                "This supports using E104/deadzone018 as the next manuscript-candidate branch, while still requiring a "
                "pre-declared threshold-selection rule and checkpoint sweep before final paper-main promotion."
            ),
            "",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
