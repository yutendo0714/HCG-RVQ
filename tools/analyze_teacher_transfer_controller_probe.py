#!/usr/bin/env python3
"""Analyze the split-teacher reliability-controller probe on holdout4096."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


VARIANT_CSV = Path(
    "experiments/analysis/teacher_transfer8192_rel075_betacommit005_holdout4096_checkpoint_sweep.csv"
)
REFERENCE_CSV = Path(
    "experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv"
)
OUT_JSON = Path(
    "experiments/analysis/teacher_transfer8192_rel075_betacommit005_seed3456_probe.json"
)
OUT_MD = Path(
    "experiments/analysis/teacher_transfer8192_rel075_betacommit005_seed3456_probe.md"
)

REF_COLUMNS = {
    "HCS": "hcs_rd",
    "old gate0.25": "old_rd",
    "min090": "min090_rd",
    "previous-local": "previous_local_rd",
    "beta005": "variant500_rd",
}

FEATURE_COLUMNS = [
    "rvq_s_q_mean",
    "rvq_latent_quant_mse",
    "rvq_householder_gate_raw",
    "rvq_householder_reliability_multiplier",
    "rvq_householder_reliability_multiplier_min",
    "rvq_householder_reliability_multiplier_max",
    "rvq_householder_risk_multiplier",
    "rvq_householder_delta_rms",
    "rvq_householder_strength",
    "rvq_dead_code_ratio",
    "rvq_perplexity",
]


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return {}

    def percentile(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = q * (n - 1)
        low = int(pos)
        high = min(low + 1, n - 1)
        frac = pos - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    mu = mean(ordered)
    var = mean([(x - mu) ** 2 for x in ordered])
    return {
        "mean": mu,
        "std": var**0.5,
        "min": ordered[0],
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def quartile_bins(rows: list[dict[str, float]], key: str) -> list[list[dict[str, float]]]:
    ordered = sorted(rows, key=lambda row: row[key])
    n = len(ordered)
    return [ordered[(n * i) // 4 : (n * (i + 1)) // 4] for i in range(4)]


def main() -> None:
    variant_rows = load_csv(VARIANT_CSV)
    ref_rows = [
        row
        for row in load_csv(REFERENCE_CSV)
        if str(row.get("seed", "")) == "3456"
    ]

    refs_by_path = {row["path"]: row for row in ref_rows}
    variant_by_step: dict[int, list[dict[str, str]]] = {}
    for row in variant_rows:
        seed = str(row.get("seed", ""))
        if seed and seed != "3456":
            continue
        variant_by_step.setdefault(int(row["step"]), []).append(row)

    report: dict[str, object] = {
        "variant_csv": str(VARIANT_CSV),
        "reference_csv": str(REFERENCE_CSV),
        "seed": 3456,
        "steps": {},
    }

    for step, rows in sorted(variant_by_step.items()):
        matched = []
        missing = 0
        for row in rows:
            ref = refs_by_path.get(row["path"])
            if ref is None:
                missing += 1
                continue
            item: dict[str, float] = {
                "rd": as_float(row, "rd_score"),
            }
            for label, col in REF_COLUMNS.items():
                item[label] = as_float(ref, col)
            for col in FEATURE_COLUMNS:
                if row.get(col, "") != "":
                    item[col] = as_float(row, col)
            matched.append(item)

        rd_values = [row["rd"] for row in matched]
        step_report: dict[str, object] = {
            "rows": len(rows),
            "matched_rows": len(matched),
            "missing_reference_rows": missing,
            "rd": summarize(rd_values),
            "nonfinite_rows": sum(int(row["has_nonfinite"]) for row in rows),
            "comparisons": {},
            "features": {},
            "hcs_difficulty_quartiles": [],
        }

        comparisons: dict[str, object] = {}
        for label in REF_COLUMNS:
            deltas = [row["rd"] - row[label] for row in matched]
            wins = [1.0 if row["rd"] < row[label] else 0.0 for row in matched]
            comparisons[label] = {
                "reference_mean_rd": mean([row[label] for row in matched]),
                "mean_delta": mean(deltas),
                "median_delta": summarize(deltas)["p50"],
                "win_fraction": mean(wins),
            }
        step_report["comparisons"] = comparisons

        features = {}
        for col in FEATURE_COLUMNS:
            values = [row[col] for row in matched if col in row]
            if values:
                features[col] = summarize(values)
        step_report["features"] = features

        quartiles = []
        for idx, qrows in enumerate(quartile_bins(matched, "HCS"), start=1):
            quartiles.append(
                {
                    "quartile": idx,
                    "rows": len(qrows),
                    "hcs_mean": mean([row["HCS"] for row in qrows]),
                    "rd": mean([row["rd"] for row in qrows]),
                    "delta_vs_hcs": mean([row["rd"] - row["HCS"] for row in qrows]),
                    "delta_vs_beta005": mean(
                        [row["rd"] - row["beta005"] for row in qrows]
                    ),
                    "delta_vs_previous_local": mean(
                        [row["rd"] - row["previous-local"] for row in qrows]
                    ),
                    "win_vs_beta005": mean(
                        [1.0 if row["rd"] < row["beta005"] else 0.0 for row in qrows]
                    ),
                }
            )
        step_report["hcs_difficulty_quartiles"] = quartiles

        report["steps"][str(step)] = step_report

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Teacher-Transfer Reliability Controller Probe",
        "",
        "This path-aligned audit compares the split-teacher controller against the fixed holdout4096 references for seed3456.",
        "",
        "| step | RD | vs HCS | vs old gate0.25 | vs min090 | vs previous-local | vs beta005 | win vs beta005 | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    steps = report["steps"]
    assert isinstance(steps, dict)
    for step, raw in sorted(steps.items(), key=lambda item: int(item[0])):
        data = raw
        assert isinstance(data, dict)
        comps = data["comparisons"]
        assert isinstance(comps, dict)
        rd_summary = data["rd"]
        assert isinstance(rd_summary, dict)
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    fmt(rd_summary["mean"]),
                    fmt(comps["HCS"]["mean_delta"]),
                    fmt(comps["old gate0.25"]["mean_delta"]),
                    fmt(comps["min090"]["mean_delta"]),
                    fmt(comps["previous-local"]["mean_delta"]),
                    fmt(comps["beta005"]["mean_delta"]),
                    fmt(comps["beta005"]["win_fraction"]),
                    str(data["nonfinite_rows"]),
                ]
            )
            + " |"
        )

    lines += ["", "## Feature Means", ""]
    lines.append(
        "| step | s_q | qMSE | raw gate | reliability | reliability min | reliability max | risk multiplier | delta RMS | strength | dead code | perplexity |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step, raw in sorted(steps.items(), key=lambda item: int(item[0])):
        data = raw
        assert isinstance(data, dict)
        features = data["features"]
        assert isinstance(features, dict)

        def fmean(key: str) -> str:
            return fmt(features[key]["mean"])

        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    fmean("rvq_s_q_mean"),
                    fmean("rvq_latent_quant_mse"),
                    fmean("rvq_householder_gate_raw"),
                    fmean("rvq_householder_reliability_multiplier"),
                    fmean("rvq_householder_reliability_multiplier_min"),
                    fmean("rvq_householder_reliability_multiplier_max"),
                    fmean("rvq_householder_risk_multiplier"),
                    fmean("rvq_householder_delta_rms"),
                    fmean("rvq_householder_strength"),
                    fmean("rvq_dead_code_ratio"),
                    fmean("rvq_perplexity"),
                ]
            )
            + " |"
        )

    lines += ["", "## HCS-Difficulty Quartiles", ""]
    lines.append(
        "| step | quartile | HCS RD | RD | vs HCS | vs beta005 | vs previous-local | win vs beta005 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step, raw in sorted(steps.items(), key=lambda item: int(item[0])):
        data = raw
        assert isinstance(data, dict)
        quartiles = data["hcs_difficulty_quartiles"]
        assert isinstance(quartiles, list)
        for qrow in quartiles:
            lines.append(
                "| "
                + " | ".join(
                    [
                        step,
                        str(qrow["quartile"]),
                        fmt(qrow["hcs_mean"]),
                        fmt(qrow["rd"]),
                        fmt(qrow["delta_vs_hcs"]),
                        fmt(qrow["delta_vs_beta005"]),
                        fmt(qrow["delta_vs_previous_local"]),
                        fmt(qrow["win_vs_beta005"]),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "Interpretation: the split-generated teacher labels contain real oracle headroom, but this rel075/rho0.10 implementation does not recover it as a single checkpoint. It is numerically stable, yet it lands near the previous-local operating point and remains clearly worse than beta005 on seed3456.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
