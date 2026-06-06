#!/usr/bin/env python3
"""Path-aligned audit for a beta005-initialized head-only teacher variant."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

REFERENCE_CSV = Path("experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv")

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


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def parse_step_csvs(items: list[str]) -> dict[int, Path]:
    step_csvs: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--step-csv must be STEP=PATH, got {item!r}")
        step_text, path_text = item.split("=", 1)
        step_csvs[int(step_text)] = Path(path_text)
    return step_csvs


def fmt(value: float, signed: bool = False) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def build_report(args: argparse.Namespace) -> dict[str, object]:
    refs = {
        row["path"]: row
        for row in load_csv(Path(args.reference_csv))
        if str(row.get("seed", "")) == str(args.seed)
    }
    report: dict[str, object] = {
        "title": args.title,
        "description": args.description,
        "interpretation": args.interpretation,
        "seed": args.seed,
        "reference_csv": args.reference_csv,
        "steps": {},
    }

    for step, path in parse_step_csvs(args.step_csv).items():
        rows = load_csv(path)
        matched = []
        missing = 0
        for row in rows:
            ref = refs.get(row["path"])
            if ref is None:
                missing += 1
                continue
            item = {"rd": float(row["rd_score"])}
            for label, col in REF_COLUMNS.items():
                item[label] = float(ref[col])
            for col in FEATURE_COLUMNS:
                if row.get(col, "") != "":
                    item[col] = float(row[col])
            matched.append(item)

        comparisons = {}
        for label in REF_COLUMNS:
            deltas = [row["rd"] - row[label] for row in matched]
            wins = [1.0 if row["rd"] < row[label] else 0.0 for row in matched]
            comparisons[label] = {
                "reference_mean_rd": mean([row[label] for row in matched]),
                "mean_delta": mean(deltas),
                "median_delta": summarize(deltas)["p50"],
                "win_fraction": mean(wins),
            }

        features = {}
        for col in FEATURE_COLUMNS:
            values = [row[col] for row in matched if col in row]
            features[col] = summarize(values)

        beta_deltas = [row["rd"] - row["beta005"] for row in matched]
        correlations = {
            "corr_reliability_rd_delta_vs_beta005": pearson(
                [row.get("rvq_householder_reliability_multiplier", float("nan")) for row in matched], beta_deltas
            ),
            "corr_rawgate_rd_delta_vs_beta005": pearson(
                [row.get("rvq_householder_gate_raw", float("nan")) for row in matched], beta_deltas
            ),
            "corr_delta_rms_rd_delta_vs_beta005": pearson(
                [row.get("rvq_householder_delta_rms", float("nan")) for row in matched], beta_deltas
            ),
            "corr_qmse_rd_delta_vs_beta005": pearson(
                [row.get("rvq_latent_quant_mse", float("nan")) for row in matched], beta_deltas
            ),
            "corr_deadcode_rd_delta_vs_beta005": pearson(
                [row.get("rvq_dead_code_ratio", float("nan")) for row in matched], beta_deltas
            ),
        }

        quartiles = []
        for idx, qrows in enumerate(quartile_bins(matched, "HCS"), start=1):
            quartiles.append(
                {
                    "quartile": idx,
                    "rows": len(qrows),
                    "hcs_mean": mean([row["HCS"] for row in qrows]),
                    "rd": mean([row["rd"] for row in qrows]),
                    "delta_vs_hcs": mean([row["rd"] - row["HCS"] for row in qrows]),
                    "delta_vs_beta005": mean([row["rd"] - row["beta005"] for row in qrows]),
                    "delta_vs_previous_local": mean([row["rd"] - row["previous-local"] for row in qrows]),
                    "win_vs_beta005": mean([1.0 if row["rd"] < row["beta005"] else 0.0 for row in qrows]),
                }
            )

        report["steps"][str(step)] = {
            "csv": str(path),
            "rows": len(rows),
            "matched_rows": len(matched),
            "missing_reference_rows": missing,
            "nonfinite_rows": sum(int(row["has_nonfinite"]) for row in rows),
            "rd": summarize([row["rd"] for row in matched]),
            "comparisons": comparisons,
            "features": features,
            "correlations": correlations,
            "hcs_difficulty_quartiles": quartiles,
        }
    return report


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        str(report.get("description", "")),
        "",
        "| step | RD | vs HCS | vs old gate0.25 | vs min090 | vs previous-local | vs beta005 | win vs beta005 | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    steps = report["steps"]
    assert isinstance(steps, dict)
    for step, data in sorted(steps.items(), key=lambda item: int(item[0])):
        comps = data["comparisons"]
        rd = data["rd"]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    fmt(float(rd["mean"])),
                    fmt(float(comps["HCS"]["mean_delta"]), signed=True),
                    fmt(float(comps["old gate0.25"]["mean_delta"]), signed=True),
                    fmt(float(comps["min090"]["mean_delta"]), signed=True),
                    fmt(float(comps["previous-local"]["mean_delta"]), signed=True),
                    fmt(float(comps["beta005"]["mean_delta"]), signed=True),
                    fmt(float(comps["beta005"]["win_fraction"])),
                    str(data["nonfinite_rows"]),
                ]
            )
            + " |"
        )

    lines += ["", "## Feature Means", ""]
    lines.append("| step | s_q | qMSE | raw gate | reliability | reliability min | reliability max | risk multiplier | delta RMS | strength | dead code | perplexity |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step, data in sorted(steps.items(), key=lambda item: int(item[0])):
        features = data["features"]
        def fmean(key: str) -> str:
            return fmt(float(features[key]["mean"]))
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

    lines += ["", "## Correlations With RD Delta vs Beta005", ""]
    lines.append("| step | reliability | raw gate | delta RMS | qMSE | dead code |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for step, data in sorted(steps.items(), key=lambda item: int(item[0])):
        corr = data["correlations"]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    fmt(float(corr["corr_reliability_rd_delta_vs_beta005"])),
                    fmt(float(corr["corr_rawgate_rd_delta_vs_beta005"])),
                    fmt(float(corr["corr_delta_rms_rd_delta_vs_beta005"])),
                    fmt(float(corr["corr_qmse_rd_delta_vs_beta005"])),
                    fmt(float(corr["corr_deadcode_rd_delta_vs_beta005"])),
                ]
            )
            + " |"
        )

    lines += ["", "## HCS-Difficulty Quartiles", ""]
    lines.append("| step | quartile | HCS RD | RD | vs HCS | vs beta005 | win vs beta005 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for step, data in sorted(steps.items(), key=lambda item: int(item[0])):
        for qrow in data["hcs_difficulty_quartiles"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        step,
                        str(qrow["quartile"]),
                        fmt(float(qrow["hcs_mean"])),
                        fmt(float(qrow["rd"])),
                        fmt(float(qrow["delta_vs_hcs"]), signed=True),
                        fmt(float(qrow["delta_vs_beta005"]), signed=True),
                        fmt(float(qrow["win_vs_beta005"])),
                    ]
                )
                + " |"
            )

    lines += ["", f"Interpretation: {report.get('interpretation', '')}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-csv", action="append", required=True, help="STEP=CSV path. May be repeated.")
    parser.add_argument("--reference-csv", default=str(REFERENCE_CSV))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--title", default="Head-Only Teacher Variant Probe")
    parser.add_argument("--description", default="Path-aligned holdout4096 audit against HCS, old gate0.25, min090, previous-local, and beta005 references.")
    parser.add_argument("--interpretation", default="")
    args = parser.parse_args()
    report = build_report(args)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
