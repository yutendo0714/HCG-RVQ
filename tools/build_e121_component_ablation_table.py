#!/usr/bin/env python3
"""Build a component ablation table from existing fixed-protocol evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e121_component_ablation_table"


def load_json(name: str) -> dict:
    with (ANALYSIS / name).open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, *, signed: bool = False) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def main() -> None:
    beta = load_json("beta005_paper_evidence_summary.json")
    e118 = load_json("e118_hcg_rvq_prototype_main_table_package.json")

    dz_rows = {
        (row["split"], row["threshold"]): row
        for row in e118["table_rows"]
    }

    component_specs = [
        {
            "component": "HCS-RVQ",
            "short_name": "hcs",
            "paper_role": "shift/scale + index entropy baseline",
            "geometry": "none",
            "reliability": "none",
            "prompt_mapping": "hyperprior-conditioned shift/scale RVQ with index entropy",
        },
        {
            "component": "old gate0.25",
            "short_name": "old",
            "paper_role": "raw Householder geometry control",
            "geometry": "Householder",
            "reliability": "fixed raw gate",
            "prompt_mapping": "shift/scale + Householder geometry",
        },
        {
            "component": "min090 risk gate",
            "short_name": "min090",
            "paper_role": "conservative risk-gated geometry",
            "geometry": "Householder",
            "reliability": "detached inverse-risk floor",
            "prompt_mapping": "geometry with reliability control diagnostic",
        },
        {
            "component": "beta005 guard",
            "short_name": "beta",
            "paper_role": "broad fixed-checkpoint guard baseline",
            "geometry": "Householder",
            "reliability": "teacher/anchor/beta-commit guard",
            "prompt_mapping": "stabilized HCG-RVQ geometry",
        },
        {
            "component": "deadzone014",
            "short_name": "dz014",
            "paper_role": "main mean-RD HCG-RVQ branch",
            "geometry": "Householder",
            "reliability": "residual-selector dead-zone 0.014",
            "prompt_mapping": "stabilized geometry + reliability selector",
        },
        {
            "component": "deadzone018",
            "short_name": "dz018",
            "paper_role": "tail-safety HCG-RVQ ablation",
            "geometry": "Householder",
            "reliability": "residual-selector dead-zone 0.018",
            "prompt_mapping": "conservative geometry + reliability selector",
        },
    ]

    split_rows: list[dict[str, object]] = []
    for beta_row in beta["rows"]:
        split = beta_row["split"]
        values = {
            "hcs": beta_row["hcs"],
            "old": beta_row["old"],
            "min090": beta_row["min090"],
            "beta": beta_row["beta"],
            "dz014": dz_rows[(split, "014")]["deadzone_rd"],
            "dz018": dz_rows[(split, "018")]["deadzone_rd"],
        }
        for spec in component_specs:
            rd = float(values[spec["short_name"]])
            split_rows.append(
                {
                    "split": split,
                    "component": spec["component"],
                    "paper_role": spec["paper_role"],
                    "geometry": spec["geometry"],
                    "reliability": spec["reliability"],
                    "prompt_mapping": spec["prompt_mapping"],
                    "rd": rd,
                    "delta_vs_hcs": rd - float(beta_row["hcs"]),
                    "delta_vs_beta005": rd - float(beta_row["beta"]),
                    "nonfinite": beta_row["nonfinite"]
                    if spec["short_name"] in {"hcs", "old", "min090", "beta"}
                    else dz_rows[(split, "014" if spec["short_name"] == "dz014" else "018")]["nonfinite_rows"],
                }
            )

    summary_rows: list[dict[str, object]] = []
    for spec in component_specs:
        rows = [row for row in split_rows if row["component"] == spec["component"]]
        summary_rows.append(
            {
                "component": spec["component"],
                "paper_role": spec["paper_role"],
                "geometry": spec["geometry"],
                "reliability": spec["reliability"],
                "prompt_mapping": spec["prompt_mapping"],
                "num_splits": len(rows),
                "mean_split_delta_vs_hcs": mean(float(row["delta_vs_hcs"]) for row in rows),
                "worst_split_delta_vs_hcs": max(float(row["delta_vs_hcs"]) for row in rows),
                "mean_split_delta_vs_beta005": mean(float(row["delta_vs_beta005"]) for row in rows),
                "all_splits_improve_hcs": all(float(row["delta_vs_hcs"]) < 0.0 for row in rows)
                if spec["short_name"] != "hcs"
                else False,
                "nonfinite": sum(int(row["nonfinite"]) for row in rows),
            }
        )

    missing_rows = [
        {
            "missing_or_partial": "explicit entropy-only / HVQ-like row",
            "status": "missing as final fixed-protocol table row",
            "reason": "HCS includes shift/scale plus index entropy; it is not a pure entropy-only control.",
            "next_action": "If time allows, add a frozen checkpoint row with hyperprior index entropy but no shift/scale geometry change.",
        },
        {
            "missing_or_partial": "multi-rate repetition",
            "status": "missing",
            "reason": "All final dead-zone evidence is at lambda=0.0035.",
            "next_action": "Repeat the fixed protocol for at least two additional lambdas before final submission.",
        },
        {
            "missing_or_partial": "strong-backbone plug-in",
            "status": "missing",
            "reason": "E118 is still MeanScaleHyperprior/RVQ prototype evidence.",
            "next_action": "Run adapter extraction first, then local CompressAI strong-backbone smoke.",
        },
    ]

    decision = {
        "component_table_status": "usable for prototype manuscript, but entropy-only and multi-rate rows remain missing",
        "best_mean_component": min(summary_rows, key=lambda row: float(row["mean_split_delta_vs_hcs"]))["component"],
        "dz014_mean_delta_vs_hcs": next(
            row for row in summary_rows if row["component"] == "deadzone014"
        )["mean_split_delta_vs_hcs"],
        "dz018_mean_delta_vs_hcs": next(
            row for row in summary_rows if row["component"] == "deadzone018"
        )["mean_split_delta_vs_hcs"],
        "beta005_mean_delta_vs_hcs": next(
            row for row in summary_rows if row["component"] == "beta005 guard"
        )["mean_split_delta_vs_hcs"],
    }

    payload = {
        "decision": decision,
        "summary_rows": summary_rows,
        "split_rows": split_rows,
        "missing_rows": missing_rows,
        "sources": {
            "beta005": "beta005_paper_evidence_summary.json",
            "deadzone": "e118_hcg_rvq_prototype_main_table_package.json",
        },
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".summary.csv"), summary_rows)
    write_csv(OUT_PREFIX.with_suffix(".splits.csv"), split_rows)
    write_csv(OUT_PREFIX.with_suffix(".missing.csv"), missing_rows)

    lines = [
        "# E121 Component Ablation Table",
        "",
        "This table converts existing fixed-protocol evidence into a paper-readable component ablation view.",
        "",
        "## Decision",
        "",
        f"- Status: {decision['component_table_status']}",
        f"- Best mean component: `{decision['best_mean_component']}`",
        f"- dz014 mean delta vs HCS: `{fmt(decision['dz014_mean_delta_vs_hcs'], signed=True)}`",
        f"- dz018 mean delta vs HCS: `{fmt(decision['dz018_mean_delta_vs_hcs'], signed=True)}`",
        f"- beta005 mean delta vs HCS: `{fmt(decision['beta005_mean_delta_vs_hcs'], signed=True)}`",
        "",
        "## Summary",
        "",
        "| component | role | geometry | reliability | mean delta vs HCS | worst delta vs HCS | mean delta vs beta005 | all improve HCS | nonfinite |",
        "|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {component} | {paper_role} | {geometry} | {reliability} | {mean_hcs} | {worst_hcs} | {mean_beta} | {all_hcs} | {nonfinite} |".format(
                component=row["component"],
                paper_role=row["paper_role"],
                geometry=row["geometry"],
                reliability=row["reliability"],
                mean_hcs=fmt(row["mean_split_delta_vs_hcs"], signed=True),
                worst_hcs=fmt(row["worst_split_delta_vs_hcs"], signed=True),
                mean_beta=fmt(row["mean_split_delta_vs_beta005"], signed=True),
                all_hcs=row["all_splits_improve_hcs"],
                nonfinite=int(row["nonfinite"]),
            )
        )
    lines.extend(
        [
            "",
            "## Missing Or Partial Rows",
            "",
            "| item | status | reason | next action |",
            "|---|---|---|---|",
        ]
    )
    for row in missing_rows:
        lines.append(
            f"| {row['missing_or_partial']} | {row['status']} | {row['reason']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_suffix('.summary.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.splits.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.missing.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
