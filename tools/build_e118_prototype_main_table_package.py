#!/usr/bin/env python3
"""Build prototype manuscript tables after the E117 external audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e118_hcg_rvq_prototype_main_table_package"
THRESHOLDS = ("014", "018")


def load_json(name: str) -> dict:
    with (ANALYSIS / name).open(encoding="utf-8") as f:
        return json.load(f)


def find(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def fmt(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


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


def main() -> None:
    beta = load_json("beta005_paper_evidence_summary.json")
    e108 = load_json("e108_deadzone_transfer_threshold_selection_audit.json")
    e109 = load_json("e109_deadzone014_holdout_confirmation_audit.json")
    e117 = load_json("e117_deadzone_external_fixed_protocol_audit.json")

    beta_by_split = {row["split"]: row for row in beta["rows"]}
    split_sources = [
        {
            "paper_split": "OpenImages transfer start8192",
            "deadzone_split": "transfer_start8192",
            "protocol_role": "threshold-selection split",
            "rows": e108["thresholds"],
        },
        {
            "paper_split": "OpenImages trusted holdout4096",
            "deadzone_split": "holdout4096",
            "protocol_role": "holdout confirmation split",
            "rows": e109["thresholds"],
        },
        {
            "paper_split": "Kodak",
            "deadzone_split": "kodak",
            "protocol_role": "external fixed-protocol split",
            "rows": e117["split_threshold_rows"],
        },
        {
            "paper_split": "CLIC mobile valid",
            "deadzone_split": "clic_mobile_valid",
            "protocol_role": "external fixed-protocol split",
            "rows": e117["split_threshold_rows"],
        },
        {
            "paper_split": "CLIC professional valid",
            "deadzone_split": "clic_professional_valid",
            "protocol_role": "external fixed-protocol split",
            "rows": e117["split_threshold_rows"],
        },
    ]

    table_rows: list[dict[str, object]] = []
    for source in split_sources:
        beta_row = beta_by_split[source["paper_split"]]
        for threshold in THRESHOLDS:
            if "split" in source["rows"][0]:
                dz_row = find(source["rows"], split=source["deadzone_split"], threshold=threshold)
            else:
                dz_row = find(source["rows"], threshold=threshold)
            dz_vs_beta = float(dz_row["mean_delta"])
            table_rows.append(
                {
                    "split": source["paper_split"],
                    "protocol_role": source["protocol_role"],
                    "threshold": threshold,
                    "beta005_rd": beta_row["beta"],
                    "beta005_vs_hcs": beta_row["beta_vs_hcs"],
                    "deadzone_rd": dz_row["mean_rd"],
                    "deadzone_vs_beta005": dz_vs_beta,
                    "deadzone_vs_hcs": float(beta_row["beta_vs_hcs"]) + dz_vs_beta,
                    "win_rate_vs_beta005": dz_row["win_rate"],
                    "q95_delta_vs_beta005": dz_row["q95_delta"],
                    "nonfinite_rows": dz_row["nonfinite_rows"],
                    "source": "E108/E109/E117",
                }
            )

    threshold_summary: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        rows = [row for row in table_rows if row["threshold"] == threshold]
        threshold_summary.append(
            {
                "threshold": threshold,
                "num_splits": len(rows),
                "all_splits_improve_beta005": all(float(row["deadzone_vs_beta005"]) < 0.0 for row in rows),
                "mean_split_delta_vs_beta005": mean(float(row["deadzone_vs_beta005"]) for row in rows),
                "worst_split_delta_vs_beta005": max(float(row["deadzone_vs_beta005"]) for row in rows),
                "mean_win_rate": mean(float(row["win_rate_vs_beta005"]) for row in rows),
                "worst_q95_delta": max(float(row["q95_delta_vs_beta005"]) for row in rows),
                "nonfinite_rows": sum(int(row["nonfinite_rows"]) for row in rows),
            }
        )

    dz014 = find(threshold_summary, threshold="014")
    dz018 = find(threshold_summary, threshold="018")
    decision = {
        "main_branch": "deadzone014",
        "safety_ablation": "deadzone018",
        "paper_baseline_context": "beta005 guard remains the broad historical prototype baseline, but dz014/dz018 are now externally confirmed stronger controller branches.",
        "dz014_mean_split_delta_vs_beta005": dz014["mean_split_delta_vs_beta005"],
        "dz018_mean_split_delta_vs_beta005": dz018["mean_split_delta_vs_beta005"],
        "dz014_worst_q95": dz014["worst_q95_delta"],
        "dz018_worst_q95": dz018["worst_q95_delta"],
        "interpretation": "dz014 is the mean-RD headline, while dz018 is the lower-tail-risk ablation.",
    }

    payload = {
        "decision": decision,
        "threshold_summary": threshold_summary,
        "table_rows": table_rows,
        "sources": {
            "beta005": "beta005_paper_evidence_summary.json",
            "transfer_selection": "e108_deadzone_transfer_threshold_selection_audit.json",
            "holdout_confirmation": "e109_deadzone014_holdout_confirmation_audit.json",
            "external_confirmation": "e117_deadzone_external_fixed_protocol_audit.json",
        },
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".table.csv"), table_rows)
    write_csv(OUT_PREFIX.with_suffix(".threshold_summary.csv"), threshold_summary)

    lines = [
        "# E118 HCG-RVQ Prototype Main Table Package",
        "",
        "This package freezes the current prototype manuscript table after E117 external confirmation.",
        "",
        "## Decision",
        "",
        f"- Main HCG-RVQ branch: `{decision['main_branch']}`",
        f"- Safety ablation: `{decision['safety_ablation']}`",
        f"- Interpretation: {decision['interpretation']}",
        "",
        "## Threshold Summary",
        "",
        "| threshold | splits | all improve beta005 | mean delta | worst split delta | mean win | worst q95 | nonfinite |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_summary:
        lines.append(
            "| dz{threshold} | {splits} | {all_ok} | {mean_delta} | {worst_delta} | {win} | {q95} | {nonfinite} |".format(
                threshold=row["threshold"],
                splits=int(row["num_splits"]),
                all_ok=row["all_splits_improve_beta005"],
                mean_delta=fmt(row["mean_split_delta_vs_beta005"], signed=True),
                worst_delta=fmt(row["worst_split_delta_vs_beta005"], signed=True),
                win=fmt(row["mean_win_rate"]),
                q95=fmt(row["worst_q95_delta"], signed=True),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## Prototype Table",
            "",
            "| split | role | threshold | beta005 RD | dz RD | dz-beta005 | dz-HCS | win | q95 | nonfinite |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table_rows:
        lines.append(
            "| {split} | {role} | dz{threshold} | {beta_rd} | {dz_rd} | {dz_beta} | {dz_hcs} | {win} | {q95} | {nonfinite} |".format(
                split=row["split"],
                role=row["protocol_role"],
                threshold=row["threshold"],
                beta_rd=fmt(row["beta005_rd"]),
                dz_rd=fmt(row["deadzone_rd"]),
                dz_beta=fmt(row["deadzone_vs_beta005"], signed=True),
                dz_hcs=fmt(row["deadzone_vs_hcs"], signed=True),
                win=fmt(row["win_rate_vs_beta005"]),
                q95=fmt(row["q95_delta_vs_beta005"], signed=True),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paper Use",
            "",
            "- Use dz014 as the main mean-RD row for the current MeanScaleHyperprior/RVQ prototype.",
            "- Keep dz018 as a conservative reliability/tail ablation.",
            "- Keep beta005 as the earlier broad guard baseline and initialization context.",
            "- Keep max500 out of the main row until its checkpoint policy or conditional reliability control is fixed.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_suffix('.table.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.threshold_summary.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
