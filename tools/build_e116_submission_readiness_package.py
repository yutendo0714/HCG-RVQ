#!/usr/bin/env python3
"""Build a manuscript-readiness package for the current HCG-RVQ evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e116_hcg_rvq_submission_readiness_package"


def load_json(name: str) -> dict:
    with (ANALYSIS / name).open(encoding="utf-8") as f:
        return json.load(f)


def fmt(value: object, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def find(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    e110 = load_json("e110_residual_selector_max500_checkpoint_audit.json")
    e111 = load_json("e111_residual_selector_max500_transfer_checkpoint_audit.json")
    e113 = load_json("e113_max500_selector_cap_multiseed_audit.json")
    e114 = load_json("e114_max500_per_image_cap_headroom.json")
    e115 = load_json("e115_max500_learned_cap_selector_cv.json")

    beta_rows = beta["rows"]
    beta_min_vs_hcs = max(float(row["beta_vs_hcs"]) for row in beta_rows)
    beta_max_nonfinite = max(int(row["nonfinite"]) for row in beta_rows)

    transfer014 = find(e108["thresholds"], threshold="014")
    transfer018 = find(e108["thresholds"], threshold="018")
    holdout014 = find(e109["thresholds"], threshold="014")
    holdout018 = find(e109["thresholds"], threshold="018")
    holdout_max500_018 = find(e110["budget_threshold"], budget="max500", threshold="018")
    transfer_max500_018 = find(e111["budget_threshold"], budget="max500", threshold="018")

    claim_rows = [
        {
            "claim_layer": "broad fixed-checkpoint prototype",
            "candidate": "beta005 guard",
            "evidence": "OpenImages holdout/transfer + Kodak + CLIC mobile/professional",
            "mean_delta": None,
            "best_split_delta": min(float(row["beta_vs_hcs"]) for row in beta_rows),
            "weakest_split_delta": beta_min_vs_hcs,
            "nonfinite": beta_max_nonfinite,
            "status": "paper-safe prototype baseline",
            "limitation": "does not include the newer dead-zone residual-selector gain",
        },
        {
            "claim_layer": "stronger HCG-RVQ branch",
            "candidate": "step250 dz014",
            "evidence": "transfer-selected threshold, holdout-confirmed",
            "transfer_delta": transfer014["mean_delta"],
            "holdout_delta": holdout014["mean_delta"],
            "win_rate_holdout": holdout014["win_rate"],
            "q95_holdout": holdout014["q95_delta"],
            "nonfinite": holdout014["nonfinite_rows"],
            "status": "current mean-RD manuscript candidate",
            "limitation": "needs external/Kodak/CLIC confirmation before replacing beta005 as broad headline",
        },
        {
            "claim_layer": "conservative safety ablation",
            "candidate": "step250 dz018",
            "evidence": "near-identical transfer/holdout gain, lower q95 damage",
            "transfer_delta": transfer018["mean_delta"],
            "holdout_delta": holdout018["mean_delta"],
            "win_rate_holdout": holdout018["win_rate"],
            "q95_holdout": holdout018["q95_delta"],
            "nonfinite": holdout018["nonfinite_rows"],
            "status": "appendix/conservative ablation",
            "limitation": "slightly weaker mean RD than dz014",
        },
        {
            "claim_layer": "high-mean headroom",
            "candidate": "max500 dz018",
            "evidence": "holdout and transfer checkpoint audit",
            "transfer_delta": transfer_max500_018["mean_delta"],
            "holdout_delta": holdout_max500_018["mean_delta"],
            "win_rate_holdout": holdout_max500_018["win_rate"],
            "q95_holdout": holdout_max500_018["q95_delta"],
            "nonfinite": holdout_max500_018["nonfinite_rows"],
            "status": "not paper-main yet",
            "limitation": "seed3456 regression and larger q95 tail",
        },
    ]

    selector_rows = [
        {
            "question": "Can a global deploy-time cap make max500 clean?",
            "result": "No",
            "metric": "E113 transfer-selected cap",
            "value": e113["decision"]["best_transfer_cap"],
            "delta": e113["decision"]["best_transfer_delta"],
            "interpretation": "3-seed transfer selects cap0.50; lowering cap helps seed3456 but loses seed1234/2345 gain.",
        },
        {
            "question": "Is there per-image cap headroom?",
            "result": "Yes, modest",
            "metric": "E114 oracle gain vs cap0.50",
            "value": e114["oracle_gain_vs_cap050"],
            "delta": e114["per_image_oracle"]["mean_delta"],
            "interpretation": "Oracle fixes seed3456 sign, but this is not deployable.",
        },
        {
            "question": "Does a learned multi-feature selector generalize?",
            "result": "Slightly",
            "metric": "E115 leave-one-seed gain vs cap0.50",
            "value": e115["learned_cv_gain_vs_cap050"],
            "delta": e115["learned_cv"]["mean_delta"],
            "interpretation": "Learned selector improves transfer mean slightly but recovers too little oracle headroom.",
        },
    ]

    readiness = {
        "overall": "promising_but_not_submission_complete",
        "short_answer": (
            "The research is progressing well for a serious conference submission: the safe prototype claim is broad, "
            "and the newer dead-zone branch is protocol-clean on OpenImages. It is not final yet because the stronger "
            "branch still needs external split confirmation and SOTA/backbone comparisons."
        ),
        "green": [
            "beta005 fixed-checkpoint prototype improves HCS/old/min090 across OpenImages, Kodak, and CLIC-style splits",
            "step250 dz014 is selected on independent transfer and confirmed on holdout with zero nonfinite rows",
            "intermediate qMSE, s_q, dead-code, and Householder statistics stay controlled for dz014/dz018",
        ],
        "yellow": [
            "max500 has larger mean gain but seed3456/q95 tail risk prevents paper-main promotion",
            "learned cap selection shows signal but not enough recovered oracle headroom",
            "dead-zone branch needs external/Kodak/CLIC confirmation before becoming the broad headline row",
        ],
        "red_or_missing": [
            "SOTA/backbone plug-in is not yet executed",
            "final manuscript tables need compact ablation formatting and broader competitor baselines",
        ],
        "next_actions": [
            "Evaluate step250 dz014/dz018 on Kodak and CLIC fixed protocol against beta005 references",
            "Freeze the paper table order: beta005 broad prototype, dz014 stronger OpenImages branch, dz018 safety ablation, max500 headroom",
            "Only after those tables are frozen, start SOTA-backbone plug-in experiments",
        ],
    }

    payload = {
        "readiness": readiness,
        "claim_rows": claim_rows,
        "selector_rows": selector_rows,
        "sources": {
            "beta005": "beta005_paper_evidence_summary.json",
            "threshold_selection": "e108_deadzone_transfer_threshold_selection_audit.json",
            "holdout_confirmation": "e109_deadzone014_holdout_confirmation_audit.json",
            "checkpoint_holdout": "e110_residual_selector_max500_checkpoint_audit.json",
            "checkpoint_transfer": "e111_residual_selector_max500_transfer_checkpoint_audit.json",
            "cap_global": "e113_max500_selector_cap_multiseed_audit.json",
            "cap_oracle": "e114_max500_per_image_cap_headroom.json",
            "cap_learned": "e115_max500_learned_cap_selector_cv.json",
        },
    }

    md = ["# E116 HCG-RVQ Submission Readiness Package", ""]
    md.append("## Readiness")
    md.append("")
    md.append(readiness["short_answer"])
    md.append("")
    md.append("## Claim Stack")
    md.append("")
    md.append("| layer | candidate | evidence | transfer delta | holdout delta | win/q95 holdout | nonfinite | status | limitation |")
    md.append("|---|---|---|---:|---:|---|---:|---|---|")
    for row in claim_rows:
        win_q95 = "n/a"
        if "win_rate_holdout" in row:
            win_q95 = f"{fmt(row['win_rate_holdout'])} / {fmt(row['q95_holdout'], signed=True)}"
        md.append(
            "| {layer} | {candidate} | {evidence} | {transfer} | {holdout} | {win_q95} | {nonfinite} | {status} | {limitation} |".format(
                layer=row["claim_layer"],
                candidate=row["candidate"],
                evidence=row["evidence"],
                transfer=fmt(row.get("transfer_delta"), signed=True),
                holdout=fmt(row.get("holdout_delta"), signed=True),
                win_q95=win_q95,
                nonfinite=row["nonfinite"],
                status=row["status"],
                limitation=row["limitation"],
            )
        )
    md.append("")
    md.append("## Max500 Selector Controls")
    md.append("")
    md.append("| question | result | metric | value | selected/mean delta | interpretation |")
    md.append("|---|---|---|---:|---:|---|")
    for row in selector_rows:
        md.append(
            f"| {row['question']} | {row['result']} | {row['metric']} | "
            f"{fmt(row['value'])} | {fmt(row['delta'], signed=True)} | {row['interpretation']} |"
        )
    md.append("")
    md.append("## Current Answer")
    md.append("")
    md.append("- Yes, the research is going well enough to justify continuing toward an international submission.")
    md.append("- The safest current story is broad beta005 evidence plus the newer dz014 dead-zone controller as the stronger OpenImages branch.")
    md.append("- The strongest missing evidence is external confirmation of dz014/dz018 and later SOTA/backbone plug-in.")
    md.append("")
    md.append("## Next Actions")
    md.append("")
    for item in readiness["next_actions"]:
        md.append(f"- {item}")
    md.append("")

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    write_csv(OUT_PREFIX.with_suffix(".claim_rows.csv"), claim_rows)
    write_csv(OUT_PREFIX.with_suffix(".selector_rows.csv"), selector_rows)
    print(OUT_PREFIX.with_suffix(".md"))
    print(OUT_PREFIX.with_suffix(".json"))


if __name__ == "__main__":
    main()
