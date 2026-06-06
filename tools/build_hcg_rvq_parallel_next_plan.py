#!/usr/bin/env python3
"""Build the parallel paper-claim and method-improvement plan for HCG-RVQ."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"


def load_json(name: str) -> dict:
    with (ANALYSIS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float | int | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def optional_json(name: str) -> dict | None:
    path = ANALYSIS / name
    if not path.exists():
        return None
    return load_json(name)


def main() -> None:
    tables = load_json("beta005_paper_tables.json")
    claim = load_json("beta005_paper_claim_matrix.json")
    selector = load_json("excessrisk090_prevlocal_selector_headroom_holdout4096.json")
    beta003 = optional_json("excessrisk090_local_cap080_rho1_betacommit003_after250_holdout4096_checkpoint_sweep.json")
    beta007 = optional_json("excessrisk090_local_cap080_rho1_betacommit007_after250_holdout4096_checkpoint_sweep.json")
    beta007_kodak = optional_json("excessrisk090_local_cap080_rho1_betacommit007_after250_kodak_checkpoint_sweep.json")

    evidence_rows = tables["cross_split_evidence"]
    beta_hcs = [float(row["beta_minus_hcs"]) for row in evidence_rows]
    beta_old = [float(row["beta_minus_old"]) for row in evidence_rows]
    beta_min090 = [float(row["beta_minus_min090"]) for row in evidence_rows]
    nonfinite = [int(row["nonfinite_rows"]) for row in evidence_rows]

    best_threshold = selector["top_thresholds"][0]
    oracle = selector["oracle_previous_local_or_excess500"]
    beta005_seed3456 = 2.156416
    beta_commit_notes = []
    if beta003 is not None:
        beta_commit_notes.append(
            {
                "setting": "beta_commit 0.03 after step250, seed3456 holdout",
                "rd": beta003["aggregate"]["step500_rd"],
                "delta_vs_beta005_seed3456": beta003["aggregate"]["step500_rd"] - beta005_seed3456,
                "decision": "weaker than beta005 on the fragile OpenImages seed",
            }
        )
    if beta007 is not None:
        beta_commit_notes.append(
            {
                "setting": "beta_commit 0.07 after step250, seed3456 holdout",
                "rd": beta007["aggregate"]["step500_rd"],
                "delta_vs_beta005_seed3456": beta007["aggregate"]["step500_rd"] - beta005_seed3456,
                "decision": "not enough to replace beta005 without full 3-seed evidence",
            }
        )
    if beta007_kodak is not None:
        beta_commit_notes.append(
            {
                "setting": "beta_commit 0.07 after step250, seed3456 Kodak",
                "rd": beta007_kodak["aggregate"]["step500_rd"],
                "delta_vs_beta005_seed3456": beta007_kodak["aggregate"]["step500_rd"] - 2.106310,
                "decision": "interesting transfer signal, but single-seed and not paper-main",
            }
        )

    tracks = [
        {
            "track": "paper claim hardening",
            "status": "active and mostly frozen for the prototype stage",
            "purpose": "Make the current HCG-RVQ claim safe: fixed checkpoints, aligned splits, no selector/oracle headline.",
            "evidence": (
                "Beta005 improves HCS, old gate0.25, and min090 on all five aligned rows; "
                f"worst beta-HCS is {fmt(max(beta_hcs), signed=True)} and total nonfinite rows are {sum(nonfinite)}."
            ),
            "next_action": "Use beta005 tables as the manuscript prototype result and keep guardrails in appendix.",
        },
        {
            "track": "method improvement",
            "status": "active, but should be narrower than broad sweeps",
            "purpose": "Turn the diagnostic headroom into a stronger single HCG-RVQ method.",
            "evidence": (
                f"The previous-local/excess500 oracle reaches {fmt(oracle['rd'])} RD "
                f"({fmt(oracle['delta_vs_hcs'], signed=True)} vs HCS), and a simple delta-RMS threshold reaches "
                f"{fmt(best_threshold['rd'])} RD ({fmt(best_threshold['delta_vs_hcs'], signed=True)} vs HCS)."
            ),
            "next_action": "Design one single-checkpoint reliability controller instead of switching checkpoints per image.",
        },
        {
            "track": "strong-backbone/SOTA plug-in",
            "status": "next-stage, not yet the highest GPU priority",
            "purpose": "Show that the quantizer-geometry idea also helps a stronger LIC/GIC backbone.",
            "evidence": "The prototype table is now strong enough to justify an interface audit, but not a SOTA-dominance claim.",
            "next_action": "Start with official-baseline/repository feasibility and one chosen backbone after the prototype table is locked.",
        },
    ]

    priorities = [
        {
            "rank": 1,
            "name": "Freeze paper-facing beta005 prototype tables",
            "type": "paper claim",
            "why": "The five aligned rows already support the controlled MeanScaleHyperprior/RVQ claim.",
            "risk": "Low; avoid overclaiming SOTA.",
            "gpu": "none",
        },
        {
            "rank": 2,
            "name": "Single-checkpoint reliability controller from selector headroom",
            "type": "method improvement",
            "why": "It directly targets the measured gap between beta005 and the local hard-tail specialist without leaving the prompt thesis.",
            "risk": "Medium; must not become a per-image checkpoint selector.",
            "gpu": "focused 1-seed probe first, then 3-seed if it beats beta005 on both average and hard tail.",
        },
        {
            "rank": 3,
            "name": "Strong-backbone plug-in feasibility audit",
            "type": "SOTA bridge",
            "why": "International-conference positioning eventually needs official baselines or a plug-in demonstration.",
            "risk": "Medium-high; external code can confound novelty and consume GPU time.",
            "gpu": "start with code/protocol audit, then one small plug-in experiment.",
        },
        {
            "rank": 4,
            "name": "Validation-selected beta_commit boundary check",
            "type": "method improvement",
            "why": "0.07 has a small single-seed Kodak signal, but 0.05 is better on the fragile OpenImages seed.",
            "risk": "Medium; easy to overfit a scalar knob.",
            "gpu": "only after defining an independent validation criterion.",
        },
        {
            "rank": 5,
            "name": "Stage context/gating and index-prior expansion",
            "type": "prompt extension",
            "why": "Aligned with the full HCG-RVQ spec, but less urgent than stabilizing geometry.",
            "risk": "Medium; adds moving parts before the geometry claim is completely locked.",
            "gpu": "defer until prototype paper table and controller result are stable.",
        },
    ]

    payload = {
        "prompt_anchor": (
            "Hyperprior should not only predict entropy parameters; it should also generate the local geometry of the quantizer."
        ),
        "summary": {
            "num_aligned_splits": len(evidence_rows),
            "all_beta_improve_hcs": all(v < 0 for v in beta_hcs),
            "all_beta_improve_old": all(v < 0 for v in beta_old),
            "all_beta_improve_min090": all(v < 0 for v in beta_min090),
            "worst_beta_hcs": max(beta_hcs),
            "best_beta_hcs": min(beta_hcs),
            "total_nonfinite_rows": sum(nonfinite),
        },
        "tracks": tracks,
        "priorities": priorities,
        "beta_commit_boundary": beta_commit_notes,
        "decision": (
            "Proceed in parallel, but not with equal GPU allocation at every moment: keep paper-claim "
            "tables frozen while the next method-improvement GPU run focuses on a single-controller reliability design."
        ),
    }

    md = ["# HCG-RVQ Parallel Strategy and Next Experiments", ""]
    md.append("## Prompt Anchor")
    md.append("")
    md.append(payload["prompt_anchor"])
    md.append("")
    md.append("## Current Evidence State")
    md.append("")
    md.append(
        f"Beta005 has {payload['summary']['num_aligned_splits']} aligned fixed-checkpoint rows. "
        f"It improves HCS on every row; worst beta-HCS is {fmt(payload['summary']['worst_beta_hcs'], signed=True)} "
        f"and best beta-HCS is {fmt(payload['summary']['best_beta_hcs'], signed=True)}. "
        f"Total nonfinite rows: {payload['summary']['total_nonfinite_rows']}."
    )
    md.append("")
    md.append("| split | beta-HCS | beta-old | beta-min090 | nonfinite |")
    md.append("|---|---:|---:|---:|---:|")
    for row in evidence_rows:
        md.append(
            "| {split} | {hcs} | {old} | {min090} | {nonfinite} |".format(
                split=row["split"],
                hcs=fmt(row["beta_minus_hcs"], signed=True),
                old=fmt(row["beta_minus_old"], signed=True),
                min090=fmt(row["beta_minus_min090"], signed=True),
                nonfinite=row["nonfinite_rows"],
            )
        )
    md.append("")
    md.append("## Parallel Tracks")
    md.append("")
    md.append("| track | status | purpose | next action |")
    md.append("|---|---|---|---|")
    for row in tracks:
        md.append(f"| {row['track']} | {row['status']} | {row['purpose']} {row['evidence']} | {row['next_action']} |")
    md.append("")
    md.append("## Next Experiment Priority")
    md.append("")
    md.append("| rank | experiment | type | why | GPU policy |")
    md.append("|---:|---|---|---|---|")
    for row in priorities:
        md.append(f"| {row['rank']} | {row['name']} | {row['type']} | {row['why']} Risk: {row['risk']} | {row['gpu']} |")
    md.append("")
    md.append("## Beta-Commit Boundary")
    md.append("")
    if beta_commit_notes:
        md.append("| setting | RD | delta vs reference | decision |")
        md.append("|---|---:|---:|---|")
        for row in beta_commit_notes:
            md.append(
                f"| {row['setting']} | {fmt(row['rd'])} | {fmt(row['delta_vs_beta005_seed3456'], signed=True)} | {row['decision']} |"
            )
    else:
        md.append("No beta-commit boundary artifacts were available.")
    md.append("")
    md.append("## Decision")
    md.append("")
    md.append(payload["decision"])
    md.append("")
    md.append("The immediate research move is therefore not a broad SOTA clone or a random hyperparameter sweep. It is a narrow, single-checkpoint reliability-control experiment, while the official/SOTA backbone work starts as an interface and protocol audit.")
    md.append("")

    (ANALYSIS / "hcg_rvq_parallel_next_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "hcg_rvq_parallel_next_plan.md").write_text("\n".join(md), encoding="utf-8")
    print(ANALYSIS / "hcg_rvq_parallel_next_plan.md")
    print(ANALYSIS / "hcg_rvq_parallel_next_plan.json")


if __name__ == "__main__":
    main()
