#!/usr/bin/env python3
"""Build manuscript-oriented tables for the beta005 HCG-RVQ candidate."""

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


def main() -> None:
    holdout = load_json("excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.json")
    cap060 = load_json("excessrisk090_local_cap080to060_rho1_tail_holdout4096.json")
    posthoc = load_json("posthoc_single_checkpoint_controller_val4096_holdout4096_current.json")
    min095 = load_json("risk_floor_min095_val4096_holdout4096_current.json")
    delta010 = load_json("delta_reg010_seed3456_val4096_holdout4096_current.json")
    claim = load_json("beta005_paper_claim_matrix.json")

    summaries = holdout["summaries"]
    quartiles = {row["quartile"]: row for row in holdout["quartiles_by_hcs_difficulty"]}
    features = holdout["feature_summaries"]

    main_ablation = [
        {
            "row": "HCS-RVQ",
            "role": "shift/scale RVQ baseline",
            "rd": summaries["hcs"]["rd"],
            "delta_vs_hcs": 0.0,
            "q4_delta_vs_hcs": 0.0,
            "delta_rms": None,
            "qmse": None,
            "paper_use": "baseline",
        },
        {
            "row": "HCG old gate0.25",
            "role": "adds Householder geometry",
            "rd": summaries["old"]["rd"],
            "delta_vs_hcs": summaries["old"]["delta_vs_hcs"],
            "q4_delta_vs_hcs": quartiles["Q4"]["old_minus_hcs"],
            "delta_rms": features["old"]["rvq_householder_delta_rms"],
            "qmse": features["old"]["rvq_latent_quant_mse"],
            "paper_use": "geometry ablation",
        },
        {
            "row": "HCG min090 risk",
            "role": "risk-suppressed geometry",
            "rd": summaries["min090"]["rd"],
            "delta_vs_hcs": summaries["min090"]["delta_vs_hcs"],
            "q4_delta_vs_hcs": quartiles["Q4"]["min090_minus_hcs"],
            "delta_rms": features["min090"]["rvq_householder_delta_rms"],
            "qmse": features["min090"]["rvq_latent_quant_mse"],
            "paper_use": "reliability ablation",
        },
        {
            "row": "local cap080/rho1 step250",
            "role": "hard-tail specialist",
            "rd": summaries["previous_local"]["rd"],
            "delta_vs_hcs": summaries["previous_local"]["delta_vs_hcs"],
            "q4_delta_vs_hcs": quartiles["Q4"]["previous_local_minus_hcs"],
            "delta_rms": features["previous_local"]["rvq_householder_delta_rms"],
            "qmse": features["previous_local"]["rvq_latent_quant_mse"],
            "paper_use": "mechanism, not main codec",
        },
        {
            "row": "beta005 guard step500",
            "role": "fixed-checkpoint stabilized HCG",
            "rd": summaries["variant500"]["rd"],
            "delta_vs_hcs": summaries["variant500"]["delta_vs_hcs"],
            "q4_delta_vs_hcs": quartiles["Q4"]["variant500_minus_hcs"],
            "delta_rms": features["variant500"]["rvq_householder_delta_rms"],
            "qmse": features["variant500"]["rvq_latent_quant_mse"],
            "paper_use": "paper-main prototype row",
        },
    ]

    guardrails = [
        {
            "row": "posthoc min090 on old weights",
            "scope": "3-seed holdout4096",
            "delta_vs_hcs": posthoc["overall"]["posthoc_delta_vs_hcs"],
            "why_rejected": "pure inference-time risk switching collapses RD and dead-code use",
        },
        {
            "row": "trained min095 risk floor",
            "scope": "3-seed holdout4096",
            "delta_vs_hcs": min095["overall"]["trained_min095_delta_vs_hcs"],
            "why_rejected": "too-high risk floor drives high qMSE and broad degradation",
        },
        {
            "row": "delta_reg010",
            "scope": "seed3456 holdout4096",
            "delta_vs_hcs": delta010["methods"]["delta_reg010"]["delta_vs_hcs"],
            "why_rejected": "directly shrinking delta RMS breaks codebook/latent usage",
        },
        {
            "row": "cap080-to-cap060 step500",
            "scope": "3-seed holdout4096",
            "delta_vs_hcs": cap060["summaries"]["variant500"]["delta_vs_hcs"],
            "why_rejected": "smaller local cap loses the beta005 average and hard-tail balance",
        },
    ]

    payload = {
        "main_ablation": main_ablation,
        "cross_split_evidence": claim["evidence_rows"],
        "guardrails": guardrails,
        "recommended_caption": (
            "Ablation of hyperprior-conditioned quantizer geometry. "
            "Beta005 uses the same fixed checkpoint rule across splits and is not selected per image or per seed."
        ),
    }

    md = ["# Beta005 Paper Tables", ""]
    md.append("## Main Ablation Table Candidate")
    md.append("")
    md.append("| row | role | RD | vs HCS | Q4 vs HCS | delta RMS | qMSE | paper use |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in main_ablation:
        md.append(
            "| {row} | {role} | {rd} | {delta} | {q4} | {delta_rms} | {qmse} | {paper_use} |".format(
                row=row["row"],
                role=row["role"],
                rd=fmt(row["rd"]),
                delta=fmt(row["delta_vs_hcs"], signed=True),
                q4=fmt(row["q4_delta_vs_hcs"], signed=True),
                delta_rms=fmt(row["delta_rms"]),
                qmse=fmt(row["qmse"]),
                paper_use=row["paper_use"],
            )
        )
    md.append("")
    md.append("## Cross-Split Fixed-Checkpoint Evidence")
    md.append("")
    md.append("| split | images/seed | beta-HCS | beta-old | beta-min090 | nonfinite |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for row in claim["evidence_rows"]:
        md.append(
            "| {split} | {n} | {hcs} | {old} | {min090} | {nonfinite} |".format(
                split=row["split"],
                n=row["images_per_seed"],
                hcs=fmt(row["beta_minus_hcs"], signed=True),
                old=fmt(row["beta_minus_old"], signed=True),
                min090=fmt(row["beta_minus_min090"], signed=True),
                nonfinite=row["nonfinite_rows"],
            )
        )
    md.append("")
    md.append("## Appendix Guardrail Table Candidate")
    md.append("")
    md.append("| row | scope | vs HCS | why rejected as paper-main |")
    md.append("|---|---|---:|---|")
    for row in guardrails:
        md.append(
            f"| {row['row']} | {row['scope']} | {fmt(row['delta_vs_hcs'], signed=True)} | {row['why_rejected']} |"
        )
    md.append("")
    md.append("## Caption")
    md.append("")
    md.append(payload["recommended_caption"])
    md.append("")

    (ANALYSIS / "beta005_paper_tables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "beta005_paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(ANALYSIS / "beta005_paper_tables.md")
    print(ANALYSIS / "beta005_paper_tables.json")


if __name__ == "__main__":
    main()
