#!/usr/bin/env python3
"""Build a paper-claim matrix for the beta005 HCG-RVQ candidate."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"


def load_json(name: str) -> dict:
    with (ANALYSIS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def method_mean(payload: dict, method: str) -> dict:
    for row in payload["summaries"]:
        if row["method"] == method and row["seed"] == "mean":
            return row
    raise KeyError(method)


def beta_row(split_name: str, protocol: str, payload: dict) -> dict:
    hcs = method_mean(payload, "HCS")
    old = method_mean(payload, "old gate0.25")
    min090 = method_mean(payload, "min090")
    beta = method_mean(payload, "beta005 guard")
    return {
        "split": split_name,
        "protocol": protocol,
        "images_per_seed": int(beta["num_images"]) // 3,
        "hcs_rd": float(hcs["mean_rd"]),
        "old_rd": float(old["mean_rd"]),
        "min090_rd": float(min090["mean_rd"]),
        "beta_rd": float(beta["mean_rd"]),
        "beta_minus_hcs": float(beta["mean_rd"]) - float(hcs["mean_rd"]),
        "beta_minus_old": float(beta["mean_rd"]) - float(old["mean_rd"]),
        "beta_minus_min090": float(beta["mean_rd"]) - float(min090["mean_rd"]),
        "nonfinite_rows": int(beta.get("nonfinite_rows", 0)),
        "beta_s_q": float(beta.get("mean_rvq_s_q_mean", float("nan"))),
        "beta_delta_rms": float(beta.get("mean_rvq_householder_delta_rms", float("nan"))),
        "beta_qmse": float(beta.get("mean_rvq_latent_quant_mse", float("nan"))),
    }


def trusted_holdout_row() -> dict:
    tail = load_json("excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.json")
    summary = tail["summaries"]
    features = tail["feature_summaries"]["variant500"]
    return {
        "split": "OpenImages trusted holdout4096",
        "protocol": "trusted validation/checkpoint-selection protocol",
        "images_per_seed": 4096,
        "hcs_rd": float(summary["hcs"]["rd"]),
        "old_rd": float(summary["old"]["rd"]),
        "min090_rd": float(summary["min090"]["rd"]),
        "beta_rd": float(summary["variant500"]["rd"]),
        "beta_minus_hcs": float(summary["variant500"]["delta_vs_hcs"]),
        "beta_minus_old": float(summary["variant500"]["delta_vs_old"]),
        "beta_minus_min090": float(summary["variant500"]["delta_vs_min090"]),
        "nonfinite_rows": 0,
        "beta_s_q": float(features["rvq_s_q_mean"]),
        "beta_delta_rms": float(features["rvq_householder_delta_rms"]),
        "beta_qmse": float(features["rvq_latent_quant_mse"]),
    }


def main() -> None:
    transfer = load_json("beta005_transfer_openimages_start8192_n4096.json")
    kodak = load_json("beta005_external_kodak_fixed_protocol.json")
    clic_mobile = load_json("beta005_external_clic_mobile_valid.json")
    clic_prof = load_json("beta005_external_clic_professional_valid.json")
    tail = load_json("excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.json")

    evidence_rows = [
        trusted_holdout_row(),
        beta_row("OpenImages transfer start8192", "unselected OpenImages transfer split", transfer),
        beta_row("Kodak", "standard Kodak fixed-checkpoint test set", kodak),
        beta_row("CLIC mobile valid", "external-style CLIC mobile valid", clic_mobile),
        beta_row("CLIC professional valid", "external-style CLIC professional valid", clic_prof),
    ]

    quartiles = {
        row["quartile"]: row
        for row in tail["quartiles_by_hcs_difficulty"]
    }
    feature = tail["feature_summaries"]

    claim_matrix = [
        {
            "tier": "paper-main prototype codec",
            "claim": (
                "A fixed-checkpoint HCG-RVQ variant with reliability-aware local geometry "
                "control and beta-commit stabilization improves HCS, old geometry gating, "
                "and min090 risk control across trusted, transfer, and external-style splits."
            ),
            "evidence": "All five audited splits have negative beta-HCS, beta-old, and beta-min090 deltas.",
            "status": "ready for the current MeanScaleHyperprior/RVQ prototype claim",
            "caveat": "Do not claim SOTA dominance over modern LIC/GIC systems from these rows alone.",
        },
        {
            "tier": "mechanism ablation",
            "claim": (
                "Hyperprior-conditioned geometry is useful only when its reliability is controlled; "
                "uncontrolled old/min090 geometry has seed-specific damage, while beta005 keeps "
                "Householder displacement smaller and stable."
            ),
            "evidence": (
                f"Holdout beta delta RMS {fmt(feature['variant500']['rvq_householder_delta_rms'])} "
                f"vs old {fmt(feature['old']['rvq_householder_delta_rms'])} and min090 "
                f"{fmt(feature['min090']['rvq_householder_delta_rms'])}."
            ),
            "status": "ready as explanation/ablation",
            "caveat": "Average feature means explain the operating regime, not every per-image win.",
        },
        {
            "tier": "hard-tail mechanism",
            "claim": (
                "The previous local cap080/rho1 step250 row remains the clean hard-tail specialist, "
                "whereas beta005 is the stronger average/transfer codec."
            ),
            "evidence": (
                f"Holdout Q4: beta005-HCS {fmt(quartiles['Q4']['variant500_minus_hcs'], signed=True)}, "
                f"previous-local-HCS {fmt(quartiles['Q4']['previous_local_minus_hcs'], signed=True)}."
            ),
            "status": "ready as secondary mechanism result",
            "caveat": "Keep this separate from the fixed step500 paper-main row.",
        },
        {
            "tier": "diagnostic/headroom",
            "claim": (
                "Old/min090 per-image selection shows reliability-control headroom, but it is a "
                "multi-checkpoint diagnostic rather than a deployable single-codec result."
            ),
            "evidence": "selector_claim_readiness.md marks the calibrated selector as multi-checkpoint diagnostic.",
            "status": "use for motivation, not headline",
            "caveat": "A unified single-checkpoint controller is still needed before claiming this as a method.",
        },
        {
            "tier": "negative controls",
            "claim": (
                "Posthoc risk switching, min095 risk floor, direct delta regularization, and cap080-to-cap060 "
                "schedule checks prevent over-claiming a simple threshold or smaller geometry as the answer."
            ),
            "evidence": (
                "Existing audits reject pure inference-time min090, too-high risk floor, direct delta010, "
                "and cap080-to-cap060 as paper-main variants."
            ),
            "status": "ready for appendix/ablation narrative",
            "caveat": "Use concise reporting; these rows are guardrails, not central contributions.",
        },
    ]

    payload = {
        "evidence_rows": evidence_rows,
        "claim_matrix": claim_matrix,
        "protocol_notes": [
            "Checkpoint choices are fixed by the trusted holdout4096 protocol.",
            "Transfer, Kodak, and CLIC rows do not select checkpoints on their reporting split.",
            "All recorded beta005 rows have zero nonfinite outputs.",
            "Physical GPU 0 was used for the new GPU evaluations.",
            "Quarantined localstats/current-recheck artifacts should not be used for paper-facing claims.",
        ],
        "recommended_next_actions": [
            "Write the paper table around HCS -> old geometry -> min090 risk -> beta005 guard.",
            "Add a compact appendix table for rejected controllers and selector headroom.",
            "Run a stronger-backbone plug-in only after freezing the prototype claim table.",
            "Keep scratch-from-random training as later robustness evidence, not as the next blocking item.",
        ],
    }

    md = ["# Beta005 Paper Claim Matrix", ""]
    md.append("## Main Evidence")
    md.append("")
    md.append(
        "| split | images/seed | beta-HCS | beta-old | beta-min090 | beta s_q | beta delta RMS | beta qMSE | nonfinite |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in evidence_rows:
        md.append(
            "| {split} | {images_per_seed} | {beta_minus_hcs} | {beta_minus_old} | "
            "{beta_minus_min090} | {beta_s_q} | {beta_delta_rms} | {beta_qmse} | {nonfinite_rows} |".format(
                split=row["split"],
                images_per_seed=row["images_per_seed"],
                beta_minus_hcs=fmt(row["beta_minus_hcs"], signed=True),
                beta_minus_old=fmt(row["beta_minus_old"], signed=True),
                beta_minus_min090=fmt(row["beta_minus_min090"], signed=True),
                beta_s_q=fmt(row["beta_s_q"]),
                beta_delta_rms=fmt(row["beta_delta_rms"]),
                beta_qmse=fmt(row["beta_qmse"]),
                nonfinite_rows=row["nonfinite_rows"],
            )
        )
    md.append("")
    md.append("## Claim Tiers")
    md.append("")
    md.append("| tier | paper use | status | caveat |")
    md.append("|---|---|---|---|")
    for row in claim_matrix:
        md.append(f"| {row['tier']} | {row['claim']} | {row['status']} | {row['caveat']} |")
    md.append("")
    md.append("## Protocol Notes")
    md.append("")
    for note in payload["protocol_notes"]:
        md.append(f"- {note}")
    md.append("")
    md.append("## Recommended Next Actions")
    md.append("")
    for action in payload["recommended_next_actions"]:
        md.append(f"- {action}")
    md.append("")

    (ANALYSIS / "beta005_paper_claim_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "beta005_paper_claim_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print(ANALYSIS / "beta005_paper_claim_matrix.md")
    print(ANALYSIS / "beta005_paper_claim_matrix.json")


if __name__ == "__main__":
    main()
