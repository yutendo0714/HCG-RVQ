#!/usr/bin/env python3
"""Build a SOTA/backbone plug-in readiness audit for HCG-RVQ."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e119_sota_plugin_readiness_audit"


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


def main() -> None:
    e118 = load_json("e118_hcg_rvq_prototype_main_table_package.json")
    dz014 = next(row for row in e118["threshold_summary"] if row["threshold"] == "014")
    dz018 = next(row for row in e118["threshold_summary"] if row["threshold"] == "018")

    readiness_rows = [
        {
            "track": "paper claim",
            "item": "prototype main table",
            "status": "ready",
            "evidence": "E118: dz014/dz018 improve beta005 on all five reporting splits",
            "next_action": "Use dz014 as main row and dz018 as tail-safety ablation.",
        },
        {
            "track": "paper claim",
            "item": "checkpoint and feature audit",
            "status": "ready for prototype",
            "evidence": "E108-E118 include threshold selection, holdout confirmation, external confirmation, qMSE/s_q/dead-code/nonfinite checks",
            "next_action": "Keep max500 as headroom until its checkpoint policy is stabilized.",
        },
        {
            "track": "paper claim",
            "item": "component ablation against prompt requirements",
            "status": "partial",
            "evidence": "HCS/no-transform/HCG-H lineage exists, but final table should explicitly name entropy-only or HVQ-like control if feasible",
            "next_action": "Add a compact ablation table separating index entropy, shift/scale, geometry, and reliability selector.",
        },
        {
            "track": "paper claim",
            "item": "multi-rate curves",
            "status": "missing",
            "evidence": "Current strongest frozen table is lambda=0.0035 only",
            "next_action": "Repeat final dz014/dz018 protocol at at least two more lambda/rate points after the prototype branch is frozen.",
        },
        {
            "track": "paper claim",
            "item": "SOTA or strong-backbone comparison",
            "status": "missing",
            "evidence": "No external strong-backbone plug-in row is in E118",
            "next_action": "Start with a local CompressAI-compatible backbone, then move to official external repos only after the adapter boundary is clean.",
        },
        {
            "track": "method strengthening",
            "item": "max500 high-mean branch",
            "status": "promising but not main",
            "evidence": "E110-E115 show larger mean gain but seed3456/q95 risk; learned cap recovers only a small part of oracle headroom",
            "next_action": "Train a conditional reliability/cap controller only if it has an independent teacher split and held-seed validation.",
        },
    ]

    adapter_rows = [
        {
            "module_boundary": "HCG quantizer adapter input",
            "current_location": "HCGMeanScaleHyperprior._conditioned_rvq",
            "required_tensors": "y [B,M,H,W], hyper_features [B,N,H,W], image_hw",
            "output_contract": "y_hat, indices, commit_loss, rvq_stats, conditioning_tensors",
            "portability": "high if extracted from backbone",
            "risk": "currently coupled to HCGMeanScaleHyperprior helper methods and config fields",
        },
        {
            "module_boundary": "hyperprior-conditioned heads",
            "current_location": "mu_head, log_s_head, householder_head, householder_gate_head, residual_selector_head",
            "required_tensors": "hyper_features with known channel count",
            "output_contract": "mu_q, s_q, v, raw/effective Householder gate, residual selector stats",
            "portability": "medium",
            "risk": "strong backbones may have different hyper decoder channel counts and spatial scales",
        },
        {
            "module_boundary": "RVQ core",
            "current_location": "hcg_rvq.quantizers.ResidualVectorQuantizer",
            "required_tensors": "normalized/geometry-transformed y with channels divisible by group_size",
            "output_contract": "u_hat, per-stage indices, commit loss, usage stats",
            "portability": "high",
            "risk": "fixed one-stage g64/k128 setting should be revalidated at new backbone latent dimensionality",
        },
        {
            "module_boundary": "index entropy prior",
            "current_location": "hcg_rvq.entropy.IndexEntropyModel",
            "required_tensors": "hyper_features and RVQ indices",
            "output_contract": "bpp_y_index and logits",
            "portability": "medium",
            "risk": "competitor backbones with autoregressive/context priors need a clear no-confound comparison",
        },
        {
            "module_boundary": "loss hooks",
            "current_location": "RateDistortionLoss conditioning_tensors hooks",
            "required_tensors": "rvq_stats and conditioning_tensors",
            "output_contract": "RD + commitment + optional teacher/anchor/selector losses",
            "portability": "medium",
            "risk": "teacher labels and anchor checkpoints are protocol-sensitive; they must not leak holdout labels",
        },
    ]

    backbone_rows = [
        {
            "priority": 1,
            "candidate": "CompressAI JointAutoregressiveHierarchicalPriors / mbt2018_mean",
            "why": "local dependency is already installed and shares hyperprior-style latent contracts",
            "first_action": "Create an HCG adapter prototype without touching external repos.",
            "claim_role": "stronger internal backbone, low engineering risk",
        },
        {
            "priority": 2,
            "candidate": "CompressAI Cheng2020Attention / cheng2020_attn",
            "why": "local installed model gives a stronger transform/context baseline than MeanScaleHyperprior",
            "first_action": "Audit latent M, hyper feature shape, and whether replacing y quantization confounds autoregressive coding.",
            "claim_role": "stronger LIC baseline within reproducible local framework",
        },
        {
            "priority": 3,
            "candidate": "DCAE official repo",
            "why": "prompt marks it as dictionary entropy modeling; conceptually close but on entropy side rather than quantizer geometry",
            "first_action": "Clone/read official implementation after adapter boundary exists.",
            "claim_role": "external SOTA comparison or plug-in target",
        },
        {
            "priority": 4,
            "candidate": "MambaIC / HPCM official repos",
            "why": "strong recent context/backbone directions, but high integration risk",
            "first_action": "Use as later plug-in targets after local CompressAI proof-of-portability.",
            "claim_role": "late-stage SOTA plug-in evidence",
        },
    ]

    next_experiments = [
        {
            "order": 1,
            "experiment": "E120 adapter extraction smoke",
            "goal": "Extract HCG quantizer logic behind an adapter interface and prove bit-identical output on one checkpoint/image.",
            "gpu": "optional; use CUDA_VISIBLE_DEVICES=0 if evaluating",
            "promotion_rule": "No RD/stat drift versus current HCGMeanScaleHyperprior direct path.",
        },
        {
            "order": 2,
            "experiment": "E121 explicit component ablation table",
            "goal": "Make the prompt-required ablations paper-readable: index entropy only, HCS shift/scale, geometry, residual selector.",
            "gpu": "mostly analysis; GPU only if a missing entropy-only row must be evaluated",
            "promotion_rule": "Every row uses fixed checkpoint protocol and reports qMSE/s_q/dead-code/nonfinite.",
        },
        {
            "order": 3,
            "experiment": "E122 local CompressAI strong-backbone smoke",
            "goal": "Attach the adapter to a local CompressAI stronger backbone and run a tiny non-paper smoke.",
            "gpu": "GPU0 only",
            "promotion_rule": "Finite loss/RD, stable feature stats, no device1 usage.",
        },
        {
            "order": 4,
            "experiment": "E123 multi-rate prototype expansion",
            "goal": "Repeat dz014/dz018 table at additional lambda points after adapter and ablation clarity.",
            "gpu": "GPU0 only",
            "promotion_rule": "At least two additional rate points with predeclared checkpoint/selection protocol.",
        },
    ]

    decision = {
        "conference_status": "promising prototype claim, not final SOTA claim",
        "main_result": {
            "dz014_mean_split_delta_vs_beta005": dz014["mean_split_delta_vs_beta005"],
            "dz018_mean_split_delta_vs_beta005": dz018["mean_split_delta_vs_beta005"],
            "dz014_all_splits_improve": dz014["all_splits_improve_beta005"],
            "dz018_all_splits_improve": dz018["all_splits_improve_beta005"],
        },
        "next_primary_action": "Extract/audit an HCG quantizer adapter before SOTA plug-in.",
        "next_paper_action": "Build the explicit component-ablation table.",
    }

    payload = {
        "decision": decision,
        "readiness_rows": readiness_rows,
        "adapter_rows": adapter_rows,
        "backbone_rows": backbone_rows,
        "next_experiments": next_experiments,
        "sources": {
            "prototype_table": "e118_hcg_rvq_prototype_main_table_package.json",
            "model_impl": "hcg_rvq/models/hyperprior_rvq.py",
            "rvq_impl": "hcg_rvq/quantizers/residual_vq.py",
            "index_entropy_impl": "hcg_rvq/entropy/index_entropy_model.py",
        },
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".readiness.csv"), readiness_rows)
    write_csv(OUT_PREFIX.with_suffix(".adapter.csv"), adapter_rows)
    write_csv(OUT_PREFIX.with_suffix(".backbones.csv"), backbone_rows)
    write_csv(OUT_PREFIX.with_suffix(".next_experiments.csv"), next_experiments)

    lines = [
        "# E119 SOTA/Backbone Plug-In Readiness Audit",
        "",
        "This audit separates the now-frozen prototype claim from the next SOTA/backbone integration work.",
        "",
        "## Decision",
        "",
        f"- Conference status: {decision['conference_status']}",
        f"- Next primary method action: {decision['next_primary_action']}",
        f"- Next paper action: {decision['next_paper_action']}",
        "",
        "## Readiness",
        "",
        "| track | item | status | evidence | next action |",
        "|---|---|---|---|---|",
    ]
    for row in readiness_rows:
        lines.append(
            f"| {row['track']} | {row['item']} | {row['status']} | {row['evidence']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Adapter Boundary",
            "",
            "| boundary | current location | required tensors | output contract | portability | risk |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in adapter_rows:
        lines.append(
            "| {module_boundary} | {current_location} | {required_tensors} | {output_contract} | {portability} | {risk} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Backbone Priority",
            "",
            "| priority | candidate | why | first action | claim role |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in backbone_rows:
        lines.append(
            f"| {row['priority']} | {row['candidate']} | {row['why']} | {row['first_action']} | {row['claim_role']} |"
        )
    lines.extend(
        [
            "",
            "## Next Experiments",
            "",
            "| order | experiment | goal | GPU | promotion rule |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in next_experiments:
        lines.append(
            f"| {row['order']} | {row['experiment']} | {row['goal']} | {row['gpu']} | {row['promotion_rule']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_suffix('.readiness.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.adapter.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.backbones.csv')}`",
            f"- `{OUT_PREFIX.with_suffix('.next_experiments.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
