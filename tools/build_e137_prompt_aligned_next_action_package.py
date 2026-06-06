#!/usr/bin/env python3
"""Build a prompt-aligned next-action package from the current HCG-RVQ evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e137_prompt_aligned_next_action_package"


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


def fnum(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, *, signed: bool = False) -> str:
    value = fnum(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def find(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def main() -> None:
    e118 = load_json("e118_hcg_rvq_prototype_main_table_package.json")
    e119 = load_json("e119_sota_plugin_readiness_audit.json")
    e121 = load_json("e121_component_ablation_table.json")
    e135 = load_json("e135_decoder_reproducible_guard_audit.json")
    e136 = load_json("e136_decoder_proxy_supervised_probe.json")

    component_summary = {row["component"]: row for row in e121["summary_rows"]}
    dz014 = component_summary["deadzone014"]
    dz018 = component_summary["deadzone018"]
    beta005 = component_summary["beta005 guard"]
    old_gate = component_summary["old gate0.25"]
    min090 = component_summary["min090 risk gate"]

    e118_by_threshold = {row["threshold"]: row for row in e118["threshold_summary"]}
    e135_summary = e135["summary"]
    e136_top_deployable = e136["top_deployable"]
    e136_top_reference = e136["top_reference"]

    candidate_forward_005 = find(
        e135_summary,
        tier="all_candidate_forward",
        objective="mean_dead_budget",
        budget="0.05",
    )
    hyper_preindex_005 = find(
        e135_summary,
        tier="hyper_preindex",
        objective="mean_dead_budget",
        budget="0.05",
    )
    hyper_preindex_075 = find(
        e135_summary,
        tier="hyper_preindex",
        objective="mean_dead_budget",
        budget="0.075",
    )
    best_decoder_proxy = min(
        [row for row in e136_top_deployable if row["deployability"] == "decoder_preindex_no_side_bit_candidate"],
        key=lambda row: fnum(row["mean_test_delta_rd"]),
    )
    best_reference_proxy = min(e136_top_reference, key=lambda row: fnum(row["mean_test_delta_rd"]))

    prompt_status_rows = [
        {
            "prompt_requirement": "Hyperprior generates local quantizer geometry",
            "status": "supported_prototype",
            "evidence": "E121 deadzone014/dz018 and fixed Householder rows improve HCS on all five reporting splits.",
            "key_metric": "dz014 mean delta vs HCS",
            "value": fnum(dz014["mean_split_delta_vs_hcs"]),
            "gap": "Need entropy-only/HVQ-like row to isolate index-prior-only from shift/scale/geometry.",
        },
        {
            "prompt_requirement": "Checkpoint-selected evaluation",
            "status": "strong_for_prototype",
            "evidence": "E118 freezes dz014/dz018 across OpenImages transfer/holdout, Kodak, and CLIC-style external splits.",
            "key_metric": "dz014 worst split delta vs beta005",
            "value": fnum(e118_by_threshold["014"]["worst_split_delta_vs_beta005"]),
            "gap": "Repeat at additional lambda/rate points before final manuscript curves.",
        },
        {
            "prompt_requirement": "Intermediate feature and distribution analysis",
            "status": "active_and_useful",
            "evidence": "E135 separates candidate-forward, decoder-preindex, index-usage, and encoder-error feature tiers; E136 trains split-protocol probes.",
            "key_metric": "candidate-forward budget0.05 delta RD",
            "value": fnum(candidate_forward_005["mean_delta_rd"]),
            "gap": "Promoted controller still needs held-out checkpoint evaluation with qMSE/dead-code/nonfinite checks.",
        },
        {
            "prompt_requirement": "Decoder-reproducible controller",
            "status": "partial",
            "evidence": "Hyper-preindex features are deployable without side bits, but are weaker than diagnostic/candidate-forward features.",
            "key_metric": "best decoder-preindex delta RD",
            "value": fnum(best_decoder_proxy["mean_test_delta_rd"]),
            "gap": "Decide whether the small deployable gain is worth implementation, or keep it as diagnostic while prioritizing multi-rate and entropy-only rows.",
        },
        {
            "prompt_requirement": "SOTA or strong-backbone bridge",
            "status": "not_ready_as_headline",
            "evidence": e119["decision"]["conference_status"],
            "key_metric": "next strong-backbone action order",
            "value": 1.0,
            "gap": "Use adapter/local CompressAI stronger-backbone smoke after the prototype table and component ablations are frozen.",
        },
    ]

    method_evidence_rows = [
        {
            "method_or_controller": "old gate0.25",
            "role": "raw geometry control",
            "mean_delta_vs_hcs": fnum(old_gate["mean_split_delta_vs_hcs"]),
            "worst_delta_vs_hcs": fnum(old_gate["worst_split_delta_vs_hcs"]),
            "nonfinite": int(old_gate["nonfinite"]),
            "interpretation": "Confirms Householder geometry is useful, but weaker than guarded branches.",
        },
        {
            "method_or_controller": "min090 risk gate",
            "role": "conservative geometry diagnostic",
            "mean_delta_vs_hcs": fnum(min090["mean_split_delta_vs_hcs"]),
            "worst_delta_vs_hcs": fnum(min090["worst_split_delta_vs_hcs"]),
            "nonfinite": int(min090["nonfinite"]),
            "interpretation": "Risk control helps some fragile cases but is not the strongest final row.",
        },
        {
            "method_or_controller": "beta005 guard",
            "role": "broad fixed-checkpoint prototype baseline",
            "mean_delta_vs_hcs": fnum(beta005["mean_split_delta_vs_hcs"]),
            "worst_delta_vs_hcs": fnum(beta005["worst_split_delta_vs_hcs"]),
            "nonfinite": int(beta005["nonfinite"]),
            "interpretation": "Paper-safe broad evidence baseline.",
        },
        {
            "method_or_controller": "deadzone014",
            "role": "current main mean-RD HCG-RVQ branch",
            "mean_delta_vs_hcs": fnum(dz014["mean_split_delta_vs_hcs"]),
            "worst_delta_vs_hcs": fnum(dz014["worst_split_delta_vs_hcs"]),
            "nonfinite": int(dz014["nonfinite"]),
            "interpretation": "Best current prototype row; all reporting splits improve HCS.",
        },
        {
            "method_or_controller": "deadzone018",
            "role": "tail-safety HCG-RVQ branch",
            "mean_delta_vs_hcs": fnum(dz018["mean_split_delta_vs_hcs"]),
            "worst_delta_vs_hcs": fnum(dz018["worst_split_delta_vs_hcs"]),
            "nonfinite": int(dz018["nonfinite"]),
            "interpretation": "Slightly weaker mean, lower q95 damage than dz014.",
        },
    ]

    controller_rows = [
        {
            "controller_scope": "candidate-forward diagnostic",
            "deployability": candidate_forward_005["deployability"],
            "budget": fnum(candidate_forward_005["budget"]),
            "mean_delta_rd": fnum(candidate_forward_005["mean_delta_rd"]),
            "mean_delta_dead": fnum(candidate_forward_005["mean_delta_dead"]),
            "mean_q95_damage_rd": fnum(candidate_forward_005["mean_q95_damage_rd"]),
            "selected": fnum(candidate_forward_005["mean_selected"]),
            "protocols": f"{candidate_forward_005['positive_protocols']}/{candidate_forward_005['num_protocols']}",
            "use": "reference upper bound for a practical controller; not directly decoder-preindex.",
        },
        {
            "controller_scope": "hyper-preindex safe budget",
            "deployability": hyper_preindex_005["deployability"],
            "budget": fnum(hyper_preindex_005["budget"]),
            "mean_delta_rd": fnum(hyper_preindex_005["mean_delta_rd"]),
            "mean_delta_dead": fnum(hyper_preindex_005["mean_delta_dead"]),
            "mean_q95_damage_rd": fnum(hyper_preindex_005["mean_q95_damage_rd"]),
            "selected": fnum(hyper_preindex_005["mean_selected"]),
            "protocols": f"{hyper_preindex_005['positive_protocols']}/{hyper_preindex_005['num_protocols']}",
            "use": "deployable conservative fallback; gain is small.",
        },
        {
            "controller_scope": "hyper-preindex mean-RD budget",
            "deployability": hyper_preindex_075["deployability"],
            "budget": fnum(hyper_preindex_075["budget"]),
            "mean_delta_rd": fnum(hyper_preindex_075["mean_delta_rd"]),
            "mean_delta_dead": fnum(hyper_preindex_075["mean_delta_dead"]),
            "mean_q95_damage_rd": fnum(hyper_preindex_075["mean_q95_damage_rd"]),
            "selected": fnum(hyper_preindex_075["mean_selected"]),
            "protocols": f"{hyper_preindex_075['positive_protocols']}/{hyper_preindex_075['num_protocols']}",
            "use": "best no-side-bit mean signal, but higher dead-code/q95 cost.",
        },
        {
            "controller_scope": "logistic decoder proxy",
            "deployability": best_decoder_proxy["deployability"],
            "budget": fnum(best_decoder_proxy["budget"]),
            "mean_delta_rd": fnum(best_decoder_proxy["mean_test_delta_rd"]),
            "mean_delta_dead": fnum(best_decoder_proxy["mean_test_delta_dead"]),
            "mean_q95_damage_rd": fnum(best_decoder_proxy["mean_test_q95_damage_rd"]),
            "selected": fnum(best_decoder_proxy["mean_test_selected"]),
            "protocols": f"{best_decoder_proxy['positive_protocols']}/{best_decoder_proxy['num_protocols']}",
            "use": "candidate for no-side-bit controller only if followed by checkpoint evaluation.",
        },
        {
            "controller_scope": "reference logistic probe",
            "deployability": best_reference_proxy["deployability"],
            "budget": fnum(best_reference_proxy["budget"]),
            "mean_delta_rd": fnum(best_reference_proxy["mean_test_delta_rd"]),
            "mean_delta_dead": fnum(best_reference_proxy["mean_test_delta_dead"]),
            "mean_q95_damage_rd": fnum(best_reference_proxy["mean_test_q95_damage_rd"]),
            "selected": fnum(best_reference_proxy["mean_test_selected"]),
            "protocols": f"{best_reference_proxy['positive_protocols']}/{best_reference_proxy['num_protocols']}",
            "use": "diagnostic target; requires baseline/candidate-side information.",
        },
    ]

    next_action_rows = [
        {
            "priority": 1,
            "track": "paper-claim",
            "action": "Add explicit entropy-only / HVQ-like ablation row",
            "why_now": "Prompt requires separating index entropy from shift/scale and geometry; this is the largest current component-table gap.",
            "promotion_rule": "Matched protocol, nonfinite=0, reports RD/bpp/qMSE/dead-code, and shows HCS/HCG gains are not just index-prior gains.",
        },
        {
            "priority": 2,
            "track": "paper-claim",
            "action": "Repeat dz014/dz018 fixed protocol at two additional lambdas",
            "why_now": "Current final evidence is lambda=0.0035 only; international submission needs a curve, not one point.",
            "promotion_rule": "Each rate point uses fixed checkpoint selection, GPU0, no stale artifacts, and all main tables include q95/nonfinite diagnostics.",
        },
        {
            "priority": 3,
            "track": "method-strengthening",
            "action": "If controller work continues, implement the no-side-bit hyper-preindex proxy first as a checkpoint-evaluated pilot",
            "why_now": "E135/E136 show decoder-known features are weaker but deployable; candidate-forward features are stronger but need side/proxy design.",
            "promotion_rule": "Must beat the deterministic hyper-preindex threshold and preserve qMSE/dead-code/nonfinite on held-out checkpoint evaluation.",
        },
        {
            "priority": 4,
            "track": "SOTA-bridge",
            "action": "Run local CompressAI strong-backbone adapter smoke after the above tables are frozen",
            "why_now": "E119 says plug-in is promising but not ready as a headline; adapter boundary exists, but should not distract from missing ablations.",
            "promotion_rule": "Finite loss/RD, no device1 usage, stable feature stats, and no claim of SOTA dominance until matched baselines exist.",
        },
    ]

    decision = {
        "status": "on_track_but_not_submission_complete",
        "main_claim": "HCG-RVQ should remain a hyperprior-conditioned local quantizer-geometry paper, not merely an index-entropy or SOTA-backbone paper.",
        "current_best_row": "deadzone014",
        "current_safety_row": "deadzone018",
        "prototype_strength": "strong enough to continue seriously toward an international submission",
        "largest_gap": "explicit entropy-only ablation plus multi-rate repetition",
        "controller_takeaway": "decoder-known reliability control is plausible but currently weaker than diagnostic candidate-forward guards",
    }

    payload = {
        "decision": decision,
        "prompt_status_rows": prompt_status_rows,
        "method_evidence_rows": method_evidence_rows,
        "controller_rows": controller_rows,
        "next_action_rows": next_action_rows,
        "sources": {
            "prototype_main_table": "e118_hcg_rvq_prototype_main_table_package.json",
            "sota_readiness": "e119_sota_plugin_readiness_audit.json",
            "component_ablation": "e121_component_ablation_table.json",
            "decoder_reproducible_guard": "e135_decoder_reproducible_guard_audit.json",
            "decoder_proxy_probe": "e136_decoder_proxy_supervised_probe.json",
        },
    }

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT_PREFIX.with_suffix(".prompt_status.csv"), prompt_status_rows)
    write_csv(OUT_PREFIX.with_suffix(".method_evidence.csv"), method_evidence_rows)
    write_csv(OUT_PREFIX.with_suffix(".controllers.csv"), controller_rows)
    write_csv(OUT_PREFIX.with_suffix(".next_actions.csv"), next_action_rows)

    lines = [
        "# E137 Prompt-Aligned Next Action Package",
        "",
        "This package reconnects the current evidence to `docs/prompt.txt` after the literature/code refresh and the E135/E136 deployability analyses.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Main claim: {decision['main_claim']}",
        f"- Current best row: `{decision['current_best_row']}`",
        f"- Current safety row: `{decision['current_safety_row']}`",
        f"- Largest gap: {decision['largest_gap']}",
        f"- Controller takeaway: {decision['controller_takeaway']}",
        "",
        "## Prompt Alignment",
        "",
        "| requirement | status | key metric | value | gap |",
        "|---|---|---|---:|---|",
    ]
    for row in prompt_status_rows:
        lines.append(
            f"| {row['prompt_requirement']} | {row['status']} | {row['key_metric']} | "
            f"{fmt(row['value'], signed=True)} | {row['gap']} |"
        )
    lines.extend([
        "",
        "## Method Evidence",
        "",
        "| method | role | mean vs HCS | worst vs HCS | nonfinite | interpretation |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in method_evidence_rows:
        lines.append(
            f"| {row['method_or_controller']} | {row['role']} | {fmt(row['mean_delta_vs_hcs'], signed=True)} | "
            f"{fmt(row['worst_delta_vs_hcs'], signed=True)} | {row['nonfinite']} | {row['interpretation']} |"
        )
    lines.extend([
        "",
        "## Controller Deployability",
        "",
        "| scope | deployability | budget | mean delta RD | delta dead | q95 damage | selected | protocols | use |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in controller_rows:
        lines.append(
            f"| {row['controller_scope']} | {row['deployability']} | {fmt(row['budget'])} | "
            f"{fmt(row['mean_delta_rd'], signed=True)} | {fmt(row['mean_delta_dead'], signed=True)} | "
            f"{fmt(row['mean_q95_damage_rd'], signed=True)} | {fmt(row['selected'])} | {row['protocols']} | {row['use']} |"
        )
    lines.extend([
        "",
        "## Next Actions",
        "",
        "| priority | track | action | why now | promotion rule |",
        "|---:|---|---|---|---|",
    ])
    for row in next_action_rows:
        lines.append(
            f"| {row['priority']} | {row['track']} | {row['action']} | {row['why_now']} | {row['promotion_rule']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The project is still aligned with the prompt goal. The evidence is good enough to keep building a serious submission package, but the next action should be disciplined: add the missing entropy-only control and multi-rate repetitions before making broad SOTA claims. Reliability-control work should continue in parallel only when it is decoder-reproducible or explicitly accounted for as a diagnostic upper bound.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_PREFIX.with_suffix('.json')}`",
        f"- `{OUT_PREFIX.with_suffix('.prompt_status.csv')}`",
        f"- `{OUT_PREFIX.with_suffix('.method_evidence.csv')}`",
        f"- `{OUT_PREFIX.with_suffix('.controllers.csv')}`",
        f"- `{OUT_PREFIX.with_suffix('.next_actions.csv')}`",
    ])
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
