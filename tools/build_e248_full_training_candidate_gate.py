#!/usr/bin/env python3
"""Build the E248 full-training candidate gate for EF-LIC and GLC.

The report consolidates recent short-cycle evidence into a promotion decision
matrix.  It does not claim final performance; it decides which candidates are
ready for codec-loop/full-training experiments under a paper-safe objective.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "e248_full_training_candidate_gate"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def best_min(rows: list[dict[str, str]], key: str, dataset: str | None = None) -> dict[str, str] | None:
    candidates = rows
    if dataset is not None:
        candidates = [r for r in rows if r.get("dataset") == dataset]
    candidates = [r for r in candidates if finite(as_float(r, key))]
    if not candidates:
        return None
    return min(candidates, key=lambda r: as_float(r, key))


def first_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in matches.items()):
            return row
    return None


def safe_num(value: float, digits: int = 6) -> str:
    if not finite(value):
        return "NA"
    return f"{value:.{digits}f}"


def summarize_e234() -> dict[str, Any]:
    fixed = read_csv(ANALYSIS / "e234_eflic_branch_controller_scaffold_summary_fixed.csv")
    oracle = read_csv(ANALYSIS / "e234_eflic_branch_controller_scaffold_summary_oracle.csv")
    summary = {}
    for dataset in ("pooled", "clicpro41", "kodak24"):
        best = best_min(fixed, "score_dists_3lpips", dataset)
        oracle_row = first_row(oracle, dataset=dataset)
        summary[dataset] = {
            "best_fixed": best,
            "oracle": oracle_row,
        }
    return summary


def summarize_e236() -> dict[str, Any]:
    fixed = read_csv(ANALYSIS / "e236_eflic_local_controller_map_summary_fixed.csv")
    oracle = read_csv(ANALYSIS / "e236_eflic_local_controller_map_summary_oracle.csv")
    summary = {}
    for dataset in ("pooled", "clicpro41", "kodak24"):
        best = best_min(fixed, "score_dists_3lpips", dataset)
        oracle_row = first_row(oracle, dataset=dataset)
        summary[dataset] = {
            "best_fixed": best,
            "oracle": oracle_row,
        }
    return summary


def summarize_split(rows: list[dict[str, str]], score_key: str = "score") -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in (
        "pooled_resub",
        "pooled_loio",
        "train_clicpro41_test_kodak24",
        "train_kodak24_test_clicpro41",
    ):
        split_rows = [r for r in rows if r.get("split") == split]
        if not split_rows:
            continue
        oracle = first_row(split_rows, method="oracle_all")
        true_family = first_row(split_rows, method="true_family_train_best_preset")
        if true_family is None:
            true_family = first_row(split_rows, method="true_family_train_best_policy")
        best_fixed = first_row(split_rows, method="best_fixed_train")
        learned = [
            r
            for r in split_rows
            if r.get("method", "").startswith(("ridge", "nearest"))
            or "fallback" in r.get("method", "")
        ]
        learned_best = min(learned, key=lambda r: as_float(r, score_key), default=None)
        summary[split] = {
            "oracle": oracle,
            "true_family": true_family,
            "best_fixed": best_fixed,
            "learned_best": learned_best,
        }
    return summary


def summarize_e246() -> dict[str, Any]:
    active = read_csv(ANALYSIS / "e246_eflic_decoder_safe_feature_groups.active_summary.csv")
    family = read_csv(ANALYSIS / "e246_eflic_decoder_safe_feature_groups.family_summary.csv")

    out: dict[str, Any] = {"activation": {}, "family": {}}
    for split in ("pooled_resub", "train_clicpro41_test_kodak24", "train_kodak24_test_clicpro41"):
        rows = [r for r in active if r.get("split") == split]
        if rows:
            out["activation"][split] = {
                "best_f1": max(
                    (r for r in rows if r.get("threshold_mode") == "best_f1"),
                    key=lambda r: as_float(r, "f1"),
                    default=None,
                ),
                "min_risk": min(
                    (r for r in rows if r.get("threshold_mode") == "min_weighted_risk"),
                    key=lambda r: as_float(r, "weighted_risk"),
                    default=None,
                ),
                "fpr010": max(
                    (r for r in rows if r.get("threshold_mode") == "fpr_le_010"),
                    key=lambda r: as_float(r, "recall"),
                    default=None,
                ),
            }
        fam_rows = [r for r in family if r.get("split") == split]
        if fam_rows:
            out["family"][split] = max(
                fam_rows, key=lambda r: as_float(r, "family_accuracy"), default=None
            )

    loio = [r for r in active if r.get("split", "").startswith("loio__")]
    if loio:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in loio:
            grouped[(row.get("feature_group", ""), row.get("threshold_mode", ""))].append(row)
        loio_rows = []
        for (feature_group, threshold_mode), rs in grouped.items():
            loio_rows.append(
                {
                    "feature_group": feature_group,
                    "threshold_mode": threshold_mode,
                    "test_images": str(len(rs)),
                    "recall": safe_num(mean(as_float(r, "recall") for r in rs)),
                    "fpr": safe_num(mean(as_float(r, "fpr") for r in rs)),
                    "f1": safe_num(mean(as_float(r, "f1") for r in rs)),
                    "weighted_risk": safe_num(mean(as_float(r, "weighted_risk") for r in rs)),
                }
            )
        out["activation"]["pooled_loio_aggregated"] = min(
            loio_rows, key=lambda r: float(r["weighted_risk"])
        )
    return out


def summarize_e247() -> dict[str, Any]:
    rows = read_csv(ANALYSIS / "e247_loss_objective_audit.csv")
    total = len(rows)
    rd_only = sum(1 for r in rows if int(float(r.get("noncore_count") or 0)) == 0)
    teacher = sum(1 for r in rows if int(float(r.get("teacher_selector_count") or 0)) > 0)
    anchor = sum(1 for r in rows if int(float(r.get("anchor_count") or 0)) > 0)
    teacher_or_anchor = sum(
        1
        for r in rows
        if int(float(r.get("teacher_selector_count") or 0)) > 0
        or int(float(r.get("anchor_count") or 0)) > 0
    )
    regularizer = sum(1 for r in rows if int(float(r.get("regularizer_count") or 0)) > 0)
    return {
        "configs": total,
        "rd_commit_only": rd_only,
        "teacher_selector": teacher,
        "anchor": anchor,
        "teacher_selector_or_anchor": teacher_or_anchor,
        "geometry_gate_regularizer": regularizer,
    }


def summarize_glc() -> dict[str, Any]:
    e170 = read_csv(ANALYSIS / "e170_glc_tail_vq_probe_kodak24_summary.csv")
    e171 = read_csv(ANALYSIS / "e171_glc_tail_rvq_stage_probe_k2_kodak24_summary.csv")
    e181 = read_csv(ANALYSIS / "e181_glc_decoder_aware_tail_vq_split_train_q0_oi16_kodak8.csv")

    e170_pg = [r for r in e170 if r.get("scope") == "part_group"]
    best_e170 = min(e170_pg, key=lambda r: as_float(r, "mse_ratio"), default=None)
    efficient_e170 = min(
        (r for r in e170_pg if as_float(r, "empirical_bpp_delta") <= 0.006),
        key=lambda r: as_float(r, "mse_ratio"),
        default=None,
    )
    e171_pg = [r for r in e171 if r.get("scope") == "part_group"]
    best_e171 = min(e171_pg, key=lambda r: as_float(r, "mse_ratio"), default=None)

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in e181:
        by_label[row.get("label", "")].append(row)
    e181_summary = {}
    for label, rows in by_label.items():
        if not label:
            continue
        e181_summary[label] = {
            "images": len(rows),
            "active_mse_ratio": mean(as_float(r, "active_mse_ratio") for r in rows),
            "empirical_bpp_delta": mean(as_float(r, "empirical_bpp_delta") for r in rows),
            "fixed_bpp_delta": mean(as_float(r, "fixed_bpp_delta") for r in rows),
            "delta_psnr": mean(as_float(r, "branch_psnr") - as_float(r, "base_psnr") for r in rows),
            "delta_ms_ssim": mean(as_float(r, "branch_ms_ssim") - as_float(r, "base_ms_ssim") for r in rows),
            "delta_lpips": mean(as_float(r, "branch_lpips") - as_float(r, "base_lpips") for r in rows),
            "delta_dists": mean(as_float(r, "branch_dists") - as_float(r, "base_dists") for r in rows),
            "nonfinite": sum(int(float(r.get("nonfinite") or 0)) for r in rows),
        }

    return {
        "e170_best_part_group": best_e170,
        "e170_efficient_part_group": efficient_e170,
        "e171_best_part_group": best_e171,
        "e181_decoder_aware": e181_summary,
    }


def fmt_row_metric(row: dict[str, str] | None, label_key: str, score_key: str) -> str:
    if not row:
        return "NA"
    label = row.get(label_key, "NA")
    score = safe_num(as_float(row, score_key))
    return f"{label} ({score_key} {score})"


def build_decisions(
    e234: dict[str, Any],
    e236: dict[str, Any],
    e235: dict[str, Any],
    e237: dict[str, Any],
    e246: dict[str, Any],
    e247: dict[str, Any],
    glc: dict[str, Any],
) -> list[dict[str, Any]]:
    eflic_e236_pooled = e236["pooled"]
    e237_loio = e237.get("pooled_loio", {})
    e246_loio = e246["activation"].get("pooled_loio_aggregated", {})

    glc_e181_trained = glc["e181_decoder_aware"].get("trained_eval", {})
    glc_e170_best = glc["e170_best_part_group"]

    decisions = [
        {
            "target": "EF-LIC",
            "candidate": "decoder-safe compact/local HCG branch with conservative fallback",
            "promote": "yes: mid-scale codec-loop gate, then full-training if stable",
            "evidence": (
                "E236 pooled oracle score "
                f"{safe_num(as_float(eflic_e236_pooled['oracle'], 'score_dists_3lpips'))}; "
                "best fixed "
                f"{fmt_row_metric(eflic_e236_pooled['best_fixed'], 'policy', 'score_dists_3lpips')}; "
                "E237 true-family almost closes oracle but learned selectors remain weak"
            ),
            "risk": (
                "E246 pooled LOIO activation still high-FPR/weak-family; do not use a "
                f"teacher-heavy classifier as paper-main (LOIO best risk {e246_loio.get('weighted_risk', 'NA')})"
            ),
            "next_action": (
                "Implement a small trainable decoder-safe branch whose default is zero/scalar, "
                "compare baseline/all-on/fixed-guarded/oracle/learned, and keep EF-LIC "
                "R-D/perceptual loss dominant."
            ),
            "full_training_gate": (
                "Advance only if finite checkpoints improve DISTS+3*LPIPS or RD on Kodak and "
                "CLIC professional, with no uncounted side information and activation not collapsing all-on."
            ),
        },
        {
            "target": "EF-LIC",
            "candidate": "frozen teacher/selector controller from E244/E246 features",
            "promote": "no: diagnostic/warmup only",
            "evidence": (
                "Feature-rich summaries memorize labels but do not generalize; family selection "
                "near chance under cross-dataset/LOIO."
            ),
            "risk": (
                "Could produce apparent gains by fitting teacher labels rather than proving "
                "hyperprior-conditioned quantizer geometry."
            ),
            "next_action": (
                "Use labels only for initialization, weak regularization, or oracle upper bounds; "
                "paper rows need codec-objective-dominant learned behavior."
            ),
            "full_training_gate": "Not eligible without larger independent labels or clean codec-loop evidence.",
        },
        {
            "target": "GLC",
            "candidate": "bit-aware q0 tail VQ/HCG branch over active residual states",
            "promote": "yes: integrated smoke, then mid-scale training gate",
            "evidence": (
                "E170 part-group residual VQ has strong q0 MSE headroom "
                f"({fmt_row_metric(glc_e170_best, 'scope', 'mse_ratio')}); "
                "E181 decoder-aware split improves PSNR/MS-SSIM/LPIPS on Kodak8 after OI16 training."
            ),
            "risk": (
                "Residual-MSE-only VQ increases empirical index bpp and can worsen DISTS; "
                "later q stages are fragile."
            ),
            "next_action": (
                "Add explicit index-rate/perceptual accounting to the q0 branch, then test "
                "baseline/scalar/K4/K8/RVQ-stage/HCG variants before scaling q1-q3."
            ),
            "full_training_gate": (
                "Advance only if PSNR/MS-SSIM/LPIPS gains survive while DISTS and bpp are "
                "not worsened beyond a predeclared tolerance."
            ),
        },
        {
            "target": "GLC",
            "candidate": "dense or residual-MSE-only all-on RVQ branch",
            "promote": "no: mechanism probe only",
            "evidence": (
                "E170/E171 prove representational headroom, but E181 shows metric conflict "
                f"(trained empirical bpp delta {safe_num(glc_e181_trained.get('empirical_bpp_delta', math.nan))}, "
                f"DISTS delta {safe_num(glc_e181_trained.get('delta_dists', math.nan))})."
            ),
            "risk": "Can look good in residual MSE or PSNR while losing rate/perceptual quality.",
            "next_action": "Use as ablation and initialization check, not the main submission design.",
            "full_training_gate": "Not eligible until rate/perceptual-aware objective and reliability are added.",
        },
    ]

    for decision in decisions:
        decision["loss_policy"] = (
            f"E247 guardrail: {e247['rd_commit_only']}/{e247['configs']} configs are clean "
            "RD/commit-only; paper-main runs should stay in that family or document weak auxiliaries."
        )
    return decisions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(payload: dict[str, Any]) -> str:
    decisions = payload["decisions"]
    e234 = payload["eflic"]["e234"]
    e236 = payload["eflic"]["e236"]
    glc = payload["glc"]
    e247 = payload["loss_policy"]

    lines = [
        "# E248 Full-Training Candidate Gate",
        "",
        "## Purpose",
        "",
        "This package converts the recent short-cycle EF-LIC/GLC diagnostics into a",
        "paper-safe promotion gate.  It does not treat smoke-test numbers as final",
        "performance.  Instead, it decides which HCG-RVQ variants are worth moving",
        "toward codec-loop, mid-scale, and eventually full-training/full-evaluation",
        "under an objective dominated by the original codec loss.",
        "",
        "## EF-LIC Evidence Snapshot",
        "",
        "| source | pooled best fixed | pooled oracle | decision signal |",
        "|---|---:|---:|---|",
        "| E234 compact branch vocabulary | "
        f"{fmt_row_metric(e234['pooled']['best_fixed'], 'preset', 'score_dists_3lpips')} | "
        f"{safe_num(as_float(e234['pooled']['oracle'], 'score_dists_3lpips'))} | "
        "fixed presets are small but oracle is large |",
        "| E236 local policy map | "
        f"{fmt_row_metric(e236['pooled']['best_fixed'], 'policy', 'score_dists_3lpips')} | "
        f"{safe_num(as_float(e236['pooled']['oracle'], 'score_dists_3lpips'))} | "
        "local policy headroom is stronger than fixed rows |",
        "",
        "E235/E237 show that true-family/oracle policies almost close the local-policy",
        "upper bound, but E246 shows that frozen decoder-safe feature classifiers do",
        "not transfer reliably.  Therefore the EF-LIC promotion target is not a",
        "teacher-heavy selector; it is an R-D/perceptual-dominant learned branch with",
        "zero/scalar fallback and explicit oracle/fixed/all-on ablations.",
        "",
        "## GLC Evidence Snapshot",
        "",
        "| source | key result | decision signal |",
        "|---|---|---|",
    ]

    best_e170 = glc["e170_best_part_group"]
    efficient_e170 = glc["e170_efficient_part_group"]
    best_e171 = glc["e171_best_part_group"]
    lines.extend(
        [
            "| E170 tail VQ active residual probe | "
            f"best part-group K={best_e170.get('k', 'NA')} q{best_e170.get('q_index', 'NA')} "
            f"MSE ratio {safe_num(as_float(best_e170, 'mse_ratio'))}, empirical bpp delta "
            f"{safe_num(as_float(best_e170, 'empirical_bpp_delta'))} | "
            "residual headroom exists but index rate must be counted |",
            "| E170 rate-lighter part-group row | "
            f"K={efficient_e170.get('k', 'NA')} q{efficient_e170.get('q_index', 'NA')} "
            f"MSE ratio {safe_num(as_float(efficient_e170, 'mse_ratio'))}, empirical bpp delta "
            f"{safe_num(as_float(efficient_e170, 'empirical_bpp_delta'))} | "
            "useful ablation for bit-aware design |",
            "| E171 K=2 RVQ stage probe | "
            f"best q{best_e171.get('q_index', 'NA')} stages={best_e171.get('stages', 'NA')} "
            f"MSE ratio {safe_num(as_float(best_e171, 'mse_ratio'))}, empirical bpp delta "
            f"{safe_num(as_float(best_e171, 'empirical_bpp_delta'))} | "
            "stages help but later q and bpp are fragile |",
        ]
    )

    e181 = glc["e181_decoder_aware"]
    if e181:
        lines.extend(["", "### E181 Decoder-Aware Split Aggregate", ""])
        lines.append("| label | images | active MSE ratio | emp bpp d | PSNR d | MS-SSIM d | LPIPS d | DISTS d | nonfinite |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, row in sorted(e181.items()):
            lines.append(
                f"| {label} | {row['images']} | {safe_num(row['active_mse_ratio'])} | "
                f"{safe_num(row['empirical_bpp_delta'])} | {safe_num(row['delta_psnr'])} | "
                f"{safe_num(row['delta_ms_ssim'])} | {safe_num(row['delta_lpips'])} | "
                f"{safe_num(row['delta_dists'])} | {row['nonfinite']} |"
            )

    lines.extend(
        [
            "",
            "## Promotion Decisions",
            "",
            "| target | candidate | promote | next action |",
            "|---|---|---|---|",
        ]
    )
    for row in decisions:
        lines.append(
            f"| {row['target']} | {row['candidate']} | {row['promote']} | {row['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Full-Training Guardrails",
            "",
            "1. Keep original EF-LIC/GLC R-D/perceptual terms dominant.  HCG-specific",
            "   labels may initialize or weakly regularize, but they must not carry the",
            "   paper-main improvement.",
            "2. Every promoted row must include matched baseline, zero/scalar fallback,",
            "   all-on HCG, fixed guarded HCG or scalar/VQ ablation, oracle/teacher upper",
            "   bound, and learned HCG under the same split and checkpoint policy.",
            "3. Report bpp/PSNR/MS-SSIM plus perceptual metrics and intermediate",
            "   statistics: codebook usage, dead codes, residual MSE per stage, index",
            "   entropy, activation fraction, and failure cases.",
            "4. Use GPU0 only for future CUDA jobs on this machine.",
            "",
            "## Loss Audit Link",
            "",
            f"E247 scanned {e247['configs']} configs: {e247['rd_commit_only']} are clean",
            f"RD/commit-only, {e247['teacher_selector_or_anchor']} include teacher/selector",
            f"or anchor terms ({e247['teacher_selector']} teacher/selector; {e247['anchor']}",
            f"anchor), and {e247['geometry_gate_regularizer']}",
            "include geometry/gate regularizers.  This gate treats heavy auxiliary losses",
            "as diagnostic or warmup tools unless a matched ablation proves otherwise.",
            "",
            "## Recommended Immediate Experiments",
            "",
            "1. EF-LIC E249: implement a small trainable decoder-safe branch with",
            "   zero/scalar fallback and original EF-LIC objective dominance; smoke on",
            "   Kodak24 and CLIC professional before any full run.",
            "2. EF-LIC E250: same branch with fixed guarded/all-on/oracle controls to",
            "   quantify whether learned control beats simple policies.",
            "3. GLC E249: make the q0 tail VQ/HCG branch bit-aware and perceptual-aware,",
            "   then rerun OI16-to-Kodak8 before scaling q1-q3.",
            "4. Full training: promote only after the mid-scale gate shows finite",
            "   checkpoints, counted index rate, no all-on collapse, and aligned",
            "   RD/perceptual/intermediate improvements.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    e234 = summarize_e234()
    e236 = summarize_e236()
    e235 = summarize_split(read_csv(ANALYSIS / "e235_eflic_compact_controller_readiness.summary.csv"))
    e237 = summarize_split(
        read_csv(ANALYSIS / "e237_eflic_local_policy_controller_split.summary.csv")
    )
    e246 = summarize_e246()
    e247 = summarize_e247()
    glc = summarize_glc()

    decisions = build_decisions(e234, e236, e235, e237, e246, e247, glc)
    payload = {
        "eflic": {
            "e234": e234,
            "e236": e236,
            "e235_split": e235,
            "e237_split": e237,
            "e246_feature_gate": e246,
        },
        "glc": glc,
        "loss_policy": e247,
        "decisions": decisions,
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_PREFIX.with_suffix(".csv"), decisions)
    OUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    OUT_PREFIX.with_suffix(".md").write_text(build_markdown(payload), encoding="utf-8")
    print(f"wrote {OUT_PREFIX.with_suffix('.md')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.json')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
