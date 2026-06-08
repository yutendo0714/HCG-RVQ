#!/usr/bin/env python3
"""Build the E291 promotion audit for EF-LIC/GLC HCG-RVQ integration.

The audit is intentionally a reporting artifact, not a training/evaluation
script. It consolidates recent short-cycle evidence into the next paper-facing
implementation decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

E287 = ROOT / "experiments/analysis/e287_glc_signal_accounted_clictail9_kodak16_current_subset.json"
E288 = ROOT / "experiments/analysis/e288_eflic_fallback_gate_context_after_glc_contract_t8_e8_s24.json"
E289 = ROOT / "experiments/analysis/e289_eflic_branch_controller_current_kodak4_contract_smoke.json"
E290 = ROOT / "experiments/analysis/e290_eflic_branch_controller_current_clic_tail4_contract_smoke.json"
E234 = ROOT / "experiments/analysis/e234_eflic_branch_controller_scaffold_summary.md"
E235 = ROOT / "experiments/analysis/e235_eflic_compact_controller_readiness.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def fmt(value: Any, signed: bool = False, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.{digits}f}"


def find_row(rows: list[dict[str, Any]], domain: str, label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("domain") == domain and row.get("label") == label:
            return row
    raise KeyError((domain, label))


def compact_branch_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ["zero", "sparse_prev005", "sparse_prev010", "constant020", "soft_support020"]
    by_name = {row["preset"]: row for row in summary}
    return [by_name[name] for name in keep if name in by_name]


def build_report(output_prefix: Path) -> None:
    e287 = load_json(E287)
    e288 = load_json(E288)
    e289 = load_json(E289)
    e290 = load_json(E290)

    glc_rows = e287["summary"]
    glc_focus_labels = [
        "derived_rate_cap_replacement_soft_cap0p003",
        "derived_rate_cap_replacement_soft_cap0p003_sig8b",
        "trained_rate_cap_replacement_soft_cap0p0035",
        "trained_rate_cap_replacement_soft_cap0p0035_sig8b",
        "trained_rate_cap_replacement_soft_cap0p004",
        "trained_replacement_all_on",
    ]
    glc_focus = [find_row(glc_rows, "all", label) for label in glc_focus_labels]

    eflic_runs = {
        "kodak4": {
            "dataset": e289["dataset"],
            "images": len(e289["images"]),
            "rows": compact_branch_rows(e289["summary"]),
        },
        "clic_tail4": {
            "dataset": e290["dataset"],
            "images": len(e290["images"]),
            "rows": compact_branch_rows(e290["summary"]),
        },
    }

    best_eflic_rows: list[dict[str, Any]] = []
    for run_name, run in eflic_runs.items():
        nonzero = [row for row in run["rows"] if row["preset"] != "zero"]
        best = max(nonzero, key=lambda row: float(row["delta_psnr"]))
        safest = min(nonzero, key=lambda row: abs(float(row["geometry_delta_rms"])))
        best_eflic_rows.append(
            {
                "split": run_name,
                "best_preset": best["preset"],
                "best_delta_psnr": best["delta_psnr"],
                "safe_preset": safest["preset"],
                "safe_delta_psnr": safest["delta_psnr"],
                "safe_geometry_delta_rms": safest["geometry_delta_rms"],
                "zero_exact": next(row for row in run["rows"] if row["preset"] == "zero")["max_decode_diff"] == 0.0,
                "nonfinite_rows": sum(int(row["nonfinite_rows"]) for row in run["rows"]),
            }
        )

    controller_smoke = {
        "all_checks_passed": bool(e288["all_checks_passed"]),
        "trained_eval_loss": e288["trained_eval"]["loss"],
        "trained_eval_rel_loss": e288["trained_eval"]["rel_loss"],
        "trained_eval_gate_mean": e288["trained_eval"]["gate_mean"],
        "trained_eval_hard_base_exact": e288["trained_eval"]["hard_base_exact"],
        "trained_eval_nonfinite": e288["trained_eval"].get("nonfinite_records", e288["trained_eval"].get("nonfinite", 0)),
    }

    audit = {
        "purpose": "E291 EF-LIC/GLC HCG-RVQ promotion audit",
        "inputs": {
            "glc_signal_accounted": str(E287.relative_to(ROOT)),
            "eflic_controller_smoke": str(E288.relative_to(ROOT)),
            "eflic_kodak4_codec_smoke": str(E289.relative_to(ROOT)),
            "eflic_clic_tail4_codec_smoke": str(E290.relative_to(ROOT)),
            "eflic_branch_headroom": str(E234.relative_to(ROOT)),
            "eflic_controller_readiness": str(E235.relative_to(ROOT)),
        },
        "controller_smoke": controller_smoke,
        "glc_focus": glc_focus,
        "eflic_best_rows": best_eflic_rows,
        "decisions": [
            {
                "track": "GLC",
                "decision": "Promote selected replacement, not dense all-on quantization.",
                "default": "cap0.0035 for balanced paper-facing controlled evidence; cap0.0030 for strict fixed-index/no-entropy framing.",
                "risk": "cap0.0040 has stronger mean score but includes CLIC-tail failures, so it remains an aggressive branch with failure analysis.",
            },
            {
                "track": "EF-LIC",
                "decision": "Insert HCG-RVQ as decoder-safe local quantizer geometry inside the existing EF-LIC sequential support/context loop.",
                "default": "Start from sparse/constant no-sidebit presets and then train a fallback-gated local controller.",
                "risk": "Fixed all-image/all-position choices are not reliable enough; E235 shows simple held-out predictors are weak.",
            },
            {
                "track": "Loss",
                "decision": "Keep the original codec objective dominant.",
                "default": "Add only minimal VQ/index/false-positive/fallback terms needed to make selected replacement measurable.",
                "risk": "Extra auxiliary losses can dilute RD/perceptual optimization if they are not tied to codec accounting.",
            },
        ],
    }

    lines: list[str] = []
    lines.append("# E291 EF-LIC/GLC HCG-RVQ Promotion Audit")
    lines.append("")
    lines.append(
        "This audit consolidates recent short-cycle evidence into the next implementation "
        "decision for paper-facing HCG-RVQ integration. It is not a final full-training "
        "or full-evaluation claim."
    )
    lines.append("")
    lines.append("## Contract Summary")
    lines.append("")
    lines.append(
        "HCG-RVQ remains aligned with the prompt goal when the hyperprior/context state "
        "generates local quantizer geometry, not hidden side information. For EF-LIC, "
        "the insertion point is after `_mean_scale` and normalized slice construction, "
        "before the existing RVQ slice quantizer. The original EF-LIC `h_a/h_s`, "
        "representation-domain decorrelation, support buffer, adaptor, context predictor, "
        "and sequential decoder loop are preserved."
    )
    lines.append("")
    lines.append("## GLC Signal-Accounted Controller Rows")
    lines.append("")
    lines.append(
        "| label | images | score | fixed score | delta bpp | fixed delta bpp | signal bpp | selected | selected win | selected fixed win | worst fixed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in glc_focus:
        lines.append(
            "| {label} | {images} | {score} | {fixed_score} | {dbpp} | {fixed_dbpp} | {signal} | {selected} | {sel_win} | {sel_fixed_win} | {worst_fixed} |".format(
                label=row["label"],
                images=row["images"],
                score=fmt(row["score"], signed=True),
                fixed_score=fmt(row["fixed_score"], signed=True),
                dbpp=fmt(row["delta_bpp"], signed=True),
                fixed_dbpp=fmt(row["fixed_delta_bpp"], signed=True),
                signal=fmt(row["selection_signal_bpp"]),
                selected=fmt(row["selected_frac"]),
                sel_win=fmt(row["selected_win_frac"]),
                sel_fixed_win=fmt(row["selected_fixed_win_frac"]),
                worst_fixed=fmt(row["worst_fixed_score"], signed=True),
            )
        )
    lines.append("")
    lines.append(
        "GLC decision: cap `0.0035` is the balanced controller because selected empirical "
        "win remains `1.000000` and selected fixed-index win remains `0.950000`; cap "
        "`0.0030` is the strict EF-LIC-compatible fixed-index/no-entropy candidate; "
        "cap `0.0040` is higher-gain but must carry CLIC-tail failure analysis. Dense "
        "all-on replacement is a negative control, not a candidate."
    )
    lines.append("")
    lines.append("## EF-LIC Current Codec-Path Contract Smokes")
    lines.append("")
    lines.append(
        "| split | preset | family | delta bpp | delta PSNR | decode diff | nonfinite | y mismatch | geom RMS | index entropy |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, run in eflic_runs.items():
        for row in run["rows"]:
            lines.append(
                "| {split} | {preset} | {family} | {dbpp} | {dpsnr} | {decode} | {nonfinite} | {mismatch} | {geom} | {entropy} |".format(
                    split=split,
                    preset=row["preset"],
                    family=row["family"],
                    dbpp=fmt(row["delta_bpp"], signed=True),
                    dpsnr=fmt(row["delta_psnr"], signed=True),
                    decode=fmt(row["max_decode_diff"]),
                    nonfinite=row["nonfinite_rows"],
                    mismatch=fmt(row["y_mismatch_frac"]),
                    geom=fmt(row["geometry_delta_rms"]),
                    entropy=fmt(row["index_entropy"]),
                )
            )
    lines.append("")
    lines.append(
        "EF-LIC decision: the zero preset exactly preserves the baseline payload/reconstruction, "
        "nonzero HCG geometry remains decoder reproducible, and no nonfinite rows appeared "
        "on these GPU0 smokes. The CLIC-tail smoke is especially useful because every tested "
        "nonzero preset improved mean emitted PSNR with `delta_bpp = 0`, but this is still "
        "a four-image PSNR-only contract check rather than a final RD/perceptual claim."
    )
    lines.append("")
    lines.append("## Controller Wiring")
    lines.append("")
    lines.append(
        "| check | passed | trained eval loss | rel loss | gate mean | fallback exact | nonfinite |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| E288 fallback-gated context smoke | {passed} | {loss} | {rel} | {gate} | {exact} | {nonfinite} |".format(
            passed=str(controller_smoke["all_checks_passed"]),
            loss=fmt(controller_smoke["trained_eval_loss"]),
            rel=fmt(controller_smoke["trained_eval_rel_loss"]),
            gate=fmt(controller_smoke["trained_eval_gate_mean"]),
            exact=fmt(controller_smoke["trained_eval_hard_base_exact"]),
            nonfinite=controller_smoke["trained_eval_nonfinite"],
        )
    )
    lines.append("")
    lines.append(
        "E234/E235 remain the key caveat: fixed presets show headroom and per-image oracle "
        "is strong, but simple held-out posthoc predictors are weak. Therefore the paper-main "
        "EF-LIC path should be an in-codec trained fallback controller, not a posthoc global "
        "selector."
    )
    lines.append("")
    lines.append("## Promotion Plan")
    lines.append("")
    lines.append("1. Implement EF-LIC `HCGBranchController` next to `_mean_scale`, preserving the existing support-buffer/RDC loop and exact zero fallback.")
    lines.append("2. Keep the original EF-LIC objective dominant; add only VQ/index/rate and false-positive/fallback regularizers that are directly tied to codec accounting.")
    lines.append("3. Evaluate force index `0` and `1` first, because the target is low-bitrate EF-LIC; report bpp, PSNR, MS-SSIM/LPIPS/DISTS where available, index entropy, code usage, selected fraction, and failure cases.")
    lines.append("4. For GLC, promote cap `0.0035` and cap `0.0030` selected replacement into longer paper-aligned runs; keep cap `0.0040` as an aggressive branch.")
    lines.append("5. Treat Kodak-only wins as insufficient. Use Kodak24 plus CLIC Professional validation as the minimum paper-facing controlled evidence; CLIC Mobile is optional stress testing, not the main benchmark.")
    lines.append("6. After these mid-scale checks pass, run paper-aligned full training/full evaluation for EF-LIC and GLC against their original settings.")
    lines.append("")
    lines.append("## Main Claim Wording")
    lines.append("")
    lines.append(
        "A safe current claim is: HCG-RVQ can be integrated as decoder-safe, "
        "hyperprior/context-conditioned local quantizer geometry in VQ-LIC systems. "
        "Short-cycle GLC evidence supports selected replacement under explicit signal/rate "
        "accounting, and EF-LIC smokes show the same contract can preserve the no-entropy "
        "codec path while exposing useful local geometry headroom. The next claim threshold "
        "is learned-controller EF-LIC/GLC evaluation under the original paper protocols."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for name, path in audit["inputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_prefix.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e291_eflic_glc_hcg_promotion_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.output_prefix)


if __name__ == "__main__":
    main()
