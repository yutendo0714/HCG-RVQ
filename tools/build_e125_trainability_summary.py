#!/usr/bin/env python3
"""Summarize E125 direct-HCG failure versus HCS warmup trainability."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_summary"
CASES = {
    "direct_hcg_old_loss_fail": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_direct_hcg_fail.json",
    "direct_hcg_lossguard": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_direct_hcg_lossguard.json",
    "hcs_warmup": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_hcs_warmup.json",
    "gated_hcg_initbias001_gate001": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_gated_hcg_initbias001_gate001.json",
    "staged_hcs30_gated_hcg": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias001_gate001.json",
    "staged_hcs30_gated_hcg_half": ANALYSIS_DIR / "e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias0005_gate0005.json",
}


def is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_case(name: str, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    train_rows = payload.get("train_rows", [])
    eval_rows = payload.get("eval_rows", [])
    first_bad_grad = None
    first_bad_output = None
    skipped_steps = 0
    for row in train_rows:
        grad_norm = row.get("grad_norm")
        grad_nonfinite = row.get("grad_nonfinite", 0)
        nonfinite = row.get("nonfinite", 0)
        skipped = row.get("skipped_step", 0)
        if skipped:
            skipped_steps += 1
        if first_bad_grad is None and (not is_finite(grad_norm) or float(grad_nonfinite or 0) > 0):
            first_bad_grad = int(row["step"])
        if first_bad_output is None and float(nonfinite or 0) > 0:
            first_bad_output = int(row["step"])
    eval_by_step = {int(row["step"]): row for row in eval_rows}
    step0 = eval_by_step.get(0, {})
    final_eval = eval_rows[-1] if eval_rows else {}
    return {
        "case": name,
        "source": str(path),
        "status": payload.get("status"),
        "variant": payload.get("config", {}).get("variant", "hcg_rvq_h"),
        "completed_steps": payload.get("completed_steps", len(train_rows)),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "first_bad_grad_step": first_bad_grad,
        "first_bad_output_step": first_bad_output,
        "skipped_steps": skipped_steps,
        "step0_rd": step0.get("rd_score"),
        "final_step": final_eval.get("step"),
        "final_rd": final_eval.get("rd_score"),
        "final_nonfinite": final_eval.get("nonfinite"),
        "step0_qmse": step0.get("latent_quant_mse"),
        "final_qmse": final_eval.get("latent_quant_mse"),
        "step0_dead": step0.get("dead_code_ratio"),
        "final_dead": final_eval.get("dead_code_ratio"),
        "step0_s_q_mean": step0.get("s_q_mean"),
        "final_s_q_mean": final_eval.get("s_q_mean"),
        "step0_householder_delta_rms": step0.get("householder_delta_rms"),
        "final_householder_delta_rms": final_eval.get("householder_delta_rms"),
        "step0_householder_v_abs_mean": step0.get("householder_v_abs_mean"),
        "final_householder_v_abs_mean": final_eval.get("householder_v_abs_mean"),
        "delta_rd": (final_eval.get("rd_score") - step0.get("rd_score"))
        if isinstance(final_eval.get("rd_score"), (int, float)) and isinstance(step0.get("rd_score"), (int, float))
        else None,
        "delta_qmse": (final_eval.get("latent_quant_mse") - step0.get("latent_quant_mse"))
        if isinstance(final_eval.get("latent_quant_mse"), (int, float)) and isinstance(step0.get("latent_quant_mse"), (int, float))
        else None,
        "delta_dead": (final_eval.get("dead_code_ratio") - step0.get("dead_code_ratio"))
        if isinstance(final_eval.get("dead_code_ratio"), (int, float)) and isinstance(step0.get("dead_code_ratio"), (int, float))
        else None,
    }


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    rows = [load_case(name, path) for name, path in CASES.items()]
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps({"cases": rows}, indent=2, sort_keys=True) + "\n")
    csv_path = OUT_PREFIX.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E125 Trainability Summary",
        "",
        "This audit separates a direct geometry failure from the safe shift/scale warmup path on the same frozen local CompressAI mbt2018_mean adapter pilot. These are trainability diagnostics, not quality or SOTA results.",
        "",
        "| case | status | completed | first bad grad | first bad output | final RD | delta RD | final qMSE | final dead | final H-delta | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {status} | {completed_steps} | {first_bad_grad_step} | {first_bad_output_step} | {final_rd} | {delta_rd} | {final_qmse} | {final_dead} | {final_householder_delta_rms} | {final_nonfinite} |".format(
                case=row["case"],
                status=row["status"],
                completed_steps=row["completed_steps"],
                first_bad_grad_step=fmt(row["first_bad_grad_step"]),
                first_bad_output_step=fmt(row["first_bad_output_step"]),
                final_rd=fmt(row["final_rd"]),
                delta_rd=fmt(row["delta_rd"]),
                final_qmse=fmt(row["final_qmse"]),
                final_dead=fmt(row["final_dead"]),
                final_householder_delta_rms=fmt(row["final_householder_delta_rms"]),
                final_nonfinite=fmt(row["final_nonfinite"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The old direct `hcg_rvq_h` failure was reproduced as a loss-plumbing issue: the forward pass was finite, but the first training row had non-finite gradient norm, and later checkpoints became non-finite. The fix is to avoid adding zero-weight conditioning terms to the graph, especially `householder_delta_rms=sqrt(0)`.",
            "- With the zero-coefficient loss guard, direct `hcg_rvq_h` completes all 30 steps on physical GPU0 with zero nonfinite outputs, zero nonfinite gradients, and no skipped optimizer steps. This clears the earlier NaN as an implementation bug rather than a device or dataset failure.",
            "- However, direct `hcg_rvq_h` still has `householder_delta_rms=0` and `householder_v_abs_mean=0` through step 30, so geometry is stable but inactive. The current optimizer is learning the HCS/RVQ path, not a useful Householder geometry yet.",
            "- `hcs_rvq` warmup also completes all 30 steps and improves qMSE from `0.002228` to `0.000488`, with dead-code ratio improving from `0.698242` to `0.505859`.",
            "- The gated/nonzero-bias geometry pilot also completes all 30 steps with zero nonfinite values and activates geometry: `householder_delta_rms` moves from about `7.69e-05` to `2.60e-04`, and `householder_v_abs_mean` moves from about `0.0074` to `0.0309`. Its RD is worse in this random-backbone smoke, so it is not a quality claim; it is a safe activation proof.",
            "- The staged HCS30 -> gated HCG run connects the two pieces directly. It loads the HCS warmup step30 adapter, resets only the Householder direction/gate, and completes another 30 steps with zero nonfinite values. Its eval RD improves slightly from `38.172090` to `38.171590` and qMSE improves from `0.000488` to `0.000437`, while dead-code worsens from `0.506836` to `0.579102`.",
            "- This makes the staged direction more credible than the scratch gated run, but the next strong-backbone path needs a codebook-usage/perplexity guard so active geometry does not buy qMSE by narrowing the used index set.",
            "- Halving the staged gate/bias also stays finite, but it does not solve the usage trade-off: RD is essentially flat/slightly worse (`38.172089` to `38.172128`), qMSE still improves (`0.000488` to `0.000428`), and dead-code still rises (`0.506836` to `0.564453`). So the next lever should be an explicit usage-aware constraint or selection rule, not merely smaller geometry amplitude.",
            "",
            "Artifacts:",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{csv_path}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
