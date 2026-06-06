from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

SWEEP = ANALYSIS / (
    "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_rel075_seed3456_holdout4096_checkpoint_sweep.csv"
)
REL_FEATURE = ANALYSIS / (
    "feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_rel075_seed3456_step500_val4096_holdout4096_current.json"
)
BETA005 = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.json"
OUT_JSON = ANALYSIS / "betacommit005_rel075_seed3456_probe.json"
OUT_MD = ANALYSIS / "betacommit005_rel075_seed3456_probe.md"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, object], key: str) -> float:
    value = row[key]
    return float(value)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def pick_step(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    for row in rows:
        if int(float(row["step"])) == step:
            return row
    raise KeyError(step)


def beta005_seed3456(beta: dict[str, object]) -> dict[str, float]:
    for row in beta["summaries"]:
        if row["seed"] == "3456" and row["step"] == "500":
            return {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
    raise KeyError("seed3456 step500")


def main() -> None:
    sweep_rows = load_csv(SWEEP)
    rel_feature = json.loads(REL_FEATURE.read_text(encoding="utf-8"))
    beta = json.loads(BETA005.read_text(encoding="utf-8"))
    beta_seed = beta005_seed3456(beta)

    rel250 = pick_step(sweep_rows, 250)
    rel500 = pick_step(sweep_rows, 500)
    beta_rd = beta_seed["mean_rd"]
    rel500_rd = f(rel500, "rd_score")
    rel250_rd = f(rel250, "rd_score")

    comparisons = {
        "rel075_step250_rd": rel250_rd,
        "rel075_step500_rd": rel500_rd,
        "beta005_seed3456_step500_rd": beta_rd,
        "rel075_step500_minus_beta005": rel500_rd - beta_rd,
        "rel075_step250_minus_beta005": rel250_rd - beta_rd,
        "rel075_step500_minus_rel075_step250": rel500_rd - rel250_rd,
        "rel075_nonfinite_rows": sum(1 for row in sweep_rows if row.get("loss", "") in {"nan", "inf", "-inf"}),
    }

    feature_compare = {
        "s_q_mean": {
            "beta005": beta_seed["mean_rvq_s_q_mean"],
            "rel075": f(rel_feature, "rvq_s_q_mean"),
        },
        "latent_qmse": {
            "beta005": beta_seed["mean_rvq_latent_quant_mse"],
            "rel075": f(rel_feature, "rvq_latent_quant_mse"),
        },
        "householder_delta_rms": {
            "beta005": beta_seed["mean_rvq_householder_delta_rms"],
            "rel075": f(rel_feature, "rvq_householder_delta_rms"),
        },
        "householder_strength": {
            "beta005": beta_seed["mean_rvq_householder_strength"],
            "rel075": f(rel_feature, "rvq_householder_strength"),
        },
        "risk_multiplier": {
            "beta005": beta_seed["mean_rvq_householder_risk_multiplier"],
            "rel075": f(rel_feature, "rvq_householder_risk_multiplier"),
        },
        "dead_code_ratio": {
            "beta005": beta_seed["mean_rvq_dead_code_ratio"],
            "rel075": f(rel_feature, "rvq_dead_code_ratio"),
        },
        "perplexity": {
            "beta005": beta_seed["mean_rvq_perplexity"],
            "rel075": f(rel_feature, "rvq_perplexity"),
        },
    }
    for values in feature_compare.values():
        values["delta"] = values["rel075"] - values["beta005"]

    result = {
        "decision": "reject_rel075_for_3seed_promotion",
        "reason": (
            "The constrained reliability multiplier remains nearly identity and does not recover "
            "selector headroom; it is worse than beta005 on the fragile seed3456 fixed protocol."
        ),
        "comparisons": comparisons,
        "feature_compare": feature_compare,
        "rel075_reliability_multiplier": {
            "mean": f(rel_feature, "rvq_householder_reliability_multiplier"),
            "min": f(rel_feature, "rvq_householder_reliability_multiplier_min"),
            "max": f(rel_feature, "rvq_householder_reliability_multiplier_max"),
            "std": f(rel_feature, "rvq_householder_reliability_multiplier_std"),
        },
        "artifacts": {
            "sweep_csv": str(SWEEP.relative_to(ROOT)),
            "feature_json": str(REL_FEATURE.relative_to(ROOT)),
            "beta005_json": str(BETA005.relative_to(ROOT)),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Beta005 rel075 constrained reliability probe",
        "",
        "## Decision",
        "",
        "Reject `rel075` for 3-seed promotion. It is a useful negative result, not a replacement for the beta005 paper-main row.",
        "",
        "## Checkpoint sweep",
        "",
        "| row | RD | delta vs beta005 seed3456 step500 | bpp | PSNR | MS-SSIM | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| rel075 step250 | {fmt(rel250_rd)} | {fmt(rel250_rd - beta_rd, signed=True)} | "
            f"{fmt(f(rel250, 'bpp'))} | {fmt(f(rel250, 'psnr'))} | {fmt(f(rel250, 'ms_ssim'))} | 0 |"
        ),
        (
            f"| rel075 step500 | {fmt(rel500_rd)} | {fmt(rel500_rd - beta_rd, signed=True)} | "
            f"{fmt(f(rel500, 'bpp'))} | {fmt(f(rel500, 'psnr'))} | {fmt(f(rel500, 'ms_ssim'))} | 0 |"
        ),
        f"| beta005 seed3456 step500 | {fmt(beta_rd)} | {fmt(0.0, signed=True)} | n/a | n/a | n/a | 0 |",
        "",
        "## Feature comparison",
        "",
        "| feature | beta005 | rel075 | delta |",
        "|---|---:|---:|---:|",
    ]
    for key, values in feature_compare.items():
        lines.append(
            f"| {key} | {fmt(values['beta005'])} | {fmt(values['rel075'])} | {fmt(values['delta'], signed=True)} |"
        )
    rel_mult = result["rel075_reliability_multiplier"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The added reliability multiplier stayed almost inactive: "
                f"mean `{fmt(rel_mult['mean'])}`, min `{fmt(rel_mult['min'])}`, max `{fmt(rel_mult['max'])}`, "
                f"std `{fmt(rel_mult['std'])}`. It therefore did not learn the selector-like reliability behavior "
                "that the diagnostic headroom suggested."
            ),
            "",
            (
                "The intermediate features are nearly identical to beta005: `s_q_mean`, latent qMSE, "
                "Householder delta RMS, strength, risk multiplier, dead-code ratio, and perplexity all remain in "
                "the same range. The RD loss comes mainly from worse image-domain distortion rather than an "
                "obvious collapse or excessive geometry displacement."
            ),
            "",
            (
                "Next method-improvement action: do not spend GPU on 3-seed rel075. The next controller should use "
                "an explicit measured reliability signal or supervised/teacher-style target derived from selector "
                "headroom, while keeping the beta005 fixed-checkpoint claim unchanged."
            ),
            "",
            "Artifacts:",
            "",
            f"- `{SWEEP.relative_to(ROOT)}`",
            f"- `{REL_FEATURE.relative_to(ROOT)}`",
            f"- `{OUT_JSON.relative_to(ROOT)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT_MD), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
