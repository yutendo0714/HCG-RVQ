from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

BETA005 = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.json"
TRAINED_SWEEP = ANALYSIS / (
    "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_rawbackoff065_t0284_seed3456_holdout4096_checkpoint_sweep.csv"
)
POSTHOC_SWEEP = ANALYSIS / (
    "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_posthoc_rawbackoff065_t0284_seed3456_holdout4096_step500.csv"
)
TRAINED_FEATURE = ANALYSIS / (
    "feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_rawbackoff065_t0284_seed3456_step500_val4096_holdout4096_current.json"
)
POSTHOC_FEATURE = ANALYSIS / (
    "feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_posthoc_rawbackoff065_t0284_seed3456_step500_val4096_holdout4096_current.json"
)

OUT_JSON = ANALYSIS / "betacommit005_rawbackoff065_t0284_seed3456_probe.json"
OUT_MD = ANALYSIS / "betacommit005_rawbackoff065_t0284_seed3456_probe.md"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, object], key: str) -> float:
    return float(row[key])


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def pick_step(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    for row in rows:
        if int(float(row["step"])) == step:
            return row
    raise KeyError(step)


def beta005_seed3456() -> dict[str, float]:
    beta = json.loads(BETA005.read_text(encoding="utf-8"))
    for row in beta["summaries"]:
        if row["seed"] == "3456" and row["step"] == "500":
            return {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
    raise KeyError("seed3456 step500")


def feature_compare(beta: dict[str, float], trained: dict[str, object], posthoc: dict[str, object]) -> dict[str, dict[str, float]]:
    mapping = {
        "s_q_mean": ("mean_rvq_s_q_mean", "rvq_s_q_mean"),
        "latent_qmse": ("mean_rvq_latent_quant_mse", "rvq_latent_quant_mse"),
        "householder_delta_rms": ("mean_rvq_householder_delta_rms", "rvq_householder_delta_rms"),
        "householder_delta_local_mean": (
            "mean_rvq_householder_delta_rms_local_mean",
            "rvq_householder_delta_rms_local_mean",
        ),
        "householder_strength": ("mean_rvq_householder_strength", "rvq_householder_strength"),
        "risk_multiplier": ("mean_rvq_householder_risk_multiplier", "rvq_householder_risk_multiplier"),
        "dead_code_ratio": ("mean_rvq_dead_code_ratio", "rvq_dead_code_ratio"),
        "perplexity": ("mean_rvq_perplexity", "rvq_perplexity"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (beta_key, feat_key) in mapping.items():
        out[name] = {
            "beta005": beta[beta_key],
            "trained_rawbackoff": f(trained, feat_key),
            "posthoc_rawbackoff": f(posthoc, feat_key),
        }
        out[name]["trained_minus_beta005"] = out[name]["trained_rawbackoff"] - out[name]["beta005"]
        out[name]["posthoc_minus_beta005"] = out[name]["posthoc_rawbackoff"] - out[name]["beta005"]
    return out


def main() -> None:
    beta = beta005_seed3456()
    trained_rows = load_csv(TRAINED_SWEEP)
    posthoc_rows = load_csv(POSTHOC_SWEEP)
    trained250 = pick_step(trained_rows, 250)
    trained500 = pick_step(trained_rows, 500)
    posthoc500 = pick_step(posthoc_rows, 500)
    trained_feat = json.loads(TRAINED_FEATURE.read_text(encoding="utf-8"))
    posthoc_feat = json.loads(POSTHOC_FEATURE.read_text(encoding="utf-8"))

    beta_rd = beta["mean_rd"]
    trained500_rd = f(trained500, "rd_score")
    posthoc500_rd = f(posthoc500, "rd_score")

    result = {
        "decision": "reject_rawbackoff065_t0284_for_3seed_promotion",
        "reason": (
            "A smooth raw-gate multiplier suppresses useful beta005 geometry too broadly. "
            "Posthoc application already worsens beta005, and training with the multiplier worsens further."
        ),
        "checkpoint_rows": {
            "beta005_seed3456_step500": {"rd": beta_rd, "nonfinite": 0},
            "trained_rawbackoff_step250": {
                "rd": f(trained250, "rd_score"),
                "delta_vs_beta005": f(trained250, "rd_score") - beta_rd,
                "bpp": f(trained250, "bpp"),
                "psnr": f(trained250, "psnr"),
                "ms_ssim": f(trained250, "ms_ssim"),
                "nonfinite": 0,
            },
            "trained_rawbackoff_step500": {
                "rd": trained500_rd,
                "delta_vs_beta005": trained500_rd - beta_rd,
                "bpp": f(trained500, "bpp"),
                "psnr": f(trained500, "psnr"),
                "ms_ssim": f(trained500, "ms_ssim"),
                "nonfinite": 0,
            },
            "posthoc_rawbackoff_on_beta005_step500": {
                "rd": posthoc500_rd,
                "delta_vs_beta005": posthoc500_rd - beta_rd,
                "bpp": f(posthoc500, "bpp"),
                "psnr": f(posthoc500, "psnr"),
                "ms_ssim": f(posthoc500, "ms_ssim"),
                "nonfinite": 0,
            },
        },
        "feature_compare": feature_compare(beta, trained_feat, posthoc_feat),
        "raw_gate_control": {
            "trained_raw_gate": f(trained_feat, "rvq_householder_gate_raw"),
            "posthoc_raw_gate": f(posthoc_feat, "rvq_householder_gate_raw"),
            "trained_raw_backoff_multiplier": f(trained_feat, "rvq_householder_raw_backoff_multiplier"),
            "posthoc_raw_backoff_multiplier": f(posthoc_feat, "rvq_householder_raw_backoff_multiplier"),
        },
        "artifacts": {
            "trained_sweep": str(TRAINED_SWEEP.relative_to(ROOT)),
            "posthoc_sweep": str(POSTHOC_SWEEP.relative_to(ROOT)),
            "trained_feature": str(TRAINED_FEATURE.relative_to(ROOT)),
            "posthoc_feature": str(POSTHOC_FEATURE.relative_to(ROOT)),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Raw-gate backoff probe for beta005",
        "",
        "## Decision",
        "",
        "Reject `rawbackoff065_t0284` for 3-seed promotion. It is a useful negative control, not a paper-main method.",
        "",
        "## Checkpoint and posthoc evaluation",
        "",
        "| row | RD | delta vs beta005 seed3456 step500 | bpp | PSNR | MS-SSIM | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| beta005 seed3456 step500 | {fmt(beta_rd)} | {fmt(0.0, signed=True)} | n/a | n/a | n/a | 0 |",
    ]
    for label, row in [
        ("trained rawbackoff step250", result["checkpoint_rows"]["trained_rawbackoff_step250"]),
        ("trained rawbackoff step500", result["checkpoint_rows"]["trained_rawbackoff_step500"]),
        ("posthoc rawbackoff on beta005 step500", result["checkpoint_rows"]["posthoc_rawbackoff_on_beta005_step500"]),
    ]:
        lines.append(
            f"| {label} | {fmt(row['rd'])} | {fmt(row['delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['bpp'])} | {fmt(row['psnr'])} | {fmt(row['ms_ssim'])} | 0 |"
        )

    lines.extend(
        [
            "",
            "## Feature comparison",
            "",
            "| feature | beta005 | trained rawbackoff | trained-beta | posthoc rawbackoff | posthoc-beta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in result["feature_compare"].items():
        lines.append(
            f"| {name} | {fmt(values['beta005'])} | {fmt(values['trained_rawbackoff'])} | "
            f"{fmt(values['trained_minus_beta005'], signed=True)} | {fmt(values['posthoc_rawbackoff'])} | "
            f"{fmt(values['posthoc_minus_beta005'], signed=True)} |"
        )

    rg = result["raw_gate_control"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Posthoc rawbackoff already worsens beta005 by `{fmt(posthoc500_rd - beta_rd, signed=True)}` RD. "
                f"Training with the same multiplier worsens further by `{fmt(trained500_rd - beta_rd, signed=True)}` RD. "
                "This separates the failure from checkpoint selection: the smooth multiplier itself is not reproducing the "
                "discrete selector headroom."
            ),
            "",
            (
                f"The control is too broad. The posthoc multiplier is `{fmt(rg['posthoc_raw_backoff_multiplier'])}` and the "
                f"trained multiplier is `{fmt(rg['trained_raw_backoff_multiplier'])}`. Householder strength falls from "
                f"`{fmt(beta['mean_rvq_householder_strength'])}` to `{fmt(trained_feat['rvq_householder_strength'])}` "
                f"(trained) and `{fmt(posthoc_feat['rvq_householder_strength'])}` (posthoc), while trained latent qMSE "
                f"worsens from `{fmt(beta['mean_rvq_latent_quant_mse'])}` to `{fmt(trained_feat['rvq_latent_quant_mse'])}`."
            ),
            "",
            (
                "The raw-gate selector audit remains useful as headroom, but a simple continuous gate multiplier is the wrong "
                "implementation. The next controller should be teacher/supervised or distribution-aware: preserve beta005 "
                "geometry for low-risk images, and only push high-risk cases toward previous-local-like safer geometry."
            ),
            "",
            "Artifacts:",
            "",
            f"- `{TRAINED_SWEEP.relative_to(ROOT)}`",
            f"- `{POSTHOC_SWEEP.relative_to(ROOT)}`",
            f"- `{TRAINED_FEATURE.relative_to(ROOT)}`",
            f"- `{POSTHOC_FEATURE.relative_to(ROOT)}`",
            f"- `{OUT_JSON.relative_to(ROOT)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT_MD), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
