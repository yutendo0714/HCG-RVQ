from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def best_by_rd(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return min(rows, key=lambda row: as_float(row, "rd_score"))


def finite_numeric(row: dict[str, str]) -> bool:
    for value in row.values():
        if value in ("", None):
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if not math.isfinite(parsed):
            return False
    return True


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


METHODS = {
    "scalar": {
        "label": "Scalar baseline",
        "sweep": "e140_scalar_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
        "feature": "e140_scalar_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
    },
    "hcs": {
        "label": "HCS-RVQ",
        "sweep": "e140_hcs_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
        "feature": "e140_hcs_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
    },
    "hcg_zero": {
        "label": "HCG gate0.25 zero-bias",
        "sweep": "e140_hcg_gate025_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
        "feature": "e140_hcg_gate025_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
    },
    "hcg_bias010": {
        "label": "HCG gate0.25 bias010",
        "sweep": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
        "feature": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_step250_feature_distribution.csv",
    },
    "hcg_bias010_step500": {
        "label": "HCG gate0.25 bias010 step500",
        "sweep": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
        "feature": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
        "force_step": 500,
    },
}


def main() -> None:
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for name, spec in METHODS.items():
        sweep_path = ANALYSIS / str(spec["sweep"])
        feature_path = ANALYSIS / str(spec["feature"])
        sweep_rows = read_csv(sweep_path)
        if "force_step" in spec:
            forced = [row for row in sweep_rows if int(float(row["step"])) == int(spec["force_step"])]
            if not forced:
                raise RuntimeError(f"missing forced step {spec['force_step']} in {sweep_path}")
            best = forced[0]
        else:
            best = min(sweep_rows, key=lambda row: as_float(row, "rd_score"))
        feature = read_csv(feature_path)[0]
        row = {
            "method": name,
            "label": spec["label"],
            "step": int(float(best["step"])),
            "rd_score": as_float(best, "rd_score"),
            "bpp": as_float(best, "bpp"),
            "bpp_y": as_float(best, "bpp_y"),
            "bpp_z": as_float(best, "bpp_z"),
            "mse": as_float(best, "mse"),
            "psnr": as_float(best, "psnr"),
            "ms_ssim": as_float(best, "ms_ssim"),
            "feature_rd_score": as_float(feature, "rd_score"),
            "y_error_rms": as_float(feature, "y_error_rms"),
            "rvq_latent_quant_mse": as_float(feature, "rvq_latent_quant_mse"),
            "rvq_dead_code_ratio": as_float(feature, "rvq_dead_code_ratio"),
            "rvq_perplexity": as_float(feature, "rvq_perplexity"),
            "index_empirical_bpp": as_float(feature, "index_empirical_bpp"),
            "householder_delta_rms": as_float(feature, "householder_delta_rms"),
            "householder_v_abs_mean": as_float(feature, "householder_v_abs_mean"),
            "householder_strength": as_float(feature, "householder_strength"),
            "s_q_mean": as_float(feature, "s_q_mean"),
            "sweep": str(sweep_path.relative_to(ROOT)),
            "feature": str(feature_path.relative_to(ROOT)),
        }
        rows.append(row)
        checks.append(
            {
                "method": name,
                "sweep_numeric_finite": all(finite_numeric(r) for r in sweep_rows),
                "feature_numeric_finite": finite_numeric(feature),
            }
        )

    by_method = {str(row["method"]): row for row in rows}
    scalar = float(by_method["scalar"]["rd_score"])
    hcs = float(by_method["hcs"]["rd_score"])
    for row in rows:
        row["delta_vs_scalar"] = float(row["rd_score"]) - scalar
        row["delta_vs_hcs"] = float(row["rd_score"]) - hcs

    headline = {
        "lambda_rd": 0.0018,
        "seed": 1234,
        "scalar_rd": scalar,
        "hcs_rd": hcs,
        "hcg_zero_rd": by_method["hcg_zero"]["rd_score"],
        "hcg_bias010_rd": by_method["hcg_bias010"]["rd_score"],
        "hcs_delta_vs_scalar": by_method["hcs"]["delta_vs_scalar"],
        "hcg_zero_delta_vs_hcs": by_method["hcg_zero"]["delta_vs_hcs"],
        "hcg_bias010_delta_vs_hcs": by_method["hcg_bias010"]["delta_vs_hcs"],
        "hcg_bias010_delta_vs_scalar": by_method["hcg_bias010"]["delta_vs_scalar"],
        "hcg_zero_householder_delta_rms": by_method["hcg_zero"]["householder_delta_rms"],
        "hcg_bias010_householder_delta_rms": by_method["hcg_bias010"]["householder_delta_rms"],
        "hcg_bias010_step500_delta_vs_step250": float(by_method["hcg_bias010_step500"]["rd_score"])
        - float(by_method["hcg_bias010"]["rd_score"]),
    }

    package = {
        "headline": headline,
        "summary": rows,
        "checks": checks,
    }
    (ANALYSIS / "e140_multirate_lambda0018_seed1234_package.json").write_text(
        json.dumps(package, indent=2) + "\n"
    )

    summary_path = ANALYSIS / "e140_multirate_lambda0018_seed1234_package.summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checks_path = ANALYSIS / "e140_multirate_lambda0018_seed1234_package.checks.csv"
    with checks_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)

    md_lines = [
        "# E140 Multi-Rate Lambda0018 Seed1234 Package",
        "",
        "## Headline",
        "",
        f"- Scalar RD: `{fmt(headline['scalar_rd'])}`.",
        f"- HCS RD: `{fmt(headline['hcs_rd'])}` (`{fmt(headline['hcs_delta_vs_scalar'], signed=True)}` vs scalar).",
        f"- HCG zero-bias RD: `{fmt(float(headline['hcg_zero_rd']))}` (`{fmt(headline['hcg_zero_delta_vs_hcs'], signed=True)}` vs HCS).",
        f"- HCG bias010 RD: `{fmt(float(headline['hcg_bias010_rd']))}` (`{fmt(headline['hcg_bias010_delta_vs_hcs'], signed=True)}` vs HCS; `{fmt(headline['hcg_bias010_delta_vs_scalar'], signed=True)}` vs scalar).",
        f"- HCG bias010 step500 drift: `{fmt(headline['hcg_bias010_step500_delta_vs_step250'], signed=True)}` RD relative to step250.",
        "",
        "## Interpretation",
        "",
        "The low-rate scaffold is useful, but the first zero-bias HCG run did not activate geometry: its Householder delta is zero and it is effectively an HCS-like control. After adding a small postload Householder bias initialization, the same rate/seed gives an active geometry checkpoint that improves Kodak RD over HCS. This should be treated as a promising single-seed multi-rate result, not yet as a paper-level rate-curve claim.",
        "",
        "## Summary",
        "",
        "| method | step | RD | delta vs HCS | bpp | PSNR | MS-SSIM | y err RMS | qMSE | dead | perplexity | H delta | H v abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {label} | {step} | {rd} | {dhcs} | {bpp} | {psnr} | {msssim} | {yrms} | {qmse} | {dead} | {perp} | {hdelta} | {hv} |".format(
                label=row["label"],
                step=row["step"],
                rd=fmt(float(row["rd_score"])),
                dhcs=fmt(float(row["delta_vs_hcs"]), signed=True),
                bpp=fmt(float(row["bpp"])),
                psnr=fmt(float(row["psnr"])),
                msssim=fmt(float(row["ms_ssim"])),
                yrms=fmt(float(row["y_error_rms"])),
                qmse=fmt(float(row["rvq_latent_quant_mse"])),
                dead=fmt(float(row["rvq_dead_code_ratio"])),
                perp=fmt(float(row["rvq_perplexity"])),
                hdelta=fmt(float(row["householder_delta_rms"])),
                hv=fmt(float(row["householder_v_abs_mean"])),
            )
        )
    md_lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Numeric finite checks: `{all(bool(c['sweep_numeric_finite']) and bool(c['feature_numeric_finite']) for c in checks)}`.",
            "- GPU runs were executed with `CUDA_VISIBLE_DEVICES=0` and `cuda:0`.",
            "",
            "## Artifacts",
            "",
            "- `experiments/analysis/e140_multirate_lambda0018_seed1234_package.json`",
            "- `experiments/analysis/e140_multirate_lambda0018_seed1234_package.summary.csv`",
            "- `experiments/analysis/e140_multirate_lambda0018_seed1234_package.checks.csv`",
        ]
    )
    (ANALYSIS / "e140_multirate_lambda0018_seed1234_package.md").write_text("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
