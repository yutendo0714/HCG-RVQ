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


def best_by_rd(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: as_float(row, "rd_score"))


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


METHODS = {
    "scalar": "Scalar baseline",
    "hcs": "HCS-RVQ",
    "hcg_bias010": "HCG gate0.25 bias010",
}

SPECS = {
    1234: {
        "scalar": {
            "sweep": "e140_scalar_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
            "feature": "e140_scalar_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
        },
        "hcs": {
            "sweep": "e140_hcs_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcs_lambda0018_seed1234_kodak_step500_feature_distribution.csv",
        },
        "hcg_bias010": {
            "sweep": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcg_gate025_bias010_lambda0018_seed1234_kodak_step250_feature_distribution.csv",
        },
    },
    2345: {
        "scalar": {
            "sweep": "e140_scalar_lambda0018_seed2345_kodak_checkpoint_sweep.csv",
            "feature": "e140_scalar_lambda0018_seed2345_kodak_step500_feature_distribution.csv",
        },
        "hcs": {
            "sweep": "e140_hcs_lambda0018_seed2345_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcs_lambda0018_seed2345_kodak_step250_feature_distribution.csv",
        },
        "hcg_bias010": {
            "sweep": "e140_hcg_gate025_bias010_lambda0018_seed2345_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcg_gate025_bias010_lambda0018_seed2345_kodak_step250_feature_distribution.csv",
        },
    },
}


def build_method_row(seed: int, method: str, spec: dict[str, str]) -> tuple[dict[str, object], dict[str, object]]:
    sweep_path = ANALYSIS / spec["sweep"]
    feature_path = ANALYSIS / spec["feature"]
    sweep_rows = read_csv(sweep_path)
    feature = read_csv(feature_path)[0]
    best = best_by_rd(sweep_rows)
    step250 = next((row for row in sweep_rows if int(float(row["step"])) == 250), None)
    step500 = next((row for row in sweep_rows if int(float(row["step"])) == 500), None)
    row = {
        "seed": seed,
        "method": method,
        "label": METHODS[method],
        "best_step": int(float(best["step"])),
        "rd_score": as_float(best, "rd_score"),
        "bpp": as_float(best, "bpp"),
        "bpp_y": as_float(best, "bpp_y"),
        "bpp_z": as_float(best, "bpp_z"),
        "psnr": as_float(best, "psnr"),
        "ms_ssim": as_float(best, "ms_ssim"),
        "step250_rd": as_float(step250, "rd_score") if step250 else math.nan,
        "step500_rd": as_float(step500, "rd_score") if step500 else math.nan,
        "step500_minus_step250": (
            as_float(step500, "rd_score") - as_float(step250, "rd_score")
            if step250 and step500
            else math.nan
        ),
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
    check = {
        "seed": seed,
        "method": method,
        "sweep_numeric_finite": all(finite_numeric(r) for r in sweep_rows),
        "feature_numeric_finite": finite_numeric(feature),
    }
    return row, check


def main() -> None:
    method_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for seed, seed_specs in SPECS.items():
        for method, spec in seed_specs.items():
            row, check = build_method_row(seed, method, spec)
            method_rows.append(row)
            checks.append(check)

    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in method_rows}
    seed_rows: list[dict[str, object]] = []
    for seed in sorted(SPECS):
        scalar = by_seed_method[(seed, "scalar")]
        hcs = by_seed_method[(seed, "hcs")]
        hcg = by_seed_method[(seed, "hcg_bias010")]
        seed_rows.append(
            {
                "seed": seed,
                "scalar_rd": scalar["rd_score"],
                "hcs_rd": hcs["rd_score"],
                "hcg_bias010_rd": hcg["rd_score"],
                "hcs_delta_vs_scalar": float(hcs["rd_score"]) - float(scalar["rd_score"]),
                "hcg_delta_vs_hcs": float(hcg["rd_score"]) - float(hcs["rd_score"]),
                "hcg_delta_vs_scalar": float(hcg["rd_score"]) - float(scalar["rd_score"]),
                "hcg_wins_hcs": float(hcg["rd_score"]) < float(hcs["rd_score"]),
                "hcg_qmse_delta_vs_hcs": float(hcg["rvq_latent_quant_mse"]) - float(hcs["rvq_latent_quant_mse"]),
                "hcg_dead_delta_vs_hcs": float(hcg["rvq_dead_code_ratio"]) - float(hcs["rvq_dead_code_ratio"]),
                "hcg_perplexity_delta_vs_hcs": float(hcg["rvq_perplexity"]) - float(hcs["rvq_perplexity"]),
                "hcg_y_error_delta_vs_hcs": float(hcg["y_error_rms"]) - float(hcs["y_error_rms"]),
                "hcg_householder_delta_rms": hcg["householder_delta_rms"],
                "hcg_best_step": hcg["best_step"],
                "hcg_step500_minus_step250": hcg["step500_minus_step250"],
                "hcs_best_step": hcs["best_step"],
                "hcs_step500_minus_step250": hcs["step500_minus_step250"],
            }
        )

    headline = {
        "lambda_rd": 0.0018,
        "num_seeds": len(seed_rows),
        "seeds": sorted(SPECS),
        "scalar_rd_mean": mean([float(row["scalar_rd"]) for row in seed_rows]),
        "hcs_rd_mean": mean([float(row["hcs_rd"]) for row in seed_rows]),
        "hcg_bias010_rd_mean": mean([float(row["hcg_bias010_rd"]) for row in seed_rows]),
        "hcs_delta_vs_scalar_mean": mean([float(row["hcs_delta_vs_scalar"]) for row in seed_rows]),
        "hcg_delta_vs_hcs_mean": mean([float(row["hcg_delta_vs_hcs"]) for row in seed_rows]),
        "hcg_delta_vs_scalar_mean": mean([float(row["hcg_delta_vs_scalar"]) for row in seed_rows]),
        "hcg_win_count_vs_hcs": sum(1 for row in seed_rows if bool(row["hcg_wins_hcs"])),
        "hcg_mean_dead_delta_vs_hcs": mean([float(row["hcg_dead_delta_vs_hcs"]) for row in seed_rows]),
        "hcg_mean_qmse_delta_vs_hcs": mean([float(row["hcg_qmse_delta_vs_hcs"]) for row in seed_rows]),
        "hcg_mean_perplexity_delta_vs_hcs": mean([float(row["hcg_perplexity_delta_vs_hcs"]) for row in seed_rows]),
        "all_numeric_finite": all(
            bool(check["sweep_numeric_finite"]) and bool(check["feature_numeric_finite"])
            for check in checks
        ),
    }

    package = {
        "headline": headline,
        "seed_summary": seed_rows,
        "method_summary": method_rows,
        "checks": checks,
    }
    (ANALYSIS / "e140_multirate_lambda0018_two_seed_package.json").write_text(
        json.dumps(package, indent=2) + "\n"
    )

    for suffix, rows in [
        ("seed_summary", seed_rows),
        ("method_summary", method_rows),
        ("checks", checks),
    ]:
        path = ANALYSIS / f"e140_multirate_lambda0018_two_seed_package.{suffix}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_lines = [
        "# E140 Multi-Rate Lambda0018 Two-Seed Package",
        "",
        "## Headline",
        "",
        f"- Seeds: `{headline['seeds']}`.",
        f"- Scalar mean RD: `{fmt(headline['scalar_rd_mean'])}`.",
        f"- HCS mean RD: `{fmt(headline['hcs_rd_mean'])}` (`{fmt(headline['hcs_delta_vs_scalar_mean'], signed=True)}` vs scalar).",
        f"- HCG bias010 mean RD: `{fmt(headline['hcg_bias010_rd_mean'])}` (`{fmt(headline['hcg_delta_vs_hcs_mean'], signed=True)}` vs HCS; `{fmt(headline['hcg_delta_vs_scalar_mean'], signed=True)}` vs scalar).",
        f"- HCG wins vs HCS: `{headline['hcg_win_count_vs_hcs']}/{headline['num_seeds']}`.",
        f"- Mean dead-code delta vs HCS: `{fmt(headline['hcg_mean_dead_delta_vs_hcs'], signed=True)}`.",
        f"- Mean qMSE delta vs HCS: `{fmt(headline['hcg_mean_qmse_delta_vs_hcs'], signed=True)}`.",
        f"- Mean perplexity delta vs HCS: `{fmt(headline['hcg_mean_perplexity_delta_vs_hcs'], signed=True)}`.",
        f"- Numeric finite checks: `{headline['all_numeric_finite']}`.",
        "",
        "## Interpretation",
        "",
        "The active low-rate HCG geometry result now repeats on two seeds. Both seeds select the step250 HCG checkpoint and improve RD over the matched HCS checkpoint. This is a method-strengthening signal, not yet a final paper rate curve: seed3456, holdout4096, and another lambda point are still required.",
        "",
        "The intermediate-feature picture is mixed in the useful way. Householder geometry is active in both seeds, but codebook usage worsens relative to HCS: dead-code increases and perplexity decreases. Therefore this result supports continuing active geometry at low rate, while preserving checkpoint selection and usage-control audits as required safeguards.",
        "",
        "## Seed Summary",
        "",
        "| seed | scalar RD | HCS RD | HCG RD | HCG-HCS | HCG step | H delta | qMSE delta | dead delta | perplexity delta | HCG step500-step250 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        md_lines.append(
            "| {seed} | {scalar} | {hcs} | {hcg} | {delta} | {step} | {hdelta} | {qmse} | {dead} | {perp} | {drift} |".format(
                seed=row["seed"],
                scalar=fmt(float(row["scalar_rd"])),
                hcs=fmt(float(row["hcs_rd"])),
                hcg=fmt(float(row["hcg_bias010_rd"])),
                delta=fmt(float(row["hcg_delta_vs_hcs"]), signed=True),
                step=row["hcg_best_step"],
                hdelta=fmt(float(row["hcg_householder_delta_rms"])),
                qmse=fmt(float(row["hcg_qmse_delta_vs_hcs"]), signed=True),
                dead=fmt(float(row["hcg_dead_delta_vs_hcs"]), signed=True),
                perp=fmt(float(row["hcg_perplexity_delta_vs_hcs"]), signed=True),
                drift=fmt(float(row["hcg_step500_minus_step250"]), signed=True),
            )
        )
    md_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `experiments/analysis/e140_multirate_lambda0018_two_seed_package.json`",
            "- `experiments/analysis/e140_multirate_lambda0018_two_seed_package.seed_summary.csv`",
            "- `experiments/analysis/e140_multirate_lambda0018_two_seed_package.method_summary.csv`",
            "- `experiments/analysis/e140_multirate_lambda0018_two_seed_package.checks.csv`",
        ]
    )
    (ANALYSIS / "e140_multirate_lambda0018_two_seed_package.md").write_text(
        "\n".join(md_lines) + "\n"
    )


if __name__ == "__main__":
    main()
