from __future__ import annotations

import csv
import json
from pathlib import Path

from build_e140_multirate_lambda0018_two_seed_package import (
    ANALYSIS,
    METHODS,
    ROOT,
    SPECS as TWO_SEED_SPECS,
    build_method_row,
    fmt,
    mean,
)


SPECS = {
    **TWO_SEED_SPECS,
    3456: {
        "scalar": {
            "sweep": "e140_scalar_lambda0018_seed3456_kodak_checkpoint_sweep.csv",
            "feature": "e140_scalar_lambda0018_seed3456_kodak_step500_feature_distribution.csv",
        },
        "hcs": {
            "sweep": "e140_hcs_lambda0018_seed3456_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcs_lambda0018_seed3456_kodak_step500_feature_distribution.csv",
        },
        "hcg_bias010": {
            "sweep": "e140_hcg_gate025_bias010_lambda0018_seed3456_kodak_checkpoint_sweep.csv",
            "feature": "e140_hcg_gate025_bias010_lambda0018_seed3456_kodak_step500_feature_distribution.csv",
        },
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
                "hcg_index_bpp_delta_vs_hcs": float(hcg["index_empirical_bpp"]) - float(hcs["index_empirical_bpp"]),
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
        "hcg_mean_y_error_delta_vs_hcs": mean([float(row["hcg_y_error_delta_vs_hcs"]) for row in seed_rows]),
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
    (ANALYSIS / "e140_multirate_lambda0018_three_seed_package.json").write_text(
        json.dumps(package, indent=2) + "\n"
    )
    write_csv(ANALYSIS / "e140_multirate_lambda0018_three_seed_package.seed_summary.csv", seed_rows)
    write_csv(ANALYSIS / "e140_multirate_lambda0018_three_seed_package.method_summary.csv", method_rows)
    write_csv(ANALYSIS / "e140_multirate_lambda0018_three_seed_package.checks.csv", checks)

    md_lines = [
        "# E140 Multi-Rate Lambda0018 Three-Seed Package",
        "",
        "## Headline",
        "",
        f"- Seeds: `{headline['seeds']}`.",
        f"- Scalar mean RD: `{fmt(headline['scalar_rd_mean'])}`.",
        f"- HCS mean RD: `{fmt(headline['hcs_rd_mean'])}` (`{fmt(headline['hcs_delta_vs_scalar_mean'], signed=True)}` vs scalar).",
        f"- HCG bias010 mean RD: `{fmt(headline['hcg_bias010_rd_mean'])}` (`{fmt(headline['hcg_delta_vs_hcs_mean'], signed=True)}` vs HCS; `{fmt(headline['hcg_delta_vs_scalar_mean'], signed=True)}` vs scalar).",
        f"- HCG wins vs HCS: `{headline['hcg_win_count_vs_hcs']}/{headline['num_seeds']}`.",
        f"- Mean y-error delta vs HCS: `{fmt(headline['hcg_mean_y_error_delta_vs_hcs'], signed=True)}`.",
        f"- Mean dead-code delta vs HCS: `{fmt(headline['hcg_mean_dead_delta_vs_hcs'], signed=True)}`.",
        f"- Mean qMSE delta vs HCS: `{fmt(headline['hcg_mean_qmse_delta_vs_hcs'], signed=True)}`.",
        f"- Mean perplexity delta vs HCS: `{fmt(headline['hcg_mean_perplexity_delta_vs_hcs'], signed=True)}`.",
        f"- Numeric finite checks: `{headline['all_numeric_finite']}`.",
        "",
        "## Interpretation",
        "",
        "The active low-rate HCG result remains useful but is no longer cleanly strong after adding seed3456. It wins two of three seeds and the mean is still slightly better than HCS, but seed3456 reverses the result and worsens RD by about `+0.029` vs HCS. This should be treated as a fragile method-strengthening signal, not a paper-main rate claim.",
        "",
        "The failure seed is informative. HCG remains active, but it worsens y-error, MS-SSIM, and index bpp relative to HCS. This points toward reliability or usage control rather than simply increasing geometry strength.",
        "",
        "## Seed Summary",
        "",
        "| seed | scalar RD | HCS RD | HCG RD | HCG-HCS | HCG step | H delta | yerr delta | qMSE delta | dead delta | perplexity delta | HCG step500-step250 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        md_lines.append(
            "| {seed} | {scalar} | {hcs} | {hcg} | {delta} | {step} | {hdelta} | {yerr} | {qmse} | {dead} | {perp} | {drift} |".format(
                seed=row["seed"],
                scalar=fmt(float(row["scalar_rd"])),
                hcs=fmt(float(row["hcs_rd"])),
                hcg=fmt(float(row["hcg_bias010_rd"])),
                delta=fmt(float(row["hcg_delta_vs_hcs"]), signed=True),
                step=row["hcg_best_step"],
                hdelta=fmt(float(row["hcg_householder_delta_rms"])),
                yerr=fmt(float(row["hcg_y_error_delta_vs_hcs"]), signed=True),
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
            "- `experiments/analysis/e140_multirate_lambda0018_three_seed_package.json`",
            "- `experiments/analysis/e140_multirate_lambda0018_three_seed_package.seed_summary.csv`",
            "- `experiments/analysis/e140_multirate_lambda0018_three_seed_package.method_summary.csv`",
            "- `experiments/analysis/e140_multirate_lambda0018_three_seed_package.checks.csv`",
        ]
    )
    (ANALYSIS / "e140_multirate_lambda0018_three_seed_package.md").write_text(
        "\n".join(md_lines) + "\n"
    )


if __name__ == "__main__":
    main()
