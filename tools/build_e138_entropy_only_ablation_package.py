from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_BASE = ANALYSIS / "e138_entropy_only_ablation_package"

SEEDS = [1234, 2345, 3456]
LATEST_STEP = 1_000_000_000_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def best_by_rd(rows: list[dict[str, str]]) -> dict[str, str]:
    min_rd = min(f(row, "rd_score") for row in rows)
    tied_rows = [row for row in rows if f(row, "rd_score") <= min_rd + 1e-6]

    def key(row: dict[str, str]) -> tuple[int, int, float]:
        step = i(row, "step", LATEST_STEP)
        latest_penalty = 1 if step >= LATEST_STEP else 0
        return (latest_penalty, step, f(row, "rd_score"))

    return min(tied_rows, key=key)


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def load_entropy_best(seed: int) -> dict[str, str]:
    return best_by_rd(read_csv(ANALYSIS / f"e138_entropy_only_seed{seed}_kodak_checkpoint_sweep.csv"))


def load_hcs_best(seed: int) -> dict[str, str]:
    name = "pilot_hcs_rvq_frozen_kodak.csv" if seed == 1234 else f"pilot_hcs_rvq_frozen_seed{seed}_kodak.csv"
    return best_by_rd(read_csv(ANALYSIS / name))


def load_feature(seed: int) -> dict[str, str]:
    return read_csv(ANALYSIS / f"e138_entropy_only_seed{seed}_kodak_step500_feature_distribution.csv")[0]


def main() -> None:
    global_best = best_by_rd(read_csv(ANALYSIS / "pilot_global_rvq_frozen_kodak.csv"))

    per_seed: list[dict[str, object]] = []
    checkpoint_choices: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        entropy = load_entropy_best(seed)
        hcs = load_hcs_best(seed)
        feature = load_feature(seed)

        row = {
            "seed": seed,
            "entropy_step": i(entropy, "step"),
            "hcs_step": i(hcs, "step"),
            "entropy_rd": f(entropy, "rd_score"),
            "hcs_rd": f(hcs, "rd_score"),
            "entropy_minus_hcs_rd": f(entropy, "rd_score") - f(hcs, "rd_score"),
            "entropy_bpp": f(entropy, "bpp"),
            "hcs_bpp": f(hcs, "bpp"),
            "entropy_minus_hcs_bpp": f(entropy, "bpp") - f(hcs, "bpp"),
            "entropy_bpp_y": f(entropy, "bpp_y"),
            "hcs_bpp_y": f(hcs, "bpp_y"),
            "entropy_minus_hcs_bpp_y": f(entropy, "bpp_y") - f(hcs, "bpp_y"),
            "entropy_mse": f(entropy, "mse"),
            "hcs_mse": f(hcs, "mse"),
            "entropy_minus_hcs_mse": f(entropy, "mse") - f(hcs, "mse"),
            "entropy_psnr": f(entropy, "psnr"),
            "hcs_psnr": f(hcs, "psnr"),
            "entropy_minus_hcs_psnr": f(entropy, "psnr") - f(hcs, "psnr"),
            "entropy_ms_ssim": f(entropy, "ms_ssim"),
            "hcs_ms_ssim": f(hcs, "ms_ssim"),
            "entropy_minus_hcs_ms_ssim": f(entropy, "ms_ssim") - f(hcs, "ms_ssim"),
        }
        if seed == 1234:
            row["entropy_minus_global_rd"] = f(entropy, "rd_score") - f(global_best, "rd_score")
            row["entropy_minus_global_bpp_y"] = f(entropy, "bpp_y") - f(global_best, "bpp_y")
        else:
            row["entropy_minus_global_rd"] = math.nan
            row["entropy_minus_global_bpp_y"] = math.nan
        per_seed.append(row)

        checkpoint_choices.append(
            {
                "seed": seed,
                "method": "entropy_only_index_global_rvq",
                "best_step": i(entropy, "step"),
                "best_rd": f(entropy, "rd_score"),
                "best_checkpoint": entropy["checkpoint"],
            }
        )
        checkpoint_choices.append(
            {
                "seed": seed,
                "method": "hcs_rvq",
                "best_step": i(hcs, "step"),
                "best_rd": f(hcs, "rd_score"),
                "best_checkpoint": hcs["checkpoint"],
            }
        )

        feature_rows.append(
            {
                "seed": seed,
                "rd_score": f(feature, "rd_score"),
                "bpp": f(feature, "bpp"),
                "bpp_y": f(feature, "bpp_y"),
                "y_error_rms": f(feature, "y_error_rms"),
                "rvq_latent_quant_mse": f(feature, "rvq_latent_quant_mse"),
                "rvq_dead_code_ratio": f(feature, "rvq_dead_code_ratio"),
                "rvq_perplexity": f(feature, "rvq_perplexity"),
                "index_empirical_entropy": f(feature, "index_empirical_entropy"),
                "index_empirical_bpp": f(feature, "index_empirical_bpp"),
                "index_dead_code_ratio": f(feature, "index_dead_code_ratio"),
                "global_s_q_mean": f(feature, "global_s_q_mean"),
                "u_std": f(feature, "u_std"),
            }
        )

    summary = {
        "experiment": "E138 entropy-only / HVQ-like ablation",
        "status": "done",
        "interpretation": "Entropy-only index conditioning is a strong and necessary ablation, but it is not the final HCG geometry claim.",
        "protocol": "Kodak24, checkpoint-selected per method and seed by minimum RD score, CUDA device 0.",
        "entropy_mean_rd": mean([float(row["entropy_rd"]) for row in per_seed]),
        "hcs_mean_rd": mean([float(row["hcs_rd"]) for row in per_seed]),
        "entropy_minus_hcs_mean_rd": mean([float(row["entropy_minus_hcs_rd"]) for row in per_seed]),
        "wins_vs_hcs": sum(1 for row in per_seed if float(row["entropy_minus_hcs_rd"]) < 0),
        "num_seeds": len(SEEDS),
        "entropy_mean_bpp": mean([float(row["entropy_bpp"]) for row in per_seed]),
        "hcs_mean_bpp": mean([float(row["hcs_bpp"]) for row in per_seed]),
        "entropy_mean_bpp_y": mean([float(row["entropy_bpp_y"]) for row in per_seed]),
        "hcs_mean_bpp_y": mean([float(row["hcs_bpp_y"]) for row in per_seed]),
        "entropy_mean_psnr": mean([float(row["entropy_psnr"]) for row in per_seed]),
        "hcs_mean_psnr": mean([float(row["hcs_psnr"]) for row in per_seed]),
        "entropy_mean_ms_ssim": mean([float(row["entropy_ms_ssim"]) for row in per_seed]),
        "hcs_mean_ms_ssim": mean([float(row["hcs_ms_ssim"]) for row in per_seed]),
        "seed1234_global_rd": f(global_best, "rd_score"),
        "seed1234_entropy_minus_global_rd": per_seed[0]["entropy_minus_global_rd"],
        "feature_mean_index_empirical_bpp": mean([float(row["index_empirical_bpp"]) for row in feature_rows]),
        "feature_mean_rvq_dead_code_ratio": mean([float(row["rvq_dead_code_ratio"]) for row in feature_rows]),
        "feature_mean_rvq_latent_quant_mse": mean([float(row["rvq_latent_quant_mse"]) for row in feature_rows]),
        "feature_mean_y_error_rms": mean([float(row["y_error_rms"]) for row in feature_rows]),
    }

    package = {
        "summary": summary,
        "per_seed": per_seed,
        "checkpoint_choices": checkpoint_choices,
        "feature_rows": feature_rows,
        "inputs": {
            "entropy_sweeps": [
                str((ANALYSIS / f"e138_entropy_only_seed{seed}_kodak_checkpoint_sweep.csv").relative_to(ROOT))
                for seed in SEEDS
            ],
            "entropy_features": [
                str((ANALYSIS / f"e138_entropy_only_seed{seed}_kodak_step500_feature_distribution.csv").relative_to(ROOT))
                for seed in SEEDS
            ],
            "hcs_sweeps": [
                "experiments/analysis/pilot_hcs_rvq_frozen_kodak.csv",
                "experiments/analysis/pilot_hcs_rvq_frozen_seed2345_kodak.csv",
                "experiments/analysis/pilot_hcs_rvq_frozen_seed3456_kodak.csv",
            ],
            "global_reference": "experiments/analysis/pilot_global_rvq_frozen_kodak.csv",
        },
    }

    OUT_BASE.with_suffix(".json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_BASE.with_suffix(".per_seed.csv"), per_seed)
    write_csv(OUT_BASE.with_suffix(".checkpoint_choices.csv"), checkpoint_choices)
    write_csv(OUT_BASE.with_suffix(".features.csv"), feature_rows)
    write_csv(OUT_BASE.with_suffix(".summary.csv"), [summary])

    lines = [
        "# E138 Entropy-Only / HVQ-like Ablation Package",
        "",
        "## Conclusion",
        "",
        (
            "The entropy-only index-conditioned Global RVQ row is a real and important ablation. "
            "It improves over the single-seed Global RVQ reference and is competitive with checkpoint-selected HCS on Kodak24. "
            "This makes the paper story safer because HCG geometry can be compared against a hyperprior-index-prior-only control, not only against plain Global RVQ."
        ),
        "",
        (
            "It should not replace the main HCG geometry claim. "
            "The prompt's core claim is that hyperprior generates local quantizer geometry; entropy-only only changes the index entropy path while keeping global normalization and no local shift/scale/Householder geometry."
        ),
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| entropy-only mean RD | {fmt(summary['entropy_mean_rd'])} |",
        f"| HCS mean RD | {fmt(summary['hcs_mean_rd'])} |",
        f"| entropy-only - HCS mean RD | {fmt(summary['entropy_minus_hcs_mean_rd'], True)} |",
        f"| wins vs HCS | {summary['wins_vs_hcs']} / {summary['num_seeds']} |",
        f"| entropy-only mean bpp_y | {fmt(summary['entropy_mean_bpp_y'])} |",
        f"| HCS mean bpp_y | {fmt(summary['hcs_mean_bpp_y'])} |",
        f"| entropy-only mean PSNR | {fmt(summary['entropy_mean_psnr'])} |",
        f"| HCS mean PSNR | {fmt(summary['hcs_mean_psnr'])} |",
        f"| entropy-only mean MS-SSIM | {fmt(summary['entropy_mean_ms_ssim'])} |",
        f"| HCS mean MS-SSIM | {fmt(summary['hcs_mean_ms_ssim'])} |",
        f"| seed1234 entropy-only - Global RVQ RD | {fmt(summary['seed1234_entropy_minus_global_rd'], True)} |",
        f"| mean empirical index bpp | {fmt(summary['feature_mean_index_empirical_bpp'])} |",
        f"| mean RVQ dead-code ratio | {fmt(summary['feature_mean_rvq_dead_code_ratio'])} |",
        "",
        "## Per-Seed Checkpoint-Selected RD",
        "",
        "| seed | entropy step | entropy RD | HCS step | HCS RD | entropy-HCS | entropy bpp_y | HCS bpp_y | entropy MS-SSIM | HCS MS-SSIM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["entropy_step"]),
                    fmt(float(row["entropy_rd"])),
                    str(row["hcs_step"]),
                    fmt(float(row["hcs_rd"])),
                    fmt(float(row["entropy_minus_hcs_rd"]), True),
                    fmt(float(row["entropy_bpp_y"])),
                    fmt(float(row["hcs_bpp_y"])),
                    fmt(float(row["entropy_ms_ssim"])),
                    fmt(float(row["hcs_ms_ssim"])),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Intermediate Feature Readout",
        "",
        "| seed | y_error_rms | latent qMSE | dead-code | perplexity | empirical index bpp | global_s_q_mean | u_std |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in feature_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt(float(row["y_error_rms"])),
                    fmt(float(row["rvq_latent_quant_mse"])),
                    fmt(float(row["rvq_dead_code_ratio"])),
                    fmt(float(row["rvq_perplexity"])),
                    fmt(float(row["index_empirical_bpp"])),
                    fmt(float(row["global_s_q_mean"])),
                    fmt(float(row["u_std"])),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paper-Use Guidance",
        "",
        "- Use this row as the required HVQ-like / entropy-only ablation requested by `docs/prompt.txt`.",
        "- Report it as a strong control, not as the final method. It validates that index entropy conditioning helps, then leaves room to test whether local geometry adds beyond it.",
        "- Be cautious with MS-SSIM wording: HCS keeps higher MS-SSIM on Kodak even when entropy-only is better on MSE-based RD.",
        "- Next paper-facing step: run the same entropy-only row on the OpenImages holdout/start8192 protocols used by dz014/dz018, then compare against HCG geometry under the same split.",
        "",
    ]
    OUT_BASE.with_suffix(".md").write_text("\n".join(lines))

    print(f"wrote {OUT_BASE.with_suffix('.json')}")
    print(f"entropy-HCS mean RD delta: {summary['entropy_minus_hcs_mean_rd']:+.6f}")
    print(f"wins vs HCS: {summary['wins_vs_hcs']} / {summary['num_seeds']}")


if __name__ == "__main__":
    main()
