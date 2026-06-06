#!/usr/bin/env python3
"""Evaluate E146 low-rate bias010 head-only reliability controller on holdout4096."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_e143_lowrate_bias010_holdout_selector as e143
from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
BASE_PREFIX = ANALYSIS / "e146_lowrate_bias010_teacher_headonly_holdout4096"
E143_PAIRS = ANALYSIS / "e143_lowrate_bias010_holdout4096_selector.pairs.csv"
DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
SEED = "3456"
RUN_NAME = "pilot_hcg_rvq_h_gate025_bias010_teacher_transfer8192_relmin000_rho050_headonly_g64_l1_k128_lambda0018_seed3456"
CONFIG = ROOT / "configs/pilot_hcg_rvq_h_gate025_bias010_teacher_transfer8192_relmin000_rho050_headonly_lambda0018_seed3456.yaml"
CHECKPOINT_DIR = ROOT / "experiments" / RUN_NAME
DEFAULT_START_INDEX = 4096
DEFAULT_MAX_IMAGES = 4096


def parse_value(value: str) -> object:
    if value == "":
        return ""
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def q95(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    return finite[min(len(finite) - 1, math.ceil(0.95 * len(finite)) - 1)]


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def output_prefix(max_images: int, start_index: int) -> Path:
    if max_images == DEFAULT_MAX_IMAGES and start_index == DEFAULT_START_INDEX:
        return BASE_PREFIX
    return ANALYSIS / f"e146_lowrate_bias010_teacher_headonly_start{start_index}_n{max_images}"


def evaluate_steps(steps: list[int], start_index: int, max_images: int) -> list[dict[str, object]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows_all: list[dict[str, object]] = []
    for step in steps:
        checkpoint = CHECKPOINT_DIR / f"checkpoint_step_{step}.pth.tar"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        rows, _summary = evaluate_mode(
            mode="exact",
            config_path=str(CONFIG),
            checkpoint_path=str(checkpoint),
            data_root=DATA_ROOT,
            device=device,
            max_images=max_images,
            start_index=start_index,
            patch_size=None,
            reference={},
        )
        for row in rows:
            row.update(
                {
                    "seed": SEED,
                    "step": step,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "config": str(CONFIG.relative_to(ROOT)),
                }
            )
        rows_all.extend(rows)
    return rows_all


def align(rows: list[dict[str, object]], refs: list[dict[str, object]]) -> list[dict[str, object]]:
    ref_by_path = {str(row["path"]): row for row in refs if str(row["seed"]) == SEED}
    out: list[dict[str, object]] = []
    for row in rows:
        ref = ref_by_path[str(row["path"])]
        rd = float(row["rd_score"])
        hcs_rd = float(ref["hcs_rd"])
        fixed_rd = float(ref["hcg_rd"])
        item: dict[str, object] = {
            "seed": SEED,
            "step": int(row["step"]),
            "path": row["path"],
            "hcs_rd": hcs_rd,
            "fixed_hcg_rd": fixed_rd,
            "teacher_head_rd": rd,
            "fixed_hcg_minus_hcs": float(ref["hcg_minus_hcs"]),
            "teacher_head_minus_hcs": rd - hcs_rd,
            "teacher_head_minus_fixed_hcg": rd - fixed_rd,
            "hcs_bpp": float(ref["hcs_bpp"]),
            "fixed_hcg_bpp": float(ref["hcg_bpp"]),
            "teacher_head_bpp": float(row["bpp"]),
            "hcs_psnr": float(ref["hcs_psnr"]),
            "fixed_hcg_psnr": float(ref["hcg_psnr"]),
            "teacher_head_psnr": float(row["psnr"]),
            "hcs_ms_ssim": float(ref["hcs_ms_ssim"]),
            "fixed_hcg_ms_ssim": float(ref["hcg_ms_ssim"]),
            "teacher_head_ms_ssim": float(row["ms_ssim"]),
            "fixed_hcg_strength": float(ref.get("hcg_rvq_householder_strength", float("nan"))),
            "teacher_head_strength": float(row.get("rvq_householder_strength", float("nan"))),
            "teacher_head_reliability": float(row.get("rvq_householder_reliability_multiplier", float("nan"))),
            "teacher_head_reliability_min": float(row.get("rvq_householder_reliability_multiplier_min", float("nan"))),
            "teacher_head_reliability_max": float(row.get("rvq_householder_reliability_multiplier_max", float("nan"))),
            "teacher_head_delta_rms": float(row.get("rvq_householder_delta_rms", float("nan"))),
            "teacher_head_latent_qmse": float(row.get("rvq_latent_quant_mse", float("nan"))),
            "teacher_head_s_q_mean": float(row.get("rvq_s_q_mean", float("nan"))),
            "teacher_head_dead_code_ratio": float(row.get("rvq_dead_code_ratio", float("nan"))),
            "teacher_head_perplexity": float(row.get("rvq_perplexity", float("nan"))),
            "nonfinite": int(row.get("has_nonfinite", 0)),
        }
        out.append(item)
    return out


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    hcs = [float(row["hcs_rd"]) for row in rows]
    fixed = [float(row["fixed_hcg_rd"]) for row in rows]
    teacher = [float(row["teacher_head_rd"]) for row in rows]
    d_hcs = [float(row["teacher_head_minus_hcs"]) for row in rows]
    d_fixed = [float(row["teacher_head_minus_fixed_hcg"]) for row in rows]
    return {
        "num_images": len(rows),
        "hcs_rd": mean(hcs),
        "fixed_hcg_rd": mean(fixed),
        "teacher_head_rd": mean(teacher),
        "fixed_hcg_minus_hcs": mean([float(row["fixed_hcg_minus_hcs"]) for row in rows]),
        "teacher_head_minus_hcs": mean(d_hcs),
        "teacher_head_minus_fixed_hcg": mean(d_fixed),
        "teacher_head_win_hcs_count": sum(delta < 0.0 for delta in d_hcs),
        "teacher_head_win_fixed_hcg_count": sum(delta < 0.0 for delta in d_fixed),
        "q95_teacher_head_damage_vs_hcs": q95([max(0.0, delta) for delta in d_hcs]),
        "q95_teacher_head_damage_vs_fixed_hcg": q95([max(0.0, delta) for delta in d_fixed]),
        "nonfinite_rows": sum(int(row["nonfinite"]) for row in rows),
        "fixed_hcg_strength": mean([float(row["fixed_hcg_strength"]) for row in rows]),
        "teacher_head_strength": mean([float(row["teacher_head_strength"]) for row in rows]),
        "teacher_head_reliability": mean([float(row["teacher_head_reliability"]) for row in rows]),
        "teacher_head_reliability_min": mean([float(row["teacher_head_reliability_min"]) for row in rows]),
        "teacher_head_reliability_max": mean([float(row["teacher_head_reliability_max"]) for row in rows]),
        "teacher_head_delta_rms": mean([float(row["teacher_head_delta_rms"]) for row in rows]),
        "teacher_head_latent_qmse": mean([float(row["teacher_head_latent_qmse"]) for row in rows]),
        "teacher_head_s_q_mean": mean([float(row["teacher_head_s_q_mean"]) for row in rows]),
        "teacher_head_dead_code_ratio": mean([float(row["teacher_head_dead_code_ratio"]) for row in rows]),
        "teacher_head_perplexity": mean([float(row["teacher_head_perplexity"]) for row in rows]),
    }


def by_step(rows: list[dict[str, object]], steps: list[int]) -> list[dict[str, object]]:
    return [{"step": step, **summarize([row for row in rows if int(row["step"]) == step])} for step in steps]


def quartiles(rows: list[dict[str, object]], steps: list[int]) -> list[dict[str, object]]:
    out = []
    for step in steps:
        step_rows = [row for row in rows if int(row["step"]) == step]
        ordered = sorted(step_rows, key=lambda row: float(row["hcs_rd"]))
        qsize = len(ordered) // 4
        for index in range(4):
            chunk = ordered[index * qsize : (index + 1) * qsize]
            hcs = [float(row["hcs_rd"]) for row in chunk]
            out.append(
                {
                    "step": step,
                    "quartile": f"Q{index + 1}",
                    "hcs_rd_min": min(hcs),
                    "hcs_rd_max": max(hcs),
                    **summarize(chunk),
                }
            )
    return out


def write_markdown(prefix: Path, result: dict[str, object]) -> None:
    steps = result["by_step"]
    assert isinstance(steps, list)
    lines = [
        "# E146 Low-Rate Bias010 Teacher-Head Reliability Controller",
        "",
        "This evaluates a single HCG bias010 checkpoint with only the reliability head trained on the independent E144 transfer split. It is a deployable single-checkpoint test, unlike the E144 HCS/HCG diagnostic switch.",
        "",
        "## Headline",
        "",
        "| step | HCS RD | fixed HCG-HCS | teacher-HCS | teacher-fixed | wins vs HCS | q95 damage | reliability | strength | qMSE | dead | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in steps:
        lines.append(
            "| {step} | {hcs} | {fixed_hcs} | {teacher_hcs} | {teacher_fixed} | {wins}/{n} | {q95} | {rel} | {strength} | {qmse} | {dead} | {nonfinite} |".format(
                step=row["step"],
                hcs=fmt(float(row["hcs_rd"])),
                fixed_hcs=fmt(float(row["fixed_hcg_minus_hcs"]), True),
                teacher_hcs=fmt(float(row["teacher_head_minus_hcs"]), True),
                teacher_fixed=fmt(float(row["teacher_head_minus_fixed_hcg"]), True),
                wins=row["teacher_head_win_hcs_count"],
                n=row["num_images"],
                q95=fmt(float(row["q95_teacher_head_damage_vs_hcs"])),
                rel=fmt(float(row["teacher_head_reliability"])),
                strength=fmt(float(row["teacher_head_strength"])),
                qmse=fmt(float(row["teacher_head_latent_qmse"])),
                dead=fmt(float(row["teacher_head_dead_code_ratio"])),
                nonfinite=row["nonfinite_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Promote this line only if the teacher-head checkpoint reduces the seed3456 fixed-HCG damage without erasing the useful geometry regime. If it loses to HCS or simply suppresses geometry into a worse operating point, keep E144 as controlled evidence and move to a richer reliability/fallback design.",
            "",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[250, 500])
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    args = parser.parse_args()

    refs = load_csv(E143_PAIRS)
    rows = evaluate_steps(args.steps, args.start_index, args.max_images)
    aligned = align(rows, refs)
    prefix = output_prefix(args.max_images, args.start_index)
    step_rows = by_step(aligned, args.steps)
    quartile_rows = quartiles(aligned, args.steps)
    result = {
        "experiment": "E146 low-rate bias010 teacher-head reliability controller",
        "seed": SEED,
        "run_name": RUN_NAME,
        "start_index": args.start_index,
        "max_images": args.max_images,
        "by_step": step_rows,
        "quartiles": quartile_rows,
    }
    write_csv(prefix.with_suffix(".rows.csv"), rows)
    write_csv(prefix.with_suffix(".aligned.csv"), aligned)
    write_csv(prefix.with_suffix(".by_step.csv"), step_rows)
    write_csv(prefix.with_suffix(".quartiles.csv"), quartile_rows)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(prefix, result)
    print(prefix.with_suffix(".md"))
    print(json.dumps({"by_step": step_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
