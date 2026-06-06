#!/usr/bin/env python3
"""Audit E149 low-rate bias010 strong exact-default residual selector on transfer labels."""

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

from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
LABELS = ANALYSIS / "e146_lowrate_bias010_transfer8192_reliability_teacher_labels.csv"
BASE_PREFIX = ANALYSIS / "e149_lowrate_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_fit"
DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
SEED = "3456"
RUN_NAME = "pilot_hcg_rvq_h_gate025_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_g64_l1_k128_lambda0018_seed3456"
CONFIG = ROOT / "configs/pilot_hcg_rvq_h_gate025_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_lambda0018_seed3456.yaml"
CHECKPOINT_DIR = ROOT / "experiments" / RUN_NAME
DEFAULT_START_INDEX = 8192
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


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def auc(labels: list[float], scores: list[float]) -> float:
    pairs = [(score, label) for label, score in zip(labels, scores) if math.isfinite(label) and math.isfinite(score)]
    pos = sum(1 for _, label in pairs if label >= 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ordered):
        j = idx + 1
        while j < len(ordered) and ordered[j][0] == ordered[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in ordered[idx:j] if label >= 0.5)
        idx = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def bce(labels: list[float], probs: list[float]) -> float:
    vals = []
    for label, prob in zip(labels, probs):
        if math.isfinite(label) and math.isfinite(prob):
            p = min(max(prob, 1e-6), 1.0 - 1e-6)
            vals.append(-(label * math.log(p) + (1.0 - label) * math.log(1.0 - p)))
    return mean(vals)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def output_prefix(max_images: int, start_index: int) -> Path:
    if max_images == DEFAULT_MAX_IMAGES and start_index == DEFAULT_START_INDEX:
        return BASE_PREFIX
    return ANALYSIS / f"e148_lowrate_bias010_residualselector_transfer_fit_start{start_index}_n{max_images}"


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
            row.update({"seed": SEED, "step": step, "checkpoint": str(checkpoint.relative_to(ROOT))})
        rows_all.extend(rows)
    return rows_all


def align(rows: list[dict[str, object]], labels: list[dict[str, object]]) -> list[dict[str, object]]:
    label_by_path = {str(row["path"]): row for row in labels if str(row.get("seed")) == SEED}
    out: list[dict[str, object]] = []
    missing: list[str] = []
    for row in rows:
        key = str(row["path"])
        ref = label_by_path.get(key)
        if ref is None:
            missing.append(key)
            continue
        rd = float(row["rd_score"])
        hcs_rd = float(ref["hcs_rd"])
        fixed_rd = float(ref["hcg_rd"])
        suppress = float(ref["householder_reliability_suppress"])
        prob = float(row.get("rvq_householder_residual_selector_prob", float("nan")))
        mult = float(row.get("rvq_householder_residual_selector_multiplier", float("nan")))
        out.append({
            "seed": SEED,
            "step": int(row["step"]),
            "path": row["path"],
            "teacher_suppress": suppress,
            "teacher_keep": float(ref["householder_reliability_keep"]),
            "teacher_weight_margin_balanced": float(ref.get("householder_reliability_weight_margin_balanced", float("nan"))),
            "hcs_rd": hcs_rd,
            "fixed_hcg_rd": fixed_rd,
            "selector_rd": rd,
            "fixed_hcg_minus_hcs": float(ref["hcg_minus_hcs"]),
            "selector_minus_hcs": rd - hcs_rd,
            "selector_minus_fixed_hcg": rd - fixed_rd,
            "selector_prob": prob,
            "selector_multiplier": mult,
            "selector_pred_suppress014": float(prob >= 0.014) if math.isfinite(prob) else float("nan"),
            "selector_pred_suppress050": float(prob >= 0.5) if math.isfinite(prob) else float("nan"),
            "selector_strength": float(row.get("rvq_householder_strength", float("nan"))),
            "selector_delta_rms": float(row.get("rvq_householder_delta_rms", float("nan"))),
            "selector_latent_qmse": float(row.get("rvq_latent_quant_mse", float("nan"))),
            "selector_s_q_mean": float(row.get("rvq_s_q_mean", float("nan"))),
            "selector_dead_code_ratio": float(row.get("rvq_dead_code_ratio", float("nan"))),
            "selector_perplexity": float(row.get("rvq_perplexity", float("nan"))),
            "nonfinite": int(row.get("has_nonfinite", 0)),
        })
    if missing:
        raise RuntimeError(f"missing {len(missing)} evaluated paths in labels; first={missing[0]}")
    return out


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = [float(row["teacher_suppress"]) for row in rows]
    probs = [float(row["selector_prob"]) for row in rows]
    d_hcs = [float(row["selector_minus_hcs"]) for row in rows]
    d_fixed = [float(row["selector_minus_fixed_hcg"]) for row in rows]
    suppress_rows = [row for row in rows if float(row["teacher_suppress"]) >= 0.5]
    keep_rows = [row for row in rows if float(row["teacher_suppress"]) < 0.5]
    return {
        "num_images": len(rows),
        "teacher_suppress_fraction": mean(labels),
        "pred_suppress_fraction_p014": mean([float(row["selector_pred_suppress014"]) for row in rows]),
        "pred_suppress_fraction_p050": mean([float(row["selector_pred_suppress050"]) for row in rows]),
        "hcs_rd": mean([float(row["hcs_rd"]) for row in rows]),
        "fixed_hcg_rd": mean([float(row["fixed_hcg_rd"]) for row in rows]),
        "selector_rd": mean([float(row["selector_rd"]) for row in rows]),
        "fixed_hcg_minus_hcs": mean([float(row["fixed_hcg_minus_hcs"]) for row in rows]),
        "selector_minus_hcs": mean(d_hcs),
        "selector_minus_fixed_hcg": mean(d_fixed),
        "selector_win_hcs_count": sum(delta < 0.0 for delta in d_hcs),
        "selector_win_fixed_hcg_count": sum(delta < 0.0 for delta in d_fixed),
        "q95_selector_damage_vs_hcs": q95([max(0.0, delta) for delta in d_hcs]),
        "selector_prob_mean": mean(probs),
        "selector_prob_suppress_label_mean": mean([float(row["selector_prob"]) for row in suppress_rows]),
        "selector_prob_keep_label_mean": mean([float(row["selector_prob"]) for row in keep_rows]),
        "selector_target_corr": pearson(labels, probs),
        "selector_target_auc": auc(labels, probs),
        "selector_target_bce": bce(labels, probs),
        "selector_multiplier": mean([float(row["selector_multiplier"]) for row in rows]),
        "strength": mean([float(row["selector_strength"]) for row in rows]),
        "delta_rms": mean([float(row["selector_delta_rms"]) for row in rows]),
        "latent_qmse": mean([float(row["selector_latent_qmse"]) for row in rows]),
        "dead_code_ratio": mean([float(row["selector_dead_code_ratio"]) for row in rows]),
        "perplexity": mean([float(row["selector_perplexity"]) for row in rows]),
        "nonfinite_rows": sum(int(row["nonfinite"]) for row in rows),
    }


def write_markdown(prefix: Path, by_step: list[dict[str, object]]) -> None:
    lines = [
        "# E149 Low-Rate Bias010 Strong Residual Selector Transfer-Fit Audit",
        "",
        "This tests an exact-default residual selector trained on independent transfer suppress labels. High selector probability weakens the Householder gate, so the target is the suppress label, not the keep label.",
        "",
        "| step | suppress label frac | pred suppress p>=0.014 | selector prob suppress/keep | AUC | BCE | selector-HCS | selector-fixed | wins vs HCS | q95 damage | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_step:
        lines.append(
            f"| {int(row['step'])} | {fmt(float(row['teacher_suppress_fraction']))} | "
            f"{fmt(float(row['pred_suppress_fraction_p014']))} | "
            f"{fmt(float(row['selector_prob_suppress_label_mean']))}/{fmt(float(row['selector_prob_keep_label_mean']))} | "
            f"{fmt(float(row['selector_target_auc']))} | {fmt(float(row['selector_target_bce']))} | "
            f"{fmt(float(row['selector_minus_hcs']), signed=True)} | {fmt(float(row['selector_minus_fixed_hcg']), signed=True)} | "
            f"{int(row['selector_win_hcs_count'])}/{int(row['num_images'])} | "
            f"{fmt(float(row['q95_selector_damage_vs_hcs']))} | {int(row['nonfinite_rows'])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "A useful E149 result must satisfy both sides: selector probability separates suppress vs keep labels, and RD stays close to fixed HCG while reducing HCS damage. If label separation improves but RD collapses like E147, the action path is still wrong; if RD stays stable but labels do not separate, the selector objective is too weak.",
    ])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=[250])
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    args = parser.parse_args()

    labels = load_csv(LABELS)
    rows = evaluate_steps(args.steps, args.start_index, args.max_images)
    aligned = align(rows, labels)
    by_step = [{"step": step, **summarize([row for row in aligned if int(row["step"]) == step])} for step in args.steps]
    prefix = output_prefix(args.max_images, args.start_index)
    write_csv(prefix.with_suffix(".aligned.csv"), aligned)
    write_csv(prefix.with_suffix(".by_step.csv"), by_step)
    result = {"by_step": by_step}
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(prefix, by_step)
    print(prefix.with_suffix(".md"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
