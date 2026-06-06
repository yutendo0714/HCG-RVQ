#!/usr/bin/env python3
"""Audit whether the E146 teacher-head learned its transfer-split labels."""

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

from tools import analyze_e146_lowrate_bias010_teacher_headonly as e146
from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
LABELS = ANALYSIS / "e146_lowrate_bias010_transfer8192_reliability_teacher_labels.csv"
BASE_PREFIX = ANALYSIS / "e146_lowrate_bias010_teacher_headonly_transfer8192_fit"
DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
SEED = "3456"
RUN_NAME = e146.RUN_NAME
CONFIG = e146.CONFIG
CHECKPOINT_DIR = e146.CHECKPOINT_DIR
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


def output_prefix(max_images: int, start_index: int) -> Path:
    if max_images == DEFAULT_MAX_IMAGES and start_index == DEFAULT_START_INDEX:
        return BASE_PREFIX
    return ANALYSIS / f"e146_lowrate_bias010_teacher_headonly_transfer_fit_start{start_index}_n{max_images}"


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


def align(rows: list[dict[str, object]], labels: list[dict[str, object]]) -> list[dict[str, object]]:
    label_by_path = {
        str(row["path"]): row
        for row in labels
        if str(row.get("seed")) == SEED
    }
    out: list[dict[str, object]] = []
    missing = []
    for row in rows:
        key = str(row["path"])
        ref = label_by_path.get(key)
        if ref is None:
            missing.append(key)
            continue
        rd = float(row["rd_score"])
        hcs_rd = float(ref["hcs_rd"])
        fixed_rd = float(ref["hcg_rd"])
        target = float(ref["householder_reliability_keep"])
        rel = float(row.get("rvq_householder_reliability_multiplier", float("nan")))
        item: dict[str, object] = {
            "seed": SEED,
            "step": int(row["step"]),
            "path": row["path"],
            "teacher_keep": target,
            "teacher_weight_margin_balanced": float(ref.get("householder_reliability_weight_margin_balanced", float("nan"))),
            "hcs_rd": hcs_rd,
            "fixed_hcg_rd": fixed_rd,
            "teacher_head_rd": rd,
            "fixed_hcg_minus_hcs": float(ref["hcg_minus_hcs"]),
            "teacher_head_minus_hcs": rd - hcs_rd,
            "teacher_head_minus_fixed_hcg": rd - fixed_rd,
            "teacher_head_reliability": rel,
            "teacher_head_pred_keep05": float(rel >= 0.5) if math.isfinite(rel) else float("nan"),
            "teacher_head_pred_keep09": float(rel >= 0.9) if math.isfinite(rel) else float("nan"),
            "teacher_head_strength": float(row.get("rvq_householder_strength", float("nan"))),
            "teacher_head_delta_rms": float(row.get("rvq_householder_delta_rms", float("nan"))),
            "teacher_head_latent_qmse": float(row.get("rvq_latent_quant_mse", float("nan"))),
            "teacher_head_s_q_mean": float(row.get("rvq_s_q_mean", float("nan"))),
            "teacher_head_dead_code_ratio": float(row.get("rvq_dead_code_ratio", float("nan"))),
            "teacher_head_perplexity": float(row.get("rvq_perplexity", float("nan"))),
            "nonfinite": int(row.get("has_nonfinite", 0)),
        }
        out.append(item)
    if missing:
        raise RuntimeError(f"missing {len(missing)} evaluated paths in labels; first={missing[0]}")
    return out


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = [float(row["teacher_keep"]) for row in rows]
    rels = [float(row["teacher_head_reliability"]) for row in rows]
    d_hcs = [float(row["teacher_head_minus_hcs"]) for row in rows]
    d_fixed = [float(row["teacher_head_minus_fixed_hcg"]) for row in rows]
    keep_rows = [row for row in rows if float(row["teacher_keep"]) >= 0.5]
    suppress_rows = [row for row in rows if float(row["teacher_keep"]) < 0.5]
    return {
        "num_images": len(rows),
        "teacher_keep_fraction": mean(labels),
        "pred_keep_fraction_rel05": mean([float(row["teacher_head_pred_keep05"]) for row in rows]),
        "pred_keep_fraction_rel09": mean([float(row["teacher_head_pred_keep09"]) for row in rows]),
        "hcs_rd": mean([float(row["hcs_rd"]) for row in rows]),
        "fixed_hcg_rd": mean([float(row["fixed_hcg_rd"]) for row in rows]),
        "teacher_head_rd": mean([float(row["teacher_head_rd"]) for row in rows]),
        "fixed_hcg_minus_hcs": mean([float(row["fixed_hcg_minus_hcs"]) for row in rows]),
        "teacher_head_minus_hcs": mean(d_hcs),
        "teacher_head_minus_fixed_hcg": mean(d_fixed),
        "teacher_head_win_hcs_count": sum(delta < 0.0 for delta in d_hcs),
        "teacher_head_win_fixed_hcg_count": sum(delta < 0.0 for delta in d_fixed),
        "q95_teacher_head_damage_vs_hcs": q95([max(0.0, delta) for delta in d_hcs]),
        "nonfinite_rows": sum(int(row["nonfinite"]) for row in rows),
        "reliability_mean": mean(rels),
        "reliability_keep_label_mean": mean([float(row["teacher_head_reliability"]) for row in keep_rows]),
        "reliability_suppress_label_mean": mean([float(row["teacher_head_reliability"]) for row in suppress_rows]),
        "reliability_target_corr": pearson(labels, rels),
        "reliability_target_auc": auc(labels, rels),
        "reliability_target_bce": bce(labels, rels),
        "strength": mean([float(row["teacher_head_strength"]) for row in rows]),
        "delta_rms": mean([float(row["teacher_head_delta_rms"]) for row in rows]),
        "latent_qmse": mean([float(row["teacher_head_latent_qmse"]) for row in rows]),
        "dead_code_ratio": mean([float(row["teacher_head_dead_code_ratio"]) for row in rows]),
        "perplexity": mean([float(row["teacher_head_perplexity"]) for row in rows]),
    }


def by_step(rows: list[dict[str, object]], steps: list[int]) -> list[dict[str, object]]:
    return [{"step": step, **summarize([row for row in rows if int(row["step"]) == step])} for step in steps]


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def write_markdown(prefix: Path, result: dict[str, object]) -> None:
    rows = result["by_step"]
    assert isinstance(rows, list)
    lines = [
        "# E146 Transfer-Fit Audit",
        "",
        "This checks whether the seed3456 teacher-head controller learned the independent transfer-split labels it was trained on. A useful head should lower reliability on suppress-label images before we judge holdout transfer.",
        "",
        "| step | keep label frac | pred keep rel>=0.5 | reliability | rel keep/suppress | label corr | AUC | BCE | teacher-HCS | teacher-fixed | wins vs HCS | q95 damage | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {step} | {keep} | {pred} | {rel} | {rel_keep}/{rel_sup} | {corr} | {auc} | {bce} | {dhcs} | {dfix} | {wins}/{n} | {q95} | {nonfinite} |".format(
                step=row["step"],
                keep=fmt(float(row["teacher_keep_fraction"])),
                pred=fmt(float(row["pred_keep_fraction_rel05"])),
                rel=fmt(float(row["reliability_mean"])),
                rel_keep=fmt(float(row["reliability_keep_label_mean"])),
                rel_sup=fmt(float(row["reliability_suppress_label_mean"])),
                corr=fmt(float(row["reliability_target_corr"]), True),
                auc=fmt(float(row["reliability_target_auc"])),
                bce=fmt(float(row["reliability_target_bce"])),
                dhcs=fmt(float(row["teacher_head_minus_hcs"]), True),
                dfix=fmt(float(row["teacher_head_minus_fixed_hcg"]), True),
                wins=row["teacher_head_win_hcs_count"],
                n=row["num_images"],
                q95=fmt(float(row["q95_teacher_head_damage_vs_hcs"])),
                nonfinite=row["nonfinite_rows"],
            )
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "If reliability remains high and barely separates keep/suppress labels on the transfer split, the head-only BCE route is underpowered or poorly coupled to the deployed geometry gate. The next method should use a deterministic decoder-known gate, richer local target/weighting, or an explicit fallback/selector path rather than repeating the same head-only setting.",
        "",
    ])
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[250, 500])
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    args = parser.parse_args()

    labels = load_csv(LABELS)
    rows = evaluate_steps(args.steps, args.start_index, args.max_images)
    aligned = align(rows, labels)
    step_rows = by_step(aligned, args.steps)
    prefix = output_prefix(args.max_images, args.start_index)
    result = {
        "experiment": "E146 transfer-fit audit",
        "seed": SEED,
        "run_name": RUN_NAME,
        "start_index": args.start_index,
        "max_images": args.max_images,
        "by_step": step_rows,
    }
    write_csv(prefix.with_suffix(".rows.csv"), rows)
    write_csv(prefix.with_suffix(".aligned.csv"), aligned)
    write_csv(prefix.with_suffix(".by_step.csv"), step_rows)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(prefix, result)
    print(prefix.with_suffix(".md"))
    print(json.dumps({"by_step": step_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
