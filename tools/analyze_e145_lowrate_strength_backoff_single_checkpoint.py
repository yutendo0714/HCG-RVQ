#!/usr/bin/env python3
"""Evaluate E144 strength-threshold evidence as a single-checkpoint HCG gate."""


from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcg_rvq.utils import load_config
from tools import analyze_e143_lowrate_bias010_holdout_selector as e143
from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e145_lowrate_strength_backoff_single_checkpoint"
CONFIG_DIR = ANALYSIS / "e145_strength_backoff_configs"
E144_JSON = ANALYSIS / "e144_lowrate_bias010_transfer_to_holdout_controller.json"
E143_PAIRS = ANALYSIS / "e143_lowrate_bias010_holdout4096_selector.pairs.csv"
DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
DEFAULT_START_INDEX = 4096
DEFAULT_MAX_IMAGES = 4096

RUNS = [run for run in e143.RUNS if run["method"] == "hcg_bias010"]


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
    with path.open(newline="") as handle:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
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


def load_threshold() -> float:
    payload = json.loads(E144_JSON.read_text(encoding="utf-8"))
    return float(payload["preset_householder_strength_low"]["threshold"])


def make_config(run: dict[str, object], min_value: float, threshold: float, sharpness: float) -> Path:
    config = load_config(ROOT / str(run["config"]))
    qcfg = dict(config.get("quantizer", {}))
    qcfg.update(
        {
            "householder_gate_strength_backoff_enabled": True,
            "householder_gate_strength_backoff_threshold": float(threshold),
            "householder_gate_strength_backoff_min": float(min_value),
            "householder_gate_strength_backoff_sharpness": float(sharpness),
            "householder_gate_strength_backoff_detach": True,
            "householder_gate_strength_backoff_use_image_mean": True,
        }
    )
    config["quantizer"] = qcfg
    base_name = str(config.get("run_name", "hcg"))
    config["run_name"] = f"{base_name}_strengthbackoff_min{min_value:.2f}".replace(".", "p")
    seed = str(run["seed"])
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"seed{seed}_min{min_value:.2f}_t{threshold:.9f}.yaml".replace(".", "p")
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out


def evaluate_variant(min_value: float, threshold: float, sharpness: float, start_index: int, max_images: int) -> list[dict[str, object]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows_all: list[dict[str, object]] = []
    for run in RUNS:
        config_path = make_config(run, min_value, threshold, sharpness)
        rows, _summary = evaluate_mode(
            mode="exact",
            config_path=str(config_path),
            checkpoint_path=str(ROOT / str(run["checkpoint"])),
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
                    "variant": f"strength_backoff_min{min_value:.2f}",
                    "threshold": threshold,
                    "backoff_min": min_value,
                    "backoff_sharpness": sharpness,
                    "seed": run["seed"],
                    "checkpoint": run["checkpoint"],
                    "config": str(config_path.relative_to(ROOT)),
                }
            )
        rows_all.extend(rows)
    return rows_all


def align_with_references(rows: list[dict[str, object]], reference_pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(row["seed"]), str(row["path"])): row for row in reference_pairs}
    out: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["seed"]), str(row["path"]))
        ref = by_key[key]
        backoff_rd = float(row["rd_score"])
        hcs_rd = float(ref["hcs_rd"])
        fixed_hcg_rd = float(ref["hcg_rd"])
        item: dict[str, object] = {
            "variant": row["variant"],
            "seed": row["seed"],
            "path": row["path"],
            "hcs_rd": hcs_rd,
            "fixed_hcg_rd": fixed_hcg_rd,
            "backoff_rd": backoff_rd,
            "fixed_hcg_minus_hcs": float(ref["hcg_minus_hcs"]),
            "backoff_minus_hcs": backoff_rd - hcs_rd,
            "backoff_minus_fixed_hcg": backoff_rd - fixed_hcg_rd,
            "hcs_bpp": float(ref["hcs_bpp"]),
            "fixed_hcg_bpp": float(ref["hcg_bpp"]),
            "backoff_bpp": float(row["bpp"]),
            "hcs_psnr": float(ref["hcs_psnr"]),
            "fixed_hcg_psnr": float(ref["hcg_psnr"]),
            "backoff_psnr": float(row["psnr"]),
            "hcs_ms_ssim": float(ref["hcs_ms_ssim"]),
            "fixed_hcg_ms_ssim": float(ref["hcg_ms_ssim"]),
            "backoff_ms_ssim": float(row["ms_ssim"]),
            "fixed_hcg_strength": float(ref.get("hcg_rvq_householder_strength", float("nan"))),
            "backoff_strength": float(row.get("rvq_householder_strength", float("nan"))),
            "backoff_strength_backoff_multiplier": float(
                row.get("rvq_householder_strength_backoff_multiplier", float("nan"))
            ),
            "backoff_delta_rms": float(row.get("rvq_householder_delta_rms", float("nan"))),
            "backoff_local_delta_mean": float(row.get("rvq_householder_delta_rms_local_mean", float("nan"))),
            "backoff_latent_qmse": float(row.get("rvq_latent_quant_mse", float("nan"))),
            "backoff_s_q_mean": float(row.get("rvq_s_q_mean", float("nan"))),
            "backoff_dead_code_ratio": float(row.get("rvq_dead_code_ratio", float("nan"))),
            "backoff_perplexity": float(row.get("rvq_perplexity", float("nan"))),
            "nonfinite": int(row.get("has_nonfinite", 0)),
        }
        out.append(item)
    return out


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    hcs = [float(row["hcs_rd"]) for row in rows]
    fixed = [float(row["fixed_hcg_rd"]) for row in rows]
    backoff = [float(row["backoff_rd"]) for row in rows]
    d_hcs = [float(row["backoff_minus_hcs"]) for row in rows]
    d_fixed = [float(row["backoff_minus_fixed_hcg"]) for row in rows]
    return {
        "num_images": len(rows),
        "hcs_rd": mean(hcs),
        "fixed_hcg_rd": mean(fixed),
        "backoff_rd": mean(backoff),
        "fixed_hcg_minus_hcs": mean([float(row["fixed_hcg_minus_hcs"]) for row in rows]),
        "backoff_minus_hcs": mean(d_hcs),
        "backoff_minus_fixed_hcg": mean(d_fixed),
        "backoff_win_hcs_count": sum(delta < 0.0 for delta in d_hcs),
        "backoff_win_fixed_hcg_count": sum(delta < 0.0 for delta in d_fixed),
        "q95_backoff_damage_vs_hcs": q95([max(0.0, delta) for delta in d_hcs]),
        "q95_backoff_damage_vs_fixed_hcg": q95([max(0.0, delta) for delta in d_fixed]),
        "nonfinite_rows": sum(int(row["nonfinite"]) for row in rows),
        "backoff_strength": mean([float(row["backoff_strength"]) for row in rows]),
        "fixed_hcg_strength": mean([float(row["fixed_hcg_strength"]) for row in rows]),
        "backoff_strength_backoff_multiplier": mean(
            [float(row["backoff_strength_backoff_multiplier"]) for row in rows]
        ),
        "backoff_delta_rms": mean([float(row["backoff_delta_rms"]) for row in rows]),
        "backoff_latent_qmse": mean([float(row["backoff_latent_qmse"]) for row in rows]),
        "backoff_s_q_mean": mean([float(row["backoff_s_q_mean"]) for row in rows]),
        "backoff_dead_code_ratio": mean([float(row["backoff_dead_code_ratio"]) for row in rows]),
        "backoff_perplexity": mean([float(row["backoff_perplexity"]) for row in rows]),
    }


def per_seed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for seed in e143.SEEDS:
        chunk = [row for row in rows if str(row["seed"]) == seed]
        out.append({"seed": seed, **summarize(chunk)})
    return out


def quartiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["hcs_rd"]))
    qsize = len(ordered) // 4
    out = []
    for index in range(4):
        chunk = ordered[index * qsize : (index + 1) * qsize]
        hcs = [float(row["hcs_rd"]) for row in chunk]
        summary = summarize(chunk)
        out.append(
            {
                "quartile": f"Q{index + 1}",
                "hcs_rd_min": min(hcs),
                "hcs_rd_max": max(hcs),
                **summary,
            }
        )
    return out


def write_markdown(result: dict[str, object]) -> None:
    summary = result["summary"]
    lines = [
        "# E145 Low-Rate Strength Backoff Single Checkpoint",
        "",
        "This evaluates the E144 transfer-trained householder-strength threshold as a decoder-reproducible gate inside the same HCG bias010 checkpoint. It is not an HCS/HCG oracle switch.",
        "",
        "## Headline",
        "",
        "- threshold: `{:.9f}`".format(float(result["threshold"])),
        "- backoff min: `{:.3f}`".format(float(result["backoff_min"])),
        "- HCS RD: `{}`".format(fmt(float(summary["hcs_rd"]))),
        "- fixed HCG RD: `{}` (`{}` vs HCS)".format(
            fmt(float(summary["fixed_hcg_rd"])),
            fmt(float(summary["fixed_hcg_minus_hcs"]), True),
        ),
        "- strength-backoff HCG RD: `{}` (`{}` vs HCS, `{}` vs fixed HCG)".format(
            fmt(float(summary["backoff_rd"])),
            fmt(float(summary["backoff_minus_hcs"]), True),
            fmt(float(summary["backoff_minus_fixed_hcg"]), True),
        ),
        "- wins vs HCS: `{}/{}`".format(summary["backoff_win_hcs_count"], summary["num_images"]),
        "- wins vs fixed HCG: `{}/{}`".format(
            summary["backoff_win_fixed_hcg_count"], summary["num_images"]
        ),
        "- q95 damage vs HCS: `{}`".format(fmt(float(summary["q95_backoff_damage_vs_hcs"]))),
        "- nonfinite rows: `{}`".format(summary["nonfinite_rows"]),
        "",
        "## Per Seed",
        "",
        "| seed | HCS RD | fixed HCG-HCS | backoff-HCS | backoff-fixed | strength | mult | qMSE | dead | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            "| {seed} | {hcs} | {fixed} | {backoff} | {bfixed} | {strength} | {mult} | {qmse} | {dead} | {nf} |".format(
                seed=row["seed"],
                hcs=fmt(float(row["hcs_rd"])),
                fixed=fmt(float(row["fixed_hcg_minus_hcs"]), True),
                backoff=fmt(float(row["backoff_minus_hcs"]), True),
                bfixed=fmt(float(row["backoff_minus_fixed_hcg"]), True),
                strength=fmt(float(row["backoff_strength"])),
                mult=fmt(float(row["backoff_strength_backoff_multiplier"])),
                qmse=fmt(float(row["backoff_latent_qmse"])),
                dead=fmt(float(row["backoff_dead_code_ratio"])),
                nf=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## HCS Difficulty Quartiles",
            "",
            "| quartile | HCS range | fixed HCG-HCS | backoff-HCS | backoff-fixed | wins vs HCS | q95 damage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["quartiles"]:
        lines.append(
            "| {q} | {lo}-{hi} | {fixed} | {backoff} | {bfixed} | {wins}/{n} | {q95v} |".format(
                q=row["quartile"],
                lo=fmt(float(row["hcs_rd_min"])),
                hi=fmt(float(row["hcs_rd_max"])),
                fixed=fmt(float(row["fixed_hcg_minus_hcs"]), True),
                backoff=fmt(float(row["backoff_minus_hcs"]), True),
                bfixed=fmt(float(row["backoff_minus_fixed_hcg"]), True),
                wins=row["backoff_win_hcs_count"],
                n=row["num_images"],
                q95v=fmt(float(row["q95_backoff_damage_vs_hcs"])),
            )
        )
    lines.extend(["", "## Decision", "", str(result["decision"])])
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--backoff-min", type=float, default=0.0)
    parser.add_argument("--sharpness", type=float, default=80.0)
    args = parser.parse_args()

    threshold = load_threshold()
    reference_pairs = load_csv(E143_PAIRS)
    rows = evaluate_variant(args.backoff_min, threshold, args.sharpness, args.start_index, args.max_images)
    aligned = align_with_references(rows, reference_pairs)
    summary = summarize(aligned)
    result = {
        "experiment": "E145 low-rate strength backoff single checkpoint",
        "threshold": threshold,
        "backoff_min": args.backoff_min,
        "sharpness": args.sharpness,
        "start_index": args.start_index,
        "max_images": args.max_images,
        "summary": summary,
        "per_seed": per_seed(aligned),
        "quartiles": quartiles(aligned),
        "decision": (
            "Promote this only if it improves or nearly preserves fixed HCG while reducing seed3456/tail damage. "
            "If it merely explains the E144 oracle switch but weakens fixed HCG, keep it as controlled evidence and move to a learned reliability head or smoother training-time calibration."
        ),
    }
    write_csv(PREFIX.with_suffix(".rows.csv"), rows)
    write_csv(PREFIX.with_suffix(".aligned.csv"), aligned)
    write_csv(PREFIX.with_suffix(".per_seed.csv"), result["per_seed"])
    write_csv(PREFIX.with_suffix(".quartiles.csv"), result["quartiles"])
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result)
    print(PREFIX.with_suffix(".md"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
