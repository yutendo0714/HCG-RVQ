#!/usr/bin/env python3
"""Directly evaluate a signaled HCS/HCG branch on the low-rate holdout protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - side-bit accounting can fall back to patch cost.
    Image = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools import analyze_e143_lowrate_bias010_holdout_selector as e143
from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e151_signaled_branch_direct_eval"
DEFAULT_FEATURE = "hcg_rvq_householder_strength"
DEFAULT_DIRECTION = "low"
DEFAULT_THRESHOLD = 0.27135278284549713


def finite_mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def q95(values: list[float]) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]


def side_bit_bpp(path: str, patch_size: int | None) -> float:
    if Image is not None:
        try:
            with Image.open(path) as image:
                width, height = image.size
            if width > 0 and height > 0:
                return 1.0 / float(width * height)
        except OSError:
            pass
    assumed = patch_size or 256
    return 1.0 / float(assumed * assumed)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose(use_value: float, direction: str, threshold: float) -> bool:
    if not math.isfinite(use_value):
        return False
    if direction == "low":
        return use_value <= threshold
    if direction == "high":
        return use_value >= threshold
    raise ValueError(f"unknown direction: {direction}")


def evaluate_runs(
    *,
    data_root: str,
    start_index: int,
    max_images: int,
    patch_size: int | None,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for run in e143.RUNS:
        rows, summary = evaluate_mode(
            mode="exact",
            config_path=str(ROOT / str(run["config"])),
            checkpoint_path=str(ROOT / str(run["checkpoint"])),
            data_root=data_root,
            device=device,
            max_images=max_images,
            start_index=start_index,
            patch_size=patch_size,
            reference={},
        )
        for row in rows:
            row.update(
                {
                    "seed": run["seed"],
                    "method": run["method"],
                    "selected_step": run["step"],
                    "config": run["config"],
                    "checkpoint": run["checkpoint"],
                }
            )
        summary.update(
            {
                "seed": run["seed"],
                "method": run["method"],
                "selected_step": run["step"],
                "config": run["config"],
                "checkpoint": run["checkpoint"],
            }
        )
        all_rows.extend(rows)
        summaries.append(summary)
    return all_rows, summaries


def align_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(row["seed"]), str(row["method"]), str(row["path"])): row for row in rows}
    pairs: list[dict[str, object]] = []
    for seed in e143.SEEDS:
        paths = sorted({str(row["path"]) for row in rows if str(row["seed"]) == seed})
        for path in paths:
            hcs = by_key[(seed, "hcs", path)]
            hcg = by_key[(seed, "hcg_bias010", path)]
            pair: dict[str, object] = {
                "seed": seed,
                "path": path,
                "index": hcs["index"],
                "hcs_rd": hcs["rd_score"],
                "hcg_rd": hcg["rd_score"],
                "hcg_minus_hcs": float(hcg["rd_score"]) - float(hcs["rd_score"]),
                "hcs_bpp": hcs["bpp"],
                "hcg_bpp": hcg["bpp"],
                "hcs_psnr": hcs["psnr"],
                "hcg_psnr": hcg["psnr"],
                "hcs_ms_ssim": hcs["ms_ssim"],
                "hcg_ms_ssim": hcg["ms_ssim"],
                "hcs_nonfinite": hcs.get("has_nonfinite", 0),
                "hcg_nonfinite": hcg.get("has_nonfinite", 0),
            }
            for key, value in hcg.items():
                if str(key).startswith("rvq_"):
                    pair[f"hcg_{key}"] = value
            pairs.append(pair)
    return pairs


def summarize_pairs(
    pairs: list[dict[str, object]],
    *,
    feature: str,
    direction: str,
    threshold: float,
    patch_size: int | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for row in pairs:
        value = float(row[feature])
        use_hcg = choose(value, direction, threshold)
        signal_bpp = side_bit_bpp(str(row["path"]), patch_size)
        branch_rd = float(row["hcg_rd"]) if use_hcg else float(row["hcs_rd"])
        branch_rd_signaled = branch_rd + signal_bpp
        delta = branch_rd - float(row["hcs_rd"])
        rows.append(
            {
                **row,
                "branch_feature": feature,
                "branch_direction": direction,
                "branch_threshold": threshold,
                "branch_use_hcg": float(use_hcg),
                "branch_signal_bpp": signal_bpp,
                "branch_rd": branch_rd,
                "branch_rd_signaled": branch_rd_signaled,
                "branch_minus_hcs": delta,
                "branch_signaled_minus_hcs": branch_rd_signaled - float(row["hcs_rd"]),
            }
        )

    hcs_rd = finite_mean([float(row["hcs_rd"]) for row in rows])
    hcg_rd = finite_mean([float(row["hcg_rd"]) for row in rows])
    branch_rd = finite_mean([float(row["branch_rd"]) for row in rows])
    branch_rd_signaled = finite_mean([float(row["branch_rd_signaled"]) for row in rows])
    deltas = [float(row["hcg_minus_hcs"]) for row in rows]
    branch_deltas = [float(row["branch_minus_hcs"]) for row in rows]
    side_bpps = [float(row["branch_signal_bpp"]) for row in rows]
    per_seed: list[dict[str, object]] = []
    for seed in e143.SEEDS:
        chunk = [row for row in rows if str(row["seed"]) == seed]
        seed_hcs = finite_mean([float(row["hcs_rd"]) for row in chunk])
        seed_hcg = finite_mean([float(row["hcg_rd"]) for row in chunk])
        seed_branch = finite_mean([float(row["branch_rd"]) for row in chunk])
        seed_signaled = finite_mean([float(row["branch_rd_signaled"]) for row in chunk])
        per_seed.append(
            {
                "seed": seed,
                "num_images": len(chunk),
                "hcs_rd": seed_hcs,
                "hcg_rd": seed_hcg,
                "hcg_minus_hcs": seed_hcg - seed_hcs,
                "branch_rd": seed_branch,
                "branch_minus_hcs": seed_branch - seed_hcs,
                "branch_signaled_rd": seed_signaled,
                "branch_signaled_minus_hcs": seed_signaled - seed_hcs,
                "selected_frac": finite_mean([float(row["branch_use_hcg"]) for row in chunk]),
                "q95_branch_damage": q95([max(0.0, float(row["branch_minus_hcs"])) for row in chunk]),
                "nonfinite_rows": sum(int(row["hcs_nonfinite"]) + int(row["hcg_nonfinite"]) for row in chunk),
            }
        )
    summary = {
        "experiment": "E151 signaled HCS/HCG branch direct eval",
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "num_rows": len(rows),
        "hcs_rd": hcs_rd,
        "hcg_rd": hcg_rd,
        "hcg_minus_hcs": hcg_rd - hcs_rd,
        "branch_rd": branch_rd,
        "branch_minus_hcs": branch_rd - hcs_rd,
        "branch_minus_hcg": branch_rd - hcg_rd,
        "branch_signaled_rd": branch_rd_signaled,
        "branch_signaled_minus_hcs": branch_rd_signaled - hcs_rd,
        "branch_signaled_minus_hcg": branch_rd_signaled - hcg_rd,
        "selected_frac": finite_mean([float(row["branch_use_hcg"]) for row in rows]),
        "hcg_win_hcs_count": sum(delta < 0.0 for delta in deltas),
        "branch_win_hcs_count": sum(delta < 0.0 for delta in branch_deltas),
        "q95_hcg_damage": q95([max(0.0, delta) for delta in deltas]),
        "q95_branch_damage": q95([max(0.0, delta) for delta in branch_deltas]),
        "mean_signal_bpp": finite_mean(side_bpps),
        "nonfinite_rows": sum(int(row["hcs_nonfinite"]) + int(row["hcg_nonfinite"]) for row in rows),
        "per_seed": per_seed,
    }
    return summary, per_seed, rows


def write_markdown(prefix: Path, summary: dict[str, object], per_seed: list[dict[str, object]]) -> None:
    def fmt(value: float, signed: bool = False) -> str:
        if not math.isfinite(value):
            return "n/a"
        return f"{value:+.6f}" if signed else f"{value:.6f}"

    lines = [
        "# E151 Signaled Branch Direct Evaluation",
        "",
        "This reruns the matched HCS and active-HCG checkpoints directly, then applies the transfer-trained branch rule.",
        "",
        "## Summary",
        "",
        f"- HCS RD: `{fmt(float(summary['hcs_rd']))}`",
        f"- fixed HCG RD: `{fmt(float(summary['hcg_rd']))}` ({fmt(float(summary['hcg_minus_hcs']), True)} vs HCS)",
        f"- branch RD: `{fmt(float(summary['branch_rd']))}` ({fmt(float(summary['branch_minus_hcs']), True)} vs HCS, {fmt(float(summary['branch_minus_hcg']), True)} vs fixed HCG)",
        f"- signaled branch RD: `{fmt(float(summary['branch_signaled_rd']))}` ({fmt(float(summary['branch_signaled_minus_hcs']), True)} vs HCS)",
        f"- selected frac: `{fmt(float(summary['selected_frac']))}`",
        f"- q95 branch damage: `{fmt(float(summary['q95_branch_damage']))}`",
        f"- mean one-bit signal bpp: `{float(summary['mean_signal_bpp']):.9f}`",
        f"- nonfinite rows: `{int(summary['nonfinite_rows'])}`",
        "",
        "## Per Seed",
        "",
        "| seed | HCS RD | HCG-HCS | branch-HCS | signaled-HCS | selected frac | q95 damage | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {hcs} | {hcg_delta} | {branch_delta} | {signaled_delta} | {selected} | {q95} | {nonfinite} |".format(
                seed=row["seed"],
                hcs=fmt(float(row["hcs_rd"])),
                hcg_delta=fmt(float(row["hcg_minus_hcs"]), True),
                branch_delta=fmt(float(row["branch_minus_hcs"]), True),
                signaled_delta=fmt(float(row["branch_signaled_minus_hcs"]), True),
                selected=fmt(float(row["selected_frac"])),
                q95=fmt(float(row["q95_branch_damage"])),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use this as the reproducible branch/fallback protocol before moving the same idea into a backbone plug-in. A full-method claim still needs a cleaner codec integration, but this is the right state-preserving control experiment.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=e143.DATA_ROOT)
    parser.add_argument("--start-index", type=int, default=e143.START_INDEX)
    parser.add_argument("--max-images", type=int, default=e143.MAX_IMAGES)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument("--direction", choices=("low", "high"), default=DEFAULT_DIRECTION)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-prefix", default=str(PREFIX))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    raw_rows, run_summaries = evaluate_runs(
        data_root=args.data_root,
        start_index=args.start_index,
        max_images=args.max_images,
        patch_size=args.patch_size,
        device=device,
    )
    pairs = align_pairs(raw_rows)
    summary, per_seed, branch_rows = summarize_pairs(
        pairs,
        feature=args.feature,
        direction=args.direction,
        threshold=args.threshold,
        patch_size=args.patch_size,
    )
    summary.update(
        {
            "data_root": args.data_root,
            "start_index": args.start_index,
            "max_images": args.max_images,
            "patch_size": args.patch_size,
            "device": str(device),
            "run_summaries": run_summaries,
        }
    )
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".raw_rows.csv"), raw_rows)
    write_csv(prefix.with_suffix(".pairs.csv"), branch_rows)
    write_csv(prefix.with_suffix(".per_seed.csv"), per_seed)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(prefix, summary, per_seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
