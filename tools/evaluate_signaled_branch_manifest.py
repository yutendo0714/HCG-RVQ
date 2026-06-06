#!/usr/bin/env python3
"""Evaluate a manifest-defined state-preserving HCS/HCG branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcg_rvq.utils import load_config
from tools.probe_householder_inverse_modes import evaluate_mode


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


def choose(value: float, direction: str, threshold: float) -> bool:
    if not math.isfinite(value):
        return False
    if direction == "low":
        return value <= threshold
    if direction == "high":
        return value >= threshold
    raise ValueError(f"unknown branch direction: {direction}")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def evaluate_runs(manifest: dict, device: torch.device) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data_cfg = manifest.get("data", {})
    data_root = str(data_cfg["root"])
    max_images = int(data_cfg.get("max_images", 4096))
    start_index = int(data_cfg.get("start_index", 0))
    patch_size = data_cfg.get("patch_size")
    rows_all: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for run in manifest["runs"]:
        config_path = resolve(run["config"])
        checkpoint_path = resolve(run["checkpoint"])
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        rows, summary = evaluate_mode(
            mode="exact",
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
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
                    "seed": str(run["seed"]),
                    "method": str(run["method"]),
                    "selected_step": int(run.get("step", -1)),
                    "config": str(run["config"]),
                    "checkpoint": str(run["checkpoint"]),
                }
            )
        summary.update(
            {
                "seed": str(run["seed"]),
                "method": str(run["method"]),
                "selected_step": int(run.get("step", -1)),
                "config": str(run["config"]),
                "checkpoint": str(run["checkpoint"]),
            }
        )
        rows_all.extend(rows)
        summaries.append(summary)
    return rows_all, summaries


def align_rows(
    raw_rows: list[dict[str, object]],
    *,
    base_method: str,
    candidate_method: str,
) -> list[dict[str, object]]:
    by_key = {
        (str(row["seed"]), str(row["method"]), str(row["path"])): row
        for row in raw_rows
    }
    pairs: list[dict[str, object]] = []
    seeds = sorted({str(row["seed"]) for row in raw_rows})
    for seed in seeds:
        paths = sorted({str(row["path"]) for row in raw_rows if str(row["seed"]) == seed})
        for path in paths:
            base = by_key[(seed, base_method, path)]
            candidate = by_key[(seed, candidate_method, path)]
            row: dict[str, object] = {
                "seed": seed,
                "path": path,
                "index": base["index"],
                "base_method": base_method,
                "candidate_method": candidate_method,
                "base_rd": base["rd_score"],
                "candidate_rd": candidate["rd_score"],
                "candidate_minus_base": float(candidate["rd_score"]) - float(base["rd_score"]),
                "base_bpp": base["bpp"],
                "candidate_bpp": candidate["bpp"],
                "base_psnr": base["psnr"],
                "candidate_psnr": candidate["psnr"],
                "base_ms_ssim": base["ms_ssim"],
                "candidate_ms_ssim": candidate["ms_ssim"],
                "base_nonfinite": base.get("has_nonfinite", 0),
                "candidate_nonfinite": candidate.get("has_nonfinite", 0),
            }
            for method, source in ((base_method, base), (candidate_method, candidate)):
                for key, value in source.items():
                    if str(key).startswith("rvq_"):
                        row[f"{method}_{key}"] = value
            pairs.append(row)
    return pairs


def summarize_branch(
    pairs: list[dict[str, object]],
    manifest: dict,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    data_cfg = manifest.get("data", {})
    branch_cfg = manifest["branch"]
    feature = str(branch_cfg["feature"])
    direction = str(branch_cfg.get("direction", "low"))
    threshold = float(branch_cfg["threshold"])
    include_signal = bool(branch_cfg.get("include_one_bit_signal", True))
    patch_size = data_cfg.get("patch_size")

    rows: list[dict[str, object]] = []
    for row in pairs:
        value = float(row[feature])
        use_candidate = choose(value, direction, threshold)
        signal_bpp = side_bit_bpp(str(row["path"]), patch_size) if include_signal else 0.0
        branch_rd = float(row["candidate_rd"]) if use_candidate else float(row["base_rd"])
        branch_signaled_rd = branch_rd + signal_bpp
        rows.append(
            {
                **row,
                "branch_feature": feature,
                "branch_direction": direction,
                "branch_threshold": threshold,
                "branch_use_candidate": float(use_candidate),
                "branch_signal_bpp": signal_bpp,
                "branch_rd": branch_rd,
                "branch_signaled_rd": branch_signaled_rd,
                "branch_minus_base": branch_rd - float(row["base_rd"]),
                "branch_signaled_minus_base": branch_signaled_rd - float(row["base_rd"]),
            }
        )

    base_rd = finite_mean([float(row["base_rd"]) for row in rows])
    candidate_rd = finite_mean([float(row["candidate_rd"]) for row in rows])
    branch_rd = finite_mean([float(row["branch_rd"]) for row in rows])
    branch_signaled_rd = finite_mean([float(row["branch_signaled_rd"]) for row in rows])
    candidate_deltas = [float(row["candidate_minus_base"]) for row in rows]
    branch_deltas = [float(row["branch_minus_base"]) for row in rows]
    per_seed: list[dict[str, object]] = []
    for seed in sorted({str(row["seed"]) for row in rows}):
        chunk = [row for row in rows if str(row["seed"]) == seed]
        seed_base = finite_mean([float(row["base_rd"]) for row in chunk])
        seed_candidate = finite_mean([float(row["candidate_rd"]) for row in chunk])
        seed_branch = finite_mean([float(row["branch_rd"]) for row in chunk])
        seed_signaled = finite_mean([float(row["branch_signaled_rd"]) for row in chunk])
        per_seed.append(
            {
                "seed": seed,
                "num_images": len(chunk),
                "base_rd": seed_base,
                "candidate_rd": seed_candidate,
                "candidate_minus_base": seed_candidate - seed_base,
                "branch_rd": seed_branch,
                "branch_minus_base": seed_branch - seed_base,
                "branch_signaled_rd": seed_signaled,
                "branch_signaled_minus_base": seed_signaled - seed_base,
                "selected_frac": finite_mean([float(row["branch_use_candidate"]) for row in chunk]),
                "q95_branch_damage": q95([max(0.0, float(row["branch_minus_base"])) for row in chunk]),
                "nonfinite_rows": sum(
                    int(row["base_nonfinite"]) + int(row["candidate_nonfinite"]) for row in chunk
                ),
            }
        )

    summary = {
        "experiment": str(manifest.get("name", "signaled_branch_manifest")),
        "description": str(manifest.get("description", "")),
        "base_method": str(branch_cfg["base_method"]),
        "candidate_method": str(branch_cfg["candidate_method"]),
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "num_rows": len(rows),
        "base_rd": base_rd,
        "candidate_rd": candidate_rd,
        "candidate_minus_base": candidate_rd - base_rd,
        "branch_rd": branch_rd,
        "branch_minus_base": branch_rd - base_rd,
        "branch_minus_candidate": branch_rd - candidate_rd,
        "branch_signaled_rd": branch_signaled_rd,
        "branch_signaled_minus_base": branch_signaled_rd - base_rd,
        "branch_signaled_minus_candidate": branch_signaled_rd - candidate_rd,
        "selected_frac": finite_mean([float(row["branch_use_candidate"]) for row in rows]),
        "candidate_win_base_count": sum(delta < 0.0 for delta in candidate_deltas),
        "branch_win_base_count": sum(delta < 0.0 for delta in branch_deltas),
        "q95_candidate_damage": q95([max(0.0, delta) for delta in candidate_deltas]),
        "q95_branch_damage": q95([max(0.0, delta) for delta in branch_deltas]),
        "mean_signal_bpp": finite_mean([float(row["branch_signal_bpp"]) for row in rows]),
        "nonfinite_rows": sum(int(row["base_nonfinite"]) + int(row["candidate_nonfinite"]) for row in rows),
        "per_seed": per_seed,
    }
    return summary, per_seed, rows


def write_markdown(prefix: Path, summary: dict[str, object], per_seed: list[dict[str, object]]) -> None:
    def fmt(value: float, signed: bool = False) -> str:
        if not math.isfinite(value):
            return "n/a"
        return f"{value:+.6f}" if signed else f"{value:.6f}"

    lines = [
        "# Signaled Branch Manifest Evaluation",
        "",
        str(summary.get("description", "")).strip(),
        "",
        "## Summary",
        "",
        f"- base `{summary['base_method']}` RD: `{fmt(float(summary['base_rd']))}`",
        f"- candidate `{summary['candidate_method']}` RD: `{fmt(float(summary['candidate_rd']))}` ({fmt(float(summary['candidate_minus_base']), True)} vs base)",
        f"- branch RD: `{fmt(float(summary['branch_rd']))}` ({fmt(float(summary['branch_minus_base']), True)} vs base, {fmt(float(summary['branch_minus_candidate']), True)} vs candidate)",
        f"- signaled branch RD: `{fmt(float(summary['branch_signaled_rd']))}` ({fmt(float(summary['branch_signaled_minus_base']), True)} vs base)",
        f"- rule: `{summary['feature']} {summary['direction']} {float(summary['threshold']):.9f}`",
        f"- selected frac: `{fmt(float(summary['selected_frac']))}`",
        f"- q95 branch damage: `{fmt(float(summary['q95_branch_damage']))}`",
        f"- mean one-bit signal bpp: `{float(summary['mean_signal_bpp']):.9f}`",
        f"- nonfinite rows: `{int(summary['nonfinite_rows'])}`",
        "",
        "## Per Seed",
        "",
        "| seed | base RD | candidate-base | branch-base | signaled-base | selected frac | q95 damage | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {base} | {candidate_delta} | {branch_delta} | {signaled_delta} | {selected} | {q95} | {nonfinite} |".format(
                seed=row["seed"],
                base=fmt(float(row["base_rd"])),
                candidate_delta=fmt(float(row["candidate_minus_base"]), True),
                branch_delta=fmt(float(row["branch_minus_base"]), True),
                signaled_delta=fmt(float(row["branch_signaled_minus_base"]), True),
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
            "This manifest turns the E151 branch into a reusable state-preserving protocol. It is still a branch over matched states, but it is now portable to new splits and to future SOTA/backbone adapter probes.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    manifest = load_config(args.manifest)
    manifest = dict(manifest)
    manifest["data"] = dict(manifest.get("data", {}))
    if args.data_root is not None:
        manifest["data"]["root"] = args.data_root
    if args.start_index is not None:
        manifest["data"]["start_index"] = args.start_index
    if args.max_images is not None:
        manifest["data"]["max_images"] = args.max_images
    if args.patch_size is not None:
        manifest["data"]["patch_size"] = args.patch_size
    output_prefix = Path(
        args.output_prefix
        or ROOT / "experiments" / "analysis" / str(manifest.get("name", "signaled_branch_manifest"))
    )
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    raw_rows, run_summaries = evaluate_runs(manifest, device)
    branch_cfg = manifest["branch"]
    pairs = align_rows(
        raw_rows,
        base_method=str(branch_cfg["base_method"]),
        candidate_method=str(branch_cfg["candidate_method"]),
    )
    summary, per_seed, branch_rows = summarize_branch(pairs, manifest)
    summary.update(
        {
            "manifest": str(args.manifest),
            "device": str(device),
            "data": manifest.get("data", {}),
            "run_summaries": run_summaries,
        }
    )

    write_csv(output_prefix.with_suffix(".raw_rows.csv"), raw_rows)
    write_csv(output_prefix.with_suffix(".pairs.csv"), branch_rows)
    write_csv(output_prefix.with_suffix(".per_seed.csv"), per_seed)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_prefix, summary, per_seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
