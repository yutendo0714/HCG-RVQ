#!/usr/bin/env python3
"""Transfer/external-split audit for the beta-commit guard paper candidate.

The checkpoint choices are fixed from the trusted holdout4096 protocol and
then applied unchanged to another image split or dataset.
"""

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
DEFAULT_DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
RUN_SUFFIX = "g64_l1_k128_lambda0035"

METHODS = [
    {
        "name": "HCS",
        "configs": {
            "1234": "configs/pilot_hcs_rvq_frozen.yaml",
            "2345": "configs/pilot_hcs_rvq_frozen_seed2345.yaml",
            "3456": "configs/pilot_hcs_rvq_frozen_seed3456.yaml",
        },
        "dirs": {
            "1234": f"experiments/pilot_hcs_rvq_frozen_{RUN_SUFFIX}",
            "2345": f"experiments/pilot_hcs_rvq_frozen_{RUN_SUFFIX}_seed2345_seed2345",
            "3456": f"experiments/pilot_hcs_rvq_frozen_{RUN_SUFFIX}_seed3456_seed3456",
        },
        "steps": {"1234": "500", "2345": "250", "3456": "250"},
    },
    {
        "name": "old gate0.25",
        "config_prefix": "configs/pilot_hcg_rvq_h_gate025_frozen_seed{seed}.yaml",
        "dir_prefix": f"experiments/pilot_hcg_rvq_h_gate025_frozen_{RUN_SUFFIX}_seed{{seed}}",
        "steps": {"1234": "250", "2345": "250", "3456": "500"},
    },
    {
        "name": "min090",
        "config_prefix": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed{seed}.yaml",
        "dir_prefix": f"experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_{RUN_SUFFIX}_seed{{seed}}",
        "steps": {"1234": "500", "2345": "250", "3456": "500"},
    },
    {
        "name": "beta005 guard",
        "config_prefix": (
            "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_"
            "local_delta_cap080_rho1_excessrisk090_betacommit005_after250_frozen_seed{seed}.yaml"
        ),
        "dir_prefix": (
            f"experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_"
            f"local_delta_cap080_rho1_excessrisk090_betacommit005_after250_frozen_{RUN_SUFFIX}_seed{{seed}}"
        ),
        "steps": {"1234": "500", "2345": "500", "3456": "500"},
    },
]

FEATURE_KEYS = [
    "rvq_s_q_mean",
    "rvq_householder_strength",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
]


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def method_config(method: dict[str, object], seed: str) -> Path:
    configs = method.get("configs")
    if isinstance(configs, dict):
        return ROOT / str(configs[seed])
    return ROOT / str(method["config_prefix"]).format(seed=seed)


def method_checkpoint(method: dict[str, object], seed: str) -> Path:
    dirs = method.get("dirs")
    if isinstance(dirs, dict):
        directory = ROOT / str(dirs[seed])
    else:
        directory = ROOT / str(method["dir_prefix"]).format(seed=seed)
    step = str(method["steps"][seed])  # type: ignore[index]
    return directory / f"checkpoint_step_{step}.pth.tar"


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "num_images": len(rows),
        "mean_rd": mean([float(row["rd_score"]) for row in rows]),
        "mean_bpp": mean([float(row["bpp"]) for row in rows]),
        "mean_psnr": mean([float(row["psnr"]) for row in rows]),
        "mean_ms_ssim": mean([float(row["ms_ssim"]) for row in rows]),
        "nonfinite_rows": sum(int(row.get("has_nonfinite", 0)) for row in rows),
    }
    for key in FEATURE_KEYS:
        vals = [float(row[key]) for row in rows if key in row and str(row[key]) != ""]
        if vals:
            out[f"mean_{key}"] = mean(vals)
    return out


def quartiles(rows: list[dict[str, object]], hcs_by_key: dict[tuple[str, str], float]) -> list[dict[str, float | int | str]]:
    keyed = []
    for row in rows:
        key = (str(row["seed"]), str(row["path"]))
        hcs_rd = hcs_by_key.get(key)
        if hcs_rd is not None:
            keyed.append((hcs_rd, row))
    keyed.sort(key=lambda item: item[0])
    if not keyed:
        return []
    out = []
    n = len(keyed)
    for qi in range(4):
        part = keyed[qi * n // 4 : (qi + 1) * n // 4]
        deltas = [float(row["rd_score"]) - hcs for hcs, row in part]
        out.append(
            {
                "quartile": f"Q{qi + 1}",
                "num_images": len(part),
                "mean_hcs_rd": mean([hcs for hcs, _ in part]),
                "mean_delta_vs_hcs": mean(deltas),
                "win_vs_hcs": mean([1.0 if delta < 0.0 else 0.0 for delta in deltas]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-index", type=int, default=8192)
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_prefix = args.output_prefix or f"beta005_transfer_openimages_start{args.start_index}_n{args.max_images}"
    data_root = str(Path(args.data_root))

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    by_method: dict[str, list[dict[str, object]]] = {}

    for method in METHODS:
        method_rows: list[dict[str, object]] = []
        for seed in ("1234", "2345", "3456"):
            config = method_config(method, seed)
            checkpoint = method_checkpoint(method, seed)
            if not config.exists():
                raise FileNotFoundError(config)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            rows, _ = evaluate_mode(
                mode="exact",
                config_path=str(config),
                checkpoint_path=str(checkpoint),
                data_root=data_root,
                device=device,
                max_images=args.max_images,
                start_index=args.start_index,
                patch_size=None,
                reference={},
            )
            for row in rows:
                row["method"] = method["name"]
                row["seed"] = seed
                row["selected_step"] = str(method["steps"][seed])  # type: ignore[index]
                row["config"] = str(config.relative_to(ROOT))
                row["checkpoint"] = str(checkpoint.relative_to(ROOT))
            method_rows.extend(rows)
            all_rows.extend(rows)
            seed_summary = summarize(rows)
            seed_summary.update({"method": method["name"], "seed": seed, "selected_step": str(method["steps"][seed])})  # type: ignore[index]
            summaries.append(seed_summary)
        by_method[str(method["name"])] = method_rows
        method_summary = summarize(method_rows)
        method_summary.update({"method": method["name"], "seed": "mean", "selected_step": "validation-fixed"})
        summaries.append(method_summary)

    hcs_rows = by_method["HCS"]
    hcs_by_key = {(str(row["seed"]), str(row["path"])): float(row["rd_score"]) for row in hcs_rows}
    hcs_mean = float([row for row in summaries if row["method"] == "HCS" and row["seed"] == "mean"][0]["mean_rd"])
    hcs_seed_means = {
        str(row["seed"]): float(row["mean_rd"])
        for row in summaries
        if row["method"] == "HCS" and row["seed"] != "mean"
    }
    method_means = {str(row["method"]): float(row["mean_rd"]) for row in summaries if row["seed"] == "mean"}
    for row in summaries:
        seed = str(row["seed"])
        reference = hcs_mean if seed == "mean" else hcs_seed_means[seed]
        row["delta_vs_hcs"] = float(row["mean_rd"]) - reference

    tail = {name: quartiles(rows, hcs_by_key) for name, rows in by_method.items() if name != "HCS"}

    out_csv = ANALYSIS / f"{output_prefix}.csv"
    out_json = ANALYSIS / f"{output_prefix}.json"
    out_md = ANALYSIS / f"{output_prefix}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({key for row in all_rows for key in row}))
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "data_root": data_root,
        "start_index": args.start_index,
        "max_images_per_seed": args.max_images,
        "device": str(device),
        "summaries": summaries,
        "method_means": method_means,
        "tail_vs_hcs": tail,
        "outputs": {
            "csv": str(out_csv.relative_to(ROOT)),
            "json": str(out_json.relative_to(ROOT)),
            "md": str(out_md.relative_to(ROOT)),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Beta-Commit Guard Transfer Split Audit",
        "",
        f"Split: `{data_root}`, start_index={args.start_index}, max_images_per_seed={args.max_images}.",
        "Checkpoint selection is fixed from the trusted holdout4096 protocol; no checkpoint is selected on this audit split.",
        "",
        "| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | strength | delta RMS | qMSE | dead code | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [item for item in summaries if item["seed"] == "mean"]:
        lines.append(
            "| {method} | {rd} | {dhcs} | {bpp} | {psnr} | {msssim} | {sq} | {strength} | {delta} | {qmse} | {dead} | {nonfinite} |".format(
                method=row["method"],
                rd=fmt(float(row["mean_rd"])),
                dhcs=fmt(float(row["delta_vs_hcs"]), signed=True),
                bpp=fmt(float(row["mean_bpp"])),
                psnr=fmt(float(row["mean_psnr"])),
                msssim=fmt(float(row["mean_ms_ssim"])),
                sq=fmt(float(row.get("mean_rvq_s_q_mean", float("nan")))),
                strength=fmt(float(row.get("mean_rvq_householder_strength", float("nan")))),
                delta=fmt(float(row.get("mean_rvq_householder_delta_rms", float("nan")))),
                qmse=fmt(float(row.get("mean_rvq_latent_quant_mse", float("nan")))),
                dead=fmt(float(row.get("mean_rvq_dead_code_ratio", float("nan")))),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )
    lines.extend(["", "## Per Seed", "", "| method | seed | step | RD | vs seed HCS | nonfinite |", "|---|---:|---:|---:|---:|---:|"])
    for row in [item for item in summaries if item["seed"] != "mean"]:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['selected_step']} | {fmt(float(row['mean_rd']))} | {fmt(float(row['delta_vs_hcs']), signed=True)} | {int(row['nonfinite_rows'])} |"
        )
    lines.extend(["", "## HCS Difficulty Quartiles", ""])
    for name, qs in tail.items():
        lines.extend([f"### {name}", "", "| quartile | images | HCS RD | method-HCS | win vs HCS |", "|---|---:|---:|---:|---:|"])
        for row in qs:
            lines.append(
                f"| {row['quartile']} | {row['num_images']} | {fmt(float(row['mean_hcs_rd']))} | {fmt(float(row['mean_delta_vs_hcs']), signed=True)} | {fmt(float(row['win_vs_hcs']))} |"
            )
        lines.append("")
    lines.extend(
        [
            "Artifacts:",
            "",
            f"- `{out_csv.relative_to(ROOT)}`",
            f"- `{out_json.relative_to(ROOT)}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_md": str(out_md), "method_means": method_means}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
