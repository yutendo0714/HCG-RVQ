#!/usr/bin/env python3
"""Per-image audit for E126 staged HCS-to-gated-HCG adapter checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_psnr
from tools.run_e125_mbt2018_hcg_adapter_trainability_pilot import (
    FrozenMbt2018HCG,
    crop_to_hw,
    make_model,
    nonfinite_output,
    pad_to_multiple,
)

DEFAULT_OUT_PREFIX = ROOT / "experiments" / "analysis" / "e127_staged_geometry_per_image_audit"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    checkpoint: str
    variant: str
    gate_enabled: bool = False
    gate_max: float = 0.45
    gate_init: float = 0.25


CASES = [
    CaseSpec(
        name="hcs_warmup_step30",
        checkpoint="experiments/e125_mbt2018_hcg_adapter_trainability_pilot_hcs_warmup/checkpoint_step_30.pth.tar",
        variant="hcs_rvq",
    ),
    CaseSpec(
        name="staged_gate001_step30",
        checkpoint="experiments/e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias001_gate001/checkpoint_step_30.pth.tar",
        variant="hcg_rvq_h",
        gate_enabled=True,
        gate_max=0.1,
        gate_init=0.01,
    ),
    CaseSpec(
        name="staged_gate0005_step30",
        checkpoint="experiments/e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias0005_gate0005/checkpoint_step_30.pth.tar",
        variant="hcg_rvq_h",
        gate_enabled=True,
        gate_max=0.05,
        gate_init=0.005,
    ),
]


def scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def make_args(spec: CaseSpec, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        variant=spec.variant,
        group_size=args.group_size,
        num_stages=args.num_stages,
        codebook_size=args.codebook_size,
        householder_bias_init_scale=0.0,
        householder_gate_enabled=spec.gate_enabled,
        householder_gate_max=spec.gate_max,
        householder_gate_init=spec.gate_init,
    )


def load_model(spec: CaseSpec, args: argparse.Namespace, device: torch.device) -> FrozenMbt2018HCG:
    torch.manual_seed(args.model_seed)
    model = make_model(make_args(spec, args), device)
    checkpoint = torch.load(ROOT / spec.checkpoint, map_location=device)
    load_result = model.adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"{spec.name} checkpoint mismatch: {load_result}")
    model.eval()
    return model


def evaluate_case(
    spec: CaseSpec,
    model: FrozenMbt2018HCG,
    loader: DataLoader,
    criterion: RateDistortionLoss,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            path = batch["path"][0]
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            row: dict[str, object] = {
                "case": spec.name,
                "image_index": idx,
                "path": path,
                "rd_score": float((losses["bpp_total"] + criterion.lambda_rd * criterion.mse_scale * losses["mse"]).detach().cpu()),
                "loss": float(losses["loss"].detach().cpu()),
                "bpp": float(losses["bpp_total"].detach().cpu()),
                "bpp_y": float(losses["bpp_y"].detach().cpu()),
                "bpp_z": float(losses["bpp_z"].detach().cpu()),
                "mse": float(losses["mse"].detach().cpu()),
                "psnr": float(compute_psnr(x, output["x_hat"]).detach().cpu()),
                "nonfinite": nonfinite_output(output, losses),
            }
            rvq_stats = output["rvq_stats"]
            for key in (
                "latent_quant_mse",
                "dead_code_ratio",
                "perplexity",
                "stage_entropy",
                "s_q_mean",
                "s_q_std",
                "mu_q_abs_mean",
                "householder_delta_rms",
                "householder_v_abs_mean",
            ):
                value = scalar(rvq_stats.get(key))
                if value is not None:
                    row[key] = value
            rows.append(row)
    return rows


def add_baseline_deltas(rows: list[dict[str, object]], baseline_name: str) -> None:
    baseline = {int(row["image_index"]): row for row in rows if row["case"] == baseline_name}
    keys = ["rd_score", "latent_quant_mse", "dead_code_ratio", "perplexity", "stage_entropy", "s_q_mean", "s_q_std"]
    for row in rows:
        base = baseline[int(row["image_index"])]
        for key in keys:
            if key in row and key in base:
                row[f"delta_{key}"] = float(row[key]) - float(base[key])


def aggregate(rows: list[dict[str, object]], baseline_name: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cases = sorted({str(row["case"]) for row in rows})
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        deltas = [float(row.get("delta_rd_score", 0.0)) for row in case_rows]
        dead_deltas = [float(row.get("delta_dead_code_ratio", 0.0)) for row in case_rows]
        out.append(
            {
                "case": case,
                "num_images": len(case_rows),
                "mean_rd": finite_mean([float(row["rd_score"]) for row in case_rows]),
                "mean_delta_rd": finite_mean(deltas),
                "win_rate_rd": sum(1 for v in deltas if v < 0.0) / len(deltas),
                "q95_damage_rd": percentile([max(0.0, v) for v in deltas], 0.95),
                "max_damage_rd": max([max(0.0, v) for v in deltas]) if deltas else float("nan"),
                "mean_delta_qmse": finite_mean([float(row.get("delta_latent_quant_mse", 0.0)) for row in case_rows]),
                "mean_delta_dead": finite_mean(dead_deltas),
                "mean_delta_perplexity": finite_mean([float(row.get("delta_perplexity", 0.0)) for row in case_rows]),
                "mean_householder_delta_rms": finite_mean([float(row.get("householder_delta_rms", 0.0)) for row in case_rows]),
                "nonfinite_sum": sum(float(row.get("nonfinite", 0.0)) for row in case_rows),
                "is_baseline": case == baseline_name,
            }
        )
    return out


def selector_rows(rows: list[dict[str, object]], baseline_name: str) -> list[dict[str, object]]:
    by_image: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_image.setdefault(int(row["image_index"]), []).append(row)
    out: list[dict[str, object]] = []
    thresholds = [0.0, 0.025, 0.05, 0.075, 0.10]
    baseline_total = 0.0
    for image_rows in by_image.values():
        baseline_total += float(next(row["rd_score"] for row in image_rows if row["case"] == baseline_name))
    baseline_mean = baseline_total / len(by_image)

    policies: dict[str, list[dict[str, object]]] = {"oracle_min_rd": []}
    for threshold in thresholds:
        policies[f"rd_win_dead_delta_le_{threshold:.3f}"] = []

    for image_index, image_rows in by_image.items():
        baseline = next(row for row in image_rows if row["case"] == baseline_name)
        candidates = [row for row in image_rows if row["case"] != baseline_name]
        best = min(image_rows, key=lambda row: float(row["rd_score"]))
        policies["oracle_min_rd"].append(best)
        for threshold in thresholds:
            valid = [
                row
                for row in candidates
                if float(row.get("delta_rd_score", 0.0)) < 0.0
                and float(row.get("delta_dead_code_ratio", 0.0)) <= threshold
            ]
            policies[f"rd_win_dead_delta_le_{threshold:.3f}"].append(
                min(valid, key=lambda row: float(row["rd_score"])) if valid else baseline
            )

    for policy, selected in policies.items():
        mean_rd = finite_mean([float(row["rd_score"]) for row in selected])
        out.append(
            {
                "policy": policy,
                "num_images": len(selected),
                "mean_rd": mean_rd,
                "delta_vs_baseline": mean_rd - baseline_mean,
                "hcg_selected": sum(1 for row in selected if row["case"] != baseline_name),
                "mean_delta_dead": finite_mean([float(row.get("delta_dead_code_ratio", 0.0)) for row in selected]),
                "mean_delta_qmse": finite_mean([float(row.get("delta_latent_quant_mse", 0.0)) for row in selected]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default="/dpl/kodak")
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda-rd", type=float, default=0.0035)
    parser.add_argument("--beta-commit", type=float, default=0.05)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--out-prefix", default=str(DEFAULT_OUT_PREFIX))
    args = parser.parse_args()
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = ROOT / out_prefix

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    dataset = ImageFolderDataset([args.eval_root], training=False, max_images=args.eval_images, return_path=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    criterion = RateDistortionLoss(lambda_rd=args.lambda_rd, beta_commit=args.beta_commit)

    per_image: list[dict[str, object]] = []
    for spec in CASES:
        model = load_model(spec, args, device)
        per_image.extend(evaluate_case(spec, model, loader, criterion, device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_name = CASES[0].name
    add_baseline_deltas(per_image, baseline_name)
    summary = aggregate(per_image, baseline_name)
    selectors = selector_rows(per_image, baseline_name)
    result = {
        "experiment": "E127 staged geometry per-image audit",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "baseline": baseline_name,
        "model_seed": args.model_seed,
        "cases": [spec.__dict__ for spec in CASES],
        "summary": summary,
        "selectors": selectors,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(out_prefix.with_name(out_prefix.name + "_per_image.csv"), per_image)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary)
    write_csv(out_prefix.with_name(out_prefix.name + "_selectors.csv"), selectors)

    lines = [
        "# E127 Staged Geometry Per-Image Audit",
        "",
        f"- Device: `{result['device']}`, CUDA_VISIBLE_DEVICES=`{result['cuda_visible_devices']}`",
        f"- Baseline: `{baseline_name}`",
        f"- Images: `{args.eval_images}`",
        "",
        "## Summary",
        "",
        "| case | mean RD | delta RD | win rate | q95 damage | delta qMSE | delta dead | delta perplexity | H-delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {case} | {mean_rd} | {mean_delta_rd} | {win_rate_rd} | {q95_damage_rd} | {mean_delta_qmse} | {mean_delta_dead} | {mean_delta_perplexity} | {mean_householder_delta_rms} |".format(
                case=row["case"],
                mean_rd=fmt(row["mean_rd"]),
                mean_delta_rd=fmt(row["mean_delta_rd"]),
                win_rate_rd=fmt(row["win_rate_rd"]),
                q95_damage_rd=fmt(row["q95_damage_rd"]),
                mean_delta_qmse=fmt(row["mean_delta_qmse"]),
                mean_delta_dead=fmt(row["mean_delta_dead"]),
                mean_delta_perplexity=fmt(row["mean_delta_perplexity"]),
                mean_householder_delta_rms=fmt(row["mean_householder_delta_rms"]),
            )
        )
    lines.extend(["", "## Selector Headroom", "", "| policy | mean RD | delta vs baseline | HCG selected | delta dead | delta qMSE |", "|---|---:|---:|---:|---:|---:|"])
    for row in selectors:
        lines.append(
            "| {policy} | {mean_rd} | {delta_vs_baseline} | {hcg_selected} | {mean_delta_dead} | {mean_delta_qmse} |".format(
                policy=row["policy"],
                mean_rd=fmt(row["mean_rd"]),
                delta_vs_baseline=fmt(row["delta_vs_baseline"]),
                hcg_selected=row["hcg_selected"],
                mean_delta_dead=fmt(row["mean_delta_dead"]),
                mean_delta_qmse=fmt(row["mean_delta_qmse"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit compares staged geometry checkpoints against the HCS warmup checkpoint on the same images. It is meant to diagnose checkpoint behavior, intermediate features, and codebook usage, not to make a SOTA quality claim.",
            "",
            "## Artifacts",
            "",
            f"- `{out_prefix.with_suffix('.json')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_per_image.csv')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_summary.csv')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_selectors.csv')}`",
        ]
    )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
