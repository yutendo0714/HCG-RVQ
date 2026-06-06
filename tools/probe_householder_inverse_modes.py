#!/usr/bin/env python3
"""Probe HCG-RVQ Householder inverse conventions on path-aligned images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_msssim, compute_psnr
from hcg_rvq.models import build_model
from hcg_rvq.quantizers import householder_transform
from hcg_rvq.utils import load_config


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def _finite_mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def load_reference(path: str | None, column: str) -> dict[str, float]:
    if path is None:
        return {}
    ref_path = Path(path)
    with ref_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return {}
    if column not in rows[0]:
        lowered = {key.lower(): key for key in rows[0]}
        column = lowered.get(column.lower(), column)
    return {row["path"]: float(row[column]) for row in rows if row.get("path") and row.get(column, "") != ""}


def patch_inverse_mode(model: torch.nn.Module, mode: str) -> None:
    if mode == "exact":
        return

    def same_partial(self, x: torch.Tensor, v: torch.Tensor, strength: torch.Tensor | None = None) -> torch.Tensor:
        return self._partial_householder_transform(x, v, strength)

    def full_householder(self, x: torch.Tensor, v: torch.Tensor, strength: torch.Tensor | None = None) -> torch.Tensor:
        return householder_transform(x, v, eps=self.eps)

    def identity(self, x: torch.Tensor, v: torch.Tensor, strength: torch.Tensor | None = None) -> torch.Tensor:
        return x

    patches = {
        "same_partial": same_partial,
        "full_householder": full_householder,
        "identity": identity,
    }
    if mode not in patches:
        raise ValueError(f"unknown inverse mode: {mode}")
    model._inverse_partial_householder_transform = types.MethodType(patches[mode], model)


def build_patched_model(config_path: str, checkpoint_path: str, device: torch.device, mode: str) -> torch.nn.Module:
    config = load_config(config_path)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    patch_inverse_mode(model, mode)
    model.eval()
    return model


def tensor_scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    return None


def evaluate_mode(
    *,
    mode: str,
    config_path: str,
    checkpoint_path: str,
    data_root: str,
    device: torch.device,
    max_images: int,
    start_index: int,
    patch_size: int | None,
    reference: dict[str, float],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    config = load_config(config_path)
    model = build_patched_model(config_path, checkpoint_path, device, mode)
    loss_cfg = config.get("loss", {})
    eval_loss_cfg = dict(loss_cfg)
    eval_loss_cfg["rho_householder_reliability_teacher"] = 0.0
    eval_loss_cfg["rho_householder_residual_selector_teacher"] = 0.0
    eval_loss_cfg["rho_householder_residual_selector_noop"] = 0.0
    for anchor_name in (
        "rho_anchor_mu",
        "rho_anchor_log_s",
        "rho_anchor_u",
        "rho_anchor_y_hat",
        "rho_anchor_selected_distortion_margin",
    ):
        eval_loss_cfg[anchor_name] = 0.0
    criterion = RateDistortionLoss(**eval_loss_cfg)
    dataset = ImageFolderDataset(
        [data_root],
        patch_size=patch_size,
        training=False,
        max_images=max_images,
        start_index=start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.get("eval", {}).get("num_workers", 2))

    rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc=mode, dynamic_ncols=True)):
            x = batch.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            mse = float(losses["mse"].cpu())
            bpp = float(losses["bpp_total"].cpu())
            rd_score = bpp + float(loss_cfg.get("lambda_rd", 0.0)) * float(loss_cfg.get("mse_scale", 255.0 * 255.0)) * mse
            image_path = str(dataset.paths[index])
            row: dict[str, float | int | str] = {
                "mode": mode,
                "index": index,
                "path": image_path,
                "rd_score": rd_score,
                "bpp": bpp,
                "bpp_y": float(losses["bpp_y"].cpu()),
                "bpp_z": float(losses["bpp_z"].cpu()),
                "mse": mse,
                "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
                "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
                "commit_loss": float(losses["commit_loss"].cpu()),
            }
            if image_path in reference:
                row["reference_rd_score"] = reference[image_path]
                row["rd_minus_reference"] = rd_score - reference[image_path]
                row["abs_rd_minus_reference"] = abs(rd_score - reference[image_path])
            for key, value in output.get("rvq_stats", {}).items():
                scalar = tensor_scalar(value)
                if scalar is not None:
                    row[f"rvq_{key}"] = scalar
            if not all(math.isfinite(float(v)) for k, v in row.items() if isinstance(v, (float, int)) and k != "index"):
                row["has_nonfinite"] = 1
            else:
                row["has_nonfinite"] = 0
            rows.append(row)

    summary: dict[str, float | int | str] = {
        "mode": mode,
        "num_images": len(rows),
        "mean_rd": _finite_mean([float(row["rd_score"]) for row in rows]),
        "mean_bpp": _finite_mean([float(row["bpp"]) for row in rows]),
        "mean_psnr": _finite_mean([float(row["psnr"]) for row in rows]),
        "mean_ms_ssim": _finite_mean([float(row["ms_ssim"]) for row in rows]),
        "nonfinite_rows": sum(int(row.get("has_nonfinite", 0)) for row in rows),
    }
    for key in ("rd_minus_reference", "abs_rd_minus_reference"):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"mean_{key}"] = _finite_mean(values)
            summary[f"max_{key}"] = max(values)
    for key in (
        "rvq_householder_delta_rms",
        "rvq_householder_delta_rms_local_mean",
        "rvq_householder_strength",
        "rvq_latent_quant_mse",
        "rvq_s_q_mean",
        "rvq_perplexity",
        "rvq_dead_code_ratio",
    ):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"mean_{key}"] = _finite_mean(values)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=["exact", "same_partial", "full_householder", "identity"])
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--reference-column", default="rd_score")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = load_reference(args.reference_csv, args.reference_column)

    all_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for mode in args.modes:
        rows, summary = evaluate_mode(
            mode=mode,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            device=device,
            max_images=args.max_images,
            start_index=args.start_index,
            patch_size=args.patch_size,
            reference=reference,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in all_rows for key in row})
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "start_index": args.start_index,
        "max_images": args.max_images,
        "patch_size": args.patch_size,
        "reference_csv": args.reference_csv,
        "reference_column": args.reference_column,
        "summaries": summaries,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Householder Inverse Mode Probe",
        "",
        f"Config: `{args.config}`",
        f"Checkpoint: `{args.checkpoint}`",
        f"Split: `{args.data_root}`, start_index={args.start_index}, max_images={args.max_images}, patch_size={args.patch_size}",
        "",
        "| mode | mean RD | mean RD-ref | max abs RD-ref | bpp | PSNR | MS-SSIM | delta RMS | strength | latent qMSE | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {mode} | {rd:.6f} | {dref} | {maxref} | {bpp:.6f} | {psnr:.6f} | {msssim:.6f} | {delta} | {strength} | {qmse} | {nonfinite} |".format(
                mode=s["mode"],
                rd=float(s["mean_rd"]),
                dref="n/a" if "mean_rd_minus_reference" not in s else f"{float(s['mean_rd_minus_reference']):+.6f}",
                maxref="n/a" if "max_abs_rd_minus_reference" not in s else f"{float(s['max_abs_rd_minus_reference']):.6f}",
                bpp=float(s["mean_bpp"]),
                psnr=float(s["mean_psnr"]),
                msssim=float(s["mean_ms_ssim"]),
                delta="n/a" if "mean_rvq_householder_delta_rms" not in s else f"{float(s['mean_rvq_householder_delta_rms']):.6f}",
                strength="n/a" if "mean_rvq_householder_strength" not in s else f"{float(s['mean_rvq_householder_strength']):.6f}",
                qmse="n/a" if "mean_rvq_latent_quant_mse" not in s else f"{float(s['mean_rvq_latent_quant_mse']):.6f}",
                nonfinite=int(s["nonfinite_rows"]),
            )
        )
    best = min(
        (s for s in summaries if "mean_abs_rd_minus_reference" in s),
        key=lambda s: float(s["mean_abs_rd_minus_reference"]),
        default=None,
    )
    lines.extend([""])
    if best is not None:
        lines.append(
            f"Best historical-match mode by mean absolute RD error: `{best['mode']}` ({float(best['mean_abs_rd_minus_reference']):.6f})."
        )
    else:
        lines.append("No reference CSV was supplied, so only within-probe mode comparisons are available.")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
