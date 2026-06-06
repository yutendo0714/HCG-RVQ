from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"{prefix}_mean": float(x.mean().cpu()),
        f"{prefix}_std": float(x.std(unbiased=False).cpu()),
        f"{prefix}_abs_mean": float(x.abs().mean().cpu()),
        f"{prefix}_rms": float(x.pow(2).mean().sqrt().cpu()),
        f"{prefix}_min": float(x.min().cpu()),
        f"{prefix}_max": float(x.max().cpu()),
    }


def update_average(acc: dict[str, list[float]], values: dict[str, float]) -> None:
    for key, value in values.items():
        if math.isfinite(value):
            acc[key].append(value)


def summarize_averages(acc: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / len(values) for key, values in sorted(acc.items()) if values}


def to_grouped(x: torch.Tensor, group_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    ng = c // group_size
    return x.view(b, ng, group_size, h, w).permute(0, 1, 3, 4, 2).contiguous()


def from_grouped(x: torch.Tensor) -> torch.Tensor:
    b, ng, h, w, g = x.shape
    return x.permute(0, 1, 4, 2, 3).contiguous().view(b, ng * g, h, w)


def index_usage_summary(
    index_counts: list[torch.Tensor],
    num_pixels: int,
    codebook_size: int,
) -> dict[str, float]:
    if not index_counts:
        return {}
    stage_entropies = []
    stage_perplexities = []
    stage_dead = []
    total_symbols = 0
    for counts in index_counts:
        counts = counts.float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        nz = probs[probs > 0]
        entropy = -(nz * nz.log2()).sum()
        stage_entropies.append(float(entropy.cpu()))
        stage_perplexities.append(float(torch.pow(torch.tensor(2.0), entropy).cpu()))
        stage_dead.append(float((counts == 0).float().mean().cpu()))
        total_symbols += int(total.item())

    empirical_bits = sum(stage_entropies) * (total_symbols / len(index_counts))
    fixed_bits = len(index_counts) * (total_symbols / len(index_counts)) * math.log2(codebook_size)
    return {
        "index_empirical_entropy": sum(stage_entropies) / len(stage_entropies),
        "index_empirical_perplexity": sum(stage_perplexities) / len(stage_perplexities),
        "index_dead_code_ratio": sum(stage_dead) / len(stage_dead),
        "index_empirical_bpp": empirical_bits / num_pixels,
        "index_fixed_bpp_from_counts": fixed_bits / num_pixels,
    }


def conditioning_stats(model, output: dict[str, object]) -> dict[str, float]:
    if model.variant == "scalar":
        hyper_features = output["hyper_features"]
        gaussian_params = model.scalar_head(hyper_features)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        values = {}
        values.update(tensor_stats("scalar_mean_hat", means_hat))
        values.update(tensor_stats("scalar_scale_hat", scales_hat))
        return values

    y = output["y"]
    hyper_features = output["hyper_features"]
    if model.variant == "global_rvq":
        values = {}
        if model.use_global_norm:
            s_q = (F.softplus(model.global_log_s) + model.eps).clamp(model.scale_min, model.scale_max)
            u = (y - model.global_mu) / s_q
            values.update(tensor_stats("global_mu", model.global_mu))
            values.update(tensor_stats("global_s_q", s_q))
            values.update(tensor_stats("u", u))
        return values

    mu_q = model.mu_head(hyper_features)
    log_s_q = model.log_s_head(hyper_features)
    s_q = (F.softplus(log_s_q) + model.eps).clamp(model.scale_min, model.scale_max)
    y_norm = (y - mu_q) / s_q
    values = {}
    values.update(tensor_stats("mu_q", mu_q))
    values.update(tensor_stats("s_q", s_q))
    values.update(tensor_stats("y_norm", y_norm))

    if model.variant in {"hcg_rvq_h", "hcg_rvq_h_gate", "hcg_rvq_h_no_transform"}:
        v = model.householder_head(hyper_features)
        values.update(tensor_stats("householder_v", v))
        if model.variant in {"hcg_rvq_h", "hcg_rvq_h_gate"}:
            v_g = F.normalize(to_grouped(v, model.group_size), dim=-1, eps=model.eps)
            y_norm_g = to_grouped(y_norm, model.group_size)
            if hasattr(model, "_partial_householder_transform"):
                strength_g = model._householder_gate(hyper_features, s_q=s_q) if hasattr(model, "_householder_gate") else None
                u = from_grouped(model._partial_householder_transform(y_norm_g, v_g, strength_g))
                if strength_g is not None:
                    values.update(tensor_stats("householder_strength", strength_g))
                    values["householder_strength"] = float(strength_g.mean().cpu())
                    if hasattr(model, "_raw_householder_gate"):
                        raw_gate = model._raw_householder_gate(hyper_features)
                        if raw_gate is not None:
                            values.update(tensor_stats("householder_gate_raw", raw_gate))
                    if hasattr(model, "_householder_gate_risk_multiplier"):
                        risk_multiplier = model._householder_gate_risk_multiplier(s_q)
                        if risk_multiplier is not None:
                            values.update(tensor_stats("householder_risk_multiplier", risk_multiplier))
                else:
                    values["householder_strength"] = float(getattr(model, "householder_strength", 1.0))
            else:
                u = from_grouped(householder_transform(y_norm_g, v_g, eps=model.eps))
                values["householder_strength"] = float(getattr(model, "householder_strength", 1.0))
            values.update(tensor_stats("householder_delta", u - y_norm))
            values.update(tensor_stats("u", u))
    return values


def analyze_checkpoint(
    config: dict,
    checkpoint_path: Path,
    root: str,
    device: torch.device,
    max_images: int | None,
    start_index: int = 0,
    patch_size: int | None = None,
) -> dict[str, float | str]:
    data_cfg = config.get("eval", {})
    dataset = ImageFolderDataset(
        [root],
        patch_size=patch_size,
        training=False,
        max_images=max_images,
        start_index=start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_cfg.get("num_workers", 2))

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_info = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    eval_loss_cfg = dict(config.get("loss", {}))
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

    averages: dict[str, list[float]] = defaultdict(list)
    index_counts = [
        torch.zeros(model.codebook_size, dtype=torch.long)
        for _ in range(model.num_stages if model.variant != "scalar" else 0)
    ]
    num_pixels_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=checkpoint_path.name, dynamic_ncols=True):
            x = batch.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)

            losses = criterion(output, x)
            num_pixels_total += x.shape[0] * x.shape[-2] * x.shape[-1]
            update_average(
                averages,
                {
                    "loss": float(losses["loss"].cpu()),
                    "bpp": float(losses["bpp_total"].cpu()),
                    "bpp_y": float(losses["bpp_y"].cpu()),
                    "bpp_z": float(losses["bpp_z"].cpu()),
                    "mse": float(losses["mse"].cpu()),
                    "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
                    "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
                    "commit_loss": float(losses["commit_loss"].cpu()),
                },
            )

            update_average(averages, tensor_stats("y", output["y"]))
            update_average(averages, tensor_stats("y_hat", output["y_hat"]))
            update_average(averages, tensor_stats("y_error", output["y_hat"] - output["y"]))
            update_average(averages, tensor_stats("hyper_features", output["hyper_features"]))
            update_average(averages, conditioning_stats(model, output))

            rvq_stats = output.get("rvq_stats", {})
            for key, value in rvq_stats.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    update_average(averages, {f"rvq_{key}": float(value.detach().cpu())})

            for stage, idx in enumerate(output.get("indices", [])):
                index_counts[stage] += torch.bincount(
                    idx.detach().cpu().reshape(-1),
                    minlength=model.codebook_size,
                )

    summary: dict[str, float | str] = {
        "checkpoint": str(checkpoint_path),
        "variant": model.variant,
        "group_size": model.group_size,
        "num_stages": model.num_stages,
        "codebook_size": model.codebook_size,
        "num_images": len(dataset),
        "missing_keys": ";".join(load_info.missing_keys),
        "unexpected_keys": ";".join(load_info.unexpected_keys),
    }
    summary.update(summarize_averages(averages))
    summary.update(index_usage_summary(index_counts, num_pixels_total, model.codebook_size))
    if "bpp" in summary and "mse" in summary:
        loss_cfg = config.get("loss", {})
        summary["rd_score"] = float(summary["bpp"]) + float(loss_cfg.get("lambda_rd", 0.0)) * float(
            loss_cfg.get("mse_scale", 255.0 * 255.0)
        ) * float(summary["mse"])
    return summary


def write_csv(summary: dict[str, float | str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze intermediate feature and index distributions for one checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config.get("eval", {})
    root = args.data_root or data_cfg.get("root", "/dpl/kodak")
    max_images = args.max_images if args.max_images is not None else data_cfg.get("max_images")
    start_index = args.start_index if args.start_index is not None else data_cfg.get("start_index", 0)
    patch_size = args.patch_size if args.patch_size is not None else data_cfg.get("patch_size")
    summary = analyze_checkpoint(
        config,
        Path(args.checkpoint),
        root,
        device,
        max_images,
        start_index=start_index,
        patch_size=patch_size,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        write_csv(summary, Path(args.output_csv))

    for key in ["variant", "bpp", "psnr", "ms_ssim", "rd_score", "y_std", "y_error_rms", "index_empirical_bpp"]:
        if key in summary:
            print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
