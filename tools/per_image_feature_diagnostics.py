from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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


def to_grouped(x: torch.Tensor, group_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    ng = c // group_size
    return x.view(b, ng, group_size, h, w).permute(0, 1, 3, 4, 2).contiguous()


def from_grouped(x: torch.Tensor) -> torch.Tensor:
    b, ng, h, w, g = x.shape
    return x.permute(0, 1, 4, 2, 3).contiguous().view(b, ng * g, h, w)


def per_image_index_stats(indices: list[torch.Tensor], codebook_size: int, image_hw: tuple[int, int]) -> dict[str, float]:
    if not indices:
        return {}
    h_img, w_img = image_hw
    num_pixels = h_img * w_img
    entropies = []
    perplexities = []
    dead_ratios = []
    symbols_per_stage = 0
    for idx in indices:
        counts = torch.bincount(idx.detach().cpu().reshape(-1), minlength=codebook_size).float()
        total = counts.sum().clamp_min(1.0)
        symbols_per_stage = int(total.item())
        probs = counts / total
        nz = probs[probs > 0]
        entropy = -(nz * nz.log2()).sum()
        entropies.append(float(entropy.cpu()))
        perplexities.append(float(torch.pow(torch.tensor(2.0), entropy).cpu()))
        dead_ratios.append(float((counts == 0).float().mean().cpu()))
    entropy_sum = sum(entropies)
    return {
        "index_empirical_entropy": sum(entropies) / len(entropies),
        "index_empirical_perplexity": sum(perplexities) / len(perplexities),
        "index_dead_code_ratio": sum(dead_ratios) / len(dead_ratios),
        "index_empirical_bpp": entropy_sum * symbols_per_stage / num_pixels,
        "index_fixed_bpp_from_counts": len(indices) * symbols_per_stage * math.log2(codebook_size) / num_pixels,
    }


def conditioning_values(model: torch.nn.Module, output: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    y = output["y"]
    hyper_features = output["hyper_features"]
    values.update(tensor_stats("y", y))
    values.update(tensor_stats("y_hat", output["y_hat"]))
    values.update(tensor_stats("y_error", output["y_hat"] - y))
    values.update(tensor_stats("hyper_features", hyper_features))

    if model.variant == "scalar":
        return values
    if model.variant == "global_rvq":
        if getattr(model, "use_global_norm", False):
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
    values.update(tensor_stats("mu_q", mu_q))
    values.update(tensor_stats("s_q", s_q))
    values.update(tensor_stats("y_norm", y_norm))

    if model.variant not in {"hcg_rvq_h", "hcg_rvq_h_gate", "hcg_rvq_h_no_transform"}:
        return values

    v = model.householder_head(hyper_features)
    values.update(tensor_stats("householder_v", v))
    if model.variant == "hcg_rvq_h_no_transform":
        return values

    v_g = F.normalize(to_grouped(v, model.group_size), dim=-1, eps=model.eps)
    y_norm_g = to_grouped(y_norm, model.group_size)
    strength = model._householder_gate(hyper_features, s_q=s_q) if hasattr(model, "_householder_gate") else None
    raw_gate = model._raw_householder_gate(hyper_features) if hasattr(model, "_raw_householder_gate") else None
    reliability_multiplier = (
        model._householder_gate_reliability_multiplier(hyper_features)
        if hasattr(model, "_householder_gate_reliability_multiplier")
        else None
    )
    risk_multiplier = (
        model._householder_gate_risk_multiplier(s_q)
        if hasattr(model, "_householder_gate_risk_multiplier")
        else None
    )
    if raw_gate is not None:
        values.update(tensor_stats("householder_gate_raw", raw_gate))
    if reliability_multiplier is not None:
        values.update(tensor_stats("householder_reliability_multiplier", reliability_multiplier))
    if risk_multiplier is not None:
        values.update(tensor_stats("householder_risk_multiplier", risk_multiplier))
    if strength is not None:
        values.update(tensor_stats("householder_strength", strength))
        u = from_grouped(model._partial_householder_transform(y_norm_g, v_g, strength))
    else:
        values["householder_strength_mean"] = float(getattr(model, "householder_strength", 1.0))
        u = from_grouped(model._partial_householder_transform(y_norm_g, v_g, None))
    values.update(tensor_stats("householder_delta", u - y_norm))
    values.update(tensor_stats("u", u))
    return values


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"num_images": len(rows)}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, int)) and key not in {"index"}
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))]
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-image HCG/RVQ intermediate feature diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args.config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    criterion = RateDistortionLoss(**config.get("loss", {}))

    dataset = ImageFolderDataset(
        [args.data_root],
        patch_size=args.patch_size,
        training=False,
        max_images=args.max_images,
        start_index=args.start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.get("eval", {}).get("num_workers", 2))

    rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="per-image-features", dynamic_ncols=True)):
            x = batch.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            mse = float(losses["mse"].cpu())
            bpp = float(losses["bpp_total"].cpu())
            loss_cfg = config.get("loss", {})
            row: dict[str, float | int | str] = {
                "index": index,
                "path": str(dataset.paths[index]),
                "loss": float(losses["loss"].cpu()),
                "rd_score": bpp
                + float(loss_cfg.get("lambda_rd", 0.0)) * float(loss_cfg.get("mse_scale", 255.0 * 255.0)) * mse,
                "bpp": bpp,
                "bpp_y": float(losses["bpp_y"].cpu()),
                "bpp_z": float(losses["bpp_z"].cpu()),
                "mse": mse,
                "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
                "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
                "commit_loss": float(losses["commit_loss"].cpu()),
            }
            row.update(conditioning_values(model, output))
            for key, value in output.get("rvq_stats", {}).items():
                if torch.is_tensor(value) and value.numel() == 1:
                    row[f"rvq_{key}"] = float(value.detach().cpu())
            row.update(per_image_index_stats(output.get("indices", []), model.codebook_size, x.shape[-2:]))
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
