#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, from_minus1_1_to_0_1, get_state_dict  # noqa: E402
from tools.run_e162_glc_pretrained_baseline import list_images, load_image, psnr01  # noqa: E402
from tools.run_e170_glc_tail_vq_rate_distortion_probe import (  # noqa: E402
    ResidualSet,
    active_key,
    collect_residual_set,
    entropy_bits,
    key_values,
    nearest_indices,
    sample_rows,
    train_rvq_codebooks,
    vectors_for_key,
)


@dataclass
class PreparedImage:
    path: Path
    img01: torch.Tensor
    x: torch.Tensor
    x_pad: torch.Tensor
    padding: tuple[int, int, int, int]
    height: int
    width: int


class TrainableRVQCodebooks(nn.Module):
    def __init__(self, initial: dict[int, list[torch.Tensor]], device: torch.device) -> None:
        super().__init__()
        self.keys = sorted(initial)
        self.stage_counts = {k: len(v) for k, v in initial.items()}
        self.params = nn.ParameterDict()
        for key, books in initial.items():
            for stage, book in enumerate(books):
                self.params[self.param_name(key, stage)] = nn.Parameter(book.float().to(device))

    @staticmethod
    def param_name(key: int, stage: int) -> str:
        prefix = f"m{abs(key)}" if key < 0 else f"p{key}"
        return f"{prefix}_s{stage}"

    def for_key(self, key: int) -> list[torch.Tensor]:
        return [self.params[self.param_name(key, stage)] for stage in range(self.stage_counts[key])]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e175_glc_decoder_aware_tail_vq_train_smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--kmeans-iters", type=int, default=8)
    p.add_argument("--max-train-vectors", type=int, default=20000)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.02)
    p.add_argument("--dists-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--limit", type=int, default=2)
    return p.parse_args()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def mean_psnr(rows: list[dict[str, Any]], key: str) -> float:
    mses = [10 ** (-float(r[key]) / 10.0) for r in rows if math.isfinite(float(r[key]))]
    mse = float(np.mean(mses)) if mses else float("nan")
    return -10.0 * math.log10(mse) if mse > 0 else float("inf")


def sum_float(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach().float()).sum().item())


def ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return 0.0
    return float(num / den)


def dists_call(dists_fn, x01: torch.Tensor, target01: torch.Tensor, require_grad: bool) -> torch.Tensor:
    try:
        return dists_fn(x01, target01, require_grad=require_grad).mean()
    except TypeError:
        return dists_fn(x01, target01).mean()


def encode_trainable_rvq(x: torch.Tensor, codebooks: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    residual = x.float()
    recon = torch.zeros_like(residual)
    assignments: list[torch.Tensor] = []
    for codebook in codebooks:
        cb = codebook.to(device=x.device, dtype=residual.dtype)
        idx = nearest_indices(residual, cb)
        q = cb[idx]
        recon = recon + q
        residual = residual - q
        assignments.append(idx.detach().cpu())
    return recon.to(dtype=x.dtype), assignments


def build_initial_codebooks(
    items: list[ResidualSet],
    scope: str,
    k: int,
    stages: int,
    iters: int,
    max_train_vectors: int,
    seed: int,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    initial: dict[int, list[torch.Tensor]] = {}
    for key in key_values(items, scope):
        train_vectors = vectors_for_key(items, key, scope)
        train_vectors = sample_rows(train_vectors, max_train_vectors, seed + key)
        initial[key] = train_rvq_codebooks(
            train_vectors,
            k=k,
            stages=stages,
            iters=iters,
            seed=seed + key * 1009,
            device=device,
        )
    return initial


def replace_group_vectors_trainable(
    y_q: torch.Tensor,
    y_res: torch.Tensor,
    bits: torch.Tensor,
    mask: torch.Tensor,
    part_idx: int,
    group_size: int,
    active_groups: set[int],
    active_parts: set[int],
    codebooks: TrainableRVQCodebooks,
    scope: str,
    k: int,
    stats: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if part_idx not in active_parts:
        return y_q, bits * mask
    out = y_q.clone()
    active_bits = torch.zeros_like(bits)
    valid_bits = bits * mask
    C = y_q.shape[1]
    for group in sorted(active_groups):
        start = group * group_size
        end = min(start + group_size, C)
        if start >= C:
            continue
        spatial = mask[0, start].bool()
        if not spatial.any():
            continue
        key = -1 if scope == "shared" else active_key(part_idx, group)
        if key not in codebooks.stage_counts:
            continue
        vec = y_res[0, start:end].permute(1, 2, 0)[spatial]
        scalar_vec = y_q[0, start:end].permute(1, 2, 0)[spatial]
        recon, assignments = encode_trainable_rvq(vec, codebooks.for_key(key))
        out_group = out[0, start:end].permute(1, 2, 0)
        out_group[spatial] = recon
        out[0, start:end] = out_group.permute(2, 0, 1)

        active_bits_group = bits[0, start:end].permute(1, 2, 0)[spatial]
        active_bits_sum = sum_float(active_bits_group)
        scalar_mse = float(((scalar_vec.detach().float() - vec.detach().float()) ** 2).mean().item())
        rvq_mse = float(((recon.detach().float() - vec.detach().float()) ** 2).mean().item())
        empirical_bits = 0.0
        entropy_sum = 0.0
        used_sum = 0.0
        dead_sum = 0.0
        for idx in assignments:
            entropy, _, used_frac, dead_frac = entropy_bits(idx, k)
            entropy_sum += entropy
            used_sum += used_frac
            dead_sum += dead_frac
            empirical_bits += float(idx.numel()) * entropy
        nstages = max(1, len(assignments))
        stats["active_scalar_bits"] = stats.get("active_scalar_bits", 0.0) + active_bits_sum
        stats["active_rvq_fixed_bits"] = stats.get("active_rvq_fixed_bits", 0.0) + float(vec.shape[0] * nstages * math.log2(k))
        stats["active_rvq_empirical_bits"] = stats.get("active_rvq_empirical_bits", 0.0) + empirical_bits
        stats["active_scalar_mse_sum"] = stats.get("active_scalar_mse_sum", 0.0) + scalar_mse * float(vec.numel())
        stats["active_rvq_mse_sum"] = stats.get("active_rvq_mse_sum", 0.0) + rvq_mse * float(vec.numel())
        stats["active_scalar_count"] = stats.get("active_scalar_count", 0.0) + float(vec.numel())
        stats["index_entropy_sum"] = stats.get("index_entropy_sum", 0.0) + entropy_sum
        stats["index_used_sum"] = stats.get("index_used_sum", 0.0) + used_sum
        stats["index_dead_sum"] = stats.get("index_dead_sum", 0.0) + dead_sum
        stats["index_stage_count"] = stats.get("index_stage_count", 0.0) + float(nstages)
        active_bits[0, start:end].permute(1, 2, 0)[spatial] = active_bits_group
    inactive_bits = valid_bits - active_bits
    return out, inactive_bits


def trainable_tail_forward_four_part_prior(
    self: GLC_Image,
    y: torch.Tensor,
    common_params: torch.Tensor,
    y_spatial_prior_adaptor_1,
    y_spatial_prior_adaptor_2,
    y_spatial_prior_adaptor_3,
    y_spatial_prior,
    y_spatial_prior_reduction=None,
    write: bool = False,
):
    del write
    q_enc, q_dec, scales, means = self.separate_prior(common_params)
    if y_spatial_prior_reduction is not None:
        common_params = y_spatial_prior_reduction(common_params)
    B, C, H, W = y.size()
    masks = self.get_mask_four_parts(B, C, H, W, y.dtype, y.device)
    y_scaled = y * q_enc
    active_groups = set(getattr(self, "_e175_active_groups"))
    active_parts = set(getattr(self, "_e175_active_parts"))
    group_size = int(getattr(self, "_e175_group_size"))
    codebooks = getattr(self, "_e175_codebooks")
    scope = str(getattr(self, "_e175_scope"))
    k = int(getattr(self, "_e175_k"))
    stats: dict[str, float] = {}
    y_res_parts = []
    y_q_parts = []
    y_hat_parts = []
    s_hat_parts = []
    y_hat_so_far = None
    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            part_scales, part_means = scales, means
        else:
            assert y_hat_so_far is not None
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                y_spatial_prior_adaptor_1
                if part_idx == 1
                else y_spatial_prior_adaptor_2
                if part_idx == 2
                else y_spatial_prior_adaptor_3
            )
            part_scales, part_means = y_spatial_prior(adaptor(params)).chunk(2, 1)
        scales_hat = part_scales * mask
        means_hat = part_means * mask
        y_res = (y_scaled - means_hat) * mask
        scalar_y_q = self.quant(y_res)
        scalar_bits = self.get_y_gaussian_bits(scalar_y_q, scales_hat) * mask
        y_q, inactive_bits = replace_group_vectors_trainable(
            scalar_y_q,
            y_res,
            scalar_bits,
            mask,
            part_idx,
            group_size,
            active_groups,
            active_parts,
            codebooks,
            scope,
            k,
            stats,
        )
        y_hat_part = y_q + means_hat
        stats["inactive_scalar_bits"] = stats.get("inactive_scalar_bits", 0.0) + sum_float(inactive_bits)
        stats["original_scalar_bits"] = stats.get("original_scalar_bits", 0.0) + sum_float(scalar_bits)
        y_res_parts.append(y_res)
        y_q_parts.append(y_q)
        y_hat_parts.append(y_hat_part)
        s_hat_parts.append(scales_hat)
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
    y_res_all = sum(y_res_parts)
    y_q_all = sum(y_q_parts)
    y_hat = sum(y_hat_parts) * q_dec
    scales_hat_all = sum(s_hat_parts)
    active_count = stats.get("active_scalar_count", 0.0)
    stats["active_scalar_mse"] = ratio(stats.get("active_scalar_mse_sum", 0.0), active_count)
    stats["active_rvq_mse"] = ratio(stats.get("active_rvq_mse_sum", 0.0), active_count)
    stats["active_mse_ratio"] = (
        stats["active_rvq_mse"] / stats["active_scalar_mse"] if stats["active_scalar_mse"] > 0 else float("nan")
    )
    stats["index_entropy_mean"] = ratio(stats.get("index_entropy_sum", 0.0), stats.get("index_stage_count", 0.0))
    stats["index_used_frac_mean"] = ratio(stats.get("index_used_sum", 0.0), stats.get("index_stage_count", 0.0))
    stats["index_dead_frac_mean"] = ratio(stats.get("index_dead_sum", 0.0), stats.get("index_stage_count", 0.0))
    stats["hybrid_fixed_bits_y"] = stats.get("inactive_scalar_bits", 0.0) + stats.get("active_rvq_fixed_bits", 0.0)
    stats["hybrid_empirical_bits_y"] = stats.get("inactive_scalar_bits", 0.0) + stats.get("active_rvq_empirical_bits", 0.0)
    self._e175_tail_vq_stats = stats
    return y_res_all, y_q_all, y_hat, scales_hat_all


def install_trainable_branch(
    net: GLC_Image,
    codebooks: TrainableRVQCodebooks,
    args: argparse.Namespace,
) -> None:
    net._e175_codebooks = codebooks
    net._e175_group_size = args.group_size
    net._e175_active_groups = args.active_groups
    net._e175_active_parts = args.active_parts
    net._e175_scope = args.scope
    net._e175_k = args.k
    net.forward_four_part_prior = types.MethodType(trainable_tail_forward_four_part_prior, net)


def run_instrumented(net: GLC_Image, x_pad: torch.Tensor, q: int) -> tuple[torch.Tensor, dict[str, float]]:
    curr_q_enc = net.q_enc[q : q + 1]
    curr_q_dec = net.q_dec[q : q + 1]
    y_ori = net.vqgan.encoder(x_pad)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    with torch.no_grad():
        index = net.z_vq.get_indices(z)
        z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)
    y_res, y_q, y_hat_prior, scales_hat = net.forward_four_part_prior(
        y,
        params,
        net.y_spatial_prior_adaptor_1,
        net.y_spatial_prior_adaptor_2,
        net.y_spatial_prior_adaptor_3,
        net.y_spatial_prior,
        y_spatial_prior_reduction=net.y_spatial_prior_reduction,
    )
    y_hat_dec = net.dec(y_hat_prior, curr_q_dec)
    x_hat = net.vqgan.generator(y_hat_dec)
    bit_y = float(net.get_y_gaussian_bits(y_q.detach(), scales_hat.detach()).sum().item())
    bit_z = float(z_hat.shape[-2] * z_hat.shape[-1] * math.log2(net.codebook_size))
    stats = {
        "gaussian_bits_y": bit_y,
        "bits_z": bit_z,
        "gaussian_bits_total": bit_y + bit_z,
        "y_res_std": float(y_res.detach().float().std(unbiased=False).item()),
        "y_q_std": float(y_q.detach().float().std(unbiased=False).item()),
        "nonfinite_forward": int(
            (not torch.isfinite(x_hat).all().item())
            or (not torch.isfinite(y_res).all().item())
            or (not torch.isfinite(y_q).all().item())
            or (not torch.isfinite(scales_hat).all().item())
        ),
    }
    stats.update(getattr(net, "_e175_tail_vq_stats", {}))
    return x_hat, stats


def prepare_images(paths: list[Path], device: torch.device, padding_size: int) -> list[PreparedImage]:
    out = []
    for path in paths:
        img01 = load_image(path, device)
        x = from_0_1_to_minus1_1(img01)
        _, _, h, w = x.shape
        padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, padding_size)
        x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
        out.append(PreparedImage(path, img01, x, x_pad, (padding_l, padding_r, padding_t, padding_b), h, w))
    return out


def crop_to_image(x_pad: torch.Tensor, item: PreparedImage) -> torch.Tensor:
    padding_l, padding_r, padding_t, padding_b = item.padding
    return F.pad(x_pad, (-padding_l, -padding_r, -padding_t, -padding_b)).clamp(-1, 1)


def evaluate_rows(
    net: GLC_Image,
    official_forward,
    codebooks_by_q: dict[int, TrainableRVQCodebooks],
    prepared: list[PreparedImage],
    q_indexes: list[int],
    args: argparse.Namespace,
    lpips_fn,
    dists_fn,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for q in q_indexes:
            for item in prepared:
                pixels = float(item.height * item.width)
                net.forward_four_part_prior = official_forward
                base_pad, base_stats = run_instrumented(net, item.x_pad, q)
                install_trainable_branch(net, codebooks_by_q[q], args)
                branch_pad, branch_stats = run_instrumented(net, item.x_pad, q)
                net.forward_four_part_prior = official_forward

                base = crop_to_image(base_pad, item)
                branch = crop_to_image(branch_pad, item)
                base01 = from_minus1_1_to_0_1(base).clamp(0, 1)
                branch01 = from_minus1_1_to_0_1(branch).clamp(0, 1)
                base_bpp = float(base_stats["gaussian_bits_total"]) / pixels
                hybrid_fixed_bpp = (float(branch_stats["hybrid_fixed_bits_y"]) + float(branch_stats["bits_z"])) / pixels
                hybrid_emp_bpp = (float(branch_stats["hybrid_empirical_bits_y"]) + float(branch_stats["bits_z"])) / pixels
                row = {
                    "label": label,
                    "q_index": q,
                    "image": item.path.name,
                    "height": item.height,
                    "width": item.width,
                    "base_bpp": base_bpp,
                    "branch_gaussian_bpp": float(branch_stats["gaussian_bits_total"]) / pixels,
                    "branch_hybrid_fixed_bpp": hybrid_fixed_bpp,
                    "branch_hybrid_empirical_bpp": hybrid_emp_bpp,
                    "empirical_bpp_delta": hybrid_emp_bpp - base_bpp,
                    "fixed_bpp_delta": hybrid_fixed_bpp - base_bpp,
                    "base_psnr": psnr01(base01, item.img01),
                    "branch_psnr": psnr01(branch01, item.img01),
                    "base_ms_ssim": float(ms_ssim(base01, item.img01, data_range=1.0).item()),
                    "branch_ms_ssim": float(ms_ssim(branch01, item.img01, data_range=1.0).item()),
                    "base_lpips": float(lpips_fn(base, item.x).mean().item()),
                    "branch_lpips": float(lpips_fn(branch, item.x).mean().item()),
                    "base_dists": float(dists_call(dists_fn, base01, item.img01, require_grad=False).detach().item()),
                    "branch_dists": float(dists_call(dists_fn, branch01, item.img01, require_grad=False).detach().item()),
                    "active_scalar_mse": float(branch_stats["active_scalar_mse"]),
                    "active_rvq_mse": float(branch_stats["active_rvq_mse"]),
                    "active_mse_ratio": float(branch_stats["active_mse_ratio"]),
                    "index_entropy_mean": float(branch_stats["index_entropy_mean"]),
                    "index_used_frac_mean": float(branch_stats["index_used_frac_mean"]),
                    "index_dead_frac_mean": float(branch_stats["index_dead_frac_mean"]),
                    "nonfinite": int(
                        base_stats["nonfinite_forward"]
                        or branch_stats["nonfinite_forward"]
                        or any(isinstance(v, float) and not math.isfinite(v) for v in branch_stats.values())
                    ),
                }
                rows.append(row)
                print(
                    f"{label} q={q} {item.path.name} psnr {row['base_psnr']:.3f}->{row['branch_psnr']:.3f} "
                    f"lpips {row['base_lpips']:.4f}->{row['branch_lpips']:.4f} "
                    f"dists {row['base_dists']:.4f}->{row['branch_dists']:.4f} "
                    f"emp_dbpp={row['empirical_bpp_delta']:+.5f} nonfinite={row['nonfinite']}"
                )
    return rows


def training_loss(
    branch: torch.Tensor,
    branch01: torch.Tensor,
    target: PreparedImage,
    lpips_fn,
    dists_fn,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(branch01, target.img01)
    lpips_val = lpips_fn(branch, target.x).mean()
    dists_val = dists_call(dists_fn, branch01, target.img01, require_grad=True)
    loss = args.mse_weight * mse + args.lpips_weight * lpips_val + args.dists_weight * dists_val
    return loss, {
        "mse": float(mse.detach().item()),
        "lpips": float(lpips_val.detach().item()),
        "dists": float(dists_val.detach().item()),
        "loss": float(loss.detach().item()),
    }


def summarize(rows: list[dict[str, Any]], q_indexes: list[int]) -> list[dict[str, Any]]:
    summary = []
    labels = sorted({str(r["label"]) for r in rows})
    for label in labels:
        for q in q_indexes:
            subset = [r for r in rows if str(r["label"]) == label and int(r["q_index"]) == q]
            if not subset:
                continue
            summary.append(
                {
                    "label": label,
                    "q_index": q,
                    "images": len(subset),
                    "base_bpp": mean(subset, "base_bpp"),
                    "branch_hybrid_empirical_bpp": mean(subset, "branch_hybrid_empirical_bpp"),
                    "empirical_bpp_delta": mean(subset, "empirical_bpp_delta"),
                    "base_psnr": mean_psnr(subset, "base_psnr"),
                    "branch_psnr": mean_psnr(subset, "branch_psnr"),
                    "base_ms_ssim": mean(subset, "base_ms_ssim"),
                    "branch_ms_ssim": mean(subset, "branch_ms_ssim"),
                    "base_lpips": mean(subset, "base_lpips"),
                    "branch_lpips": mean(subset, "branch_lpips"),
                    "base_dists": mean(subset, "base_dists"),
                    "branch_dists": mean(subset, "branch_dists"),
                    "active_mse_ratio": mean(subset, "active_mse_ratio"),
                    "index_entropy_mean": mean(subset, "index_entropy_mean"),
                    "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
                }
            )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: list[dict[str, Any]], train_trace: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "E175 GLC decoder-aware trainable active tail VQ diagnostic",
        "note": "Diagnostic upper-bound only. Codebooks are optimized on the evaluated small subset; this is not a paper-quality training split.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
        "train_trace": train_trace,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E175 GLC Decoder-Aware Tail VQ Training Diagnostic",
        "",
        "Diagnostic upper-bound only: trainable active codebooks are optimized on the evaluated small subset. Use this to decide whether matched GLC fine-tuning is worth implementing, not as a final paper row.",
        "",
        f"Input: `{args.input_dir}`",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Active parts/groups: `{args.active_parts}` / `{args.active_groups}`",
        f"Scope/K/stages: `{args.scope}` / `{args.k}` / `{args.stages}`",
        f"Steps/lr/loss weights: `{args.steps}` / `{args.lr}` / mse `{args.mse_weight}`, lpips `{args.lpips_weight}`, dists `{args.dists_weight}`",
        "",
        "| label | q | images | base bpp | emp bpp | emp dbpp | PSNR base | PSNR branch | MS base | MS branch | LPIPS base | LPIPS branch | DISTS base | DISTS branch | active MSE ratio | H | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['label']} | {s['q_index']} | {s['images']} | {s['base_bpp']:.6f} | "
            f"{s['branch_hybrid_empirical_bpp']:.6f} | {s['empirical_bpp_delta']:+.6f} | "
            f"{s['base_psnr']:.4f} | {s['branch_psnr']:.4f} | {s['base_ms_ssim']:.5f} | {s['branch_ms_ssim']:.5f} | "
            f"{s['base_lpips']:.5f} | {s['branch_lpips']:.5f} | {s['base_dists']:.5f} | {s['branch_dists']:.5f} | "
            f"{s['active_mse_ratio']:.4f} | {s['index_entropy_mean']:.4f} | {s['nonfinite_rows']} |"
        )
    if train_trace:
        lines.extend(["", "## Train Trace", "", "| step | loss | mse | lpips | dists |", "|---:|---:|---:|---:|---:|"])
        for t in train_trace:
            lines.append(f"| {t['step']} | {t['loss']:.6f} | {t['mse']:.6f} | {t['lpips']:.6f} | {t['dists']:.6f} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    dists_fn = dists.DISTS().to(device).eval()
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    # Decoder-aware optimization needs autograd through the frozen GLC decoder path.
    # Use out-of-place activations to avoid backward conflicts with chunk views.
    net = GLC_Image(inplace=False).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    active_parts = set(args.active_parts)
    active_groups = set(args.active_groups)

    prepared = prepare_images(images, device, args.padding_size)
    data_by_q: dict[int, list[ResidualSet]] = {}
    codebooks_by_q: dict[int, TrainableRVQCodebooks] = {}
    with torch.no_grad():
        for q in args.q_indexes:
            items = [
                collect_residual_set(net, item.path, q, device, args.padding_size, args.group_size, active_parts, active_groups)
                for item in prepared
            ]
            data_by_q[q] = items
            initial = build_initial_codebooks(
                items,
                args.scope,
                args.k,
                args.stages,
                args.kmeans_iters,
                args.max_train_vectors,
                args.seed + q * 10000,
                device,
            )
            codebooks_by_q[q] = TrainableRVQCodebooks(initial, device).to(device)
            print(f"initialized q={q} keys={len(initial)} vectors={sum(int(x.vectors.shape[0]) for x in items)}")

    # Residual collection runs in inference_mode and can populate cached masks
    # with inference tensors. Clear them before any autograd-enabled forward.
    net.masks = {}

    rows = evaluate_rows(net, official_forward, codebooks_by_q, prepared, args.q_indexes, args, lpips_fn, dists_fn, "init")
    params = [p for module in codebooks_by_q.values() for p in module.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    train_trace: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = None
        loss_accum = {"mse": 0.0, "lpips": 0.0, "dists": 0.0, "loss": 0.0}
        count = 0
        for q in args.q_indexes:
            install_trainable_branch(net, codebooks_by_q[q], args)
            for item in prepared:
                branch_pad, _ = run_instrumented(net, item.x_pad, q)
                branch = crop_to_image(branch_pad, item)
                branch01 = from_minus1_1_to_0_1(branch).clamp(0, 1)
                loss, parts = training_loss(branch, branch01, item, lpips_fn, dists_fn, args)
                total = loss if total is None else total + loss
                for key in loss_accum:
                    loss_accum[key] += parts[key]
                count += 1
        assert total is not None
        total = total / max(1, count)
        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        trace_row = {"step": step}
        for key, val in loss_accum.items():
            trace_row[key] = val / max(1, count)
        train_trace.append(trace_row)
        print(
            f"step={step}/{args.steps} loss={trace_row['loss']:.6f} mse={trace_row['mse']:.6f} "
            f"lpips={trace_row['lpips']:.6f} dists={trace_row['dists']:.6f}"
        )
    net.forward_four_part_prior = official_forward
    print(f"training_ms={(time.perf_counter() - t0) * 1000.0:.1f}")

    rows.extend(evaluate_rows(net, official_forward, codebooks_by_q, prepared, args.q_indexes, args, lpips_fn, dists_fn, "trained"))
    summary = summarize(rows, args.q_indexes)
    write_outputs(args, rows, summary, train_trace)


if __name__ == "__main__":
    main()
