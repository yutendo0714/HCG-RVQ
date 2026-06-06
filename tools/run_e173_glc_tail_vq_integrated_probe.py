#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, from_minus1_1_to_0_1, get_state_dict  # noqa: E402
from tools.run_e162_glc_pretrained_baseline import list_images, load_image, psnr01, write_image  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e173_glc_tail_vq_integrated_k8_kodak24")
    p.add_argument("--recon-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--kmeans-iters", type=int, default=12)
    p.add_argument("--max-train-vectors", type=int, default=30000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def mean_psnr(rows: list[dict[str, Any]], key: str) -> float:
    mses = [10 ** (-float(r[key]) / 10.0) for r in rows if math.isfinite(float(r[key]))]
    mse = float(np.mean(mses)) if mses else float("nan")
    return -10.0 * math.log10(mse) if mse > 0 else float("inf")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def sum_float(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach().float()).sum().item())


def ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return 0.0
    return float(num / den)


def encode_rvq_device(x: torch.Tensor, codebooks: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
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


def train_codebooks_for_item(
    items: list[ResidualSet],
    heldout_idx: int,
    scope: str,
    k: int,
    stages: int,
    iters: int,
    max_train_vectors: int,
    seed: int,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    train_items = [x for i, x in enumerate(items) if i != heldout_idx]
    codebooks_by_key: dict[int, list[torch.Tensor]] = {}
    for key in key_values(train_items, scope):
        train_vectors = vectors_for_key(train_items, key, scope)
        train_vectors = sample_rows(train_vectors, max_train_vectors, seed + heldout_idx + key)
        codebooks_by_key[key] = train_rvq_codebooks(
            train_vectors,
            k=k,
            stages=stages,
            iters=iters,
            seed=seed + heldout_idx * 100 + key,
            device=device,
        )
    return codebooks_by_key


def replace_group_vectors(
    y_q: torch.Tensor,
    y_res: torch.Tensor,
    bits: torch.Tensor,
    mask: torch.Tensor,
    part_idx: int,
    group_size: int,
    active_groups: set[int],
    active_parts: set[int],
    codebooks_by_key: dict[int, list[torch.Tensor]],
    scope: str,
    k: int,
    stats: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if part_idx not in active_parts:
        return y_q, bits
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
        if key not in codebooks_by_key:
            continue
        vec = y_res[0, start:end].permute(1, 2, 0)[spatial]
        scalar_vec = y_q[0, start:end].permute(1, 2, 0)[spatial]
        recon, assignments = encode_rvq_device(vec, codebooks_by_key[key])
        out_group = out[0, start:end].permute(1, 2, 0)
        out_group[spatial] = recon
        out[0, start:end] = out_group.permute(2, 0, 1)
        active_bits_group = bits[0, start:end].permute(1, 2, 0)
        active_bits_group = active_bits_group[spatial]
        active_bits_sum = sum_float(active_bits_group)
        scalar_mse = float(((scalar_vec.float() - vec.float()) ** 2).mean().item())
        rvq_mse = float(((recon.float() - vec.float()) ** 2).mean().item())
        fixed_bits = float(vec.shape[0] * len(assignments) * math.log2(k))
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
        prefix = f"part{part_idx}_group{group}"
        stats[f"{prefix}_scalar_mse"] = scalar_mse
        stats[f"{prefix}_rvq_mse"] = rvq_mse
        stats[f"{prefix}_mse_ratio"] = rvq_mse / scalar_mse if scalar_mse > 0 else float("nan")
        stats[f"{prefix}_scalar_active_bits"] = active_bits_sum
        stats[f"{prefix}_rvq_fixed_bits"] = fixed_bits
        stats[f"{prefix}_rvq_empirical_bits"] = empirical_bits
        stats[f"{prefix}_index_entropy_mean"] = entropy_sum / nstages
        stats[f"{prefix}_index_used_frac_mean"] = used_sum / nstages
        stats[f"{prefix}_index_dead_frac_mean"] = dead_sum / nstages
        stats["active_scalar_bits"] = stats.get("active_scalar_bits", 0.0) + active_bits_sum
        stats["active_rvq_fixed_bits"] = stats.get("active_rvq_fixed_bits", 0.0) + fixed_bits
        stats["active_rvq_empirical_bits"] = stats.get("active_rvq_empirical_bits", 0.0) + empirical_bits
        stats["active_scalar_mse_sum"] = stats.get("active_scalar_mse_sum", 0.0) + scalar_mse * float(vec.numel())
        stats["active_rvq_mse_sum"] = stats.get("active_rvq_mse_sum", 0.0) + rvq_mse * float(vec.numel())
        stats["active_scalar_count"] = stats.get("active_scalar_count", 0.0) + float(vec.numel())
        stats["index_entropy_sum"] = stats.get("index_entropy_sum", 0.0) + entropy_sum
        stats["index_stage_count"] = stats.get("index_stage_count", 0.0) + float(nstages)
        active_bits[0, start:end].permute(1, 2, 0)[spatial] = bits[0, start:end].permute(1, 2, 0)[spatial]
    inactive_bits = valid_bits - active_bits
    return out, inactive_bits


def tail_vq_forward_four_part_prior(
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
    active_groups = set(getattr(self, "_e173_active_groups"))
    active_parts = set(getattr(self, "_e173_active_parts"))
    group_size = int(getattr(self, "_e173_group_size"))
    codebooks_by_key = getattr(self, "_e173_codebooks_by_key")
    scope = str(getattr(self, "_e173_scope"))
    k = int(getattr(self, "_e173_k"))
    stats: dict[str, float] = {
        "active_parts": float(len(active_parts)),
        "active_groups": float(len(active_groups)),
        "group_size": float(group_size),
    }
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
        y_q, inactive_bits = replace_group_vectors(
            scalar_y_q,
            y_res,
            scalar_bits,
            mask,
            part_idx,
            group_size,
            active_groups,
            active_parts,
            codebooks_by_key,
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
    stats["hybrid_fixed_bits_y"] = stats.get("inactive_scalar_bits", 0.0) + stats.get("active_rvq_fixed_bits", 0.0)
    stats["hybrid_empirical_bits_y"] = stats.get("inactive_scalar_bits", 0.0) + stats.get("active_rvq_empirical_bits", 0.0)
    self._e173_tail_vq_stats = stats
    return y_res_all, y_q_all, y_hat, scales_hat_all


def install_tail_vq_branch(
    net: GLC_Image,
    codebooks_by_key: dict[int, list[torch.Tensor]],
    args: argparse.Namespace,
) -> None:
    net._e173_codebooks_by_key = codebooks_by_key
    net._e173_group_size = args.group_size
    net._e173_active_groups = args.active_groups
    net._e173_active_parts = args.active_parts
    net._e173_scope = args.scope
    net._e173_k = args.k
    net.forward_four_part_prior = types.MethodType(tail_vq_forward_four_part_prior, net)


@torch.inference_mode()
def run_instrumented(net: GLC_Image, x_pad: torch.Tensor, q: int) -> tuple[torch.Tensor, dict[str, float]]:
    curr_q_enc = net.q_enc[q : q + 1]
    curr_q_dec = net.q_dec[q : q + 1]
    y_ori = net.vqgan.encoder(x_pad)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
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
    bit_y = float(net.get_y_gaussian_bits(y_q, scales_hat).sum().item())
    bit_z = float(z_hat.shape[-2] * z_hat.shape[-1] * math.log2(net.codebook_size))
    stats = {
        "gaussian_bits_y": bit_y,
        "bits_z": bit_z,
        "gaussian_bits_total": bit_y + bit_z,
        "y_res_std": float(y_res.float().std(unbiased=False).item()),
        "y_q_std": float(y_q.float().std(unbiased=False).item()),
        "nonfinite_forward": int(
            (not torch.isfinite(x_hat).all().item())
            or (not torch.isfinite(y_res).all().item())
            or (not torch.isfinite(y_q).all().item())
            or (not torch.isfinite(scales_hat).all().item())
        ),
    }
    stats.update(getattr(net, "_e173_tail_vq_stats", {}))
    return x_hat, stats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    official_forward = net.forward_four_part_prior
    active_parts = set(args.active_parts)
    active_groups = set(args.active_groups)

    data_by_q: dict[int, list[ResidualSet]] = {}
    for q in args.q_indexes:
        data_by_q[q] = []
        for path in images:
            item = collect_residual_set(net, path, q, device, args.padding_size, args.group_size, active_parts, active_groups)
            data_by_q[q].append(item)
            print(f"collected q={q} {path.name} vectors={item.vectors.shape[0]}")

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        items = data_by_q[q]
        for heldout_idx, path in enumerate(images):
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            pixels = float(h * w)

            net.forward_four_part_prior = official_forward
            t0 = time.perf_counter()
            base_pad, base_stats = run_instrumented(net, x_pad, q)
            if device.type == "cuda":
                torch.cuda.synchronize()
            base_ms = (time.perf_counter() - t0) * 1000.0

            codebooks = train_codebooks_for_item(
                items,
                heldout_idx,
                args.scope,
                args.k,
                args.stages,
                args.kmeans_iters,
                args.max_train_vectors,
                args.seed,
                device,
            )
            install_tail_vq_branch(net, codebooks, args)
            t0 = time.perf_counter()
            branch_pad, branch_stats = run_instrumented(net, x_pad, q)
            if device.type == "cuda":
                torch.cuda.synchronize()
            branch_ms = (time.perf_counter() - t0) * 1000.0
            net.forward_four_part_prior = official_forward

            base = F.pad(base_pad, (-padding_l, -padding_r, -padding_t, -padding_b)).clamp(-1, 1)
            branch = F.pad(branch_pad, (-padding_l, -padding_r, -padding_t, -padding_b)).clamp(-1, 1)
            base01 = from_minus1_1_to_0_1(base).clamp(0, 1)
            branch01 = from_minus1_1_to_0_1(branch).clamp(0, 1)
            base_bpp = float(base_stats["gaussian_bits_total"]) / pixels
            hybrid_fixed_bpp = (float(branch_stats["hybrid_fixed_bits_y"]) + float(branch_stats["bits_z"])) / pixels
            hybrid_emp_bpp = (float(branch_stats["hybrid_empirical_bits_y"]) + float(branch_stats["bits_z"])) / pixels
            row: dict[str, Any] = {
                "q_index": q,
                "image": path.name,
                "height": h,
                "width": w,
                "base_bpp": base_bpp,
                "branch_gaussian_bpp": float(branch_stats["gaussian_bits_total"]) / pixels,
                "branch_hybrid_fixed_bpp": hybrid_fixed_bpp,
                "branch_hybrid_empirical_bpp": hybrid_emp_bpp,
                "fixed_bpp_delta": hybrid_fixed_bpp - base_bpp,
                "empirical_bpp_delta": hybrid_emp_bpp - base_bpp,
                "base_psnr": psnr01(base01, img01),
                "branch_psnr": psnr01(branch01, img01),
                "base_ms_ssim": float(ms_ssim(base01, img01, data_range=1.0).item()),
                "branch_ms_ssim": float(ms_ssim(branch01, img01, data_range=1.0).item()),
                "base_lpips": float(lpips_fn(base, x).mean().item()),
                "branch_lpips": float(lpips_fn(branch, x).mean().item()),
                "base_dists": float(dists_fn(base01, img01, require_grad=False).detach().mean().item()),
                "branch_dists": float(dists_fn(branch01, img01, require_grad=False).detach().mean().item()),
                "base_ms": base_ms,
                "branch_ms": branch_ms,
                "xhat_mse_vs_base": float(((branch01 - base01) ** 2).mean().item()),
                "max_abs_xhat_diff_vs_base": float((branch - base).abs().max().item()),
                "nonfinite": int(
                    base_stats["nonfinite_forward"]
                    or branch_stats["nonfinite_forward"]
                    or any(isinstance(v, float) and not math.isfinite(v) for v in branch_stats.values())
                ),
                "active_scalar_mse": float(branch_stats["active_scalar_mse"]),
                "active_rvq_mse": float(branch_stats["active_rvq_mse"]),
                "active_mse_ratio": float(branch_stats["active_mse_ratio"]),
                "index_entropy_mean": float(branch_stats["index_entropy_mean"]),
                "original_scalar_bpp_y": float(branch_stats["original_scalar_bits"]) / pixels,
                "inactive_scalar_bpp_y": float(branch_stats["inactive_scalar_bits"]) / pixels,
                "active_scalar_bpp_y": float(branch_stats["active_scalar_bits"]) / pixels,
                "active_rvq_fixed_bpp_y": float(branch_stats["active_rvq_fixed_bits"]) / pixels,
                "active_rvq_empirical_bpp_y": float(branch_stats["active_rvq_empirical_bits"]) / pixels,
                "base_y_q_std": float(base_stats["y_q_std"]),
                "branch_y_q_std": float(branch_stats["y_q_std"]),
            }
            rows.append(row)
            if args.recon_dir is not None:
                write_image(args.recon_dir / "base" / f"q{q}" / path.name, base)
                write_image(args.recon_dir / "branch" / f"q{q}" / path.name, branch)
            print(
                f"q={q} {path.name} base={base_bpp:.5f} fixed={hybrid_fixed_bpp:.5f} emp={hybrid_emp_bpp:.5f} "
                f"psnr {row['base_psnr']:.3f}->{row['branch_psnr']:.3f} "
                f"dists {row['base_dists']:.5f}->{row['branch_dists']:.5f} "
                f"active_mse_ratio={row['active_mse_ratio']:.3f} nonfinite={row['nonfinite']}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for q in args.q_indexes:
        subset = [r for r in rows if int(r["q_index"]) == q]
        if not subset:
            continue
        summary.append(
            {
                "q_index": q,
                "images": len(subset),
                "base_bpp": mean(subset, "base_bpp"),
                "branch_hybrid_fixed_bpp": mean(subset, "branch_hybrid_fixed_bpp"),
                "branch_hybrid_empirical_bpp": mean(subset, "branch_hybrid_empirical_bpp"),
                "fixed_bpp_delta": mean(subset, "fixed_bpp_delta"),
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

    payload = {
        "experiment": "E173 GLC integrated tail VQ/RVQ diagnostic branch",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "q_indexes": args.q_indexes,
        "active_parts": args.active_parts,
        "active_groups": args.active_groups,
        "group_size": args.group_size,
        "scope": args.scope,
        "k": args.k,
        "stages": args.stages,
        "kmeans_iters": args.kmeans_iters,
        "rows": len(rows),
        "summary": summary,
        "note": "Diagnostic only. Codebooks are trained leave-one-image-out on evaluation residuals and are not a paper-quality training protocol.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E173 GLC Integrated Tail VQ/RVQ Diagnostic Branch",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        f"Active parts: `{args.active_parts}`",
        f"Active groups: `{args.active_groups}`",
        f"Scope/K/stages: `{args.scope}` / `{args.k}` / `{args.stages}`",
        "",
        "Diagnostic only: codebooks are trained leave-one-image-out on evaluation residuals. The table reports hybrid bpp estimates, not a final bitstream.",
        "",
        "| q | images | base bpp | fixed bpp | emp bpp | fixed dbpp | emp dbpp | PSNR base | PSNR branch | MS-SSIM base | MS-SSIM branch | LPIPS base | LPIPS branch | DISTS base | DISTS branch | active MSE ratio | H | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['q_index']} | {s['images']} | {s['base_bpp']:.6f} | {s['branch_hybrid_fixed_bpp']:.6f} | "
            f"{s['branch_hybrid_empirical_bpp']:.6f} | {s['fixed_bpp_delta']:+.6f} | {s['empirical_bpp_delta']:+.6f} | "
            f"{s['base_psnr']:.4f} | {s['branch_psnr']:.4f} | {s['base_ms_ssim']:.5f} | {s['branch_ms_ssim']:.5f} | "
            f"{s['base_lpips']:.5f} | {s['branch_lpips']:.5f} | {s['base_dists']:.5f} | {s['branch_dists']:.5f} | "
            f"{s['active_mse_ratio']:.4f} | {s['index_entropy_mean']:.4f} | {s['nonfinite_rows']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
