#!/usr/bin/env python3
"""Symbol round-trip audit for HCG-RVQ branches inside the GLC image codec.

This script is intentionally stricter than the fast export scripts:

* Base GLC is decoded from transmitted z indices plus quantized y symbols.
* The HCG branch is decoded from transmitted z indices, inactive scalar y
  symbols, and active RVQ indices.
* Hard-gate rows pay for exactly one selected stream plus an image-level mode
  signal.
* Soft-gate rows pay for both base and branch streams plus the quantized gate
  signal, because both images are needed to reproduce a soft blend.

The goal is not to implement a production arithmetic coder.  It verifies that
the claimed decoder output is reproducible from the symbols being counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import get_state_dict, init_func  # noqa: E402
from hcg_rvq.reliability_index_controller import (  # noqa: E402
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    mix_with_fallback,
)
from tools.eval_glc_qaware_branch_checkpoint import codebooks_from_state_dict  # noqa: E402
from tools.run_e170_glc_tail_vq_rate_distortion_probe import (  # noqa: E402
    active_key,
    entropy_bits,
    nearest_indices,
)
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    TrainableRVQCodebooks,
    crop_to_image,
    install_trainable_branch,
    run_instrumented,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    list_images,
    prepare_images,
)
from tools.run_e263_glc_fallback_gate_codec_loop_pilot import (  # noqa: E402
    FEATURES,
    branch_feature_dict,
    feature_tensor,
    image_signal_bpp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch-checkpoint", type=Path, required=True)
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=4)
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--active-threshold", type=float, default=0.05)
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--controller-hidden", type=int, default=None)
    p.add_argument("--quantize-soft-gate-bits", type=int, default=8)
    p.add_argument("--selection-signal-bits", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=1e-5)
    return p.parse_args()


def sum_float(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach().float()).sum().item())


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def max_or_nan(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(max(vals)) if vals else float("nan")


def encode_trainable_rvq_with_indices(
    x: torch.Tensor,
    codebooks: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    residual = x.float()
    recon = torch.zeros_like(residual)
    indices: list[torch.Tensor] = []
    for codebook in codebooks:
        cb = codebook.to(device=x.device, dtype=residual.dtype)
        idx = nearest_indices(residual, cb)
        q = cb[idx]
        recon = recon + q
        residual = residual - q
        indices.append(idx.detach().cpu())
    return recon.to(dtype=x.dtype), indices


def z_to_params(net: GLC_Image, z_index: torch.Tensor, z_get_shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    idx = z_index.to(device)
    z_hat = net.z_vq.get_quan_feat(idx, z_get_shape)
    params = net.hyper_dec(z_hat)
    return net.y_prior_fusion(params)


def encoder_common(net: GLC_Image, x_pad: torch.Tensor, q: int) -> dict[str, Any]:
    curr_q_enc = net.q_enc[q : q + 1]
    y_ori = net.vqgan.encoder(x_pad)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    z_index = net.z_vq.get_indices(z)
    z_get_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
    z_hat = net.z_vq.get_quan_feat(z_index, z_get_shape)
    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)
    bits_z = float(z_hat.shape[-2] * z_hat.shape[-1] * math.log2(net.codebook_size))
    return {
        "y": y,
        "z_index": z_index.detach().cpu(),
        "z_get_shape": z_get_shape,
        "params": params,
        "bits_z": bits_z,
    }


def prior_step(
    net: GLC_Image,
    part_idx: int,
    mask: torch.Tensor,
    common_params: torch.Tensor,
    base_scales: torch.Tensor,
    base_means: torch.Tensor,
    y_hat_so_far: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if part_idx == 0:
        return base_scales, base_means
    assert y_hat_so_far is not None
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    adaptor = (
        net.y_spatial_prior_adaptor_1
        if part_idx == 1
        else net.y_spatial_prior_adaptor_2
        if part_idx == 2
        else net.y_spatial_prior_adaptor_3
    )
    del mask
    return net.y_spatial_prior(adaptor(params)).chunk(2, 1)


def decode_from_yq_parts(
    net: GLC_Image,
    q: int,
    symbol: dict[str, Any],
    y_q_parts: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    params = z_to_params(net, symbol["z_index"], tuple(symbol["z_get_shape"]), device)
    _, q_dec, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    first = y_q_parts[0].to(device)
    b, c, h, w = first.shape
    masks = net.get_mask_four_parts(b, c, h, w, first.dtype, device)
    y_hat_parts: list[torch.Tensor] = []
    y_hat_so_far = None
    for part_idx, mask in enumerate(masks):
        part_scales, part_means = prior_step(net, part_idx, mask, common_params, scales, means, y_hat_so_far)
        del part_scales
        means_hat = part_means * mask
        y_q_part = y_q_parts[part_idx].to(device)
        y_hat_part = y_q_part + means_hat
        y_hat_parts.append(y_hat_part)
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
    y_hat = sum(y_hat_parts) * q_dec
    y_hat_dec = net.dec(y_hat, net.q_dec[q : q + 1])
    return net.vqgan.generator(y_hat_dec)


def encode_base_symbols(net: GLC_Image, x_pad: torch.Tensor, q: int) -> tuple[torch.Tensor, dict[str, Any], dict[str, float]]:
    common = encoder_common(net, x_pad, q)
    y = common["y"]
    params = common["params"]
    q_enc, q_dec, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    b, c, h, w = y.shape
    masks = net.get_mask_four_parts(b, c, h, w, y.dtype, y.device)
    y_scaled = y * q_enc
    y_q_parts: list[torch.Tensor] = []
    y_hat_parts: list[torch.Tensor] = []
    scales_hat_parts: list[torch.Tensor] = []
    y_hat_so_far = None
    bits_y = 0.0
    for part_idx, mask in enumerate(masks):
        part_scales, part_means = prior_step(net, part_idx, mask, common_params, scales, means, y_hat_so_far)
        scales_hat = part_scales * mask
        means_hat = part_means * mask
        y_res = (y_scaled - means_hat) * mask
        y_q = net.quant(y_res)
        y_hat_part = y_q + means_hat
        bits_y += sum_float(net.get_y_gaussian_bits(y_q, scales_hat) * mask)
        y_q_parts.append(y_q.detach().cpu())
        y_hat_parts.append(y_hat_part)
        scales_hat_parts.append(scales_hat)
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
    y_hat = sum(y_hat_parts) * q_dec
    x_hat = net.vqgan.generator(net.dec(y_hat, net.q_dec[q : q + 1]))
    symbol = {
        "z_index": common["z_index"],
        "z_get_shape": common["z_get_shape"],
        "y_q_parts": y_q_parts,
    }
    stats = {
        "gaussian_bits_y": bits_y,
        "bits_z": float(common["bits_z"]),
        "gaussian_bits_total": bits_y + float(common["bits_z"]),
        "z_bpp": 0.0,
    }
    del scales_hat_parts
    return x_hat, symbol, stats


def reconstruct_active_group(
    codebooks: TrainableRVQCodebooks,
    key: int,
    indices: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    recon = None
    books = codebooks.for_key(key)
    for stage, idx_cpu in enumerate(indices):
        cb = books[stage].to(device)
        idx = idx_cpu.to(device).long()
        q = cb[idx]
        recon = q if recon is None else recon + q
    assert recon is not None
    return recon


def encode_branch_symbols(
    net: GLC_Image,
    x_pad: torch.Tensor,
    q: int,
    codebooks: TrainableRVQCodebooks,
    branch_args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float]]:
    common = encoder_common(net, x_pad, q)
    y = common["y"]
    params = common["params"]
    q_enc, q_dec, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    b, c, h, w = y.shape
    masks = net.get_mask_four_parts(b, c, h, w, y.dtype, y.device)
    y_scaled = y * q_enc
    active_groups = set(int(v) for v in branch_args.active_groups)
    active_parts = set(int(v) for v in branch_args.active_parts)
    group_size = int(branch_args.group_size)
    scope = str(branch_args.scope)
    k = int(branch_args.k)

    inactive_y_q_parts: list[torch.Tensor] = []
    y_hat_parts: list[torch.Tensor] = []
    active_entries: list[dict[str, Any]] = []
    y_hat_so_far = None
    stats: dict[str, float] = {
        "inactive_scalar_bits": 0.0,
        "active_scalar_bits": 0.0,
        "active_rvq_empirical_bits": 0.0,
        "active_rvq_fixed_bits": 0.0,
        "original_scalar_bits": 0.0,
        "index_entropy_sum": 0.0,
        "index_used_sum": 0.0,
        "index_dead_sum": 0.0,
        "index_stage_count": 0.0,
    }
    for part_idx, mask in enumerate(masks):
        part_scales, part_means = prior_step(net, part_idx, mask, common_params, scales, means, y_hat_so_far)
        scales_hat = part_scales * mask
        means_hat = part_means * mask
        y_res = (y_scaled - means_hat) * mask
        scalar_y_q = net.quant(y_res)
        scalar_bits = net.get_y_gaussian_bits(scalar_y_q, scales_hat) * mask
        y_q_part = scalar_y_q.clone()
        inactive_y_q = scalar_y_q.clone()
        active_bits = torch.zeros_like(scalar_bits)

        if part_idx in active_parts:
            for group in sorted(active_groups):
                start = group * group_size
                end = min(start + group_size, c)
                if start >= c:
                    continue
                key = -1 if scope == "shared" else active_key(part_idx, group)
                if key not in codebooks.stage_counts:
                    continue
                spatial = mask[0, start].bool()
                if not spatial.any():
                    continue
                vec = y_res[0, start:end].permute(1, 2, 0)[spatial]
                recon, indices = encode_trainable_rvq_with_indices(vec, codebooks.for_key(key))
                yq_group = y_q_part[0, start:end].permute(1, 2, 0)
                yq_group[spatial] = recon
                y_q_part[0, start:end] = yq_group.permute(2, 0, 1)
                inactive_group = inactive_y_q[0, start:end].permute(1, 2, 0)
                inactive_group[spatial] = 0.0
                inactive_y_q[0, start:end] = inactive_group.permute(2, 0, 1)

                active_bits_group = scalar_bits[0, start:end].permute(1, 2, 0)[spatial]
                active_bits[0, start:end].permute(1, 2, 0)[spatial] = active_bits_group
                empirical_bits = 0.0
                entropy_sum = 0.0
                used_sum = 0.0
                dead_sum = 0.0
                for idx in indices:
                    entropy, _, used_frac, dead_frac = entropy_bits(idx, k)
                    entropy_sum += entropy
                    used_sum += used_frac
                    dead_sum += dead_frac
                    empirical_bits += float(idx.numel()) * entropy
                nstages = max(1, len(indices))
                stats["active_scalar_bits"] += sum_float(active_bits_group)
                stats["active_rvq_empirical_bits"] += empirical_bits
                stats["active_rvq_fixed_bits"] += float(vec.shape[0] * nstages * math.log2(k))
                stats["index_entropy_sum"] += entropy_sum
                stats["index_used_sum"] += used_sum
                stats["index_dead_sum"] += dead_sum
                stats["index_stage_count"] += float(nstages)
                active_entries.append(
                    {
                        "part": part_idx,
                        "group": group,
                        "key": key,
                        "indices": indices,
                    }
                )
        valid_bits = scalar_bits * mask
        stats["inactive_scalar_bits"] += sum_float(valid_bits - active_bits)
        stats["original_scalar_bits"] += sum_float(valid_bits)
        y_hat_part = y_q_part + means_hat
        inactive_y_q_parts.append(inactive_y_q.detach().cpu())
        y_hat_parts.append(y_hat_part)
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    y_hat = sum(y_hat_parts) * q_dec
    x_hat = net.vqgan.generator(net.dec(y_hat, net.q_dec[q : q + 1]))
    stage_count = stats["index_stage_count"]
    stats.update(
        {
            "bits_z": float(common["bits_z"]),
            "hybrid_empirical_bits_y": stats["inactive_scalar_bits"] + stats["active_rvq_empirical_bits"],
            "hybrid_fixed_bits_y": stats["inactive_scalar_bits"] + stats["active_rvq_fixed_bits"],
            "index_entropy_mean": stats["index_entropy_sum"] / stage_count if stage_count > 0 else 0.0,
            "index_used_frac_mean": stats["index_used_sum"] / stage_count if stage_count > 0 else 0.0,
            "index_dead_frac_mean": stats["index_dead_sum"] / stage_count if stage_count > 0 else 0.0,
        }
    )
    symbol = {
        "z_index": common["z_index"],
        "z_get_shape": common["z_get_shape"],
        "inactive_y_q_parts": inactive_y_q_parts,
        "active_entries": active_entries,
    }
    return x_hat, symbol, stats


def decode_branch_symbols(
    net: GLC_Image,
    q: int,
    symbol: dict[str, Any],
    codebooks: TrainableRVQCodebooks,
    branch_args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    params = z_to_params(net, symbol["z_index"], tuple(symbol["z_get_shape"]), device)
    _, q_dec, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    first = symbol["inactive_y_q_parts"][0].to(device)
    b, c, h, w = first.shape
    masks = net.get_mask_four_parts(b, c, h, w, first.dtype, device)
    entries_by_part: dict[int, list[dict[str, Any]]] = {}
    for entry in symbol["active_entries"]:
        entries_by_part.setdefault(int(entry["part"]), []).append(entry)

    y_hat_parts: list[torch.Tensor] = []
    y_hat_so_far = None
    group_size = int(branch_args.group_size)
    for part_idx, mask in enumerate(masks):
        part_scales, part_means = prior_step(net, part_idx, mask, common_params, scales, means, y_hat_so_far)
        del part_scales
        means_hat = part_means * mask
        y_q_part = symbol["inactive_y_q_parts"][part_idx].to(device).clone()
        for entry in entries_by_part.get(part_idx, []):
            group = int(entry["group"])
            start = group * group_size
            end = min(start + group_size, c)
            spatial = mask[0, start].bool()
            recon = reconstruct_active_group(codebooks, int(entry["key"]), entry["indices"], device)
            yq_group = y_q_part[0, start:end].permute(1, 2, 0)
            yq_group[spatial] = recon
            y_q_part[0, start:end] = yq_group.permute(2, 0, 1)
        y_hat_part = y_q_part + means_hat
        y_hat_parts.append(y_hat_part)
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
    y_hat = sum(y_hat_parts) * q_dec
    return net.vqgan.generator(net.dec(y_hat, net.q_dec[q : q + 1]))


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def load_branch(args: argparse.Namespace, device: torch.device):
    payload: dict[str, Any] = torch.load(args.branch_checkpoint, map_location="cpu")
    hidden = args.controller_hidden or int(payload.get("args", {}).get("controller_hidden", 16))
    codebooks_by_q = {
        int(q): codebooks_from_state_dict(state, device)
        for q, state in payload["codebooks_by_q"].items()
        if int(q) in set(args.q_indexes)
    }
    controller = ReliabilityIndexMLP(
        ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=hidden, zero_bias=-2.0)
    ).to(device)
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval()
    feature_mu = {str(k): float(v) for k, v in payload["feature_mu"].items()}
    feature_std = {str(k): float(v) for k, v in payload["feature_std"].items()}
    return payload, codebooks_by_q, controller, feature_mu, feature_std


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=2, default=str) + "\n")
    lines = [
        "# GLC/HCG Symbol Round-Trip Audit",
        "",
        f"Checkpoint: `{args.branch_checkpoint}`",
        f"Input: `{args.input_path}`",
        f"Rows: `{len(rows)}`",
        "",
        "## Summary",
        "",
        f"- max base symbol diff: `{summary['max_base_symbol_diff']:.6g}`",
        f"- max branch symbol diff: `{summary['max_branch_symbol_diff']:.6g}`",
        f"- max hard-gate symbol diff: `{summary['max_hard_gate_symbol_diff']:.6g}`",
        f"- max soft-dual symbol diff: `{summary['max_soft_dual_symbol_diff']:.6g}`",
        f"- all diffs <= tolerance `{args.tolerance}`: `{summary['roundtrip_pass']}`",
        f"- mean base bpp: `{summary['mean_base_bpp']:.6f}`",
        f"- mean branch exact bpp: `{summary['mean_branch_exact_bpp']:.6f}`",
        f"- mean branch fixed bpp: `{summary['mean_branch_fixed_bpp']:.6f}`",
        f"- mean old replacement bpp: `{summary['mean_old_replacement_bpp']:.6f}`",
        f"- mean exact hard-gate bpp: `{summary['mean_hard_exact_bpp']:.6f}`",
        f"- mean fixed hard-gate bpp: `{summary['mean_hard_fixed_bpp']:.6f}`",
        f"- mean exact soft-dual bpp: `{summary['mean_soft_dual_exact_bpp']:.6f}`",
        f"- mean fixed soft-dual bpp: `{summary['mean_soft_dual_fixed_bpp']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- `branch_exact_bpp` counts RVQ indices with their empirical entropy; this mirrors GLC-style analytical rate accounting, not a production bitstream.",
        "- `branch_fixed_bpp` is the conservative fixed-length RVQ-index alternative.",
        "- `hard_exact_bpp` / `hard_fixed_bpp` send either the base stream or the branch stream plus a mode signal.",
        "- `soft_dual_exact_bpp` / `soft_dual_fixed_bpp` send z once, both base-y and branch-y/RVQ symbols, and the quantized soft gate. This is the strict rate required to reproduce a soft blend.",
        "- `old_replacement_bpp` is kept only to expose how much the earlier optimistic accounting differs from symbol-exact accounting.",
        "",
        "## Per-Image Rows",
        "",
        "| q | image | gate | selected | base bpp | branch exact | branch fixed | old repl | hard exact | hard fixed | soft dual exact | soft dual fixed | base diff | branch diff | hard diff | soft diff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['q_index']} | {row['image']} | {row['soft_gate']:.4f} | {row['hard_selected']} | "
            f"{row['base_bpp']:.6f} | {row['branch_exact_bpp']:.6f} | {row['branch_fixed_bpp']:.6f} | {row['old_replacement_bpp']:.6f} | "
            f"{row['hard_exact_bpp']:.6f} | {row['hard_fixed_bpp']:.6f} | {row['soft_dual_exact_bpp']:.6f} | {row['soft_dual_fixed_bpp']:.6f} | "
            f"{row['base_symbol_diff']:.2e} | {row['branch_symbol_diff']:.2e} | "
            f"{row['hard_gate_symbol_diff']:.2e} | {row['soft_dual_symbol_diff']:.2e} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main() -> None:
    init_func()
    args = parse_args()
    device = torch.device(args.device)
    payload, codebooks_by_q, controller, feature_mu, feature_std = load_branch(args, device)

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for param in net.parameters():
        param.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    net.masks = {}

    branch_args = SimpleNamespace(
        group_size=args.group_size,
        active_groups=args.active_groups,
        active_parts=args.active_parts,
        scope=args.scope,
        k=args.k,
        stages=args.stages,
    )

    eval_paths = list_images(args.input_path, args.eval_start_index, args.eval_limit)
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.input_path}")
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for q in args.q_indexes:
            if q not in codebooks_by_q:
                raise SystemExit(f"q={q} missing from branch checkpoint")
            codebooks = codebooks_by_q[q]
            for idx, path in enumerate(eval_paths):
                item = prepare_images([path], device, args.padding_size, args.eval_crop_size)[0]
                pixels = float(item.height * item.width)

                net.forward_four_part_prior = official_forward
                base_forward_pad, base_run_stats = run_instrumented(net, item.x_pad, q)
                base_symbol_pad, base_symbol, base_symbol_stats = encode_base_symbols(net, item.x_pad, q)
                base_decode_pad = decode_from_yq_parts(net, q, base_symbol, base_symbol["y_q_parts"], device)

                install_trainable_branch(net, codebooks, branch_args)
                branch_forward_pad, branch_run_stats = run_instrumented(net, item.x_pad, q)
                net.forward_four_part_prior = official_forward
                branch_symbol_pad, branch_symbol, branch_symbol_stats = encode_branch_symbols(
                    net, item.x_pad, q, codebooks, branch_args
                )
                branch_decode_pad = decode_branch_symbols(net, q, branch_symbol, codebooks, branch_args, device)

                base = crop_to_image(base_forward_pad, item)
                base_dec = crop_to_image(base_decode_pad, item)
                branch = crop_to_image(branch_forward_pad, item)
                branch_dec = crop_to_image(branch_decode_pad, item)
                base_symbol_img = crop_to_image(base_symbol_pad, item)
                branch_symbol_img = crop_to_image(branch_symbol_pad, item)

                feature_row = branch_feature_dict(base_run_stats, branch_run_stats, pixels)
                features = feature_tensor(feature_row, feature_mu, feature_std, device)
                ctrl = controller(features)
                soft_mixed, soft_gate = mix_with_fallback(
                    base,
                    branch,
                    ctrl["active_logit"],
                    active_threshold=args.active_threshold,
                    hard=False,
                    max_gate=args.max_gate,
                )
                hard_mixed, hard_gate = mix_with_fallback(
                    base,
                    branch,
                    ctrl["active_logit"],
                    active_threshold=args.active_threshold,
                    hard=True,
                    max_gate=args.max_gate,
                )
                soft_gate_mean = float(soft_gate.mean().item())
                hard_gate_mean = float(hard_gate.mean().item())
                hard_selected = hard_gate_mean > 0.5
                if args.quantize_soft_gate_bits > 0:
                    levels = float((1 << int(args.quantize_soft_gate_bits)) - 1)
                    qgate = torch.round(soft_gate.mean().clamp(0, args.max_gate) * levels) / levels
                    gate_bits = float(args.quantize_soft_gate_bits)
                else:
                    qgate = soft_gate.mean().clamp(0, args.max_gate)
                    gate_bits = 0.0
                soft_quantized = base + qgate * (branch - base)
                hard_decode = branch_dec if hard_selected else base_dec
                soft_dual_decode = base_dec + qgate * (branch_dec - base_dec)

                base_bpp = float(base_symbol_stats["gaussian_bits_total"]) / pixels
                z_bpp = float(base_symbol_stats["bits_z"]) / pixels
                branch_exact_bpp = (
                    float(branch_symbol_stats["hybrid_empirical_bits_y"]) + float(branch_symbol_stats["bits_z"])
                ) / pixels
                branch_fixed_bpp = (
                    float(branch_symbol_stats["hybrid_fixed_bits_y"]) + float(branch_symbol_stats["bits_z"])
                ) / pixels
                old_replacement_bpp = base_bpp + float(feature_row["active_replacement_delta_bpp"])
                signal_bpp = image_signal_bpp(args.selection_signal_bits, item)
                gate_bpp = image_signal_bpp(gate_bits, item)
                hard_exact_bpp = (branch_exact_bpp if hard_selected else base_bpp) + signal_bpp
                hard_fixed_bpp = (branch_fixed_bpp if hard_selected else base_bpp) + signal_bpp
                soft_dual_exact_bpp = base_bpp + branch_exact_bpp - z_bpp + gate_bpp + signal_bpp
                soft_dual_fixed_bpp = base_bpp + branch_fixed_bpp - z_bpp + gate_bpp + signal_bpp

                row = {
                    "checkpoint_step": int(payload.get("step", -1)),
                    "q_index": int(q),
                    "image": path.name,
                    "height": int(item.height),
                    "width": int(item.width),
                    "base_bpp": base_bpp,
                    "z_bpp": z_bpp,
                    "branch_exact_bpp": branch_exact_bpp,
                    "branch_fixed_bpp": branch_fixed_bpp,
                    "old_replacement_bpp": old_replacement_bpp,
                    "hard_exact_bpp": hard_exact_bpp,
                    "hard_fixed_bpp": hard_fixed_bpp,
                    "soft_dual_exact_bpp": soft_dual_exact_bpp,
                    "soft_dual_fixed_bpp": soft_dual_fixed_bpp,
                    "active_rvq_empirical_bpp": float(branch_symbol_stats["active_rvq_empirical_bits"]) / pixels,
                    "active_rvq_fixed_bpp": float(branch_symbol_stats["active_rvq_fixed_bits"]) / pixels,
                    "inactive_scalar_bpp": float(branch_symbol_stats["inactive_scalar_bits"]) / pixels,
                    "active_scalar_bpp": float(branch_symbol_stats["active_scalar_bits"]) / pixels,
                    "soft_gate": soft_gate_mean,
                    "quantized_soft_gate": float(qgate.item()),
                    "hard_gate": hard_gate_mean,
                    "hard_selected": int(hard_selected),
                    "index_entropy_mean": float(branch_symbol_stats["index_entropy_mean"]),
                    "base_symbol_diff": max(tensor_max_abs(base, base_dec), tensor_max_abs(base, base_symbol_img)),
                    "branch_symbol_diff": max(tensor_max_abs(branch, branch_dec), tensor_max_abs(branch, branch_symbol_img)),
                    "hard_gate_symbol_diff": tensor_max_abs(hard_mixed, hard_decode),
                    "soft_dual_symbol_diff": tensor_max_abs(soft_quantized, soft_dual_decode),
                }
                row["old_replacement_under_count_bpp"] = row["branch_exact_bpp"] - row["old_replacement_bpp"]
                row["soft_dual_extra_vs_old_replacement_bpp"] = row["soft_dual_exact_bpp"] - row["old_replacement_bpp"]
                rows.append(row)
                print(
                    f"[audit] q={q} {idx + 1}/{len(eval_paths)} {path.name} "
                    f"base_diff={row['base_symbol_diff']:.2e} branch_diff={row['branch_symbol_diff']:.2e} "
                    f"hard_diff={row['hard_gate_symbol_diff']:.2e} soft_diff={row['soft_dual_symbol_diff']:.2e} "
                    f"hard_bpp={hard_exact_bpp:.6f} soft_dual_bpp={soft_dual_exact_bpp:.6f}"
                )
                del item
                torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(args.branch_checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "rows": len(rows),
        "max_base_symbol_diff": max_or_nan([r["base_symbol_diff"] for r in rows]),
        "max_branch_symbol_diff": max_or_nan([r["branch_symbol_diff"] for r in rows]),
        "max_hard_gate_symbol_diff": max_or_nan([r["hard_gate_symbol_diff"] for r in rows]),
        "max_soft_dual_symbol_diff": max_or_nan([r["soft_dual_symbol_diff"] for r in rows]),
        "mean_base_bpp": mean([r["base_bpp"] for r in rows]),
        "mean_branch_exact_bpp": mean([r["branch_exact_bpp"] for r in rows]),
        "mean_branch_fixed_bpp": mean([r["branch_fixed_bpp"] for r in rows]),
        "mean_old_replacement_bpp": mean([r["old_replacement_bpp"] for r in rows]),
        "mean_hard_exact_bpp": mean([r["hard_exact_bpp"] for r in rows]),
        "mean_hard_fixed_bpp": mean([r["hard_fixed_bpp"] for r in rows]),
        "mean_soft_dual_exact_bpp": mean([r["soft_dual_exact_bpp"] for r in rows]),
        "mean_soft_dual_fixed_bpp": mean([r["soft_dual_fixed_bpp"] for r in rows]),
        "mean_old_replacement_under_count_bpp": mean([r["old_replacement_under_count_bpp"] for r in rows]),
        "mean_soft_dual_extra_vs_old_replacement_bpp": mean([r["soft_dual_extra_vs_old_replacement_bpp"] for r in rows]),
    }
    summary["roundtrip_pass"] = bool(
        summary["max_base_symbol_diff"] <= args.tolerance
        and summary["max_branch_symbol_diff"] <= args.tolerance
        and summary["max_hard_gate_symbol_diff"] <= args.tolerance
        and summary["max_soft_dual_symbol_diff"] <= args.tolerance
    )
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
