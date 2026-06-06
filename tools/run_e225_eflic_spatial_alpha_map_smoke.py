#!/usr/bin/env python3
"""EF-LIC projected-HCG spatial alpha-map smoke.

This is a codec-path scaffold for the next HCG-RVQ step after the E221-E224
offline selector probes. It replaces a scalar alpha with a decoder-reproducible
spatial alpha map computed from predecision EF-LIC context. The goal is not to
claim final performance, but to verify that local fallback/strength control can
be represented without side bits and with exact decode reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(EFLIC_DIR))
sys.path.insert(0, str(ROOT / "tools"))

from EF_LIC import G_CH_Y, N_E, ResidualVectorQuantizeDropInfer, VectorQuantizerProjInfer, model  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import (  # noqa: E402
    compare_inds,
    hcg_rvq_encode_decode,
    index_stats,
    make_projected_direction,
    mean_psnr,
    tensor_stats,
)
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


def reduce_map(x: torch.Tensor, kind: str) -> torch.Tensor:
    xf = x.detach().float()
    if xf.shape[1] == 0:
        return xf.new_zeros((xf.shape[0], 1, xf.shape[2], xf.shape[3]))
    if kind == "abs_mean":
        return xf.abs().mean(dim=1, keepdim=True)
    if kind == "rms":
        return torch.sqrt(xf.square().mean(dim=1, keepdim=True).clamp_min(0.0))
    if kind == "std":
        return xf.std(dim=1, keepdim=True, unbiased=False)
    raise ValueError(f"unknown map reduction {kind!r}")


def top_fraction_mask(score: torch.Tensor, active_frac: float) -> torch.Tensor:
    if active_frac <= 0.0:
        return torch.zeros_like(score, dtype=torch.bool)
    if active_frac >= 1.0:
        return torch.ones_like(score, dtype=torch.bool)
    b = score.shape[0]
    flat = score.reshape(b, -1)
    k = max(1, int(round(flat.shape[1] * active_frac)))
    idx = torch.topk(flat, k=k, dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.view_as(score)


def soft_alpha_from_score(score: torch.Tensor, alpha: float) -> torch.Tensor:
    """Map a decoder-reproducible local score to a smooth alpha map.

    The normalization is per image and uses only mean score magnitude, avoiding
    learned or transmitted thresholds. The average strength is intentionally
    conservative compared with `constant`, while still allowing every position
    to receive a small geometry update.
    """

    score_f = score.detach().float().clamp_min(0.0)
    flat = score_f.reshape(score_f.shape[0], -1)
    denom = flat.mean(dim=1, keepdim=True).clamp_min(1e-6).view(score_f.shape[0], 1, 1, 1)
    normalized = (score_f / (2.0 * denom)).clamp(0.0, 1.0)
    return normalized.to(dtype=score.dtype) * float(alpha)


def build_alpha_map(
    *,
    mode: str,
    alpha: float,
    active_frac: float,
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
) -> torch.Tensor:
    if mode == "zero":
        return mean.new_zeros((mean.shape[0], 1, mean.shape[2], mean.shape[3]))
    if mode == "constant":
        return mean.new_full((mean.shape[0], 1, mean.shape[2], mean.shape[3]), float(alpha))

    c = G_CH_Y
    known_support = support_buf[:, : (4 + slice_id) * c]
    soft = mode.endswith("_soft")
    base_mode = mode[: -len("_soft")] if soft else mode
    if base_mode == "mean_abs_top":
        score = reduce_map(mean, "abs_mean")
    elif base_mode == "scale_rms_top":
        score = reduce_map(scale, "rms")
    elif base_mode == "support_rms_top":
        score = reduce_map(known_support, "rms")
    elif base_mode == "prev_rms_top":
        if slice_id == 0:
            return mean.new_zeros((mean.shape[0], 1, mean.shape[2], mean.shape[3]))
        prev = support_buf[:, 4 * c : (4 + slice_id) * c]
        score = reduce_map(prev, "rms")
    elif base_mode == "prev_over_scale_top":
        if slice_id == 0:
            return mean.new_zeros((mean.shape[0], 1, mean.shape[2], mean.shape[3]))
        prev = support_buf[:, 4 * c : (4 + slice_id) * c]
        score = reduce_map(prev, "rms") / reduce_map(scale, "rms").clamp_min(1e-6)
    elif base_mode == "support_over_scale_top":
        score = reduce_map(known_support, "rms") / reduce_map(scale, "rms").clamp_min(1e-6)
    else:
        raise ValueError(f"unknown alpha-map mode {mode!r}")

    if soft:
        return soft_alpha_from_score(score, alpha)

    mask = top_fraction_mask(score, active_frac)
    return mask.to(dtype=mean.dtype) * float(alpha)


def apply_rank1_geometry_map(x: torch.Tensor, v: torch.Tensor, alpha_map: torch.Tensor) -> torch.Tensor:
    if float(alpha_map.abs().max().item()) == 0.0:
        return x
    dot = (x * v).sum(dim=1, keepdim=True)
    return x - (2.0 * alpha_map) * dot * v


def invert_rank1_geometry_map(x: torch.Tensor, v: torch.Tensor, alpha_map: torch.Tensor) -> torch.Tensor:
    if float(alpha_map.abs().max().item()) == 0.0:
        return x
    denom = (1.0 - 2.0 * alpha_map).clamp_min(1e-6)
    coeff = (2.0 * alpha_map) / denom
    dot = (x * v).sum(dim=1, keepdim=True)
    return x + coeff * dot * v


def projected_hcg_stage_map(
    q: VectorQuantizerProjInfer,
    residual: torch.Tensor,
    context: torch.Tensor,
    alpha_map: torch.Tensor,
    direction_source: str,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    z_proj = q.in_proj(residual)
    v = make_projected_direction(q, context, direction_source, stage)
    z_query = apply_rank1_geometry_map(z_proj, v, alpha_map)
    b, c, h, w = z_query.shape
    flat = z_query.permute(0, 2, 3, 1).reshape(-1, c)
    dist = torch.mm(flat, q._codebook_T)
    dist.mul_(-2.0).add_(q._codebook_norm_sq)
    ind_flat = dist.argmin(1)
    code = q.embedding(ind_flat).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    code_inv = invert_rank1_geometry_map(code, v, alpha_map)
    decoded = q.out_proj(code_inv)
    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_query - z_proj, "geometry_delta"))
    stats.update(tensor_stats(decoded - residual, "residual_error"))
    stats.update(index_stats(ind_flat.view(b, h, w), q.n_e, "index"))
    return ind_flat.view(b, h, w), decoded, stats


def hcg_rvq_encode_decode_map(
    rvq: ResidualVectorQuantizeDropInfer,
    z: torch.Tensor,
    context: torch.Tensor,
    alpha_map: torch.Tensor,
    direction_source: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    residual = z.clone()
    z_q = torch.zeros_like(z)
    packed = None
    stage_stats: list[dict[str, float]] = []
    last = rvq.n_codebooks - 1
    for i, q in enumerate(rvq.quantizers):
        ind_i, z_q_i, stats_i = projected_hcg_stage_map(q, residual, context, alpha_map, direction_source, i)
        z_q = z_q + z_q_i
        if i != last:
            residual = residual - z_q_i
        if packed is None:
            packed = ind_i.new_empty((rvq.n_codebooks, *ind_i.shape))
        packed[i].copy_(ind_i)
        stage_stats.append({f"stage{i}_{k}": v for k, v in stats_i.items()})
    assert packed is not None
    out: dict[str, float] = {}
    for item in stage_stats:
        out.update(item)
    out["avg_index_entropy"] = float(np.mean([s[f"stage{i}_index_entropy"] for i, s in enumerate(stage_stats)]))
    out["avg_index_used_frac"] = float(np.mean([s[f"stage{i}_index_used_frac"] for i, s in enumerate(stage_stats)]))
    out["avg_geometry_delta_rms"] = float(np.mean([s[f"stage{i}_geometry_delta_rms"] for i, s in enumerate(stage_stats)]))
    out["avg_residual_error_rms"] = float(np.mean([s[f"stage{i}_residual_error_rms"] for i, s in enumerate(stage_stats)]))
    return packed, z_q, out


def hcg_rvq_decode_map(
    rvq: ResidualVectorQuantizeDropInfer,
    inds: torch.Tensor,
    context: torch.Tensor,
    alpha_map: torch.Tensor,
    direction_source: str,
) -> torch.Tensor:
    acc = None
    for i, q in enumerate(rvq.quantizers):
        b, h, w = inds[i].shape
        v = make_projected_direction(q, context, direction_source, i)
        code = q.embedding(inds[i].reshape(-1)).view(b, h, w, q.e_dim).permute(0, 3, 1, 2).contiguous()
        code_inv = invert_rank1_geometry_map(code, v, alpha_map)
        decoded = q.out_proj(code_inv)
        acc = decoded if acc is None else acc + decoded
    assert acc is not None
    return acc


@torch.inference_mode()
def active_compress_forward_map(
    net: model,
    x: torch.Tensor,
    *,
    force_ind: int,
    mode: str,
    alpha: float,
    active_frac: float,
    direction_source: str,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    y = net.g_a(x)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    y_inds = []
    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(index_stats(z_inds, int(N_E[-1]), "z_index"))
    for i in range(4):
        mean, scale = net._mean_scale(support_buf, i)
        net._qt_select(y, i, y_slice)
        y_norm = (y_slice - mean) / scale
        alpha_map = build_alpha_map(
            mode=mode,
            alpha=alpha,
            active_frac=active_frac,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=i,
        )
        context = scale if direction_source == "logscale" else mean
        ind_i, y_hat_norm_i, slice_stats = hcg_rvq_encode_decode_map(
            net.quantizes[force_ind][i], y_norm, context, alpha_map, direction_source
        )
        y_hat_i = y_hat_norm_i * scale + mean
        net._qt_put_(y_hat, y_hat_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)
        stats[f"slice{i}_alpha_mean"] = float(alpha_map.mean().item())
        stats[f"slice{i}_alpha_active_frac"] = float((alpha_map > 0).float().mean().item())
        for key, value in slice_stats.items():
            stats[f"slice{i}_{key}"] = value
    for metric in ["alpha_mean", "alpha_active_frac", "avg_index_entropy", "avg_index_used_frac", "avg_geometry_delta_rms", "avg_residual_error_rms"]:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, net.g_s(y_hat), stats


@torch.inference_mode()
def active_decompress_map(
    net: model,
    inds: dict[str, Any],
    *,
    force_ind: int,
    mode: str,
    alpha: float,
    active_frac: float,
    direction_source: str,
) -> torch.Tensor:
    z_hat = net.quantizes[force_ind][-1].decoding(inds["z_inds"])
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    for i, ind_i in enumerate(inds["y_inds"]):
        mean, scale = net._mean_scale(support_buf, i)
        alpha_map = build_alpha_map(
            mode=mode,
            alpha=alpha,
            active_frac=active_frac,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=i,
        )
        context = scale if direction_source == "logscale" else mean
        y_i = hcg_rvq_decode_map(net.quantizes[force_ind][i], ind_i, context, alpha_map, direction_source)
        y_i = y_i * scale + mean
        net._qt_put_(y_hat, y_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_i)
    return net.g_s(y_hat)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak_first4")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e225_eflic_spatial_alpha_map_smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--mode", nargs="+", default=["zero", "constant", "mean_abs_top", "scale_rms_top", "support_rms_top", "prev_rms_top"])
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--active-frac", type=float, default=0.25)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--with-perceptual", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.image_dir)[args.start_index : args.start_index + args.max_images]
    if not images:
        raise SystemExit(f"no images selected from {args.image_dir}")
    if abs(1.0 - 2.0 * args.alpha) < 1e-6:
        raise ValueError("alpha=0.5 is singular")

    lpips_fn = None
    dists_fn = None
    if args.with_perceptual:
        import DISTS_pytorch as dists
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        dists_fn = dists.DISTS().to(device).eval()

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    rows: list[dict[str, Any]] = []
    for mode in args.mode:
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            orig_inds = net.compress(padded.clone(), force_ind=args.force_ind)
            orig_payload, _, _ = pack_inds(net, orig_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=args.force_ind)[:, :, :h, :w]

            if mode == "zero":
                active_inds, active_x_hat_forward, stats = active_compress_forward_map(
                    net,
                    padded.clone(),
                    force_ind=args.force_ind,
                    mode=mode,
                    alpha=0.0,
                    active_frac=0.0,
                    direction_source=args.direction_source,
                )
            elif mode == "constant":
                ind_scalar, x_scalar, stats_scalar = hcg_rvq_scalar_path(
                    net, padded.clone(), args.force_ind, args.alpha, args.direction_source
                )
                active_inds, active_x_hat_forward, stats = ind_scalar, x_scalar, stats_scalar
            else:
                active_inds, active_x_hat_forward, stats = active_compress_forward_map(
                    net,
                    padded.clone(),
                    force_ind=args.force_ind,
                    mode=mode,
                    alpha=args.alpha,
                    active_frac=args.active_frac,
                    direction_source=args.direction_source,
                )

            active_payload, _, _ = pack_inds(net, active_inds)
            active_x_hat_dec = active_decompress_map(
                net,
                active_inds,
                force_ind=args.force_ind,
                mode="constant" if mode == "constant" else mode,
                alpha=args.alpha if mode != "zero" else 0.0,
                active_frac=1.0 if mode == "constant" else args.active_frac,
                direction_source=args.direction_source,
            )[:, :, :h, :w]
            active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
            active_x_hat = active_x_hat_dec
            diff = (active_x_hat_fwd - active_x_hat_dec).abs()
            mismatch = compare_inds(orig_inds, active_inds)
            row: dict[str, Any] = {
                "mode": mode,
                "image": path.name,
                "force_ind": args.force_ind,
                "direction_source": args.direction_source,
                "alpha": 0.0 if mode == "zero" else args.alpha,
                "active_frac_target": 0.0 if mode == "zero" else (1.0 if mode == "constant" else args.active_frac),
                "bpp": len(active_payload) * 8.0 / float(h * w),
                "delta_bpp": (len(active_payload) - len(orig_payload)) * 8.0 / float(h * w),
                "payload_len_equal": int(len(active_payload) == len(orig_payload)),
                "payload_equal": int(active_payload == orig_payload),
                "base_psnr": psnr_from_mse(mse01(orig_x_hat, frame)),
                "active_psnr": psnr_from_mse(mse01(active_x_hat, frame)),
                "max_decode_diff": float(diff.max().item()),
                "mean_decode_diff": float(diff.mean().item()),
                "nonfinite": int(
                    (not torch.isfinite(active_x_hat).all().item())
                    or (not torch.isfinite(active_x_hat_fwd).all().item())
                    or any(isinstance(v, float) and not math.isfinite(v) for v in stats.values())
                ),
            }
            row["delta_psnr"] = row["active_psnr"] - row["base_psnr"]
            if lpips_fn is not None and dists_fn is not None:
                row["base_lpips"] = float(lpips_fn(orig_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                row["active_lpips"] = float(lpips_fn(active_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                row["delta_lpips"] = row["active_lpips"] - row["base_lpips"]
                row["base_dists"] = float(
                    dists_fn(((orig_x_hat + 1.0) * 0.5).clamp(0, 1), ((frame + 1.0) * 0.5).clamp(0, 1), require_grad=False)
                    .detach()
                    .mean()
                    .item()
                )
                row["active_dists"] = float(
                    dists_fn(((active_x_hat + 1.0) * 0.5).clamp(0, 1), ((frame + 1.0) * 0.5).clamp(0, 1), require_grad=False)
                    .detach()
                    .mean()
                    .item()
                )
                row["delta_dists"] = row["active_dists"] - row["base_dists"]
            row.update(mismatch)
            row.update(stats)
            rows.append(row)
            metric_text = f"dPSNR={row['delta_psnr']:+.4f}"
            if "delta_dists" in row:
                metric_text += f" dDISTS={row['delta_dists']:+.5f} dLPIPS={row['delta_lpips']:+.5f}"
            print(
                f"mode={mode} image={path.name} {metric_text} "
                f"dbpp={row['delta_bpp']:+.6f} decmax={row['max_decode_diff']:.2e} nonfinite={row['nonfinite']}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for mode in args.mode:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        item: dict[str, Any] = {
            "mode": mode,
            "images": len(subset),
            "bpp": float(np.mean([r["bpp"] for r in subset])),
            "delta_bpp": float(np.mean([r["delta_bpp"] for r in subset])),
            "base_psnr": mean_psnr(subset, "base_psnr"),
            "active_psnr": mean_psnr(subset, "active_psnr"),
            "delta_psnr": float(np.mean([r["delta_psnr"] for r in subset])),
            "max_decode_diff": float(max(r["max_decode_diff"] for r in subset)),
            "nonfinite_rows": int(sum(r["nonfinite"] for r in subset)),
            "y_mismatch_frac": float(np.sum([r["y_mismatch"] for r in subset]) / max(1, np.sum([r["y_total"] for r in subset]))),
            "alpha_active_frac": float(np.mean([r.get("y_alpha_active_frac", 0.0) for r in subset])),
            "geometry_delta_rms": float(np.mean([r.get("y_avg_geometry_delta_rms", 0.0) for r in subset])),
        }
        if "delta_dists" in subset[0]:
            item.update(
                {
                    "delta_dists": float(np.mean([r["delta_dists"] for r in subset])),
                    "delta_lpips": float(np.mean([r["delta_lpips"] for r in subset])),
                }
            )
        summary.append(item)

    payload = {
        "experiment": "E225 EF-LIC spatial alpha-map smoke",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.image_dir),
        "device": str(device),
        "force_ind": args.force_ind,
        "direction_source": args.direction_source,
        "alpha": args.alpha,
        "active_frac": args.active_frac,
        "images": [p.name for p in images],
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    with md_path.open("w") as f:
        f.write("# E225 EF-LIC Spatial Alpha-Map Smoke\n\n")
        f.write("This verifies decoder-reproducible local alpha maps for projected HCG inside EF-LIC.\n\n")
        f.write(f"Dataset: `{args.image_dir}`  \n")
        f.write(f"Images: `{len(images)}`  \n")
        f.write(f"Force index: `{args.force_ind}`  \n")
        f.write(f"Alpha: `{args.alpha}`  \n")
        f.write(f"Active fraction: `{args.active_frac}`  \n")
        f.write(f"Perceptual metrics: `{bool(args.with_perceptual)}`  \n\n")
        keys = ["mode", "images", "delta_bpp", "delta_psnr", "max_decode_diff", "nonfinite_rows", "y_mismatch_frac", "alpha_active_frac", "geometry_delta_rms"]
        if summary and "delta_dists" in summary[0]:
            keys.extend(["delta_dists", "delta_lpips"])
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for item in summary:
            vals = []
            for key in keys:
                val = item.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:+.8f}" if key.startswith("delta") else f"{val:.8f}")
                else:
                    vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\nInterpretation:\n\n")
        f.write("- `zero` should reproduce the baseline path; nonzero decode diff there would indicate a scaffold bug.\n")
        f.write("- `constant` is the scalar projected-HCG reference expressed through the same alpha-map decoder path.\n")
        f.write("- Top-fraction modes are no-sidebit deterministic local controllers; they are diagnostics before training a codec-aware local head.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


@torch.inference_mode()
def hcg_rvq_scalar_path(
    net: model,
    x: torch.Tensor,
    force_ind: int,
    alpha: float,
    direction_source: str,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    y = net.g_a(x)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    y_inds = []
    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(index_stats(z_inds, int(N_E[-1]), "z_index"))
    for i in range(4):
        mean, scale = net._mean_scale(support_buf, i)
        net._qt_select(y, i, y_slice)
        y_norm = (y_slice - mean) / scale
        context = scale if direction_source == "logscale" else mean
        ind_i, y_hat_norm_i, slice_stats = hcg_rvq_encode_decode(
            net.quantizes[force_ind][i], y_norm, context, alpha, direction_source
        )
        y_hat_i = y_hat_norm_i * scale + mean
        net._qt_put_(y_hat, y_hat_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)
        stats[f"slice{i}_alpha_mean"] = float(alpha)
        stats[f"slice{i}_alpha_active_frac"] = 1.0 if alpha > 0 else 0.0
        for key, value in slice_stats.items():
            stats[f"slice{i}_{key}"] = value
    for metric in ["alpha_mean", "alpha_active_frac", "avg_index_entropy", "avg_index_used_frac", "avg_geometry_delta_rms", "avg_residual_error_rms"]:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, net.g_s(y_hat), stats


if __name__ == "__main__":
    main()
