#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(EFLIC_DIR))

from EF_LIC import G_CH_Y, N_E, ResidualVectorQuantizeDropInfer, VectorQuantizerProjInfer, model  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


def tensor_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    x = x.detach()
    finite = torch.isfinite(x)
    out: dict[str, float] = {f"{prefix}_finite_frac": float(finite.float().mean().item())}
    if finite.any():
        xf = x[finite].float()
        out.update(
            {
                f"{prefix}_mean": float(xf.mean().item()),
                f"{prefix}_std": float(xf.std(unbiased=False).item()),
                f"{prefix}_abs_mean": float(xf.abs().mean().item()),
                f"{prefix}_rms": float(torch.sqrt((xf * xf).mean()).item()),
                f"{prefix}_min": float(xf.min().item()),
                f"{prefix}_max": float(xf.max().item()),
            }
        )
    else:
        out.update({f"{prefix}_{k}": float("nan") for k in ["mean", "std", "abs_mean", "rms", "min", "max"]})
    return out


def index_stats(indices: torch.Tensor, n_e: int, prefix: str) -> dict[str, float]:
    flat = indices.detach().to("cpu", torch.long).reshape(-1)
    counts = torch.bincount(flat, minlength=n_e).float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    nz = probs > 0
    entropy = float((-(probs[nz] * torch.log2(probs[nz])).sum()).item())
    return {
        f"{prefix}_entropy": entropy,
        f"{prefix}_perplexity": float(2.0**entropy),
        f"{prefix}_used_frac": float((counts > 0).float().mean().item()),
        f"{prefix}_max_prob": float(probs.max().item()),
    }


def normalize_v(v: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-6)
    return v / norm


def make_projected_direction(
    q: VectorQuantizerProjInfer,
    context: torch.Tensor,
    source: str,
    stage: int,
) -> torch.Tensor:
    b, _, h, w = context.shape
    if source == "mean":
        v = q.in_proj(context)
    elif source == "logscale":
        v = q.in_proj(torch.log(context.clamp_min(1e-6)))
    elif source == "fixed":
        v = context.new_zeros((b, q.e_dim, h, w))
        v[:, stage % q.e_dim].fill_(1.0)
    else:
        raise ValueError(f"unknown direction source: {source}")
    return normalize_v(v)


def apply_rank1_geometry(x: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0:
        return x
    dot = (x * v).sum(dim=1, keepdim=True)
    return x - (2.0 * alpha) * dot * v


def invert_rank1_geometry(x: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0:
        return x
    denom = 1.0 - 2.0 * alpha
    if abs(denom) < 1e-6:
        raise ValueError("alpha too close to singular value 0.5")
    coeff = (2.0 * alpha) / denom
    dot = (x * v).sum(dim=1, keepdim=True)
    return x + coeff * dot * v


def projected_hcg_stage(
    q: VectorQuantizerProjInfer,
    residual: torch.Tensor,
    context: torch.Tensor,
    alpha: float,
    direction_source: str,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    z_proj = q.in_proj(residual)
    v = make_projected_direction(q, context, direction_source, stage)
    z_query = apply_rank1_geometry(z_proj, v, alpha)
    b, c, h, w = z_query.shape
    flat = z_query.permute(0, 2, 3, 1).reshape(-1, c)
    dist = torch.mm(flat, q._codebook_T)
    dist.mul_(-2.0).add_(q._codebook_norm_sq)
    ind_flat = dist.argmin(1)

    code = q.embedding(ind_flat).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    code_inv = invert_rank1_geometry(code, v, alpha)
    decoded = q.out_proj(code_inv)

    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_proj, "proj"))
    stats.update(tensor_stats(z_query - z_proj, "geometry_delta"))
    stats.update(tensor_stats(decoded - residual, "residual_error"))
    stats.update(tensor_stats(code_inv - z_proj, "proj_quant_error"))
    stats.update(index_stats(ind_flat.view(b, h, w), q.n_e, "index"))
    stats["alpha"] = float(alpha)
    stats["direction_norm_mean"] = float(torch.linalg.vector_norm(v, dim=1).mean().item())
    return ind_flat.view(b, h, w), decoded, stats


def projected_hcg_decode_stage(
    q: VectorQuantizerProjInfer,
    indices: torch.Tensor,
    context: torch.Tensor,
    alpha: float,
    direction_source: str,
    stage: int,
) -> torch.Tensor:
    b, h, w = indices.shape
    c = q.e_dim
    v = make_projected_direction(q, context, direction_source, stage)
    code = q.embedding(indices.reshape(-1)).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    code_inv = invert_rank1_geometry(code, v, alpha)
    return q.out_proj(code_inv)


def hcg_rvq_encode_decode(
    rvq: ResidualVectorQuantizeDropInfer,
    z: torch.Tensor,
    context: torch.Tensor,
    alpha: float,
    direction_source: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    residual = z.clone()
    z_q = torch.zeros_like(z)
    packed = None
    stats_by_stage: list[dict[str, float]] = []
    last = rvq.n_codebooks - 1
    for i, q in enumerate(rvq.quantizers):
        ind_i, z_q_i, stats_i = projected_hcg_stage(q, residual, context, alpha, direction_source, i)
        z_q = z_q + z_q_i
        if i != last:
            residual = residual - z_q_i
        if packed is None:
            packed = ind_i.new_empty((rvq.n_codebooks, *ind_i.shape))
        packed[i].copy_(ind_i)
        stats_by_stage.append({f"stage{i}_{k}": v for k, v in stats_i.items()})
    assert packed is not None
    flat_stats: dict[str, float] = {}
    for s in stats_by_stage:
        flat_stats.update(s)
    flat_stats["avg_index_entropy"] = float(np.mean([s[f"stage{i}_index_entropy"] for i, s in enumerate(stats_by_stage)]))
    flat_stats["avg_index_used_frac"] = float(np.mean([s[f"stage{i}_index_used_frac"] for i, s in enumerate(stats_by_stage)]))
    flat_stats["avg_geometry_delta_rms"] = float(np.mean([s[f"stage{i}_geometry_delta_rms"] for i, s in enumerate(stats_by_stage)]))
    flat_stats["avg_residual_error_rms"] = float(np.mean([s[f"stage{i}_residual_error_rms"] for i, s in enumerate(stats_by_stage)]))
    flat_stats.update(tensor_stats(z_q, "decoded_sum"))
    return packed, z_q, flat_stats


def hcg_rvq_decode(
    rvq: ResidualVectorQuantizeDropInfer,
    inds: torch.Tensor,
    context: torch.Tensor,
    alpha: float,
    direction_source: str,
) -> torch.Tensor:
    acc = None
    for i, q in enumerate(rvq.quantizers[: inds.shape[0]]):
        dec_i = projected_hcg_decode_stage(q, inds[i], context, alpha, direction_source, i)
        acc = dec_i if acc is None else acc + dec_i
    assert acc is not None
    return acc


def parse_slice_alpha_schedule(spec: str) -> tuple[str, tuple[float, float, float, float]]:
    if ":" in spec:
        name, values_text = spec.split(":", 1)
        name = name.strip()
    else:
        name = ""
        values_text = spec
    values = tuple(float(v.strip()) for v in values_text.split(",") if v.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            f"slice alpha schedule must contain 4 comma-separated values, got {len(values)} in {spec!r}"
        )
    if not name:
        name = "slice_" + "_".join(f"{v:.6g}" for v in values)
    return name, (values[0], values[1], values[2], values[3])


def format_alpha_values(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.6g}" for v in values)


def build_alpha_schedules(args: argparse.Namespace) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    schedules: list[tuple[str, float, tuple[float, float, float, float]]] = []
    if args.slice_alpha_schedule:
        seen: set[str] = set()
        for spec in args.slice_alpha_schedule:
            name, values = parse_slice_alpha_schedule(spec)
            if name in seen:
                raise ValueError(f"duplicate slice alpha schedule name: {name}")
            seen.add(name)
            schedules.append((name, float(np.mean(values)), values))
    else:
        for alpha in args.alpha:
            value = float(alpha)
            schedules.append((f"alpha{value:.6g}", value, (value, value, value, value)))
    for name, _, values in schedules:
        if any(abs(1.0 - 2.0 * a) < 1e-6 for a in values):
            raise ValueError(f"{name} contains alpha=0.5, which is singular")
    return schedules


@torch.inference_mode()
def active_compress_forward(
    net: model,
    x: torch.Tensor,
    force_ind: int,
    slice_alphas: Sequence[float],
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
    stats.update(tensor_stats(y, "y"))
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(index_stats(z_inds, int(N_E[-1]), "z_index"))

    for i in range(4):
        alpha_i = float(slice_alphas[i])
        mean, scale = net._mean_scale(support_buf, i)
        net._qt_select(y, i, y_slice)
        y_norm = (y_slice - mean) / scale
        context = scale if direction_source == "logscale" else mean
        ind_i, y_hat_norm_i, slice_stats = hcg_rvq_encode_decode(
            net.quantizes[force_ind][i], y_norm, context, alpha_i, direction_source
        )
        y_hat_i = y_hat_norm_i * scale + mean
        net._qt_put_(y_hat, y_hat_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)

        stats.update(tensor_stats(mean, f"slice{i}_mean"))
        stats.update(tensor_stats(scale, f"slice{i}_scale"))
        stats.update(tensor_stats(y_norm, f"slice{i}_y_norm"))
        stats[f"slice{i}_selected_alpha"] = alpha_i
        for key, value in slice_stats.items():
            stats[f"slice{i}_{key}"] = value

    for metric in ["avg_index_entropy", "avg_index_used_frac", "avg_geometry_delta_rms", "avg_residual_error_rms"]:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))

    x_hat = net.g_s(y_hat)
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, x_hat, stats


@torch.inference_mode()
def active_decompress(
    net: model,
    inds: dict[str, Any],
    force_ind: int,
    slice_alphas: Sequence[float],
    direction_source: str,
) -> torch.Tensor:
    z_hat = net.quantizes[force_ind][-1].decoding(inds["z_inds"])
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    for i, ind_i in enumerate(inds["y_inds"]):
        alpha_i = float(slice_alphas[i])
        mean, scale = net._mean_scale(support_buf, i)
        context = scale if direction_source == "logscale" else mean
        y_i = hcg_rvq_decode(net.quantizes[force_ind][i], ind_i, context, alpha_i, direction_source)
        y_i = y_i * scale + mean
        net._qt_put_(y_hat, y_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_i)
    return net.g_s(y_hat)


def compare_inds(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    z_mismatch = int((a["z_inds"] != b["z_inds"]).sum().item())
    y_mismatch = 0
    y_total = 0
    for ai, bi in zip(a["y_inds"], b["y_inds"]):
        y_mismatch += int((ai != bi).sum().item())
        y_total += int(ai.numel())
    return {"z_mismatch": z_mismatch, "z_total": int(a["z_inds"].numel()), "y_mismatch": y_mismatch, "y_total": y_total}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kodak-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak_first4")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e160_eflic_projected_hcg_first4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--alpha", type=float, nargs="*", default=[0.02, 0.05, 0.10])
    p.add_argument(
        "--slice-alpha-schedule",
        action="append",
        default=[],
        help=(
            "Optional fixed per-slice alpha schedule in the form name:a0,a1,a2,a3. "
            "When set, --alpha is ignored and each schedule is evaluated as a deterministic no-sidebit controller."
        ),
    )
    p.add_argument("--direction-source", nargs="*", default=["mean"], choices=["mean", "logscale", "fixed"])
    p.add_argument("--start-index", type=int, default=0, help="Start offset after deterministic path sorting.")
    p.add_argument("--max-images", type=int, default=None, help="Maximum number of images to evaluate after start-index.")
    return p.parse_args()


def mean_psnr(rows: list[dict[str, Any]], key: str) -> float:
    return psnr_from_mse(float(np.mean([10 ** (-float(r[key]) / 10.0) for r in rows])))


def main() -> None:
    args = parse_args()
    alpha_schedules = build_alpha_schedules(args)
    device = torch.device(args.device)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive when set")
    all_images = list_images(args.kodak_dir)
    end_index = None if args.max_images is None else args.start_index + args.max_images
    images = all_images[args.start_index:end_index]
    if not images:
        raise SystemExit(f"no images selected from {args.kodak_dir} start={args.start_index} max={args.max_images}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    rows: list[dict[str, Any]] = []
    for force_ind in args.force_ind:
        net.prepare_inference_(force_ind=force_ind)
        for source in args.direction_source:
            for schedule_name, alpha, slice_alphas in alpha_schedules:
                for path in images:
                    frame = load_image(path, device)
                    _, _, h, w = frame.shape
                    padded = replicate_pad(frame, h, w)

                    orig_inds = net.compress(padded.clone(), force_ind=force_ind)
                    orig_payload, _, _ = pack_inds(net, orig_inds)
                    orig_x_hat = net.decompress(orig_inds, force_ind=force_ind)[:, :, :h, :w]

                    active_inds, active_x_hat_forward, stats = active_compress_forward(
                        net,
                        padded.clone(),
                        force_ind=force_ind,
                        slice_alphas=slice_alphas,
                        direction_source=source,
                    )
                    active_payload, _, _ = pack_inds(net, active_inds)
                    active_x_hat_dec = active_decompress(
                        net,
                        active_inds,
                        force_ind=force_ind,
                        slice_alphas=slice_alphas,
                        direction_source=source,
                    )[:, :, :h, :w]
                    active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
                    decode_diff = (active_x_hat_fwd - active_x_hat_dec).abs()
                    active_x_hat = active_x_hat_dec

                    mismatch = compare_inds(orig_inds, active_inds)
                    base_lpips = float(lpips_fn(orig_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                    active_lpips = float(lpips_fn(active_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                    base_dists = float(
                        dists_fn(((orig_x_hat + 1.0) * 0.5).clamp(0, 1), ((frame + 1.0) * 0.5).clamp(0, 1), require_grad=False)
                        .detach()
                        .mean()
                        .item()
                    )
                    active_dists = float(
                        dists_fn(((active_x_hat + 1.0) * 0.5).clamp(0, 1), ((frame + 1.0) * 0.5).clamp(0, 1), require_grad=False)
                        .detach()
                        .mean()
                        .item()
                    )
                    base_psnr = psnr_from_mse(mse01(orig_x_hat, frame))
                    active_psnr = psnr_from_mse(mse01(active_x_hat, frame))
                    nonfinite = int(
                        (not torch.isfinite(active_x_hat).all().item())
                        or (not torch.isfinite(active_x_hat_fwd).all().item())
                        or any(isinstance(v, float) and not math.isfinite(v) for v in stats.values())
                    )
                    row: dict[str, Any] = {
                        "force_ind": force_ind,
                        "direction_source": source,
                        "alpha": alpha,
                        "alpha_schedule": schedule_name,
                        "alpha_values": format_alpha_values(slice_alphas),
                        "image": path.name,
                        "bpp": len(active_payload) * 8.0 / float(h * w),
                        "payload_bytes": len(active_payload),
                        "payload_len_equal": int(len(active_payload) == len(orig_payload)),
                        "payload_equal": int(active_payload == orig_payload),
                        "base_psnr": base_psnr,
                        "active_psnr": active_psnr,
                        "delta_psnr": active_psnr - base_psnr,
                        "base_lpips": base_lpips,
                        "active_lpips": active_lpips,
                        "delta_lpips": active_lpips - base_lpips,
                        "base_dists": base_dists,
                        "active_dists": active_dists,
                        "delta_dists": active_dists - base_dists,
                        "max_decode_diff": float(decode_diff.max().item()),
                        "mean_decode_diff": float(decode_diff.mean().item()),
                        "nonfinite": nonfinite,
                    }
                    row.update(mismatch)
                    row.update(stats)
                    rows.append(row)
                    print(
                        f"force={force_ind} src={source} schedule={schedule_name} "
                        f"alphas={format_alpha_values(slice_alphas)} {path.name} "
                        f"dLPIPS={row['delta_lpips']:+.5f} dDISTS={row['delta_dists']:+.5f} "
                        f"dPSNR={row['delta_psnr']:+.4f} ymis={row['y_mismatch']} "
                        f"decmax={row['max_decode_diff']:.2e} nonfinite={nonfinite}"
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
    for force_ind in args.force_ind:
        for source in args.direction_source:
            for schedule_name, alpha, slice_alphas in alpha_schedules:
                subset = [
                    r
                    for r in rows
                    if r["force_ind"] == force_ind
                    and r["direction_source"] == source
                    and r["alpha_schedule"] == schedule_name
                ]
                if not subset:
                    continue
                summary.append(
                    {
                        "force_ind": force_ind,
                        "direction_source": source,
                        "alpha": alpha,
                        "alpha_schedule": schedule_name,
                        "alpha_values": format_alpha_values(slice_alphas),
                        "images": len(subset),
                        "bpp": float(np.mean([r["bpp"] for r in subset])),
                        "base_psnr": mean_psnr(subset, "base_psnr"),
                        "active_psnr": mean_psnr(subset, "active_psnr"),
                        "delta_psnr": float(np.mean([r["delta_psnr"] for r in subset])),
                        "base_lpips": float(np.mean([r["base_lpips"] for r in subset])),
                        "active_lpips": float(np.mean([r["active_lpips"] for r in subset])),
                        "delta_lpips": float(np.mean([r["delta_lpips"] for r in subset])),
                        "base_dists": float(np.mean([r["base_dists"] for r in subset])),
                        "active_dists": float(np.mean([r["active_dists"] for r in subset])),
                        "delta_dists": float(np.mean([r["delta_dists"] for r in subset])),
                        "payload_len_equal_rows": int(sum(r["payload_len_equal"] for r in subset)),
                        "payload_equal_rows": int(sum(r["payload_equal"] for r in subset)),
                        "y_mismatch_frac": float(np.sum([r["y_mismatch"] for r in subset]) / max(1, np.sum([r["y_total"] for r in subset]))),
                        "max_decode_diff": float(max(r["max_decode_diff"] for r in subset)),
                        "nonfinite_rows": int(sum(r["nonfinite"] for r in subset)),
                        "y_index_entropy": float(np.mean([r["y_avg_index_entropy"] for r in subset])),
                        "y_index_used_frac": float(np.mean([r["y_avg_index_used_frac"] for r in subset])),
                        "geometry_delta_rms": float(np.mean([r["y_avg_geometry_delta_rms"] for r in subset])),
                        "residual_error_rms": float(np.mean([r["y_avg_residual_error_rms"] for r in subset])),
                    }
                )

    payload = {
        "experiment": "E160 EF-LIC projected HCG active smoke",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.kodak_dir),
        "start_index": args.start_index,
        "max_images": args.max_images,
        "alpha_schedules": [
            {"name": name, "alpha": alpha, "values": list(values)} for name, alpha, values in alpha_schedules
        ],
        "evaluated_images": [p.name for p in images],
        "device": str(device),
        "rows": len(rows),
        "summary": summary,
        "interpretation": "Diagnostic active geometry only; not a trained quality claim.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E160 EF-LIC Projected HCG Active Smoke",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Dataset: `{args.kodak_dir}`",
        f"Start index: `{args.start_index}`",
        f"Max images: `{args.max_images}`",
        f"Device: `{device}`",
        "",
        "This is a diagnostic active-geometry smoke, not a trained quality claim.",
        "",
        "| force | source | schedule | alpha values | images | bpp | base LPIPS | active LPIPS | dLPIPS | base DISTS | active DISTS | dDISTS | dPSNR | y mismatch frac | max decode diff | nonfinite | geom RMS |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['force_ind']} | {s['direction_source']} | {s['alpha_schedule']} | {s['alpha_values']} | "
            f"{s['images']} | {s['bpp']:.6f} | "
            f"{s['base_lpips']:.5f} | {s['active_lpips']:.5f} | {s['delta_lpips']:+.5f} | "
            f"{s['base_dists']:.5f} | {s['active_dists']:.5f} | {s['delta_dists']:+.5f} | {s['delta_psnr']:+.4f} | "
            f"{s['y_mismatch_frac']:.6f} | {s['max_decode_diff']:.3e} | {s['nonfinite_rows']} | {s['geometry_delta_rms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `max_decode_diff` near zero means the active geometry is decoder-reproducible and therefore bitstream-valid under the same signaled indices.",
            "- Positive `dLPIPS`/`dDISTS` is worse. This smoke is expected to be conservative diagnostics because no HCG heads were trained for EF-LIC.",
            "- The next step should use these deltas and feature stats to choose whether to train a small projected-HCG head or build a teacher-label controller.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
