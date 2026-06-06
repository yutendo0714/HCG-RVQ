#!/usr/bin/env python3
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

from EF_LIC import G_CH_Y, ResidualVectorQuantizeDropInfer, VectorQuantizerProjInfer, model  # noqa: E402
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
                f"{prefix}_min": float(xf.min().item()),
                f"{prefix}_max": float(xf.max().item()),
            }
        )
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


def projected_hcg_identity_stage(q: VectorQuantizerProjInfer, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One EF-LIC projected VQ stage with explicit HCG identity hooks.

    This intentionally keeps the same in/out projection and codebook search as
    EF-LIC. The HCG fields are identity: mu=0, scale=1, no Householder transform.
    If this does not reproduce the original path exactly, active HCG geometry
    would be scientifically unsafe to test.
    """

    z_proj = q.in_proj(residual)
    z_hcg = z_proj  # identity shift/scale/geometry in projected 8-D code space
    b, c, h, w = z_hcg.shape
    flat = z_hcg.permute(0, 2, 3, 1).reshape(-1, c)
    dist = torch.mm(flat, q._codebook_T)
    dist.mul_(-2.0).add_(q._codebook_norm_sq)
    ind_flat = dist.argmin(1)
    decoded_flat = q._decode_flat(ind_flat)
    decoded = decoded_flat.view(b, h, w, q._decode_dim).permute(0, 3, 1, 2).contiguous()
    stats = {}
    stats.update(tensor_stats(z_proj, "proj"))
    stats.update(index_stats(ind_flat.view(b, h, w), q.n_e, "index"))
    return ind_flat.view(b, h, w), decoded, stats


def projected_hcg_identity_encode_decode(rvq: ResidualVectorQuantizeDropInfer, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    residual = z
    z_q = torch.zeros_like(z)
    packed = None
    last = rvq.n_codebooks - 1
    stats_by_stage: list[dict[str, float]] = []
    for i, q in enumerate(rvq.quantizers):
        ind_i, z_q_i, stats_i = projected_hcg_identity_stage(q, residual)
        z_q.add_(z_q_i)
        if i != last:
            residual.sub_(z_q_i)
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
    flat_stats.update(tensor_stats(z_q, "decoded_sum"))
    return packed, z_q, flat_stats


def projected_hcg_identity_encoding(rvq: ResidualVectorQuantizeDropInfer, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    residual = z
    packed = None
    last = rvq.n_codebooks - 1
    stats_by_stage: list[dict[str, float]] = []
    for i, q in enumerate(rvq.quantizers):
        ind_i, z_q_i, stats_i = projected_hcg_identity_stage(q, residual)
        if i != last:
            residual.sub_(z_q_i)
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
    return packed, flat_stats


def projected_hcg_identity_decoding(rvq: ResidualVectorQuantizeDropInfer, inds: torch.Tensor) -> torch.Tensor:
    n = inds.shape[0]
    _, b, h, w = inds.shape
    m = b * h * w
    c = rvq.quantizers[0]._decode_dim
    acc = rvq.quantizers[0]._decode_codebook.new_zeros((m, c))
    buf = acc.new_empty((m, c))
    for i in range(n):
        rvq.quantizers[i]._decode_flat(inds[i], out=buf)
        acc.add_(buf)
    return acc.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


@torch.inference_mode()
def wrapper_compress(net: model, x: torch.Tensor, force_ind: int) -> tuple[dict[str, Any], dict[str, float]]:
    y = net.g_a(x)
    z_inds, z_hat, z_stats = projected_hcg_identity_encode_decode(net.quantizes[force_ind][-1], net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))

    b, _, h2, w2 = support_buf.shape
    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    y_inds = []
    stats: dict[str, float] = {f"z_{k}": v for k, v in z_stats.items()}
    stats.update(tensor_stats(y, "y"))
    stats.update(tensor_stats(z_hat, "z_hat"))

    for i in range(4):
        mean, scale = net._mean_scale(support_buf, i)
        net._qt_select(y, i, y_slice)
        y_norm = y_slice.sub(mean).div(scale)
        stats.update(tensor_stats(mean, f"slice{i}_mean"))
        stats.update(tensor_stats(scale, f"slice{i}_scale"))
        stats.update(tensor_stats(y_norm, f"slice{i}_y_norm"))
        if i < 3:
            ind_i, y_hat_i, slice_stats = projected_hcg_identity_encode_decode(net.quantizes[force_ind][i], y_norm)
            y_hat_i.mul_(scale).add_(mean)
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        else:
            ind_i, slice_stats = projected_hcg_identity_encoding(net.quantizes[force_ind][i], y_norm)
        y_inds.append(ind_i)
        for key, value in slice_stats.items():
            stats[f"slice{i}_{key}"] = value

    for metric in ["avg_index_entropy", "avg_index_used_frac"]:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, stats


def compare_inds(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    z_mismatch = int((a["z_inds"] != b["z_inds"]).sum().item())
    y_mismatch = 0
    y_total = 0
    for ai, bi in zip(a["y_inds"], b["y_inds"]):
        y_mismatch += int((ai != bi).sum().item())
        y_total += int(ai.numel())
    return {
        "z_mismatch": z_mismatch,
        "z_total": int(a["z_inds"].numel()),
        "y_mismatch": y_mismatch,
        "y_total": y_total,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kodak-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak_first4")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e158_eflic_projected_identity_first4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.kodak_dir)

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    rows: list[dict[str, Any]] = []
    for force_ind in args.force_ind:
        net.prepare_inference_(force_ind=force_ind)
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            orig_inds = net.compress(padded.clone(), force_ind=force_ind)
            wrap_inds, stats = wrapper_compress(net, padded.clone(), force_ind=force_ind)
            mismatch = compare_inds(orig_inds, wrap_inds)

            orig_payload, _, _ = pack_inds(net, orig_inds)
            wrap_payload, _, _ = pack_inds(net, wrap_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=force_ind)[:, :, :h, :w]
            wrap_x_hat = net.decompress(wrap_inds, force_ind=force_ind)[:, :, :h, :w]
            xhat_diff = (orig_x_hat - wrap_x_hat).abs()
            mse = mse01(wrap_x_hat, frame)

            row: dict[str, Any] = {
                "force_ind": force_ind,
                "image": path.name,
                "bpp": len(wrap_payload) * 8.0 / float(h * w),
                "orig_payload_bytes": len(orig_payload),
                "wrap_payload_bytes": len(wrap_payload),
                "payload_equal": int(orig_payload == wrap_payload),
                "psnr": psnr_from_mse(mse),
                "max_abs_xhat_diff": float(xhat_diff.max().item()),
                "mean_abs_xhat_diff": float(xhat_diff.mean().item()),
                "nonfinite": int((not torch.isfinite(orig_x_hat).all().item()) or (not torch.isfinite(wrap_x_hat).all().item())),
            }
            row.update(mismatch)
            row.update(stats)
            rows.append(row)
            print(
                f"force={force_ind} {path.name} payload_equal={row['payload_equal']} "
                f"z_mis={row['z_mismatch']} y_mis={row['y_mismatch']} "
                f"maxdiff={row['max_abs_xhat_diff']:.3e} nonfinite={row['nonfinite']}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for force_ind in args.force_ind:
        subset = [r for r in rows if r["force_ind"] == force_ind]
        summary.append(
            {
                "force_ind": force_ind,
                "images": len(subset),
                "payload_equal_rows": int(sum(r["payload_equal"] for r in subset)),
                "z_mismatch": int(sum(r["z_mismatch"] for r in subset)),
                "y_mismatch": int(sum(r["y_mismatch"] for r in subset)),
                "max_abs_xhat_diff": float(max(r["max_abs_xhat_diff"] for r in subset)),
                "mean_abs_xhat_diff": float(np.mean([r["mean_abs_xhat_diff"] for r in subset])),
                "nonfinite_rows": int(sum(r["nonfinite"] for r in subset)),
                "bpp": float(np.mean([r["bpp"] for r in subset])),
                "psnr": psnr_from_mse(float(np.mean([10 ** (-float(r["psnr"]) / 10.0) for r in subset]))),
                "y_index_entropy": float(np.mean([r["y_avg_index_entropy"] for r in subset])),
                "y_index_used_frac": float(np.mean([r["y_avg_index_used_frac"] for r in subset])),
            }
        )

    payload = {
        "experiment": "E158 EF-LIC projected HCG identity wrapper",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.kodak_dir),
        "device": str(device),
        "rows": len(rows),
        "summary": summary,
        "decision": "Identity projected-space wrapper must match original EF-LIC before active HCG geometry is tested.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E158 EF-LIC Projected Identity Wrapper",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Dataset: `{args.kodak_dir}`",
        f"Device: `{device}`",
        "",
        "| force_ind | images | payload equal | z mismatch | y mismatch | max xhat diff | mean xhat diff | nonfinite | bpp | PSNR | y index H | y used frac |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['force_ind']} | {s['images']} | {s['payload_equal_rows']} | {s['z_mismatch']} | {s['y_mismatch']} | "
            f"{s['max_abs_xhat_diff']:.3e} | {s['mean_abs_xhat_diff']:.3e} | {s['nonfinite_rows']} | "
            f"{s['bpp']:.6f} | {s['psnr']:.4f} | {s['y_index_entropy']:.4f} | {s['y_index_used_frac']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This wrapper explicitly exposes the HCG insertion point in EF-LIC's 8-D projected RVQ space.",
            "- A zero-mismatch result means future active geometry tests can be attributed to HCG changes rather than wrapper/protocol drift.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
