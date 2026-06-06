#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(EFLIC_DIR))

from EF_LIC import G_CH_Y, N_E, model  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad, unpack_inds  # noqa: E402


def tensor_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    x = x.detach()
    finite = torch.isfinite(x)
    out = {f"{prefix}_finite_frac": float(finite.float().mean().item())}
    if finite.any():
        xf = x[finite].float()
        out.update(
            {
                f"{prefix}_mean": float(xf.mean().item()),
                f"{prefix}_std": float(xf.std(unbiased=False).item()),
                f"{prefix}_min": float(xf.min().item()),
                f"{prefix}_max": float(xf.max().item()),
            }
        )
    else:
        out.update(
            {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
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


@torch.inference_mode()
def instrumented_forward(net: model, padded: torch.Tensor, force_ind: int):
    y = net.g_a(padded)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))

    b, _, h2, w2 = support_buf.shape
    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    y_inds = []

    y_norm_stats = []
    mean_stats = []
    scale_stats = []
    y_index_stats = []

    for i in range(4):
        mean, scale = net._mean_scale(support_buf, i)
        net._qt_select(y, i, y_slice)
        y_norm = (y_slice - mean).div(scale)

        ind_i, y_hat_i = net.quantizes[force_ind][i].encode_decode(y_norm.clone())
        y_hat_i = y_hat_i.mul(scale).add(mean)
        net._qt_put_(y_hat, y_hat_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)

        y_norm_stats.append(tensor_stats(y_norm, f"slice{i}_y_norm"))
        mean_stats.append(tensor_stats(mean, f"slice{i}_mean"))
        scale_stats.append(tensor_stats(scale, f"slice{i}_scale"))
        y_index_stats.append(index_stats(ind_i, int(N_E[i]), f"slice{i}_index"))

    x_hat = net.g_s(y_hat)
    inds = {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}

    row_stats: dict[str, float] = {}
    row_stats.update(tensor_stats(y, "y"))
    row_stats.update(tensor_stats(z_hat, "z_hat"))
    row_stats.update(index_stats(z_inds, int(N_E[-1]), "z_index"))
    for stats_group in (y_norm_stats, mean_stats, scale_stats, y_index_stats):
        for stats in stats_group:
            row_stats.update(stats)

    for key_family in ["y_norm", "mean", "scale", "index"]:
        vals: dict[str, list[float]] = {}
        for i in range(4):
            for key, val in row_stats.items():
                if key.startswith(f"slice{i}_{key_family}"):
                    suffix = key.split(f"slice{i}_{key_family}_", 1)[1]
                    vals.setdefault(suffix, []).append(val)
        for suffix, arr in vals.items():
            row_stats[f"slice_avg_{key_family}_{suffix}"] = float(np.nanmean(arr))

    return x_hat, inds, row_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kodak-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e156_eflic_pretrained_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.kodak_dir)

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    rows: list[dict[str, float | int | str]] = []
    for force_ind in args.force_ind:
        net.prepare_inference_(force_ind=force_ind)
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_hat_padded, inds, stats = instrumented_forward(net, padded, force_ind)
            payload, meta, total_bits = pack_inds(net, inds)
            if device.type == "cuda":
                torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t0) * 1000.0

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            decoded_inds = unpack_inds(payload, meta, total_bits, device)
            x_hat = net.decompress(decoded_inds, force_ind=force_ind)[:, :, :h, :w]
            if device.type == "cuda":
                torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - t1) * 1000.0

            mse = mse01(x_hat, frame)
            row = {
                "force_ind": force_ind,
                "image": path.name,
                "bpp": len(payload) * 8.0 / float(h * w),
                "psnr": psnr_from_mse(mse),
                "lpips": float(lpips_fn(x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
                "dists": float(
                    dists_fn(
                        ((x_hat + 1.0) * 0.5).clamp(0, 1),
                        ((frame + 1.0) * 0.5).clamp(0, 1),
                        require_grad=False,
                    ).detach().mean().item()
                ),
                "enc_ms": enc_ms,
                "dec_ms": dec_ms,
                "nonfinite": int(
                    (not torch.isfinite(x_hat).all().item())
                    or (not torch.isfinite(x_hat_padded).all().item())
                    or any(v != v for v in stats.values())
                ),
            }
            row.update(stats)
            rows.append(row)
            print(
                f"force={force_ind} {path.name} bpp={row['bpp']:.5f} psnr={row['psnr']:.4f} "
                f"lpips={row['lpips']:.5f} dists={row['dists']:.5f} nonfinite={row['nonfinite']}"
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

    summary = []
    for force_ind in args.force_ind:
        subset = [r for r in rows if r["force_ind"] == force_ind]
        summary.append(
            {
                "force_ind": force_ind,
                "images": len(subset),
                "bpp": float(np.mean([r["bpp"] for r in subset])),
                "psnr": psnr_from_mse(float(np.mean([10 ** (-float(r["psnr"]) / 10.0) for r in subset]))),
                "lpips": float(np.mean([r["lpips"] for r in subset])),
                "dists": float(np.mean([r["dists"] for r in subset])),
                "enc_ms": float(np.mean([r["enc_ms"] for r in subset])),
                "dec_ms": float(np.mean([r["dec_ms"] for r in subset])),
                "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
                "y_index_entropy": float(np.mean([r["slice_avg_index_entropy"] for r in subset])),
                "y_index_used_frac": float(np.mean([r["slice_avg_index_used_frac"] for r in subset])),
                "y_norm_std": float(np.mean([r["slice_avg_y_norm_std"] for r in subset])),
                "scale_mean": float(np.mean([r["slice_avg_scale_mean"] for r in subset])),
            }
        )

    payload = {
        "checkpoint": str(args.ckpt_path),
        "kodak_dir": str(args.kodak_dir),
        "device": str(device),
        "rows": len(rows),
        "summary": summary,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = ["# E156 EF-LIC Pretrained Baseline", "", f"Checkpoint: `{args.ckpt_path}`", f"Dataset: `{args.kodak_dir}`", f"Device: `{device}`", "", "| force_ind | images | bpp | PSNR | LPIPS | DISTS | enc ms | dec ms | nonfinite | y index H | y used frac | y norm std | scale mean |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(
            f"| {s['force_ind']} | {s['images']} | {s['bpp']:.6f} | {s['psnr']:.4f} | {s['lpips']:.5f} | {s['dists']:.5f} | "
            f"{s['enc_ms']:.2f} | {s['dec_ms']:.2f} | {s['nonfinite_rows']} | {s['y_index_entropy']:.4f} | "
            f"{s['y_index_used_frac']:.4f} | {s['y_norm_std']:.4f} | {s['scale_mean']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
