#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, from_minus1_1_to_0_1, get_state_dict  # noqa: E402


def list_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def write_image(path: Path, x_hat_m11: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = from_minus1_1_to_0_1(x_hat_m11.detach().clamp(-1, 1)).squeeze(0).cpu()
    arr = out.permute(1, 2, 0).numpy()
    arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


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


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


@torch.inference_mode()
def instrumented_glc_test(net: GLC_Image, x_m11: torch.Tensor, q_index: int) -> tuple[torch.Tensor, dict[str, float]]:
    curr_q_enc = net.q_enc[q_index : q_index + 1]
    curr_q_dec = net.q_dec[q_index : q_index + 1]

    y_ori = net.vqgan.encoder(x_m11)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)

    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))

    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)
    q_enc, q_dec, prior_scales, prior_means = net.separate_prior(params)
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

    stats: dict[str, float] = {
        "bit_y": bit_y,
        "bit_z": bit_z,
        "bit_total": bit_y + bit_z,
        "q_enc_scalar_mean": float(curr_q_enc.mean().item()),
        "q_dec_scalar_mean": float(curr_q_dec.mean().item()),
    }
    stats.update(tensor_stats(y_ori, "y_ori"))
    stats.update(tensor_stats(y, "y"))
    stats.update(tensor_stats(z, "z"))
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(tensor_stats(params, "params"))
    stats.update(tensor_stats(q_enc, "prior_q_enc"))
    stats.update(tensor_stats(q_dec, "prior_q_dec"))
    stats.update(tensor_stats(prior_scales, "prior_scales"))
    stats.update(tensor_stats(prior_means, "prior_means"))
    stats.update(tensor_stats(y_res, "y_res"))
    stats.update(tensor_stats(y_q, "y_q"))
    stats.update(tensor_stats(y_hat_prior, "y_hat_prior"))
    stats.update(tensor_stats(y_hat_dec, "y_hat_dec"))
    stats.update(tensor_stats(scales_hat, "scales_hat"))
    stats.update(index_stats(index, int(net.codebook_size), "z_vq_index"))
    return x_hat, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak_first4")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e162_glc_pretrained_kodak_first4")
    p.add_argument("--recon-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def mean_psnr(rows: list[dict[str, Any]]) -> float:
    mses = [10 ** (-float(r["psnr"]) / 10.0) for r in rows]
    mse = float(np.mean(mses))
    return -10.0 * math.log10(mse) if mse > 0 else float("inf")


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

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for path in images:
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_hat_pad, stats = instrumented_glc_test(net, x_pad, q)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encdec_ms = (time.perf_counter() - t0) * 1000.0

            x_hat = F.pad(x_hat_pad, (-padding_l, -padding_r, -padding_t, -padding_b)).clamp(-1, 1)
            x_hat01 = from_minus1_1_to_0_1(x_hat).clamp(0, 1)
            bit_total = float(stats["bit_total"])
            row: dict[str, Any] = {
                "q_index": q,
                "image": path.name,
                "height": h,
                "width": w,
                "bpp": bit_total / float(h * w),
                "bpp_y": float(stats["bit_y"]) / float(h * w),
                "bpp_z": float(stats["bit_z"]) / float(h * w),
                "psnr": psnr01(x_hat01, img01),
                "ms_ssim": float(ms_ssim(x_hat01, img01, data_range=1.0).item()),
                "lpips": float(lpips_fn(x_hat.clamp(-1, 1), x.clamp(-1, 1)).mean().item()),
                "dists": float(dists_fn(x_hat01, img01, require_grad=False).detach().mean().item()),
                "encdec_ms": encdec_ms,
                "nonfinite": int((not torch.isfinite(x_hat).all().item()) or any(isinstance(v, float) and not math.isfinite(v) for v in stats.values())),
            }
            row.update(stats)
            rows.append(row)
            if args.recon_dir is not None:
                write_image(args.recon_dir / f"q{q}" / path.name, x_hat)
            print(
                f"q={q} {path.name} bpp={row['bpp']:.5f} y={row['bpp_y']:.5f} z={row['bpp_z']:.5f} "
                f"psnr={row['psnr']:.3f} lpips={row['lpips']:.5f} dists={row['dists']:.5f} nonfinite={row['nonfinite']}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    summary=[]
    for q in args.q_indexes:
        subset=[r for r in rows if int(r["q_index"]) == q]
        if not subset:
            continue
        summary.append(
            {
                "q_index": q,
                "images": len(subset),
                "bpp": float(np.mean([r["bpp"] for r in subset])),
                "bpp_y": float(np.mean([r["bpp_y"] for r in subset])),
                "bpp_z": float(np.mean([r["bpp_z"] for r in subset])),
                "psnr": mean_psnr(subset),
                "ms_ssim": float(np.mean([r["ms_ssim"] for r in subset])),
                "lpips": float(np.mean([r["lpips"] for r in subset])),
                "dists": float(np.mean([r["dists"] for r in subset])),
                "encdec_ms": float(np.mean([r["encdec_ms"] for r in subset])),
                "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
                "z_index_entropy": float(np.mean([r["z_vq_index_entropy"] for r in subset])),
                "z_index_used_frac": float(np.mean([r["z_vq_index_used_frac"] for r in subset])),
                "y_std": float(np.mean([r["y_std"] for r in subset])),
                "z_std": float(np.mean([r["z_std"] for r in subset])),
                "y_q_std": float(np.mean([r["y_q_std"] for r in subset])),
                "scales_hat_mean": float(np.mean([r["scales_hat_mean"] for r in subset])),
            }
        )

    payload={
        "experiment": "E162 GLC pretrained baseline",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "lpips_net": args.lpips_net,
        "padding_size": args.padding_size,
        "rows": len(rows),
        "summary": summary,
        "note": "FID/KID are intentionally not computed here; this evaluator focuses on reproducible per-image metrics and intermediate distributions.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines=[
        "# E162 GLC Pretrained Baseline",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        f"LPIPS net: `{args.lpips_net}`",
        "",
        "FID/KID are not computed in this evaluator; the purpose is per-image metric and intermediate-feature reproduction.",
        "",
        "| q | images | bpp | bpp_y | bpp_z | PSNR | MS-SSIM | LPIPS | DISTS | encdec ms | nonfinite | z H | z used | y std | y_q std | scales mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['q_index']} | {s['images']} | {s['bpp']:.6f} | {s['bpp_y']:.6f} | {s['bpp_z']:.6f} | "
            f"{s['psnr']:.4f} | {s['ms_ssim']:.5f} | {s['lpips']:.5f} | {s['dists']:.5f} | {s['encdec_ms']:.2f} | {s['nonfinite_rows']} | "
            f"{s['z_index_entropy']:.4f} | {s['z_index_used_frac']:.4f} | {s['y_std']:.4f} | {s['y_q_std']:.4f} | {s['scales_hat_mean']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
