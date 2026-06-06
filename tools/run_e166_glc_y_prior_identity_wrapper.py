#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, get_state_dict  # noqa: E402
from tools.run_e162_glc_pretrained_baseline import list_images, load_image, tensor_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e166_glc_y_prior_identity_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def finite_max_abs(x: torch.Tensor) -> float:
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(x[finite].abs().max().item())


def hcg_ready_identity_process_with_mask(self: GLC_Image, y: torch.Tensor, scales: torch.Tensor, means: torch.Tensor, mask: torch.Tensor):
    scales_hat = scales * mask
    means_hat = means * mask
    y_res = (y - means_hat) * mask
    y_q = self.quant(y_res)
    y_hat = y_q + means_hat
    return y_res, y_q, y_hat, scales_hat, means_hat


def hcg_ready_identity_forward_four_part_prior(
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
    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

    y_scaled = y * q_enc
    stats: dict[str, float] = {
        "mask0_frac": float(mask_0.mean().item()),
        "mask1_frac": float(mask_1.mean().item()),
        "mask2_frac": float(mask_2.mean().item()),
        "mask3_frac": float(mask_3.mean().item()),
    }
    stats.update(tensor_stats(q_enc, "q_enc"))
    stats.update(tensor_stats(q_dec, "q_dec"))
    stats.update(tensor_stats(y_scaled, "y_scaled"))

    y_res_0, y_q_0, y_hat_0, s_hat_0, m_hat_0 = hcg_ready_identity_process_with_mask(self, y_scaled, scales, means, mask_0)
    stats.update(tensor_stats(y_res_0, "part0_res"))
    stats.update(tensor_stats(y_q_0, "part0_q"))
    stats.update(tensor_stats(s_hat_0, "part0_scale"))
    stats.update(tensor_stats(m_hat_0, "part0_mean"))

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    y_res_1, y_q_1, y_hat_1, s_hat_1, m_hat_1 = hcg_ready_identity_process_with_mask(self, y_scaled, scales, means, mask_1)
    stats.update(tensor_stats(y_res_1, "part1_res"))
    stats.update(tensor_stats(y_q_1, "part1_q"))
    stats.update(tensor_stats(s_hat_1, "part1_scale"))
    stats.update(tensor_stats(m_hat_1, "part1_mean"))

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    y_res_2, y_q_2, y_hat_2, s_hat_2, m_hat_2 = hcg_ready_identity_process_with_mask(self, y_scaled, scales, means, mask_2)
    stats.update(tensor_stats(y_res_2, "part2_res"))
    stats.update(tensor_stats(y_q_2, "part2_q"))
    stats.update(tensor_stats(s_hat_2, "part2_scale"))
    stats.update(tensor_stats(m_hat_2, "part2_mean"))

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    y_res_3, y_q_3, y_hat_3, s_hat_3, m_hat_3 = hcg_ready_identity_process_with_mask(self, y_scaled, scales, means, mask_3)
    stats.update(tensor_stats(y_res_3, "part3_res"))
    stats.update(tensor_stats(y_q_3, "part3_q"))
    stats.update(tensor_stats(s_hat_3, "part3_scale"))
    stats.update(tensor_stats(m_hat_3, "part3_mean"))

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    y_hat = y_hat * q_dec

    stats.update(tensor_stats(y_res, "combined_res"))
    stats.update(tensor_stats(y_q, "combined_q"))
    stats.update(tensor_stats(y_hat, "combined_y_hat"))
    stats.update(tensor_stats(scales_hat, "combined_scales"))
    self._e166_y_prior_identity_stats = stats
    return y_res, y_q, y_hat, scales_hat


def install_identity_y_prior_wrapper(net: GLC_Image) -> None:
    net.forward_four_part_prior = types.MethodType(hcg_ready_identity_forward_four_part_prior, net)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    official_forward_four_part_prior = net.forward_four_part_prior

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for path in images:
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            net.forward_four_part_prior = official_forward_four_part_prior
            original = net.test(x_pad, q)
            install_identity_y_prior_wrapper(net)
            wrapped = net.test(x_pad, q)
            net.forward_four_part_prior = official_forward_four_part_prior
            stats = getattr(net, "_e166_y_prior_identity_stats", {})

            x_diff = original["x_hat"] - wrapped["x_hat"]
            ref_diff = original["ref_latent"] - wrapped["ref_latent"]
            bit_y_diff = float(original["bit_y"]) - float(wrapped["bit_y"])
            bit_z_diff = float(original["bit_z"]) - float(wrapped["bit_z"])
            bit_total_diff = float(original["bit"]) - float(wrapped["bit"])
            nonfinite = int(
                (not torch.isfinite(original["x_hat"]).all().item())
                or (not torch.isfinite(wrapped["x_hat"]).all().item())
                or any(not math.isfinite(v) for v in [bit_y_diff, bit_z_diff, bit_total_diff])
                or any(isinstance(v, float) and not math.isfinite(v) for v in stats.values())
            )

            row: dict[str, Any] = {
                "q_index": q,
                "image": path.name,
                "height": h,
                "width": w,
                "max_abs_xhat_diff": finite_max_abs(x_diff),
                "mean_abs_xhat_diff": float(torch.nan_to_num(x_diff.abs()).mean().item()),
                "max_abs_ref_latent_diff": finite_max_abs(ref_diff),
                "mean_abs_ref_latent_diff": float(torch.nan_to_num(ref_diff.abs()).mean().item()),
                "bit_y_diff": bit_y_diff,
                "bit_z_diff": bit_z_diff,
                "bit_total_diff": bit_total_diff,
                "orig_bit_y": float(original["bit_y"]),
                "wrapped_bit_y": float(wrapped["bit_y"]),
                "orig_bit_z": float(original["bit_z"]),
                "wrapped_bit_z": float(wrapped["bit_z"]),
                "nonfinite": nonfinite,
            }
            row.update(stats)
            rows.append(row)
            print(
                f"q={q} {path.name} max_x={row['max_abs_xhat_diff']:.3e} "
                f"max_ref={row['max_abs_ref_latent_diff']:.3e} bit={row['bit_total_diff']:.3e} "
                f"res_std={row.get('combined_res_std', float('nan')):.4f} nonfinite={nonfinite}"
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
                "max_abs_xhat_diff": float(np.max([r["max_abs_xhat_diff"] for r in subset])),
                "max_abs_ref_latent_diff": float(np.max([r["max_abs_ref_latent_diff"] for r in subset])),
                "max_abs_bit_y_diff": float(np.max(np.abs([r["bit_y_diff"] for r in subset]))),
                "max_abs_bit_z_diff": float(np.max(np.abs([r["bit_z_diff"] for r in subset]))),
                "max_abs_bit_total_diff": float(np.max(np.abs([r["bit_total_diff"] for r in subset]))),
                "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
                "combined_res_std": float(np.mean([r["combined_res_std"] for r in subset])),
                "combined_q_std": float(np.mean([r["combined_q_std"] for r in subset])),
                "combined_scales_mean": float(np.mean([r["combined_scales_mean"] for r in subset])),
                "part0_res_std": float(np.mean([r["part0_res_std"] for r in subset])),
                "part1_res_std": float(np.mean([r["part1_res_std"] for r in subset])),
                "part2_res_std": float(np.mean([r["part2_res_std"] for r in subset])),
                "part3_res_std": float(np.mean([r["part3_res_std"] for r in subset])),
            }
        )

    payload = {
        "experiment": "E166 GLC y-prior HCG-ready identity wrapper",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "padding_size": args.padding_size,
        "rows": len(rows),
        "summary": summary,
        "note": "This exposes the main GLC y-path HCG insertion boundary while preserving official behavior exactly.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E166 GLC y-Prior HCG-Ready Identity Wrapper",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        "",
        "This replaces `forward_four_part_prior()` with an identity wrapper that exposes per-part HCG insertion hooks.",
        "",
        "| q | images | max xhat diff | max ref-latent diff | max bit_y diff | max bit_z diff | max total-bit diff | nonfinite | res std | q std | scales mean | part res std 0/1/2/3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        part = f"{s['part0_res_std']:.4f}/{s['part1_res_std']:.4f}/{s['part2_res_std']:.4f}/{s['part3_res_std']:.4f}"
        lines.append(
            f"| {s['q_index']} | {s['images']} | {s['max_abs_xhat_diff']:.3e} | {s['max_abs_ref_latent_diff']:.3e} | "
            f"{s['max_abs_bit_y_diff']:.3e} | {s['max_abs_bit_z_diff']:.3e} | {s['max_abs_bit_total_diff']:.3e} | "
            f"{s['nonfinite_rows']} | {s['combined_res_std']:.4f} | {s['combined_q_std']:.4f} | "
            f"{s['combined_scales_mean']:.4f} | {part} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
