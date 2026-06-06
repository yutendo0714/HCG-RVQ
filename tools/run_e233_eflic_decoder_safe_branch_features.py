#!/usr/bin/env python3
"""Dump decoder-safe EF-LIC context features for branch-controller design.

E232 showed that multiple codec-valid HCG branch families are useful, but a
paper-main controller must choose them from information available at both
encoder and decoder before reconstructing each slice. This script records only
that predecision context: z_hat, z index statistics, mean/scale maps, support
buffer maps, and previous decoded support maps.

It intentionally does not evaluate new reconstruction quality. It creates the
feature table needed to test whether the E232 oracle branch labels are
predictable from decoder-safe context.
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

from EF_LIC import G_CH_Y, N_E, model  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import index_stats, tensor_stats  # noqa: E402
from run_e225_eflic_spatial_alpha_map_smoke import reduce_map  # noqa: E402
from test import load_checkpoint, load_image, list_images, replicate_pad  # noqa: E402


def quantile_value(x: torch.Tensor, q: float) -> float:
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, q).item())


def map_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    xf = x.detach().float()
    finite = torch.isfinite(xf)
    out = {f"{prefix}_finite_frac": float(finite.float().mean().item())}
    if not finite.any():
        for key in ["mean", "std", "abs_mean", "rms", "min", "max", "p50", "p90", "p95", "top25_mean"]:
            out[f"{prefix}_{key}"] = float("nan")
        return out
    vals = xf[finite]
    out.update(
        {
            f"{prefix}_mean": float(vals.mean().item()),
            f"{prefix}_std": float(vals.std(unbiased=False).item()),
            f"{prefix}_abs_mean": float(vals.abs().mean().item()),
            f"{prefix}_rms": float(torch.sqrt(vals.square().mean()).item()),
            f"{prefix}_min": float(vals.min().item()),
            f"{prefix}_max": float(vals.max().item()),
            f"{prefix}_p50": quantile_value(vals, 0.50),
            f"{prefix}_p90": quantile_value(vals, 0.90),
            f"{prefix}_p95": quantile_value(vals, 0.95),
        }
    )
    flat = vals.reshape(-1)
    k = max(1, int(round(flat.numel() * 0.25)))
    out[f"{prefix}_top25_mean"] = float(torch.topk(flat, k=k).values.mean().item())
    return out


def add_slice_context_features(
    row: dict[str, Any],
    *,
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
) -> None:
    c = G_CH_Y
    known_support = support_buf[:, : (4 + slice_id) * c]
    if slice_id == 0:
        prev = support_buf.new_zeros((support_buf.shape[0], 1, support_buf.shape[2], support_buf.shape[3]))
    else:
        prev = support_buf[:, 4 * c : (4 + slice_id) * c]

    mean_abs = reduce_map(mean, "abs_mean")
    mean_rms = reduce_map(mean, "rms")
    scale_rms = reduce_map(scale, "rms").clamp_min(1e-6)
    support_rms = reduce_map(known_support, "rms")
    prev_rms = reduce_map(prev, "rms")
    support_over_scale = support_rms / scale_rms
    prev_over_scale = prev_rms / scale_rms

    maps = {
        "mean_abs": mean_abs,
        "mean_rms": mean_rms,
        "scale_rms": scale_rms,
        "support_rms": support_rms,
        "prev_rms": prev_rms,
        "support_over_scale": support_over_scale,
        "prev_over_scale": prev_over_scale,
    }
    for name, value in maps.items():
        row.update(map_stats(value, f"slice{slice_id}_{name}"))

    row[f"slice{slice_id}_support_channels"] = float(known_support.shape[1])
    row[f"slice{slice_id}_prev_channels"] = float(0 if slice_id == 0 else prev.shape[1])


@torch.inference_mode()
def feature_row(net: model, x: torch.Tensor, image: str, dataset: str, force_ind: int) -> dict[str, Any]:
    y = net.g_a(x)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))

    row: dict[str, Any] = {
        "dataset": dataset,
        "image": image,
        "force_ind": force_ind,
    }
    row.update(tensor_stats(z_hat, "z_hat"))
    row.update(index_stats(z_inds, int(N_E[-1]), "z_index"))

    b, _, h2, w2 = support_buf.shape
    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    for i in range(4):
        mean, scale = net._mean_scale(support_buf, i)
        add_slice_context_features(row, support_buf=support_buf, mean=mean, scale=scale, slice_id=i)

        # Use the original EF-LIC reconstruction path to make previous support
        # available for later slices. These indices are not recorded as
        # controller features because they are not predecision context.
        if i < 3:
            net._qt_select(y, i, y_slice)
            y_norm = (y_slice - mean) / scale
            _, y_hat_i = net.quantizes[force_ind][i].encode_decode(y_norm)
            y_hat_i = y_hat_i * scale + mean
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--dataset-label", required=True)
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0, help="0 means all images after start-index")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.image_dir)
    if args.max_images > 0:
        images = images[args.start_index : args.start_index + args.max_images]
    else:
        images = images[args.start_index :]
    if not images:
        raise SystemExit(f"no images selected from {args.image_dir}")

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    rows: list[dict[str, Any]] = []
    for path in images:
        frame = load_image(path, device)
        _, _, h, w = frame.shape
        padded = replicate_pad(frame, h, w)
        row = feature_row(net, padded, path.name, args.dataset_label, args.force_ind)
        row["height"] = h
        row["width"] = w
        row["pixels"] = h * w
        row["nonfinite_features"] = int(
            any(isinstance(v, float) and not math.isfinite(v) for v in row.values())
        )
        rows.append(row)
        print(f"features dataset={args.dataset_label} image={path.name} nonfinite={row['nonfinite_features']}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "E233 EF-LIC decoder-safe branch features",
        "dataset": args.dataset_label,
        "image_dir": str(args.image_dir),
        "images": len(rows),
        "force_ind": args.force_ind,
        "device": str(device),
        "nonfinite_rows": int(sum(int(r["nonfinite_features"]) for r in rows)),
        "feature_columns": len(fieldnames),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    with md_path.open("w") as fobj:
        fobj.write("# E233 EF-LIC Decoder-Safe Branch Features\n\n")
        fobj.write("This file records predecision context features for future branch-controller diagnostics.\n\n")
        for key, value in summary.items():
            fobj.write(f"- `{key}`: `{value}`\n")
        fobj.write("\nFeature families include z statistics, z-index statistics, and per-slice mean/scale/support/previous-support map summaries.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
