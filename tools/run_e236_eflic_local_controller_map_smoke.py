#!/usr/bin/env python3
"""EF-LIC decoder-safe local HCG controller-map smoke.

E234 exposed a compact branch vocabulary, and E235 showed that image-level
post-hoc selectors are not robust enough. This script moves the next candidate
one step deeper into the codec path: each policy builds a local alpha map from
decoder-reproducible support/mean/scale context after EF-LIC `_mean_scale`.

The policies here are still hand-coded diagnostics, not the final learned head.
Their purpose is to verify the interface and test whether local branch/strength
composition can reduce the all-on false-positive problem while preserving some
of the HCG geometry gain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(EFLIC_DIR))
sys.path.insert(0, str(ROOT / "tools"))

from EF_LIC import G_CH_Y, N_E, model  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import compare_inds, index_stats, mean_psnr, tensor_stats  # noqa: E402
from run_e225_eflic_spatial_alpha_map_smoke import (  # noqa: E402
    build_alpha_map,
    hcg_rvq_decode_map,
    hcg_rvq_encode_decode_map,
)
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


@dataclass(frozen=True)
class LocalControllerPolicy:
    name: str
    family: str
    description: str


POLICIES: dict[str, LocalControllerPolicy] = {
    "zero": LocalControllerPolicy(
        name="zero",
        family="zero",
        description="Safe fallback. Should exactly reproduce the original EF-LIC path.",
    ),
    "constant020": LocalControllerPolicy(
        name="constant020",
        family="constant",
        description="All-position HCG reference at alpha 0.02.",
    ),
    "soft_prev_support010_max": LocalControllerPolicy(
        name="soft_prev_support010_max",
        family="soft_blend",
        description="Max-combine smooth prev/scale and support/scale maps at alpha 0.01.",
    ),
    "soft_prev010_support020_mean": LocalControllerPolicy(
        name="soft_prev010_support020_mean",
        family="soft_blend",
        description="Average weak previous-context and stronger support/scale soft maps.",
    ),
    "sparse_prev_support010_union": LocalControllerPolicy(
        name="sparse_prev_support010_union",
        family="sparse_union",
        description="Union of previous-support and support RMS top-25% maps at alpha 0.01.",
    ),
    "hybrid_prev005_support010": LocalControllerPolicy(
        name="hybrid_prev005_support010",
        family="hybrid",
        description="Conservative previous-support sparse map plus support/scale soft map.",
    ),
    "guarded_support020_top50": LocalControllerPolicy(
        name="guarded_support020_top50",
        family="guarded_support",
        description="Strong support/scale soft map gated to top 50% support/scale positions.",
    ),
    "guarded_constant020_support25": LocalControllerPolicy(
        name="guarded_constant020_support25",
        family="guarded_constant",
        description="All-on alpha 0.02 restricted to top 25% support/scale positions.",
    ),
}


def _branch_map(
    *,
    mode: str,
    alpha: float,
    active_frac: float,
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
) -> torch.Tensor:
    return build_alpha_map(
        mode=mode,
        alpha=alpha,
        active_frac=active_frac,
        support_buf=support_buf,
        mean=mean,
        scale=scale,
        slice_id=slice_id,
    )


def build_controller_alpha_map(
    *,
    policy: str,
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
) -> torch.Tensor:
    """Build a decoder-reproducible local alpha map for an E236 policy."""

    if policy == "zero":
        return mean.new_zeros((mean.shape[0], 1, mean.shape[2], mean.shape[3]))
    if policy == "constant020":
        return _branch_map(
            mode="constant",
            alpha=0.02,
            active_frac=1.0,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )

    prev010 = _branch_map(
        mode="prev_over_scale_top_soft",
        alpha=0.01,
        active_frac=1.0,
        support_buf=support_buf,
        mean=mean,
        scale=scale,
        slice_id=slice_id,
    )
    support010 = _branch_map(
        mode="support_over_scale_top_soft",
        alpha=0.01,
        active_frac=1.0,
        support_buf=support_buf,
        mean=mean,
        scale=scale,
        slice_id=slice_id,
    )
    support020 = _branch_map(
        mode="support_over_scale_top_soft",
        alpha=0.02,
        active_frac=1.0,
        support_buf=support_buf,
        mean=mean,
        scale=scale,
        slice_id=slice_id,
    )

    if policy == "soft_prev_support010_max":
        return torch.maximum(prev010, support010)
    if policy == "soft_prev010_support020_mean":
        return 0.5 * (prev010 + support020)
    if policy == "sparse_prev_support010_union":
        prev_sparse = _branch_map(
            mode="prev_rms_top",
            alpha=0.01,
            active_frac=0.25,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )
        support_sparse = _branch_map(
            mode="support_rms_top",
            alpha=0.01,
            active_frac=0.25,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )
        return torch.maximum(prev_sparse, support_sparse)
    if policy == "hybrid_prev005_support010":
        prev_sparse = _branch_map(
            mode="prev_rms_top",
            alpha=0.005,
            active_frac=0.25,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )
        return torch.maximum(prev_sparse, support010)
    if policy == "guarded_support020_top50":
        guard = _branch_map(
            mode="support_over_scale_top",
            alpha=1.0,
            active_frac=0.50,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        ).clamp(0.0, 1.0)
        return support020 * guard
    if policy == "guarded_constant020_support25":
        guard = _branch_map(
            mode="support_over_scale_top",
            alpha=1.0,
            active_frac=0.25,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        ).clamp(0.0, 1.0)
        return guard * 0.02

    raise ValueError(f"unknown local controller policy {policy!r}")


def alpha_stats(alpha_map: torch.Tensor, prefix: str) -> dict[str, float]:
    alpha = alpha_map.detach().float()
    return {
        f"{prefix}_alpha_mean": float(alpha.mean().item()),
        f"{prefix}_alpha_std": float(alpha.std(unbiased=False).item()),
        f"{prefix}_alpha_max": float(alpha.max().item()),
        f"{prefix}_alpha_active_frac": float((alpha > 0).float().mean().item()),
    }


@torch.inference_mode()
def active_compress_forward_controller_map(
    net: model,
    x: torch.Tensor,
    *,
    force_ind: int,
    policy: str,
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

    for slice_id in range(4):
        mean, scale = net._mean_scale(support_buf, slice_id)
        net._qt_select(y, slice_id, y_slice)
        y_norm = (y_slice - mean) / scale
        alpha_map = build_controller_alpha_map(
            policy=policy,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )
        context = scale if direction_source == "logscale" else mean
        ind_i, y_hat_norm_i, slice_stats = hcg_rvq_encode_decode_map(
            net.quantizes[force_ind][slice_id],
            y_norm,
            context,
            alpha_map,
            direction_source,
        )
        y_hat_i = y_hat_norm_i * scale + mean
        net._qt_put_(y_hat, y_hat_i, slice_id)
        if slice_id < 3:
            support_buf[:, (4 + slice_id) * G_CH_Y : (5 + slice_id) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)
        stats.update(alpha_stats(alpha_map, f"slice{slice_id}"))
        for key, value in slice_stats.items():
            stats[f"slice{slice_id}_{key}"] = value

    for metric in [
        "alpha_mean",
        "alpha_std",
        "alpha_max",
        "alpha_active_frac",
        "avg_index_entropy",
        "avg_index_used_frac",
        "avg_geometry_delta_rms",
        "avg_residual_error_rms",
    ]:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, net.g_s(y_hat), stats


@torch.inference_mode()
def active_decompress_controller_map(
    net: model,
    inds: dict[str, Any],
    *,
    force_ind: int,
    policy: str,
    direction_source: str,
) -> torch.Tensor:
    z_hat = net.quantizes[force_ind][-1].decoding(inds["z_inds"])
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    for slice_id, ind_i in enumerate(inds["y_inds"]):
        mean, scale = net._mean_scale(support_buf, slice_id)
        alpha_map = build_controller_alpha_map(
            policy=policy,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        )
        context = scale if direction_source == "logscale" else mean
        y_i = hcg_rvq_decode_map(
            net.quantizes[force_ind][slice_id],
            ind_i,
            context,
            alpha_map,
            direction_source,
        )
        y_i = y_i * scale + mean
        net._qt_put_(y_hat, y_i, slice_id)
        if slice_id < 3:
            support_buf[:, (4 + slice_id) * G_CH_Y : (5 + slice_id) * G_CH_Y].copy_(y_i)
    return net.g_s(y_hat)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--policies", nargs="+", default=list(POLICIES))
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--with-perceptual", action="store_true")
    return p.parse_args()


def validate_policies(names: list[str]) -> list[LocalControllerPolicy]:
    missing = [name for name in names if name not in POLICIES]
    if missing:
        raise SystemExit(f"unknown policy(s): {missing}. Available: {sorted(POLICIES)}")
    return [POLICIES[name] for name in names]


def finite_row(row: dict[str, Any], stats: dict[str, float], *tensors: torch.Tensor) -> int:
    if any(not torch.isfinite(t).all().item() for t in tensors):
        return 1
    if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
        return 1
    if any(isinstance(v, float) and not math.isfinite(v) for v in stats.values()):
        return 1
    return 0


def summarize(rows: list[dict[str, Any]], policies: list[LocalControllerPolicy]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy.name]
        if not subset:
            continue
        item: dict[str, Any] = {
            "policy": policy.name,
            "family": policy.family,
            "images": len(subset),
            "bpp": float(np.mean([row["bpp"] for row in subset])),
            "delta_bpp": float(np.mean([row["delta_bpp"] for row in subset])),
            "base_psnr": mean_psnr(subset, "base_psnr"),
            "active_psnr": mean_psnr(subset, "active_psnr"),
            "delta_psnr": float(np.mean([row["delta_psnr"] for row in subset])),
            "max_decode_diff": float(max(row["max_decode_diff"] for row in subset)),
            "nonfinite_rows": int(sum(row["nonfinite"] for row in subset)),
            "payload_len_equal_frac": float(np.mean([row["payload_len_equal"] for row in subset])),
            "payload_equal_frac": float(np.mean([row["payload_equal"] for row in subset])),
            "y_mismatch_frac": float(np.sum([row["y_mismatch"] for row in subset]) / max(1, np.sum([row["y_total"] for row in subset]))),
            "alpha_mean": float(np.mean([row.get("y_alpha_mean", 0.0) for row in subset])),
            "alpha_std": float(np.mean([row.get("y_alpha_std", 0.0) for row in subset])),
            "alpha_max": float(np.mean([row.get("y_alpha_max", 0.0) for row in subset])),
            "alpha_active_frac": float(np.mean([row.get("y_alpha_active_frac", 0.0) for row in subset])),
            "geometry_delta_rms": float(np.mean([row.get("y_avg_geometry_delta_rms", 0.0) for row in subset])),
            "index_entropy": float(np.mean([row.get("y_avg_index_entropy", 0.0) for row in subset])),
            "index_used_frac": float(np.mean([row.get("y_avg_index_used_frac", 0.0) for row in subset])),
        }
        if "delta_dists" in subset[0]:
            item.update(
                {
                    "delta_dists": float(np.mean([row["delta_dists"] for row in subset])),
                    "delta_lpips": float(np.mean([row["delta_lpips"] for row in subset])),
                    "score_dists_3lpips": float(np.mean([row["score_dists_3lpips"] for row in subset])),
                    "score_win_frac": float(np.mean([row["score_dists_3lpips"] < 0.0 for row in subset])),
                }
            )
        summary.append(item)
    return summary


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    policies: list[LocalControllerPolicy],
    images: list[Path],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "E236 EF-LIC local controller-map smoke",
        "purpose": "Test decoder-safe local branch/strength composition inside the EF-LIC codec path.",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.image_dir),
        "images": [path.name for path in images],
        "device": str(args.device),
        "force_ind": args.force_ind,
        "direction_source": args.direction_source,
        "with_perceptual": bool(args.with_perceptual),
        "policies": [asdict(policy) for policy in policies],
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    keys = [
        "policy",
        "family",
        "images",
        "delta_bpp",
        "delta_psnr",
        "max_decode_diff",
        "nonfinite_rows",
        "y_mismatch_frac",
        "alpha_mean",
        "alpha_std",
        "alpha_max",
        "alpha_active_frac",
        "geometry_delta_rms",
        "index_entropy",
    ]
    if summary and "delta_dists" in summary[0]:
        keys.extend(["delta_dists", "delta_lpips", "score_dists_3lpips", "score_win_frac"])

    with md_path.open("w") as fobj:
        fobj.write("# E236 EF-LIC Local Controller-Map Smoke\n\n")
        fobj.write(
            "This diagnostic composes local HCG alpha maps from decoder-reproducible EF-LIC context "
            "after `_mean_scale(support_buf, i)`. It is a bridge toward a trained codec-path local head.\n\n"
        )
        fobj.write(f"- Dataset: `{args.image_dir}`\n")
        fobj.write(f"- Images: `{len(images)}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Force index: `{args.force_ind}`\n")
        fobj.write(f"- Direction source: `{args.direction_source}`\n")
        fobj.write(f"- Perceptual metrics: `{bool(args.with_perceptual)}`\n\n")
        fobj.write("| " + " | ".join(keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for item in summary:
            vals = []
            for key in keys:
                val = item.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:+.8f}" if key.startswith("delta") or key.startswith("score") else f"{val:.8f}")
                else:
                    vals.append(str(val))
            fobj.write("| " + " | ".join(vals) + " |\n")
        fobj.write("\nPolicy vocabulary:\n\n")
        for policy in policies:
            fobj.write(f"- `{policy.name}`: family=`{policy.family}`. {policy.description}\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- `zero` is the contract check: unchanged bpp and exact decode reproduction are required.\n")
        fobj.write("- `constant020` is the aggressive E234 reference, included to detect whether guards reduce false positives.\n")
        fobj.write("- Other policies combine support/previous-context local maps without transmitting side bits.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    policies = validate_policies(args.policies)
    images = list_images(args.image_dir)
    images = images[args.start_index : args.start_index + args.max_images]
    if not images:
        raise SystemExit(f"no images selected from {args.image_dir}")

    device = torch.device(args.device)
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
    for policy in policies:
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            orig_inds = net.compress(padded.clone(), force_ind=args.force_ind)
            orig_payload, _, _ = pack_inds(net, orig_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=args.force_ind)[:, :, :h, :w]

            active_inds, active_x_hat_forward, stats = active_compress_forward_controller_map(
                net,
                padded.clone(),
                force_ind=args.force_ind,
                policy=policy.name,
                direction_source=args.direction_source,
            )
            active_payload, _, _ = pack_inds(net, active_inds)
            active_x_hat_dec = active_decompress_controller_map(
                net,
                active_inds,
                force_ind=args.force_ind,
                policy=policy.name,
                direction_source=args.direction_source,
            )[:, :, :h, :w]
            active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
            active_x_hat = active_x_hat_dec
            diff = (active_x_hat_fwd - active_x_hat_dec).abs()
            mismatch = compare_inds(orig_inds, active_inds)

            row: dict[str, Any] = {
                "policy": policy.name,
                "family": policy.family,
                "image": path.name,
                "force_ind": args.force_ind,
                "direction_source": args.direction_source,
                "bpp": len(active_payload) * 8.0 / float(h * w),
                "delta_bpp": (len(active_payload) - len(orig_payload)) * 8.0 / float(h * w),
                "payload_len_equal": int(len(active_payload) == len(orig_payload)),
                "payload_equal": int(active_payload == orig_payload),
                "base_psnr": psnr_from_mse(mse01(orig_x_hat, frame)),
                "active_psnr": psnr_from_mse(mse01(active_x_hat, frame)),
                "max_decode_diff": float(diff.max().item()),
                "mean_decode_diff": float(diff.mean().item()),
            }
            row["delta_psnr"] = row["active_psnr"] - row["base_psnr"]
            if lpips_fn is not None and dists_fn is not None:
                row["base_lpips"] = float(lpips_fn(orig_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                row["active_lpips"] = float(lpips_fn(active_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item())
                row["delta_lpips"] = row["active_lpips"] - row["base_lpips"]
                row["base_dists"] = float(
                    dists_fn(
                        ((orig_x_hat + 1.0) * 0.5).clamp(0, 1),
                        ((frame + 1.0) * 0.5).clamp(0, 1),
                        require_grad=False,
                    )
                    .detach()
                    .mean()
                    .item()
                )
                row["active_dists"] = float(
                    dists_fn(
                        ((active_x_hat + 1.0) * 0.5).clamp(0, 1),
                        ((frame + 1.0) * 0.5).clamp(0, 1),
                        require_grad=False,
                    )
                    .detach()
                    .mean()
                    .item()
                )
                row["delta_dists"] = row["active_dists"] - row["base_dists"]
                row["score_dists_3lpips"] = row["delta_dists"] + 3.0 * row["delta_lpips"]
            row.update(mismatch)
            row.update(stats)
            row["nonfinite"] = finite_row(row, stats, active_x_hat, active_x_hat_fwd)
            rows.append(row)

            metric_text = f"dPSNR={row['delta_psnr']:+.4f}"
            if "delta_dists" in row:
                metric_text += (
                    f" dDISTS={row['delta_dists']:+.5f}"
                    f" dLPIPS={row['delta_lpips']:+.5f}"
                    f" score={row['score_dists_3lpips']:+.5f}"
                )
            print(
                f"policy={policy.name} family={policy.family} image={path.name} "
                f"{metric_text} dbpp={row['delta_bpp']:+.6f} "
                f"decmax={row['max_decode_diff']:.2e} nonfinite={row['nonfinite']}"
            )

    summary = summarize(rows, policies)
    write_outputs(args=args, rows=rows, summary=summary, policies=policies, images=images)


if __name__ == "__main__":
    main()
