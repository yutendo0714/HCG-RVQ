#!/usr/bin/env python3
"""EF-LIC decoder-safe HCG branch-controller scaffold smoke.

E232/E233 show that useful HCG geometry states form a small branch vocabulary
but that post-hoc image-level selectors are not robust enough. This script is
the next implementation bridge: it evaluates that branch vocabulary through a
single controller-like preset interface inside the codec path.

This is not a trained controller yet. It verifies the contract required by the
paper method:

* every branch decision is derived from decoder-reproducible context,
* no side bits are added,
* encoder forward and decoder reconstruction match exactly,
* the preset vocabulary covers zero fallback, strong all-on, sparse local, and
  smooth local support/previous-context geometry.
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

from EF_LIC import model  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import compare_inds, mean_psnr  # noqa: E402
from run_e225_eflic_spatial_alpha_map_smoke import active_compress_forward_map, active_decompress_map  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


@dataclass(frozen=True)
class ControllerPreset:
    name: str
    family: str
    mode: str
    alpha: float
    active_frac: float
    description: str


PRESETS: dict[str, ControllerPreset] = {
    "zero": ControllerPreset(
        name="zero",
        family="zero",
        mode="zero",
        alpha=0.0,
        active_frac=0.0,
        description="Safe fallback. Should reproduce the EF-LIC baseline exactly.",
    ),
    "constant020": ControllerPreset(
        name="constant020",
        family="constant",
        mode="constant",
        alpha=0.02,
        active_frac=1.0,
        description="Aggressive all-position projected geometry; useful on Kodak-like cases.",
    ),
    "sparse_prev005": ControllerPreset(
        name="sparse_prev005",
        family="sparse_prev",
        mode="prev_rms_top",
        alpha=0.005,
        active_frac=0.25,
        description="Very conservative previous-support local geometry; safest CLIC fixed branch.",
    ),
    "sparse_prev010": ControllerPreset(
        name="sparse_prev010",
        family="sparse_prev",
        mode="prev_rms_top",
        alpha=0.01,
        active_frac=0.25,
        description="Weak previous-support local geometry; best pooled fixed row in E232.",
    ),
    "sparse_support010": ControllerPreset(
        name="sparse_support010",
        family="sparse_support",
        mode="support_rms_top",
        alpha=0.01,
        active_frac=0.25,
        description="Sparse local geometry from known support-buffer energy.",
    ),
    "soft_prev010": ControllerPreset(
        name="soft_prev010",
        family="soft_prev",
        mode="prev_over_scale_top_soft",
        alpha=0.01,
        active_frac=1.0,
        description="Smooth previous-context geometry normalized by predicted scale.",
    ),
    "soft_support020": ControllerPreset(
        name="soft_support020",
        family="soft_support",
        mode="support_over_scale_top_soft",
        alpha=0.02,
        active_frac=1.0,
        description="Smooth support/scale-conditioned geometry; high-value oracle branch.",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--presets", nargs="+", default=list(PRESETS))
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--with-perceptual", action="store_true")
    return p.parse_args()


def validate_presets(names: list[str]) -> list[ControllerPreset]:
    missing = [name for name in names if name not in PRESETS]
    if missing:
        raise SystemExit(f"unknown preset(s): {missing}. Available: {sorted(PRESETS)}")
    return [PRESETS[name] for name in names]


def finite_row(row: dict[str, Any], stats: dict[str, float], *tensors: torch.Tensor) -> int:
    if any(not torch.isfinite(t).all().item() for t in tensors):
        return 1
    if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
        return 1
    if any(isinstance(v, float) and not math.isfinite(v) for v in stats.values()):
        return 1
    return 0


def summarize(rows: list[dict[str, Any]], presets: list[ControllerPreset]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for preset in presets:
        subset = [row for row in rows if row["preset"] == preset.name]
        if not subset:
            continue
        item: dict[str, Any] = {
            "preset": preset.name,
            "family": preset.family,
            "mode": preset.mode,
            "alpha": preset.alpha,
            "active_frac_target": preset.active_frac,
            "images": len(subset),
            "bpp": float(np.mean([r["bpp"] for r in subset])),
            "delta_bpp": float(np.mean([r["delta_bpp"] for r in subset])),
            "base_psnr": mean_psnr(subset, "base_psnr"),
            "active_psnr": mean_psnr(subset, "active_psnr"),
            "delta_psnr": float(np.mean([r["delta_psnr"] for r in subset])),
            "max_decode_diff": float(max(r["max_decode_diff"] for r in subset)),
            "nonfinite_rows": int(sum(r["nonfinite"] for r in subset)),
            "payload_len_equal_frac": float(np.mean([r["payload_len_equal"] for r in subset])),
            "payload_equal_frac": float(np.mean([r["payload_equal"] for r in subset])),
            "y_mismatch_frac": float(np.sum([r["y_mismatch"] for r in subset]) / max(1, np.sum([r["y_total"] for r in subset]))),
            "alpha_mean": float(np.mean([r.get("y_alpha_mean", 0.0) for r in subset])),
            "alpha_active_frac": float(np.mean([r.get("y_alpha_active_frac", 0.0) for r in subset])),
            "geometry_delta_rms": float(np.mean([r.get("y_avg_geometry_delta_rms", 0.0) for r in subset])),
            "index_entropy": float(np.mean([r.get("y_avg_index_entropy", 0.0) for r in subset])),
            "index_used_frac": float(np.mean([r.get("y_avg_index_used_frac", 0.0) for r in subset])),
        }
        if "delta_dists" in subset[0]:
            item.update(
                {
                    "delta_dists": float(np.mean([r["delta_dists"] for r in subset])),
                    "delta_lpips": float(np.mean([r["delta_lpips"] for r in subset])),
                    "score_dists_3lpips": float(np.mean([r["delta_dists"] + 3.0 * r["delta_lpips"] for r in subset])),
                }
            )
        summary.append(item)
    return summary


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    presets: list[ControllerPreset],
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
        "experiment": "E234 EF-LIC branch-controller scaffold smoke",
        "purpose": "Validate a decoder-safe HCG branch vocabulary through one codec-path controller interface.",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.image_dir),
        "images": [path.name for path in images],
        "device": str(args.device),
        "force_ind": args.force_ind,
        "direction_source": args.direction_source,
        "with_perceptual": bool(args.with_perceptual),
        "presets": [asdict(preset) for preset in presets],
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    keys = [
        "preset",
        "family",
        "images",
        "delta_bpp",
        "delta_psnr",
        "max_decode_diff",
        "nonfinite_rows",
        "y_mismatch_frac",
        "alpha_mean",
        "alpha_active_frac",
        "geometry_delta_rms",
        "index_entropy",
    ]
    if summary and "delta_dists" in summary[0]:
        keys.extend(["delta_dists", "delta_lpips", "score_dists_3lpips"])

    with md_path.open("w") as fobj:
        fobj.write("# E234 EF-LIC Branch-Controller Scaffold Smoke\n\n")
        fobj.write(
            "This run treats the E232 HCG branch library as a decoder-safe controller vocabulary. "
            "It is an implementation bridge toward a trained local controller, not a final performance claim.\n\n"
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
        fobj.write("\nPreset vocabulary:\n\n")
        for preset in presets:
            fobj.write(
                f"- `{preset.name}`: family=`{preset.family}`, mode=`{preset.mode}`, "
                f"alpha=`{preset.alpha}`, active_frac=`{preset.active_frac}`. {preset.description}\n"
            )
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- `zero` is the contract check: bpp, payload length, and reconstruction should match the EF-LIC baseline.\n")
        fobj.write("- Nonzero presets are deterministic no-sidebit HCG states computed inside the same EF-LIC sequential support-buffer loop.\n")
        fobj.write("- A trained E235-style controller can now output this same vocabulary, replacing the fixed preset selector.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    presets = validate_presets(args.presets)
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
    for preset in presets:
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            orig_inds = net.compress(padded.clone(), force_ind=args.force_ind)
            orig_payload, _, _ = pack_inds(net, orig_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=args.force_ind)[:, :, :h, :w]

            active_inds, active_x_hat_forward, stats = active_compress_forward_map(
                net,
                padded.clone(),
                force_ind=args.force_ind,
                mode=preset.mode,
                alpha=preset.alpha,
                active_frac=preset.active_frac,
                direction_source=args.direction_source,
            )
            active_payload, _, _ = pack_inds(net, active_inds)
            active_x_hat_dec = active_decompress_map(
                net,
                active_inds,
                force_ind=args.force_ind,
                mode=preset.mode,
                alpha=preset.alpha,
                active_frac=preset.active_frac,
                direction_source=args.direction_source,
            )[:, :, :h, :w]
            active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
            active_x_hat = active_x_hat_dec
            diff = (active_x_hat_fwd - active_x_hat_dec).abs()
            mismatch = compare_inds(orig_inds, active_inds)

            row: dict[str, Any] = {
                "preset": preset.name,
                "family": preset.family,
                "mode": preset.mode,
                "image": path.name,
                "force_ind": args.force_ind,
                "direction_source": args.direction_source,
                "alpha": preset.alpha,
                "active_frac_target": preset.active_frac,
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
                f"preset={preset.name} family={preset.family} image={path.name} "
                f"{metric_text} dbpp={row['delta_bpp']:+.6f} "
                f"decmax={row['max_decode_diff']:.2e} nonfinite={row['nonfinite']}"
            )

    summary = summarize(rows, presets)
    write_outputs(args=args, rows=rows, summary=summary, presets=presets, images=images)


if __name__ == "__main__":
    main()
