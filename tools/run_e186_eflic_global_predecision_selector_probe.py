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
TOOLS = ROOT / "tools"
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(EFLIC_DIR))

from run_e160_eflic_projected_hcg_smoke import (  # noqa: E402
    active_compress_forward,
    active_decompress,
    mean_psnr,
    tensor_stats,
)
from EF_LIC import model  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kodak-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e186_eflic_force0_global_predecision_selector",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--selector-feature", default="slice0_mean_abs_mean")
    p.add_argument("--selector-op", choices=["<=", ">="], default="<=")
    p.add_argument("--selector-threshold", type=float, default=0.455596)
    return p.parse_args()


@torch.inference_mode()
def predecision_stats_from_padded(net: model, x: torch.Tensor, force_ind: int) -> tuple[torch.Tensor, dict[str, float]]:
    y = net.g_a(x)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))
    mean0, scale0 = net._mean_scale(support_buf, 0)
    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(tensor_stats(mean0, "slice0_mean"))
    stats.update(tensor_stats(scale0, "slice0_scale"))
    return z_inds, stats


@torch.inference_mode()
def predecision_stats_from_z_inds(net: model, z_inds: torch.Tensor, force_ind: int) -> dict[str, float]:
    z_hat = net.quantizes[force_ind][-1].decoding(z_inds)
    support_buf = net._support_buf(net.h_s(z_hat))
    mean0, scale0 = net._mean_scale(support_buf, 0)
    stats: dict[str, float] = {}
    stats.update(tensor_stats(z_hat, "z_hat"))
    stats.update(tensor_stats(mean0, "slice0_mean"))
    stats.update(tensor_stats(scale0, "slice0_scale"))
    return stats


def decide(stats: dict[str, float], feature: str, op: str, threshold: float) -> bool:
    value = float(stats[feature])
    if op == "<=":
        return value <= threshold
    return value >= threshold


def dists_value(fn: Any, x_hat: torch.Tensor, frame: torch.Tensor) -> float:
    return float(
        fn(
            ((x_hat + 1.0) * 0.5).clamp(0, 1),
            ((frame + 1.0) * 0.5).clamp(0, 1),
            require_grad=False,
        )
        .detach()
        .mean()
        .item()
    )


def summarize(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    n = max(1, len(rows))
    return {
        "branch": branch,
        "images": len(rows),
        "branch_share": float(np.mean([float(r["use_active"]) for r in rows])) if branch == "selected" else float(branch == "active"),
        "bpp": float(np.mean([r[f"{branch}_bpp"] for r in rows])),
        "psnr": mean_psnr(rows, f"{branch}_psnr"),
        "lpips": float(np.mean([r[f"{branch}_lpips"] for r in rows])),
        "dists": float(np.mean([r[f"{branch}_dists"] for r in rows])),
        "delta_psnr_vs_base": float(np.mean([r[f"{branch}_psnr"] - r["base_psnr"] for r in rows])),
        "delta_lpips_vs_base": float(np.mean([r[f"{branch}_lpips"] - r["base_lpips"] for r in rows])),
        "delta_dists_vs_base": float(np.mean([r[f"{branch}_dists"] - r["base_dists"] for r in rows])),
        "dists_wins_vs_base": int(sum(r[f"{branch}_dists"] < r["base_dists"] for r in rows)),
        "lpips_wins_vs_base": int(sum(r[f"{branch}_lpips"] < r["base_lpips"] for r in rows)),
        "psnr_wins_vs_base": int(sum(r[f"{branch}_psnr"] > r["base_psnr"] for r in rows)),
    }


def main() -> None:
    args = parse_args()
    if abs(1.0 - 2.0 * args.alpha) < 1e-6:
        raise ValueError("alpha=0.5 is singular")

    device = torch.device(args.device)
    images = list_images(args.kodak_dir)

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    rows: list[dict[str, Any]] = []
    for path in images:
        frame = load_image(path, device)
        _, _, h, w = frame.shape
        padded = replicate_pad(frame, h, w)

        enc_z_inds, enc_pre_stats = predecision_stats_from_padded(net, padded.clone(), args.force_ind)
        enc_decision = decide(enc_pre_stats, args.selector_feature, args.selector_op, args.selector_threshold)
        dec_pre_stats = predecision_stats_from_z_inds(net, enc_z_inds, args.force_ind)
        dec_decision = decide(dec_pre_stats, args.selector_feature, args.selector_op, args.selector_threshold)

        base_inds = net.compress(padded.clone(), force_ind=args.force_ind)
        base_payload, _, _ = pack_inds(net, base_inds)
        base_x_hat = net.decompress(base_inds, force_ind=args.force_ind)[:, :, :h, :w]

        active_inds, active_x_hat_forward, active_stats = active_compress_forward(
            net,
            padded.clone(),
            force_ind=args.force_ind,
            alpha=args.alpha,
            direction_source=args.direction_source,
        )
        active_payload, _, _ = pack_inds(net, active_inds)
        active_x_hat_dec = active_decompress(
            net,
            active_inds,
            force_ind=args.force_ind,
            alpha=args.alpha,
            direction_source=args.direction_source,
        )[:, :, :h, :w]
        active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
        decode_diff = (active_x_hat_fwd - active_x_hat_dec).abs()

        selected_x_hat = active_x_hat_dec if enc_decision else base_x_hat
        selected_payload = active_payload if enc_decision else base_payload

        row: dict[str, Any] = {
            "image": path.name,
            "selector_feature": args.selector_feature,
            "selector_op": args.selector_op,
            "selector_threshold": args.selector_threshold,
            "selector_value_encoder": float(enc_pre_stats[args.selector_feature]),
            "selector_value_decoder": float(dec_pre_stats[args.selector_feature]),
            "selector_value_abs_diff": abs(float(enc_pre_stats[args.selector_feature]) - float(dec_pre_stats[args.selector_feature])),
            "use_active": int(enc_decision),
            "decoder_decision_match": int(enc_decision == dec_decision),
            "base_bpp": len(base_payload) * 8.0 / float(h * w),
            "active_bpp": len(active_payload) * 8.0 / float(h * w),
            "selected_bpp": len(selected_payload) * 8.0 / float(h * w),
            "base_psnr": psnr_from_mse(mse01(base_x_hat, frame)),
            "active_psnr": psnr_from_mse(mse01(active_x_hat_dec, frame)),
            "selected_psnr": psnr_from_mse(mse01(selected_x_hat, frame)),
            "base_lpips": float(lpips_fn(base_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "active_lpips": float(lpips_fn(active_x_hat_dec.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "selected_lpips": float(lpips_fn(selected_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "base_dists": dists_value(dists_fn, base_x_hat, frame),
            "active_dists": dists_value(dists_fn, active_x_hat_dec, frame),
            "selected_dists": dists_value(dists_fn, selected_x_hat, frame),
            "active_max_decode_diff": float(decode_diff.max().item()),
            "active_mean_decode_diff": float(decode_diff.mean().item()),
            "nonfinite": int(
                (not torch.isfinite(base_x_hat).all().item())
                or (not torch.isfinite(active_x_hat_dec).all().item())
                or (not torch.isfinite(selected_x_hat).all().item())
                or any(isinstance(v, float) and not math.isfinite(v) for v in active_stats.values())
            ),
        }
        row["active_delta_dists"] = row["active_dists"] - row["base_dists"]
        row["selected_delta_dists"] = row["selected_dists"] - row["base_dists"]
        row["active_delta_lpips"] = row["active_lpips"] - row["base_lpips"]
        row["selected_delta_lpips"] = row["selected_lpips"] - row["base_lpips"]
        row["active_delta_psnr"] = row["active_psnr"] - row["base_psnr"]
        row["selected_delta_psnr"] = row["selected_psnr"] - row["base_psnr"]
        rows.append(row)
        print(
            f"{path.name} use_active={row['use_active']} val={row['selector_value_encoder']:.6f} "
            f"dDISTS active={row['active_delta_dists']:+.6f} selected={row['selected_delta_dists']:+.6f} "
            f"match={row['decoder_decision_match']} nonfinite={row['nonfinite']}"
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = [summarize(rows, "base"), summarize(rows, "active"), summarize(rows, "selected")]
    payload = {
        "experiment": "E186 EF-LIC global predecision selector direct probe",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.kodak_dir),
        "device": str(device),
        "force_ind": args.force_ind,
        "alpha": args.alpha,
        "direction_source": args.direction_source,
        "selector_rule": f"{args.selector_feature} {args.selector_op} {args.selector_threshold}",
        "rows": len(rows),
        "summary": summary,
        "interpretation": "Direct deployability probe for a Kodak-fitted scalar selector; not final paper evidence.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# EF-LIC Global Predecision Selector Probe",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Dataset: `{args.kodak_dir}`",
        f"Device: `{device}`",
        f"Rule: `{args.selector_feature} {args.selector_op} {args.selector_threshold}`",
        "",
        "This is a direct deployability probe for the E185 global-predecision rule. The threshold is selected on the same Kodak diagnostic table, so this is not a final paper-quality claim.",
        "",
        "| branch | images | branch share | bpp | PSNR | LPIPS | DISTS | dPSNR | dLPIPS | dDISTS | DISTS wins | LPIPS wins | PSNR wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['branch']} | {s['images']} | {s['branch_share']:.3f} | {s['bpp']:.6f} | "
            f"{s['psnr']:.4f} | {s['lpips']:.5f} | {s['dists']:.5f} | "
            f"{s['delta_psnr_vs_base']:+.4f} | {s['delta_lpips_vs_base']:+.6f} | {s['delta_dists_vs_base']:+.6f} | "
            f"{s['dists_wins_vs_base']}/{s['images']} | {s['lpips_wins_vs_base']}/{s['images']} | {s['psnr_wins_vs_base']}/{s['images']} |"
        )
    lines.extend(
        [
            "",
            "Checks:",
            "",
            f"- Encoder/decoder selector decision matches: `{sum(r['decoder_decision_match'] for r in rows)}/{len(rows)}`",
            f"- Max selector value abs diff: `{max(r['selector_value_abs_diff'] for r in rows):.6e}`",
            f"- Max active decode diff: `{max(r['active_max_decode_diff'] for r in rows):.6e}`",
            f"- Nonfinite rows: `{sum(r['nonfinite'] for r in rows)}`",
            "",
            "Next:",
            "",
            "- Replace Kodak-fitted scalar thresholds with train-split/OpenImages-fitted or learned reliability controllers.",
            "- Keep this as implementation evidence that global predecision selection can be decoder-reproduced without a side bit.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


if __name__ == "__main__":
    main()
