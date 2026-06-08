#!/usr/bin/env python3
"""EF-LIC HCG branch-controller integration smoke.

E292-E294 showed that fixed nonzero EF-LIC HCG branches are codec-valid but not
reliable enough to use as paper-main policies. This script connects the new
decoder-safe `EFLICHCGBranchController` to the actual EF-LIC slice/RVQ loop and
checks the contract needed before training:

* `force_zero` exactly recovers the original EF-LIC path,
* the fallback-biased hard controller also stays exact at initialization,
* the soft controller produces finite, decoder-reproducible alpha maps.

This is a wiring/contract smoke, not a final RD claim.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from EF_LIC import G_CH_Y, N_E, model  # noqa: E402
from hcg_rvq.eflic_local_controller import (  # noqa: E402
    EFLICHCGBranchController,
    EFLICHCGBranchControllerConfig,
    build_local_context_maps,
)
from run_e160_eflic_projected_hcg_smoke import compare_inds, index_stats, mean_psnr, tensor_stats  # noqa: E402
from run_e225_eflic_spatial_alpha_map_smoke import hcg_rvq_decode_map, hcg_rvq_encode_decode_map  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


MODE_TO_GATE: dict[str, tuple[bool, bool]] = {
    "force_zero": (True, True),
    "init_hard": (False, True),
    "init_soft": (False, False),
    "trained_hard": (False, True),
    "trained_soft": (False, False),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e295_eflic_hcg_branch_controller_integration_smoke",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--modes", nargs="+", default=["force_zero", "init_hard", "init_soft"], choices=sorted(MODE_TO_GATE))
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--max-alpha", type=float, default=0.02)
    p.add_argument("--active-threshold", type=float, default=0.5)
    p.add_argument("--max-risk", type=float, default=0.0)
    p.add_argument("--risk-temperature", type=float, default=1.0)
    p.add_argument(
        "--compute-perceptual",
        action="store_true",
        help="Also compute MS-SSIM, LPIPS, and DISTS. PSNR remains a diagnostic codec-health metric.",
    )
    p.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--seed", type=int, default=295)
    p.add_argument("--controller-state", type=Path, default=None)
    p.add_argument(
        "--active-slices",
        default="all",
        help="Comma-separated EF-LIC y-slice ids to allow HCG perturbation on, or all. "
        "Disabled slices use exact zero-alpha fallback. This supports slice-isolation probes.",
    )
    return p.parse_args()


def _finite_stats(stats: dict[str, float]) -> bool:
    return all(math.isfinite(v) for v in stats.values())


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def build_metric_fns(args: argparse.Namespace, device: torch.device) -> dict[str, Any] | None:
    if not args.compute_perceptual:
        return None

    import DISTS_pytorch as dists
    import lpips
    from pytorch_msssim import ms_ssim

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for param in lpips_fn.parameters():
        param.requires_grad_(False)
    dists_fn = dists.DISTS().to(device).eval()
    for param in dists_fn.parameters():
        param.requires_grad_(False)
    return {"lpips": lpips_fn, "dists": dists_fn, "ms_ssim": ms_ssim}


@torch.inference_mode()
def compute_perceptual_metrics(
    metric_fns: dict[str, Any] | None,
    *,
    base_x_hat: torch.Tensor,
    active_x_hat: torch.Tensor,
    frame: torch.Tensor,
) -> dict[str, float]:
    if metric_fns is None:
        return {}

    frame_m11 = frame.clamp(-1.0, 1.0)
    base_m11 = base_x_hat.clamp(-1.0, 1.0)
    active_m11 = active_x_hat.clamp(-1.0, 1.0)
    frame01 = _to_01(frame)
    base01 = _to_01(base_x_hat)
    active01 = _to_01(active_x_hat)

    lpips_fn = metric_fns["lpips"]
    dists_fn = metric_fns["dists"]
    ms_ssim_fn = metric_fns["ms_ssim"]

    base_ms_ssim = float(ms_ssim_fn(base01, frame01, data_range=1.0).item())
    active_ms_ssim = float(ms_ssim_fn(active01, frame01, data_range=1.0).item())
    base_lpips = float(lpips_fn(base_m11, frame_m11).mean().item())
    active_lpips = float(lpips_fn(active_m11, frame_m11).mean().item())
    base_dists = float(dists_fn(base01, frame01, require_grad=False).detach().mean().item())
    active_dists = float(dists_fn(active01, frame01, require_grad=False).detach().mean().item())
    return {
        "base_ms_ssim": base_ms_ssim,
        "active_ms_ssim": active_ms_ssim,
        "delta_ms_ssim": active_ms_ssim - base_ms_ssim,
        "base_lpips": base_lpips,
        "active_lpips": active_lpips,
        "delta_lpips": active_lpips - base_lpips,
        "base_dists": base_dists,
        "active_dists": active_dists,
        "delta_dists": active_dists - base_dists,
    }


def parse_active_slices(value: str) -> set[int] | None:
    value = str(value).strip().lower()
    if value in {"", "all"}:
        return None
    if value in {"none", "off", "zero"}:
        return set()
    out: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0 or idx >= 4:
            raise ValueError(f"active slice id must be in [0, 3], got {idx}")
        out.add(idx)
    if not out:
        return None
    return out


def _apply_slice_isolation(
    alpha_map: torch.Tensor,
    controller_stats: dict[str, float],
    *,
    slice_id: int,
    active_slices: set[int] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    enabled = active_slices is None or slice_id in active_slices
    stats = dict(controller_stats)
    stats["slice_enabled"] = float(enabled)
    if enabled:
        return alpha_map, stats
    zero = torch.zeros_like(alpha_map)
    stats["alpha_mean"] = 0.0
    stats["alpha_max"] = 0.0
    stats["alpha_active_frac"] = 0.0
    return zero, stats


def _controller_alpha(
    controller: EFLICHCGBranchController,
    *,
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
    force_zero: bool,
    hard: bool,
    active_threshold: float,
    max_risk: float,
    risk_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    context_maps = build_local_context_maps(
        support_buf,
        mean,
        scale,
        slice_id,
        group_channels=G_CH_Y,
    )
    decision = controller(
        context_maps,
        hard=hard,
        force_zero=force_zero,
        active_threshold=active_threshold,
        max_risk=max_risk,
        risk_temperature=risk_temperature,
    )
    alpha_map = decision["alpha_map"]
    stats = {
        "alpha_mean": float(alpha_map.mean().item()),
        "alpha_max": float(alpha_map.max().item()),
        "alpha_active_frac": float((alpha_map > 0).float().mean().item()),
        "gate_mean": float(decision["gate"].float().mean().item()),
        "gate_max": float(decision["gate"].float().max().item()),
        "strength_mean": float(decision["strength"].float().mean().item()),
        "local_score_mean": float(decision["local_score"].float().mean().item()),
        "active_logit_mean": float(decision["active_logit"].float().mean().item()),
        "risk_score_mean": float(decision["risk_score"].float().mean().item()),
        "family_zero_prob_mean": float(decision["family_logits"].softmax(dim=1)[:, 0:1].float().mean().item()),
    }
    return alpha_map, stats


@torch.inference_mode()
def controller_compress_forward(
    net: model,
    controller: EFLICHCGBranchController,
    x: torch.Tensor,
    *,
    force_ind: int,
    mode: str,
    direction_source: str,
    active_threshold: float,
    max_risk: float,
    risk_temperature: float,
    active_slices: set[int] | None = None,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    force_zero, hard = MODE_TO_GATE[mode]
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
        alpha_map, controller_stats = _controller_alpha(
            controller,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=i,
            force_zero=force_zero,
            hard=hard,
            active_threshold=active_threshold,
            max_risk=max_risk,
            risk_temperature=risk_temperature,
        )
        alpha_map, controller_stats = _apply_slice_isolation(
            alpha_map, controller_stats, slice_id=i, active_slices=active_slices
        )
        context = scale if direction_source == "logscale" else mean
        ind_i, y_hat_norm_i, slice_stats = hcg_rvq_encode_decode_map(
            net.quantizes[force_ind][i],
            y_norm,
            context,
            alpha_map,
            direction_source,
        )
        y_hat_i = y_hat_norm_i * scale + mean
        net._qt_put_(y_hat, y_hat_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_hat_i)
        y_inds.append(ind_i)
        for key, value in controller_stats.items():
            stats[f"slice{i}_{key}"] = value
        for key, value in slice_stats.items():
            stats[f"slice{i}_{key}"] = value
    aggregate_metrics = [
        "alpha_mean",
        "alpha_max",
        "alpha_active_frac",
        "gate_mean",
        "gate_max",
        "strength_mean",
        "local_score_mean",
        "active_logit_mean",
        "risk_score_mean",
        "family_zero_prob_mean",
        "avg_index_entropy",
        "avg_index_used_frac",
        "avg_geometry_delta_rms",
        "avg_residual_error_rms",
        "slice_enabled",
    ]
    for metric in aggregate_metrics:
        stats[f"y_{metric}"] = float(np.mean([stats[f"slice{i}_{metric}"] for i in range(4)]))
    return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}, net.g_s(y_hat), stats


@torch.inference_mode()
def controller_decompress(
    net: model,
    controller: EFLICHCGBranchController,
    inds: dict[str, Any],
    *,
    force_ind: int,
    mode: str,
    direction_source: str,
    active_threshold: float,
    max_risk: float,
    risk_temperature: float,
    active_slices: set[int] | None = None,
) -> torch.Tensor:
    force_zero, hard = MODE_TO_GATE[mode]
    z_hat = net.quantizes[force_ind][-1].decoding(inds["z_inds"])
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    y_hat = support_buf.new_empty(b, G_CH_Y, h2 << 1, w2 << 1)
    for i, ind_i in enumerate(inds["y_inds"]):
        mean, scale = net._mean_scale(support_buf, i)
        alpha_map, controller_stats = _controller_alpha(
            controller,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=i,
            force_zero=force_zero,
            hard=hard,
            active_threshold=active_threshold,
            max_risk=max_risk,
            risk_temperature=risk_temperature,
        )
        alpha_map, _ = _apply_slice_isolation(
            alpha_map, controller_stats, slice_id=i, active_slices=active_slices
        )
        context = scale if direction_source == "logscale" else mean
        y_i = hcg_rvq_decode_map(net.quantizes[force_ind][i], ind_i, context, alpha_map, direction_source)
        y_i = y_i * scale + mean
        net._qt_put_(y_hat, y_i, i)
        if i < 3:
            support_buf[:, (4 + i) * G_CH_Y : (5 + i) * G_CH_Y].copy_(y_i)
    return net.g_s(y_hat)


def summarize(rows: list[dict[str, Any]], modes: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        item = {
            "mode": mode,
            "images": len(subset),
            "bpp": float(np.mean([r["bpp"] for r in subset])),
            "delta_bpp": float(np.mean([r["delta_bpp"] for r in subset])),
            "base_psnr": mean_psnr(subset, "base_psnr"),
            "active_psnr": mean_psnr(subset, "active_psnr"),
            "delta_psnr": float(np.mean([r["delta_psnr"] for r in subset])),
            "max_decode_diff": float(max(r["max_decode_diff"] for r in subset)),
            "max_baseline_diff": float(max(r["max_baseline_diff"] for r in subset)),
            "nonfinite_rows": int(sum(r["nonfinite"] for r in subset)),
            "payload_len_equal_frac": float(np.mean([r["payload_len_equal"] for r in subset])),
            "payload_equal_frac": float(np.mean([r["payload_equal"] for r in subset])),
            "y_mismatch_frac": float(np.sum([r["y_mismatch"] for r in subset]) / max(1, np.sum([r["y_total"] for r in subset]))),
            "alpha_mean": float(np.mean([r["y_alpha_mean"] for r in subset])),
            "alpha_max": float(max(r["y_alpha_max"] for r in subset)),
            "gate_mean": float(np.mean([r["y_gate_mean"] for r in subset])),
            "gate_max": float(max(r["y_gate_max"] for r in subset)),
            "family_zero_prob_mean": float(np.mean([r["y_family_zero_prob_mean"] for r in subset])),
            "geometry_delta_rms": float(np.mean([r["y_avg_geometry_delta_rms"] for r in subset])),
            "index_entropy": float(np.mean([r["y_avg_index_entropy"] for r in subset])),
            "slice_enabled_frac": float(np.mean([r.get("y_slice_enabled", 1.0) for r in subset])),
        }
        if any("delta_lpips" in row for row in subset):
            for key in (
                "base_ms_ssim",
                "active_ms_ssim",
                "delta_ms_ssim",
                "base_lpips",
                "active_lpips",
                "delta_lpips",
                "base_dists",
                "active_dists",
                "delta_dists",
            ):
                item[key] = float(np.mean([r[key] for r in subset]))
        summary.append(item)
    return summary


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
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
        "experiment": "E295 EF-LIC HCG branch-controller integration smoke",
        "purpose": "Validate decoder-safe branch-controller wiring in the actual EF-LIC RVQ loop.",
        "checkpoint": str(args.ckpt_path),
        "dataset": str(args.image_dir),
        "images": [path.name for path in images],
        "device": str(args.device),
        "force_ind": args.force_ind,
        "direction_source": args.direction_source,
        "active_slices": args.active_slices,
        "controller_config": {
            "max_alpha": args.max_alpha,
            "active_threshold": args.active_threshold,
            "max_risk": args.max_risk,
            "risk_temperature": args.risk_temperature,
            "seed": args.seed,
            "controller_state": str(args.controller_state) if args.controller_state is not None else None,
        },
        "metric_protocol": {
            "psnr_role": "diagnostic codec-health metric; not the primary generative/perceptual claim",
            "compute_perceptual": bool(args.compute_perceptual),
            "lpips_net": args.lpips_net if args.compute_perceptual else None,
            "perceptual_metrics": ["MS-SSIM", "LPIPS", "DISTS"] if args.compute_perceptual else [],
            "metric_direction": {
                "delta_ms_ssim": "positive is better",
                "delta_lpips": "negative is better",
                "delta_dists": "negative is better",
            },
        },
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    keys = [
        "mode",
        "images",
        "delta_bpp",
        "delta_psnr",
        "max_decode_diff",
        "max_baseline_diff",
        "nonfinite_rows",
        "payload_equal_frac",
        "y_mismatch_frac",
        "alpha_mean",
        "alpha_max",
        "gate_mean",
        "gate_max",
        "family_zero_prob_mean",
        "geometry_delta_rms",
        "index_entropy",
        "slice_enabled_frac",
    ]
    if any("delta_lpips" in item for item in summary):
        insert_at = keys.index("delta_psnr") + 1
        keys[insert_at:insert_at] = [
            "delta_ms_ssim",
            "delta_lpips",
            "delta_dists",
            "base_ms_ssim",
            "active_ms_ssim",
            "base_lpips",
            "active_lpips",
            "base_dists",
            "active_dists",
        ]

    with md_path.open("w") as fobj:
        fobj.write("# E295 EF-LIC HCG Branch-Controller Integration Smoke\n\n")
        fobj.write(
            "This run inserts the decoder-safe HCG branch controller into the EF-LIC "
            "slice/RVQ loop. It validates wiring and fallback contracts before any "
            "paper-facing controller training.\n\n"
        )
        fobj.write(
            "PSNR is kept as a diagnostic codec-health metric. When "
            "`--compute-perceptual` is enabled, MS-SSIM, LPIPS, and DISTS are "
            "reported for the generative/perceptual compression claim.\n\n"
        )
        fobj.write(f"- Dataset: `{args.image_dir}`\n")
        fobj.write(f"- Images: `{len(images)}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Force index: `{args.force_ind}`\n")
        fobj.write(f"- Direction source: `{args.direction_source}`\n")
        fobj.write(f"- Active slices: `{args.active_slices}`\n")
        fobj.write(f"- Max alpha: `{args.max_alpha}`\n")
        fobj.write(f"- Perceptual metrics: `{bool(args.compute_perceptual)}`\n")
        if args.compute_perceptual:
            fobj.write(f"- LPIPS backbone: `{args.lpips_net}`\n")
        fobj.write("\n")
        fobj.write("| " + " | ".join(keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for item in summary:
            vals = []
            for key in keys:
                val = item.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:+.8f}" if key.startswith("delta") else f"{val:.8f}")
                else:
                    vals.append(str(val))
            fobj.write("| " + " | ".join(vals) + " |\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- `force_zero` must exactly reproduce the original EF-LIC branch.\n")
        fobj.write("- `init_hard` should also stay exact because the default controller is fallback-biased.\n")
        fobj.write("- `trained_hard`/`trained_soft` are for checkpoints loaded through `--controller-state`; they are codec-loop smoke modes, not final RD claims.\n")
        fobj.write("- `init_soft`/`trained_soft` may perturb the branch slightly, but forward/decode must remain matched and finite.\n")
        fobj.write("- Any nonzero `delta_bpp` comes only from changed fixed-length index payloads, not entropy side bits.\n")
        fobj.write("- For perceptual metrics, positive `delta_ms_ssim` is better, while negative `delta_lpips` and `delta_dists` are better.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    active_slices = parse_active_slices(args.active_slices)
    metric_fns = build_metric_fns(args, device)
    images = list_images(args.image_dir)[args.start_index : args.start_index + args.max_images]
    if not images:
        raise SystemExit(f"no images selected from {args.image_dir}")

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    state = None
    config_kwargs: dict[str, Any] = {"max_alpha": args.max_alpha}
    if args.controller_state is not None:
        state = torch.load(args.controller_state, map_location=device, weights_only=False)
        saved_config = state.get("config") if isinstance(state, dict) else None
        if isinstance(saved_config, dict):
            config_kwargs.update(saved_config)
    controller = EFLICHCGBranchController(
        EFLICHCGBranchControllerConfig(**config_kwargs)
    ).to(device).eval()
    if state is not None:
        controller.load_state_dict(state.get("model", state), strict=True)
        controller.eval()

    rows: list[dict[str, Any]] = []
    for mode in args.modes:
        for path in images:
            frame = load_image(path, device)
            _, _, h, w = frame.shape
            padded = replicate_pad(frame, h, w)

            orig_inds = net.compress(padded.clone(), force_ind=args.force_ind)
            orig_payload, _, _ = pack_inds(net, orig_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=args.force_ind)[:, :, :h, :w]

            active_inds, active_x_hat_forward, stats = controller_compress_forward(
                net,
                controller,
                padded.clone(),
                force_ind=args.force_ind,
                mode=mode,
                direction_source=args.direction_source,
                active_threshold=args.active_threshold,
                max_risk=args.max_risk,
                risk_temperature=args.risk_temperature,
                active_slices=active_slices,
            )
            active_payload, _, _ = pack_inds(net, active_inds)
            active_x_hat_dec = controller_decompress(
                net,
                controller,
                active_inds,
                force_ind=args.force_ind,
                mode=mode,
                direction_source=args.direction_source,
                active_threshold=args.active_threshold,
                max_risk=args.max_risk,
                risk_temperature=args.risk_temperature,
                active_slices=active_slices,
            )[:, :, :h, :w]
            active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
            decode_diff = (active_x_hat_fwd - active_x_hat_dec).abs()
            baseline_diff = (active_x_hat_dec - orig_x_hat).abs()
            mismatch = compare_inds(orig_inds, active_inds)
            perceptual = compute_perceptual_metrics(
                metric_fns,
                base_x_hat=orig_x_hat,
                active_x_hat=active_x_hat_dec,
                frame=frame,
            )
            nonfinite = int(
                (not torch.isfinite(active_x_hat_dec).all().item())
                or (not torch.isfinite(active_x_hat_fwd).all().item())
                or (not _finite_stats(stats))
                or (bool(perceptual) and not _finite_stats(perceptual))
            )
            row: dict[str, Any] = {
                "mode": mode,
                "image": path.name,
                "force_ind": args.force_ind,
                "direction_source": args.direction_source,
                "active_slices": args.active_slices,
                "bpp": len(active_payload) * 8.0 / float(h * w),
                "delta_bpp": (len(active_payload) - len(orig_payload)) * 8.0 / float(h * w),
                "payload_len_equal": int(len(active_payload) == len(orig_payload)),
                "payload_equal": int(active_payload == orig_payload),
                "base_psnr": psnr_from_mse(mse01(orig_x_hat, frame)),
                "active_psnr": psnr_from_mse(mse01(active_x_hat_dec, frame)),
                "max_decode_diff": float(decode_diff.max().item()),
                "mean_decode_diff": float(decode_diff.mean().item()),
                "max_baseline_diff": float(baseline_diff.max().item()),
                "mean_baseline_diff": float(baseline_diff.mean().item()),
                "nonfinite": nonfinite,
            }
            row["delta_psnr"] = row["active_psnr"] - row["base_psnr"]
            row.update(perceptual)
            row.update(mismatch)
            row.update(stats)
            rows.append(row)
            extra_metrics = ""
            if perceptual:
                extra_metrics = (
                    f" dMS={row['delta_ms_ssim']:+.6f} "
                    f"dLPIPS={row['delta_lpips']:+.6f} dDISTS={row['delta_dists']:+.6f}"
                )
            print(
                f"mode={mode} image={path.name} dPSNR={row['delta_psnr']:+.6f}{extra_metrics} "
                f"dbpp={row['delta_bpp']:+.6f} decmax={row['max_decode_diff']:.2e} "
                f"base_delta={row['max_baseline_diff']:.2e} alpha={row['y_alpha_mean']:.3e} "
                f"gate={row['y_gate_mean']:.3e} active_slices={args.active_slices} nonfinite={row['nonfinite']}"
            )

    summary = summarize(rows, args.modes)
    write_outputs(args=args, rows=rows, summary=summary, images=images)


if __name__ == "__main__":
    main()
