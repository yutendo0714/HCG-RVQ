#!/usr/bin/env python3
"""E263 GLC codec-loop fallback-gate pilot.

E261 showed that offline thresholds over GLC branch diagnostics are not robust.
E262 added a safe fallback-mix primitive.  E263 connects those pieces in a small
codec-loop pilot: train the E250-style q0 local RVQ branch and a compact
reliability/index gate together, using the original image objective plus a
direct gate-weighted rate penalty.

This is a short-cycle design probe, not a full-training claim.  It reports base,
all-on branch, soft learned gate, and hard learned fallback under the same split.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_msssim import ms_ssim

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_minus1_1_to_0_1, get_state_dict  # noqa: E402
from hcg_rvq.reliability_index_controller import (  # noqa: E402
    QAwareThresholdControllerSpec,
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    mix_with_fallback,
    qaware_threshold_gate,
)
from tools.run_e162_glc_pretrained_baseline import psnr01  # noqa: E402
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    TrainableRVQCodebooks,
    build_initial_codebooks,
    crop_to_image,
    dists_call,
    install_trainable_branch,
    run_instrumented,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    collect_residual_set_from_prepared,
    list_images,
    prepare_images,
)
from tools.run_e250_glc_bitaware_tail_vq_split_train import (  # noqa: E402
    image_loss,
    soft_usage_entropy,
)


FEATURES = [
    "active_mse_ratio",
    "active_scalar_mse",
    "active_rvq_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "empirical_bpp_delta",
    "fixed_bpp_delta",
    "base_bpp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=Path("/dpl/openimages/open-images-v6/train/data"))
    p.add_argument("--eval-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e263_glc_fallback_gate_codec_loop_pilot_smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--train-crop-size", type=int, default=256)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--train-start-index", type=int, default=8192)
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=2)
    p.add_argument("--eval-limit", type=int, default=2)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--kmeans-iters", type=int, default=6)
    p.add_argument("--max-train-vectors", type=int, default=12000)
    p.add_argument("--max-rate-vectors", type=int, default=2048)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save HCG/RVQ branch checkpoints every N training steps; 0 disables checkpointing.")
    p.add_argument("--lr-codebook", type=float, default=2e-3)
    p.add_argument("--lr-controller", type=float, default=1e-3)
    p.add_argument("--mse-weight", type=float, default=0.0)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--soft-index-weight", type=float, default=0.01)
    p.add_argument("--soft-index-target", type=float, default=2.0)
    p.add_argument("--soft-index-temp", type=float, default=0.05)
    p.add_argument("--gate-rate-weight", type=float, default=1.0)
    p.add_argument("--gate-l1-weight", type=float, default=0.02)
    p.add_argument("--active-threshold", type=float, default=0.5)
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--rate-cap-dbpp", type=float, default=-1.0, help="If non-negative, emit rate-cap policy rows that pay full branch bpp for selected soft/all-on outputs.")
    p.add_argument("--emit-progressive-extra-rows", action="store_true", help="Emit rows for a base-plus-active-RVQ progressive enhancement rate model.")
    p.add_argument("--progressive-extra-cap-bpp", type=float, default=-1.0, help="If non-negative, cap progressive enhancement rows by active RVQ extra bpp.")
    p.add_argument("--emit-replacement-rows", action="store_true", help="Emit rows for an active-scalar-to-active-RVQ replacement rate model.")
    p.add_argument("--replacement-cap-dbpp", type=float, default=-1.0, help="If non-negative, cap replacement rows by active RVQ minus active scalar bpp.")
    p.add_argument("--replacement-cap-dbpp-values", type=float, nargs="*", default=[], help="Additional replacement delta-bpp caps to emit with suffixed labels, e.g. 0.0025 0.0035 0.0040.")
    p.add_argument("--replacement-signal-bits", type=float, nargs="*", default=[], help="Optional image-level selection/mode signal costs to add to capped replacement rows, e.g. 1 8.")
    p.add_argument("--qaware-controller-json", type=Path, default=None, help="Optional E379-style q-aware deployment JSON. Emits replacement rows selected by the exported controller spec.")
    p.add_argument("--qaware-policy-modes", nargs="*", default=["q-aware", "global"], help="Policy modes to load from --qaware-controller-json. Use an empty list to load all modes.")
    p.add_argument("--controller-hidden", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--wandb-enabled", action="store_true", help="Log training trace and final perceptual summary rows to Weights & Biases.")
    p.add_argument("--wandb-project", default="HCG-RVQ")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb_enabled:
        return None
    import wandb

    run_name = args.wandb_name or args.output_prefix.name
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
        dir=str(args.output_prefix.parent),
    )


def wandb_log_summary(wandb_run: Any, summary: list[dict[str, Any]]) -> None:
    if wandb_run is None:
        return
    for row in summary:
        label = safe_token(str(row["label"]))
        payload = {
            f"final/{label}/bpp": row["bpp"],
            f"final/{label}/delta_bpp": row["delta_bpp"],
            f"final/{label}/ms_ssim": row["ms_ssim"],
            f"final/{label}/delta_ms_ssim": row["delta_ms_ssim"],
            f"final/{label}/lpips": row["lpips"],
            f"final/{label}/delta_lpips": row["delta_lpips"],
            f"final/{label}/dists": row["dists"],
            f"final/{label}/delta_dists": row["delta_dists"],
            f"final/{label}/score": row["score"],
            f"final/{label}/selected_frac": row["selected_frac"],
            f"final/{label}/gate_mean": row["gate_mean"],
            f"final/{label}/index_entropy_mean": row["index_entropy_mean"],
            f"final/{label}/nonfinite_rows": row["nonfinite_rows"],
        }
        wandb_run.log(payload)


def save_branch_checkpoint(
    args: argparse.Namespace,
    step: int,
    controller: torch.nn.Module,
    codebooks_by_q: dict[int, list[torch.Tensor]],
    feature_mu: torch.Tensor,
    feature_std: torch.Tensor,
    trace: list[dict[str, float]],
) -> Path:
    ckpt_path = args.output_prefix.parent / f"{args.output_prefix.name}_step{step:04d}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "controller_state_dict": controller.state_dict(),
        "codebooks_by_q": {
            int(q): [cb.detach().cpu() for cb in codebooks]
            for q, codebooks in codebooks_by_q.items()
        },
        "feature_mu": feature_mu.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "trace": trace,
    }
    torch.save(payload, ckpt_path)
    print(f"[checkpoint] saved {ckpt_path}")
    return ckpt_path


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def mean_psnr(values: list[float]) -> float:
    mses = [10.0 ** (-v / 10.0) for v in values if math.isfinite(v)]
    mse = mean(mses)
    return -10.0 * math.log10(mse) if mse > 0 else float("inf")


def cap_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def image_signal_bpp(bits: float, item) -> float:
    pixels = max(1.0, float(item.height * item.width))
    return max(0.0, float(bits)) / pixels


def safe_token(text: str) -> str:
    out = []
    for char in str(text):
        if char.isalnum():
            out.append(char.lower())
        elif char == ".":
            out.append("p")
        else:
            out.append("_")
    return "_".join(part for part in "".join(out).split("_") if part)


def load_qaware_specs(path: Path | None, policy_modes: list[str]) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = payload.get("policies")
    if policies is None:
        policies = [payload.get("main_policy")] if payload.get("main_policy") else []
    allowed = {str(mode) for mode in policy_modes}
    specs: list[dict[str, Any]] = []
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        mode = str(policy.get("mode", "policy"))
        if allowed and mode not in allowed:
            continue
        feature = str(policy.get("feature", "index_entropy_mean"))
        raw_spec = policy.get("controller_spec", {})
        raw_thresholds = raw_spec.get("thresholds", policy.get("deployment_thresholds", policy.get("thresholds", {})))
        if not isinstance(raw_thresholds, dict):
            continue
        thresholds = {int(q): float(value) for q, value in raw_thresholds.items()}
        if not thresholds:
            continue
        direction = str(raw_spec.get("direction", policy.get("direction", ">=")))
        soft_width = float(raw_spec.get("soft_width", 0.0))
        margin = policy.get("threshold_margin", "nomargin")
        tag = safe_token(f"qaware_{mode}_{feature}_m{margin}")
        specs.append(
            {
                "tag": tag,
                "mode": mode,
                "feature": feature,
                "spec": QAwareThresholdControllerSpec(thresholds=thresholds, direction=direction, soft_width=soft_width),
            }
        )
    return specs


def branch_feature_dict(base_stats: dict[str, Any], branch_stats: dict[str, Any], pixels: float) -> dict[str, float]:
    base_bpp = finite(base_stats.get("gaussian_bits_total")) / pixels
    fixed_bpp = (finite(branch_stats.get("hybrid_fixed_bits_y")) + finite(branch_stats.get("bits_z"))) / pixels
    empirical_bpp = (finite(branch_stats.get("hybrid_empirical_bits_y")) + finite(branch_stats.get("bits_z"))) / pixels
    return {
        "active_mse_ratio": finite(branch_stats.get("active_mse_ratio")),
        "active_scalar_mse": finite(branch_stats.get("active_scalar_mse")),
        "active_rvq_mse": finite(branch_stats.get("active_rvq_mse")),
        "index_entropy_mean": finite(branch_stats.get("index_entropy_mean")),
        "index_used_frac_mean": finite(branch_stats.get("index_used_frac_mean")),
        "index_dead_frac_mean": finite(branch_stats.get("index_dead_frac_mean")),
        "empirical_bpp_delta": empirical_bpp - base_bpp,
        "fixed_bpp_delta": fixed_bpp - base_bpp,
        "base_bpp": base_bpp,
        "inactive_scalar_bpp": finite(branch_stats.get("inactive_scalar_bits")) / pixels,
        "active_scalar_bpp": finite(branch_stats.get("active_scalar_bits")) / pixels,
        "active_rvq_fixed_bpp": finite(branch_stats.get("active_rvq_fixed_bits")) / pixels,
        "active_rvq_empirical_bpp": finite(branch_stats.get("active_rvq_empirical_bits")) / pixels,
        "active_rvq_extra_bpp": finite(branch_stats.get("active_rvq_empirical_bits")) / pixels,
        "active_replacement_delta_bpp": (finite(branch_stats.get("active_rvq_empirical_bits")) - finite(branch_stats.get("active_scalar_bits"))) / pixels,
    }


def standardizer(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    mu = {key: mean([row[key] for row in rows]) for key in FEATURES}
    std = {}
    for key in FEATURES:
        var = mean([(row[key] - mu[key]) ** 2 for row in rows])
        std[key] = math.sqrt(var) if math.isfinite(var) and var > 1e-12 else 1.0
    return mu, std


def feature_tensor(row: dict[str, float], mu: dict[str, float], std: dict[str, float], device: torch.device) -> torch.Tensor:
    values = [(finite(row.get(key), mu[key]) - mu[key]) / std[key] for key in FEATURES]
    return torch.tensor([values], dtype=torch.float32, device=device)


@torch.no_grad()
def collect_initial_feature_rows(
    net: GLC_Image,
    official_forward,
    codebooks_by_q: dict[int, TrainableRVQCodebooks],
    prepared,
    q_indexes: list[int],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for q in q_indexes:
        install_trainable_branch(net, codebooks_by_q[q], args)
        for item in prepared:
            pixels = float(item.height * item.width)
            net.forward_four_part_prior = official_forward
            _, base_stats = run_instrumented(net, item.x_pad, q)
            install_trainable_branch(net, codebooks_by_q[q], args)
            _, branch_stats = run_instrumented(net, item.x_pad, q)
            rows.append(branch_feature_dict(base_stats, branch_stats, pixels))
    net.forward_four_part_prior = official_forward
    return rows


def metric_values(x_hat: torch.Tensor, x_hat01: torch.Tensor, item, lpips_fn, dists_fn) -> dict[str, float]:
    return {
        "psnr": psnr01(x_hat01, item.img01),
        "ms_ssim": float(ms_ssim(x_hat01, item.img01, data_range=1.0).item()),
        "lpips": float(lpips_fn(x_hat, item.x).mean().item()),
        "dists": float(dists_call(dists_fn, x_hat01, item.img01, require_grad=False).detach().item()),
    }


def add_policy_row(
    rows: list[dict[str, Any]],
    *,
    label: str,
    q: int,
    item,
    bpp: float,
    gate: float,
    selected: bool,
    metrics: dict[str, float],
    base_metrics: dict[str, float],
    base_bpp: float,
    feature_row: dict[str, float],
    nonfinite: int,
    selection_signal_bpp: float = 0.0,
) -> None:
    rows.append(
        {
            "label": label,
            "q_index": q,
            "image": item.path.name,
            "height": item.height,
            "width": item.width,
            "bpp": bpp,
            "base_bpp": base_bpp,
            "selection_signal_bpp": selection_signal_bpp,
            "delta_bpp": bpp - base_bpp,
            "psnr": metrics["psnr"],
            "delta_psnr": metrics["psnr"] - base_metrics["psnr"],
            "ms_ssim": metrics["ms_ssim"],
            "delta_ms_ssim": metrics["ms_ssim"] - base_metrics["ms_ssim"],
            "lpips": metrics["lpips"],
            "delta_lpips": metrics["lpips"] - base_metrics["lpips"],
            "dists": metrics["dists"],
            "delta_dists": metrics["dists"] - base_metrics["dists"],
            "score": (metrics["dists"] - base_metrics["dists"]) + 3.0 * (metrics["lpips"] - base_metrics["lpips"]) + (bpp - base_bpp),
            "gate_mean": gate,
            "selected": int(selected),
            "active_mse_ratio": feature_row["active_mse_ratio"],
            "active_rvq_mse": feature_row["active_rvq_mse"],
            "active_scalar_mse": feature_row["active_scalar_mse"],
            "index_entropy_mean": feature_row["index_entropy_mean"],
            "index_used_frac_mean": feature_row["index_used_frac_mean"],
            "index_dead_frac_mean": feature_row["index_dead_frac_mean"],
            "inactive_scalar_bpp": feature_row.get("inactive_scalar_bpp", 0.0),
            "active_scalar_bpp": feature_row.get("active_scalar_bpp", 0.0),
            "active_rvq_fixed_bpp": feature_row.get("active_rvq_fixed_bpp", 0.0),
            "active_rvq_empirical_bpp": feature_row.get("active_rvq_empirical_bpp", 0.0),
            "active_rvq_extra_bpp": feature_row.get("active_rvq_extra_bpp", 0.0),
            "active_replacement_delta_bpp": feature_row.get("active_replacement_delta_bpp", 0.0),
            "nonfinite": nonfinite,
        }
    )


@torch.no_grad()
def evaluate_policies(
    net: GLC_Image,
    official_forward,
    controller: ReliabilityIndexMLP,
    codebooks_by_q: dict[int, TrainableRVQCodebooks],
    prepared,
    q_indexes: list[int],
    args: argparse.Namespace,
    lpips_fn,
    dists_fn,
    mu: dict[str, float],
    std: dict[str, float],
    label_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qaware_specs = load_qaware_specs(args.qaware_controller_json, args.qaware_policy_modes)
    for q in q_indexes:
        for item in prepared:
            pixels = float(item.height * item.width)
            net.forward_four_part_prior = official_forward
            base_pad, base_stats = run_instrumented(net, item.x_pad, q)
            install_trainable_branch(net, codebooks_by_q[q], args)
            branch_pad, branch_stats = run_instrumented(net, item.x_pad, q)
            net.forward_four_part_prior = official_forward

            base = crop_to_image(base_pad, item)
            branch = crop_to_image(branch_pad, item)
            base01 = from_minus1_1_to_0_1(base).clamp(0, 1)
            branch01 = from_minus1_1_to_0_1(branch).clamp(0, 1)
            base_metrics = metric_values(base, base01, item, lpips_fn, dists_fn)
            branch_metrics = metric_values(branch, branch01, item, lpips_fn, dists_fn)
            feature_row = branch_feature_dict(base_stats, branch_stats, pixels)
            base_bpp = feature_row["base_bpp"]
            branch_bpp = base_bpp + feature_row["empirical_bpp_delta"]
            features = feature_tensor(feature_row, mu, std, base.device)
            ctrl = controller(features)
            soft_mixed, soft_gate = mix_with_fallback(
                base,
                branch,
                ctrl["active_logit"],
                active_threshold=args.active_threshold,
                hard=False,
                max_gate=args.max_gate,
            )
            hard_mixed, hard_gate = mix_with_fallback(
                base,
                branch,
                ctrl["active_logit"],
                active_threshold=args.active_threshold,
                hard=True,
                max_gate=args.max_gate,
            )
            soft01 = from_minus1_1_to_0_1(soft_mixed).clamp(0, 1)
            hard01 = from_minus1_1_to_0_1(hard_mixed).clamp(0, 1)
            soft_gate_mean = float(soft_gate.mean().item())
            hard_gate_mean = float(hard_gate.mean().item())
            soft_metrics = metric_values(soft_mixed, soft01, item, lpips_fn, dists_fn)
            hard_metrics = metric_values(hard_mixed, hard01, item, lpips_fn, dists_fn)
            nonfinite = int(
                base_stats["nonfinite_forward"]
                or branch_stats["nonfinite_forward"]
                or not torch.isfinite(soft_mixed).all().item()
                or not torch.isfinite(hard_mixed).all().item()
            )
            add_policy_row(
                rows,
                label=f"{label_prefix}_base",
                q=q,
                item=item,
                bpp=base_bpp,
                gate=0.0,
                selected=False,
                metrics=base_metrics,
                base_metrics=base_metrics,
                base_bpp=base_bpp,
                feature_row=feature_row,
                nonfinite=nonfinite,
            )
            add_policy_row(
                rows,
                label=f"{label_prefix}_all_on",
                q=q,
                item=item,
                bpp=branch_bpp,
                gate=1.0,
                selected=True,
                metrics=branch_metrics,
                base_metrics=base_metrics,
                base_bpp=base_bpp,
                feature_row=feature_row,
                nonfinite=nonfinite,
            )
            add_policy_row(
                rows,
                label=f"{label_prefix}_soft_gate",
                q=q,
                item=item,
                bpp=base_bpp + soft_gate_mean * feature_row["empirical_bpp_delta"],
                gate=soft_gate_mean,
                selected=soft_gate_mean > 0.0,
                metrics=soft_metrics,
                base_metrics=base_metrics,
                base_bpp=base_bpp,
                feature_row=feature_row,
                nonfinite=nonfinite,
            )
            add_policy_row(
                rows,
                label=f"{label_prefix}_hard_gate",
                q=q,
                item=item,
                bpp=base_bpp + hard_gate_mean * feature_row["empirical_bpp_delta"],
                gate=hard_gate_mean,
                selected=hard_gate_mean > 0.0,
                metrics=hard_metrics,
                base_metrics=base_metrics,
                base_bpp=base_bpp,
                feature_row=feature_row,
                nonfinite=nonfinite,
            )
            print(
                f"{label_prefix} q={q} {item.path.name} all_on_score={rows[-3]['score']:+.6f} "
                f"soft_score={rows[-2]['score']:+.6f} hard_score={rows[-1]['score']:+.6f} "
                f"soft_gate={soft_gate_mean:.4f} hard_gate={hard_gate_mean:.4f} nonfinite={nonfinite}"
            )
            if args.rate_cap_dbpp >= 0.0:
                cap_selected = feature_row["empirical_bpp_delta"] <= args.rate_cap_dbpp
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_rate_cap_soft",
                    q=q,
                    item=item,
                    bpp=branch_bpp if cap_selected else base_bpp,
                    gate=soft_gate_mean if cap_selected else 0.0,
                    selected=cap_selected,
                    metrics=soft_metrics if cap_selected else base_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_rate_cap_all_on",
                    q=q,
                    item=item,
                    bpp=branch_bpp if cap_selected else base_bpp,
                    gate=1.0 if cap_selected else 0.0,
                    selected=cap_selected,
                    metrics=branch_metrics if cap_selected else base_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
            if args.emit_progressive_extra_rows:
                progressive_extra_bpp = max(0.0, feature_row.get("active_rvq_extra_bpp", 0.0))
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_progressive_extra_soft",
                    q=q,
                    item=item,
                    bpp=base_bpp + progressive_extra_bpp,
                    gate=soft_gate_mean,
                    selected=soft_gate_mean > 0.0,
                    metrics=soft_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_progressive_extra_all_on",
                    q=q,
                    item=item,
                    bpp=base_bpp + progressive_extra_bpp,
                    gate=1.0,
                    selected=True,
                    metrics=branch_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
                if args.progressive_extra_cap_bpp >= 0.0:
                    progressive_selected = progressive_extra_bpp <= args.progressive_extra_cap_bpp
                    add_policy_row(
                        rows,
                        label=f"{label_prefix}_rate_cap_progressive_extra_soft",
                        q=q,
                        item=item,
                        bpp=base_bpp + progressive_extra_bpp if progressive_selected else base_bpp,
                        gate=soft_gate_mean if progressive_selected else 0.0,
                        selected=progressive_selected,
                        metrics=soft_metrics if progressive_selected else base_metrics,
                        base_metrics=base_metrics,
                        base_bpp=base_bpp,
                        feature_row=feature_row,
                        nonfinite=nonfinite,
                    )
                    add_policy_row(
                        rows,
                        label=f"{label_prefix}_rate_cap_progressive_extra_all_on",
                        q=q,
                        item=item,
                        bpp=base_bpp + progressive_extra_bpp if progressive_selected else base_bpp,
                        gate=1.0 if progressive_selected else 0.0,
                        selected=progressive_selected,
                        metrics=branch_metrics if progressive_selected else base_metrics,
                        base_metrics=base_metrics,
                        base_bpp=base_bpp,
                        feature_row=feature_row,
                        nonfinite=nonfinite,
                    )
            if args.emit_replacement_rows:
                replacement_dbpp = feature_row.get("active_replacement_delta_bpp", 0.0)
                replacement_bpp = base_bpp + replacement_dbpp
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_replacement_soft",
                    q=q,
                    item=item,
                    bpp=replacement_bpp,
                    gate=soft_gate_mean,
                    selected=soft_gate_mean > 0.0,
                    metrics=soft_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
                add_policy_row(
                    rows,
                    label=f"{label_prefix}_replacement_all_on",
                    q=q,
                    item=item,
                    bpp=replacement_bpp,
                    gate=1.0,
                    selected=True,
                    metrics=branch_metrics,
                    base_metrics=base_metrics,
                    base_bpp=base_bpp,
                    feature_row=feature_row,
                    nonfinite=nonfinite,
                )
                replacement_cap_specs: list[tuple[str, float]] = []
                if args.replacement_cap_dbpp >= 0.0:
                    replacement_cap_specs.append(("", args.replacement_cap_dbpp))
                seen_replacement_caps = {round(cap, 12) for _, cap in replacement_cap_specs}
                for cap in args.replacement_cap_dbpp_values:
                    if cap < 0.0:
                        continue
                    rounded = round(float(cap), 12)
                    if rounded in seen_replacement_caps:
                        continue
                    seen_replacement_caps.add(rounded)
                    replacement_cap_specs.append((f"_cap{cap_token(float(cap))}", float(cap)))
                for cap_suffix, cap_value in replacement_cap_specs:
                    replacement_selected = replacement_dbpp <= cap_value
                    add_policy_row(
                        rows,
                        label=f"{label_prefix}_rate_cap_replacement_soft{cap_suffix}",
                        q=q,
                        item=item,
                        bpp=replacement_bpp if replacement_selected else base_bpp,
                        gate=soft_gate_mean if replacement_selected else 0.0,
                        selected=replacement_selected,
                        metrics=soft_metrics if replacement_selected else base_metrics,
                        base_metrics=base_metrics,
                        base_bpp=base_bpp,
                        feature_row=feature_row,
                        nonfinite=nonfinite,
                    )
                    add_policy_row(
                        rows,
                        label=f"{label_prefix}_rate_cap_replacement_all_on{cap_suffix}",
                        q=q,
                        item=item,
                        bpp=replacement_bpp if replacement_selected else base_bpp,
                        gate=1.0 if replacement_selected else 0.0,
                        selected=replacement_selected,
                        metrics=branch_metrics if replacement_selected else base_metrics,
                        base_metrics=base_metrics,
                        base_bpp=base_bpp,
                        feature_row=feature_row,
                        nonfinite=nonfinite,
                    )
                    for signal_bits in args.replacement_signal_bits:
                        if signal_bits < 0.0:
                            continue
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_suffix = f"_sig{cap_token(float(signal_bits))}b"
                        add_policy_row(
                            rows,
                            label=f"{label_prefix}_rate_cap_replacement_soft{cap_suffix}{signal_suffix}",
                            q=q,
                            item=item,
                            bpp=(replacement_bpp if replacement_selected else base_bpp) + signal_bpp,
                            gate=soft_gate_mean if replacement_selected else 0.0,
                            selected=replacement_selected,
                            metrics=soft_metrics if replacement_selected else base_metrics,
                            base_metrics=base_metrics,
                            base_bpp=base_bpp,
                            feature_row=feature_row,
                            nonfinite=nonfinite,
                            selection_signal_bpp=signal_bpp,
                        )
                        add_policy_row(
                            rows,
                            label=f"{label_prefix}_rate_cap_replacement_all_on{cap_suffix}{signal_suffix}",
                            q=q,
                            item=item,
                            bpp=(replacement_bpp if replacement_selected else base_bpp) + signal_bpp,
                            gate=1.0 if replacement_selected else 0.0,
                            selected=replacement_selected,
                            metrics=branch_metrics if replacement_selected else base_metrics,
                            base_metrics=base_metrics,
                            base_bpp=base_bpp,
                            feature_row=feature_row,
                            nonfinite=nonfinite,
                            selection_signal_bpp=signal_bpp,
                        )
                for spec_row in qaware_specs:
                    feature_name = str(spec_row["feature"])
                    feature_value = finite(feature_row.get(feature_name), float("nan"))
                    selected = False
                    if math.isfinite(feature_value):
                        gate_tensor = qaware_threshold_gate(
                            torch.tensor([feature_value], dtype=torch.float32, device=base.device),
                            int(q),
                            spec_row["spec"],
                            hard=True,
                        )
                        selected = bool(float(gate_tensor.item()) > 0.5)
                    label = f"{label_prefix}_{spec_row['tag']}_replacement_soft"
                    add_policy_row(
                        rows,
                        label=label,
                        q=q,
                        item=item,
                        bpp=replacement_bpp if selected else base_bpp,
                        gate=soft_gate_mean if selected else 0.0,
                        selected=selected,
                        metrics=soft_metrics if selected else base_metrics,
                        base_metrics=base_metrics,
                        base_bpp=base_bpp,
                        feature_row=feature_row,
                        nonfinite=nonfinite,
                    )
                    for signal_bits in args.replacement_signal_bits:
                        if signal_bits < 0.0:
                            continue
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_suffix = f"_sig{cap_token(float(signal_bits))}b"
                        add_policy_row(
                            rows,
                            label=f"{label}{signal_suffix}",
                            q=q,
                            item=item,
                            bpp=(replacement_bpp if selected else base_bpp) + signal_bpp,
                            gate=soft_gate_mean if selected else 0.0,
                            selected=selected,
                            metrics=soft_metrics if selected else base_metrics,
                            base_metrics=base_metrics,
                            base_bpp=base_bpp,
                            feature_row=feature_row,
                            nonfinite=nonfinite,
                            selection_signal_bpp=signal_bpp,
                        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label in sorted({str(row["label"]) for row in rows}):
        subset = [row for row in rows if str(row["label"]) == label]
        out.append(
            {
                "label": label,
                "images": len(subset),
                "bpp": mean([finite(row["bpp"], float("nan")) for row in subset]),
                "delta_bpp": mean([finite(row["delta_bpp"], float("nan")) for row in subset]),
                "selection_signal_bpp": mean([finite(row.get("selection_signal_bpp", 0.0), float("nan")) for row in subset]),
                "psnr": mean_psnr([finite(row["psnr"], float("nan")) for row in subset]),
                "delta_psnr": mean([finite(row["delta_psnr"], float("nan")) for row in subset]),
                "ms_ssim": mean([finite(row["ms_ssim"], float("nan")) for row in subset]),
                "delta_ms_ssim": mean([finite(row["delta_ms_ssim"], float("nan")) for row in subset]),
                "lpips": mean([finite(row["lpips"], float("nan")) for row in subset]),
                "delta_lpips": mean([finite(row["delta_lpips"], float("nan")) for row in subset]),
                "dists": mean([finite(row["dists"], float("nan")) for row in subset]),
                "delta_dists": mean([finite(row["delta_dists"], float("nan")) for row in subset]),
                "score": mean([finite(row["score"], float("nan")) for row in subset]),
                "gate_mean": mean([finite(row["gate_mean"], float("nan")) for row in subset]),
                "selected_frac": mean([finite(row["selected"], float("nan")) for row in subset]),
                "active_mse_ratio": mean([finite(row["active_mse_ratio"], float("nan")) for row in subset]),
                "index_entropy_mean": mean([finite(row["index_entropy_mean"], float("nan")) for row in subset]),
                "nonfinite_rows": int(sum(int(row["nonfinite"]) for row in subset)),
            }
        )
    return out


def write_outputs(
    args: argparse.Namespace,
    train_paths: list[Path],
    eval_paths: list[Path],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    feature_mu: dict[str, float],
    feature_std: dict[str, float],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "E263 GLC fallback-gate codec-loop pilot",
        "note": "Short-cycle design probe. Not a full-training claim.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_images": [str(path) for path in train_paths],
        "eval_images": [str(path) for path in eval_paths],
        "features": FEATURES,
        "feature_mu": feature_mu,
        "feature_std": feature_std,
        "summary": summary,
        "train_trace": trace,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E263 GLC Fallback-Gate Codec-Loop Pilot",
        "",
        "Short-cycle design probe after E261/E262.  Trains an E250-style local RVQ branch and a compact fallback gate together.",
        "The loss is intentionally simple: image loss plus gate-weighted empirical bpp delta and a weak gate sparsity penalty.",
        "",
        f"Train dir/start/limit/crop: `{args.train_dir}` / `{args.train_start_index}` / `{args.train_limit}` / `{args.train_crop_size}`",
        f"Eval dir/start/limit/crop: `{args.eval_dir}` / `{args.eval_start_index}` / `{args.eval_limit}` / `{args.eval_crop_size}`",
        f"Loss weights: mse `{args.mse_weight}`, lpips `{args.lpips_weight}`, dists `{args.dists_weight}`, soft-index `{args.soft_index_weight}`, gate-rate `{args.gate_rate_weight}`, gate-l1 `{args.gate_l1_weight}`",
        "",
        "| label | images | bpp | dbpp | signal bpp | score | dPSNR | dMS-SSIM | dLPIPS | dDISTS | gate | selected | active MSE ratio | H | nonfinite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['images']} | {row['bpp']:.6f} | {row['delta_bpp']:+.6f} | "
            f"{row['selection_signal_bpp']:.8f} | "
            f"{row['score']:+.6f} | {row['delta_psnr']:+.6f} | {row['delta_ms_ssim']:+.6f} | "
            f"{row['delta_lpips']:+.6f} | {row['delta_dists']:+.6f} | {row['gate_mean']:.6f} | "
            f"{row['selected_frac']:.6f} | {row['active_mse_ratio']:.6f} | {row['index_entropy_mean']:.6f} | {row['nonfinite_rows']} |"
        )
    if trace:
        lines.extend(
            [
                "",
                "## Train Trace",
                "",
                "| step | loss | image | rate | gate l1 | soft H | soft excess | gate | dists |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in trace:
            lines.append(
                f"| {row['step']} | {row['loss']:.6f} | {row['image_loss']:.6f} | {row['rate_loss']:.6f} | "
                f"{row['gate_l1_loss']:.6f} | {row['soft_index_entropy']:.6f} | {row['soft_index_excess']:.6f} | "
                f"{row['gate_mean']:.6f} | {row['dists']:.6f} |"
            )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


def main() -> None:
    args = parse_args()
    wandb_run = maybe_init_wandb(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_paths = list_images(args.train_dir, args.train_start_index, args.train_limit)
    eval_paths = list_images(args.eval_dir, args.eval_start_index, args.eval_limit)
    if not train_paths:
        raise SystemExit(f"no train images in {args.train_dir}")
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.eval_dir}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    dists_fn = dists.DISTS().to(device).eval()
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    net = GLC_Image(inplace=False).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    active_parts = set(args.active_parts)
    active_groups = set(args.active_groups)

    train_prepared = prepare_images(train_paths, device, args.padding_size, args.train_crop_size)
    eval_prepared = prepare_images(eval_paths, device, args.padding_size, args.eval_crop_size)
    train_items_by_q = {}
    codebooks_by_q: dict[int, TrainableRVQCodebooks] = {}
    with torch.no_grad():
        for q in args.q_indexes:
            train_items = [
                collect_residual_set_from_prepared(net, item, q, args.group_size, active_parts, active_groups)
                for item in train_prepared
            ]
            train_items_by_q[q] = train_items
            initial = build_initial_codebooks(
                train_items,
                args.scope,
                args.k,
                args.stages,
                args.kmeans_iters,
                args.max_train_vectors,
                args.seed + q * 10000,
                device,
            )
            codebooks_by_q[q] = TrainableRVQCodebooks(initial, device).to(device)
            print(f"initialized q={q} keys={len(initial)} train_vectors={sum(int(x.vectors.shape[0]) for x in train_items)}")
    net.masks = {}

    feature_rows = collect_initial_feature_rows(net, official_forward, codebooks_by_q, train_prepared, args.q_indexes, args)
    feature_mu, feature_std = standardizer(feature_rows)
    controller = ReliabilityIndexMLP(
        ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=args.controller_hidden, zero_bias=-2.0)
    ).to(device)

    rows = evaluate_policies(
        net,
        official_forward,
        controller,
        codebooks_by_q,
        eval_prepared,
        args.q_indexes,
        args,
        lpips_fn,
        dists_fn,
        feature_mu,
        feature_std,
        "init",
    )

    params = [
        {"params": [p for module in codebooks_by_q.values() for p in module.parameters()], "lr": args.lr_codebook},
        {"params": list(controller.parameters()), "lr": args.lr_controller},
    ]
    opt = torch.optim.Adam(params)
    trace: list[dict[str, Any]] = []
    num_train_losses = max(1, len(args.q_indexes) * len(train_prepared))
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        accum = {
            "loss": 0.0,
            "image_loss": 0.0,
            "rate_loss": 0.0,
            "gate_l1_loss": 0.0,
            "soft_index_entropy": 0.0,
            "soft_index_excess": 0.0,
            "gate_mean": 0.0,
            "mse": 0.0,
            "lpips": 0.0,
            "dists": 0.0,
        }
        count = 0
        for q in args.q_indexes:
            install_trainable_branch(net, codebooks_by_q[q], args)
            for item in train_prepared:
                pixels = float(item.height * item.width)
                with torch.no_grad():
                    net.forward_four_part_prior = official_forward
                    base_pad, base_stats = run_instrumented(net, item.x_pad, q)
                    base = crop_to_image(base_pad, item).detach()
                install_trainable_branch(net, codebooks_by_q[q], args)
                branch_pad, branch_stats = run_instrumented(net, item.x_pad, q)
                branch = crop_to_image(branch_pad, item)
                feat = branch_feature_dict(base_stats, branch_stats, pixels)
                ctrl = controller(feature_tensor(feat, feature_mu, feature_std, device))
                mixed, gate = mix_with_fallback(
                    base,
                    branch,
                    ctrl["active_logit"],
                    active_threshold=args.active_threshold,
                    hard=False,
                    max_gate=args.max_gate,
                )
                mixed01 = from_minus1_1_to_0_1(mixed).clamp(0, 1)
                img_loss, parts = image_loss(mixed, mixed01, item, lpips_fn, dists_fn, args)
                gate_mean = gate.mean()
                rate_delta = max(0.0, feat["empirical_bpp_delta"])
                rate_loss = gate_mean * mixed.new_tensor(rate_delta * float(args.gate_rate_weight))
                gate_l1_loss = gate_mean * float(args.gate_l1_weight)
                loss = (img_loss + rate_loss + gate_l1_loss) / num_train_losses
                loss.backward()
                accum["loss"] += float(loss.detach().item())
                accum["image_loss"] += float(img_loss.detach().item())
                accum["rate_loss"] += float(rate_loss.detach().item())
                accum["gate_l1_loss"] += float(gate_l1_loss.detach().item())
                accum["gate_mean"] += float(gate_mean.detach().item())
                for key in ("mse", "lpips", "dists"):
                    accum[key] += parts[key]
                count += 1
                del base_pad, base, branch_pad, branch, mixed, mixed01, img_loss, rate_loss, gate_l1_loss, loss
            proxy, proxy_parts = soft_usage_entropy(codebooks_by_q[q], train_items_by_q[q], args, q, device)
            soft_loss = args.soft_index_weight * proxy / max(1, len(args.q_indexes))
            soft_loss.backward()
            accum["loss"] += float(soft_loss.detach().item())
            accum["soft_index_entropy"] += proxy_parts["soft_index_entropy"]
            accum["soft_index_excess"] += proxy_parts["soft_index_excess"]
            del proxy, soft_loss
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        trace_row = {"step": step}
        for key, value in accum.items():
            denom = max(1, count) if key not in {"soft_index_entropy", "soft_index_excess"} else max(1, len(args.q_indexes))
            trace_row[key] = value / denom
        trace.append(trace_row)
        print(
            f"step={step}/{args.steps} loss={trace_row['loss']:.6f} image={trace_row['image_loss']:.6f} "
            f"rate={trace_row['rate_loss']:.6f} gate={trace_row['gate_mean']:.4f} dists={trace_row['dists']:.6f}"
        )
        if wandb_run is not None:
            train_payload = {f"train/{key}": value for key, value in trace_row.items() if key != "step"}
            wandb_run.log(train_payload, step=step)
        if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
            save_branch_checkpoint(args, step, controller, codebooks_by_q, feature_mu, feature_std, trace)
    net.forward_four_part_prior = official_forward
    print(f"training_ms={(time.perf_counter() - t0) * 1000.0:.1f}")

    rows.extend(
        evaluate_policies(
            net,
            official_forward,
            controller,
            codebooks_by_q,
            eval_prepared,
            args.q_indexes,
            args,
            lpips_fn,
            dists_fn,
            feature_mu,
            feature_std,
            "trained",
        )
    )
    summary = summarize(rows)
    write_outputs(args, train_paths, eval_paths, rows, summary, trace, feature_mu, feature_std)
    wandb_log_summary(wandb_run, summary)
    if wandb_run is not None:
        wandb_run.finish()
    del train_prepared, eval_prepared, train_items_by_q, codebooks_by_q, controller, net
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
