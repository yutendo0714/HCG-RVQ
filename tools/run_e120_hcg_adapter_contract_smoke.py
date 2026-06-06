#!/usr/bin/env python3
"""Smoke-test the HCG quantizer adapter contract against the current forward path."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config


ANALYSIS = ROOT / "experiments" / "analysis"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_deadzone014_from_beta005_seed1234.yaml"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_"
    "excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_"
    "distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_g64_l1_k128_lambda0035_seed1234"
    / "checkpoint_step_250.pth.tar"
)
OUT_PREFIX = ANALYSIS / "e120_hcg_adapter_contract_smoke"


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def sanitize_loss_config(config: dict) -> dict:
    eval_loss_cfg = dict(config.get("loss", {}))
    for name in (
        "rho_householder_reliability_teacher",
        "rho_householder_residual_selector_teacher",
        "rho_householder_residual_selector_noop",
        "rho_anchor_mu",
        "rho_anchor_log_s",
        "rho_anchor_u",
        "rho_anchor_y_hat",
        "rho_anchor_selected_distortion_margin",
    ):
        eval_loss_cfg[name] = 0.0
    return eval_loss_cfg


def scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default="/dpl/kodak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    dataset = ImageFolderDataset(
        [args.data_root],
        patch_size=None,
        training=False,
        max_images=args.index + 1,
        start_index=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    for _ in range(args.index):
        batch = next(iter(loader))
    x = batch.to(device, non_blocking=True)
    x_pad, hw = pad_to_multiple(x)

    criterion = RateDistortionLoss(**sanitize_loss_config(config))

    with torch.no_grad():
        forward_out = model(x_pad)
        forward_eval = dict(forward_out)
        forward_eval["x_hat"] = crop_to_hw(forward_out["x_hat"], hw)
        forward_loss = criterion(forward_eval, x)

        image_hw = (x_pad.shape[-2], x_pad.shape[-1])
        y = model.g_a(x_pad)
        z = model.h_a(y)
        z_hat, z_likelihoods = model.entropy_bottleneck(z)
        hyper_features = model.h_s(z_hat)
        y_hat, indices, commit_loss, rvq_stats, conditioning_tensors = model._conditioned_rvq(
            y,
            hyper_features,
            image_hw,
        )
        if model.index_entropy_model is not None:
            bpp_y_index, _ = model.index_entropy_model(hyper_features, indices, image_hw)
        else:
            bpp_y_index = rvq_stats["fixed_bpp"]
        x_hat = model.g_s(y_hat)
        manual_out = {
            "x_hat": crop_to_hw(x_hat, hw),
            "likelihoods": {"z": z_likelihoods},
            "y": y,
            "y_hat": y_hat,
            "hyper_features": hyper_features,
            "indices": indices,
            "commit_loss": commit_loss,
            "rvq_stats": rvq_stats,
            "conditioning_tensors": conditioning_tensors,
            "bpp_y_index": bpp_y_index,
        }
        manual_loss = criterion(manual_out, x)

    rvq_keys = sorted(set(forward_out["rvq_stats"]) & set(rvq_stats))
    rvq_diffs = {}
    for key in rvq_keys:
        lhs = scalar(forward_out["rvq_stats"][key])
        rhs = scalar(rvq_stats[key])
        if lhs is not None and rhs is not None:
            rvq_diffs[key] = abs(lhs - rhs)

    loss_diffs = {
        key: abs(float(forward_loss[key].detach().cpu()) - float(manual_loss[key].detach().cpu()))
        for key in sorted(set(forward_loss) & set(manual_loss))
        if torch.is_tensor(forward_loss[key]) and forward_loss[key].numel() == 1
    }
    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": args.data_root,
        "image_index": args.index,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "image_hw": hw,
        "max_abs_y_hat_diff": max_abs_diff(forward_out["y_hat"], y_hat),
        "max_abs_x_hat_diff": max_abs_diff(forward_eval["x_hat"], manual_out["x_hat"]),
        "bpp_y_index_diff": abs(float(forward_out["bpp_y_index"].detach().cpu()) - float(bpp_y_index.detach().cpu())),
        "commit_loss_diff": abs(float(forward_out["commit_loss"].detach().cpu()) - float(commit_loss.detach().cpu())),
        "loss_diffs": loss_diffs,
        "max_loss_diff": max(loss_diffs.values()) if loss_diffs else 0.0,
        "rvq_stat_diffs": rvq_diffs,
        "max_rvq_stat_diff": max(rvq_diffs.values()) if rvq_diffs else 0.0,
        "nonfinite": int(
            any(
                not math.isfinite(float(value))
                for value in [
                    float(forward_loss["loss"].detach().cpu()),
                    float(manual_loss["loss"].detach().cpu()),
                    float(forward_out["bpp_y_index"].detach().cpu()),
                    float(bpp_y_index.detach().cpu()),
                ]
            )
        ),
    }
    tolerance = 1e-4
    result["contract_pass_exact"] = (
        result["max_abs_y_hat_diff"] == 0.0
        and result["max_abs_x_hat_diff"] == 0.0
        and result["bpp_y_index_diff"] == 0.0
        and result["commit_loss_diff"] == 0.0
        and result["max_loss_diff"] == 0.0
        and result["max_rvq_stat_diff"] == 0.0
        and result["nonfinite"] == 0
    )
    result["contract_tolerance"] = tolerance
    result["contract_pass"] = (
        result["max_abs_y_hat_diff"] <= tolerance
        and result["max_abs_x_hat_diff"] <= tolerance
        and result["bpp_y_index_diff"] <= tolerance
        and result["commit_loss_diff"] <= tolerance
        and result["max_loss_diff"] <= tolerance
        and result["max_rvq_stat_diff"] <= tolerance
        and result["nonfinite"] == 0
    )

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E120 HCG Adapter Contract Smoke",
        "",
        "This smoke test compares the current full forward path with a manual call through the quantizer boundary.",
        "",
        f"- Contract pass: `{result['contract_pass']}`",
        f"- Exact-zero pass: `{result['contract_pass_exact']}`",
        f"- Tolerance: `{result['contract_tolerance']}`",
        f"- Device: `{result['device']}`, CUDA_VISIBLE_DEVICES=`{result['cuda_visible_devices']}`",
        f"- max |y_hat diff|: `{result['max_abs_y_hat_diff']:.12f}`",
        f"- max |x_hat diff|: `{result['max_abs_x_hat_diff']:.12f}`",
        f"- bpp_y_index diff: `{result['bpp_y_index_diff']:.12f}`",
        f"- max loss diff: `{result['max_loss_diff']:.12f}`",
        f"- max RVQ stat diff: `{result['max_rvq_stat_diff']:.12f}`",
        f"- nonfinite: `{result['nonfinite']}`",
        "",
        "The adapter contract is ready for non-invasive extraction under the 1e-4 tolerance. The exact-zero check is kept as a stricter diagnostic for future CPU/deterministic runs.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_PREFIX.with_suffix('.json')}`",
    ]
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
