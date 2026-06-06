#!/usr/bin/env python3
"""E262 controller fallback-mix smoke.

After E261, the next full-training candidate is not an offline threshold but a
codec-loop reliability/index controller with no-branch fallback.  This smoke
verifies the shared insertion primitive before touching EF-LIC or GLC internals:
soft gates are differentiable for training, hard gates can exactly recover the
original codec path, and conservative initialization keeps activation small.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.reliability_index_controller import (  # noqa: E402
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    SpatialReliabilityIndexConfig,
    SpatialReliabilityIndexHead,
    mix_with_fallback,
    reliability_index_loss,
    select_with_fallback,
)


def finite(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def tensor_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def main() -> None:
    torch.manual_seed(262)
    out_prefix = ROOT / "experiments" / "analysis" / "e262_controller_fallback_mix_smoke"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Spatial EF-LIC-style insertion: [B, C, H, W] branch/base tensors.
    spatial = SpatialReliabilityIndexHead(SpatialReliabilityIndexConfig(input_channels=7, hidden_channels=8))
    context = torch.randn(2, 7, 4, 5, requires_grad=True)
    base = torch.randn(2, 3, 4, 5)
    branch = (base + 0.05 * torch.randn_like(base)).detach().requires_grad_(True)
    target = base + 0.02 * torch.randn_like(base)

    spatial_out = spatial(context)
    soft_mixed, soft_gate = mix_with_fallback(
        base,
        branch,
        spatial_out["active_logit"],
        risk_score=spatial_out["risk_score"],
        risk_temperature=0.5,
        hard=False,
    )
    hard_mixed, hard_gate = mix_with_fallback(
        base,
        branch,
        torch.full_like(spatial_out["active_logit"], -12.0),
        risk_score=torch.ones_like(spatial_out["risk_score"]),
        hard=True,
    )
    hard_reconstructs_base = bool(torch.allclose(hard_mixed, base, atol=0.0, rtol=0.0))

    target_active = torch.zeros_like(spatial_out["active_logit"])
    target_score = torch.zeros_like(spatial_out["risk_score"])
    loss = F.mse_loss(soft_mixed, target) + reliability_index_loss(
        spatial_out["active_logit"],
        target_active,
        risk_score=spatial_out["risk_score"],
        target_score=target_score,
        false_positive_weight=4.0,
        score_weight=0.1,
    )
    loss.backward()
    spatial_grads_finite = all(p.grad is None or tensor_finite(p.grad) for p in spatial.parameters())

    # Per-image GLC-style insertion: [B, F] diagnostics controlling [B, D].
    mlp = ReliabilityIndexMLP(ReliabilityIndexMLPConfig(input_dim=5, hidden_dim=8))
    features = torch.randn(3, 5, requires_grad=True)
    base_vec = torch.randn(3, 12)
    branch_vec = (base_vec + 0.03 * torch.randn_like(base_vec)).detach().requires_grad_(True)
    mlp_out = mlp(features)
    vec_mixed, vec_gate = mix_with_fallback(base_vec, branch_vec, mlp_out["active_logit"], risk_score=mlp_out["risk_score"])
    vec_loss = vec_mixed.pow(2).mean() + reliability_index_loss(
        mlp_out["active_logit"],
        torch.zeros_like(mlp_out["active_logit"]),
        risk_score=mlp_out["risk_score"],
        target_score=torch.zeros_like(mlp_out["risk_score"]),
        score_weight=0.1,
    )
    vec_loss.backward()
    mlp_grads_finite = all(p.grad is None or tensor_finite(p.grad) for p in mlp.parameters())

    hard_mask = select_with_fallback(
        torch.full((3, 1), -12.0),
        risk_score=torch.ones(3, 1),
        active_threshold=0.5,
        max_risk=0.0,
    )

    summary = {
        "spatial_soft_gate_mean": float(soft_gate.detach().mean().item()),
        "spatial_soft_gate_max": float(soft_gate.detach().max().item()),
        "spatial_soft_delta_abs_mean": float((soft_mixed - base).detach().abs().mean().item()),
        "spatial_hard_gate_sum": float(hard_gate.detach().sum().item()),
        "spatial_hard_reconstructs_base": hard_reconstructs_base,
        "spatial_loss_finite": tensor_finite(loss.detach()),
        "spatial_grads_finite": spatial_grads_finite,
        "mlp_soft_gate_mean": float(vec_gate.detach().mean().item()),
        "mlp_soft_gate_max": float(vec_gate.detach().max().item()),
        "mlp_soft_delta_abs_mean": float((vec_mixed - base_vec).detach().abs().mean().item()),
        "mlp_loss_finite": tensor_finite(vec_loss.detach()),
        "mlp_grads_finite": mlp_grads_finite,
        "hard_mask_selected": int(hard_mask.sum().item()),
    }
    summary["all_checks_passed"] = all(
        [
            summary["spatial_hard_reconstructs_base"],
            summary["spatial_loss_finite"],
            summary["spatial_grads_finite"],
            summary["mlp_loss_finite"],
            summary["mlp_grads_finite"],
            summary["hard_mask_selected"] == 0,
        ]
    ) and all(finite(v) for k, v in summary.items() if k != "all_checks_passed")

    out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E262 Controller Fallback-Mix Smoke",
        "",
        "This smoke verifies the shared codec-loop insertion primitive after E261: soft gate for training, hard no-branch fallback for evaluation, and finite gradients.",
        "",
        "| check | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{float(value):.6f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    out_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_prefix": str(out_prefix), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
