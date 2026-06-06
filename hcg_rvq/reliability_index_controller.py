"""Small reliability/index controllers for HCG-RVQ branches.

These heads are intentionally compact.  E246-E259 indicate that the useful
signal is local but fragile: the branch should be allowed to stay silent unless
its predicted perceptual/rate risk is worth paying.  The module is independent
from EF-LIC/GLC internals so both backbones can share the same controller
contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilityIndexMLPConfig:
    input_dim: int
    hidden_dim: int = 32
    zero_bias: float = -2.0
    risk_bias: float = 0.0


@dataclass(frozen=True)
class SpatialReliabilityIndexConfig:
    input_channels: int
    hidden_channels: int = 32
    zero_bias: float = -2.0
    risk_bias: float = 0.0


class ReliabilityIndexMLP(nn.Module):
    """Per-image reliability/index-risk controller.

    The first output is an active logit.  The second output is a signed branch
    risk/score prediction where lower is better.  A negative active bias makes a
    newly inserted controller conservative before calibration.
    """

    def __init__(self, config: ReliabilityIndexMLPConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            last.bias.data[0] = float(self.config.zero_bias)
            last.bias.data[1] = float(self.config.risk_bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"ReliabilityIndexMLP expects [B, F], got {tuple(features.shape)}")
        out = self.net(features)
        return {"active_logit": out[:, :1], "risk_score": out[:, 1:2]}


class SpatialReliabilityIndexHead(nn.Module):
    """Decoder-safe spatial reliability/index-risk controller for LIC slices."""

    def __init__(self, config: SpatialReliabilityIndexConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.hidden_channels, 2, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            last.bias.data[0] = float(self.config.zero_bias)
            last.bias.data[1] = float(self.config.risk_bias)

    def forward(self, context_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        if context_maps.ndim != 4:
            raise ValueError(f"SpatialReliabilityIndexHead expects [B, C, H, W], got {tuple(context_maps.shape)}")
        out = self.net(context_maps)
        return {"active_logit": out[:, :1], "risk_score": out[:, 1:2]}


def reliability_index_loss(
    active_logit: torch.Tensor,
    target_active: torch.Tensor,
    *,
    risk_score: torch.Tensor | None = None,
    target_score: torch.Tensor | None = None,
    false_positive_weight: float = 4.0,
    missed_active_weight: float = 1.0,
    score_weight: float = 0.25,
) -> torch.Tensor:
    """Asymmetric activation loss plus optional signed risk regression.

    `target_active=1` means the HCG-RVQ branch should be used.  False-positive
    branch activation is weighted more heavily because E246-E259 show that
    all-on activation is the dominant failure mode.  `target_score`, when
    provided, should be the actual guarded branch score where lower is better.
    """

    if target_active.shape != active_logit.shape:
        if target_active.ndim == active_logit.ndim - 1:
            target_active = target_active.unsqueeze(1)
        if target_active.shape != active_logit.shape:
            raise ValueError(
                f"target_active shape {tuple(target_active.shape)} is incompatible with "
                f"active_logit {tuple(active_logit.shape)}"
            )
    target = (target_active > 0).to(device=active_logit.device, dtype=active_logit.dtype)
    bce = F.binary_cross_entropy_with_logits(active_logit, target, reduction="none")
    weight = torch.where(
        target > 0.5,
        torch.full_like(bce, float(missed_active_weight)),
        torch.full_like(bce, float(false_positive_weight)),
    )
    loss = (bce * weight).mean()

    if risk_score is not None and target_score is not None:
        if target_score.shape != risk_score.shape:
            if target_score.ndim == risk_score.ndim - 1:
                target_score = target_score.unsqueeze(1)
            if target_score.shape != risk_score.shape:
                raise ValueError(
                    f"target_score shape {tuple(target_score.shape)} is incompatible with "
                    f"risk_score {tuple(risk_score.shape)}"
                )
        target_score = target_score.to(device=risk_score.device, dtype=risk_score.dtype)
        loss = loss + float(score_weight) * F.smooth_l1_loss(risk_score, target_score)
    return loss


def select_with_fallback(
    active_logit: torch.Tensor,
    *,
    risk_score: torch.Tensor | None = None,
    active_threshold: float = 0.5,
    max_risk: float = 0.0,
) -> torch.Tensor:
    """Return a boolean branch-use mask with explicit fallback."""

    active = torch.sigmoid(active_logit) >= float(active_threshold)
    if risk_score is not None:
        active = active & (risk_score <= float(max_risk))
    return active



def controller_gate(
    active_logit: torch.Tensor,
    *,
    risk_score: torch.Tensor | None = None,
    active_threshold: float = 0.5,
    max_risk: float = 0.0,
    risk_temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """Return a soft or hard branch gate with explicit risk fallback.

    Soft mode is used during training so the controller and branch receive
    gradients.  Hard mode is used for deterministic evaluation/bit accounting.
    """

    if hard:
        return select_with_fallback(
            active_logit,
            risk_score=risk_score,
            active_threshold=active_threshold,
            max_risk=max_risk,
        ).to(dtype=active_logit.dtype)

    gate = torch.sigmoid(active_logit)
    if risk_score is not None:
        temperature = max(float(risk_temperature), 1e-6)
        gate = gate * torch.sigmoid((float(max_risk) - risk_score) / temperature)
    return gate


def _broadcast_gate(gate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if gate.shape[0] != target.shape[0]:
        raise ValueError(f"gate batch {gate.shape[0]} does not match target batch {target.shape[0]}")
    while gate.ndim < target.ndim:
        gate = gate.unsqueeze(-1)
    return gate.to(device=target.device, dtype=target.dtype)


def mix_with_fallback(
    base: torch.Tensor,
    branch: torch.Tensor,
    active_logit: torch.Tensor,
    *,
    risk_score: torch.Tensor | None = None,
    active_threshold: float = 0.5,
    max_risk: float = 0.0,
    risk_temperature: float = 1.0,
    hard: bool = False,
    max_gate: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend branch output into base output through a conservative fallback gate.

    `base` is the original codec path.  `branch` is the HCG-RVQ candidate path.
    The returned gate is broadcastable to `base`/`branch`; zero exactly recovers
    the original codec.  This is the shared insertion primitive for EF-LIC/GLC
    codec-loop pilots after E261.
    """

    if base.shape != branch.shape:
        raise ValueError(f"base shape {tuple(base.shape)} must match branch shape {tuple(branch.shape)}")
    gate = controller_gate(
        active_logit,
        risk_score=risk_score,
        active_threshold=active_threshold,
        max_risk=max_risk,
        risk_temperature=risk_temperature,
        hard=hard,
    ).clamp(0.0, float(max_gate))
    gate = _broadcast_gate(gate, base)
    mixed = base + gate * (branch - base)
    return mixed, gate
