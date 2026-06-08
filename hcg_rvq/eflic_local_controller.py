"""Decoder-safe EF-LIC local HCG family controller utilities.

This module is intentionally small and independent from the EF-LIC third-party
package. EF-LIC integration code can call `build_local_context_maps` after
`_mean_scale(support_buf, i)` and feed the result to `LocalHCGFamilyHead`.

The first target is a conservative confidence-gated controller:

* class 0 is always the zero/fallback state,
* nonzero classes correspond to E236 local HCG geometry families,
* losses can penalize false-positive nonzero activation more than missed gains.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hcg_rvq.reliability_index_controller import (
    SpatialReliabilityIndexConfig,
    SpatialReliabilityIndexHead,
    controller_gate,
)

FAMILY_NAMES: tuple[str, ...] = (
    "zero",
    "constant",
    "guarded_constant",
    "guarded_support",
    "soft_blend",
    "sparse_union",
    "hybrid",
)

FAMILY_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(FAMILY_NAMES)}


@dataclass(frozen=True)
class LocalHCGHeadConfig:
    input_channels: int = 11
    hidden_channels: int = 48
    num_families: int = len(FAMILY_NAMES)
    zero_bias: float = 2.0


@dataclass(frozen=True)
class EFLICHCGBranchControllerConfig:
    """Conservative EF-LIC HCG branch-controller defaults.

    The controller is designed to be inserted immediately after EF-LIC computes
    `mean, scale = _mean_scale(support_buf, slice_id)`. It only consumes
    decoder-available context maps and therefore does not add hidden side
    information. Newly inserted heads are biased toward fallback.
    """

    input_channels: int = 11
    hidden_channels: int = 32
    max_alpha: float = 0.02
    zero_bias: float = -6.0
    risk_bias: float = 0.0
    strength_bias: float = -4.0
    support_ratio_channel: int = 5
    prev_ratio_channel: int = 7


def _reduce_rms(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
    return torch.sqrt(x.float().square().mean(dim=1, keepdim=True).clamp_min(0.0)).to(dtype=x.dtype)


def _reduce_abs_mean(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
    return x.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype)


def build_local_context_maps(
    support_buf: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    slice_id: int,
    *,
    group_channels: int,
    num_slices: int = 4,
) -> torch.Tensor:
    """Build decoder-reproducible local maps for the HCG family head.

    All inputs are available on both encoder and decoder immediately after
    EF-LIC computes `mean, scale = _mean_scale(support_buf, slice_id)`.
    The output has 11 channels by default:

    1. abs-mean of current mean map
    2. RMS of current mean map
    3. abs-mean of current scale map
    4. RMS of current scale map
    5. RMS of known support buffer
    6. support/scale RMS ratio
    7. RMS of previously decoded y slices
    8. previous/scale RMS ratio
    9-11. first three slice-id bits as constant maps

    The slice bits avoid a variable one-hot channel count while preserving a
    deterministic slice cue for the small convolutional head.
    """

    if slice_id < 0 or slice_id >= num_slices:
        raise ValueError(f"slice_id must be in [0, {num_slices}), got {slice_id}")
    c = int(group_channels)
    if c <= 0:
        raise ValueError("group_channels must be positive")

    known = support_buf[:, : (num_slices + slice_id) * c]
    prev = support_buf[:, num_slices * c : (num_slices + slice_id) * c]
    scale_rms = _reduce_rms(scale).clamp_min(1e-6)
    support_rms = _reduce_rms(known)
    prev_rms = _reduce_rms(prev)

    b, _, h, w = mean.shape
    slice_bits = []
    for bit in range(3):
        value = float((int(slice_id) >> bit) & 1)
        slice_bits.append(mean.new_full((b, 1, h, w), value))

    return torch.cat(
        [
            _reduce_abs_mean(mean),
            _reduce_rms(mean),
            _reduce_abs_mean(scale),
            scale_rms,
            support_rms,
            support_rms / scale_rms,
            prev_rms,
            prev_rms / scale_rms,
            *slice_bits,
        ],
        dim=1,
    )


@dataclass(frozen=True)
class LocalHCGActivationConfig:
    input_channels: int = 11
    hidden_channels: int = 32
    zero_bias: float = -2.0


class LocalHCGActivationHead(nn.Module):
    """Binary zero-vs-active reliability head for local HCG geometry.

    This head is intentionally separated from family prediction. E241/E242 show
    that activation calibration is the fragile part; a newly inserted head is
    initialized toward zero/fallback by using a negative final bias.
    """

    def __init__(self, config: LocalHCGActivationConfig | None = None):
        super().__init__()
        self.config = config or LocalHCGActivationConfig()
        self.net = nn.Sequential(
            nn.Conv2d(self.config.input_channels, self.config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_channels, self.config.hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_channels, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Conv2d) and last.bias is not None:
            nn.init.constant_(last.bias, float(self.config.zero_bias))

    def forward(self, context_maps: torch.Tensor) -> torch.Tensor:
        return self.net(context_maps)


class EFLICHCGBranchController(nn.Module):
    """Decoder-safe spatial fallback controller for EF-LIC HCG geometry.

    The module consumes `build_local_context_maps(...)` outputs. It returns an
    alpha map that can be passed to the projected-HCG EF-LIC RVQ path. The
    `force_zero` path is intentionally exact: it returns an all-zero alpha map
    and therefore recovers the original EF-LIC quantizer path.
    """

    def __init__(self, config: EFLICHCGBranchControllerConfig | None = None):
        super().__init__()
        self.config = config or EFLICHCGBranchControllerConfig()
        rel_cfg = SpatialReliabilityIndexConfig(
            input_channels=self.config.input_channels,
            hidden_channels=self.config.hidden_channels,
            zero_bias=self.config.zero_bias,
            risk_bias=self.config.risk_bias,
        )
        self.reliability = SpatialReliabilityIndexHead(rel_cfg)
        self.family_head = LocalHCGFamilyHead(
            LocalHCGHeadConfig(
                input_channels=self.config.input_channels,
                hidden_channels=self.config.hidden_channels,
                num_families=len(FAMILY_NAMES),
                zero_bias=2.0,
            )
        )
        self.strength_head = nn.Sequential(
            nn.Conv2d(self.config.input_channels, self.config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_channels, 1, kernel_size=1),
        )
        self.reset_strength_parameters()

    def reset_strength_parameters(self) -> None:
        for module in self.strength_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.strength_head[-1]
        if isinstance(last, nn.Conv2d) and last.bias is not None:
            nn.init.constant_(last.bias, float(self.config.strength_bias))

    def _normalized_local_score(self, context_maps: torch.Tensor) -> torch.Tensor:
        channels = context_maps.shape[1]
        candidates = []
        for idx in (self.config.support_ratio_channel, self.config.prev_ratio_channel):
            if 0 <= idx < channels:
                candidates.append(context_maps[:, idx : idx + 1].float().clamp_min(0.0))
        if not candidates:
            score = context_maps.new_ones((context_maps.shape[0], 1, context_maps.shape[2], context_maps.shape[3])).float()
        else:
            score = torch.stack(candidates, dim=0).amax(dim=0)
        denom = score.flatten(1).mean(dim=1).view(score.shape[0], 1, 1, 1).clamp_min(1e-6)
        return (score / (2.0 * denom)).clamp(0.0, 1.0).to(dtype=context_maps.dtype)

    def forward(
        self,
        context_maps: torch.Tensor,
        *,
        hard: bool = False,
        force_zero: bool = False,
        active_threshold: float = 0.5,
        max_risk: float = 0.0,
        risk_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        reliability = self.reliability(context_maps)
        family_logits = self.family_head(context_maps)
        if force_zero:
            gate = torch.zeros_like(reliability["active_logit"])
        else:
            gate = controller_gate(
                reliability["active_logit"],
                risk_score=reliability["risk_score"],
                active_threshold=active_threshold,
                max_risk=max_risk,
                risk_temperature=risk_temperature,
                hard=hard,
            )
        strength = torch.sigmoid(self.strength_head(context_maps)) * float(self.config.max_alpha)
        local_score = self._normalized_local_score(context_maps)
        alpha_map = gate.to(dtype=context_maps.dtype) * strength.to(dtype=context_maps.dtype) * local_score
        return {
            "alpha_map": alpha_map,
            "gate": gate,
            "strength": strength,
            "local_score": local_score,
            "active_logit": reliability["active_logit"],
            "risk_score": reliability["risk_score"],
            "family_logits": family_logits,
        }


def binary_activation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    false_positive_weight: float = 4.0,
    missed_active_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted binary loss for zero-vs-active HCG reliability.

    `target` may be `[B, H, W]` or `[B, 1, H, W]`; nonzero values mean active.
    False-positive active predictions on zero regions are weighted separately
    from missed active regions so calibration can match E238's asymmetric risk.
    """

    if target.ndim == 3:
        target = target[:, None]
    if target.shape != logits.shape:
        raise ValueError(f"target shape {tuple(target.shape)} is incompatible with logits {tuple(logits.shape)}")
    target = (target > 0).to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(
        target > 0.5,
        torch.full_like(loss, float(missed_active_weight)),
        torch.full_like(loss, float(false_positive_weight)),
    )
    if sample_weight is not None:
        sw = sample_weight.to(device=logits.device, dtype=logits.dtype)
        if sw.ndim == 1:
            sw = sw[:, None, None, None].expand_as(loss)
        elif sw.ndim == 3:
            sw = sw[:, None]
        weight = weight * sw
    return (loss * weight).mean()


class LocalHCGFamilyHead(nn.Module):
    """Small local family/strength head for EF-LIC HCG-RVQ integration.

    The last bias is initialized toward the zero/fallback class. With the
    default initialization, a newly inserted head should be conservative before
    supervised or codec-aware training.
    """

    def __init__(self, config: LocalHCGHeadConfig | None = None):
        super().__init__()
        self.config = config or LocalHCGHeadConfig()
        self.net = nn.Sequential(
            nn.Conv2d(self.config.input_channels, self.config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_channels, self.config.hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_channels, self.config.num_families, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Conv2d) and last.bias is not None:
            nn.init.zeros_(last.bias)
            last.bias.data[FAMILY_TO_INDEX["zero"]] = float(self.config.zero_bias)

    def forward(self, context_maps: torch.Tensor) -> torch.Tensor:
        return self.net(context_maps)


def asymmetric_family_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    false_positive_weight: float = 4.0,
    missed_active_weight: float = 1.0,
    cost_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy plus HCG-specific activation penalties.

    `target` may be `[B]`, `[B, H, W]`, or `[B, 1, H, W]`. Image-level labels are
    broadcast over spatial positions. Class 0 is interpreted as zero/fallback.
    """

    if target.ndim == 1:
        target = target[:, None, None].expand(logits.shape[0], logits.shape[2], logits.shape[3])
    elif target.ndim == 4:
        target = target.squeeze(1)
    if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError(f"target shape {tuple(target.shape)} is incompatible with logits {tuple(logits.shape)}")
    target = target.long()

    loss = F.cross_entropy(logits, target, reduction="none")
    probs = logits.softmax(dim=1)
    zero_idx = FAMILY_TO_INDEX["zero"]
    zero_target = target == zero_idx
    active_target = ~zero_target

    nonzero_prob = 1.0 - probs[:, zero_idx]
    zero_prob = probs[:, zero_idx]
    penalty = torch.zeros_like(loss)
    penalty = penalty + zero_target.float() * nonzero_prob * float(false_positive_weight)
    penalty = penalty + active_target.float() * zero_prob * float(missed_active_weight)

    if cost_matrix is not None:
        cm = cost_matrix.to(device=logits.device, dtype=logits.dtype)
        if cm.shape != (logits.shape[1], logits.shape[1]):
            raise ValueError(f"cost_matrix shape {tuple(cm.shape)} does not match {logits.shape[1]} classes")
        expected_cost = (probs * cm[target].permute(0, 3, 1, 2)).sum(dim=1)
        penalty = penalty + expected_cost

    total = loss + penalty
    if sample_weight is not None:
        weight = sample_weight.to(device=logits.device, dtype=logits.dtype)
        if weight.ndim == 1:
            weight = weight[:, None, None].expand_as(total)
        elif weight.ndim == 4:
            weight = weight.squeeze(1)
        total = total * weight
    return total.mean()
