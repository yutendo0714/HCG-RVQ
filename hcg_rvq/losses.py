from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RateDistortionLoss(nn.Module):
    """Rate-distortion objective with optional RVQ commitment and gate penalties."""

    def __init__(
        self,
        lambda_rd: float,
        beta_commit: float = 0.25,
        rho_gate: float = 0.0,
        rho_mu_q_abs: float = 0.0,
        rho_s_q_std: float = 0.0,
        rho_householder_delta: float = 0.0,
        rho_householder_delta_target: float = 0.0,
        householder_delta_target: float = 0.0,
        rho_householder_delta_local_cap: float = 0.0,
        householder_delta_local_cap: float = 0.0,
        householder_delta_local_cap_use_risk: bool = True,
        householder_delta_local_cap_detach_risk: bool = True,
        householder_delta_local_cap_risk_power: float = 1.0,
        householder_delta_local_cap_risk_offset: float = 0.0,
        householder_delta_local_cap_risk_normalize: bool = False,
        rho_householder_delta_image_tail: float = 0.0,
        householder_delta_image_tail_threshold: float = 0.047937,
        householder_delta_image_tail_detach: bool = False,
        rho_householder_gate_raw_tail: float = 0.0,
        householder_gate_raw_tail_threshold: float = 0.284059,
        householder_gate_raw_tail_use_image_mean: bool = True,
        householder_gate_raw_tail_detach: bool = False,
        rho_householder_reliability_teacher: float = 0.0,
        householder_reliability_teacher_min: float = 0.5,
        householder_reliability_teacher_target: str = "householder_reliability_keep",
        householder_reliability_teacher_weight: str | None = None,
        householder_reliability_teacher_use_image_mean: bool = True,
        householder_reliability_teacher_local_weight_source: str | None = None,
        householder_reliability_teacher_local_weight_threshold: float = 0.0,
        householder_reliability_teacher_local_weight_sharpness: float = 1.0,
        householder_reliability_teacher_local_weight_high: bool = True,
        householder_reliability_teacher_local_weight_min: float = 0.0,
        householder_reliability_teacher_local_weight_detach: bool = True,
        householder_reliability_teacher_local_target_source: str | None = None,
        householder_reliability_teacher_local_target_threshold: float = 0.0,
        householder_reliability_teacher_local_target_sharpness: float = 1.0,
        householder_reliability_teacher_local_target_high: bool = True,
        householder_reliability_teacher_local_target_keep_value: float = 1.0,
        householder_reliability_teacher_local_target_detach: bool = True,
        rho_householder_residual_selector_teacher: float = 0.0,
        householder_residual_selector_teacher_target: str = "e088_decoder_safe_selected_previous_local",
        householder_residual_selector_teacher_weight: str | None = None,
        householder_residual_selector_teacher_use_image_mean: bool = True,
        householder_residual_selector_teacher_local_weight_source: str | None = None,
        householder_residual_selector_teacher_local_weight_threshold: float = 0.0,
        householder_residual_selector_teacher_local_weight_sharpness: float = 1.0,
        householder_residual_selector_teacher_local_weight_high: bool = True,
        householder_residual_selector_teacher_local_weight_min: float = 0.0,
        householder_residual_selector_teacher_local_weight_detach: bool = True,
        rho_householder_residual_selector_noop: float = 0.0,
        householder_residual_selector_noop_weight_source: str | None = None,
        householder_residual_selector_noop_weight_threshold: float = 0.0,
        householder_residual_selector_noop_weight_sharpness: float = 1.0,
        householder_residual_selector_noop_weight_high: bool = False,
        householder_residual_selector_noop_weight_min: float = 0.0,
        householder_residual_selector_noop_weight_detach: bool = True,
        rho_anchor_mu: float = 0.0,
        rho_anchor_log_s: float = 0.0,
        rho_anchor_u: float = 0.0,
        rho_anchor_y_hat: float = 0.0,
        rho_anchor_selected_distortion_margin: float = 0.0,
        anchor_selected_distortion_target: str = "e088_decoder_safe_selected_previous_local",
        anchor_selected_distortion_weight: str | None = None,
        anchor_selected_distortion_margin: float = 0.0,
        anchor_selected_distortion_keep_margin: float = 0.0,
        anchor_selected_distortion_keep_weight: float = 0.0,
        anchor_selected_distortion_squared: bool = True,
        mse_scale: float = 255.0 * 255.0,
    ) -> None:
        super().__init__()
        self.lambda_rd = lambda_rd
        self.beta_commit = beta_commit
        self.rho_gate = rho_gate
        self.rho_mu_q_abs = rho_mu_q_abs
        self.rho_s_q_std = rho_s_q_std
        self.rho_householder_delta = rho_householder_delta
        self.rho_householder_delta_target = rho_householder_delta_target
        self.householder_delta_target = householder_delta_target
        self.rho_householder_delta_local_cap = rho_householder_delta_local_cap
        self.householder_delta_local_cap = householder_delta_local_cap
        self.householder_delta_local_cap_use_risk = householder_delta_local_cap_use_risk
        self.householder_delta_local_cap_detach_risk = householder_delta_local_cap_detach_risk
        self.householder_delta_local_cap_risk_power = householder_delta_local_cap_risk_power
        self.householder_delta_local_cap_risk_offset = householder_delta_local_cap_risk_offset
        self.householder_delta_local_cap_risk_normalize = householder_delta_local_cap_risk_normalize
        self.rho_householder_delta_image_tail = rho_householder_delta_image_tail
        self.householder_delta_image_tail_threshold = householder_delta_image_tail_threshold
        self.householder_delta_image_tail_detach = householder_delta_image_tail_detach
        self.rho_householder_gate_raw_tail = rho_householder_gate_raw_tail
        self.householder_gate_raw_tail_threshold = householder_gate_raw_tail_threshold
        self.householder_gate_raw_tail_use_image_mean = householder_gate_raw_tail_use_image_mean
        self.householder_gate_raw_tail_detach = householder_gate_raw_tail_detach
        self.rho_householder_reliability_teacher = rho_householder_reliability_teacher
        self.householder_reliability_teacher_min = householder_reliability_teacher_min
        self.householder_reliability_teacher_target = householder_reliability_teacher_target
        self.householder_reliability_teacher_weight = householder_reliability_teacher_weight
        self.householder_reliability_teacher_use_image_mean = householder_reliability_teacher_use_image_mean
        self.householder_reliability_teacher_local_weight_source = householder_reliability_teacher_local_weight_source
        self.householder_reliability_teacher_local_weight_threshold = householder_reliability_teacher_local_weight_threshold
        self.householder_reliability_teacher_local_weight_sharpness = householder_reliability_teacher_local_weight_sharpness
        self.householder_reliability_teacher_local_weight_high = householder_reliability_teacher_local_weight_high
        self.householder_reliability_teacher_local_weight_min = householder_reliability_teacher_local_weight_min
        self.householder_reliability_teacher_local_weight_detach = householder_reliability_teacher_local_weight_detach
        self.householder_reliability_teacher_local_target_source = householder_reliability_teacher_local_target_source
        self.householder_reliability_teacher_local_target_threshold = householder_reliability_teacher_local_target_threshold
        self.householder_reliability_teacher_local_target_sharpness = householder_reliability_teacher_local_target_sharpness
        self.householder_reliability_teacher_local_target_high = householder_reliability_teacher_local_target_high
        self.householder_reliability_teacher_local_target_keep_value = householder_reliability_teacher_local_target_keep_value
        self.householder_reliability_teacher_local_target_detach = householder_reliability_teacher_local_target_detach
        self.rho_householder_residual_selector_teacher = rho_householder_residual_selector_teacher
        self.householder_residual_selector_teacher_target = householder_residual_selector_teacher_target
        self.householder_residual_selector_teacher_weight = householder_residual_selector_teacher_weight
        self.householder_residual_selector_teacher_use_image_mean = householder_residual_selector_teacher_use_image_mean
        self.householder_residual_selector_teacher_local_weight_source = householder_residual_selector_teacher_local_weight_source
        self.householder_residual_selector_teacher_local_weight_threshold = householder_residual_selector_teacher_local_weight_threshold
        self.householder_residual_selector_teacher_local_weight_sharpness = householder_residual_selector_teacher_local_weight_sharpness
        self.householder_residual_selector_teacher_local_weight_high = householder_residual_selector_teacher_local_weight_high
        self.householder_residual_selector_teacher_local_weight_min = householder_residual_selector_teacher_local_weight_min
        self.householder_residual_selector_teacher_local_weight_detach = householder_residual_selector_teacher_local_weight_detach
        self.rho_householder_residual_selector_noop = rho_householder_residual_selector_noop
        self.householder_residual_selector_noop_weight_source = householder_residual_selector_noop_weight_source
        self.householder_residual_selector_noop_weight_threshold = householder_residual_selector_noop_weight_threshold
        self.householder_residual_selector_noop_weight_sharpness = householder_residual_selector_noop_weight_sharpness
        self.householder_residual_selector_noop_weight_high = householder_residual_selector_noop_weight_high
        self.householder_residual_selector_noop_weight_min = householder_residual_selector_noop_weight_min
        self.householder_residual_selector_noop_weight_detach = householder_residual_selector_noop_weight_detach
        self.rho_anchor_mu = rho_anchor_mu
        self.rho_anchor_log_s = rho_anchor_log_s
        self.rho_anchor_u = rho_anchor_u
        self.rho_anchor_y_hat = rho_anchor_y_hat
        self.rho_anchor_selected_distortion_margin = rho_anchor_selected_distortion_margin
        self.anchor_selected_distortion_target = anchor_selected_distortion_target
        self.anchor_selected_distortion_weight = anchor_selected_distortion_weight
        self.anchor_selected_distortion_margin = anchor_selected_distortion_margin
        self.anchor_selected_distortion_keep_margin = anchor_selected_distortion_keep_margin
        self.anchor_selected_distortion_keep_weight = anchor_selected_distortion_keep_weight
        self.anchor_selected_distortion_squared = anchor_selected_distortion_squared
        self.mse_scale = mse_scale

    @staticmethod
    def likelihood_bpp(likelihood: torch.Tensor, num_pixels: int) -> torch.Tensor:
        return torch.log(likelihood).sum() / (-math.log(2.0) * num_pixels)

    def forward(self, output: dict[str, object], target: torch.Tensor) -> dict[str, torch.Tensor]:
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        likelihoods = output["likelihoods"]
        bpp_z = self.likelihood_bpp(likelihoods["z"], num_pixels)

        if "y" in likelihoods:
            bpp_y = self.likelihood_bpp(likelihoods["y"], num_pixels)
        else:
            bpp_y = output["bpp_y_index"]

        bpp_total = bpp_y + bpp_z
        mse = F.mse_loss(output["x_hat"].clamp(0, 1), target)
        commit_loss = output.get("commit_loss", target.new_tensor(0.0))
        rvq_stats = output.get("rvq_stats", {})
        gate_loss = rvq_stats.get("avg_gate", target.new_tensor(0.0))
        conditioning_loss = target.new_tensor(0.0)
        if self.rho_mu_q_abs != 0.0:
            conditioning_loss = conditioning_loss + self.rho_mu_q_abs * rvq_stats.get(
                "mu_q_abs_mean",
                target.new_tensor(0.0),
            )
        if self.rho_s_q_std != 0.0:
            conditioning_loss = conditioning_loss + self.rho_s_q_std * rvq_stats.get(
                "s_q_std",
                target.new_tensor(0.0),
            )
        if self.rho_householder_delta != 0.0:
            conditioning_loss = conditioning_loss + self.rho_householder_delta * rvq_stats.get(
                "householder_delta_rms",
                target.new_tensor(0.0),
            )
        if self.rho_householder_delta_target != 0.0:
            delta_rms = rvq_stats.get("householder_delta_rms")
            if delta_rms is None:
                raise KeyError(
                    "householder_delta_target loss requested, but householder_delta_rms is missing"
                )
            delta_target = target.new_tensor(float(self.householder_delta_target))
            conditioning_loss = conditioning_loss + self.rho_householder_delta_target * (
                delta_rms - delta_target
            ).pow(2)

        anchor_loss = target.new_tensor(0.0)
        teacher_loss = target.new_tensor(0.0)
        residual_selector_teacher_loss = target.new_tensor(0.0)
        residual_selector_noop_loss = target.new_tensor(0.0)
        selected_distortion_margin_loss = target.new_tensor(0.0)
        current_conditioning = output.get("conditioning_tensors", {})
        if self.rho_householder_delta_local_cap != 0.0:
            delta_rms_map = current_conditioning.get("householder_delta_rms_map")
            if delta_rms_map is None:
                raise KeyError(
                    "local Householder delta cap requested, but householder_delta_rms_map is missing"
                )
            cap = target.new_tensor(float(self.householder_delta_local_cap))
            local_excess = F.relu(delta_rms_map - cap).pow(2)
            if self.householder_delta_local_cap_use_risk:
                risk_weight = current_conditioning.get("householder_risk_multiplier")
                if risk_weight is None:
                    raise KeyError(
                        "risk-weighted local Householder delta cap requested, but risk multiplier is missing"
                    )
                if self.householder_delta_local_cap_detach_risk:
                    risk_weight = risk_weight.detach()
                if self.householder_delta_local_cap_risk_offset != 0.0:
                    offset = float(self.householder_delta_local_cap_risk_offset)
                    risk_weight = F.relu(risk_weight - risk_weight.new_tensor(offset))
                    if self.householder_delta_local_cap_risk_normalize:
                        risk_weight = risk_weight / max(1e-6, 1.0 - offset)
                if self.householder_delta_local_cap_risk_power != 1.0:
                    risk_weight = risk_weight.clamp_min(0.0).pow(
                        float(self.householder_delta_local_cap_risk_power)
                    )
                local_excess = local_excess * risk_weight
            conditioning_loss = conditioning_loss + self.rho_householder_delta_local_cap * local_excess.mean()

        if self.rho_householder_delta_image_tail != 0.0:
            delta_rms_map = current_conditioning.get("householder_delta_rms_map")
            if delta_rms_map is None:
                raise KeyError(
                    "image Householder delta tail loss requested, but delta RMS map is missing"
                )
            delta_source = (
                delta_rms_map.detach()
                if self.householder_delta_image_tail_detach
                else delta_rms_map
            )
            image_delta = delta_source.mean(dim=(1, 2, 3, 4), keepdim=True)
            threshold = target.new_tensor(float(self.householder_delta_image_tail_threshold))
            delta_tail_excess = F.relu(image_delta - threshold).pow(2)
            conditioning_loss = (
                conditioning_loss
                + self.rho_householder_delta_image_tail * delta_tail_excess.mean()
            )

        if self.rho_householder_gate_raw_tail != 0.0:
            raw_gate = current_conditioning.get("householder_gate_raw")
            if raw_gate is None:
                raise KeyError(
                    "raw Householder gate tail loss requested, but raw gate tensor is missing"
                )
            gate_source = raw_gate.detach() if self.householder_gate_raw_tail_detach else raw_gate
            if self.householder_gate_raw_tail_use_image_mean:
                gate_source = gate_source.mean(dim=(1, 2, 3, 4), keepdim=True)
            threshold = target.new_tensor(float(self.householder_gate_raw_tail_threshold))
            tail_excess = F.relu(gate_source - threshold).pow(2)
            conditioning_loss = conditioning_loss + self.rho_householder_gate_raw_tail * tail_excess.mean()

        if self.rho_householder_reliability_teacher != 0.0:
            reliability = current_conditioning.get("householder_reliability_multiplier")
            if reliability is None:
                raise KeyError(
                    "reliability teacher loss requested, but reliability multiplier is missing"
                )
            teacher_targets = output.get("teacher_targets", {})
            reliability_target = teacher_targets.get(self.householder_reliability_teacher_target)
            if reliability_target is None:
                raise KeyError(
                    f"reliability teacher loss requested, but target {self.householder_reliability_teacher_target!r} is missing"
                )
            if not torch.is_tensor(reliability_target):
                reliability_target = torch.as_tensor(reliability_target, device=target.device, dtype=target.dtype)
            reliability_target = reliability_target.to(device=target.device, dtype=target.dtype).view(-1).clamp(0.0, 1.0)
            reliability_weight = None
            if self.householder_reliability_teacher_weight:
                reliability_weight = teacher_targets.get(self.householder_reliability_teacher_weight)
                if reliability_weight is None:
                    raise KeyError(
                        f"reliability teacher weight requested, but target {self.householder_reliability_teacher_weight!r} is missing"
                    )
                if not torch.is_tensor(reliability_weight):
                    reliability_weight = torch.as_tensor(reliability_weight, device=target.device, dtype=target.dtype)
                reliability_weight = reliability_weight.to(device=target.device, dtype=target.dtype).view(-1).clamp_min(0.0)
            reliability_source = reliability
            if self.householder_reliability_teacher_use_image_mean:
                reliability_source = reliability_source.mean(dim=(1, 2, 3, 4), keepdim=True)
            source_shape = reliability_source.shape
            reliability_source = reliability_source.view(-1)
            if reliability_target.numel() != reliability_source.numel():
                if (
                    not self.householder_reliability_teacher_use_image_mean
                    and reliability_target.numel() == source_shape[0]
                ):
                    reliability_target = reliability_target.view(
                        source_shape[0], 1, 1, 1, 1
                    ).expand(source_shape).reshape(-1)
                    if reliability_weight is not None:
                        reliability_weight = reliability_weight.view(
                            source_shape[0], 1, 1, 1, 1
                        ).expand(source_shape).reshape(-1)
                else:
                    raise ValueError(
                        "reliability teacher target size does not match reliability source: "
                        f"target={reliability_target.numel()}, source={reliability_source.numel()}"
                    )
            rel_min = target.new_tensor(float(self.householder_reliability_teacher_min))
            rel_prob = ((reliability_source - rel_min) / (1.0 - rel_min).clamp_min(1e-6)).clamp(1e-6, 1.0 - 1e-6)
            if self.householder_reliability_teacher_local_target_source:
                local_target_source = current_conditioning.get(
                    self.householder_reliability_teacher_local_target_source
                )
                if local_target_source is None:
                    raise KeyError(
                        "local reliability teacher target requested, but conditioning tensor "
                        f"{self.householder_reliability_teacher_local_target_source!r} is missing"
                    )
                if local_target_source.shape != source_shape:
                    raise ValueError(
                        "local reliability teacher target shape does not match reliability source: "
                        f"target={tuple(local_target_source.shape)}, source={tuple(source_shape)}"
                    )
                if self.householder_reliability_teacher_local_target_detach:
                    local_target_source = local_target_source.detach()
                threshold = target.new_tensor(
                    float(self.householder_reliability_teacher_local_target_threshold)
                )
                sharpness = float(self.householder_reliability_teacher_local_target_sharpness)
                selector_logits = (local_target_source - threshold) * sharpness
                if not self.householder_reliability_teacher_local_target_high:
                    selector_logits = -selector_logits
                teacher_selector = torch.sigmoid(selector_logits).reshape(-1)
                keep_value = target.new_tensor(
                    float(self.householder_reliability_teacher_local_target_keep_value)
                )
                reliability_target = (
                    teacher_selector * reliability_target
                    + (1.0 - teacher_selector) * keep_value
                ).clamp(0.0, 1.0)
            per_image_teacher_loss = F.binary_cross_entropy(
                rel_prob,
                reliability_target,
                reduction="none",
            )
            if self.householder_reliability_teacher_local_weight_source:
                local_weight_source = current_conditioning.get(
                    self.householder_reliability_teacher_local_weight_source
                )
                if local_weight_source is None:
                    raise KeyError(
                        "local reliability teacher weight requested, but conditioning tensor "
                        f"{self.householder_reliability_teacher_local_weight_source!r} is missing"
                    )
                if local_weight_source.shape != source_shape:
                    raise ValueError(
                        "local reliability teacher weight shape does not match reliability source: "
                        f"weight={tuple(local_weight_source.shape)}, source={tuple(source_shape)}"
                    )
                if self.householder_reliability_teacher_local_weight_detach:
                    local_weight_source = local_weight_source.detach()
                threshold = target.new_tensor(
                    float(self.householder_reliability_teacher_local_weight_threshold)
                )
                sharpness = float(self.householder_reliability_teacher_local_weight_sharpness)
                local_logits = (local_weight_source - threshold) * sharpness
                if not self.householder_reliability_teacher_local_weight_high:
                    local_logits = -local_logits
                local_weight = torch.sigmoid(local_logits)
                local_min = target.new_tensor(
                    float(self.householder_reliability_teacher_local_weight_min)
                )
                local_weight = local_min + (1.0 - local_min) * local_weight
                local_weight = local_weight.reshape(-1)
                if reliability_weight is None:
                    reliability_weight = local_weight
                else:
                    reliability_weight = reliability_weight * local_weight
            if reliability_weight is not None:
                teacher_loss = teacher_loss + (
                    per_image_teacher_loss * reliability_weight
                ).sum() / reliability_weight.sum().clamp_min(1e-6)
            else:
                teacher_loss = teacher_loss + per_image_teacher_loss.mean()
            conditioning_loss = conditioning_loss + self.rho_householder_reliability_teacher * teacher_loss

        if self.rho_householder_residual_selector_teacher != 0.0:
            selector_logits = current_conditioning.get("householder_residual_selector_logits")
            if selector_logits is None:
                raise KeyError(
                    "residual selector teacher loss requested, but selector logits are missing"
                )
            teacher_targets = output.get("teacher_targets", {})
            selector_target = teacher_targets.get(self.householder_residual_selector_teacher_target)
            if selector_target is None:
                raise KeyError(
                    "residual selector teacher loss requested, but target "
                    f"{self.householder_residual_selector_teacher_target!r} is missing"
                )
            if not torch.is_tensor(selector_target):
                selector_target = torch.as_tensor(selector_target, device=target.device, dtype=target.dtype)
            selector_target = selector_target.to(device=target.device, dtype=target.dtype).view(-1).clamp(0.0, 1.0)
            selector_weight = None
            if self.householder_residual_selector_teacher_weight:
                selector_weight = teacher_targets.get(self.householder_residual_selector_teacher_weight)
                if selector_weight is None:
                    raise KeyError(
                        "residual selector teacher weight requested, but target "
                        f"{self.householder_residual_selector_teacher_weight!r} is missing"
                    )
                if not torch.is_tensor(selector_weight):
                    selector_weight = torch.as_tensor(selector_weight, device=target.device, dtype=target.dtype)
                selector_weight = selector_weight.to(device=target.device, dtype=target.dtype).view(-1).clamp_min(0.0)
            selector_source = selector_logits
            if self.householder_residual_selector_teacher_use_image_mean:
                selector_source = selector_source.mean(dim=(1, 2, 3, 4), keepdim=True)
            source_shape = selector_source.shape
            selector_source = selector_source.reshape(-1)
            if selector_target.numel() != selector_source.numel():
                if (
                    not self.householder_residual_selector_teacher_use_image_mean
                    and selector_target.numel() == source_shape[0]
                ):
                    selector_target = selector_target.view(
                        source_shape[0], 1, 1, 1, 1
                    ).expand(source_shape).reshape(-1)
                    if selector_weight is not None:
                        selector_weight = selector_weight.view(
                            source_shape[0], 1, 1, 1, 1
                        ).expand(source_shape).reshape(-1)
                else:
                    raise ValueError(
                        "residual selector teacher target size does not match source: "
                        f"target={selector_target.numel()}, source={selector_source.numel()}"
                    )
            per_selector_loss = F.binary_cross_entropy_with_logits(
                selector_source,
                selector_target,
                reduction="none",
            )
            if self.householder_residual_selector_teacher_local_weight_source:
                local_weight_source = current_conditioning.get(
                    self.householder_residual_selector_teacher_local_weight_source
                )
                if local_weight_source is None:
                    raise KeyError(
                        "local residual selector teacher weight requested, but conditioning tensor "
                        f"{self.householder_residual_selector_teacher_local_weight_source!r} is missing"
                    )
                if self.householder_residual_selector_teacher_local_weight_detach:
                    local_weight_source = local_weight_source.detach()
                if self.householder_residual_selector_teacher_use_image_mean:
                    local_weight_source = local_weight_source.mean(dim=(1, 2, 3, 4), keepdim=True)
                if local_weight_source.shape != source_shape:
                    raise ValueError(
                        "local residual selector teacher weight shape does not match source: "
                        f"weight={tuple(local_weight_source.shape)}, source={tuple(source_shape)}"
                    )
                threshold = target.new_tensor(
                    float(self.householder_residual_selector_teacher_local_weight_threshold)
                )
                sharpness = float(self.householder_residual_selector_teacher_local_weight_sharpness)
                local_logits = (local_weight_source - threshold) * sharpness
                if not self.householder_residual_selector_teacher_local_weight_high:
                    local_logits = -local_logits
                local_weight = torch.sigmoid(local_logits)
                local_min = target.new_tensor(
                    float(self.householder_residual_selector_teacher_local_weight_min)
                )
                local_weight = local_min + (1.0 - local_min) * local_weight
                local_weight = local_weight.reshape(-1)
                if selector_weight is None:
                    selector_weight = local_weight
                else:
                    selector_weight = selector_weight * local_weight
            if selector_weight is not None:
                residual_selector_teacher_loss = (
                    per_selector_loss * selector_weight
                ).sum() / selector_weight.sum().clamp_min(1e-6)
            else:
                residual_selector_teacher_loss = per_selector_loss.mean()
            teacher_loss = teacher_loss + residual_selector_teacher_loss
            conditioning_loss = (
                conditioning_loss
                + self.rho_householder_residual_selector_teacher * residual_selector_teacher_loss
            )

        if self.rho_householder_residual_selector_noop != 0.0:
            selector_logits = current_conditioning.get("householder_residual_selector_logits")
            if selector_logits is None:
                raise KeyError(
                    "residual selector no-op loss requested, but selector logits are missing"
                )
            noop_source = selector_logits
            source_shape = noop_source.shape
            noop_target = torch.zeros_like(noop_source)
            per_noop_loss = F.binary_cross_entropy_with_logits(
                noop_source.reshape(-1),
                noop_target.reshape(-1),
                reduction="none",
            )
            noop_weight = None
            if self.householder_residual_selector_noop_weight_source:
                noop_weight_source = current_conditioning.get(
                    self.householder_residual_selector_noop_weight_source
                )
                if noop_weight_source is None:
                    raise KeyError(
                        "local residual selector no-op weight requested, but conditioning tensor "
                        f"{self.householder_residual_selector_noop_weight_source!r} is missing"
                    )
                if self.householder_residual_selector_noop_weight_detach:
                    noop_weight_source = noop_weight_source.detach()
                if noop_weight_source.shape != source_shape:
                    raise ValueError(
                        "local residual selector no-op weight shape does not match source: "
                        f"weight={tuple(noop_weight_source.shape)}, source={tuple(source_shape)}"
                    )
                threshold = target.new_tensor(
                    float(self.householder_residual_selector_noop_weight_threshold)
                )
                sharpness = float(self.householder_residual_selector_noop_weight_sharpness)
                noop_logits = (noop_weight_source - threshold) * sharpness
                if not self.householder_residual_selector_noop_weight_high:
                    noop_logits = -noop_logits
                noop_weight = torch.sigmoid(noop_logits)
                noop_min = target.new_tensor(
                    float(self.householder_residual_selector_noop_weight_min)
                )
                noop_weight = noop_min + (1.0 - noop_min) * noop_weight
                noop_weight = noop_weight.reshape(-1)
            if noop_weight is not None:
                residual_selector_noop_loss = (
                    per_noop_loss * noop_weight
                ).sum() / noop_weight.sum().clamp_min(1e-6)
            else:
                residual_selector_noop_loss = per_noop_loss.mean()
            conditioning_loss = (
                conditioning_loss
                + self.rho_householder_residual_selector_noop * residual_selector_noop_loss
            )

        anchor_conditioning = output.get("anchor_conditioning", {})
        anchor_specs = (
            ("mu_q", self.rho_anchor_mu),
            ("log_s_q", self.rho_anchor_log_s),
            ("u", self.rho_anchor_u),
        )
        for key, weight in anchor_specs:
            if weight == 0.0:
                continue
            if key not in current_conditioning or key not in anchor_conditioning:
                raise KeyError(f"anchor loss requested for {key}, but conditioning tensors are missing")
            anchor_loss = anchor_loss + float(weight) * F.mse_loss(
                current_conditioning[key],
                anchor_conditioning[key].detach(),
            )
        if self.rho_anchor_y_hat != 0.0:
            anchor_y_hat = output.get("anchor_y_hat")
            if anchor_y_hat is None or "y_hat" not in output:
                raise KeyError("y_hat anchor loss requested, but y_hat tensors are missing")
            anchor_loss = anchor_loss + float(self.rho_anchor_y_hat) * F.mse_loss(
                output["y_hat"],
                anchor_y_hat.detach(),
            )
        if self.rho_anchor_selected_distortion_margin != 0.0:
            anchor_x_hat = output.get("anchor_x_hat")
            if anchor_x_hat is None or "x_hat" not in output:
                raise KeyError(
                    "selected distortion margin requested, but x_hat anchor tensors are missing"
                )
            teacher_targets = output.get("teacher_targets", {})
            selected_target = teacher_targets.get(self.anchor_selected_distortion_target)
            if selected_target is None:
                raise KeyError(
                    "selected distortion margin requested, but teacher target "
                    f"{self.anchor_selected_distortion_target!r} is missing"
                )
            if not torch.is_tensor(selected_target):
                selected_target = torch.as_tensor(
                    selected_target,
                    device=target.device,
                    dtype=target.dtype,
                )
            selected_target = selected_target.to(device=target.device, dtype=target.dtype).view(-1).clamp(0.0, 1.0)
            if selected_target.numel() != target.size(0):
                raise ValueError(
                    "selected distortion target size does not match batch size: "
                    f"target={selected_target.numel()}, batch={target.size(0)}"
                )
            current_mse = (output["x_hat"].clamp(0, 1) - target).pow(2).flatten(1).mean(dim=1)
            anchor_mse = (
                anchor_x_hat.detach().clamp(0, 1) - target
            ).pow(2).flatten(1).mean(dim=1)
            current_distortion = self.lambda_rd * self.mse_scale * current_mse
            anchor_distortion = self.lambda_rd * self.mse_scale * anchor_mse
            selected_margin = target.new_tensor(float(self.anchor_selected_distortion_margin))
            selected_excess = F.relu(current_distortion - anchor_distortion + selected_margin)
            keep_margin = target.new_tensor(float(self.anchor_selected_distortion_keep_margin))
            keep_excess = F.relu(current_distortion - anchor_distortion + keep_margin)
            if self.anchor_selected_distortion_squared:
                selected_excess = selected_excess.pow(2)
                keep_excess = keep_excess.pow(2)
            selected_weight = selected_target
            if self.anchor_selected_distortion_weight:
                extra_weight = teacher_targets.get(self.anchor_selected_distortion_weight)
                if extra_weight is None:
                    raise KeyError(
                        "selected distortion margin weight requested, but teacher target "
                        f"{self.anchor_selected_distortion_weight!r} is missing"
                    )
                if not torch.is_tensor(extra_weight):
                    extra_weight = torch.as_tensor(
                        extra_weight,
                        device=target.device,
                        dtype=target.dtype,
                    )
                extra_weight = extra_weight.to(device=target.device, dtype=target.dtype).view(-1).clamp_min(0.0)
                if extra_weight.numel() != target.size(0):
                    raise ValueError(
                        "selected distortion weight size does not match batch size: "
                        f"weight={extra_weight.numel()}, batch={target.size(0)}"
                    )
                selected_weight = selected_weight * extra_weight
            keep_weight = (1.0 - selected_target) * float(self.anchor_selected_distortion_keep_weight)
            total_weight = selected_weight.sum() + keep_weight.sum()
            selected_distortion_margin_loss = (
                (selected_excess * selected_weight).sum()
                + (keep_excess * keep_weight).sum()
            ) / total_weight.clamp_min(1e-6)
            anchor_loss = (
                anchor_loss
                + float(self.rho_anchor_selected_distortion_margin)
                * selected_distortion_margin_loss
            )

        loss = bpp_total + self.lambda_rd * self.mse_scale * mse
        loss = loss + self.beta_commit * commit_loss + self.rho_gate * gate_loss + conditioning_loss + anchor_loss

        return {
            "loss": loss,
            "bpp_total": bpp_total.detach(),
            "bpp_y": bpp_y.detach(),
            "bpp_z": bpp_z.detach(),
            "mse": mse.detach(),
            "psnr": (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).detach(),
            "commit_loss": commit_loss.detach(),
            "gate_loss": gate_loss.detach(),
            "conditioning_loss": conditioning_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "teacher_loss": teacher_loss.detach(),
            "residual_selector_teacher_loss": residual_selector_teacher_loss.detach(),
            "residual_selector_noop_loss": residual_selector_noop_loss.detach(),
            "selected_distortion_margin_loss": selected_distortion_margin_loss.detach(),
        }

