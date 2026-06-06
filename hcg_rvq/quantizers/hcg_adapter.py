from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as F

from .householder import householder_transform
from .residual_vq import ResidualVectorQuantizer


def _inverse_softplus(value: float) -> float:
    return torch.log(torch.expm1(torch.tensor(value))).item()


def _inverse_sigmoid(value: float) -> float:
    value = torch.tensor(value).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(value).item()


class HCGQuantizerOwner(Protocol):
    """Minimal interface required by the HCG quantizer adapter.

    The adapter is intentionally parameter-free for now so existing checkpoints
    keep their state-dict keys. A later stronger-backbone plug-in can replace
    the owner with a real module that exposes the same contract.
    """

    variant: str
    use_global_norm: bool
    global_log_s: torch.Tensor
    global_mu: torch.Tensor
    eps: float
    scale_min: float
    scale_max: float
    householder_strength: float

    def _stage_gate(self, hyper_features: torch.Tensor) -> torch.Tensor | None: ...
    def _to_grouped(self, x: torch.Tensor) -> torch.Tensor: ...
    def _from_grouped(self, x: torch.Tensor) -> torch.Tensor: ...
    def _partial_householder_transform(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        strength: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def _inverse_partial_householder_transform(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        strength: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def _raw_householder_gate(self, hyper_features: torch.Tensor) -> torch.Tensor | None: ...
    def _householder_gate_reliability_multiplier(self, hyper_features: torch.Tensor) -> torch.Tensor | None: ...
    def _householder_gate_raw_backoff_multiplier(self, raw_gate: torch.Tensor | None) -> torch.Tensor | None: ...
    def _householder_gate_strength_backoff_multiplier(self, gate: torch.Tensor | None) -> torch.Tensor | None: ...
    def _householder_gate_strength_backoff_multiplier(self, gate: torch.Tensor | None) -> torch.Tensor | None:
        if not self.householder_gate_strength_backoff_enabled:
            return None
        if gate is None:
            raise ValueError("householder_gate_strength_backoff_enabled requires householder gate")
        source = gate.detach() if self.householder_gate_strength_backoff_detach else gate
        if self.householder_gate_strength_backoff_use_image_mean:
            source = source.mean(dim=(1, 2, 3, 4), keepdim=True)
        keep = torch.sigmoid(
            (self.householder_gate_strength_backoff_threshold - source)
            * self.householder_gate_strength_backoff_sharpness
        )
        return self.householder_gate_strength_backoff_min + (
            1.0 - self.householder_gate_strength_backoff_min
        ) * keep

    def _householder_gate_residual_selector(
        self,
        hyper_features: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: ...
    def _householder_gate_risk_multiplier(self, s_q: torch.Tensor | None) -> torch.Tensor | None: ...


def run_hcg_quantizer_adapter(
    owner: HCGQuantizerOwner,
    y: torch.Tensor,
    hyper_features: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Run the HCG-RVQ quantizer boundary without owning parameters."""

    if owner.variant == "global_rvq":
        gate = owner._stage_gate(hyper_features)
        if owner.use_global_norm:
            s_q = (F.softplus(owner.global_log_s) + owner.eps).clamp(owner.scale_min, owner.scale_max)
            u = (y - owner.global_mu) / s_q
            u_hat, indices, commit_loss, stats = owner.rvq(u, image_hw=image_hw, gate=gate)
            y_hat = owner.global_mu + s_q * u_hat
            stats.update(
                {
                    "global_s_q_mean": s_q.mean(),
                    "global_s_q_min": s_q.min(),
                    "global_s_q_max": s_q.max(),
                }
            )
        else:
            y_hat, indices, commit_loss, stats = owner.rvq(y, image_hw=image_hw, gate=gate)
        if gate is not None:
            stats["avg_gate"] = gate.mean()
        return y_hat, indices, commit_loss, stats, {}

    mu_q = owner.mu_head(hyper_features)
    log_s_q = owner.log_s_head(hyper_features)
    s_q = (F.softplus(log_s_q) + owner.eps).clamp(owner.scale_min, owner.scale_max)
    y_norm = (y - mu_q) / s_q

    apply_householder = owner.variant in {"hcg_rvq_h", "hcg_rvq_h_gate"}
    track_householder = apply_householder or owner.variant == "hcg_rvq_h_no_transform"
    if track_householder:
        v = owner.householder_head(hyper_features)
        v_g = F.normalize(owner._to_grouped(v), dim=-1, eps=owner.eps)
        stats_extra = {
            "householder_v_norm": v_g.norm(dim=-1).mean(),
            "householder_v_abs_mean": v.abs().mean(),
        }
        if apply_householder:
            y_norm_g = owner._to_grouped(y_norm)
            householder_gate_raw = owner._raw_householder_gate(hyper_features)
            householder_reliability_multiplier = owner._householder_gate_reliability_multiplier(hyper_features)
            householder_raw_backoff_multiplier = owner._householder_gate_raw_backoff_multiplier(householder_gate_raw)
            (
                householder_residual_selector_prob,
                householder_residual_selector_multiplier,
                householder_residual_selector_logits,
            ) = owner._householder_gate_residual_selector(hyper_features)
            householder_risk_multiplier = owner._householder_gate_risk_multiplier(s_q)
            householder_strength_backoff_multiplier = None
            householder_gate = householder_gate_raw
            if householder_gate is not None and householder_reliability_multiplier is not None:
                householder_gate = householder_gate * householder_reliability_multiplier
            if householder_gate is not None and householder_raw_backoff_multiplier is not None:
                householder_gate = householder_gate * householder_raw_backoff_multiplier
            if householder_gate is not None and householder_residual_selector_multiplier is not None:
                householder_gate = householder_gate * householder_residual_selector_multiplier
            if householder_gate is not None and householder_risk_multiplier is not None:
                householder_gate = householder_gate * householder_risk_multiplier
            if householder_gate is not None:
                householder_strength_backoff_multiplier = owner._householder_gate_strength_backoff_multiplier(
                    householder_gate
                )
            if householder_gate is not None and householder_strength_backoff_multiplier is not None:
                householder_gate = householder_gate * householder_strength_backoff_multiplier
            u = owner._from_grouped(owner._partial_householder_transform(y_norm_g, v_g, householder_gate))
            delta = u - y_norm
            householder_delta_rms_map = owner._to_grouped(delta).pow(2).mean(dim=-1, keepdim=True).sqrt()
            strength_for_stats = (
                householder_gate.detach()
                if householder_gate is not None
                else y.new_full((1,), owner.householder_strength)
            )
            stats_extra.update(
                {
                    "householder_delta_abs_mean": delta.abs().mean(),
                    "householder_delta_rms": delta.pow(2).mean().sqrt(),
                    "householder_delta_rms_local_mean": householder_delta_rms_map.detach().mean(),
                    "householder_delta_rms_local_max": householder_delta_rms_map.detach().max(),
                    "householder_delta_rms_local_std": householder_delta_rms_map.detach().std(unbiased=False),
                    "householder_strength": strength_for_stats.mean(),
                    "householder_strength_min": strength_for_stats.min(),
                    "householder_strength_max": strength_for_stats.max(),
                    "householder_strength_std": strength_for_stats.std(unbiased=False),
                }
            )
            if householder_gate_raw is not None:
                stats_extra["householder_gate_raw"] = householder_gate_raw.detach().mean()
                stats_extra["householder_gate_raw_image_mean"] = householder_gate_raw.detach().mean(
                    dim=(1, 2, 3, 4)
                ).mean()
            if householder_reliability_multiplier is not None:
                stats_extra.update(
                    {
                        "householder_reliability_multiplier": householder_reliability_multiplier.detach().mean(),
                        "householder_reliability_multiplier_min": householder_reliability_multiplier.detach().min(),
                        "householder_reliability_multiplier_max": householder_reliability_multiplier.detach().max(),
                        "householder_reliability_multiplier_std": householder_reliability_multiplier.detach().std(
                            unbiased=False
                        ),
                    }
                )
            if householder_raw_backoff_multiplier is not None:
                stats_extra.update(
                    {
                        "householder_raw_backoff_multiplier": householder_raw_backoff_multiplier.detach().mean(),
                        "householder_raw_backoff_multiplier_min": householder_raw_backoff_multiplier.detach().min(),
                        "householder_raw_backoff_multiplier_max": householder_raw_backoff_multiplier.detach().max(),
                        "householder_raw_backoff_multiplier_std": householder_raw_backoff_multiplier.detach().std(
                            unbiased=False
                        ),
                    }
                )
            if householder_residual_selector_prob is not None:
                stats_extra.update(
                    {
                        "householder_residual_selector_prob": householder_residual_selector_prob.detach().mean(),
                        "householder_residual_selector_prob_min": householder_residual_selector_prob.detach().min(),
                        "householder_residual_selector_prob_max": householder_residual_selector_prob.detach().max(),
                        "householder_residual_selector_prob_std": householder_residual_selector_prob.detach().std(
                            unbiased=False
                        ),
                        "householder_residual_selector_multiplier": householder_residual_selector_multiplier.detach().mean(),
                        "householder_residual_selector_multiplier_min": householder_residual_selector_multiplier.detach().min(),
                        "householder_residual_selector_multiplier_max": householder_residual_selector_multiplier.detach().max(),
                        "householder_residual_selector_multiplier_std": householder_residual_selector_multiplier.detach().std(
                            unbiased=False
                        ),
                    }
                )
            if householder_risk_multiplier is not None:
                stats_extra.update(
                    {
                        "householder_risk_multiplier": householder_risk_multiplier.detach().mean(),
                        "householder_risk_multiplier_min": householder_risk_multiplier.detach().min(),
                        "householder_risk_multiplier_max": householder_risk_multiplier.detach().max(),
                        "householder_risk_multiplier_std": householder_risk_multiplier.detach().std(unbiased=False),
                    }
                )
            if householder_strength_backoff_multiplier is not None:
                stats_extra.update(
                    {
                        "householder_strength_backoff_multiplier": householder_strength_backoff_multiplier.detach().mean(),
                        "householder_strength_backoff_multiplier_min": householder_strength_backoff_multiplier.detach().min(),
                        "householder_strength_backoff_multiplier_max": householder_strength_backoff_multiplier.detach().max(),
                        "householder_strength_backoff_multiplier_std": householder_strength_backoff_multiplier.detach().std(
                            unbiased=False
                        ),
                    }
                )
        else:
            u = y_norm
    else:
        v_g = None
        householder_gate = None
        u = y_norm
        stats_extra = {}

    gate = owner._stage_gate(hyper_features)
    u_hat, indices, commit_loss, stats = owner.rvq(u, image_hw=image_hw, gate=gate)

    if apply_householder:
        u_hat_g = owner._to_grouped(u_hat)
        y_norm_hat = owner._from_grouped(owner._inverse_partial_householder_transform(u_hat_g, v_g, householder_gate))
    else:
        y_norm_hat = u_hat

    y_hat = mu_q + s_q * y_norm_hat
    stats.update(stats_extra)
    stats.update(
        {
            "mu_q_abs_mean": mu_q.abs().mean(),
            "mu_q_std": mu_q.std(unbiased=False),
            "s_q_mean": s_q.mean(),
            "s_q_min": s_q.min(),
            "s_q_max": s_q.max(),
            "s_q_std": s_q.std(unbiased=False),
            "y_norm_abs_mean": y_norm.abs().mean(),
        }
    )
    if gate is not None:
        stats["avg_gate"] = gate.mean()
    conditioning_tensors = {
        "mu_q": mu_q,
        "log_s_q": log_s_q,
        "s_q": s_q,
        "y_norm": y_norm,
        "u": u,
    }
    if apply_householder:
        if householder_gate_raw is not None:
            conditioning_tensors["householder_gate_raw"] = householder_gate_raw
        if householder_gate is not None:
            conditioning_tensors["householder_strength"] = householder_gate
        if householder_reliability_multiplier is not None:
            conditioning_tensors["householder_reliability_multiplier"] = householder_reliability_multiplier
        conditioning_tensors["householder_delta_rms_map"] = householder_delta_rms_map
        if householder_residual_selector_prob is not None:
            conditioning_tensors["householder_residual_selector_prob"] = householder_residual_selector_prob
            conditioning_tensors["householder_residual_selector_multiplier"] = householder_residual_selector_multiplier
            conditioning_tensors["householder_residual_selector_logits"] = householder_residual_selector_logits
        if householder_risk_multiplier is not None:
            conditioning_tensors["householder_risk_multiplier"] = householder_risk_multiplier
        if householder_strength_backoff_multiplier is not None:
            conditioning_tensors["householder_strength_backoff_multiplier"] = householder_strength_backoff_multiplier
        if householder_raw_backoff_multiplier is not None:
            conditioning_tensors["householder_raw_backoff_multiplier"] = householder_raw_backoff_multiplier
    return y_hat, indices, commit_loss, stats, conditioning_tensors

class HCGQuantizerAdapter(nn.Module):
    """Standalone HCG-RVQ quantizer adapter with explicit channel contracts."""

    def __init__(
        self,
        latent_channels: int,
        hyper_channels: int,
        variant: str = "hcg_rvq_h",
        group_size: int = 64,
        num_stages: int = 1,
        codebook_size: int = 128,
        codebook_init_scale: float = 0.02,
        stage_gate_enabled: bool = False,
        scale_min: float = 0.05,
        scale_max: float = 10.0,
        householder_strength: float = 1.0,
        householder_bias_init_scale: float = 0.0,
        householder_gate_enabled: bool = False,
        householder_gate_max: float = 0.45,
        householder_gate_init: float = 0.25,
        householder_gate_risk_enabled: bool = False,
        householder_gate_risk_center: float = 0.56,
        householder_gate_risk_sharpness: float = 12.0,
        householder_gate_risk_min: float = 0.5,
        householder_gate_risk_invert: bool = False,
        householder_gate_risk_detach: bool = False,
        householder_gate_reliability_enabled: bool = False,
        householder_gate_reliability_min: float = 0.5,
        householder_gate_reliability_init: float = 0.99,
        householder_gate_reliability_detach: bool = False,
        householder_gate_raw_backoff_enabled: bool = False,
        householder_gate_raw_backoff_threshold: float = 0.284059,
        householder_gate_raw_backoff_min: float = 0.65,
        householder_gate_raw_backoff_sharpness: float = 80.0,
        householder_gate_raw_backoff_detach: bool = True,
        householder_gate_raw_backoff_use_image_mean: bool = True,
        householder_gate_strength_backoff_enabled: bool = False,
        householder_gate_strength_backoff_threshold: float = 0.271352783,
        householder_gate_strength_backoff_min: float = 0.0,
        householder_gate_strength_backoff_sharpness: float = 80.0,
        householder_gate_strength_backoff_detach: bool = True,
        householder_gate_strength_backoff_use_image_mean: bool = True,
        householder_gate_residual_selector_enabled: bool = False,
        householder_gate_residual_selector_max: float = 0.50,
        householder_gate_residual_selector_bias: float = -4.0,
        householder_gate_residual_selector_deadzone_threshold: float = 0.0,
        householder_gate_residual_selector_detach: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if latent_channels % group_size != 0:
            raise ValueError(f"latent_channels={latent_channels} must be divisible by group_size={group_size}")
        if abs(1.0 - 2.0 * float(householder_strength)) < 1e-3:
            raise ValueError("householder_strength too close to 0.5 makes the partial reflection singular")
        if householder_gate_enabled:
            if not 0.0 < float(householder_gate_max) < 0.5:
                raise ValueError("householder_gate_max must stay in (0, 0.5)")
            if not 0.0 < float(householder_gate_init) < float(householder_gate_max):
                raise ValueError("householder_gate_init must stay in (0, householder_gate_max)")

        self.variant = variant
        self.latent_channels = int(latent_channels)
        self.hyper_channels = int(hyper_channels)
        self.group_size = int(group_size)
        self.num_groups = self.latent_channels // self.group_size
        self.num_stages = int(num_stages)
        self.codebook_size = int(codebook_size)
        self.stage_gate_enabled = bool(stage_gate_enabled)
        self.use_global_norm = False
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.householder_strength = float(householder_strength)
        self.householder_bias_init_scale = float(householder_bias_init_scale)
        self.householder_gate_enabled = bool(householder_gate_enabled)
        self.householder_gate_max = float(householder_gate_max)
        self.householder_gate_init = float(householder_gate_init)
        self.householder_gate_risk_enabled = bool(householder_gate_risk_enabled)
        self.householder_gate_risk_center = float(householder_gate_risk_center)
        self.householder_gate_risk_sharpness = float(householder_gate_risk_sharpness)
        self.householder_gate_risk_min = float(householder_gate_risk_min)
        self.householder_gate_risk_invert = bool(householder_gate_risk_invert)
        self.householder_gate_risk_detach = bool(householder_gate_risk_detach)
        self.householder_gate_reliability_enabled = bool(householder_gate_reliability_enabled)
        self.householder_gate_reliability_min = float(householder_gate_reliability_min)
        self.householder_gate_reliability_init = float(householder_gate_reliability_init)
        self.householder_gate_reliability_detach = bool(householder_gate_reliability_detach)
        self.householder_gate_raw_backoff_enabled = bool(householder_gate_raw_backoff_enabled)
        self.householder_gate_raw_backoff_threshold = float(householder_gate_raw_backoff_threshold)
        self.householder_gate_raw_backoff_min = float(householder_gate_raw_backoff_min)
        self.householder_gate_raw_backoff_sharpness = float(householder_gate_raw_backoff_sharpness)
        self.householder_gate_raw_backoff_detach = bool(householder_gate_raw_backoff_detach)
        self.householder_gate_raw_backoff_use_image_mean = bool(householder_gate_raw_backoff_use_image_mean)
        self.householder_gate_strength_backoff_enabled = bool(householder_gate_strength_backoff_enabled)
        self.householder_gate_strength_backoff_threshold = float(householder_gate_strength_backoff_threshold)
        self.householder_gate_strength_backoff_min = float(householder_gate_strength_backoff_min)
        self.householder_gate_strength_backoff_sharpness = float(householder_gate_strength_backoff_sharpness)
        self.householder_gate_strength_backoff_detach = bool(householder_gate_strength_backoff_detach)
        self.householder_gate_strength_backoff_use_image_mean = bool(
            householder_gate_strength_backoff_use_image_mean
        )
        self.householder_gate_residual_selector_enabled = bool(householder_gate_residual_selector_enabled)
        self.householder_gate_residual_selector_max = float(householder_gate_residual_selector_max)
        self.householder_gate_residual_selector_bias = float(householder_gate_residual_selector_bias)
        self.householder_gate_residual_selector_deadzone_threshold = float(
            householder_gate_residual_selector_deadzone_threshold
        )
        self.householder_gate_residual_selector_detach = bool(householder_gate_residual_selector_detach)
        self.eps = float(eps)

        self.mu_head = nn.Conv2d(self.hyper_channels, self.latent_channels, 1)
        self.log_s_head = nn.Conv2d(self.hyper_channels, self.latent_channels, 1)
        self.householder_head = nn.Conv2d(self.hyper_channels, self.latent_channels, 1)
        self.householder_gate_head = nn.Conv2d(self.hyper_channels, self.num_groups, 1)
        self.householder_reliability_head = nn.Conv2d(self.hyper_channels, self.num_groups, 1)
        self.householder_residual_selector_head = nn.Conv2d(self.hyper_channels, self.num_groups, 1)
        self.gate_head = nn.Conv2d(self.hyper_channels, self.num_groups * self.num_stages, 1)
        self.global_mu = nn.Parameter(torch.zeros(1, self.latent_channels, 1, 1))
        self.global_log_s = nn.Parameter(torch.full((1, self.latent_channels, 1, 1), _inverse_softplus(1.0)))
        self.rvq = ResidualVectorQuantizer(
            dim=self.group_size,
            num_stages=self.num_stages,
            codebook_size=self.codebook_size,
            init_scale=codebook_init_scale,
        )
        self._init_conditioning_heads()

    def _init_conditioning_heads(self) -> None:
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_s_head.weight)
        nn.init.constant_(self.log_s_head.bias, _inverse_softplus(1.0))
        nn.init.zeros_(self.householder_head.weight)
        nn.init.zeros_(self.householder_head.bias)
        if self.householder_bias_init_scale > 0.0:
            nn.init.normal_(self.householder_head.bias, mean=0.0, std=self.householder_bias_init_scale)
        nn.init.zeros_(self.householder_gate_head.weight)
        gate_ratio = self.householder_gate_init / self.householder_gate_max
        nn.init.constant_(self.householder_gate_head.bias, _inverse_sigmoid(gate_ratio))
        nn.init.zeros_(self.householder_reliability_head.weight)
        reliability_ratio = (
            (self.householder_gate_reliability_init - self.householder_gate_reliability_min)
            / (1.0 - self.householder_gate_reliability_min)
        )
        nn.init.constant_(self.householder_reliability_head.bias, _inverse_sigmoid(reliability_ratio))
        nn.init.zeros_(self.householder_residual_selector_head.weight)
        nn.init.zeros_(self.householder_residual_selector_head.bias)

    def _to_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.num_groups, self.group_size, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def _from_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, ng, h, w, g = x.shape
        return x.permute(0, 1, 4, 2, 3).contiguous().view(b, ng * g, h, w)

    def _partial_householder_transform(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if strength is None and self.householder_strength == 1.0:
            return householder_transform(x, v, eps=self.eps)
        strength = x.new_tensor(self.householder_strength) if strength is None else strength.to(
            device=x.device,
            dtype=x.dtype,
        )
        dot = (x * v).sum(dim=-1, keepdim=True)
        norm = (v * v).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return x - 2.0 * strength * dot / norm * v

    def _inverse_partial_householder_transform(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if strength is None and self.householder_strength == 1.0:
            return householder_transform(x, v, eps=self.eps)
        strength = x.new_tensor(self.householder_strength) if strength is None else strength.to(
            device=x.device,
            dtype=x.dtype,
        )
        denom = 1.0 - 2.0 * strength
        dot = (x * v).sum(dim=-1, keepdim=True)
        norm = (v * v).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return x + (2.0 * strength / denom) * dot / norm * v

    def _raw_householder_gate(self, hyper_features: torch.Tensor) -> torch.Tensor | None:
        if not self.householder_gate_enabled:
            return None
        b, _, h, w = hyper_features.shape
        gate = torch.sigmoid(self.householder_gate_head(hyper_features)) * self.householder_gate_max
        return gate.view(b, self.num_groups, h, w, 1)

    def _householder_gate_risk_multiplier(self, s_q: torch.Tensor | None) -> torch.Tensor | None:
        if not self.householder_gate_risk_enabled:
            return None
        if s_q is None:
            raise ValueError("householder_gate_risk_enabled requires s_q")
        if self.householder_gate_risk_detach:
            s_q = s_q.detach()
        s_q_group = self._to_grouped(s_q).mean(dim=-1, keepdim=True)
        risk = torch.sigmoid((s_q_group - self.householder_gate_risk_center) * self.householder_gate_risk_sharpness)
        if self.householder_gate_risk_invert:
            risk = 1.0 - risk
        return self.householder_gate_risk_min + (1.0 - self.householder_gate_risk_min) * risk

    def _householder_gate_reliability_multiplier(self, hyper_features: torch.Tensor) -> torch.Tensor | None:
        if not self.householder_gate_reliability_enabled:
            return None
        if self.householder_gate_reliability_detach:
            hyper_features = hyper_features.detach()
        b, _, h, w = hyper_features.shape
        reliability = torch.sigmoid(self.householder_reliability_head(hyper_features))
        reliability = self.householder_gate_reliability_min + (
            1.0 - self.householder_gate_reliability_min
        ) * reliability
        return reliability.view(b, self.num_groups, h, w, 1)

    def _householder_gate_raw_backoff_multiplier(self, raw_gate: torch.Tensor | None) -> torch.Tensor | None:
        if not self.householder_gate_raw_backoff_enabled:
            return None
        if raw_gate is None:
            raise ValueError("householder_gate_raw_backoff_enabled requires raw householder gate")
        source = raw_gate.detach() if self.householder_gate_raw_backoff_detach else raw_gate
        if self.householder_gate_raw_backoff_use_image_mean:
            source = source.mean(dim=(1, 2, 3, 4), keepdim=True)
        keep = torch.sigmoid(
            (self.householder_gate_raw_backoff_threshold - source) * self.householder_gate_raw_backoff_sharpness
        )
        return self.householder_gate_raw_backoff_min + (1.0 - self.householder_gate_raw_backoff_min) * keep

    def _householder_gate_strength_backoff_multiplier(self, gate: torch.Tensor | None) -> torch.Tensor | None:
        if not self.householder_gate_strength_backoff_enabled:
            return None
        if gate is None:
            raise ValueError("householder_gate_strength_backoff_enabled requires householder gate")
        source = gate.detach() if self.householder_gate_strength_backoff_detach else gate
        if self.householder_gate_strength_backoff_use_image_mean:
            source = source.mean(dim=(1, 2, 3, 4), keepdim=True)
        keep = torch.sigmoid(
            (self.householder_gate_strength_backoff_threshold - source)
            * self.householder_gate_strength_backoff_sharpness
        )
        return self.householder_gate_strength_backoff_min + (
            1.0 - self.householder_gate_strength_backoff_min
        ) * keep

    def _householder_gate_residual_selector(
        self,
        hyper_features: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.householder_gate_residual_selector_enabled:
            return None, None, None
        source = hyper_features.detach() if self.householder_gate_residual_selector_detach else hyper_features
        b, _, h, w = source.shape
        logits = self.householder_residual_selector_head(source)
        bias = logits.new_tensor(self.householder_gate_residual_selector_bias)
        base = torch.sigmoid(bias)
        selector_prob = (torch.sigmoid(logits + bias) - base) / (1.0 - base).clamp_min(1e-6)
        selector_prob = selector_prob.clamp(0.0, 1.0)
        threshold = self.householder_gate_residual_selector_deadzone_threshold
        if threshold > 0.0:
            selector_prob = torch.where(
                selector_prob >= threshold,
                selector_prob,
                torch.zeros_like(selector_prob),
            )
        selector_amount = selector_prob * self.householder_gate_residual_selector_max
        selector_multiplier = 1.0 - selector_amount
        return (
            selector_prob.view(b, self.num_groups, h, w, 1),
            selector_multiplier.view(b, self.num_groups, h, w, 1),
            (logits + bias).view(b, self.num_groups, h, w, 1),
        )

    def _stage_gate(self, hyper_features: torch.Tensor) -> torch.Tensor | None:
        if not self.stage_gate_enabled:
            return None
        b, _, h, w = hyper_features.shape
        gate = self.gate_head(hyper_features)
        gate = gate.view(b, self.num_groups, self.num_stages, h, w)
        return torch.sigmoid(gate.permute(0, 1, 3, 4, 2).contiguous())

    def forward(
        self,
        y: torch.Tensor,
        hyper_features: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        return run_hcg_quantizer_adapter(self, y, hyper_features, image_hw)

