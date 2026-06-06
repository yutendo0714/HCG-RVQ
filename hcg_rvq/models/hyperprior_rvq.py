from __future__ import annotations

import torch
from torch import nn

from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

from hcg_rvq.entropy import IndexEntropyModel
from hcg_rvq.quantizers import ResidualVectorQuantizer, householder_transform
from hcg_rvq.quantizers.hcg_adapter import run_hcg_quantizer_adapter


def _inverse_softplus(value: float) -> float:
    return torch.log(torch.expm1(torch.tensor(value))).item()


def _inverse_sigmoid(value: float) -> float:
    value = torch.tensor(value).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(value).item()


class HCGMeanScaleHyperprior(CompressionModel):
    """Mean-scale hyperprior backbone with optional RVQ/HCS/HCG quantizers."""

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        variant: str = "scalar",
        group_size: int = 32,
        num_stages: int = 2,
        codebook_size: int = 256,
        index_prior_enabled: bool = False,
        index_hidden_channels: int = 192,
        stage_gate_enabled: bool = False,
        use_global_norm: bool = False,
        codebook_init_scale: float = 0.02,
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
        super().__init__(entropy_bottleneck_channels=N)
        if M % group_size != 0:
            raise ValueError(f"M={M} must be divisible by group_size={group_size}")
        if abs(1.0 - 2.0 * float(householder_strength)) < 1e-3:
            raise ValueError("householder_strength too close to 0.5 makes the partial reflection singular")
        if householder_gate_enabled:
            if not 0.0 < float(householder_gate_max) < 0.5:
                raise ValueError("householder_gate_max must stay in (0, 0.5) for an invertible gated transform")
            if not 0.0 < float(householder_gate_init) < float(householder_gate_max):
                raise ValueError("householder_gate_init must stay in (0, householder_gate_max)")
        if householder_gate_risk_enabled and not householder_gate_enabled:
            raise ValueError("householder_gate_risk_enabled requires householder_gate_enabled")
        if householder_gate_reliability_enabled and not householder_gate_enabled:
            raise ValueError("householder_gate_reliability_enabled requires householder_gate_enabled")
        if householder_gate_raw_backoff_enabled and not householder_gate_enabled:
            raise ValueError("householder_gate_raw_backoff_enabled requires householder_gate_enabled")
        if householder_gate_strength_backoff_enabled and not householder_gate_enabled:
            raise ValueError("householder_gate_strength_backoff_enabled requires householder_gate_enabled")
        if householder_gate_residual_selector_enabled and not householder_gate_enabled:
            raise ValueError("householder_gate_residual_selector_enabled requires householder_gate_enabled")
        if not 0.0 <= float(householder_gate_risk_min) <= 1.0:
            raise ValueError("householder_gate_risk_min must stay in [0, 1]")
        if float(householder_gate_risk_sharpness) < 0.0:
            raise ValueError("householder_gate_risk_sharpness must be non-negative")
        if not 0.0 <= float(householder_gate_reliability_min) < 1.0:
            raise ValueError("householder_gate_reliability_min must stay in [0, 1)")
        if not float(householder_gate_reliability_min) < float(householder_gate_reliability_init) < 1.0:
            raise ValueError("householder_gate_reliability_init must stay in (min, 1)")
        if not 0.0 <= float(householder_gate_raw_backoff_min) <= 1.0:
            raise ValueError("householder_gate_raw_backoff_min must stay in [0, 1]")
        if float(householder_gate_raw_backoff_threshold) < 0.0:
            raise ValueError("householder_gate_raw_backoff_threshold must be non-negative")
        if float(householder_gate_raw_backoff_sharpness) < 0.0:
            raise ValueError("householder_gate_raw_backoff_sharpness must be non-negative")
        if not 0.0 <= float(householder_gate_strength_backoff_min) <= 1.0:
            raise ValueError("householder_gate_strength_backoff_min must stay in [0, 1]")
        if float(householder_gate_strength_backoff_threshold) < 0.0:
            raise ValueError("householder_gate_strength_backoff_threshold must be non-negative")
        if float(householder_gate_strength_backoff_sharpness) < 0.0:
            raise ValueError("householder_gate_strength_backoff_sharpness must be non-negative")
        if not 0.0 <= float(householder_gate_residual_selector_max) < 1.0:
            raise ValueError("householder_gate_residual_selector_max must stay in [0, 1)")
        if not 0.0 <= float(householder_gate_residual_selector_deadzone_threshold) < 1.0:
            raise ValueError("householder_gate_residual_selector_deadzone_threshold must stay in [0, 1)")
        if float(householder_bias_init_scale) < 0.0:
            raise ValueError("householder_bias_init_scale must be non-negative")

        self.N = N
        self.M = M
        self.variant = variant
        self.group_size = group_size
        self.num_groups = M // group_size
        self.num_stages = num_stages
        self.codebook_size = codebook_size
        self.index_prior_enabled = index_prior_enabled
        self.stage_gate_enabled = stage_gate_enabled
        self.use_global_norm = use_global_norm
        self.scale_min = scale_min
        self.scale_max = scale_max
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
        self.eps = eps

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(N * 3 // 2, N, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.scalar_head = nn.Conv2d(N, 2 * M, 1)

        self.mu_head = nn.Conv2d(N, M, 1)
        self.log_s_head = nn.Conv2d(N, M, 1)
        self.householder_head = nn.Conv2d(N, M, 1)
        self.householder_gate_head = nn.Conv2d(N, self.num_groups, 1)
        rng_state = torch.random.get_rng_state()
        self.householder_reliability_head = nn.Conv2d(N, self.num_groups, 1)
        torch.random.set_rng_state(rng_state)
        rng_state = torch.random.get_rng_state()
        self.householder_residual_selector_head = nn.Conv2d(N, self.num_groups, 1)
        torch.random.set_rng_state(rng_state)
        self.gate_head = nn.Conv2d(N, self.num_groups * num_stages, 1)
        self.global_mu = nn.Parameter(torch.zeros(1, M, 1, 1))
        self.global_log_s = nn.Parameter(torch.full((1, M, 1, 1), _inverse_softplus(1.0)))

        self.rvq = ResidualVectorQuantizer(
            dim=group_size,
            num_stages=num_stages,
            codebook_size=codebook_size,
            init_scale=codebook_init_scale,
        )
        self.index_entropy_model = (
            IndexEntropyModel(
                hyper_channels=N,
                num_groups=self.num_groups,
                num_stages=num_stages,
                codebook_size=codebook_size,
                hidden_channels=index_hidden_channels,
            )
            if index_prior_enabled
            else None
        )
        self._init_conditioning_heads()

    def _init_conditioning_heads(self) -> None:
        """Start adaptive quantizers as close to identity transforms."""
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
        strength = x.new_tensor(self.householder_strength) if strength is None else strength.to(device=x.device, dtype=x.dtype)
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
        strength = x.new_tensor(self.householder_strength) if strength is None else strength.to(device=x.device, dtype=x.dtype)
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
            (self.householder_gate_raw_backoff_threshold - source)
            * self.householder_gate_raw_backoff_sharpness
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

    def _householder_gate(self, hyper_features: torch.Tensor, s_q: torch.Tensor | None = None) -> torch.Tensor | None:
        raw_gate = self._raw_householder_gate(hyper_features)
        if raw_gate is None:
            return None
        gate = raw_gate
        reliability_multiplier = self._householder_gate_reliability_multiplier(hyper_features)
        if reliability_multiplier is not None:
            gate = gate * reliability_multiplier
        raw_backoff_multiplier = self._householder_gate_raw_backoff_multiplier(raw_gate)
        if raw_backoff_multiplier is not None:
            gate = gate * raw_backoff_multiplier
        _, residual_selector_multiplier, _ = self._householder_gate_residual_selector(hyper_features)
        if residual_selector_multiplier is not None:
            gate = gate * residual_selector_multiplier
        risk_multiplier = self._householder_gate_risk_multiplier(s_q)
        if risk_multiplier is not None:
            gate = gate * risk_multiplier
        strength_backoff_multiplier = self._householder_gate_strength_backoff_multiplier(gate)
        if strength_backoff_multiplier is not None:
            gate = gate * strength_backoff_multiplier
        return gate

    def _stage_gate(self, hyper_features: torch.Tensor) -> torch.Tensor | None:
        if not self.stage_gate_enabled:
            return None
        b, _, h, w = hyper_features.shape
        gate = self.gate_head(hyper_features)
        gate = gate.view(b, self.num_groups, self.num_stages, h, w)
        return torch.sigmoid(gate.permute(0, 1, 3, 4, 2).contiguous())

    def _conditioned_rvq(
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

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        image_hw = (x.shape[-2], x.shape[-1])
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_features = self.h_s(z_hat)

        if self.variant == "scalar":
            gaussian_params = self.scalar_head(hyper_features)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            rvq_stats: dict[str, torch.Tensor] = {}
            commit_loss = x.new_tensor(0.0)
            indices: list[torch.Tensor] = []
            bpp_y_index = None
            conditioning_tensors: dict[str, torch.Tensor] = {}
        else:
            y_hat, indices, commit_loss, rvq_stats, conditioning_tensors = self._conditioned_rvq(
                y,
                hyper_features,
                image_hw,
            )
            y_likelihoods = None
            if self.index_entropy_model is not None:
                bpp_y_index, _ = self.index_entropy_model(hyper_features, indices, image_hw)
            else:
                bpp_y_index = rvq_stats["fixed_bpp"]

        x_hat = self.g_s(y_hat)
        out: dict[str, object] = {
            "x_hat": x_hat,
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
        if y_likelihoods is not None:
            out["likelihoods"]["y"] = y_likelihoods
        return out

