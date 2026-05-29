from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

from hcg_rvq.entropy import IndexEntropyModel
from hcg_rvq.quantizers import ResidualVectorQuantizer, householder_transform


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
        scale_min: float = 0.05,
        scale_max: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(entropy_bottleneck_channels=N)
        if M % group_size != 0:
            raise ValueError(f"M={M} must be divisible by group_size={group_size}")

        self.N = N
        self.M = M
        self.variant = variant
        self.group_size = group_size
        self.num_groups = M // group_size
        self.num_stages = num_stages
        self.codebook_size = codebook_size
        self.index_prior_enabled = index_prior_enabled
        self.stage_gate_enabled = stage_gate_enabled
        self.scale_min = scale_min
        self.scale_max = scale_max
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
        self.gate_head = nn.Conv2d(N, self.num_groups * num_stages, 1)

        self.rvq = ResidualVectorQuantizer(
            dim=group_size,
            num_stages=num_stages,
            codebook_size=codebook_size,
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

    def _to_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.num_groups, self.group_size, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def _from_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, ng, h, w, g = x.shape
        return x.permute(0, 1, 4, 2, 3).contiguous().view(b, ng * g, h, w)

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
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "global_rvq":
            gate = self._stage_gate(hyper_features)
            y_hat, indices, commit_loss, stats = self.rvq(y, image_hw=image_hw, gate=gate)
            if gate is not None:
                stats["avg_gate"] = gate.mean()
            return y_hat, indices, commit_loss, stats

        mu_q = self.mu_head(hyper_features)
        log_s_q = self.log_s_head(hyper_features)
        s_q = (F.softplus(log_s_q) + self.eps).clamp(self.scale_min, self.scale_max)
        y_norm = (y - mu_q) / s_q

        if self.variant in {"hcg_rvq_h", "hcg_rvq_h_gate"}:
            v = self.householder_head(hyper_features)
            y_norm_g = self._to_grouped(y_norm)
            v_g = F.normalize(self._to_grouped(v), dim=-1, eps=self.eps)
            u = self._from_grouped(householder_transform(y_norm_g, v_g, eps=self.eps))
            stats_extra = {
                "householder_v_norm": v_g.norm(dim=-1).mean(),
            }
        else:
            v_g = None
            u = y_norm
            stats_extra = {}

        gate = self._stage_gate(hyper_features)
        u_hat, indices, commit_loss, stats = self.rvq(u, image_hw=image_hw, gate=gate)

        if v_g is not None:
            u_hat_g = self._to_grouped(u_hat)
            y_norm_hat = self._from_grouped(householder_transform(u_hat_g, v_g, eps=self.eps))
        else:
            y_norm_hat = u_hat

        y_hat = mu_q + s_q * y_norm_hat
        stats.update(stats_extra)
        stats.update(
            {
                "s_q_mean": s_q.mean(),
                "s_q_min": s_q.min(),
                "s_q_max": s_q.max(),
            }
        )
        if gate is not None:
            stats["avg_gate"] = gate.mean()
        return y_hat, indices, commit_loss, stats

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
        else:
            y_hat, indices, commit_loss, rvq_stats = self._conditioned_rvq(y, hyper_features, image_hw)
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
            "bpp_y_index": bpp_y_index,
        }
        if y_likelihoods is not None:
            out["likelihoods"]["y"] = y_likelihoods
        return out

