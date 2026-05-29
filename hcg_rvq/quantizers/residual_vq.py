from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RVQStats:
    latent_quant_mse: torch.Tensor
    fixed_bpp: torch.Tensor
    perplexity: torch.Tensor
    dead_code_ratio: torch.Tensor
    stage_entropy: torch.Tensor


class ResidualVectorQuantizer(nn.Module):
    """Grouped residual vector quantizer for LIC latents.

    Input tensors are `[B, C, H, W]`. Channels are split into groups of
    `dim`, and one RVQ stack is shared across all groups and spatial positions.
    """

    def __init__(
        self,
        dim: int,
        num_stages: int = 2,
        codebook_size: int = 256,
        init_scale: float = 0.02,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebooks = nn.Parameter(torch.randn(num_stages, codebook_size, dim) * init_scale)

    def _to_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % self.dim != 0:
            raise ValueError(f"channels={c} must be divisible by group dim={self.dim}")
        ng = c // self.dim
        return x.view(b, ng, self.dim, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def _from_grouped(self, x: torch.Tensor) -> torch.Tensor:
        b, ng, h, w, g = x.shape
        return x.permute(0, 1, 4, 2, 3).contiguous().view(b, ng * g, h, w)

    def forward(
        self,
        x: torch.Tensor,
        image_hw: tuple[int, int] | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        xg = self._to_grouped(x)
        residual = xg
        quantized_sum = torch.zeros_like(xg)
        indices: list[torch.Tensor] = []

        for stage in range(self.num_stages):
            codebook = self.codebooks[stage]
            dist = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * torch.matmul(residual, codebook.t())
                + codebook.pow(2).sum(dim=-1)
            )
            idx = dist.argmin(dim=-1)
            q = F.embedding(idx, codebook)

            if gate is not None:
                q_used = gate[..., stage].unsqueeze(-1) * q
            else:
                q_used = q

            quantized_sum = quantized_sum + q_used
            residual = residual - q.detach()
            indices.append(idx)

        x_hat_g = xg + (quantized_sum - xg).detach()
        x_hat = self._from_grouped(x_hat_g)

        commit_loss = F.mse_loss(xg.detach(), quantized_sum) + self.commitment_weight * F.mse_loss(
            xg, quantized_sum.detach()
        )
        stats = self._stats(xg, quantized_sum, indices, x.shape, image_hw)
        return x_hat, indices, commit_loss, stats

    def _stats(
        self,
        xg: torch.Tensor,
        qg: torch.Tensor,
        indices: list[torch.Tensor],
        input_shape: tuple[int, int, int, int],
        image_hw: tuple[int, int] | None,
    ) -> dict[str, torch.Tensor]:
        b, c, h, w = input_shape
        if image_hw is None:
            image_h, image_w = h, w
        else:
            image_h, image_w = image_hw

        num_pixels = b * image_h * image_w
        num_symbols_per_stage = indices[0].numel()
        fixed_bits = self.num_stages * num_symbols_per_stage * math.log2(self.codebook_size)
        fixed_bpp = xg.new_tensor(fixed_bits / num_pixels)

        perplexities = []
        dead_ratios = []
        entropies = []
        for idx in indices:
            counts = torch.bincount(idx.reshape(-1), minlength=self.codebook_size).to(xg.dtype)
            probs = counts / counts.sum().clamp_min(1.0)
            nz_probs = probs[probs > 0]
            entropy = -(nz_probs * nz_probs.log2()).sum()
            entropies.append(entropy)
            perplexities.append(torch.pow(xg.new_tensor(2.0), entropy))
            dead_ratios.append((counts == 0).to(xg.dtype).mean())

        return {
            "latent_quant_mse": F.mse_loss(qg.detach(), xg.detach()),
            "fixed_bpp": fixed_bpp,
            "perplexity": torch.stack(perplexities).mean(),
            "dead_code_ratio": torch.stack(dead_ratios).mean(),
            "stage_entropy": torch.stack(entropies).mean(),
        }

