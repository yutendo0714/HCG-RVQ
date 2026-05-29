from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class IndexEntropyModel(nn.Module):
    """Hyperprior-conditioned categorical prior over RVQ indices."""

    def __init__(
        self,
        hyper_channels: int,
        num_groups: int,
        num_stages: int,
        codebook_size: int,
        hidden_channels: int = 192,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_stages = num_stages
        self.codebook_size = codebook_size
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hyper_channels, hidden_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, num_groups * codebook_size, 1),
                )
                for _ in range(num_stages)
            ]
        )

    def forward(
        self,
        hyper_features: torch.Tensor,
        indices: list[torch.Tensor],
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, _, h, w = hyper_features.shape
        image_h, image_w = image_hw
        ng = self.num_groups
        k = self.codebook_size
        total_ce = hyper_features.new_tensor(0.0)
        logits_list: list[torch.Tensor] = []

        for stage, head in enumerate(self.heads):
            logits = head(hyper_features)
            logits = logits.view(b, ng, k, h, w).permute(0, 1, 3, 4, 2).contiguous()
            logits_list.append(logits)
            ce = F.cross_entropy(logits.reshape(-1, k), indices[stage].reshape(-1), reduction="mean")
            total_ce = total_ce + ce

        bits_per_symbol_sum = total_ce / math.log(2.0)
        num_symbols_per_stage = b * ng * h * w
        num_pixels = b * image_h * image_w
        bpp = bits_per_symbol_sum * num_symbols_per_stage / num_pixels
        return bpp, logits_list

