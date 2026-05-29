from __future__ import annotations

import torch


def householder_transform(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply a batched Householder reflection along the last dimension."""
    dot = (x * v).sum(dim=-1, keepdim=True)
    norm = (v * v).sum(dim=-1, keepdim=True).clamp_min(eps)
    return x - 2.0 * dot / norm * v

