from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x_hat.clamp(0, 1), x.clamp(0, 1))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def compute_msssim(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch_msssim import ms_ssim

        return ms_ssim(x_hat.clamp(0, 1), x.clamp(0, 1), data_range=1.0)
    except Exception:
        return x.new_tensor(float("nan"))

