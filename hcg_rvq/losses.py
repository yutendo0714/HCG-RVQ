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
        mse_scale: float = 255.0 * 255.0,
    ) -> None:
        super().__init__()
        self.lambda_rd = lambda_rd
        self.beta_commit = beta_commit
        self.rho_gate = rho_gate
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

        loss = bpp_total + self.lambda_rd * self.mse_scale * mse
        loss = loss + self.beta_commit * commit_loss + self.rho_gate * gate_loss

        return {
            "loss": loss,
            "bpp_total": bpp_total.detach(),
            "bpp_y": bpp_y.detach(),
            "bpp_z": bpp_z.detach(),
            "mse": mse.detach(),
            "psnr": (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).detach(),
            "commit_loss": commit_loss.detach(),
            "gate_loss": gate_loss.detach(),
        }

