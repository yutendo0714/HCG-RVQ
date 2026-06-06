from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_msssim, compute_psnr
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("eval", {})
    root = args.data_root or data_cfg.get("root", "/dpl/kodak")

    dataset = ImageFolderDataset(
        [root],
        patch_size=data_cfg.get("patch_size"),
        training=False,
        max_images=data_cfg.get("max_images"),
        start_index=data_cfg.get("start_index", 0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_cfg.get("num_workers", 2))
    model = build_model(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    eval_loss_cfg = dict(config.get("loss", {}))
    eval_loss_cfg["rho_householder_reliability_teacher"] = 0.0
    eval_loss_cfg["rho_householder_residual_selector_teacher"] = 0.0
    eval_loss_cfg["rho_householder_residual_selector_noop"] = 0.0
    for anchor_name in (
        "rho_anchor_mu",
        "rho_anchor_log_s",
        "rho_anchor_u",
        "rho_anchor_y_hat",
        "rho_anchor_selected_distortion_margin",
    ):
        eval_loss_cfg[anchor_name] = 0.0
    criterion = RateDistortionLoss(**eval_loss_cfg)

    totals: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            x = batch.to(device)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            totals["bpp"].append(float(losses["bpp_total"].cpu()))
            totals["bpp_y"].append(float(losses["bpp_y"].cpu()))
            totals["bpp_z"].append(float(losses["bpp_z"].cpu()))
            totals["psnr"].append(float(compute_psnr(x, output["x_hat"]).cpu()))
            totals["ms_ssim"].append(float(compute_msssim(x, output["x_hat"]).cpu()))

    for key, values in totals.items():
        print(f"{key}: {sum(values) / len(values):.6f}")


if __name__ == "__main__":
    main()

