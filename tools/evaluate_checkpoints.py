from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def checkpoint_step(path: Path) -> int:
    if path.name == "checkpoint_latest.pth.tar":
        return 10**12
    match = re.search(r"checkpoint_step_(\d+)\.pth\.tar$", path.name)
    if match:
        return int(match.group(1))
    return 10**12 - 1


def average(values: list[float]) -> float:
    if not values:
        raise RuntimeError("no values to average")
    return sum(values) / len(values)


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | str]:
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_info = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    criterion = RateDistortionLoss(**config.get("loss", {}))

    totals: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc=checkpoint_path.name, dynamic_ncols=True):
            x = batch.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            totals["loss"].append(float(losses["loss"].cpu()))
            totals["bpp"].append(float(losses["bpp_total"].cpu()))
            totals["bpp_y"].append(float(losses["bpp_y"].cpu()))
            totals["bpp_z"].append(float(losses["bpp_z"].cpu()))
            totals["mse"].append(float(losses["mse"].cpu()))
            totals["psnr"].append(float(compute_psnr(x, output["x_hat"]).cpu()))
            totals["ms_ssim"].append(float(compute_msssim(x, output["x_hat"]).cpu()))

    mse_scale = config.get("loss", {}).get("mse_scale", 255.0 * 255.0)
    lambda_rd = config.get("loss", {}).get("lambda_rd")
    if lambda_rd is None:
        raise KeyError("loss.lambda_rd is required to compute validation RD score")

    bpp = average(totals["bpp"])
    mse = average(totals["mse"])
    return {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint_step(checkpoint_path),
        "loss": average(totals["loss"]),
        "rd_score": bpp + float(lambda_rd) * float(mse_scale) * mse,
        "bpp": bpp,
        "bpp_y": average(totals["bpp_y"]),
        "bpp_z": average(totals["bpp_z"]),
        "mse": mse,
        "psnr": average(totals["psnr"]),
        "ms_ssim": average(totals["ms_ssim"]),
        "missing_keys": ";".join(load_info.missing_keys),
        "unexpected_keys": ";".join(load_info.unexpected_keys),
    }


def write_outputs(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(rows, indent=2) + "\n")
        return

    fieldnames = ["checkpoint", "step", "loss", "rd_score", "bpp", "bpp_y", "bpp_z", "mse", "psnr", "ms_ssim", "missing_keys", "unexpected_keys"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints and select the best validation RD point.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pattern", default="checkpoint*.pth.tar")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="optional .csv or .json output path")
    parser.add_argument("--sort-by", default="rd_score", choices=["rd_score", "loss", "bpp", "psnr", "ms_ssim"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config.get("eval", {})
    root = args.data_root or data_cfg.get("root", "/dpl/kodak")
    max_images = args.max_images if args.max_images is not None else data_cfg.get("max_images")
    start_index = args.start_index if args.start_index is not None else data_cfg.get("start_index", 0)
    patch_size = args.patch_size if args.patch_size is not None else data_cfg.get("patch_size")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.pattern), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints matched {checkpoint_dir / args.pattern}")

    dataset = ImageFolderDataset(
        [root],
        patch_size=patch_size,
        training=False,
        max_images=max_images,
        start_index=start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_cfg.get("num_workers", 2))
    rows = [evaluate_checkpoint(config, ckpt, loader, device) for ckpt in checkpoints]

    reverse = args.sort_by in {"psnr", "ms_ssim"}
    best = sorted(rows, key=lambda row: float(row[args.sort_by]), reverse=reverse)[0]

    for row in rows:
        print(
            f"{Path(str(row['checkpoint'])).name}: "
            f"rd={row['rd_score']:.6f} bpp={row['bpp']:.6f} "
            f"psnr={row['psnr']:.6f} ms_ssim={row['ms_ssim']:.6f}"
        )
    print(f"best_by_{args.sort_by}: {best['checkpoint']}")

    if args.output:
        write_outputs(rows, Path(args.output))


if __name__ == "__main__":
    main()
