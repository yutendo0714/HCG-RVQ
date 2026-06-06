from __future__ import annotations

import argparse
import csv
import json
import sys
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


def load_eval_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[dict, torch.nn.Module]:
    config = load_config(config_path)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return config, model


def evaluate_one(
    model: torch.nn.Module,
    criterion: RateDistortionLoss,
    config: dict,
    x: torch.Tensor,
) -> dict[str, float]:
    x_pad, hw = pad_to_multiple(x)
    output = model(x_pad)
    output["x_hat"] = crop_to_hw(output["x_hat"], hw)
    losses = criterion(output, x)
    mse = float(losses["mse"].cpu())
    bpp = float(losses["bpp_total"].cpu())
    lambda_rd = float(config.get("loss", {})["lambda_rd"])
    mse_scale = float(config.get("loss", {}).get("mse_scale", 255.0 * 255.0))
    return {
        "loss": float(losses["loss"].cpu()),
        "rd_score": bpp + lambda_rd * mse_scale * mse,
        "bpp": bpp,
        "bpp_y": float(losses["bpp_y"].cpu()),
        "bpp_z": float(losses["bpp_z"].cpu()),
        "mse": mse,
        "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
        "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two checkpoints image-by-image on the same dataset slice.")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--a-config", required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--b-config", required=True)
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config_a, model_a = load_eval_model(args.a_config, args.a_checkpoint, device)
    config_b, model_b = load_eval_model(args.b_config, args.b_checkpoint, device)
    criterion_a = RateDistortionLoss(**config_a.get("loss", {}))
    criterion_b = RateDistortionLoss(**config_b.get("loss", {}))

    dataset = ImageFolderDataset(
        [args.data_root],
        patch_size=args.patch_size,
        training=False,
        max_images=args.max_images,
        start_index=args.start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config_a.get("eval", {}).get("num_workers", 2))

    rows: list[dict[str, float | str | int]] = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="per-image", dynamic_ncols=True)):
            x = batch.to(device, non_blocking=True)
            metrics_a = evaluate_one(model_a, criterion_a, config_a, x)
            metrics_b = evaluate_one(model_b, criterion_b, config_b, x)
            row: dict[str, float | str | int] = {
                "index": index,
                "path": str(dataset.paths[index]),
            }
            for prefix, metrics in [(args.a_name, metrics_a), (args.b_name, metrics_b)]:
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            for key in metrics_a:
                row[f"delta_{key}"] = metrics_b[key] - metrics_a[key]
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "num_images": len(rows),
        "mean_delta_rd_score": average([float(row["delta_rd_score"]) for row in rows]),
        "mean_delta_bpp": average([float(row["delta_bpp"]) for row in rows]),
        "mean_delta_psnr": average([float(row["delta_psnr"]) for row in rows]),
        "mean_delta_ms_ssim": average([float(row["delta_ms_ssim"]) for row in rows]),
        "b_better_rd_count": sum(1 for row in rows if float(row["delta_rd_score"]) < 0.0),
        "b_worse_rd_count": sum(1 for row in rows if float(row["delta_rd_score"]) > 0.0),
        "top_b_worse_by_rd": sorted(rows, key=lambda row: float(row["delta_rd_score"]), reverse=True)[:20],
        "top_b_better_by_rd": sorted(rows, key=lambda row: float(row["delta_rd_score"]))[:20],
    }
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
