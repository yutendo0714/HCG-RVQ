#!/usr/bin/env python3
"""Evaluate a pretrained DCAE checkpoint on a small path-fixed subset.

This is a baseline reproduction smoke for the SOTA/backbone lane. It mirrors
the official DCAE forward-eval path, but saves per-image metrics and latent
feature summaries for later same-backbone HCG plug-in comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DCAE_ROOT = ROOT / "third_party" / "DCAE"
if str(DCAE_ROOT) not in sys.path:
    sys.path.insert(0, str(DCAE_ROOT))

from models import DCAE


ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e155_dcae_lambda0018_kodak_first4_baseline_smoke"


def pad(x: torch.Tensor, p: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse <= 0.0 else -10.0 * math.log10(mse)


def compute_msssim_db(a: torch.Tensor, b: torch.Tensor) -> float:
    score = ms_ssim(a, b, data_range=1.0).item()
    return -10.0 * math.log10(max(1.0 - score, 1e-12))


def compute_bpp(out_net: dict[str, object], num_pixels: int) -> float:
    likelihoods = out_net["likelihoods"]
    assert isinstance(likelihoods, dict)
    total = 0.0
    for tensor in likelihoods.values():
        assert torch.is_tensor(tensor)
        total += torch.log(tensor).sum().item() / (-math.log(2.0) * num_pixels)
    return total


def finite_ratio(tensor: torch.Tensor) -> float:
    return float(torch.isfinite(tensor).float().mean().detach().cpu())


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float | list[int]]:
    data = tensor.detach()
    finite = torch.isfinite(data)
    safe = data[finite]
    row: dict[str, float | list[int]] = {
        f"{prefix}_shape": list(data.shape),
        f"{prefix}_finite_ratio": float(finite.float().mean().cpu()),
    }
    if safe.numel() > 0:
        row.update(
            {
                f"{prefix}_mean": float(safe.mean().cpu()),
                f"{prefix}_std": float(safe.std(unbiased=False).cpu()),
                f"{prefix}_abs_mean": float(safe.abs().mean().cpu()),
                f"{prefix}_min": float(safe.min().cpu()),
                f"{prefix}_max": float(safe.max().cpu()),
            }
        )
    return row


def load_dcae(checkpoint_path: Path, device: torch.device) -> DCAE:
    net = DCAE().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    net.load_state_dict(state_dict)
    return net


def image_paths(data: Path, max_images: int | None) -> list[Path]:
    paths = sorted(
        [
            path
            for path in data.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    return paths if max_images is None else paths[:max_images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="third_party/checkpoints/dcae/dcae_lambda0018_mse.pth.tar")
    parser.add_argument("--data", default="experiments/data/kodak_first4")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--pad", type=int, default=128)
    parser.add_argument("--output-prefix", default=str(PREFIX))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    data_path = (ROOT / args.data).resolve()
    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    net = load_dcae(checkpoint_path, device)
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for idx, path in enumerate(image_paths(data_path, args.max_images)):
            x = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            x_padded, padding = pad(x, args.pad)
            out = net(x_padded)
            x_hat = crop(out["x_hat"], padding).clamp_(0.0, 1.0)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            row: dict[str, object] = {
                "index": idx,
                "path": str(path),
                "filename": path.name,
                "height": int(x.shape[2]),
                "width": int(x.shape[3]),
                "padded_height": int(x_padded.shape[2]),
                "padded_width": int(x_padded.shape[3]),
                "bpp": compute_bpp(out, num_pixels),
                "psnr": compute_psnr(x, x_hat),
                "ms_ssim_db": compute_msssim_db(x, x_hat),
                "x_hat_finite_ratio": finite_ratio(x_hat),
                "nonfinite": int(finite_ratio(x_hat) < 1.0),
            }
            para = out.get("para", {})
            if isinstance(para, dict):
                for key in ("y", "means", "scales"):
                    value = para.get(key)
                    if torch.is_tensor(value):
                        row.update(tensor_stats(key, value))
            likelihoods = out.get("likelihoods", {})
            if isinstance(likelihoods, dict):
                for key, value in likelihoods.items():
                    if torch.is_tensor(value):
                        row[f"{key}_likelihood_finite_ratio"] = finite_ratio(value)
                        row[f"{key}_likelihood_min"] = float(value.detach().min().cpu())
            rows.append(row)

    metric_keys = ["bpp", "psnr", "ms_ssim_db"]
    summary = {
        "experiment": "E155 DCAE pretrained baseline smoke",
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "num_images": len(rows),
        "nonfinite_rows": sum(int(row["nonfinite"]) for row in rows),
        "averages": {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in metric_keys
            if rows
        },
        "decision": (
            "This is a pretrained DCAE baseline reproduction smoke on a "
            "path-fixed image set, not a full multi-rate SOTA benchmark."
        ),
    }

    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if rows:
        with (prefix.with_suffix(".csv")).open("w", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# E155 DCAE Pretrained Baseline Smoke",
        "",
        f"- device: `{summary['device']}`",
        f"- images: `{summary['num_images']}`",
        f"- nonfinite rows: `{summary['nonfinite_rows']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary["averages"].items():
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.extend(
        [
            "",
            "Decision: " + summary["decision"],
            "",
            "Per-image metrics and feature summaries are in the CSV artifact.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
