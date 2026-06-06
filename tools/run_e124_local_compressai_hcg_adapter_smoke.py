#!/usr/bin/env python3
"""Smoke-test standalone HCG adapters on local CompressAI backbones."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from compressai.zoo import cheng2020_attn, mbt2018_mean

from hcg_rvq.quantizers import HCGQuantizerAdapter


OUT_PREFIX = ROOT / "experiments" / "analysis" / "e124_local_compressai_hcg_adapter_smoke"


def tensor_nonfinite_count(value: torch.Tensor) -> int:
    return int((~torch.isfinite(value)).sum().detach().cpu())


def scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def smoke_one(name: str, factory, x: torch.Tensor, device: torch.device) -> dict[str, object]:
    model = factory(quality=1, pretrained=False).to(device).eval()
    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)
        z_hat, z_likelihoods = model.entropy_bottleneck(z)
        hyper_features = model.h_s(z_hat)
        adapter = HCGQuantizerAdapter(
            latent_channels=int(y.shape[1]),
            hyper_channels=int(hyper_features.shape[1]),
            variant="hcg_rvq_h",
            group_size=64,
            num_stages=1,
            codebook_size=128,
        ).to(device).eval()
        y_hat, indices, commit_loss, rvq_stats, conditioning_tensors = adapter(
            y,
            hyper_features,
            (int(x.shape[-2]), int(x.shape[-1])),
        )
        x_hat = model.g_s(y_hat)

    stat_scalars = {
        key: scalar(value)
        for key, value in rvq_stats.items()
        if scalar(value) is not None
    }
    conditioning_shapes = {
        key: list(value.shape)
        for key, value in conditioning_tensors.items()
        if torch.is_tensor(value)
    }
    nonfinite = (
        tensor_nonfinite_count(y)
        + tensor_nonfinite_count(hyper_features)
        + tensor_nonfinite_count(y_hat)
        + tensor_nonfinite_count(x_hat)
        + tensor_nonfinite_count(commit_loss)
        + tensor_nonfinite_count(z_likelihoods)
    )
    for value in rvq_stats.values():
        if torch.is_tensor(value):
            nonfinite += tensor_nonfinite_count(value)
    result = {
        "name": name,
        "class": type(model).__name__,
        "adapter_class": type(adapter).__name__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "z_shape": list(z.shape),
        "hyper_features_shape": list(hyper_features.shape),
        "y_hat_shape": list(y_hat.shape),
        "x_hat_shape": list(x_hat.shape),
        "num_indices": len(indices),
        "index_shapes": [list(index.shape) for index in indices],
        "commit_loss": float(commit_loss.detach().cpu()),
        "rvq_stats": stat_scalars,
        "conditioning_shapes": conditioning_shapes,
        "adapter_num_parameters": sum(p.numel() for p in adapter.parameters()),
        "nonfinite": int(nonfinite),
    }
    result["pass"] = (
        result["y_hat_shape"] == result["y_shape"]
        and result["x_hat_shape"] == result["x_shape"]
        and result["num_indices"] == 1
        and result["nonfinite"] == 0
        and math.isfinite(result["commit_loss"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    x = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    rows = [
        smoke_one("mbt2018_mean", mbt2018_mean, x, device),
        smoke_one("cheng2020_attn", cheng2020_attn, x, device),
    ]
    result = {
        "experiment": "E124 local CompressAI HCG adapter smoke",
        "status": "pass" if all(row["pass"] for row in rows) else "fail",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_shape": list(x.shape),
        "rows": rows,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E124 Local CompressAI HCG Adapter Smoke",
        "",
        f"- Status: `{result['status']}`",
        f"- Device: `{result['device']}`, CUDA_VISIBLE_DEVICES=`{result['cuda_visible_devices']}`",
        "",
        "## Backbones",
        "",
    ]
    for row in rows:
        lines.append(
            "- `{name}`: pass `{passed}`, y `{y}`, h_s `{hyper}`, y_hat `{y_hat}`, x_hat `{x_hat}`, nonfinite `{nonfinite}`.".format(
                name=row["name"],
                passed=row["pass"],
                y=row["y_shape"],
                hyper=row["hyper_features_shape"],
                y_hat=row["y_hat_shape"],
                x_hat=row["x_hat_shape"],
                nonfinite=row["nonfinite"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The standalone adapter accepts explicit latent and hyper feature channel counts and can run inside both local CompressAI backbone contracts. This is a shape/numerics smoke only; it does not claim RD quality before training or checkpoint transfer.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
