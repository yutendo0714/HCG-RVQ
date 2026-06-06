#!/usr/bin/env python3
"""Forward-only DCAE + HCG adapter smoke.

This is an architecture/provenance test, not a quality claim. It checks whether
the HCG-RVQ adapter can consume DCAE hyper features and feed DCAE's synthesis
transform with finite tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DCAE_ROOT = ROOT / "third_party" / "DCAE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DCAE_ROOT) not in sys.path:
    sys.path.insert(0, str(DCAE_ROOT))

from hcg_rvq.quantizers.hcg_adapter import HCGQuantizerAdapter
from models import DCAE
from models.dcae import ste_round


ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e153_dcae_hcg_adapter_smoke"


def scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def finite_ratio(tensor: torch.Tensor) -> float:
    return float(torch.isfinite(tensor).float().mean().detach().cpu())


def tensor_summary(prefix: str, tensor: torch.Tensor) -> dict[str, float | list[int]]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    safe = detached[finite]
    out: dict[str, float | list[int]] = {
        f"{prefix}_shape": list(detached.shape),
        f"{prefix}_finite_ratio": float(finite.float().mean().cpu()),
    }
    if safe.numel() > 0:
        out.update(
            {
                f"{prefix}_mean": float(safe.mean().cpu()),
                f"{prefix}_std": float(safe.std(unbiased=False).cpu()),
                f"{prefix}_abs_mean": float(safe.abs().mean().cpu()),
                f"{prefix}_min": float(safe.min().cpu()),
                f"{prefix}_max": float(safe.max().cpu()),
            }
        )
    return out


def build_adapter(
    *,
    variant: str,
    latent_channels: int,
    hyper_channels: int,
    group_size: int,
    num_stages: int,
    codebook_size: int,
    active_geometry: bool,
) -> HCGQuantizerAdapter:
    return HCGQuantizerAdapter(
        latent_channels=latent_channels,
        hyper_channels=hyper_channels,
        variant=variant,
        group_size=group_size,
        num_stages=num_stages,
        codebook_size=codebook_size,
        householder_bias_init_scale=0.01 if active_geometry else 0.0,
        householder_gate_enabled=active_geometry,
        householder_gate_max=0.45,
        householder_gate_init=0.25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--output-prefix", default=str(PREFIX))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    x = torch.rand(1, 3, args.height, args.width, device=device)

    net = DCAE().to(device).eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        baseline = net(x)
        y = net.g_a(x)
        z = net.h_a(y)
        _, z_likelihoods = net.entropy_bottleneck(z)
        z_offset = net.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        latent_scales = net.h_z_s1(z_hat)
        latent_means = net.h_z_s2(z_hat)
        hyper_features = torch.cat([latent_scales, latent_means], dim=1)

        base_row: dict[str, object] = {
            "case": "dcae_baseline_forward",
            "device": str(device),
            "input_hw": [args.height, args.width],
            "x_hat_finite_ratio": finite_ratio(baseline["x_hat"]),
            "y_finite_ratio": finite_ratio(y),
            "z_likelihoods_finite_ratio": finite_ratio(z_likelihoods),
            "y_likelihoods_finite_ratio": finite_ratio(baseline["likelihoods"]["y"]),
            "x_hat_shape": list(baseline["x_hat"].shape),
            "y_shape": list(y.shape),
            "hyper_features_shape": list(hyper_features.shape),
            "nonfinite": int(
                finite_ratio(baseline["x_hat"]) < 1.0
                or finite_ratio(y) < 1.0
                or finite_ratio(hyper_features) < 1.0
            ),
        }
        rows.append(base_row)

        adapter_cases = [
            ("hcs_adapter_identity_init", "hcs_rvq", False),
            ("hcg_adapter_active_geometry", "hcg_rvq_h", True),
        ]
        for case_name, variant, active in adapter_cases:
            torch.manual_seed(args.seed)
            adapter = build_adapter(
                variant=variant,
                latent_channels=y.shape[1],
                hyper_channels=hyper_features.shape[1],
                group_size=args.group_size,
                num_stages=args.num_stages,
                codebook_size=args.codebook_size,
                active_geometry=active,
            ).to(device).eval()
            y_hat, indices, commit_loss, rvq_stats, conditioning = adapter(
                y,
                hyper_features,
                (args.height, args.width),
            )
            x_hat = net.g_s(y_hat)
            row: dict[str, object] = {
                "case": case_name,
                "variant": variant,
                "active_geometry": active,
                "device": str(device),
                "input_hw": [args.height, args.width],
                "x_hat_shape": list(x_hat.shape),
                "y_hat_shape": list(y_hat.shape),
                "num_indices": len(indices),
                "commit_loss": float(commit_loss.detach().cpu()),
                "x_hat_finite_ratio": finite_ratio(x_hat),
                "y_hat_finite_ratio": finite_ratio(y_hat),
                "nonfinite": int(finite_ratio(x_hat) < 1.0 or finite_ratio(y_hat) < 1.0),
            }
            row.update(tensor_summary("adapter_y_hat", y_hat))
            row.update(tensor_summary("adapter_x_hat", x_hat))
            for key, value in rvq_stats.items():
                val = scalar(value)
                if val is not None and math.isfinite(val):
                    row[f"rvq_{key}"] = val
            for key, value in conditioning.items():
                if torch.is_tensor(value):
                    row[f"conditioning_{key}_shape"] = list(value.shape)
                    row[f"conditioning_{key}_finite_ratio"] = finite_ratio(value)
            rows.append(row)

    summary = {
        "experiment": "E153 DCAE HCG adapter forward-only smoke",
        "device": str(device),
        "seed": args.seed,
        "input_hw": [args.height, args.width],
        "rows": rows,
        "nonfinite_rows": sum(int(row.get("nonfinite", 0)) for row in rows),
        "decision": (
            "Pass only means DCAE exposes a compatible HCG adapter boundary. "
            "It is not an RD or SOTA quality result."
        ),
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E153 DCAE HCG Adapter Smoke",
        "",
        "Forward-only architecture smoke for DCAE plus the local HCG adapter.",
        "",
        f"- device: `{summary['device']}`",
        f"- input: `{args.height}x{args.width}`",
        f"- nonfinite rows: `{summary['nonfinite_rows']}`",
        "",
        "| case | x_hat finite | y/y_hat finite | qMSE | dead | perplexity | householder strength | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        y_key = "y_hat_finite_ratio" if "y_hat_finite_ratio" in row else "y_finite_ratio"
        lines.append(
            "| {case} | {xfin:.6f} | {yfin:.6f} | {qmse} | {dead} | {perp} | {strength} | {delta} |".format(
                case=row["case"],
                xfin=float(row.get("x_hat_finite_ratio", float("nan"))),
                yfin=float(row.get(y_key, float("nan"))),
                qmse="n/a" if "rvq_latent_quant_mse" not in row else f"{float(row['rvq_latent_quant_mse']):.6f}",
                dead="n/a" if "rvq_dead_code_ratio" not in row else f"{float(row['rvq_dead_code_ratio']):.6f}",
                perp="n/a" if "rvq_perplexity" not in row else f"{float(row['rvq_perplexity']):.6f}",
                strength="n/a" if "rvq_householder_strength" not in row else f"{float(row['rvq_householder_strength']):.6f}",
                delta="n/a" if "rvq_householder_delta_rms" not in row else f"{float(row['rvq_householder_delta_rms']):.6f}",
            )
        )
    lines.extend(["", "Decision: " + str(summary["decision"])])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
