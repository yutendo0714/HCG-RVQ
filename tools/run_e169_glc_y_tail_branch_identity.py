#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, get_state_dict  # noqa: E402
from tools.run_e162_glc_pretrained_baseline import list_images, load_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e169_glc_y_tail_branch_identity_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def finite_max_abs(x: torch.Tensor) -> float:
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(x[finite].abs().max().item())


def sum_float(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach().float()).sum().item())


def ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den):
        return float("nan")
    if den <= 0.0:
        return 0.0
    return float(num / den)


def make_group_branch_mask(mask: torch.Tensor, group_size: int, active_groups: set[int], active: bool) -> torch.Tensor:
    if not active:
        return torch.zeros_like(mask)
    _, channels, _, _ = mask.shape
    if channels % group_size != 0:
        raise ValueError(f"channels {channels} is not divisible by group size {group_size}")
    channel_gate = torch.zeros((1, channels, 1, 1), dtype=mask.dtype, device=mask.device)
    for group in active_groups:
        start = group * group_size
        end = start + group_size
        if 0 <= start < channels:
            channel_gate[:, start:min(end, channels)] = 1
    return mask * channel_gate


def process_tail_branch_identity(
    self: GLC_Image,
    y: torch.Tensor,
    scales: torch.Tensor,
    means: torch.Tensor,
    mask: torch.Tensor,
    branch_mask: torch.Tensor,
    stats: dict[str, float],
    part_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scales_hat = scales * mask
    means_hat = means * mask
    y_res = (y - means_hat) * mask
    y_q = self.quant(y_res)
    y_hat = y_q + means_hat

    qerr = (y_q - y_res) * mask
    active = branch_mask.to(torch.bool)
    valid = mask.to(torch.bool)
    inactive = valid & (~active)

    active_count = float(active.sum().item())
    valid_count = float(valid.sum().item())
    total_count = float(mask.numel())
    res_energy_total = sum_float((y_res * valid) ** 2)
    qerr_energy_total = sum_float((qerr * valid) ** 2)
    res_energy_active = sum_float((y_res * active) ** 2)
    qerr_energy_active = sum_float((qerr * active) ** 2)
    nonzero_total = float(((y_q != 0) & valid).sum().item())
    nonzero_active = float(((y_q != 0) & active).sum().item())
    nonzero_inactive = float(((y_q != 0) & inactive).sum().item())
    inactive_count = float(inactive.sum().item())

    prefix = f"part{part_idx}"
    stats[f"{prefix}_branch_active_frac_of_part"] = ratio(active_count, valid_count)
    stats[f"{prefix}_branch_active_frac_of_all"] = ratio(active_count, total_count)
    stats[f"{prefix}_res_energy_active_frac"] = ratio(res_energy_active, res_energy_total)
    stats[f"{prefix}_qerr_energy_active_frac"] = ratio(qerr_energy_active, qerr_energy_total)
    stats[f"{prefix}_nonzero_symbol_active_frac"] = ratio(nonzero_active, nonzero_total)
    stats[f"{prefix}_active_nonzero_rate"] = ratio(nonzero_active, active_count)
    stats[f"{prefix}_inactive_nonzero_rate"] = ratio(nonzero_inactive, inactive_count)

    stats["branch_active_count"] = stats.get("branch_active_count", 0.0) + active_count
    stats["branch_valid_count"] = stats.get("branch_valid_count", 0.0) + valid_count
    stats["branch_total_count"] = stats.get("branch_total_count", 0.0) + total_count
    stats["res_energy_total"] = stats.get("res_energy_total", 0.0) + res_energy_total
    stats["qerr_energy_total"] = stats.get("qerr_energy_total", 0.0) + qerr_energy_total
    stats["res_energy_active"] = stats.get("res_energy_active", 0.0) + res_energy_active
    stats["qerr_energy_active"] = stats.get("qerr_energy_active", 0.0) + qerr_energy_active
    stats["nonzero_symbol_total"] = stats.get("nonzero_symbol_total", 0.0) + nonzero_total
    stats["nonzero_symbol_active"] = stats.get("nonzero_symbol_active", 0.0) + nonzero_active
    stats["nonzero_symbol_inactive"] = stats.get("nonzero_symbol_inactive", 0.0) + nonzero_inactive
    stats["branch_inactive_count"] = stats.get("branch_inactive_count", 0.0) + inactive_count
    return y_res, y_q, y_hat, scales_hat


def tail_branch_identity_forward_four_part_prior(
    self: GLC_Image,
    y: torch.Tensor,
    common_params: torch.Tensor,
    y_spatial_prior_adaptor_1,
    y_spatial_prior_adaptor_2,
    y_spatial_prior_adaptor_3,
    y_spatial_prior,
    y_spatial_prior_reduction=None,
    write: bool = False,
):
    del write
    q_enc, q_dec, scales, means = self.separate_prior(common_params)
    if y_spatial_prior_reduction is not None:
        common_params = y_spatial_prior_reduction(common_params)
    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)
    masks = (mask_0, mask_1, mask_2, mask_3)

    y_scaled = y * q_enc
    active_groups = set(getattr(self, "_e169_active_groups", []))
    active_parts = set(getattr(self, "_e169_active_parts", []))
    group_size = int(getattr(self, "_e169_group_size", 16))
    stats: dict[str, float] = {
        "branch_group_size": float(group_size),
        "branch_num_active_groups": float(len(active_groups)),
        "branch_num_active_parts": float(len(active_parts)),
    }

    branch_mask = make_group_branch_mask(mask_0, group_size, active_groups, 0 in active_parts)
    y_res_0, y_q_0, y_hat_0, s_hat_0 = process_tail_branch_identity(self, y_scaled, scales, means, mask_0, branch_mask, stats, 0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    branch_mask = make_group_branch_mask(mask_1, group_size, active_groups, 1 in active_parts)
    y_res_1, y_q_1, y_hat_1, s_hat_1 = process_tail_branch_identity(self, y_scaled, scales, means, mask_1, branch_mask, stats, 1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    branch_mask = make_group_branch_mask(mask_2, group_size, active_groups, 2 in active_parts)
    y_res_2, y_q_2, y_hat_2, s_hat_2 = process_tail_branch_identity(self, y_scaled, scales, means, mask_2, branch_mask, stats, 2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    branch_mask = make_group_branch_mask(mask_3, group_size, active_groups, 3 in active_parts)
    y_res_3, y_q_3, y_hat_3, s_hat_3 = process_tail_branch_identity(self, y_scaled, scales, means, mask_3, branch_mask, stats, 3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    y_hat = y_hat * q_dec

    stats["branch_active_frac_of_valid"] = ratio(stats["branch_active_count"], stats["branch_valid_count"])
    stats["branch_active_frac_of_all"] = ratio(stats["branch_active_count"], stats["branch_total_count"])
    stats["res_energy_active_frac"] = ratio(stats["res_energy_active"], stats["res_energy_total"])
    stats["qerr_energy_active_frac"] = ratio(stats["qerr_energy_active"], stats["qerr_energy_total"])
    stats["nonzero_symbol_active_frac"] = ratio(stats["nonzero_symbol_active"], stats["nonzero_symbol_total"])
    stats["active_nonzero_rate"] = ratio(stats["nonzero_symbol_active"], stats["branch_active_count"])
    stats["inactive_nonzero_rate"] = ratio(stats["nonzero_symbol_inactive"], stats["branch_inactive_count"])
    self._e169_tail_branch_identity_stats = stats
    return y_res, y_q, y_hat, scales_hat


def install_tail_branch_identity(
    net: GLC_Image,
    group_size: int,
    active_groups: list[int],
    active_parts: list[int],
) -> None:
    net._e169_group_size = group_size
    net._e169_active_groups = active_groups
    net._e169_active_parts = active_parts
    net.forward_four_part_prior = types.MethodType(tail_branch_identity_forward_four_part_prior, net)


def mean_or_nan(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    official_forward_four_part_prior = net.forward_four_part_prior

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for path in images:
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            net.forward_four_part_prior = official_forward_four_part_prior
            original = net.test(x_pad, q)
            install_tail_branch_identity(net, args.group_size, args.active_groups, args.active_parts)
            branched = net.test(x_pad, q)
            net.forward_four_part_prior = official_forward_four_part_prior
            stats = getattr(net, "_e169_tail_branch_identity_stats", {})

            x_diff = original["x_hat"] - branched["x_hat"]
            ref_diff = original["ref_latent"] - branched["ref_latent"]
            bit_y_diff = float(original["bit_y"]) - float(branched["bit_y"])
            bit_z_diff = float(original["bit_z"]) - float(branched["bit_z"])
            bit_total_diff = float(original["bit"]) - float(branched["bit"])
            row: dict[str, Any] = {
                "q_index": q,
                "image": path.name,
                "height": h,
                "width": w,
                "max_abs_xhat_diff": finite_max_abs(x_diff),
                "max_abs_ref_latent_diff": finite_max_abs(ref_diff),
                "bit_y_diff": bit_y_diff,
                "bit_z_diff": bit_z_diff,
                "bit_total_diff": bit_total_diff,
                "nonfinite": int(
                    (not torch.isfinite(original["x_hat"]).all().item())
                    or (not torch.isfinite(branched["x_hat"]).all().item())
                    or any(not math.isfinite(v) for v in [bit_y_diff, bit_z_diff, bit_total_diff])
                    or any(isinstance(v, float) and not math.isfinite(v) for v in stats.values())
                ),
            }
            row.update(stats)
            rows.append(row)
            print(
                f"q={q} {path.name} max_x={row['max_abs_xhat_diff']:.1e} bit={bit_total_diff:.1e} "
                f"active={row['branch_active_frac_of_valid']:.4f} "
                f"resE={row['res_energy_active_frac']:.4f} qerrE={row['qerr_energy_active_frac']:.4f} "
                f"nonzero={row['nonzero_symbol_active_frac']:.4f} nonfinite={row['nonfinite']}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for q in args.q_indexes:
        subset = [row for row in rows if int(row["q_index"]) == q]
        if not subset:
            continue
        summary.append(
            {
                "q_index": q,
                "images": len(subset),
                "max_abs_xhat_diff": float(np.max([row["max_abs_xhat_diff"] for row in subset])),
                "max_abs_ref_latent_diff": float(np.max([row["max_abs_ref_latent_diff"] for row in subset])),
                "max_abs_bit_total_diff": float(np.max(np.abs([row["bit_total_diff"] for row in subset]))),
                "nonfinite_rows": int(sum(int(row["nonfinite"]) for row in subset)),
                "branch_active_frac_of_valid": mean_or_nan(subset, "branch_active_frac_of_valid"),
                "branch_active_frac_of_all": mean_or_nan(subset, "branch_active_frac_of_all"),
                "res_energy_active_frac": mean_or_nan(subset, "res_energy_active_frac"),
                "qerr_energy_active_frac": mean_or_nan(subset, "qerr_energy_active_frac"),
                "nonzero_symbol_active_frac": mean_or_nan(subset, "nonzero_symbol_active_frac"),
                "active_nonzero_rate": mean_or_nan(subset, "active_nonzero_rate"),
                "inactive_nonzero_rate": mean_or_nan(subset, "inactive_nonzero_rate"),
            }
        )

    payload = {
        "experiment": "E169 GLC y-path tail branch identity scaffold",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "padding_size": args.padding_size,
        "group_size": args.group_size,
        "active_groups": args.active_groups,
        "active_parts": args.active_parts,
        "rows": len(rows),
        "summary": summary,
        "note": "This is an identity branch scaffold. Active and fallback states both use original scalar rounding.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E169 GLC y-Path Tail Branch Identity Scaffold",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        f"Group size: `{args.group_size}`",
        f"Active parts: `{args.active_parts}`",
        f"Active groups: `{args.active_groups}`",
        "",
        "Both branch states use original scalar rounding. Exact equality is required; the table measures how much residual and rounding-error energy the selected active state would cover.",
        "",
        "| q | images | max xhat diff | max ref diff | max bit diff | nonfinite | active frac(valid) | active frac(all) | residual energy covered | qerr energy covered | nonzero symbols covered | active nonzero rate | inactive nonzero rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['q_index']} | {row['images']} | {row['max_abs_xhat_diff']:.3e} | "
            f"{row['max_abs_ref_latent_diff']:.3e} | {row['max_abs_bit_total_diff']:.3e} | "
            f"{row['nonfinite_rows']} | {row['branch_active_frac_of_valid']:.4f} | "
            f"{row['branch_active_frac_of_all']:.4f} | {row['res_energy_active_frac']:.4f} | "
            f"{row['qerr_energy_active_frac']:.4f} | {row['nonzero_symbol_active_frac']:.4f} | "
            f"{row['active_nonzero_rate']:.4f} | {row['inactive_nonzero_rate']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
