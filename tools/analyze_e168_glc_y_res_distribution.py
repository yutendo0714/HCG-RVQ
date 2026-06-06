#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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


METRIC_FIELDS = ("bpp", "bpp_y", "bpp_z", "psnr", "ms_ssim", "lpips", "dists")
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
RES_TAILS = (0.125, 0.25, 0.5, 1.0)
QERR_TAILS = (0.25, 0.49)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--metrics-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e162_glc_pretrained_kodak24.csv")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e168_glc_y_res_distribution_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_metric_rows(path: Path | None) -> dict[tuple[int, str], dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["q_index"]), row["image"])
            out[key] = {k: float(row[k]) for k in METRIC_FIELDS if k in row and row[k] != ""}
    return out


def sanitize_name(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def finite_values(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    finite = torch.isfinite(x)
    if not finite.any():
        return x.new_empty((0,))
    return x[finite]


def safe_stat_values(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = finite_values(values)
    out: dict[str, float] = {f"{prefix}_count": int(values.numel())}
    if values.numel() == 0:
        for name in ("mean", "std", "abs_mean", "rms", "min", "max"):
            out[f"{prefix}_{name}"] = float("nan")
        for q in QUANTILES:
            out[f"{prefix}_p{int(q * 100):02d}"] = float("nan")
        return out

    abs_values = values.abs()
    out.update(
        {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_abs_mean": float(abs_values.mean().item()),
            f"{prefix}_rms": float(torch.sqrt((values * values).mean()).item()),
            f"{prefix}_min": float(values.min().item()),
            f"{prefix}_max": float(values.max().item()),
        }
    )
    qs = torch.tensor(QUANTILES, dtype=values.dtype, device=values.device)
    qvals = torch.quantile(abs_values, qs)
    for q, qv in zip(QUANTILES, qvals):
        out[f"{prefix}_abs_p{int(q * 100):02d}"] = float(qv.item())
    return out


def quantized_symbol_stats(q_values: torch.Tensor, prefix: str) -> dict[str, float]:
    q_values = finite_values(q_values)
    if q_values.numel() == 0:
        return {
            f"{prefix}_zero_frac": float("nan"),
            f"{prefix}_symbol_entropy": float("nan"),
            f"{prefix}_symbol_perplexity": float("nan"),
            f"{prefix}_symbol_used": float("nan"),
            f"{prefix}_symbol_max_prob": float("nan"),
        }

    q_int = q_values.round().to(torch.int64).cpu()
    unique, counts = torch.unique(q_int, sorted=True, return_counts=True)
    probs = counts.float() / counts.sum().clamp_min(1)
    entropy = float((-(probs * torch.log2(probs.clamp_min(1e-12))).sum()).item())
    return {
        f"{prefix}_zero_frac": float((q_int == 0).float().mean().item()),
        f"{prefix}_symbol_entropy": entropy,
        f"{prefix}_symbol_perplexity": float(2.0**entropy),
        f"{prefix}_symbol_used": int(unique.numel()),
        f"{prefix}_symbol_max_prob": float(probs.max().item()),
    }


def active_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask.to(torch.bool).expand_as(x)]


def summarize_active(
    y_res: torch.Tensor,
    y_q: torch.Tensor,
    qerr: torch.Tensor,
    scales_hat: torch.Tensor,
    means_hat: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    res_values = active_tensor(y_res, mask)
    q_values = active_tensor(y_q, mask)
    qerr_values = active_tensor(qerr, mask)
    scale_values = active_tensor(scales_hat, mask)
    mean_values = active_tensor(means_hat, mask)

    row: dict[str, float] = {}
    row.update(safe_stat_values(res_values, "res"))
    row.update(safe_stat_values(q_values, "q"))
    row.update(safe_stat_values(qerr_values, "qerr"))
    row.update(safe_stat_values(scale_values, "scale"))
    row.update(safe_stat_values(mean_values, "mean"))
    row.update(quantized_symbol_stats(q_values, "q"))

    res_abs = finite_values(res_values).abs()
    qerr_abs = finite_values(qerr_values).abs()
    for tail in RES_TAILS:
        key = f"res_abs_gt_{sanitize_name(tail)}"
        row[key] = float((res_abs > tail).float().mean().item()) if res_abs.numel() else float("nan")
    for tail in QERR_TAILS:
        key = f"qerr_abs_gt_{sanitize_name(tail)}"
        row[key] = float((qerr_abs > tail).float().mean().item()) if qerr_abs.numel() else float("nan")
    return row


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ya = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def mean_or_nan(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)

    metric_keys = (
        "res_std",
        "res_abs_mean",
        "res_abs_p90",
        "res_abs_p95",
        "res_abs_p99",
        "res_rms",
        "q_std",
        "q_zero_frac",
        "q_symbol_entropy",
        "q_symbol_perplexity",
        "q_symbol_used",
        "qerr_rms",
        "qerr_abs_mean",
        "qerr_abs_p90",
        "qerr_abs_p95",
        "qerr_abs_p99",
        "qerr_abs_gt_0p25",
        "qerr_abs_gt_0p49",
        "scale_mean",
        "scale_abs_p95",
        "mean_abs_mean",
        "res_abs_gt_0p125",
        "res_abs_gt_0p25",
        "res_abs_gt_0p5",
        "res_abs_gt_1p0",
    )

    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        row: dict[str, Any] = {k: v for k, v in zip(keys, group_key)}
        row["rows"] = len(group_rows)
        for key in metric_keys:
            row[key] = mean_or_nan(group_rows, key)
        row["corr_res_abs_p95_dists"] = pearson(
            [float(r["res_abs_p95"]) for r in group_rows],
            [float(r.get("metric_dists", float("nan"))) for r in group_rows],
        )
        row["corr_qerr_rms_dists"] = pearson(
            [float(r["qerr_rms"]) for r in group_rows],
            [float(r.get("metric_dists", float("nan"))) for r in group_rows],
        )
        row["corr_res_abs_p95_psnr"] = pearson(
            [float(r["res_abs_p95"]) for r in group_rows],
            [float(r.get("metric_psnr", float("nan"))) for r in group_rows],
        )
        out.append(row)
    return out


def assign_difficulty_quartiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[int(row["q_index"])].append(row)

    out: list[dict[str, Any]] = []
    for q, q_rows in by_q.items():
        image_scores: dict[str, float] = {}
        for row in q_rows:
            score = float(row.get("metric_dists", float("nan")))
            if math.isfinite(score):
                image_scores[str(row["image"])] = score
        ordered = sorted(image_scores.items(), key=lambda item: item[1])
        quartile: dict[str, str] = {}
        for rank, (image, _) in enumerate(ordered):
            label = f"Q{min(4, rank * 4 // max(1, len(ordered)) + 1)}"
            quartile[image] = label
        for row in q_rows:
            copied = dict(row)
            copied["difficulty_quartile"] = quartile.get(str(row["image"]), "NA")
            out.append(copied)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def collect_y_prior_rows(
    net: GLC_Image,
    x_pad: torch.Tensor,
    q_index: int,
    image_name: str,
    image_h: int,
    image_w: int,
    metric_row: dict[str, float] | None,
    group_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curr_q_enc = net.q_enc[q_index : q_index + 1]

    y_ori = net.vqgan.encoder(x_pad)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))

    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)
    q_enc, _, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    if C % group_size != 0:
        raise ValueError(f"channel count {C} is not divisible by group size {group_size}")
    masks = net.get_mask_four_parts(B, C, H, W, dtype, device)
    y_scaled = y * q_enc
    y_hat_so_far = None

    part_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            part_scales, part_means = scales, means
        else:
            assert y_hat_so_far is not None
            part_params = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1
                if part_idx == 1
                else net.y_spatial_prior_adaptor_2
                if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            part_scales, part_means = net.y_spatial_prior(adaptor(part_params)).chunk(2, 1)

        scales_hat = part_scales * mask
        means_hat = part_means * mask
        y_res = (y_scaled - means_hat) * mask
        y_q = net.quant(y_res)
        qerr = (y_q - y_res) * mask
        y_hat_part = y_q + means_hat

        base: dict[str, Any] = {
            "q_index": q_index,
            "image": image_name,
            "height": image_h,
            "width": image_w,
            "latent_height": H,
            "latent_width": W,
            "channels": C,
            "part": part_idx,
        }
        if metric_row:
            for field, value in metric_row.items():
                base[f"metric_{field}"] = value

        row = dict(base)
        row.update(summarize_active(y_res, y_q, qerr, scales_hat, means_hat, mask))
        part_rows.append(row)

        for group_idx in range(C // group_size):
            channel_slice = slice(group_idx * group_size, (group_idx + 1) * group_size)
            group_mask = mask[:, channel_slice]
            group_row = dict(base)
            group_row["group"] = group_idx
            group_row["group_size"] = group_size
            group_row["channel_start"] = group_idx * group_size
            group_row["channel_end_exclusive"] = (group_idx + 1) * group_size
            group_row.update(
                summarize_active(
                    y_res[:, channel_slice],
                    y_q[:, channel_slice],
                    qerr[:, channel_slice],
                    scales_hat[:, channel_slice],
                    means_hat[:, channel_slice],
                    group_mask,
                )
            )
            group_rows.append(group_row)

        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return part_rows, group_rows


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    part_summary: list[dict[str, Any]],
    group_summary: list[dict[str, Any]],
    quartile_summary: list[dict[str, Any]],
    rows: int,
    group_rows: int,
) -> None:
    lines = [
        "# E168 GLC y-Residual Distribution Audit",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Metrics CSV: `{args.metrics_csv}`",
        f"Device: `{args.device}`",
        f"Group size: `{args.group_size}`",
        "",
        "This is a read-only distribution audit at the E166 identity-preserving `forward_four_part_prior()` boundary.",
        "It does not change GLC outputs; it measures the residuals that a future HCG-RVQ `y`-path branch must replace or explain.",
        "",
        f"Rows: `{rows}` per-image/part rows and `{group_rows}` per-image/part/group rows.",
        "",
        "## Part Summary",
        "",
        "| q | part | rows | res std | res_abs_p95 | res_abs_p99 | qerr rms | qerr_abs_p95 | qerr_abs_p99 | q=0 frac | q entropy | scale mean | res_abs>0.25 | res_abs>0.5 | corr p95,DISTS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in part_summary:
        lines.append(
            f"| {int(row['q_index'])} | {int(row['part'])} | {int(row['rows'])} | "
            f"{row['res_std']:.5f} | {row['res_abs_p95']:.5f} | {row['res_abs_p99']:.5f} | "
            f"{row['qerr_rms']:.5f} | {row['qerr_abs_p95']:.5f} | {row['qerr_abs_p99']:.5f} | "
            f"{row['q_zero_frac']:.5f} | {row['q_symbol_entropy']:.4f} | {row['scale_mean']:.5f} | "
            f"{row['res_abs_gt_0p25']:.5f} | {row['res_abs_gt_0p5']:.5f} | {row['corr_res_abs_p95_dists']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Difficulty Quartiles",
            "",
            "Images are sorted by the reproduced GLC DISTS score within each q-index; Q1 is easier and Q4 is harder.",
            "",
            "| q | quartile | rows | res_abs_p95 | qerr rms | q=0 frac | scale mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quartile_summary:
        lines.append(
            f"| {int(row['q_index'])} | {row['difficulty_quartile']} | {int(row['rows'])} | "
            f"{row['res_abs_p95']:.5f} | {row['qerr_rms']:.5f} | {row['q_zero_frac']:.5f} | {row['scale_mean']:.5f} |"
        )

    worst_groups = sorted(group_summary, key=lambda r: float(r["res_abs_p95"]), reverse=True)[:12]
    lines.extend(
        [
            "",
            "## Largest Group Tails",
            "",
            "| q | part | group | rows | res_abs_p95 | res_abs_p99 | qerr rms | q=0 frac | scale mean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_groups:
        lines.append(
            f"| {int(row['q_index'])} | {int(row['part'])} | {int(row['group'])} | {int(row['rows'])} | "
            f"{row['res_abs_p95']:.5f} | {row['res_abs_p99']:.5f} | {row['qerr_rms']:.5f} | "
            f"{row['q_zero_frac']:.5f} | {row['scale_mean']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Transfer Implication",
            "",
            "- Part-wise and group-wise residual scales should decide the first GLC HCG-RVQ code dimension/stage schedule; a single global setting is risky.",
            "- The scalar path has a large zero-symbol mass, so an HCG-RVQ replacement must include index-prior or side-bit accounting from the start.",
            "- Any active HCG branch should be paired with a state-preserving fallback, following the E151/E167 reliability lesson.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    metric_rows = load_metric_rows(args.metrics_csv)

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)

    part_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for path in images:
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            metric_row = metric_rows.get((q, path.name))
            new_part_rows, new_group_rows = collect_y_prior_rows(
                net=net,
                x_pad=x_pad,
                q_index=q,
                image_name=path.name,
                image_h=h,
                image_w=w,
                metric_row=metric_row,
                group_size=args.group_size,
            )
            part_rows.extend(new_part_rows)
            group_rows.extend(new_group_rows)
            nonfinite = any(
                (not math.isfinite(float(r["res_std"])))
                or (not math.isfinite(float(r["qerr_rms"])))
                or (not math.isfinite(float(r["scale_mean"])))
                for r in new_part_rows
            )
            print(
                f"q={q} {path.name} parts={len(new_part_rows)} "
                f"res_p95={[round(float(r['res_abs_p95']), 4) for r in new_part_rows]} "
                f"qerr_rms={[round(float(r['qerr_rms']), 4) for r in new_part_rows]} "
                f"nonfinite={int(nonfinite)}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    part_csv = args.output_prefix.with_name(args.output_prefix.name + "_per_image_part.csv")
    group_csv = args.output_prefix.with_name(args.output_prefix.name + "_per_image_part_group.csv")
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    group_summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_group_summary.csv")
    quartile_csv = args.output_prefix.with_name(args.output_prefix.name + "_difficulty_quartiles.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    part_summary = aggregate(part_rows, ("q_index", "part"))
    group_summary = aggregate(group_rows, ("q_index", "part", "group"))
    quartile_rows = assign_difficulty_quartiles(part_rows)
    quartile_summary = aggregate(quartile_rows, ("q_index", "difficulty_quartile"))

    write_csv(part_csv, part_rows)
    write_csv(group_csv, group_rows)
    write_csv(summary_csv, part_summary)
    write_csv(group_summary_csv, group_summary)
    write_csv(quartile_csv, quartile_summary)

    payload = {
        "experiment": "E168 GLC y residual distribution audit",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "metrics_csv": str(args.metrics_csv),
        "device": str(device),
        "padding_size": args.padding_size,
        "group_size": args.group_size,
        "q_indexes": args.q_indexes,
        "per_image_part_rows": len(part_rows),
        "per_image_part_group_rows": len(group_rows),
        "part_summary": part_summary,
        "note": "Read-only audit at the E166 identity-preserving y-prior boundary.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, args, part_summary, group_summary, quartile_summary, len(part_rows), len(group_rows))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
