#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, get_state_dict  # noqa: E402
from tools.run_e170_glc_tail_vq_rate_distortion_probe import ResidualSet, active_key  # noqa: E402
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    PreparedImage,
    TrainableRVQCodebooks,
    build_initial_codebooks,
    crop_to_image,
    evaluate_rows,
    install_trainable_branch,
    run_instrumented,
    summarize,
    training_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=Path("/dpl/openimages/open-images-v6/train/data"))
    p.add_argument("--eval-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e177_glc_decoder_aware_tail_vq_split_train_smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--train-crop-size", type=int, default=256)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--train-start-index", type=int, default=8192)
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=2)
    p.add_argument("--eval-limit", type=int, default=2)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--kmeans-iters", type=int, default=6)
    p.add_argument("--max-train-vectors", type=int, default=12000)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--mse-weight", type=float, default=0.2)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    return p.parse_args()


def list_images(root: Path, start: int, limit: int | None) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in exts)
    if start:
        paths = paths[start:]
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_image_tensor(path: Path, device: torch.device, crop_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if crop_size > 0:
        w, h = img.size
        if w < crop_size or h < crop_size:
            scale = crop_size / float(min(w, h))
            new_size = (max(crop_size, int(round(w * scale))), max(crop_size, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.BICUBIC)
            w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def prepare_images(paths: list[Path], device: torch.device, padding_size: int, crop_size: int) -> list[PreparedImage]:
    prepared = []
    for path in paths:
        img01 = load_image_tensor(path, device, crop_size)
        x = from_0_1_to_minus1_1(img01)
        _, _, h, w = x.shape
        padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, padding_size)
        x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
        prepared.append(PreparedImage(path, img01, x, x_pad, (padding_l, padding_r, padding_t, padding_b), h, w))
    return prepared


def extract_group_vectors(x: torch.Tensor, mask: torch.Tensor, group: int, group_size: int) -> torch.Tensor:
    start = group * group_size
    end = start + group_size
    spatial = mask[0, start].bool()
    return x[0, start:end].permute(1, 2, 0)[spatial].detach().float().cpu()


@torch.no_grad()
def collect_residual_set_from_prepared(
    net: GLC_Image,
    item: PreparedImage,
    q_index: int,
    group_size: int,
    active_parts: set[int],
    active_groups: set[int],
) -> ResidualSet:
    curr_q_enc = net.q_enc[q_index : q_index + 1]
    y_ori = net.vqgan.encoder(item.x_pad)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)
    q_enc, _, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    B, C, H, W = y.size()
    masks = net.get_mask_four_parts(B, C, H, W, y.dtype, y.device)
    y_scaled = y * q_enc
    y_hat_so_far = None

    vecs: list[torch.Tensor] = []
    qvecs: list[torch.Tensor] = []
    bitvecs: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
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
        bits = net.get_y_gaussian_bits(y_q, scales_hat) * mask
        y_hat_part = y_q + means_hat
        if part_idx in active_parts:
            for group in sorted(active_groups):
                if group * group_size < C:
                    res_vec = extract_group_vectors(y_res, mask, group, group_size)
                    scalar_vec = extract_group_vectors(y_q, mask, group, group_size)
                    bit_vec = extract_group_vectors(bits, mask, group, group_size)
                    vecs.append(res_vec)
                    qvecs.append(scalar_vec)
                    bitvecs.append(bit_vec)
                    keys.append(torch.full((res_vec.shape[0],), active_key(part_idx, group), dtype=torch.long))
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
    return ResidualSet(
        item.path.name,
        q_index,
        item.height * item.width,
        torch.cat(vecs, dim=0),
        torch.cat(qvecs, dim=0),
        torch.cat(bitvecs, dim=0),
        torch.cat(keys, dim=0),
    )


def write_outputs(
    args: argparse.Namespace,
    train_paths: list[Path],
    eval_paths: list[Path],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    train_trace: list[dict[str, Any]],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "E177 GLC decoder-aware tail VQ split-train diagnostic",
        "note": "Diagnostic only. Active codebooks are trained on a small train split and evaluated on a separate eval split; this is not full matched GLC fine-tuning.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_images": [str(p) for p in train_paths],
        "eval_images": [str(p) for p in eval_paths],
        "summary": summary,
        "train_trace": train_trace,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E177 GLC Decoder-Aware Tail VQ Split-Train Diagnostic",
        "",
        "Diagnostic only: active codebooks are trained on a small train split and evaluated on a separate eval split. This tests transfer beyond E175 Kodak-overfit, not final matched GLC fine-tuning.",
        "",
        f"Train dir/start/limit/crop: `{args.train_dir}` / `{args.train_start_index}` / `{args.train_limit}` / `{args.train_crop_size}`",
        f"Eval dir/start/limit/crop: `{args.eval_dir}` / `{args.eval_start_index}` / `{args.eval_limit}` / `{args.eval_crop_size}`",
        f"Active parts/groups: `{args.active_parts}` / `{args.active_groups}`",
        f"Scope/K/stages: `{args.scope}` / `{args.k}` / `{args.stages}`",
        f"Steps/lr/loss weights: `{args.steps}` / `{args.lr}` / mse `{args.mse_weight}`, lpips `{args.lpips_weight}`, dists `{args.dists_weight}`",
        "",
        "Train images:",
        *[f"- `{p}`" for p in train_paths],
        "",
        "Eval images:",
        *[f"- `{p}`" for p in eval_paths],
        "",
        "| label | q | images | base bpp | emp bpp | emp dbpp | PSNR base | PSNR branch | MS base | MS branch | LPIPS base | LPIPS branch | DISTS base | DISTS branch | active MSE ratio | H | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['label']} | {s['q_index']} | {s['images']} | {s['base_bpp']:.6f} | "
            f"{s['branch_hybrid_empirical_bpp']:.6f} | {s['empirical_bpp_delta']:+.6f} | "
            f"{s['base_psnr']:.4f} | {s['branch_psnr']:.4f} | {s['base_ms_ssim']:.5f} | {s['branch_ms_ssim']:.5f} | "
            f"{s['base_lpips']:.5f} | {s['branch_lpips']:.5f} | {s['base_dists']:.5f} | {s['branch_dists']:.5f} | "
            f"{s['active_mse_ratio']:.4f} | {s['index_entropy_mean']:.4f} | {s['nonfinite_rows']} |"
        )
    if train_trace:
        lines.extend(["", "## Train Trace", "", "| step | loss | mse | lpips | dists |", "|---:|---:|---:|---:|---:|"])
        for t in train_trace:
            lines.append(f"| {t['step']} | {t['loss']:.6f} | {t['mse']:.6f} | {t['lpips']:.6f} | {t['dists']:.6f} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_paths = list_images(args.train_dir, args.train_start_index, args.train_limit)
    eval_paths = list_images(args.eval_dir, args.eval_start_index, args.eval_limit)
    if not train_paths:
        raise SystemExit(f"no train images in {args.train_dir}")
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.eval_dir}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    dists_fn = dists.DISTS().to(device).eval()
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    net = GLC_Image(inplace=False).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    active_parts = set(args.active_parts)
    active_groups = set(args.active_groups)

    train_prepared = prepare_images(train_paths, device, args.padding_size, args.train_crop_size)
    eval_prepared = prepare_images(eval_paths, device, args.padding_size, args.eval_crop_size)
    codebooks_by_q: dict[int, TrainableRVQCodebooks] = {}
    with torch.no_grad():
        for q in args.q_indexes:
            train_items = [
                collect_residual_set_from_prepared(net, item, q, args.group_size, active_parts, active_groups)
                for item in train_prepared
            ]
            initial = build_initial_codebooks(
                train_items,
                args.scope,
                args.k,
                args.stages,
                args.kmeans_iters,
                args.max_train_vectors,
                args.seed + q * 10000,
                device,
            )
            codebooks_by_q[q] = TrainableRVQCodebooks(initial, device).to(device)
            print(f"initialized q={q} keys={len(initial)} train_vectors={sum(int(x.vectors.shape[0]) for x in train_items)}")
    net.masks = {}

    rows = evaluate_rows(net, official_forward, codebooks_by_q, eval_prepared, args.q_indexes, args, lpips_fn, dists_fn, "init_eval")
    params = [p for module in codebooks_by_q.values() for p in module.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    train_trace: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    num_train_losses = max(1, len(args.q_indexes) * len(train_prepared))
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_accum = {"mse": 0.0, "lpips": 0.0, "dists": 0.0, "loss": 0.0}
        count = 0
        for q in args.q_indexes:
            install_trainable_branch(net, codebooks_by_q[q], args)
            for item in train_prepared:
                branch_pad, _ = run_instrumented(net, item.x_pad, q)
                branch = crop_to_image(branch_pad, item)
                branch01 = ((branch + 1.0) * 0.5).clamp(0, 1)
                loss, parts = training_loss(branch, branch01, item, lpips_fn, dists_fn, args)
                (loss / num_train_losses).backward()
                for key in loss_accum:
                    loss_accum[key] += parts[key]
                count += 1
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        trace_row = {"step": step}
        for key, val in loss_accum.items():
            trace_row[key] = val / max(1, count)
        train_trace.append(trace_row)
        print(
            f"step={step}/{args.steps} loss={trace_row['loss']:.6f} mse={trace_row['mse']:.6f} "
            f"lpips={trace_row['lpips']:.6f} dists={trace_row['dists']:.6f}"
        )
    net.forward_four_part_prior = official_forward
    print(f"training_ms={(time.perf_counter() - t0) * 1000.0:.1f}")

    rows.extend(evaluate_rows(net, official_forward, codebooks_by_q, eval_prepared, args.q_indexes, args, lpips_fn, dists_fn, "trained_eval"))
    summary = summarize(rows, args.q_indexes)
    write_outputs(args, train_paths, eval_paths, rows, summary, train_trace)


if __name__ == "__main__":
    main()
