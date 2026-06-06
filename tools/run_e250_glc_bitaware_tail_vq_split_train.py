#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
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
from src.utils.test_utils import get_state_dict  # noqa: E402
from tools.run_e170_glc_tail_vq_rate_distortion_probe import (  # noqa: E402
    sample_rows,
    vectors_for_key,
)
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    TrainableRVQCodebooks,
    build_initial_codebooks,
    crop_to_image,
    dists_call,
    evaluate_rows,
    install_trainable_branch,
    run_instrumented,
    summarize,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    collect_residual_set_from_prepared,
    list_images,
    prepare_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=Path("/dpl/openimages/open-images-v6/train/data"))
    p.add_argument("--eval-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e250_glc_bitaware_tail_vq_split_train_smoke")
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
    p.add_argument("--max-rate-vectors", type=int, default=2048)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--mse-weight", type=float, default=0.2)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--soft-index-weight", type=float, default=0.01)
    p.add_argument("--soft-index-target", type=float, default=2.0)
    p.add_argument("--soft-index-temp", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    return p.parse_args()


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def soft_usage_entropy(
    codebooks: TrainableRVQCodebooks,
    train_items,
    args: argparse.Namespace,
    q_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    entropies: list[torch.Tensor] = []
    excesses: list[torch.Tensor] = []
    for key in codebooks.stage_counts:
        vectors = vectors_for_key(train_items, key, args.scope)
        vectors = sample_rows(vectors, args.max_rate_vectors, args.seed + q_index * 10000 + key)
        if vectors.numel() == 0:
            continue
        residual = vectors.float().to(device)
        for book in codebooks.for_key(key):
            cb = book.float()
            dist = torch.cdist(residual, cb, p=2).pow(2)
            probs = torch.softmax(-dist / max(args.soft_index_temp, 1e-6), dim=1)
            usage = probs.mean(dim=0).clamp_min(1e-8)
            entropy = -(usage * usage.log2()).sum()
            entropies.append(entropy)
            excesses.append(F.relu(entropy - args.soft_index_target))
            residual = residual - probs.matmul(cb)
    if not entropies:
        zero = torch.zeros((), device=device)
        return zero, {"soft_index_entropy": 0.0, "soft_index_excess": 0.0}
    entropy_mean = torch.stack(entropies).mean()
    excess_mean = torch.stack(excesses).mean()
    return excess_mean, {
        "soft_index_entropy": float(entropy_mean.detach().item()),
        "soft_index_excess": float(excess_mean.detach().item()),
    }


def image_loss(
    branch: torch.Tensor,
    branch01: torch.Tensor,
    target,
    lpips_fn,
    dists_fn,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(branch01, target.img01)
    lpips_val = lpips_fn(branch, target.x).mean()
    dists_val = dists_call(dists_fn, branch01, target.img01, require_grad=True)
    loss = args.mse_weight * mse + args.lpips_weight * lpips_val + args.dists_weight * dists_val
    return loss, {
        "mse": float(mse.detach().item()),
        "lpips": float(lpips_val.detach().item()),
        "dists": float(dists_val.detach().item()),
        "image_loss": float(loss.detach().item()),
    }


def train_summary_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    subset = [r for r in rows if r["label"] == label]
    return {
        "label": label,
        "images": len(subset),
        "empirical_bpp_delta": finite_mean([float(r["empirical_bpp_delta"]) for r in subset]),
        "delta_psnr": finite_mean([float(r["branch_psnr"]) - float(r["base_psnr"]) for r in subset]),
        "delta_ms_ssim": finite_mean([float(r["branch_ms_ssim"]) - float(r["base_ms_ssim"]) for r in subset]),
        "delta_lpips": finite_mean([float(r["branch_lpips"]) - float(r["base_lpips"]) for r in subset]),
        "delta_dists": finite_mean([float(r["branch_dists"]) - float(r["base_dists"]) for r in subset]),
        "active_mse_ratio": finite_mean([float(r["active_mse_ratio"]) for r in subset]),
        "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
    }


def write_outputs(
    args: argparse.Namespace,
    train_paths: list[Path],
    eval_paths: list[Path],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "E250 GLC bit-aware tail VQ split-train smoke",
        "note": "Diagnostic only. Adds a differentiable soft index entropy excess proxy to the E177 image loss.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_images": [str(p) for p in train_paths],
        "eval_images": [str(p) for p in eval_paths],
        "summary": summary,
        "delta_summary": [train_summary_row(rows, label) for label in sorted({r["label"] for r in rows})],
        "train_trace": trace,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E250 GLC Bit-Aware Tail VQ Split-Train Smoke",
        "",
        "Diagnostic only: this extends E177 by adding a differentiable soft index",
        "entropy excess proxy.  The goal is to see whether rate-aware pressure can",
        "reduce the E181 bpp fragility without erasing quality gains.",
        "",
        f"Train dir/start/limit/crop: `{args.train_dir}` / `{args.train_start_index}` / `{args.train_limit}` / `{args.train_crop_size}`",
        f"Eval dir/start/limit/crop: `{args.eval_dir}` / `{args.eval_start_index}` / `{args.eval_limit}` / `{args.eval_crop_size}`",
        f"Loss weights: mse `{args.mse_weight}`, lpips `{args.lpips_weight}`, dists `{args.dists_weight}`, soft-index `{args.soft_index_weight}` target `{args.soft_index_target}` temp `{args.soft_index_temp}`",
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
    lines.extend(["", "## Train Trace", "", "| step | loss | image | soft entropy | soft excess | mse | lpips | dists |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for t in trace:
        lines.append(
            f"| {t['step']} | {t['loss']:.6f} | {t['image_loss']:.6f} | "
            f"{t['soft_index_entropy']:.6f} | {t['soft_index_excess']:.6f} | "
            f"{t['mse']:.6f} | {t['lpips']:.6f} | {t['dists']:.6f} |"
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


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
    train_items_by_q = {}
    codebooks_by_q: dict[int, TrainableRVQCodebooks] = {}
    with torch.no_grad():
        for q in args.q_indexes:
            train_items = [
                collect_residual_set_from_prepared(net, item, q, args.group_size, active_parts, active_groups)
                for item in train_prepared
            ]
            train_items_by_q[q] = train_items
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
    del eval_prepared
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    params = [p for module in codebooks_by_q.values() for p in module.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    trace: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    num_train_losses = max(1, len(args.q_indexes) * len(train_prepared))
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        accum = {"mse": 0.0, "lpips": 0.0, "dists": 0.0, "image_loss": 0.0}
        proxy_accum = {"soft_index_entropy": 0.0, "soft_index_excess": 0.0}
        count = 0
        loss_total_value = 0.0
        for q in args.q_indexes:
            install_trainable_branch(net, codebooks_by_q[q], args)
            for item in train_prepared:
                branch_pad, _ = run_instrumented(net, item.x_pad, q)
                branch = crop_to_image(branch_pad, item)
                branch01 = ((branch + 1.0) * 0.5).clamp(0, 1)
                loss, parts = image_loss(branch, branch01, item, lpips_fn, dists_fn, args)
                scaled_loss = loss / num_train_losses
                scaled_loss.backward()
                loss_total_value += float(scaled_loss.detach().item())
                for key in accum:
                    accum[key] += parts[key]
                count += 1
                del branch_pad, branch, branch01, loss, scaled_loss
            proxy, proxy_parts = soft_usage_entropy(codebooks_by_q[q], train_items_by_q[q], args, q, device)
            scaled_proxy = args.soft_index_weight * proxy / max(1, len(args.q_indexes))
            scaled_proxy.backward()
            loss_total_value += float(scaled_proxy.detach().item())
            for key in proxy_accum:
                proxy_accum[key] += proxy_parts[key]
            del proxy, scaled_proxy
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        trace_row = {"step": step, "loss": loss_total_value}
        for key, val in accum.items():
            trace_row[key] = val / max(1, count)
        for key, val in proxy_accum.items():
            trace_row[key] = val / max(1, len(args.q_indexes))
        trace.append(trace_row)
        print(
            f"step={step}/{args.steps} loss={trace_row['loss']:.6f} image={trace_row['image_loss']:.6f} "
            f"soft_H={trace_row['soft_index_entropy']:.4f} excess={trace_row['soft_index_excess']:.4f} "
            f"dists={trace_row['dists']:.6f}"
        )
    net.forward_four_part_prior = official_forward
    print(f"training_ms={(time.perf_counter() - t0) * 1000.0:.1f}")

    del train_prepared
    del train_items_by_q
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    eval_prepared = prepare_images(eval_paths, device, args.padding_size, args.eval_crop_size)
    rows.extend(evaluate_rows(net, official_forward, codebooks_by_q, eval_prepared, args.q_indexes, args, lpips_fn, dists_fn, "trained_eval"))
    summary = summarize(rows, args.q_indexes)
    write_outputs(args, train_paths, eval_paths, rows, summary, trace)


if __name__ == "__main__":
    main()
