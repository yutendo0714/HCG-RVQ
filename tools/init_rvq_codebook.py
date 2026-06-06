from __future__ import annotations

import argparse
import math
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
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config, load_matching_state_dict


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x.clamp_min(1e-6)))


def to_grouped(x: torch.Tensor, group_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if c % group_size != 0:
        raise ValueError(f"channels={c} must be divisible by group_size={group_size}")
    ng = c // group_size
    return x.view(b, ng, group_size, h, w).permute(0, 1, 3, 4, 2).contiguous()


def collect_channel_stats(model, loader, device: torch.device, max_batches: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    channel_sum = None
    channel_sumsq = None
    count = 0
    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(loader, desc="latent stats", dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = model.g_a(x)
            dims = (0, 2, 3)
            if channel_sum is None:
                channel_sum = y.sum(dim=dims)
                channel_sumsq = y.pow(2).sum(dim=dims)
            else:
                channel_sum = channel_sum + y.sum(dim=dims)
                channel_sumsq = channel_sumsq + y.pow(2).sum(dim=dims)
            count += y.shape[0] * y.shape[2] * y.shape[3]
    if channel_sum is None or channel_sumsq is None or count == 0:
        raise RuntimeError("no latent statistics collected")
    mean = channel_sum / count
    var = channel_sumsq / count - mean.pow(2)
    std = var.clamp_min(1e-8).sqrt()
    return mean, std


def collect_vectors(
    model,
    loader,
    device: torch.device,
    num_vectors: int,
    global_mean: torch.Tensor | None,
    global_std: torch.Tensor | None,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    collected = 0
    group_size = model.group_size
    with torch.no_grad():
        for x in tqdm(loader, desc="sample vectors", dynamic_ncols=True):
            x = x.to(device, non_blocking=True)
            y = model.g_a(x)
            if global_mean is not None and global_std is not None:
                y = (y - global_mean.view(1, -1, 1, 1)) / global_std.view(1, -1, 1, 1)
            yg = to_grouped(y, group_size).reshape(-1, group_size)
            remaining = num_vectors - collected
            if remaining <= 0:
                break
            take = min(remaining, yg.shape[0])
            perm = torch.randperm(yg.shape[0], device=yg.device)[:take]
            vectors.append(yg[perm].detach().cpu())
            collected += take
            if collected >= num_vectors:
                break
    if collected < num_vectors:
        print(f"warning: requested {num_vectors} vectors, collected {collected}")
    if not vectors:
        raise RuntimeError("no vectors collected")
    return torch.cat(vectors, dim=0)


def assign_nearest(x: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    assignments = []
    center_norm = centers.pow(2).sum(dim=1)
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        dist = chunk.pow(2).sum(dim=1, keepdim=True) - 2.0 * chunk @ centers.t() + center_norm
        assignments.append(dist.argmin(dim=1))
    return torch.cat(assignments, dim=0)


def kmeans(x: torch.Tensor, k: int, iters: int, chunk_size: int, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] < k:
        raise ValueError(f"need at least k={k} vectors, got {x.shape[0]}")
    perm = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
    centers = x[perm].clone()
    for _ in tqdm(range(iters), desc=f"kmeans K={k}", leave=False, dynamic_ncols=True):
        idx = assign_nearest(x, centers, chunk_size)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device, dtype=x.dtype)
        new_centers.index_add_(0, idx, x)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=x.dtype))
        nonempty = counts > 0
        centers[nonempty] = new_centers[nonempty] / counts[nonempty].unsqueeze(1)
    return centers


def init_codebooks(model, vectors: torch.Tensor, kmeans_iters: int, chunk_size: int, seed: int, device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    residual = vectors.to(device)
    original = residual.clone()
    codebooks = []
    quantized_sum = torch.zeros_like(residual)
    for stage in range(model.num_stages):
        centers = kmeans(residual, model.codebook_size, kmeans_iters, chunk_size, generator)
        idx = assign_nearest(residual, centers, chunk_size)
        q = centers[idx]
        quantized_sum = quantized_sum + q
        residual = residual - q
        codebooks.append(centers.detach().cpu())
        mse = F.mse_loss(quantized_sum, original).item()
        print(f"stage {stage}: sampled latent quant MSE {mse:.6f}")
    stacked = torch.stack(codebooks, dim=0).to(model.rvq.codebooks.device)
    with torch.no_grad():
        model.rvq.codebooks.copy_(stacked)
    return {
        "sampled_latent_quant_mse": F.mse_loss(quantized_sum, original).item(),
        "sampled_residual_mse": residual.pow(2).mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize RVQ codebooks from a scalar/hyperprior checkpoint.")
    parser.add_argument("--config", required=True, help="target RVQ/HCS/HCG config")
    parser.add_argument("--source-checkpoint", required=True, help="scalar or compatible checkpoint")
    parser.add_argument("--output", required=True, help="checkpoint path to write")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-vectors", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-chunk-size", type=int, default=32768)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--stats-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--postload-householder-bias-init-scale",
        type=float,
        default=0.0,
        help="Reinitialize householder_head.bias after loading a scalar source checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(config).to(device)
    load_stats = load_matching_state_dict(model, args.source_checkpoint, map_location=device, skip_prefixes=("rvq.",))
    print(f"loaded source weights: {load_stats}")
    if args.postload_householder_bias_init_scale > 0.0:
        if not hasattr(model, "householder_head"):
            raise AttributeError("target model does not have householder_head")
        with torch.no_grad():
            model.householder_head.weight.zero_()
            model.householder_head.bias.normal_(mean=0.0, std=args.postload_householder_bias_init_scale)
        print(
            "postload householder bias init: "
            f"std={args.postload_householder_bias_init_scale:.6f}, "
            f"abs_mean={model.householder_head.bias.detach().abs().mean().item():.6f}"
        )
    model.eval()

    data_cfg = config.get("data", {})
    max_images = args.max_images if args.max_images is not None else data_cfg.get("max_train_images")
    dataset = ImageFolderDataset(
        roots=data_cfg.get("train_roots", ["/dpl/openimages/open-images-v6/train/data"]),
        patch_size=data_cfg.get("patch_size", 256),
        training=True,
        max_images=max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    global_mean = None
    global_std = None
    if model.variant == "global_rvq" and model.use_global_norm:
        global_mean, global_std = collect_channel_stats(model, loader, device, args.stats_batches)
        global_std = global_std.clamp(config.get("hyper_conditioning", {}).get("scale_min", 0.05), config.get("hyper_conditioning", {}).get("scale_max", 10.0))
        with torch.no_grad():
            model.global_mu.copy_(global_mean.view(1, -1, 1, 1).to(model.global_mu.device))
            model.global_log_s.copy_(inverse_softplus(global_std).view(1, -1, 1, 1).to(model.global_log_s.device))
        print(f"global norm mean={global_mean.mean().item():.6f}, std={global_std.mean().item():.6f}")

    vectors = collect_vectors(model, loader, device, args.num_vectors, global_mean, global_std)
    init_stats = init_codebooks(model, vectors, args.kmeans_iters, args.kmeans_chunk_size, args.seed, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "config": args.config,
                "source_checkpoint": args.source_checkpoint,
                "num_vectors": args.num_vectors,
                "kmeans_iters": args.kmeans_iters,
                "postload_householder_bias_init_scale": args.postload_householder_bias_init_scale,
                "init_stats": init_stats,
            },
        },
        output,
    )
    print(f"saved initialized checkpoint to {output}")
    print(init_stats)


if __name__ == "__main__":
    main()
