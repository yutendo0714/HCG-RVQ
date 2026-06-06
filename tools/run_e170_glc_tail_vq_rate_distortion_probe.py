#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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


@dataclass
class ResidualSet:
    image: str
    q_index: int
    pixels: int
    vectors: torch.Tensor
    scalar_q: torch.Tensor
    scalar_bits: torch.Tensor
    keys: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e170_glc_tail_vq_probe_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--k-list", type=int, nargs="*", default=[2, 4, 8, 16])
    p.add_argument("--stages-list", type=int, nargs="*", default=[1])
    p.add_argument("--scopes", nargs="*", default=["shared", "part_group"], choices=["shared", "part_group"])
    p.add_argument("--kmeans-iters", type=int, default=18)
    p.add_argument("--max-train-vectors", type=int, default=50000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def nearest_indices(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    c_norm = (codebook * codebook).sum(dim=1).unsqueeze(0)
    dist = x_norm - 2.0 * x.matmul(codebook.t()) + c_norm
    return torch.argmin(dist, dim=1)


def sample_rows(x: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen)[:max_rows]
    return x[idx]


def train_stage_kmeans(x_cpu: torch.Tensor, k: int, iters: int, seed: int, device: torch.device) -> torch.Tensor:
    x = sample_rows(x_cpu, max(k, 1), seed).float().to(device)
    if x.shape[0] < k:
        reps = math.ceil(k / max(1, x.shape[0]))
        x = x.repeat((reps, 1))[:k]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    init_idx = torch.randperm(x.shape[0], generator=gen, device=device)[:k]
    codebook = x[init_idx].clone()
    train = x_cpu.float()
    train = sample_rows(train, max(k, train.shape[0]), seed + 17).to(device)
    for _ in range(iters):
        idx = nearest_indices(train, codebook)
        new_codebook = torch.zeros_like(codebook)
        counts = torch.bincount(idx, minlength=k).float().to(device)
        new_codebook.index_add_(0, idx, train)
        nonempty = counts > 0
        new_codebook[nonempty] = new_codebook[nonempty] / counts[nonempty].unsqueeze(1)
        new_codebook[~nonempty] = codebook[~nonempty]
        shift = (new_codebook - codebook).abs().max().item()
        codebook = new_codebook
        if shift < 1e-6:
            break
    return codebook.detach().cpu()


def train_rvq_codebooks(x_cpu: torch.Tensor, k: int, stages: int, iters: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    train = sample_rows(x_cpu.float(), max(1, min(x_cpu.shape[0], 50000)), seed)
    residual = train.clone()
    codebooks: list[torch.Tensor] = []
    for stage in range(stages):
        codebook = train_stage_kmeans(residual, k, iters, seed + stage * 1009, device)
        idx = nearest_indices(residual.to(device), codebook.to(device)).cpu()
        residual = residual - codebook[idx]
        codebooks.append(codebook)
    return codebooks


def encode_rvq(x_cpu: torch.Tensor, codebooks: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = x_cpu.float().to(device)
    residual = x.clone()
    recon = torch.zeros_like(x)
    assignments: list[torch.Tensor] = []
    for codebook_cpu in codebooks:
        codebook = codebook_cpu.to(device)
        idx = nearest_indices(residual, codebook)
        q = codebook[idx]
        recon = recon + q
        residual = residual - q
        assignments.append(idx.detach().cpu())
    return recon.detach().cpu(), assignments


def entropy_bits(indices: torch.Tensor, k: int) -> tuple[float, float, float, float]:
    if indices.numel() == 0:
        return 0.0, 0.0, 0.0, 1.0
    counts = torch.bincount(indices.reshape(-1).long(), minlength=k).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nz = probs > 0
    entropy = float((-(probs[nz] * torch.log2(probs[nz])).sum()).item())
    perplexity = float(2.0**entropy)
    used_frac = float((counts > 0).float().mean().item())
    dead_frac = 1.0 - used_frac
    return entropy, perplexity, used_frac, dead_frac


def active_key(part: int, group: int) -> int:
    return part * 1000 + group


def extract_group_vectors(x: torch.Tensor, mask: torch.Tensor, group: int, group_size: int) -> torch.Tensor:
    start = group * group_size
    end = start + group_size
    spatial = mask[0, start].bool()
    return x[0, start:end].permute(1, 2, 0)[spatial].detach().float().cpu()


@torch.inference_mode()
def collect_residual_set(
    net: GLC_Image,
    path: Path,
    q_index: int,
    device: torch.device,
    padding_size: int,
    group_size: int,
    active_parts: set[int],
    active_groups: set[int],
) -> ResidualSet:
    img01 = load_image(path, device)
    x = from_0_1_to_minus1_1(img01)
    _, _, h, w = x.shape
    padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, padding_size)
    x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

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
    B, C, H, W = y.size()
    masks = net.get_mask_four_parts(B, C, H, W, dtype, device)
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
                start = group * group_size
                if start < C:
                    res_vec = extract_group_vectors(y_res, mask, group, group_size)
                    scalar_vec = extract_group_vectors(y_q, mask, group, group_size)
                    bit_vec = extract_group_vectors(bits, mask, group, group_size)
                    vecs.append(res_vec)
                    qvecs.append(scalar_vec)
                    bitvecs.append(bit_vec)
                    keys.append(torch.full((res_vec.shape[0],), active_key(part_idx, group), dtype=torch.long))

        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    vectors = torch.cat(vecs, dim=0)
    scalar_q = torch.cat(qvecs, dim=0)
    scalar_bits = torch.cat(bitvecs, dim=0)
    key_tensor = torch.cat(keys, dim=0)
    return ResidualSet(path.name, q_index, h * w, vectors, scalar_q, scalar_bits, key_tensor)


def key_values(data: list[ResidualSet], scope: str) -> list[int]:
    if scope == "shared":
        return [-1]
    vals = sorted({int(k) for item in data for k in item.keys.tolist()})
    return vals


def vectors_for_key(data: list[ResidualSet], key: int, scope: str) -> torch.Tensor:
    chunks = []
    for item in data:
        if scope == "shared":
            chunks.append(item.vectors)
        else:
            mask = item.keys == key
            if mask.any():
                chunks.append(item.vectors[mask])
    if not chunks:
        return torch.empty((0, data[0].vectors.shape[1]), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def encode_item_with_codebooks(
    item: ResidualSet,
    codebooks_by_key: dict[int, list[torch.Tensor]],
    scope: str,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    recon = torch.empty_like(item.vectors)
    all_assignments: list[torch.Tensor] = []
    per_key_usage = []
    keys = [-1] if scope == "shared" else sorted(codebooks_by_key)
    for key in keys:
        if scope == "shared":
            mask = torch.ones((item.vectors.shape[0],), dtype=torch.bool)
        else:
            mask = item.keys == key
        if not mask.any():
            continue
        key_recon, assignments = encode_rvq(item.vectors[mask], codebooks_by_key[key], device)
        recon[mask] = key_recon
        for stage, idx in enumerate(assignments):
            entropy, perplexity, used_frac, dead_frac = entropy_bits(idx, k)
            per_key_usage.append((stage, entropy, perplexity, used_frac, dead_frac, idx.numel()))
            all_assignments.append(idx)

    entropy_values = [v[1] for v in per_key_usage]
    perplexity_values = [v[2] for v in per_key_usage]
    used_values = [v[3] for v in per_key_usage]
    dead_values = [v[4] for v in per_key_usage]
    usage_stats = {
        "index_entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "index_perplexity_mean": float(np.mean(perplexity_values)) if perplexity_values else 0.0,
        "index_used_frac_mean": float(np.mean(used_values)) if used_values else 0.0,
        "index_dead_frac_mean": float(np.mean(dead_values)) if dead_values else 1.0,
    }
    return recon, all_assignments, usage_stats


def evaluate_setting(
    data_by_q: dict[int, list[ResidualSet]],
    q_index: int,
    scope: str,
    k: int,
    stages: int,
    iters: int,
    max_train_vectors: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    items = data_by_q[q_index]
    for heldout_idx, item in enumerate(items):
        train_items = [x for i, x in enumerate(items) if i != heldout_idx]
        codebooks_by_key: dict[int, list[torch.Tensor]] = {}
        for key in key_values(train_items, scope):
            train_vectors = vectors_for_key(train_items, key, scope)
            train_vectors = sample_rows(train_vectors, max_train_vectors, seed + heldout_idx + key)
            codebooks_by_key[key] = train_rvq_codebooks(
                train_vectors,
                k=k,
                stages=stages,
                iters=iters,
                seed=seed + heldout_idx * 100 + key,
                device=device,
            )

        rvq_recon, assignments, usage_stats = encode_item_with_codebooks(item, codebooks_by_key, scope, k, device)
        scalar_err = item.scalar_q - item.vectors
        rvq_err = rvq_recon - item.vectors
        scalar_mse = float((scalar_err * scalar_err).mean().item())
        rvq_mse = float((rvq_err * rvq_err).mean().item())
        scalar_active_bits = float(item.scalar_bits.sum().item())
        vector_count = int(item.vectors.shape[0])
        fixed_bits = float(vector_count * stages * math.log2(k))
        empirical_bits = 0.0
        for idx in assignments:
            entropy, _, _, _ = entropy_bits(idx, k)
            empirical_bits += float(idx.numel()) * entropy

        row: dict[str, Any] = {
            "q_index": q_index,
            "image": item.image,
            "scope": scope,
            "k": k,
            "stages": stages,
            "vectors": vector_count,
            "active_scalars": int(item.vectors.numel()),
            "pixels": item.pixels,
            "scalar_mse": scalar_mse,
            "rvq_mse": rvq_mse,
            "mse_delta": rvq_mse - scalar_mse,
            "mse_ratio": rvq_mse / scalar_mse if scalar_mse > 0 else float("nan"),
            "scalar_active_bpp": scalar_active_bits / item.pixels,
            "rvq_fixed_bpp": fixed_bits / item.pixels,
            "rvq_empirical_bpp": empirical_bits / item.pixels,
            "fixed_bpp_delta": fixed_bits / item.pixels - scalar_active_bits / item.pixels,
            "empirical_bpp_delta": empirical_bits / item.pixels - scalar_active_bits / item.pixels,
        }
        row.update(usage_stats)
        rows.append(row)
        print(
            f"q={q_index} {item.image} {scope} K={k} L={stages} "
            f"mse={rvq_mse:.6f}/{scalar_mse:.6f} "
            f"bpp_fixed={row['rvq_fixed_bpp']:.5f} scalar={row['scalar_active_bpp']:.5f} "
            f"H={row['index_entropy_mean']:.3f}"
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["q_index"], row["scope"], row["k"], row["stages"])
        groups.setdefault(key, []).append(row)
    out = []
    metrics = [
        "scalar_mse",
        "rvq_mse",
        "mse_delta",
        "mse_ratio",
        "scalar_active_bpp",
        "rvq_fixed_bpp",
        "rvq_empirical_bpp",
        "fixed_bpp_delta",
        "empirical_bpp_delta",
        "index_entropy_mean",
        "index_perplexity_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
    ]
    for (q_index, scope, k, stages), subset in sorted(groups.items()):
        row = {"q_index": q_index, "scope": scope, "k": k, "stages": stages, "images": len(subset)}
        for metric in metrics:
            row[metric] = float(np.mean([float(x[metric]) for x in subset]))
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    active_parts = set(args.active_parts)
    active_groups = set(args.active_groups)

    data_by_q: dict[int, list[ResidualSet]] = {}
    for q in args.q_indexes:
        data_by_q[q] = []
        for path in images:
            item = collect_residual_set(
                net,
                path,
                q,
                device,
                args.padding_size,
                args.group_size,
                active_parts,
                active_groups,
            )
            data_by_q[q].append(item)
            print(f"collected q={q} {path.name} vectors={item.vectors.shape[0]}")

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for scope in args.scopes:
            for k in args.k_list:
                for stages in args.stages_list:
                    rows.extend(
                        evaluate_setting(
                            data_by_q,
                            q_index=q,
                            scope=scope,
                            k=k,
                            stages=stages,
                            iters=args.kmeans_iters,
                            max_train_vectors=args.max_train_vectors,
                            seed=args.seed,
                            device=device,
                        )
                    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_image_csv = args.output_prefix.with_name(args.output_prefix.name + "_per_image.csv")
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    summary = aggregate(rows)
    write_csv(per_image_csv, rows)
    write_csv(summary_csv, summary)
    payload = {
        "experiment": "E170 GLC tail VQ/RVQ leave-one-image-out residual probe",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "q_indexes": args.q_indexes,
        "active_parts": args.active_parts,
        "active_groups": args.active_groups,
        "group_size": args.group_size,
        "k_list": args.k_list,
        "stages_list": args.stages_list,
        "scopes": args.scopes,
        "kmeans_iters": args.kmeans_iters,
        "max_train_vectors": args.max_train_vectors,
        "rows": len(rows),
        "summary": summary,
        "note": "Diagnostic only: codebooks are trained leave-one-image-out on Kodak residuals. This is not a paper-quality training protocol.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E170 GLC Tail VQ/RVQ Residual Probe",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        f"Active parts: `{args.active_parts}`",
        f"Active groups: `{args.active_groups}`",
        "",
        "This is a diagnostic leave-one-image-out residual probe. It does not modify image reconstruction yet and should not be used as a final paper row.",
        "",
        "| q | scope | K | stages | images | scalar MSE | VQ/RVQ MSE | MSE ratio | scalar active bpp | fixed VQ bpp | empirical VQ bpp | H | used | dead |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['q_index']} | {row['scope']} | {row['k']} | {row['stages']} | {row['images']} | "
            f"{row['scalar_mse']:.6f} | {row['rvq_mse']:.6f} | {row['mse_ratio']:.4f} | "
            f"{row['scalar_active_bpp']:.6f} | {row['rvq_fixed_bpp']:.6f} | {row['rvq_empirical_bpp']:.6f} | "
            f"{row['index_entropy_mean']:.4f} | {row['index_used_frac_mean']:.4f} | {row['index_dead_frac_mean']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
