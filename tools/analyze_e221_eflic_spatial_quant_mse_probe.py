#!/usr/bin/env python3
"""Spatial local quantization-MSE probe for EF-LIC projected HCG.

This diagnostic intentionally uses raw y_norm only to create teacher labels.
The predictor features are decoder-reproducible predecision maps: z/index
statistics, current slice mean/scale maps, and the already available support
buffer. The goal is to test whether HCG's local geometry helps at spatial
positions that are explainable from decoder-side context.
"""
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

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(EFLIC_DIR))

from EF_LIC import G_CH_Y, N_E, model  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import hcg_rvq_encode_decode, index_stats, tensor_stats  # noqa: E402
from test import load_checkpoint, load_image, list_images, replicate_pad  # noqa: E402


FEATURE_PREFIXES = ["mean", "scale", "support", "support_hyper", "support_prev"]
FEATURE_REDUCTIONS = ["mean", "abs_mean", "rms", "std", "min", "max"]


@dataclass(frozen=True)
class InputSpec:
    dataset: str
    path: Path
    start: int
    max_images: int | None


@dataclass(frozen=True)
class Stump:
    feature: str
    threshold: float
    polarity: str
    train_score: float
    train_active_frac: float


def parse_input_spec(spec: str, default_start: int, default_max: int | None) -> InputSpec:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--input must be dataset=path[:start[:max]], got {spec!r}")
    dataset, rest = spec.split("=", 1)
    dataset = dataset.strip()
    if not dataset:
        raise argparse.ArgumentTypeError(f"empty dataset name in {spec!r}")

    start = default_start
    max_images = default_max
    parts = rest.rsplit(":", 2)
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        path_text = ":".join(parts[:-2])
        start = int(parts[-2])
        max_images = int(parts[-1])
    elif len(parts) >= 2 and parts[-1].isdigit():
        path_text = ":".join(parts[:-1])
        start = default_start
        max_images = int(parts[-1])
    else:
        path_text = rest
    return InputSpec(dataset=dataset, path=Path(path_text), start=start, max_images=max_images)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def map_reductions(x: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    xf = x.detach().float()
    if xf.shape[1] == 0:
        b, _, h, w = xf.shape
        nan_map = xf.new_full((b, h, w), float("nan"))
        return {f"{prefix}_{name}": nan_map for name in FEATURE_REDUCTIONS}
    return {
        f"{prefix}_mean": xf.mean(dim=1),
        f"{prefix}_abs_mean": xf.abs().mean(dim=1),
        f"{prefix}_rms": torch.sqrt(xf.square().mean(dim=1).clamp_min(0.0)),
        f"{prefix}_std": xf.std(dim=1, unbiased=False),
        f"{prefix}_min": xf.min(dim=1).values,
        f"{prefix}_max": xf.max(dim=1).values,
    }


def global_feature_values(z_hat: torch.Tensor, z_inds: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(tensor_stats(z_hat, "z_hat"))
    out.update(index_stats(z_inds, int(N_E[-1]), "z_index"))
    return out


def sample_flat_indices(total: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=max_samples, replace=False).astype(np.int64))


def add_sample_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    image: str,
    slice_id: int,
    h: int,
    w: int,
    sample_indices: np.ndarray,
    base_mse: torch.Tensor,
    active_mse: torch.Tensor,
    feature_maps: dict[str, torch.Tensor],
    global_features: dict[str, float],
) -> None:
    base_flat = base_mse.reshape(-1).detach().cpu().numpy()
    active_flat = active_mse.reshape(-1).detach().cpu().numpy()
    feature_flat = {k: v.reshape(-1).detach().cpu().numpy() for k, v in feature_maps.items()}
    for flat_idx in sample_indices:
        y_pos = int(flat_idx // w)
        x_pos = int(flat_idx % w)
        base = float(base_flat[flat_idx])
        active = float(active_flat[flat_idx])
        delta = active - base
        row: dict[str, Any] = {
            "dataset": dataset,
            "image": image,
            "slice_id": slice_id,
            "flat_index": int(flat_idx),
            "y_pos": y_pos,
            "x_pos": x_pos,
            "y_norm_pos": float(y_pos / max(1, h - 1)),
            "x_norm_pos": float(x_pos / max(1, w - 1)),
            "base_mse": base,
            "active_mse": active,
            "delta_mse": delta,
            "oracle_delta_mse": min(delta, 0.0),
            "helpful": int(delta < 0.0),
        }
        row.update(global_features)
        for key, values in feature_flat.items():
            row[key] = float(values[flat_idx])
        rows.append(row)


def summarize_local(rows: list[dict[str, Any]], active: np.ndarray) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "delta_mse": 0.0,
            "active_frac": 0.0,
            "helpful_frac": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    delta = np.array([safe_float(r["delta_mse"]) for r in rows], dtype=float)
    active = active.astype(bool)
    helpful = delta < 0.0
    selected = np.where(active, delta, 0.0)
    true_pos = active & helpful
    return {
        "samples": len(rows),
        "delta_mse": float(np.nanmean(selected)),
        "all_on_delta_mse": float(np.nanmean(delta)),
        "oracle_delta_mse": float(np.nanmean(np.minimum(delta, 0.0))),
        "active_frac": float(active.mean()),
        "helpful_frac": float(helpful.mean()),
        "precision": float(true_pos.sum() / max(1, active.sum())),
        "recall": float(true_pos.sum() / max(1, helpful.sum())),
    }


def valid_feature_values(rows: list[dict[str, Any]], feature: str) -> np.ndarray:
    return np.array([safe_float(r.get(feature)) for r in rows], dtype=float)


def evaluate_stump(rows: list[dict[str, Any]], stump: Stump) -> tuple[np.ndarray, dict[str, Any]]:
    vals = valid_feature_values(rows, stump.feature)
    if stump.polarity == "le":
        active = vals <= stump.threshold
    elif stump.polarity == "ge":
        active = vals >= stump.threshold
    else:
        raise ValueError(stump.polarity)
    active = active & np.isfinite(vals)
    return active, summarize_local(rows, active)


def fit_stump(rows: list[dict[str, Any]], feature_names: list[str]) -> Stump:
    best = Stump("__off__", 0.0, "ge", 0.0, 0.0)
    best_score = 0.0
    n_rows = max(1, len(rows))
    deltas = np.array([safe_float(r["delta_mse"]) for r in rows], dtype=float)
    for feature in feature_names:
        vals = valid_feature_values(rows, feature)
        finite = np.isfinite(vals) & np.isfinite(deltas)
        if not finite.any():
            continue
        finite_vals = vals[finite]
        finite_deltas = deltas[finite]
        order = np.argsort(finite_vals, kind="mergesort")
        sorted_vals = finite_vals[order]
        sorted_deltas = finite_deltas[order]
        unique, counts = np.unique(sorted_vals, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_deltas = np.cumsum(sorted_deltas)
        group_delta_sums = cum_deltas[cum_counts - 1]
        total_count = int(cum_counts[-1])
        total_delta = float(cum_deltas[-1])
        if unique.size == 1:
            thresholds = np.array([unique[0] - 1e-9, unique[0] + 1e-9], dtype=float)
        else:
            mids = (unique[:-1] + unique[1:]) * 0.5
            thresholds = np.concatenate(([unique[0] - 1e-9], mids, [unique[-1] + 1e-9]))

        le_sums = np.concatenate(([0.0], group_delta_sums))
        le_counts = np.concatenate(([0], cum_counts))
        for threshold, delta_sum, active_count in zip(thresholds, le_sums, le_counts):
            score = float(delta_sum) / n_rows
            if score < best_score - 1e-15:
                best_score = score
                best = Stump(feature, float(threshold), "le", score, float(active_count) / n_rows)

        ge_sums = np.concatenate(([total_delta], total_delta - group_delta_sums))
        ge_counts = np.concatenate(([total_count], total_count - cum_counts))
        for threshold, delta_sum, active_count in zip(thresholds, ge_sums, ge_counts):
            score = float(delta_sum) / n_rows
            if score < best_score - 1e-15:
                best_score = score
                best = Stump(feature, float(threshold), "ge", score, float(active_count) / n_rows)
    return best


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xm = x[mask]
    ym = y[mask]
    if float(xm.std()) == 0.0 or float(ym.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(xm, ym)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def feature_correlations(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    target = np.array([safe_float(r["delta_mse"]) for r in rows], dtype=float)
    out: list[dict[str, Any]] = []
    for feature in feature_names:
        vals = valid_feature_values(rows, feature)
        out.append(
            {
                "feature": feature,
                "pearson_delta_mse": pearson(vals, target),
                "spearman_delta_mse": spearman(vals, target),
                "finite_frac": float(np.isfinite(vals).mean()),
            }
        )
    out.sort(
        key=lambda r: abs(r["spearman_delta_mse"]) if math.isfinite(r["spearman_delta_mse"]) else -1.0,
        reverse=True,
    )
    return out


@torch.inference_mode()
def collect_spatial_rows(
    net: model,
    *,
    image_paths: list[Path],
    dataset: str,
    force_ind: int,
    alpha: float,
    direction_source: str,
    device: torch.device,
    max_samples_per_slice: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    for path in image_paths:
        frame = load_image(path, device)
        _, _, h0, w0 = frame.shape
        padded = replicate_pad(frame, h0, w0)

        y = net.g_a(padded)
        z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
        support_buf = net._support_buf(net.h_s(z_hat))
        global_features = global_feature_values(z_hat, z_inds)

        b, _, h2, w2 = support_buf.shape
        if b != 1:
            raise RuntimeError("this diagnostic expects one image per forward pass")
        y_slice = y.new_empty(b, G_CH_Y, h2, w2)
        total_positions = h2 * w2

        for slice_id in range(4):
            mean, scale = net._mean_scale(support_buf, slice_id)
            net._qt_select(y, slice_id, y_slice)
            y_norm = (y_slice - mean) / scale

            _, base_hat_norm = net.quantizes[force_ind][slice_id].encode_decode(y_norm.clone())
            context = scale if direction_source == "logscale" else mean
            _, active_hat_norm, active_stats = hcg_rvq_encode_decode(
                net.quantizes[force_ind][slice_id],
                y_norm,
                context,
                alpha,
                direction_source,
            )

            base_mse = (base_hat_norm - y_norm).float().square().mean(dim=1)[0]
            active_mse = (active_hat_norm - y_norm).float().square().mean(dim=1)[0]
            delta = active_mse - base_mse
            finite = torch.isfinite(delta) & torch.isfinite(base_mse) & torch.isfinite(active_mse)

            support_used = support_buf[:, : (4 + slice_id) * G_CH_Y]
            support_hyper = support_buf[:, : 4 * G_CH_Y]
            support_prev = support_buf[:, 4 * G_CH_Y : (4 + slice_id) * G_CH_Y]
            feature_maps: dict[str, torch.Tensor] = {}
            feature_maps.update(map_reductions(mean, "mean"))
            feature_maps.update(map_reductions(scale, "scale"))
            feature_maps.update(map_reductions(support_used, "support"))
            feature_maps.update(map_reductions(support_hyper, "support_hyper"))
            feature_maps.update(map_reductions(support_prev, "support_prev"))

            sample_indices = sample_flat_indices(total_positions, max_samples_per_slice, rng)
            add_sample_rows(
                rows,
                dataset=dataset,
                image=path.name,
                slice_id=slice_id,
                h=h2,
                w=w2,
                sample_indices=sample_indices,
                base_mse=base_mse,
                active_mse=active_mse,
                feature_maps={k: v[0] for k, v in feature_maps.items()},
                global_features=global_features,
            )

            image_summaries.append(
                {
                    "dataset": dataset,
                    "image": path.name,
                    "slice_id": slice_id,
                    "positions": int(total_positions),
                    "sampled_positions": int(len(sample_indices)),
                    "base_mse": float(base_mse[finite].mean().item()) if finite.any() else float("nan"),
                    "active_mse": float(active_mse[finite].mean().item()) if finite.any() else float("nan"),
                    "delta_mse": float(delta[finite].mean().item()) if finite.any() else float("nan"),
                    "oracle_delta_mse": float(torch.minimum(delta[finite], delta.new_zeros(())).mean().item()) if finite.any() else float("nan"),
                    "helpful_frac": float((delta[finite] < 0.0).float().mean().item()) if finite.any() else float("nan"),
                    "nonfinite_positions": int((~finite).sum().item()),
                    "index_entropy": float(active_stats["avg_index_entropy"]),
                    "index_used_frac": float(active_stats["avg_index_used_frac"]),
                    "geometry_delta_rms": float(active_stats["avg_geometry_delta_rms"]),
                    "residual_error_rms": float(active_stats["avg_residual_error_rms"]),
                }
            )

            y_hat_i = base_hat_norm * scale + mean
            if slice_id < 3:
                support_buf[:, (4 + slice_id) * G_CH_Y : (5 + slice_id) * G_CH_Y].copy_(y_hat_i)
        print(f"{dataset} {path.name}: collected {4 * min(max_samples_per_slice, total_positions)} spatial samples")
    return rows, image_summaries


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    if preferred:
        fields = preferred + [f for f in fields if f not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    feature_names = ["slice_id", "x_norm_pos", "y_norm_pos"]
    feature_names.extend(["z_hat_abs_mean", "z_hat_std", "z_hat_rms", "z_hat_min", "z_hat_max"])
    feature_names.extend(["z_index_entropy", "z_index_perplexity", "z_index_used_frac", "z_index_max_prob"])
    for prefix in FEATURE_PREFIXES:
        for reduction in FEATURE_REDUCTIONS:
            feature_names.append(f"{prefix}_{reduction}")
    return [f for f in feature_names if any(f in row for row in rows)]


def build_policy_summaries(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    datasets = sorted({r["dataset"] for r in rows})
    groups: list[tuple[str, list[dict[str, Any]]]] = [("pooled", rows)]
    groups.extend((d, [r for r in rows if r["dataset"] == d]) for d in datasets)
    groups.extend((f"slice{sid}", [r for r in rows if int(r["slice_id"]) == sid]) for sid in range(4))

    for name, group_rows in groups:
        if not group_rows:
            continue
        delta = np.array([safe_float(r["delta_mse"]) for r in group_rows], dtype=float)
        policies = {
            "all_off": np.zeros(len(group_rows), dtype=bool),
            "all_on": np.ones(len(group_rows), dtype=bool),
            "oracle_local": delta < 0.0,
        }
        for policy_name, active in policies.items():
            summary = summarize_local(group_rows, active)
            summary.update({"group": name, "policy": policy_name, "feature": "", "condition": ""})
            summaries.append(summary)
        stump = fit_stump(group_rows, feature_names)
        _, summary = evaluate_stump(group_rows, stump)
        summary.update(
            {
                "group": name,
                "policy": "same_table_best_stump",
                "feature": stump.feature,
                "condition": f"{stump.feature} {stump.polarity} {stump.threshold:.10g}",
                "threshold": stump.threshold,
                "polarity": stump.polarity,
            }
        )
        summaries.append(summary)

    for held in datasets:
        train = [r for r in rows if r["dataset"] != held]
        eval_rows = [r for r in rows if r["dataset"] == held]
        if not train or not eval_rows:
            continue
        stump = fit_stump(train, feature_names)
        _, summary = evaluate_stump(eval_rows, stump)
        summary.update(
            {
                "group": held,
                "policy": "leave_dataset_out_stump",
                "feature": stump.feature,
                "condition": f"{stump.feature} {stump.polarity} {stump.threshold:.10g}",
                "threshold": stump.threshold,
                "polarity": stump.polarity,
                "train_group": "+".join(sorted({r["dataset"] for r in train})),
            }
        )
        summaries.append(summary)
    return summaries


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    specs: list[InputSpec],
    rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    lines = [
        "# E221 EF-LIC Spatial Quant-MSE Probe",
        "",
        "This is a diagnostic probe, not a final paper metric row.",
        "It compares fixed EF-LIC RVQ against projected-HCG RVQ on normalized y-slice maps and asks whether local quantization-MSE improvements are predictable from decoder-reproducible context maps.",
        "",
        f"- Force index: `{args.force_ind}`",
        f"- Direction source: `{args.direction_source}`",
        f"- Alpha: `{args.alpha}`",
        f"- Max samples per slice: `{args.max_samples_per_slice}`",
        f"- Device: `{args.device}`",
        f"- Sample rows: `{len(rows)}`",
        "",
        "Inputs:",
    ]
    for spec in specs:
        lines.append(f"- `{spec.dataset}`: `{spec.path}` start={spec.start} max={spec.max_images}")

    lines.extend(
        [
            "",
            "Slice-level tensor summary:",
            "",
            "| dataset | image | slice | positions | dMSE all-on | oracle dMSE | helpful | nonfinite | entropy | geom RMS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in slice_rows[:80]:
        lines.append(
            f"| {row['dataset']} | {row['image']} | {row['slice_id']} | {row['positions']} | "
            f"{row['delta_mse']:+.8f} | {row['oracle_delta_mse']:+.8f} | {row['helpful_frac']:.3f} | "
            f"{row['nonfinite_positions']} | {row['index_entropy']:.4f} | {row['geometry_delta_rms']:.6f} |"
        )
    if len(slice_rows) > 80:
        lines.append("| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |")

    lines.extend(
        [
            "",
            "Policy/headroom summary on sampled spatial positions:",
            "",
            "| group | policy | samples | dMSE | all-on dMSE | oracle dMSE | active | helpful | precision | recall | feature | condition |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row.get('group', '')} | {row.get('policy', '')} | {int(row.get('samples', 0))} | "
            f"{float(row.get('delta_mse', 0.0)):+.8f} | {float(row.get('all_on_delta_mse', 0.0)):+.8f} | "
            f"{float(row.get('oracle_delta_mse', 0.0)):+.8f} | {float(row.get('active_frac', 0.0)):.3f} | "
            f"{float(row.get('helpful_frac', 0.0)):.3f} | {float(row.get('precision', 0.0)):.3f} | "
            f"{float(row.get('recall', 0.0)):.3f} | {row.get('feature', '')} | {row.get('condition', '')} |"
        )

    lines.extend(
        [
            "",
            "Top absolute Spearman correlations with local `delta_mse` (negative `delta_mse` means active HCG reduced quantization MSE):",
            "",
            "| rank | feature | spearman | pearson | finite |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for i, row in enumerate(correlations[:20], 1):
        lines.append(
            f"| {i} | {row['feature']} | {row['spearman_delta_mse']:+.4f} | "
            f"{row['pearson_delta_mse']:+.4f} | {row['finite_frac']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The local oracle is a teacher/headroom upper bound; it uses raw normalized latent error and is not a deployable codec rule.",
            "- Same-table stumps reveal whether decoder-side maps contain local signal. Leave-dataset-out stumps check whether a single hand rule transfers across domains.",
            "- If local oracle is strong but leave-dataset-out stumps are weak, the next paper-facing step should be a trained local HCG geometry/strength head using these decoder-safe map inputs, not another global threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="dataset=path[:start[:max]]")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--direction-source", choices=["mean", "logscale", "fixed"], default="mean")
    p.add_argument("--default-start", type=int, default=0)
    p.add_argument("--default-max-images", type=int, default=4)
    p.add_argument("--max-samples-per-slice", type=int, default=512)
    p.add_argument("--seed", type=int, default=221)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_input_spec(s, args.default_start, args.default_max_images) for s in args.input]
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    all_rows: list[dict[str, Any]] = []
    all_slice_rows: list[dict[str, Any]] = []
    selected_images: dict[str, list[str]] = {}
    for spec in specs:
        images = list_images(spec.path)
        end = None if spec.max_images is None else spec.start + spec.max_images
        subset = images[spec.start:end]
        if not subset:
            raise SystemExit(f"no images selected for {spec.dataset}: {spec.path} start={spec.start} max={spec.max_images}")
        selected_images[spec.dataset] = [p.name for p in subset]
        rows, slice_rows = collect_spatial_rows(
            net,
            image_paths=subset,
            dataset=spec.dataset,
            force_ind=args.force_ind,
            alpha=args.alpha,
            direction_source=args.direction_source,
            device=device,
            max_samples_per_slice=args.max_samples_per_slice,
            rng=rng,
        )
        all_rows.extend(rows)
        all_slice_rows.extend(slice_rows)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    samples_csv = args.output_prefix.with_suffix(".samples.csv")
    slice_csv = args.output_prefix.with_suffix(".slice_summary.csv")
    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    corr_csv = args.output_prefix.with_suffix(".correlations.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    feature_names = build_feature_names(all_rows)
    summaries = build_policy_summaries(all_rows, feature_names)
    correlations = feature_correlations(all_rows, feature_names)

    sample_preferred = [
        "dataset",
        "image",
        "slice_id",
        "flat_index",
        "y_pos",
        "x_pos",
        "base_mse",
        "active_mse",
        "delta_mse",
        "oracle_delta_mse",
        "helpful",
    ] + feature_names
    write_csv(samples_csv, all_rows, preferred=sample_preferred)
    write_csv(slice_csv, all_slice_rows)
    write_csv(summary_csv, summaries)
    write_csv(corr_csv, correlations, preferred=["feature", "spearman_delta_mse", "pearson_delta_mse", "finite_frac"])

    payload = {
        "experiment": "E221 EF-LIC spatial quant-MSE probe",
        "checkpoint": str(args.ckpt_path),
        "device": str(device),
        "force_ind": args.force_ind,
        "alpha": args.alpha,
        "direction_source": args.direction_source,
        "max_samples_per_slice": args.max_samples_per_slice,
        "inputs": [{"dataset": s.dataset, "path": str(s.path), "start": s.start, "max_images": s.max_images} for s in specs],
        "selected_images": selected_images,
        "samples": len(all_rows),
        "slice_rows": len(all_slice_rows),
        "features": feature_names,
        "summaries": summaries,
        "top_correlations": correlations[:30],
        "interpretation": (
            "Diagnostic local quantization-MSE teacher probe. Raw y_norm is used only to label "
            "where active projected-HCG reduces MSE; predictors use decoder-reproducible context maps."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(
        md_path,
        args=args,
        specs=specs,
        rows=all_rows,
        slice_rows=all_slice_rows,
        summaries=summaries,
        correlations=correlations,
    )
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
