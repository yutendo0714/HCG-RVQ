#!/usr/bin/env python3
"""Benchmark GLC and HCG-RVQ codec-path latency.

This intentionally follows the public GLC image evaluation path: it measures the
model-side encode/decode computation and GLC bit accounting, not a production
arithmetic bitstream. The GLC image release exposes ``test()`` rather than a
file-oriented compress/decompress API, so this script keeps the benchmark honest
about what is being timed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from hcg_rvq.reliability_index_controller import (  # noqa: E402
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    mix_with_fallback,
)
from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import get_state_dict, init_func  # noqa: E402
from tools.eval_glc_qaware_branch_checkpoint import codebooks_from_state_dict  # noqa: E402
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    install_trainable_branch,
    run_instrumented,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    list_images,
    prepare_images,
)
from tools.run_e263_glc_fallback_gate_codec_loop_pilot import (  # noqa: E402
    FEATURES,
    branch_feature_dict,
    feature_tensor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch-checkpoint", type=Path, required=True)
    p.add_argument("--input-path", type=Path, default=ROOT / "experiments" / "analysis" / "clic_test64_subset")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=16)
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--active-threshold", type=float, default=0.05)
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--controller-hidden", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def load_branch(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], dict[int, Any], Any, dict[str, float], dict[str, float]]:
    payload: dict[str, Any] = torch.load(args.branch_checkpoint, map_location="cpu")
    hidden = args.controller_hidden or int(payload.get("args", {}).get("controller_hidden", 16))
    codebooks_by_q = {
        int(q): codebooks_from_state_dict(state, device)
        for q, state in payload["codebooks_by_q"].items()
        if int(q) in set(args.q_indexes)
    }
    controller = ReliabilityIndexMLP(
        ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=hidden, zero_bias=-2.0)
    ).to(device)
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval()
    feature_mu = {str(k): float(v) for k, v in payload["feature_mu"].items()}
    feature_std = {str(k): float(v) for k, v in payload["feature_std"].items()}
    return payload, codebooks_by_q, controller, feature_mu, feature_std


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn: Callable[[], torch.Tensor | dict[str, Any] | tuple[Any, ...]]) -> tuple[float, Any]:
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0, result


def finite_mean(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.fmean(values)) if values else float("nan")


def finite_stdev(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return float("nan")
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def main() -> None:
    init_func()
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)

    payload, codebooks_by_q, controller, feature_mu, feature_std = load_branch(args, device)

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for param in net.parameters():
        param.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    net.masks = {}

    branch_args = SimpleNamespace(
        group_size=args.group_size,
        active_groups=args.active_groups,
        active_parts=args.active_parts,
        scope=args.scope,
        k=args.k,
        stages=args.stages,
    )

    paths = list_images(args.input_path, args.eval_start_index, args.eval_limit)
    if not paths:
        raise SystemExit(f"no eval images in {args.input_path}")

    rows: list[dict[str, Any]] = []
    total_iters = args.warmup + args.repeats
    with torch.no_grad():
        for q in args.q_indexes:
            if q not in codebooks_by_q:
                raise SystemExit(f"q={q} missing from branch checkpoint")
            for image_idx, path in enumerate(paths):
                item = prepare_images([path], device, args.padding_size, args.eval_crop_size)[0]
                pixels = float(item.height * item.width)
                for rep in range(total_iters):
                    keep = rep >= args.warmup

                    net.forward_four_part_prior = official_forward

                    def official_fn() -> dict[str, Any]:
                        result = net.test(item.x_pad, q)
                        # Touch scalar values so CUDA work cannot be optimized away.
                        _ = float(result["x_hat"].detach().float().mean().item())
                        return result

                    ms, result = timed_ms(device, official_fn)
                    if keep:
                        rows.append(
                            {
                                "path": str(path),
                                "image": path.name,
                                "image_index": image_idx,
                                "q_index": q,
                                "mode": "glc_official_test",
                                "ms": ms,
                                "bpp": float(result["bit"]) / pixels,
                                "bit_y": float(result["bit_y"]),
                                "bit_z": float(result["bit_z"]),
                                "rep": rep - args.warmup,
                            }
                        )

                    install_trainable_branch(net, codebooks_by_q[q], branch_args)

                    def branch_fn() -> tuple[torch.Tensor, dict[str, float]]:
                        x_hat, stats = run_instrumented(net, item.x_pad, q)
                        _ = float(x_hat.detach().float().mean().item())
                        return x_hat, stats

                    ms, (_, branch_stats) = timed_ms(device, branch_fn)
                    if keep:
                        rows.append(
                            {
                                "path": str(path),
                                "image": path.name,
                                "image_index": image_idx,
                                "q_index": q,
                                "mode": "hcg_branch_only",
                                "ms": ms,
                                "bpp": float(branch_stats["gaussian_bits_total"]) / pixels,
                                "bit_y": float(branch_stats["gaussian_bits_y"]),
                                "bit_z": float(branch_stats["bits_z"]),
                                "rep": rep - args.warmup,
                            }
                        )

                    net.forward_four_part_prior = official_forward

                    def soft_qgate_export_fn() -> tuple[torch.Tensor, dict[str, float]]:
                        base_pad, base_stats = run_instrumented(net, item.x_pad, q)
                        install_trainable_branch(net, codebooks_by_q[q], branch_args)
                        branch_pad, branch_stats = run_instrumented(net, item.x_pad, q)
                        feature_row = branch_feature_dict(base_stats, branch_stats, pixels)
                        features = feature_tensor(feature_row, feature_mu, feature_std, base_pad.device)
                        ctrl = controller(features)
                        mixed, gate = mix_with_fallback(
                            base_pad,
                            branch_pad,
                            ctrl["active_logit"],
                            active_threshold=args.active_threshold,
                            hard=False,
                            max_gate=args.max_gate,
                        )
                        _ = float(mixed.detach().float().mean().item()) + float(gate.detach().float().mean().item())
                        return mixed, feature_row

                    ms, (_, feature_row) = timed_ms(device, soft_qgate_export_fn)
                    if keep:
                        replacement_bpp = float(feature_row["base_bpp"]) + float(feature_row["active_replacement_delta_bpp"])
                        rows.append(
                            {
                                "path": str(path),
                                "image": path.name,
                                "image_index": image_idx,
                                "q_index": q,
                                "mode": "hcg_soft_qgate_export_path",
                                "ms": ms,
                                "bpp": replacement_bpp,
                                "bit_y": float("nan"),
                                "bit_z": float("nan"),
                                "rep": rep - args.warmup,
                            }
                        )

                    net.forward_four_part_prior = official_forward
                del item
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), int(row["q_index"])), []).append(row)

    summary = []
    for (mode, q), group_rows in sorted(grouped.items()):
        ms_vals = [float(r["ms"]) for r in group_rows]
        bpp_vals = [float(r["bpp"]) for r in group_rows]
        summary.append(
            {
                "mode": mode,
                "q_index": q,
                "images": len({r["image"] for r in group_rows}),
                "samples": len(group_rows),
                "mean_ms": finite_mean(ms_vals),
                "stdev_ms": finite_stdev(ms_vals),
                "p50_ms": percentile(ms_vals, 50.0),
                "p90_ms": percentile(ms_vals, 90.0),
                "images_per_sec": 1000.0 / finite_mean(ms_vals) if finite_mean(ms_vals) > 0 else float("nan"),
                "mean_bpp": finite_mean(bpp_vals),
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")

    with json_path.open("w") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "checkpoint_step": payload.get("step"),
                "note": "GLC image release exposes test() bit accounting; this is model codec-path timing, not arithmetic bitstream timing.",
                "summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )
    with csv_path.open("w", newline="") as f:
        fieldnames = ["mode", "q_index", "images", "samples", "mean_ms", "stdev_ms", "p50_ms", "p90_ms", "images_per_sec", "mean_bpp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    with md_path.open("w") as f:
        f.write("# GLC/HCG Codec-Path Speed Benchmark\n\n")
        f.write("This benchmark times the public GLC image `test()` path and HCG-RVQ branch paths. It does not claim production arithmetic bitstream speed.\n\n")
        f.write("| mode | q | images | mean ms/img | p50 | p90 | img/s | mean bpp |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary:
            f.write(
                f"| {row['mode']} | {row['q_index']} | {row['images']} | "
                f"{row['mean_ms']:.3f} | {row['p50_ms']:.3f} | {row['p90_ms']:.3f} | "
                f"{row['images_per_sec']:.3f} | {row['mean_bpp']:.6f} |\n"
            )
    print(f"[done] wrote {json_path}, {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
