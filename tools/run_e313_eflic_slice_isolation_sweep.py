#!/usr/bin/env python3
"""Run EF-LIC HCG slice-isolation sweeps with one model/controller load.

This extends the E312 one-image probe into a reusable small-cycle experiment for
local/slice controller-label design. It evaluates several ``--active-slices``
sets in the actual EF-LIC codec loop while keeping the original fixed-length
payload contract visible. When requested, it also computes perceptual metrics so
PSNR does not drive the generative-compression decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import run_e295_eflic_hcg_branch_controller_integration_smoke as e295  # noqa: E402
from hcg_rvq.eflic_local_controller import EFLICHCGBranchController, EFLICHCGBranchControllerConfig  # noqa: E402

ALL_SLICES = {0, 1, 2, 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments/data/kodak24")
    p.add_argument("--ckpt-path", type=Path, default=e295.EFLIC_DIR / "ckpt/checkpoint.pth.tar")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e313_eflic_slice_isolation_sweep_kodak4",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--direction-source", default="fixed", choices=["mean", "logscale", "fixed"])
    p.add_argument("--mode", default="trained_soft", choices=sorted(e295.MODE_TO_GATE))
    p.add_argument(
        "--controller-state",
        type=Path,
        default=ROOT / "experiments/analysis/e296_eflic_hcg_controller_context_train_smoke_t4_e2_s16.pth",
    )
    p.add_argument("--active-threshold", type=float, default=0.95)
    p.add_argument("--max-risk", type=float, default=999.0)
    p.add_argument("--risk-temperature", type=float, default=1.0)
    p.add_argument("--max-alpha", type=float, default=0.02)
    p.add_argument(
        "--compute-perceptual",
        action="store_true",
        help="Also compute MS-SSIM, LPIPS, and DISTS. PSNR is kept only as a codec-health diagnostic.",
    )
    p.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    p.add_argument(
        "--perceptual-score-lpips-weight",
        type=float,
        default=3.0,
        help="Weight in score = delta_DISTS + weight * delta_LPIPS. Lower is better.",
    )
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=313)
    p.add_argument(
        "--slice-sets",
        nargs="+",
        default=["all", "0", "1", "2", "3", "0,1", "1,2,3", "0,2,3", "0,1,3", "0,1,2"],
    )
    return p.parse_args()


def finite_stats(stats: dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in stats.values())


def slice_count(label: str) -> int:
    parsed = e295.parse_active_slices(label)
    if parsed is None:
        return 4
    return len(parsed)


def omitted_slice(label: str) -> str:
    parsed = e295.parse_active_slices(label)
    slices = set(ALL_SLICES if parsed is None else parsed)
    missing = sorted(ALL_SLICES - slices)
    return str(missing[0]) if len(missing) == 1 else ""


def single_slice(label: str) -> str:
    parsed = e295.parse_active_slices(label)
    if parsed is None or len(parsed) != 1:
        return ""
    return str(next(iter(parsed)))


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    images = e295.list_images(args.image_dir)[args.start_index : args.start_index + args.max_images]
    if not images:
        raise SystemExit(f"no images selected from {args.image_dir}")

    net = e295.model().to(device).eval()
    net.load_state_dict(e295.load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    state = None
    config_kwargs: dict[str, Any] = {"max_alpha": args.max_alpha}
    if args.controller_state is not None:
        state = torch.load(args.controller_state, map_location=device, weights_only=False)
        saved_config = state.get("config") if isinstance(state, dict) else None
        if isinstance(saved_config, dict):
            config_kwargs.update(saved_config)
    controller = EFLICHCGBranchController(EFLICHCGBranchControllerConfig(**config_kwargs)).to(device).eval()
    if state is not None:
        controller.load_state_dict(state.get("model", state), strict=True)
        controller.eval()
    metric_fns = e295.build_metric_fns(args, device)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for path in images:
            frame = e295.load_image(path, device)
            _, _, h, w = frame.shape
            padded = e295.replicate_pad(frame, h, w)
            orig_inds = net.compress(padded.clone(), force_ind=args.force_ind)
            orig_payload, _, _ = e295.pack_inds(net, orig_inds)
            orig_x_hat = net.decompress(orig_inds, force_ind=args.force_ind)[:, :, :h, :w]
            base_psnr = e295.psnr_from_mse(e295.mse01(orig_x_hat, frame))

            for label in args.slice_sets:
                active_slices = e295.parse_active_slices(label)
                active_inds, active_x_hat_forward, stats = e295.controller_compress_forward(
                    net,
                    controller,
                    padded.clone(),
                    force_ind=args.force_ind,
                    mode=args.mode,
                    direction_source=args.direction_source,
                    active_threshold=args.active_threshold,
                    max_risk=args.max_risk,
                    risk_temperature=args.risk_temperature,
                    active_slices=active_slices,
                )
                active_payload, _, _ = e295.pack_inds(net, active_inds)
                active_x_hat_dec = e295.controller_decompress(
                    net,
                    controller,
                    active_inds,
                    force_ind=args.force_ind,
                    mode=args.mode,
                    direction_source=args.direction_source,
                    active_threshold=args.active_threshold,
                    max_risk=args.max_risk,
                    risk_temperature=args.risk_temperature,
                    active_slices=active_slices,
                )[:, :, :h, :w]
                active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
                decode_diff = (active_x_hat_fwd - active_x_hat_dec).abs()
                baseline_diff = (active_x_hat_dec - orig_x_hat).abs()
                mismatch = e295.compare_inds(orig_inds, active_inds)
                active_psnr = e295.psnr_from_mse(e295.mse01(active_x_hat_dec, frame))
                perceptual = e295.compute_perceptual_metrics(
                    metric_fns,
                    base_x_hat=orig_x_hat,
                    active_x_hat=active_x_hat_dec,
                    frame=frame,
                )
                nonfinite = int(
                    (not torch.isfinite(active_x_hat_dec).all().item())
                    or (not torch.isfinite(active_x_hat_fwd).all().item())
                    or (not finite_stats(stats))
                    or (bool(perceptual) and not finite_stats(perceptual))
                )
                row: dict[str, Any] = {
                    "image": path.name,
                    "mode": args.mode,
                    "active_slices": label,
                    "slice_count": slice_count(label),
                    "single_slice": single_slice(label),
                    "omitted_slice": omitted_slice(label),
                    "force_ind": args.force_ind,
                    "direction_source": args.direction_source,
                    "bpp": len(active_payload) * 8.0 / float(h * w),
                    "delta_bpp": (len(active_payload) - len(orig_payload)) * 8.0 / float(h * w),
                    "payload_len_equal": int(len(active_payload) == len(orig_payload)),
                    "payload_equal": int(active_payload == orig_payload),
                    "base_psnr": base_psnr,
                    "active_psnr": active_psnr,
                    "delta_psnr": active_psnr - base_psnr,
                    "max_decode_diff": float(decode_diff.max().item()),
                    "mean_decode_diff": float(decode_diff.mean().item()),
                    "max_baseline_diff": float(baseline_diff.max().item()),
                    "mean_baseline_diff": float(baseline_diff.mean().item()),
                    "nonfinite": nonfinite,
                }
                row.update(perceptual)
                if perceptual:
                    row["perceptual_score"] = float(
                        row["delta_dists"] + args.perceptual_score_lpips_weight * row["delta_lpips"]
                    )
                    row["perceptual_score_win"] = int(float(row["perceptual_score"]) < 0.0)
                    row["triple_perceptual_win"] = int(
                        float(row["delta_dists"]) < 0.0
                        and float(row["delta_lpips"]) < 0.0
                        and float(row["delta_ms_ssim"]) > 0.0
                    )
                row.update(mismatch)
                row.update(stats)
                row["contract_ok"] = int(
                    abs(row["delta_bpp"]) <= 1e-12
                    and row["max_decode_diff"] <= 1e-10
                    and int(row["nonfinite"]) == 0
                    and int(row["payload_len_equal"]) == 1
                )
                rows.append(row)
                extra_metrics = ""
                if perceptual:
                    extra_metrics = (
                        f" score={row['perceptual_score']:+.6f}"
                        f" dMS={row['delta_ms_ssim']:+.6f}"
                        f" dLPIPS={row['delta_lpips']:+.6f}"
                        f" dDISTS={row['delta_dists']:+.6f}"
                    )
                print(
                    f"image={path.name} slices={label} dPSNR={row['delta_psnr']:+.6f}{extra_metrics} "
                    f"dbpp={row['delta_bpp']:+.6f} decmax={row['max_decode_diff']:.1e} "
                    f"enabled={row.get('y_slice_enabled', float('nan')):.2f} nonfinite={row['nonfinite']}"
                )
    return rows, images


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    has_perceptual = any("perceptual_score" in r for r in rows)
    by_set: list[dict[str, Any]] = []
    for label in sorted({str(r["active_slices"]) for r in rows}, key=lambda x: (slice_count(x), x)):
        subset = [r for r in rows if str(r["active_slices"]) == label]
        item: dict[str, Any] = {
            "active_slices": label,
            "slice_count": slice_count(label),
            "images": len(subset),
            "mean_delta_psnr": mean([float(r["delta_psnr"]) for r in subset]),
            "worst_delta_psnr": min(float(r["delta_psnr"]) for r in subset),
            "best_delta_psnr": max(float(r["delta_psnr"]) for r in subset),
            "psnr_win_frac": mean([float(r["delta_psnr"] > 0.0) for r in subset]),
            "contract_ok_frac": mean([float(r["contract_ok"]) for r in subset]),
            "mean_delta_bpp": mean([float(r["delta_bpp"]) for r in subset]),
            "mean_slice_enabled": mean([float(r.get("y_slice_enabled", 1.0)) for r in subset]),
        }
        if has_perceptual:
            scores = [float(r["perceptual_score"]) for r in subset if "perceptual_score" in r]
            item.update(
                {
                    "mean_perceptual_score": mean(scores),
                    "worst_perceptual_score": max(scores) if scores else float("nan"),
                    "best_perceptual_score": min(scores) if scores else float("nan"),
                    "perceptual_score_win_frac": mean(
                        [float(r.get("perceptual_score", float("inf")) < 0.0) for r in subset]
                    ),
                    "triple_perceptual_win_frac": mean([float(r.get("triple_perceptual_win", 0)) for r in subset]),
                    "mean_delta_ms_ssim": mean([float(r["delta_ms_ssim"]) for r in subset if "delta_ms_ssim" in r]),
                    "mean_delta_lpips": mean([float(r["delta_lpips"]) for r in subset if "delta_lpips" in r]),
                    "mean_delta_dists": mean([float(r["delta_dists"]) for r in subset if "delta_dists" in r]),
                }
            )
        by_set.append(item)

    by_image: list[dict[str, Any]] = []
    for image in sorted({str(r["image"]) for r in rows}):
        subset = [r for r in rows if str(r["image"]) == image]
        all_rows = [r for r in subset if str(r["active_slices"]) == "all"]
        all_delta = float(all_rows[0]["delta_psnr"]) if all_rows else float("nan")
        best = max(subset, key=lambda r: float(r["delta_psnr"]))
        worst = min(subset, key=lambda r: float(r["delta_psnr"]))
        if has_perceptual:
            best_score = min(subset, key=lambda r: float(r.get("perceptual_score", float("inf"))))
            worst_score = max(subset, key=lambda r: float(r.get("perceptual_score", float("-inf"))))
            all_score = float(all_rows[0]["perceptual_score"]) if all_rows and "perceptual_score" in all_rows[0] else float("nan")
        else:
            best_score = None
            worst_score = None
            all_score = float("nan")
        singles = [r for r in subset if int(r["slice_count"]) == 1]
        leaves = [r for r in subset if int(r["slice_count"]) == 3]
        leave_marginals = {}
        score_leave_marginals = {}
        if math.isfinite(all_delta):
            for r in leaves:
                leave_marginals[str(r["omitted_slice"])] = all_delta - float(r["delta_psnr"])
        if has_perceptual and math.isfinite(all_score):
            for r in leaves:
                if "perceptual_score" in r:
                    score_leave_marginals[str(r["omitted_slice"])] = float(r["perceptual_score"]) - all_score
        item = {
            "image": image,
            "all_delta_psnr": all_delta,
            "best_psnr_slice_set": best["active_slices"],
            "best_delta_psnr": float(best["delta_psnr"]),
            "best_psnr_gain_over_all": float(best["delta_psnr"]) - all_delta if math.isfinite(all_delta) else float("nan"),
            "worst_psnr_slice_set": worst["active_slices"],
            "worst_delta_psnr": float(worst["delta_psnr"]),
            "positive_single_slices": ",".join(str(r["single_slice"]) for r in singles if float(r["delta_psnr"]) > 0),
            "negative_single_slices": ",".join(str(r["single_slice"]) for r in singles if float(r["delta_psnr"]) < 0),
            "leave_one_out_marginal_json": json.dumps(leave_marginals, sort_keys=True),
            "contract_ok_frac": mean([float(r["contract_ok"]) for r in subset]),
        }
        if has_perceptual and best_score is not None and worst_score is not None:
            item.update(
                {
                    "all_perceptual_score": all_score,
                    "best_score_slice_set": best_score["active_slices"],
                    "best_perceptual_score": float(best_score["perceptual_score"]),
                    "best_score_gain_over_all": all_score - float(best_score["perceptual_score"])
                    if math.isfinite(all_score)
                    else float("nan"),
                    "worst_score_slice_set": worst_score["active_slices"],
                    "worst_perceptual_score": float(worst_score["perceptual_score"]),
                    "score_leave_one_out_marginal_json": json.dumps(score_leave_marginals, sort_keys=True),
                    "perceptual_positive_single_slices": ",".join(
                        str(r["single_slice"]) for r in singles if float(r.get("perceptual_score", 0.0)) < 0
                    ),
                    "perceptual_negative_single_slices": ",".join(
                        str(r["single_slice"]) for r in singles if float(r.get("perceptual_score", 0.0)) > 0
                    ),
                }
            )
        by_image.append(item)
    return by_set, by_image


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], images: list[Path]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    by_set, by_image = summarize(rows)
    rows_csv = args.output_prefix.with_suffix(".rows.csv")
    by_set_csv = args.output_prefix.with_suffix(".by_set.csv")
    by_image_csv = args.output_prefix.with_suffix(".by_image.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(rows_csv, rows)
    write_csv(by_set_csv, by_set)
    write_csv(by_image_csv, by_image)
    payload = {
        "experiment": "E313 EF-LIC HCG slice-isolation sweep",
        "purpose": "Measure context-dependent slice/subset HCG effects for local controller-label design.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metric_protocol": {
            "psnr_role": "diagnostic codec-health metric; not the primary generative/perceptual claim",
            "compute_perceptual": bool(args.compute_perceptual),
            "lpips_net": args.lpips_net if args.compute_perceptual else None,
            "perceptual_score": "delta_DISTS + perceptual_score_lpips_weight * delta_LPIPS; lower is better"
            if args.compute_perceptual
            else None,
            "perceptual_score_lpips_weight": args.perceptual_score_lpips_weight,
            "metric_direction": {
                "delta_ms_ssim": "positive is better",
                "delta_lpips": "negative is better",
                "delta_dists": "negative is better",
            },
        },
        "images": [p.name for p in images],
        "by_set": by_set,
        "by_image": by_image,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with md_path.open("w") as fobj:
        fobj.write("# E313 EF-LIC HCG Slice-Isolation Sweep\n\n")
        fobj.write(
            "This is a small-cycle codec-loop diagnostic for local/slice controller-label design, "
            "not final paper evidence.\n\n"
        )
        fobj.write(
            "PSNR is reported as a diagnostic codec-health metric. When perceptual metrics are enabled, "
            "the paper-facing score is `delta_DISTS + weight * delta_LPIPS`; lower is better.\n\n"
        )
        fobj.write(f"- Dataset: `{args.image_dir}`\n")
        fobj.write(f"- Images: `{len(images)}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Mode: `{args.mode}`\n")
        fobj.write(f"- Direction source: `{args.direction_source}`\n")
        fobj.write(f"- Slice sets: `{', '.join(args.slice_sets)}`\n")
        fobj.write(f"- Perceptual metrics: `{bool(args.compute_perceptual)}`\n")
        if args.compute_perceptual:
            fobj.write(f"- LPIPS backbone: `{args.lpips_net}`\n")
            fobj.write(f"- Perceptual score LPIPS weight: `{args.perceptual_score_lpips_weight}`\n")
        fobj.write("\n")

        by_set_keys = [
            "active_slices",
            "images",
            "mean_perceptual_score",
            "worst_perceptual_score",
            "best_perceptual_score",
            "perceptual_score_win_frac",
            "triple_perceptual_win_frac",
            "mean_delta_ms_ssim",
            "mean_delta_lpips",
            "mean_delta_dists",
            "mean_delta_psnr",
            "worst_delta_psnr",
            "best_delta_psnr",
            "psnr_win_frac",
            "contract_ok_frac",
            "mean_delta_bpp",
            "mean_slice_enabled",
        ]
        if not args.compute_perceptual:
            by_set_keys = [
                "active_slices",
                "images",
                "mean_delta_psnr",
                "worst_delta_psnr",
                "best_delta_psnr",
                "psnr_win_frac",
                "contract_ok_frac",
                "mean_delta_bpp",
                "mean_slice_enabled",
            ]
        fobj.write("## By Slice Set\n\n")
        fobj.write("| " + " | ".join(by_set_keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(by_set_keys)) + "|\n")
        for item in by_set:
            fobj.write("| " + " | ".join(fmt(item.get(k, "")) for k in by_set_keys) + " |\n")

        by_image_keys = [
            "image",
            "all_perceptual_score",
            "best_score_slice_set",
            "best_perceptual_score",
            "best_score_gain_over_all",
            "worst_score_slice_set",
            "worst_perceptual_score",
            "perceptual_positive_single_slices",
            "perceptual_negative_single_slices",
            "score_leave_one_out_marginal_json",
            "all_delta_psnr",
            "best_psnr_slice_set",
            "best_delta_psnr",
            "best_psnr_gain_over_all",
            "worst_psnr_slice_set",
            "worst_delta_psnr",
            "positive_single_slices",
            "negative_single_slices",
            "leave_one_out_marginal_json",
            "contract_ok_frac",
        ]
        if not args.compute_perceptual:
            by_image_keys = [
                "image",
                "all_delta_psnr",
                "best_psnr_slice_set",
                "best_delta_psnr",
                "best_psnr_gain_over_all",
                "worst_psnr_slice_set",
                "worst_delta_psnr",
                "positive_single_slices",
                "negative_single_slices",
                "leave_one_out_marginal_json",
                "contract_ok_frac",
            ]
        fobj.write("\n## By Image\n\n")
        fobj.write("| " + " | ".join(by_image_keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(by_image_keys)) + "|\n")
        for item in by_image:
            fobj.write("| " + " | ".join(fmt(item.get(k, "")) for k in by_image_keys) + " |\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- If best subsets differ from `all`, HCG activation must be context-aware rather than globally enabled.\n")
        fobj.write("- Perceptual best subsets are the controller-label target; PSNR marginals are diagnostic only.\n")
        fobj.write("- Leave-one-out marginals should be used with residual/headroom features to build sequential labels.\n")
        fobj.write("- `contract_ok_frac` must remain 1.0 before rows are used as controller-label evidence.\n")
    print(f"wrote {rows_csv}, {by_set_csv}, {by_image_csv}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    rows, images = evaluate_rows(args)
    write_outputs(args, rows, images)


if __name__ == "__main__":
    main()
