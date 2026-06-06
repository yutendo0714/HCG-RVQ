#!/usr/bin/env python3
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
TOOLS = ROOT / "tools"
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(EFLIC_DIR))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402
import analyze_e193_eflic_reliability_head as e193  # noqa: E402
from run_e160_eflic_projected_hcg_smoke import active_compress_forward, active_decompress, index_stats, mean_psnr  # noqa: E402
from run_e186_eflic_global_predecision_selector_probe import (  # noqa: E402
    dists_value,
    predecision_stats_from_padded,
    predecision_stats_from_z_inds,
)
from EF_LIC import N_E, model  # noqa: E402
from test import load_checkpoint, load_image, list_images, mse01, pack_inds, psnr_from_mse, replicate_pad  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fit-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005.csv"),
    )
    p.add_argument(
        "--fit-manifest-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005_feature_manifest.csv"),
    )
    p.add_argument("--eval-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak_first4")
    p.add_argument("--start-index", type=int, default=0, help="Start offset after deterministic eval path sorting.")
    p.add_argument("--max-images", type=int, default=None, help="Maximum eval images after start-index.")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e194_eflic_reliability_head_selector_probe",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--direction-source", default="mean", choices=["mean", "logscale", "fixed"])
    p.add_argument("--feature-set", default="global_predecision_context")
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--positive-penalty", type=float, default=20.0)
    p.add_argument("--l2", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--override-threshold", type=float, default=None, help="Optional externally calibrated selector threshold.")
    return p.parse_args()


def load_fit_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    rows = [r for r in e184.read_rows(args.fit_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in rows if int(float(r["force_ind"])) == args.force_ind]
    if not rows:
        raise SystemExit(f"no finite fit rows for force{args.force_ind}")
    manifest = e184.read_manifest(args.fit_manifest_csv)
    feature_sets = e185.feature_sets(manifest)
    if args.feature_set not in feature_sets:
        raise SystemExit(f"unknown feature set: {args.feature_set}")
    features = e184.valid_features(rows, feature_sets[args.feature_set][0])
    if not features:
        raise SystemExit(f"no valid fit features in {args.feature_set}")
    return rows, features


def head_probability(model_state: dict[str, Any], stats: dict[str, float]) -> float:
    features = model_state["features"]
    x_raw = np.asarray([[float(stats[f]) for f in features]], dtype=np.float64)
    x = (x_raw - model_state["mean"]) / model_state["std"]
    prob = e193.sigmoid(x @ model_state["weights"] + float(model_state["bias"]))
    return float(prob.reshape(-1)[0])


def decide_head(model_state: dict[str, Any], stats: dict[str, float]) -> tuple[float, bool]:
    prob = head_probability(model_state, stats)
    return prob, prob >= float(model_state["threshold"])


def add_z_index_stats(stats: dict[str, float], z_inds: torch.Tensor) -> dict[str, float]:
    out = dict(stats)
    out.update(index_stats(z_inds, int(N_E[-1]), "z_index"))
    return out


def summarize(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    n = max(1, len(rows))
    return {
        "branch": branch,
        "images": len(rows),
        "branch_share": float(np.mean([float(r["use_active"]) for r in rows])) if branch == "selected" else float(branch == "active"),
        "bpp": float(np.mean([r[f"{branch}_bpp"] for r in rows])),
        "psnr": mean_psnr(rows, f"{branch}_psnr"),
        "lpips": float(np.mean([r[f"{branch}_lpips"] for r in rows])),
        "dists": float(np.mean([r[f"{branch}_dists"] for r in rows])),
        "delta_psnr_vs_base": float(np.mean([r[f"{branch}_psnr"] - r["base_psnr"] for r in rows])),
        "delta_lpips_vs_base": float(np.mean([r[f"{branch}_lpips"] - r["base_lpips"] for r in rows])),
        "delta_dists_vs_base": float(np.mean([r[f"{branch}_dists"] - r["base_dists"] for r in rows])),
        "dists_wins_vs_base": int(sum(r[f"{branch}_dists"] < r["base_dists"] for r in rows)),
        "lpips_wins_vs_base": int(sum(r[f"{branch}_lpips"] < r["base_lpips"] for r in rows)),
        "psnr_wins_vs_base": int(sum(r[f"{branch}_psnr"] > r["base_psnr"] for r in rows)),
    }


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    model_state: dict[str, Any],
    fit_rows: list[dict[str, Any]],
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    coeffs = e193.top_coefficients(model_state, 16)
    payload = {
        "experiment": "E194 EF-LIC reliability-head direct selector probe",
        "fit_csv": str(args.fit_csv),
        "fit_rows": len(fit_rows),
        "eval_dir": str(args.eval_dir),
        "eval_start_index": args.start_index,
        "eval_max_images": args.max_images,
        "evaluated_images": [r["image"] for r in rows],
        "checkpoint": str(args.ckpt_path),
        "device": str(args.device),
        "force_ind": args.force_ind,
        "alpha": args.alpha,
        "direction_source": args.direction_source,
        "feature_set": args.feature_set,
        "head_threshold": float(model_state["threshold"]),
        "override_threshold": args.override_threshold,
        "head_bias": float(model_state["bias"]),
        "summary": summary,
        "coefficients": coeffs,
        "rows": rows,
        "interpretation": "Direct deployability probe for a fitted logistic selector. It is paper-facing only when fit_csv and eval_dir are independent.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E194 EF-LIC Reliability-Head Direct Selector Probe",
        "",
        f"Fit CSV: `{args.fit_csv}`",
        f"Fit rows: `{len(fit_rows)}`",
        f"Eval dir: `{args.eval_dir}`",
        f"Eval start index: `{args.start_index}`",
        f"Eval max images: `{args.max_images}`",
        f"Device: `{args.device}`",
        f"Feature set: `{args.feature_set}`",
        f"Head threshold: `{float(model_state['threshold']):.6f}`",
        "",
        "This runs a fitted logistic reliability head inside the direct EF-LIC active/fallback evaluation path. It is a deployability smoke unless the fit CSV and eval directory are independent.",
        "",
        "| branch | images | branch share | bpp | PSNR | LPIPS | DISTS | dPSNR | dLPIPS | dDISTS | DISTS wins | LPIPS wins | PSNR wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['branch']} | {s['images']} | {s['branch_share']:.3f} | {s['bpp']:.6f} | "
            f"{s['psnr']:.4f} | {s['lpips']:.5f} | {s['dists']:.5f} | "
            f"{s['delta_psnr_vs_base']:+.4f} | {s['delta_lpips_vs_base']:+.6f} | {s['delta_dists_vs_base']:+.6f} | "
            f"{s['dists_wins_vs_base']}/{s['images']} | {s['lpips_wins_vs_base']}/{s['images']} | {s['psnr_wins_vs_base']}/{s['images']} |"
        )
    lines.extend(
        [
            "",
            "Checks:",
            "",
            f"- Encoder/decoder selector decision matches: `{sum(r['decoder_decision_match'] for r in rows)}/{len(rows)}`",
            f"- Max selector probability abs diff: `{max(r['selector_probability_abs_diff'] for r in rows):.6e}`",
            f"- Max active decode diff: `{max(r['active_max_decode_diff'] for r in rows):.6e}`",
            f"- Nonfinite rows: `{sum(r['nonfinite'] for r in rows)}`",
            "",
            "Top standardized coefficients:",
            "",
            "| feature | coefficient |",
            "|---|---:|",
        ]
    )
    for row in coeffs:
        lines.append(f"| {row['feature']} | {row['coefficient']:+.6f} |")
    lines.extend(
        [
            "",
            "Next:",
            "",
            "- Use this same command with a non-Kodak fit CSV and held-out eval directory.",
            "- Promote the head only if held-out DISTS and LPIPS improve with exact encoder/decoder matching.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    if abs(1.0 - 2.0 * args.alpha) < 1e-6:
        raise ValueError("alpha=0.5 is singular")

    fit_rows, features = load_fit_rows(args)
    model_state = e193.fit_head(fit_rows, features, args)
    if args.override_threshold is not None:
        model_state["threshold"] = float(args.override_threshold)

    device = torch.device(args.device)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive when set")
    all_images = list_images(args.eval_dir)
    end_index = None if args.max_images is None else args.start_index + args.max_images
    images = all_images[args.start_index:end_index]
    if not images:
        raise SystemExit(f"no images selected from {args.eval_dir} start={args.start_index} max={args.max_images}")

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    rows: list[dict[str, Any]] = []
    for path in images:
        frame = load_image(path, device)
        _, _, h, w = frame.shape
        padded = replicate_pad(frame, h, w)

        enc_z_inds, enc_pre_stats = predecision_stats_from_padded(net, padded.clone(), args.force_ind)
        enc_pre_stats = add_z_index_stats(enc_pre_stats, enc_z_inds)
        enc_prob, enc_decision = decide_head(model_state, enc_pre_stats)
        dec_pre_stats = predecision_stats_from_z_inds(net, enc_z_inds, args.force_ind)
        dec_pre_stats = add_z_index_stats(dec_pre_stats, enc_z_inds)
        dec_prob, dec_decision = decide_head(model_state, dec_pre_stats)

        base_inds = net.compress(padded.clone(), force_ind=args.force_ind)
        base_payload, _, _ = pack_inds(net, base_inds)
        base_x_hat = net.decompress(base_inds, force_ind=args.force_ind)[:, :, :h, :w]

        active_inds, active_x_hat_forward, active_stats = active_compress_forward(
            net,
            padded.clone(),
            force_ind=args.force_ind,
            alpha=args.alpha,
            direction_source=args.direction_source,
        )
        active_payload, _, _ = pack_inds(net, active_inds)
        active_x_hat_dec = active_decompress(
            net,
            active_inds,
            force_ind=args.force_ind,
            alpha=args.alpha,
            direction_source=args.direction_source,
        )[:, :, :h, :w]
        active_x_hat_fwd = active_x_hat_forward[:, :, :h, :w]
        decode_diff = (active_x_hat_fwd - active_x_hat_dec).abs()

        selected_x_hat = active_x_hat_dec if enc_decision else base_x_hat
        selected_payload = active_payload if enc_decision else base_payload

        row: dict[str, Any] = {
            "image": path.name,
            "selector_probability_encoder": enc_prob,
            "selector_probability_decoder": dec_prob,
            "selector_probability_abs_diff": abs(enc_prob - dec_prob),
            "selector_threshold": float(model_state["threshold"]),
            "use_active": int(enc_decision),
            "decoder_decision_match": int(enc_decision == dec_decision),
            "base_bpp": len(base_payload) * 8.0 / float(h * w),
            "active_bpp": len(active_payload) * 8.0 / float(h * w),
            "selected_bpp": len(selected_payload) * 8.0 / float(h * w),
            "base_psnr": psnr_from_mse(mse01(base_x_hat, frame)),
            "active_psnr": psnr_from_mse(mse01(active_x_hat_dec, frame)),
            "selected_psnr": psnr_from_mse(mse01(selected_x_hat, frame)),
            "base_lpips": float(lpips_fn(base_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "active_lpips": float(lpips_fn(active_x_hat_dec.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "selected_lpips": float(lpips_fn(selected_x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()),
            "base_dists": dists_value(dists_fn, base_x_hat, frame),
            "active_dists": dists_value(dists_fn, active_x_hat_dec, frame),
            "selected_dists": dists_value(dists_fn, selected_x_hat, frame),
            "active_max_decode_diff": float(decode_diff.max().item()),
            "active_mean_decode_diff": float(decode_diff.mean().item()),
            "nonfinite": int(
                (not torch.isfinite(base_x_hat).all().item())
                or (not torch.isfinite(active_x_hat_dec).all().item())
                or (not torch.isfinite(selected_x_hat).all().item())
                or any(isinstance(v, float) and not math.isfinite(v) for v in active_stats.values())
            ),
        }
        for feature in model_state["features"]:
            row[feature] = float(enc_pre_stats[feature])
        for name in ("dists", "lpips", "psnr"):
            row[f"active_delta_{name}"] = row[f"active_{name}"] - row[f"base_{name}"]
            row[f"selected_delta_{name}"] = row[f"selected_{name}"] - row[f"base_{name}"]
        rows.append(row)
        print(
            f"{path.name} p={enc_prob:.6f} use_active={row['use_active']} "
            f"dDISTS active={row['active_delta_dists']:+.6f} selected={row['selected_delta_dists']:+.6f} "
            f"match={row['decoder_decision_match']} nonfinite={row['nonfinite']}"
        )

    summary = [summarize(rows, "base"), summarize(rows, "active"), summarize(rows, "selected")]
    write_outputs(args, rows, summary, model_state, fit_rows)


if __name__ == "__main__":
    main()
