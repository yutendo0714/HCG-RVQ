#!/usr/bin/env python3
"""Build EF-LIC controller tensors with codec-gain risk targets.

E324 broadcasts E318 powerset oracle-active slice labels over the saved E242
context maps.  That is useful for activation trainability, but its risk target is
still inherited from active magnitude.  E346 keeps the same decoder-safe context
maps and adds a `risk_target` derived from E318 contextual PSNR margins:
positive margin -> negative risk target, negative margin -> positive risk target.

The artifact is still a controller teacher bridge, not final RD evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import FAMILY_NAMES, FAMILY_TO_INDEX  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e242-manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e242_eflic_spatial_teacher_contexts_kodak24/manifest_kodak24_n24.csv",
    )
    parser.add_argument(
        "--slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/analysis/e346_eflic_codec_gain_context_teacher_kodak24",
    )
    parser.add_argument("--dataset", default="kodak24")
    parser.add_argument("--active-family", choices=FAMILY_NAMES[1:], default="constant")
    parser.add_argument("--active-alpha", type=float, default=0.02)
    parser.add_argument("--activation-source", choices=["oracle_active", "contextual_positive"], default="contextual_positive")
    parser.add_argument("--active-margin", type=float, default=0.0)
    parser.add_argument("--risk-scale", type=float, default=5.0)
    parser.add_argument("--risk-clip", type=float, default=0.10)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_label_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        margin = safe_float(row.get("contextual_margin_psnr"))
        out[(row["image"], int(row["slice"]))] = {
            "oracle_active": int(float(row.get("oracle_active", 0))),
            "contextual_positive": int(float(row.get("contextual_positive", 0))),
            "contextual_margin_psnr": margin,
            "single_delta_psnr": safe_float(row.get("single_delta_psnr")),
            "single_positive": int(float(row.get("single_positive", 0))),
        }
    return out


def clipped(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    active_family_idx = FAMILY_TO_INDEX[args.active_family]
    e242_rows = read_csv(args.e242_manifest)
    labels = build_label_lookup(read_csv(args.slice_labels))

    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    risk_values: list[float] = []
    margins: list[float] = []
    for record in e242_rows:
        tensor_path = Path(record["tensor_path"])
        if not tensor_path.is_absolute():
            tensor_path = ROOT / tensor_path
        obj = torch.load(tensor_path, map_location="cpu")
        context_maps = obj["context_maps"].to(dtype=dtype)
        if context_maps.ndim != 4 or context_maps.shape[0] != 4:
            raise RuntimeError(f"unexpected context_maps shape {tuple(context_maps.shape)} in {tensor_path}")
        slices, _, h, w = context_maps.shape
        target_map = torch.zeros((slices, h, w), dtype=torch.uint8)
        alpha_target = torch.zeros((slices, 1, h, w), dtype=dtype)
        risk_target = torch.zeros((slices, 1, h, w), dtype=dtype)
        active_slices: list[int] = []
        slice_rows: list[dict[str, Any]] = []
        for slice_id in range(slices):
            key = (record["image"], slice_id)
            if key not in labels:
                missing.append(f"{record['image']}:{slice_id}")
                continue
            label = labels[key]
            margin = float(label["contextual_margin_psnr"])
            if args.activation_source == "contextual_positive":
                active = int(label["contextual_positive"]) == 1 and margin > float(args.active_margin)
            else:
                active = int(label["oracle_active"]) == 1
            risk_value = clipped(-margin * float(args.risk_scale), float(args.risk_clip))
            risk_target[slice_id].fill_(risk_value)
            margins.append(margin)
            risk_values.append(risk_value)
            if active:
                target_map[slice_id].fill_(active_family_idx)
                alpha_target[slice_id].fill_(float(args.active_alpha))
                active_slices.append(slice_id)
            slice_rows.append({
                "slice": slice_id,
                "active": int(active),
                "contextual_margin_psnr": margin,
                "risk_target": risk_value,
                "oracle_active": label["oracle_active"],
                "contextual_positive": label["contextual_positive"],
            })

        out_path = args.output_dir / f"{args.dataset}__{Path(record['image']).stem}.pt"
        torch.save(
            {
                "context_maps": context_maps,
                "alpha_target": alpha_target,
                "target_map": target_map,
                "risk_target": risk_target,
                "target_index": int(active_family_idx),
                "target_family": args.active_family,
                "teacher_policy": "e346_contextual_margin_codec_gain_risk",
                "activation_source": args.activation_source,
                "risk_scale": float(args.risk_scale),
                "risk_clip": float(args.risk_clip),
                "active_margin": float(args.active_margin),
                "sample_weight": 1.0,
                "dataset": args.dataset,
                "image": record["image"],
                "height": int(record.get("height", obj.get("height", 0)) or 0),
                "width": int(record.get("width", obj.get("width", 0)) or 0),
                "latent_height": h,
                "latent_width": w,
                "force_ind": int(obj.get("force_ind", 0)),
                "active_slices": active_slices,
                "slice_labels": slice_rows,
                "source_tensor_path": str(tensor_path),
                "note": "E346 bridge: E242 decoder-safe contexts + E318 contextual-margin risk targets.",
            },
            out_path,
        )
        active_frac = float((target_map > 0).float().mean().item())
        risk_float = risk_target.float()
        alpha_float = alpha_target.float()
        manifest_rows.append(
            {
                "dataset": args.dataset,
                "image": record["image"],
                "tensor_path": str(out_path),
                "target_family": args.active_family if active_slices else "zero",
                "target_index": active_family_idx if active_slices else 0,
                "teacher_policy": "e346_contextual_margin_codec_gain_risk",
                "activation_source": args.activation_source,
                "active_slices": ",".join(str(v) for v in active_slices) if active_slices else "none",
                "active_slice_count": len(active_slices),
                "active_frac": active_frac,
                "alpha_mean": float(alpha_float.mean().item()),
                "alpha_max": float(alpha_float.max().item()),
                "risk_mean": float(risk_float.mean().item()),
                "risk_min": float(risk_float.min().item()),
                "risk_max": float(risk_float.max().item()),
                "risk_scale": float(args.risk_scale),
                "risk_clip": float(args.risk_clip),
                "active_margin": float(args.active_margin),
                "context_shape": "x".join(str(v) for v in context_maps.shape),
                "target_map_shape": "x".join(str(v) for v in target_map.shape),
                "risk_shape": "x".join(str(v) for v in risk_target.shape),
                "height": int(record.get("height", obj.get("height", 0)) or 0),
                "width": int(record.get("width", obj.get("width", 0)) or 0),
                "latent_height": h,
                "latent_width": w,
                "source_tensor_path": str(tensor_path),
            }
        )

    manifest_path = args.output_dir / f"manifest_{args.dataset}_n{len(manifest_rows)}.csv"
    json_path = args.output_dir / f"manifest_{args.dataset}_n{len(manifest_rows)}.json"
    md_path = args.output_dir / f"manifest_{args.dataset}_n{len(manifest_rows)}.md"
    write_csv(manifest_path, manifest_rows)
    active_counts: dict[str, int] = {}
    for row in manifest_rows:
        active_counts[row["active_slices"]] = active_counts.get(row["active_slices"], 0) + 1
    payload = {
        "experiment": "E346 EF-LIC codec-gain context teacher",
        "purpose": "Add E318 contextual-margin risk targets to decoder-safe EF-LIC controller tensors.",
        "e242_manifest": str(args.e242_manifest),
        "slice_labels": str(args.slice_labels),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "active_family": args.active_family,
        "active_alpha": float(args.active_alpha),
        "activation_source": args.activation_source,
        "active_margin": float(args.active_margin),
        "risk_scale": float(args.risk_scale),
        "risk_clip": float(args.risk_clip),
        "images": len(manifest_rows),
        "missing_oracle_keys": missing,
        "mean_active_frac": sum(float(row["active_frac"]) for row in manifest_rows) / max(1, len(manifest_rows)),
        "mean_active_slice_count": sum(float(row["active_slice_count"]) for row in manifest_rows) / max(1, len(manifest_rows)),
        "mean_contextual_margin_psnr": sum(margins) / max(1, len(margins)),
        "mean_risk_target": sum(risk_values) / max(1, len(risk_values)),
        "active_slice_set_counts": active_counts,
        "rows": manifest_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E346 EF-LIC Codec-Gain Context Teacher\n\n")
        fobj.write("This artifact keeps E242 decoder-safe context tensors, uses E318 contextual-positive slices for activation, and stores contextual-margin risk targets.\n\n")
        fobj.write(f"- Dataset: `{args.dataset}`\n")
        fobj.write(f"- Images: `{len(manifest_rows)}`\n")
        fobj.write(f"- Active family: `{args.active_family}`\n")
        fobj.write(f"- Active alpha: `{args.active_alpha}`\n")
        fobj.write(f"- Activation source: `{args.activation_source}`\n")
        fobj.write(f"- Risk target: `-contextual_margin_psnr * {args.risk_scale}`, clipped to `+/-{args.risk_clip}`\n")
        fobj.write(f"- Mean active fraction: `{payload['mean_active_frac']:.6f}`\n")
        fobj.write(f"- Mean active slice count: `{payload['mean_active_slice_count']:.6f}`\n")
        fobj.write(f"- Mean contextual margin PSNR: `{payload['mean_contextual_margin_psnr']:.6f}`\n")
        fobj.write(f"- Mean risk target: `{payload['mean_risk_target']:.6f}`\n")
        fobj.write(f"- Missing label keys: `{len(missing)}`\n\n")
        fobj.write("Interpretation:\n\n")
        fobj.write("- E346 is designed to fix the E343/E345 finding that the previous risk score was an active-like signal, not a codec-gain signal.\n")
        fobj.write("- Positive codec margin receives negative risk, so `max_risk=0` can become a meaningful decoder-safe fallback threshold.\n")
        fobj.write("- This is still a teacher bridge; codec-loop value must be checked by reloading the trained controller in E295.\n")
    print(f"wrote {manifest_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
