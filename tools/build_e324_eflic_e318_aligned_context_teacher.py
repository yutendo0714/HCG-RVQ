#!/usr/bin/env python3
"""Build E318-aligned EF-LIC controller tensors from E242 context maps.

E323 shows that E242's old spatial teacher labels are stale relative to the
latest E317/E318 fallback-aware powerset oracle. This bridge keeps the valuable
decoder-available context maps from E242, but replaces the target maps with
slice-dense E318 oracle-active labels. It is a controller-training bridge, not
final local supervision: E318 labels are image/slice-level, not per-pixel.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import FAMILY_NAMES, FAMILY_TO_INDEX  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--e242-manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e242_eflic_spatial_teacher_contexts_kodak24/manifest_kodak24_n24.csv",
    )
    p.add_argument(
        "--slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/analysis/e324_eflic_e318_aligned_context_teacher_kodak24",
    )
    p.add_argument("--dataset", default="kodak24")
    p.add_argument("--active-family", choices=FAMILY_NAMES[1:], default="constant")
    p.add_argument("--active-alpha", type=float, default=0.02)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_oracle_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for row in rows:
        out[(row["image"], int(row["slice"]))] = int(float(row["oracle_active"]))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    active_family_idx = FAMILY_TO_INDEX[args.active_family]
    e242_rows = read_csv(args.e242_manifest)
    oracle = build_oracle_lookup(read_csv(args.slice_labels))

    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
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
        active_slices: list[int] = []
        for slice_id in range(slices):
            key = (record["image"], slice_id)
            if key not in oracle:
                missing.append(f"{record['image']}:{slice_id}")
                continue
            if oracle[key] == 1:
                target_map[slice_id].fill_(active_family_idx)
                alpha_target[slice_id].fill_(float(args.active_alpha))
                active_slices.append(slice_id)

        out_path = args.output_dir / f"{args.dataset}__{Path(record['image']).stem}.pt"
        torch.save(
            {
                "context_maps": context_maps,
                "alpha_target": alpha_target,
                "target_map": target_map,
                "target_index": int(active_family_idx),
                "target_family": args.active_family,
                "teacher_policy": "e318_powerset_oracle_slice_dense",
                "sample_weight": 1.0,
                "dataset": args.dataset,
                "image": record["image"],
                "height": int(record.get("height", obj.get("height", 0)) or 0),
                "width": int(record.get("width", obj.get("width", 0)) or 0),
                "latent_height": h,
                "latent_width": w,
                "force_ind": int(obj.get("force_ind", 0)),
                "active_slices": active_slices,
                "source_tensor_path": str(tensor_path),
                "source_e242_teacher_policy": record.get("teacher_policy", ""),
                "source_e242_target_family": record.get("target_family", ""),
                "note": "E324 bridge: E242 decoder-safe contexts + E318 fallback-aware slice-dense oracle labels.",
            },
            out_path,
        )
        active_frac = float((target_map > 0).float().mean().item())
        alpha_mean = float(alpha_target.float().mean().item())
        pixel_counts = {name: int((target_map == idx).sum().item()) for idx, name in enumerate(FAMILY_NAMES)}
        pixel_counts = {k: v for k, v in pixel_counts.items() if v}
        manifest_rows.append(
            {
                "dataset": args.dataset,
                "image": record["image"],
                "tensor_path": str(out_path),
                "target_family": args.active_family if active_slices else "zero",
                "target_index": active_family_idx if active_slices else 0,
                "teacher_policy": "e318_powerset_oracle_slice_dense",
                "active_slices": ",".join(str(v) for v in active_slices) if active_slices else "none",
                "active_slice_count": len(active_slices),
                "active_frac": active_frac,
                "alpha_mean": alpha_mean,
                "alpha_max": float(alpha_target.float().max().item()),
                "context_shape": "x".join(str(v) for v in context_maps.shape),
                "target_map_shape": "x".join(str(v) for v in target_map.shape),
                "alpha_shape": "x".join(str(v) for v in alpha_target.shape),
                "context_dtype": str(context_maps.dtype).replace("torch.", ""),
                "height": int(record.get("height", obj.get("height", 0)) or 0),
                "width": int(record.get("width", obj.get("width", 0)) or 0),
                "latent_height": h,
                "latent_width": w,
                "pixel_family_counts": json.dumps(pixel_counts, sort_keys=True),
                "source_tensor_path": str(tensor_path),
                "source_e242_teacher_policy": record.get("teacher_policy", ""),
                "source_e242_target_family": record.get("target_family", ""),
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
        "experiment": "E324 EF-LIC E318-aligned context teacher",
        "purpose": "Reuse E242 decoder-safe context maps while replacing stale E242 labels with E318 fallback-aware slice-dense oracle labels.",
        "e242_manifest": str(args.e242_manifest),
        "slice_labels": str(args.slice_labels),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "active_family": args.active_family,
        "active_alpha": args.active_alpha,
        "images": len(manifest_rows),
        "missing_oracle_keys": missing,
        "mean_active_frac": sum(float(row["active_frac"]) for row in manifest_rows) / max(1, len(manifest_rows)),
        "mean_active_slice_count": sum(float(row["active_slice_count"]) for row in manifest_rows) / max(1, len(manifest_rows)),
        "active_slice_set_counts": active_counts,
        "rows": manifest_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E324 EF-LIC E318-Aligned Context Teacher\n\n")
        fobj.write("This artifact keeps E242 decoder-safe context tensors but replaces stale E242 target maps with E318 fallback-aware slice-dense oracle labels.\n\n")
        fobj.write(f"- Dataset: `{args.dataset}`\n")
        fobj.write(f"- Images: `{len(manifest_rows)}`\n")
        fobj.write(f"- Active family: `{args.active_family}`\n")
        fobj.write(f"- Active alpha: `{args.active_alpha}`\n")
        fobj.write(f"- Mean active fraction: `{payload['mean_active_frac']:.6f}`\n")
        fobj.write(f"- Mean active slice count: `{payload['mean_active_slice_count']:.6f}`\n")
        fobj.write(f"- Missing oracle keys: `{len(missing)}`\n\n")
        fobj.write("Interpretation:\n\n")
        fobj.write("- E324 is not final spatial supervision because E318 labels are image/slice-level and are broadcast over spatial positions.\n")
        fobj.write("- It is the correct next bridge after E323: trainability can be tested without reusing stale E242 activation labels.\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
