#!/usr/bin/env python3
"""Export EF-LIC decoder-safe local context maps for the E239 HCG head.

This is the first E240 data bridge: it runs the original EF-LIC forward path,
captures the local maps available immediately after `_mean_scale(support_buf, i)`,
and attaches E239 image-level teacher labels. The exported tensors are intended
for the frozen-head smoke/training stage before moving to slice/spatial labels.
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
EFLIC_DIR = ROOT / "third_party" / "EF-LIC"
sys.path.insert(0, str(EFLIC_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from EF_LIC import G_CH_Y, model  # noqa: E402
from hcg_rvq.eflic_local_controller import build_local_context_maps  # noqa: E402
from test import load_checkpoint, load_image, list_images, replicate_pad  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--dataset", default="kodak24")
    p.add_argument("--manifest-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan.manifest.csv")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "analysis" / "e240_eflic_local_head_contexts")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--allow-missing-label", action="store_true")
    return p.parse_args()


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as fobj:
        for row in csv.DictReader(fobj):
            rows[(row["dataset"], row["image"])] = row
    return rows


def target_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_index": int(row["target_index"]),
        "target_family": row["target_family"],
        "oracle_family": row["oracle_family"],
        "oracle_policy": row["oracle_policy"],
        "confident_nonzero": int(row["confident_nonzero"]),
        "sample_weight": float(row["sample_weight"]),
        "oracle_score": float(row["oracle_score"]),
        "target_score": float(row["target_score"]),
        "improvement_vs_zero": float(row["improvement_vs_zero"]),
        "family_margin": float(row["family_margin"]),
    }


def _safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace(" ", "_")


@torch.inference_mode()
def export_one(
    *,
    net: model,
    image_path: Path,
    label_row: dict[str, Any],
    dataset: str,
    output_dir: Path,
    device: torch.device,
    force_ind: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    frame = load_image(image_path, device)
    _, _, h, w = frame.shape
    padded = replicate_pad(frame, h, w)

    y = net.g_a(padded)
    z_inds, z_hat = net.quantizes[force_ind][-1].encode_decode(net.h_a(y))
    support_buf = net._support_buf(net.h_s(z_hat))
    b, _, h2, w2 = support_buf.shape
    if b != 1:
        raise RuntimeError(f"expected batch size 1, got {b}")

    y_slice = y.new_empty(b, G_CH_Y, h2, w2)
    maps = []
    finite = True
    for slice_id in range(4):
        mean, scale = net._mean_scale(support_buf, slice_id)
        context = build_local_context_maps(
            support_buf,
            mean,
            scale,
            slice_id,
            group_channels=G_CH_Y,
        )
        finite = finite and bool(torch.isfinite(context).all().item())
        maps.append(context.squeeze(0).detach().to("cpu", dtype=dtype))

        if slice_id < 3:
            net._qt_select(y, slice_id, y_slice)
            y_norm = (y_slice - mean) / scale
            _, y_hat_norm_i = net.quantizes[force_ind][slice_id].encode_decode(y_norm)
            y_hat_i = y_hat_norm_i * scale + mean
            support_buf[:, (4 + slice_id) * G_CH_Y : (5 + slice_id) * G_CH_Y].copy_(y_hat_i)

    context_maps = torch.stack(maps, dim=0)
    target = target_from_row(label_row)
    tensor_path = output_dir / f"{dataset}__{_safe_stem(image_path)}.pt"
    torch.save(
        {
            "context_maps": context_maps,
            "target_index": target["target_index"],
            "target_family": target["target_family"],
            "sample_weight": target["sample_weight"],
            "dataset": dataset,
            "image": image_path.name,
            "height": int(h),
            "width": int(w),
            "latent_height": int(h2),
            "latent_width": int(w2),
            "force_ind": int(force_ind),
            "teacher": target,
            "note": "Context maps are captured from original EF-LIC decoded-prefix teacher forcing.",
        },
        tensor_path,
    )

    return {
        "dataset": dataset,
        "image": image_path.name,
        "tensor_path": str(tensor_path),
        "target_index": target["target_index"],
        "target_family": target["target_family"],
        "confident_nonzero": target["confident_nonzero"],
        "sample_weight": target["sample_weight"],
        "height": int(h),
        "width": int(w),
        "latent_height": int(h2),
        "latent_width": int(w2),
        "context_shape": "x".join(str(v) for v in context_maps.shape),
        "context_dtype": str(context_maps.dtype).replace("torch.", ""),
        "finite_context": int(finite),
        "nonfinite_context": int(not finite),
        "context_abs_mean": float(context_maps.float().abs().mean().item()),
        "context_rms": float(torch.sqrt((context_maps.float() ** 2).mean()).item()),
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], selected: list[Path], missing: list[str]) -> None:
    prefix = args.output_dir / f"manifest_{args.dataset}_n{len(rows)}"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "E240 EF-LIC local HCG head context export",
        "dataset": args.dataset,
        "image_dir": str(args.image_dir),
        "checkpoint": str(args.ckpt_path),
        "force_ind": args.force_ind,
        "device": args.device,
        "selected_images": [p.name for p in selected],
        "exported_images": len(rows),
        "missing_labels": missing,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    finite_frac = sum(row["finite_context"] for row in rows) / max(1, len(rows))
    target_counts: dict[str, int] = {}
    for row in rows:
        target_counts[row["target_family"]] = target_counts.get(row["target_family"], 0) + 1

    with md_path.open("w") as fobj:
        fobj.write("# E240 EF-LIC Local Head Context Export\n\n")
        fobj.write(
            "This artifact exports decoder-safe EF-LIC local context maps "
            "for the frozen local-head training smoke.\n\n"
        )
        fobj.write(f"- Dataset: `{args.dataset}`\n")
        fobj.write(f"- Image dir: `{args.image_dir}`\n")
        fobj.write(f"- Exported images: `{len(rows)}` / selected `{len(selected)}`\n")
        fobj.write(f"- Force index: `{args.force_ind}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Finite context fraction: `{finite_frac:.6f}`\n")
        fobj.write(f"- Target family counts: `{json.dumps(target_counts, sort_keys=True)}`\n")
        if missing:
            fobj.write(f"- Missing labels: `{len(missing)}`\n")
        fobj.write("\nExport contract:\n\n")
        fobj.write("- `context_maps` has shape `[4, 11, H, W]` for four EF-LIC slices.\n")
        fobj.write("- Labels are E239 image-level labels and may be broadcast for the first training smoke.\n")
        fobj.write("- Support prefixes use original EF-LIC decoded-prefix teacher forcing.\n")

    print(f"wrote {csv_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    if not args.ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest_csv)
    images = list_images(args.image_dir)
    selected = images[args.start_index : args.start_index + args.max_images]
    if not selected:
        raise SystemExit(f"no images selected from {args.image_dir}")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.force_ind)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image_path in selected:
        label = manifest.get((args.dataset, image_path.name))
        if label is None:
            missing.append(image_path.name)
            if args.allow_missing_label:
                continue
            raise SystemExit(f"missing E239 label for ({args.dataset}, {image_path.name})")
        row = export_one(
            net=net,
            image_path=image_path,
            label_row=label,
            dataset=args.dataset,
            output_dir=args.output_dir,
            device=device,
            force_ind=args.force_ind,
            dtype=dtype,
        )
        rows.append(row)
        if row["nonfinite_context"] or not math.isfinite(row["context_rms"]):
            raise SystemExit(f"nonfinite context for {image_path.name}")

    write_outputs(args, rows, selected, missing)


if __name__ == "__main__":
    main()
