#!/usr/bin/env python3
"""Export EF-LIC local contexts with E236 spatial teacher maps.

E241 showed that image-level E239 labels are too coarse for a local HCG
controller. This E242 bridge re-runs the original EF-LIC forward path, captures
the same decoder-safe local context maps as E240, and additionally stores exact
E236 policy alpha maps chosen by the confidence-gated E239 teacher. Positions
with nonzero alpha receive the target HCG family; inactive positions remain the
zero/fallback class.

This is still supervision infrastructure, not a performance row. The exported
`target_map` is the first map-level teacher for the frozen local head.
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
from hcg_rvq.eflic_local_controller import FAMILY_NAMES, FAMILY_TO_INDEX, build_local_context_maps  # noqa: E402
from run_e236_eflic_local_controller_map_smoke import build_controller_alpha_map  # noqa: E402
from test import load_checkpoint, load_image, list_images, replicate_pad  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--dataset", default="kodak24")
    p.add_argument("--manifest-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan.manifest.csv")
    p.add_argument("--ckpt-path", type=Path, default=EFLIC_DIR / "ckpt" / "checkpoint.pth.tar")
    p.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-ind", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--alpha-threshold", type=float, default=0.0)
    p.add_argument("--allow-missing-label", action="store_true")
    return p.parse_args()


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as fobj:
        for row in csv.DictReader(fobj):
            rows[(row["dataset"], row["image"])] = row
    return rows


def _safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace(" ", "_")


def teacher_policy(row: dict[str, Any]) -> tuple[str, str, int]:
    target_index = int(row["target_index"])
    if target_index == FAMILY_TO_INDEX["zero"]:
        return "zero", "zero", target_index
    policy = row.get("oracle_policy") or row.get("target_policy")
    if not policy:
        raise ValueError(f"nonzero target for {row.get('image')} has no oracle_policy")
    family = row.get("target_family") or row.get("oracle_family")
    if family not in FAMILY_TO_INDEX:
        raise ValueError(f"unknown target family {family!r}")
    return policy, family, FAMILY_TO_INDEX[family]


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
    alpha_threshold: float,
) -> dict[str, Any]:
    policy, family, target_index = teacher_policy(label_row)
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
    context_items: list[torch.Tensor] = []
    alpha_items: list[torch.Tensor] = []
    target_items: list[torch.Tensor] = []
    finite = True
    finite_alpha = True

    for slice_id in range(4):
        mean, scale = net._mean_scale(support_buf, slice_id)
        context = build_local_context_maps(
            support_buf,
            mean,
            scale,
            slice_id,
            group_channels=G_CH_Y,
        )
        alpha = build_controller_alpha_map(
            policy=policy,
            support_buf=support_buf,
            mean=mean,
            scale=scale,
            slice_id=slice_id,
        ).clamp_min(0.0)
        active = alpha > float(alpha_threshold)
        target_map = torch.zeros((alpha.shape[0], alpha.shape[2], alpha.shape[3]), dtype=torch.long, device=alpha.device)
        if target_index != FAMILY_TO_INDEX["zero"]:
            target_map[active.squeeze(1)] = int(target_index)

        finite = finite and bool(torch.isfinite(context).all().item())
        finite_alpha = finite_alpha and bool(torch.isfinite(alpha).all().item())
        context_items.append(context.squeeze(0).detach().to("cpu", dtype=dtype))
        alpha_items.append(alpha.squeeze(0).detach().to("cpu", dtype=dtype))
        target_items.append(target_map.squeeze(0).detach().to("cpu", dtype=torch.uint8))

        if slice_id < 3:
            net._qt_select(y, slice_id, y_slice)
            y_norm = (y_slice - mean) / scale
            _, y_hat_norm_i = net.quantizes[force_ind][slice_id].encode_decode(y_norm)
            y_hat_i = y_hat_norm_i * scale + mean
            support_buf[:, (4 + slice_id) * G_CH_Y : (5 + slice_id) * G_CH_Y].copy_(y_hat_i)

    context_maps = torch.stack(context_items, dim=0)
    alpha_target = torch.stack(alpha_items, dim=0)
    target_map = torch.stack(target_items, dim=0)
    active_frac = float((target_map != FAMILY_TO_INDEX["zero"]).float().mean().item())
    alpha_mean = float(alpha_target.float().mean().item())
    alpha_max = float(alpha_target.float().max().item())

    tensor_path = output_dir / f"{dataset}__{_safe_stem(image_path)}.pt"
    torch.save(
        {
            "context_maps": context_maps,
            "alpha_target": alpha_target,
            "target_map": target_map,
            "target_index": int(target_index),
            "target_family": family,
            "teacher_policy": policy,
            "sample_weight": float(label_row.get("sample_weight", 1.0)),
            "dataset": dataset,
            "image": image_path.name,
            "height": int(h),
            "width": int(w),
            "latent_height": int(h2),
            "latent_width": int(w2),
            "force_ind": int(force_ind),
            "teacher": dict(label_row),
            "note": "E242 map-level teacher: E239 confidence-gated family + exact E236 policy alpha map.",
        },
        tensor_path,
    )

    pixel_counts = {name: int((target_map == idx).sum().item()) for idx, name in enumerate(FAMILY_NAMES)}
    pixel_counts = {k: v for k, v in pixel_counts.items() if v}
    return {
        "dataset": dataset,
        "image": image_path.name,
        "tensor_path": str(tensor_path),
        "target_index": int(target_index),
        "target_family": family,
        "teacher_policy": policy,
        "confident_nonzero": int(label_row.get("confident_nonzero", int(target_index != FAMILY_TO_INDEX["zero"]))),
        "sample_weight": float(label_row.get("sample_weight", 1.0)),
        "height": int(h),
        "width": int(w),
        "latent_height": int(h2),
        "latent_width": int(w2),
        "context_shape": "x".join(str(v) for v in context_maps.shape),
        "alpha_shape": "x".join(str(v) for v in alpha_target.shape),
        "target_map_shape": "x".join(str(v) for v in target_map.shape),
        "context_dtype": str(context_maps.dtype).replace("torch.", ""),
        "finite_context": int(finite),
        "finite_alpha": int(finite_alpha),
        "nonfinite_context": int(not finite),
        "nonfinite_alpha": int(not finite_alpha),
        "context_abs_mean": float(context_maps.float().abs().mean().item()),
        "context_rms": float(torch.sqrt((context_maps.float() ** 2).mean()).item()),
        "alpha_mean": alpha_mean,
        "alpha_max": alpha_max,
        "active_frac": active_frac,
        "pixel_family_counts": json.dumps(pixel_counts, sort_keys=True),
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

    family_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["target_family"]] = family_counts.get(row["target_family"], 0) + 1
        policy_counts[row["teacher_policy"]] = policy_counts.get(row["teacher_policy"], 0) + 1

    finite_context_frac = sum(row["finite_context"] for row in rows) / max(1, len(rows))
    finite_alpha_frac = sum(row["finite_alpha"] for row in rows) / max(1, len(rows))
    mean_active_frac = sum(row["active_frac"] for row in rows) / max(1, len(rows))
    mean_alpha = sum(row["alpha_mean"] for row in rows) / max(1, len(rows))

    payload = {
        "experiment": "E242 EF-LIC spatial teacher context export",
        "dataset": args.dataset,
        "image_dir": str(args.image_dir),
        "checkpoint": str(args.ckpt_path),
        "force_ind": args.force_ind,
        "device": args.device,
        "alpha_threshold": args.alpha_threshold,
        "selected_images": [p.name for p in selected],
        "exported_images": len(rows),
        "missing_labels": missing,
        "target_family_counts": family_counts,
        "teacher_policy_counts": policy_counts,
        "finite_context_frac": finite_context_frac,
        "finite_alpha_frac": finite_alpha_frac,
        "mean_active_frac": mean_active_frac,
        "mean_alpha": mean_alpha,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with md_path.open("w") as fobj:
        fobj.write("# E242 EF-LIC Spatial Teacher Context Export\n\n")
        fobj.write("This artifact exports decoder-safe EF-LIC local context maps plus map-level HCG teacher targets.\n\n")
        fobj.write(f"- Dataset: `{args.dataset}`\n")
        fobj.write(f"- Image dir: `{args.image_dir}`\n")
        fobj.write(f"- Exported images: `{len(rows)}` / selected `{len(selected)}`\n")
        fobj.write(f"- Force index: `{args.force_ind}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Alpha threshold: `{args.alpha_threshold}`\n")
        fobj.write(f"- Finite context fraction: `{finite_context_frac:.6f}`\n")
        fobj.write(f"- Finite alpha fraction: `{finite_alpha_frac:.6f}`\n")
        fobj.write(f"- Mean active target fraction: `{mean_active_frac:.6f}`\n")
        fobj.write(f"- Mean alpha: `{mean_alpha:.6f}`\n")
        fobj.write(f"- Target family counts: `{json.dumps(family_counts, sort_keys=True)}`\n")
        fobj.write(f"- Teacher policy counts: `{json.dumps(policy_counts, sort_keys=True)}`\n")
        if missing:
            fobj.write(f"- Missing labels: `{len(missing)}`\n")
        fobj.write("\nExport contract:\n\n")
        fobj.write("- `context_maps` has shape `[4, 11, H, W]`.\n")
        fobj.write("- `alpha_target` has shape `[4, 1, H, W]` and comes from the E236 policy map.\n")
        fobj.write("- `target_map` has shape `[4, H, W]`; inactive positions are class zero/fallback.\n")
        fobj.write("- Nonzero target families use the confidence-gated E239 target and oracle policy.\n")

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
            alpha_threshold=args.alpha_threshold,
        )
        rows.append(row)
        if row["nonfinite_context"] or row["nonfinite_alpha"] or not math.isfinite(row["context_rms"]):
            raise SystemExit(f"nonfinite E242 export for {image_path.name}")

    write_outputs(args, rows, selected, missing)


if __name__ == "__main__":
    main()
