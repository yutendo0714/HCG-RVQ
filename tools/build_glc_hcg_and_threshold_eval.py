#!/usr/bin/env python3
"""Compose AND-threshold HCG-RVQ eval folders from an existing threshold export.

This is a fast reliability-controller sweep: reuse already exported single-feature
fixed-gate reconstructions and choose per image between that reconstruction and
base according to an additional feature threshold.  The image is therefore exactly
one of two already-evaluated codec outputs; only the deployment decision and bpp
accounting are recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def cap_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_dims_by_png_name(input_path: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as im:
            width, height = im.size
        out[path.with_suffix(".png").name] = (width, height)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--input-path", type=Path, default=Path("/dpl/clic/professional/test"))
    p.add_argument("--q-index", type=int, default=3)
    p.add_argument("--index-thresholds", type=float, nargs="+", required=True)
    p.add_argument("--active-thresholds", type=float, nargs="+", required=True)
    p.add_argument("--fixed-gates", type=float, nargs="+", required=True)
    p.add_argument("--signal-bits", type=float, default=1.0)
    args = p.parse_args()

    payload = json.loads((args.source_root / "export_rows.json").read_text())
    rows: list[dict[str, Any]] = [r for r in payload["rows"] if int(r.get("q_index", -1)) == args.q_index]
    if not rows:
        raise SystemExit(f"no q={args.q_index} rows in {args.source_root / 'export_rows.json'}")
    dims = image_dims_by_png_name(args.input_path)
    q_dir = f"q{args.q_index}"

    args.output_root.mkdir(parents=True, exist_ok=True)
    base_src_dir = args.source_root / "base" / q_dir
    base_dst_dir = args.output_root / "base" / q_dir
    for row in rows:
        image_name = str(row["image"])
        link_or_copy(base_src_dir / image_name, base_dst_dir / image_name)
    shutil.copy2(base_src_dir / "res.txt", base_dst_dir / "res.txt")

    out_rows = []
    for index_th in args.index_thresholds:
        index_tag = cap_token(float(index_th))
        for active_th in args.active_thresholds:
            active_tag = cap_token(float(active_th))
            for gate in args.fixed_gates:
                gate_tag = cap_token(float(gate))
                src_label = f"th_indexentropymean_ge{index_tag}_g{gate_tag}_replacement_sig{cap_token(args.signal_bits)}b"
                dst_label = (
                    f"th_indexentropymean_ge{index_tag}_activemseratio_ge{active_tag}"
                    f"_g{gate_tag}_replacement_sig{cap_token(args.signal_bits)}b"
                )
                src_dir = args.source_root / src_label / q_dir
                if not src_dir.exists():
                    print(f"[skip] missing source label {src_label}")
                    continue
                dst_dir = args.output_root / dst_label / q_dir
                bpps: list[float] = []
                selected_count = 0
                for row in rows:
                    image_name = str(row["image"])
                    selected = (
                        float(row["index_entropy_mean"]) >= float(index_th)
                        and float(row["active_mse_ratio"]) >= float(active_th)
                    )
                    src_image = (src_dir if selected else base_src_dir) / image_name
                    link_or_copy(src_image, dst_dir / image_name)
                    width, height = dims[image_name]
                    signal_bpp = max(0.0, float(args.signal_bits)) / max(1.0, float(width * height))
                    core_bpp = float(row["replacement_bpp"] if selected else row["base_bpp"])
                    bpps.append(core_bpp + signal_bpp)
                    selected_count += int(selected)
                mean_bpp = sum(bpps) / len(bpps)
                with (dst_dir / "res.txt").open("w") as f:
                    f.write(f"bpp = {mean_bpp:.6f}\n")
                    f.write(f"num_images = {len(bpps)}\n")
                    f.write(f"selected_images = {selected_count}\n")
                    f.write("metrics = pending\n")
                out_rows.append({
                    "label": dst_label,
                    "q_index": args.q_index,
                    "index_entropy_threshold": float(index_th),
                    "active_mse_ratio_threshold": float(active_th),
                    "fixed_gate": float(gate),
                    "signal_bits": float(args.signal_bits),
                    "selected_images": selected_count,
                    "mean_bpp": mean_bpp,
                })
                print(f"[compose] {dst_label}: selected={selected_count}/{len(rows)} bpp={mean_bpp:.6f}")

    (args.output_root / "and_policy_rows.json").write_text(json.dumps(out_rows, indent=2))


if __name__ == "__main__":
    main()
