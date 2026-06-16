#!/usr/bin/env python3
"""Compose adaptive fixed-gate HCG-RVQ eval folders from existing gate exports."""
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


def dims(input_path: Path) -> dict[str, tuple[int, int]]:
    out = {}
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
    p.add_argument("--index-threshold", type=float, default=1.55)
    p.add_argument("--low-gate", type=float, default=0.12)
    p.add_argument("--high-gates", type=float, nargs="+", default=[0.16, 0.2])
    p.add_argument("--active-thresholds", type=float, nargs="+", default=[5.8, 6.0])
    p.add_argument("--signal-bits", type=float, default=2.0)
    args = p.parse_args()

    payload = json.loads((args.source_root / "export_rows.json").read_text())
    rows: list[dict[str, Any]] = [r for r in payload["rows"] if int(r.get("q_index", -1)) == args.q_index]
    if not rows:
        raise SystemExit(f"no rows for q={args.q_index}")
    image_dims = dims(args.input_path)
    q_dir = f"q{args.q_index}"
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_src_dir = args.source_root / "base" / q_dir
    base_dst_dir = args.output_root / "base" / q_dir
    for row in rows:
        image_name = str(row["image"])
        link_or_copy(base_src_dir / image_name, base_dst_dir / image_name)
    shutil.copy2(base_src_dir / "res.txt", base_dst_dir / "res.txt")

    index_tag = cap_token(args.index_threshold)
    low_tag = cap_token(args.low_gate)
    low_src_dir = args.source_root / f"th_indexentropymean_ge{index_tag}_g{low_tag}_replacement_sig1b" / q_dir
    if not low_src_dir.exists():
        raise SystemExit(f"missing low gate source {low_src_dir}")

    policy_rows = []
    for active_th in args.active_thresholds:
        active_tag = cap_token(active_th)
        for high_gate in args.high_gates:
            high_tag = cap_token(high_gate)
            high_src_dir = args.source_root / f"th_indexentropymean_ge{index_tag}_g{high_tag}_replacement_sig1b" / q_dir
            if not high_src_dir.exists():
                print(f"[skip] missing high gate source {high_src_dir}")
                continue
            label = (
                f"th_indexentropymean_ge{index_tag}_adapgate_lowg{low_tag}"
                f"_activemseratio_ge{active_tag}_highg{high_tag}"
                f"_replacement_sig{cap_token(args.signal_bits)}b"
            )
            dst_dir = args.output_root / label / q_dir
            bpps = []
            selected_count = 0
            high_count = 0
            for row in rows:
                image_name = str(row["image"])
                selected = float(row["index_entropy_mean"]) >= args.index_threshold
                high = selected and float(row["active_mse_ratio"]) >= active_th
                if high:
                    src_dir = high_src_dir
                elif selected:
                    src_dir = low_src_dir
                else:
                    src_dir = base_src_dir
                link_or_copy(src_dir / image_name, dst_dir / image_name)
                width, height = image_dims[image_name]
                signal_bpp = max(0.0, args.signal_bits) / max(1.0, float(width * height))
                core_bpp = float(row["replacement_bpp"] if selected else row["base_bpp"])
                bpps.append(core_bpp + signal_bpp)
                selected_count += int(selected)
                high_count += int(high)
            mean_bpp = sum(bpps) / len(bpps)
            with (dst_dir / "res.txt").open("w") as f:
                f.write(f"bpp = {mean_bpp:.6f}\n")
                f.write(f"num_images = {len(bpps)}\n")
                f.write(f"selected_images = {selected_count}\n")
                f.write(f"high_gate_images = {high_count}\n")
                f.write("metrics = pending\n")
            policy_rows.append({
                "label": label,
                "q_index": args.q_index,
                "index_entropy_threshold": args.index_threshold,
                "low_gate": args.low_gate,
                "active_mse_ratio_threshold": active_th,
                "high_gate": high_gate,
                "signal_bits": args.signal_bits,
                "selected_images": selected_count,
                "high_gate_images": high_count,
                "mean_bpp": mean_bpp,
            })
            print(f"[compose] {label}: selected={selected_count}/{len(rows)} high={high_count}/{len(rows)} bpp={mean_bpp:.6f}")
    (args.output_root / "adaptive_gate_rows.json").write_text(json.dumps(policy_rows, indent=2))


if __name__ == "__main__":
    main()
