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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1, get_state_dict  # noqa: E402
from tools.run_e162_glc_pretrained_baseline import (  # noqa: E402
    instrumented_glc_test,
    list_images,
    load_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "experiments" / "data" / "kodak24")
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e164_glc_instrumented_identity_kodak24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def finite_max_abs(x: torch.Tensor) -> float:
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(x[finite].abs().max().item())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)

    rows: list[dict[str, Any]] = []
    for q in args.q_indexes:
        for path in images:
            img01 = load_image(path, device)
            x = from_0_1_to_minus1_1(img01)
            _, _, h, w = x.shape
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(h, w, args.padding_size)
            x_pad = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            original = net.test(x_pad, q)
            inst_x_hat, inst_stats = instrumented_glc_test(net, x_pad, q)

            x_diff = original["x_hat"] - inst_x_hat
            bit_y_diff = float(original["bit_y"]) - float(inst_stats["bit_y"])
            bit_z_diff = float(original["bit_z"]) - float(inst_stats["bit_z"])
            bit_total_diff = float(original["bit"]) - float(inst_stats["bit_total"])
            nonfinite = int(
                (not torch.isfinite(original["x_hat"]).all().item())
                or (not torch.isfinite(inst_x_hat).all().item())
                or any(not math.isfinite(v) for v in [bit_y_diff, bit_z_diff, bit_total_diff])
            )

            row: dict[str, Any] = {
                "q_index": q,
                "image": path.name,
                "height": h,
                "width": w,
                "max_abs_xhat_diff": finite_max_abs(x_diff),
                "mean_abs_xhat_diff": float(torch.nan_to_num(x_diff.abs()).mean().item()),
                "bit_y_diff": bit_y_diff,
                "bit_z_diff": bit_z_diff,
                "bit_total_diff": bit_total_diff,
                "orig_bit_y": float(original["bit_y"]),
                "inst_bit_y": float(inst_stats["bit_y"]),
                "orig_bit_z": float(original["bit_z"]),
                "inst_bit_z": float(inst_stats["bit_z"]),
                "nonfinite": nonfinite,
            }
            rows.append(row)
            print(
                f"q={q} {path.name} max_x={row['max_abs_xhat_diff']:.3e} "
                f"mean_x={row['mean_abs_xhat_diff']:.3e} bit={row['bit_total_diff']:.3e} nonfinite={nonfinite}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for q in args.q_indexes:
        subset = [r for r in rows if int(r["q_index"]) == q]
        if not subset:
            continue
        summary.append(
            {
                "q_index": q,
                "images": len(subset),
                "max_abs_xhat_diff": float(np.max([r["max_abs_xhat_diff"] for r in subset])),
                "mean_abs_xhat_diff": float(np.mean([r["mean_abs_xhat_diff"] for r in subset])),
                "max_abs_bit_y_diff": float(np.max(np.abs([r["bit_y_diff"] for r in subset]))),
                "max_abs_bit_z_diff": float(np.max(np.abs([r["bit_z_diff"] for r in subset]))),
                "max_abs_bit_total_diff": float(np.max(np.abs([r["bit_total_diff"] for r in subset]))),
                "nonfinite_rows": int(sum(int(r["nonfinite"]) for r in subset)),
            }
        )

    payload = {
        "experiment": "E164 GLC official-test vs instrumented-path identity check",
        "checkpoint": str(args.ckpt_path),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "padding_size": args.padding_size,
        "rows": len(rows),
        "summary": summary,
        "note": "This checks evaluator provenance only. It is not an HCG quality experiment.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E164 GLC Official-Test vs Instrumented-Path Identity Check",
        "",
        f"Checkpoint: `{args.ckpt_path}`",
        f"Input: `{args.input_dir}`",
        f"Device: `{device}`",
        "",
        "This verifies that the E162 feature-audit path reproduces official `GLC_Image.test()` before adding HCG.",
        "",
        "| q | images | max xhat diff | mean xhat diff | max bit_y diff | max bit_z diff | max total-bit diff | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['q_index']} | {s['images']} | {s['max_abs_xhat_diff']:.3e} | "
            f"{s['mean_abs_xhat_diff']:.3e} | {s['max_abs_bit_y_diff']:.3e} | "
            f"{s['max_abs_bit_z_diff']:.3e} | {s['max_abs_bit_total_diff']:.3e} | {s['nonfinite_rows']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
