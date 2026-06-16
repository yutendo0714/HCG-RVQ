#!/usr/bin/env python3
"""Re-run GLC evaluate_quality on already exported reconstruction folders."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "GLC"))

from src.utils.metric_image import evaluate_quality  # noqa: E402


BPP_RE = re.compile(r"bpp\s*=\s*([0-9.eE+-]+)")


def read_mean_bpp(res_path: Path) -> float:
    match = BPP_RE.search(res_path.read_text())
    if match is None:
        raise ValueError(f"Could not parse bpp from {res_path}")
    return float(match.group(1))


def image_count(input_path: Path) -> int:
    return sum(1 for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--input-path", type=Path, default=Path("/dpl/clic/professional/test"))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--q-indexes", nargs="+", default=["3"])
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    labels = args.labels
    if labels is None:
        labels = sorted(p.name for p in args.eval_root.iterdir() if p.is_dir())
    count = image_count(args.input_path)
    for label in labels:
        for q in args.q_indexes:
            q_name = f"q{q}" if not str(q).startswith("q") else str(q)
            out_dir = args.eval_root / label / q_name
            res_path = out_dir / "res.txt"
            if not res_path.exists():
                print(f"[skip] missing {res_path}")
                continue
            mean_bpp = read_mean_bpp(res_path)
            print(f"[eval] {label}/{q_name}: mean_bpp={mean_bpp:.6f}, patch_size={args.patch_size}")
            evaluate_quality(
                [mean_bpp] * count,
                input_path=str(args.input_path),
                output_path=str(out_dir),
                log_path=str(out_dir),
                patch_size=args.patch_size,
            )


if __name__ == "__main__":
    main()
