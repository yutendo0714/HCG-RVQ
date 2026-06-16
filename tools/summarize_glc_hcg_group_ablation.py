#!/usr/bin/env python3
"""Aggregate group-wise HCG-RVQ/GLC evaluation directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = [
    "bpp_imp_pct",
    "lpips_imp_pct",
    "dists_imp_pct",
    "ms_ssim_imp_pct",
    "fid_imp_pct",
    "kid_imp_pct",
]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def selected_images(group_dir: Path) -> str:
    path = group_dir / "export_row_summary.csv"
    if not path.exists():
        return ""
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("key") == "hard_gate_mean":
                return row.get("selected_images", "")
    return ""


def score_row(row: dict[str, str]) -> tuple[int, float, float]:
    wins = 0
    for key in ["lpips_imp_pct", "dists_imp_pct", "ms_ssim_imp_pct"]:
        value = safe_float(row, key)
        if value is not None and value > 0:
            wins += 1
    bpp = safe_float(row, "bpp_imp_pct") or 0.0
    dists = safe_float(row, "dists_imp_pct") or 0.0
    return wins, dists, bpp


def summarize(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = group_dir / "summary.csv"
        if not summary.exists():
            continue
        for row in read_summary(summary):
            if row.get("label") == "base":
                continue
            out = {"group_set": group_dir.name, "selected_images": selected_images(group_dir)}
            out.update(row)
            rows.append(out)
    rows.sort(key=score_row, reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows = summarize(args.eval_root)
    header = ["group_set", "selected_images", "label", "q", "bpp", *METRIC_COLUMNS]
    print(",".join(header))
    for row in rows:
        print(",".join(row.get(key, "") for key in header))

    if args.write:
        out_path = args.eval_root / "group_ablation_summary.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
