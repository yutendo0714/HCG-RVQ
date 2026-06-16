#!/usr/bin/env python3
"""Summarize GLC/HCG-RVQ eval exports against the local base rows.

The script intentionally reports only perceptual/image-compression metrics used
for the current VCIP claim path. Positive improvement means better:
lower bpp/lpips/dists/fid/kid, higher ms-ssim.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


RES_RE = re.compile(
    r"bpp\s*=\s*([0-9.eE+-]+).*?"
    r"ms-ssim\s*=\s*([0-9.eE+-]+).*?"
    r"lpips\s*=\s*([0-9.eE+-]+).*?"
    r"dists\s*=\s*([0-9.eE+-]+)"
    r"(?:.*?fid\s*=\s*([0-9.eE+-]+))?"
    r"(?:.*?kid\s*=\s*([0-9.eE+-]+))?",
    re.S,
)


def read_res(path: Path) -> dict[str, float]:
    text = path.read_text()
    match = RES_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse {path}")
    bpp, ms_ssim, lpips, dists, fid, kid = match.groups()
    out = {
        "bpp": float(bpp),
        "ms_ssim": float(ms_ssim),
        "lpips": float(lpips),
        "dists": float(dists),
    }
    if fid is not None:
        out["fid"] = float(fid)
    if kid is not None:
        out["kid"] = float(kid)
    return out


def improvement(base: float, cur: float, higher_is_better: bool) -> float:
    if base == 0:
        return 0.0
    if higher_is_better:
        return (cur - base) / base * 100.0
    return (base - cur) / base * 100.0


def format_row(label: str, q: str, vals: dict[str, float], base: dict[str, float]) -> str:
    metrics = ["bpp", "lpips", "dists", "ms_ssim", "fid", "kid"]
    cells = [label, q]
    for metric in metrics:
        if metric not in vals or metric not in base:
            cells += ["", ""]
            continue
        higher = metric == "ms_ssim"
        cells += [f"{vals[metric]:.6f}", f"{improvement(base[metric], vals[metric], higher):+.3f}"]
    return ",".join(cells)


def summarize_res(eval_root: Path) -> list[str]:
    labels = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
    if "base" not in labels:
        raise ValueError(f"{eval_root} has no base directory")

    q_names = sorted(p.name for p in (eval_root / "base").iterdir() if p.is_dir())
    base = {q: read_res(eval_root / "base" / q / "res.txt") for q in q_names}

    header = [
        "label",
        "q",
        "bpp",
        "bpp_imp_pct",
        "lpips",
        "lpips_imp_pct",
        "dists",
        "dists_imp_pct",
        "ms_ssim",
        "ms_ssim_imp_pct",
        "fid",
        "fid_imp_pct",
        "kid",
        "kid_imp_pct",
    ]
    lines = [",".join(header)]
    for label in labels:
        for q in q_names:
            res_path = eval_root / label / q / "res.txt"
            if not res_path.exists():
                continue
            try:
                vals = read_res(res_path)
            except ValueError:
                continue
            lines.append(format_row(label, q, vals, base[q]))
    return lines


def summarize_export_rows(eval_root: Path) -> list[str]:
    rows_path = eval_root / "export_rows.json"
    if not rows_path.exists():
        return []
    payload = json.loads(rows_path.read_text())
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        q_value = row.get("q_index", row.get("q"))
        if q_value is None:
            continue
        q = int(q_value)
        for key in [
            "hard_gate_mean",
            "soft_gate_mean",
            "base_bpp",
            "replacement_bpp",
            "branch_bpp",
            "active_mse_ratio",
            "index_entropy_mean",
        ]:
            value = row.get(key)
            if value is not None:
                acc[q][key].append(float(value))

    if not acc:
        return []
    lines = ["q,key,mean,min,max,count,selected_images"]
    for q in sorted(acc):
        selected = ""
        if "hard_gate_mean" in acc[q]:
            selected = str(sum(v > 0.5 for v in acc[q]["hard_gate_mean"]))
        for key in sorted(acc[q]):
            values = acc[q][key]
            lines.append(
                f"{q},{key},{sum(values) / len(values):.6f},"
                f"{min(values):.6f},{max(values):.6f},{len(values)},{selected}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--write", action="store_true", help="Also write summary.csv and export_row_summary.csv into eval_root.")
    args = parser.parse_args()
    summary_lines = summarize_res(args.eval_root)
    print("\n".join(summary_lines))
    export_lines = summarize_export_rows(args.eval_root)
    if export_lines:
        print("\nexport_row_summary")
        print("\n".join(export_lines))
    if args.write:
        args.eval_root.joinpath("summary.csv").write_text("\n".join(summary_lines) + "\n")
        if export_lines:
            args.eval_root.joinpath("export_row_summary.csv").write_text("\n".join(export_lines) + "\n")


if __name__ == "__main__":
    main()
