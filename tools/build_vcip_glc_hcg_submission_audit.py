#!/usr/bin/env python3
"""Build VCIP-facing audit artifacts for deployable GLC + HCG-RVQ exports.

This script does not re-train or re-evaluate the model.  It checks that the
exported image directories, export_rows accounting, and GLC evaluate_quality
res.txt files are mutually consistent, then builds mechanism and qualitative
artifacts for the paper package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageChops, ImageDraw


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
        raise ValueError(f"could not parse {path}")
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


def seed_name(root: Path) -> str:
    for part in root.name.split("_"):
        if part.startswith("seed"):
            return part.replace("seed", "")
    return root.name


def signal_bits_from_label(label: str) -> float:
    match = re.search(r"_sig([0-9p]+)b$", label)
    if match is None:
        return 0.0
    return float(match.group(1).replace("p", "."))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def png_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".png")


def image_pixels(path: Path) -> int:
    with Image.open(path) as img:
        return int(img.width * img.height)


def row_candidate_bpp(row: dict[str, Any], candidate_label: str, image_path: Path) -> float:
    base_bpp = float(row["base_bpp"])
    if candidate_label == "base":
        return base_bpp
    if candidate_label.startswith("replacement_soft_qgate"):
        bpp = float(row["replacement_bpp"])
    elif candidate_label.startswith("replacement_soft"):
        bpp = float(row["replacement_bpp"])
    elif candidate_label.startswith("replacement_hard"):
        bpp = float(row.get("replacement_bpp", base_bpp))
    else:
        bpp = float(row.get("replacement_bpp", base_bpp))
    signal_bits = signal_bits_from_label(candidate_label)
    if signal_bits > 0:
        bpp += signal_bits / float(image_pixels(image_path))
    return bpp


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def accounting_audit(eval_roots: list[Path], candidate_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    mechanism_raw: list[dict[str, Any]] = []
    for root in eval_roots:
        seed = seed_name(root)
        payload = json.loads((root / "export_rows.json").read_text())
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_q[int(row.get("q_index", row.get("q")))].append(row)

        for q, q_rows in sorted(by_q.items()):
            base_dir = root / "base" / f"q{q}"
            cand_dir = root / candidate_label / f"q{q}"
            base_files = png_files(base_dir)
            cand_files = png_files(cand_dir)
            base_names = {p.name for p in base_files}
            cand_names = {p.name for p in cand_files}
            row_names = {str(row["image"]) for row in q_rows}

            base_res = read_res(base_dir / "res.txt")
            cand_res = read_res(cand_dir / "res.txt")

            base_bpps = [float(row["base_bpp"]) for row in q_rows]
            cand_bpps = [
                row_candidate_bpp(row, candidate_label, cand_dir / str(row["image"]))
                for row in q_rows
            ]
            base_bpp_mean, _ = mean_std(base_bpps)
            cand_bpp_mean, _ = mean_std(cand_bpps)

            audit_rows.append(
                {
                    "seed": seed,
                    "q": q,
                    "candidate_label": candidate_label,
                    "base_png_count": len(base_files),
                    "candidate_png_count": len(cand_files),
                    "export_row_count": len(q_rows),
                    "base_candidate_filenames_match": int(base_names == cand_names),
                    "base_export_filenames_match": int(base_names == row_names),
                    "candidate_export_filenames_match": int(cand_names == row_names),
                    "base_res_bpp": base_res["bpp"],
                    "base_export_bpp_mean": base_bpp_mean,
                    "base_bpp_abs_diff": abs(base_res["bpp"] - base_bpp_mean),
                    "candidate_res_bpp": cand_res["bpp"],
                    "candidate_export_bpp_mean": cand_bpp_mean,
                    "candidate_bpp_abs_diff": abs(cand_res["bpp"] - cand_bpp_mean),
                    "candidate_bpp_improvement_pct": (base_res["bpp"] - cand_res["bpp"]) / base_res["bpp"] * 100.0,
                    "lpips_improvement_pct": (base_res["lpips"] - cand_res["lpips"]) / base_res["lpips"] * 100.0,
                    "dists_improvement_pct": (base_res["dists"] - cand_res["dists"]) / base_res["dists"] * 100.0,
                    "ms_ssim_improvement_pct": (cand_res["ms_ssim"] - base_res["ms_ssim"]) / base_res["ms_ssim"] * 100.0,
                    "fid_improvement_pct": (base_res.get("fid", float("nan")) - cand_res.get("fid", float("nan"))) / base_res.get("fid", float("nan")) * 100.0,
                    "kid_improvement_pct": (base_res.get("kid", float("nan")) - cand_res.get("kid", float("nan"))) / base_res.get("kid", float("nan")) * 100.0,
                }
            )

            for row in q_rows:
                item = {"seed": seed, "q": q, "image": row["image"]}
                for key in (
                    "base_bpp",
                    "branch_bpp",
                    "replacement_bpp",
                    "soft_gate_mean",
                    "quantized_soft_gate_mean",
                    "hard_gate_mean",
                    "active_mse_ratio",
                    "index_entropy_mean",
                    "index_used_frac_mean",
                    "index_dead_frac_mean",
                ):
                    value = safe_float(row.get(key))
                    if value is not None:
                        item[key] = value
                mechanism_raw.append(item)
    return audit_rows, mechanism_raw


def aggregate_mechanism(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in sorted({int(row["q"]) for row in rows}):
        subset = [row for row in rows if int(row["q"]) == q]
        item: dict[str, Any] = {"q": q, "n_rows": len(subset)}
        keys = sorted({key for row in subset for key in row} - {"seed", "q", "image"})
        for key in keys:
            values = [float(row[key]) for row in subset if key in row]
            avg, std = mean_std(values)
            item[f"{key}_mean"] = avg
            item[f"{key}_std"] = std
        out.append(item)
    return out


def resolve_original(input_path: Path | None, image_name: str) -> Path | None:
    if input_path is None:
        return None
    direct = input_path / image_name
    if direct.exists():
        return direct
    matches = sorted(input_path.glob(Path(image_name).stem + ".*"))
    return matches[0] if matches else None


def fit_panel(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def diff_panel(base: Image.Image, cand: Image.Image, size: tuple[int, int]) -> Image.Image:
    base = base.convert("RGB")
    cand = cand.convert("RGB")
    diff = ImageChops.difference(base, cand)
    diff = diff.point(lambda v: min(255, v * 4))
    return fit_panel(diff, size)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    pad = 24
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return out


def build_qualitative(
    eval_root: Path,
    candidate_label: str,
    input_path: Path | None,
    output_dir: Path,
    q_values: list[int],
    images_per_q: int,
) -> list[dict[str, Any]]:
    payload = json.loads((eval_root / "export_rows.json").read_text())
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    selected_rows: list[dict[str, Any]] = []
    panel_size = (260, 180)
    for q in q_values:
        subset = [row for row in rows if int(row.get("q_index", row.get("q"))) == q]
        subset.sort(key=lambda row: float(row.get("active_mse_ratio", 0.0)), reverse=True)
        for row in subset[:images_per_q]:
            image_name = str(row["image"])
            orig_path = resolve_original(input_path, image_name)
            base_path = eval_root / "base" / f"q{q}" / image_name
            cand_path = eval_root / candidate_label / f"q{q}" / image_name
            if not base_path.exists() or not cand_path.exists():
                continue
            panels = []
            if orig_path is not None and orig_path.exists():
                with Image.open(orig_path) as img:
                    panels.append(draw_label(fit_panel(img, panel_size), "original"))
            with Image.open(base_path) as base_img, Image.open(cand_path) as cand_img:
                panels.append(draw_label(fit_panel(base_img, panel_size), "GLC base"))
                panels.append(draw_label(fit_panel(cand_img, panel_size), "HCG-RVQ+GLC"))
                panels.append(draw_label(diff_panel(base_img, cand_img, panel_size), "abs diff x4"))
            grid = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
            x = 0
            for panel in panels:
                grid.paste(panel, (x, 0))
                x += panel.width
            out_name = f"q{q}_{image_name}"
            out_path = output_dir / out_name
            output_dir.mkdir(parents=True, exist_ok=True)
            grid.save(out_path)
            selected_rows.append(
                {
                    "q": q,
                    "image": image_name,
                    "grid": str(out_path),
                    "active_mse_ratio": row.get("active_mse_ratio"),
                    "index_entropy_mean": row.get("index_entropy_mean"),
                    "soft_gate_mean": row.get("soft_gate_mean"),
                    "replacement_bpp": row.get("replacement_bpp"),
                }
            )
    return selected_rows


def write_summary(
    path: Path,
    candidate_label: str,
    audit_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    qualitative_rows: list[dict[str, Any]],
) -> None:
    max_base_diff = max(float(row["base_bpp_abs_diff"]) for row in audit_rows)
    max_cand_diff = max(float(row["candidate_bpp_abs_diff"]) for row in audit_rows)
    all_counts_ok = all(
        int(row["base_png_count"]) == int(row["candidate_png_count"]) == int(row["export_row_count"])
        for row in audit_rows
    )
    all_names_ok = all(
        int(row["base_candidate_filenames_match"])
        and int(row["base_export_filenames_match"])
        and int(row["candidate_export_filenames_match"])
        for row in audit_rows
    )
    lines = [
        "# VCIP GLC + HCG-RVQ Submission Audit",
        "",
        f"Candidate label: `{candidate_label}`",
        "",
        "## Accounting Consistency",
        "",
        f"- Image counts match across base/candidate/export rows: `{all_counts_ok}`",
        f"- Filenames match across base/candidate/export rows: `{all_names_ok}`",
        f"- Max base bpp absolute difference between `res.txt` and export rows: `{max_base_diff:.8f}`",
        f"- Max candidate bpp absolute difference between `res.txt` and export rows: `{max_cand_diff:.8f}`",
        "",
        "Important scope note: this audit verifies deterministic export/accounting",
        "consistency for GLC-style evaluation. The current result is not a production",
        "arithmetic bitstream compress/decompress measurement; the paper should state",
        "bit accounting explicitly unless a full bitstream wrapper is added.",
        "",
        "## Mechanism Summary",
        "",
        "| q | replacement bpp | soft gate | index entropy | used frac | dead frac | active mse ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mechanism_rows:
        lines.append(
            "| {q} | {rbpp:.6f} | {gate:.4f} | {ent:.4f} | {used} | {dead} | {mse:.4f} |".format(
                q=row["q"],
                rbpp=float(row.get("replacement_bpp_mean", float("nan"))),
                gate=float(row.get("soft_gate_mean_mean", float("nan"))),
                ent=float(row.get("index_entropy_mean_mean", float("nan"))),
                used=(
                    f"{float(row['index_used_frac_mean_mean']):.4f}"
                    if "index_used_frac_mean_mean" in row
                    else "n/a"
                ),
                dead=(
                    f"{float(row['index_dead_frac_mean_mean']):.4f}"
                    if "index_dead_frac_mean_mean" in row
                    else "n/a"
                ),
                mse=float(row.get("active_mse_ratio_mean", float("nan"))),
            )
        )
    lines.extend(["", "## Qualitative Grids", ""])
    for row in qualitative_rows:
        lines.append(f"- q{row['q']} `{row['image']}`: `{row['grid']}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-label", default="replacement_soft_qgate8b_sig8b")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualitative-seed-root", type=Path, default=None)
    parser.add_argument("--qualitative-q", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--images-per-q", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows, mechanism_raw = accounting_audit(args.eval_roots, args.candidate_label)
    mechanism_rows = aggregate_mechanism(mechanism_raw)
    write_csv(args.output_dir / "accounting_audit_by_seed_q.csv", audit_rows)
    write_csv(args.output_dir / "mechanism_raw_rows.csv", mechanism_raw)
    write_csv(args.output_dir / "mechanism_table_by_q.csv", mechanism_rows)

    qualitative_root = args.qualitative_seed_root or args.eval_roots[0]
    qualitative_rows = build_qualitative(
        qualitative_root,
        args.candidate_label,
        args.input_path,
        args.output_dir / "qualitative_grids",
        args.qualitative_q,
        args.images_per_q,
    )
    write_csv(args.output_dir / "qualitative_selection.csv", qualitative_rows)
    write_summary(
        args.output_dir / "submission_audit_summary.md",
        args.candidate_label,
        audit_rows,
        mechanism_rows,
        qualitative_rows,
    )
    print(args.output_dir / "submission_audit_summary.md")


if __name__ == "__main__":
    main()

