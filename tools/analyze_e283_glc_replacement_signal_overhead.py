#!/usr/bin/env python3
"""Audit signaling overhead for GLC selected-replacement claims.

E281/E282 showed that the GLC HCG-RVQ branch is useful when framed as a
selected replacement for inefficient scalar residual bits. This audit adds
explicit selection-signal overhead to the same rows, separating cheap coarse
signals from expensive dense/tile maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


CAPS = [0.0025, 0.0030, 0.0035, 0.0040]
PROFILES = [
    "none",
    "image1_selected",
    "image1_all",
    "image8_all",
    "tile64_selected",
    "tile32_selected",
]


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def mean(values: Iterable[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else math.nan


def source_from_path(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "clicpro16held32" in name:
        return "clic_tail9", "clic"
    if "clicpro16held" in name:
        return "clic16", "clic"
    if "clicpro8held" in name:
        return "clic8", "clic"
    if "kodak16held" in name:
        return "kodak16", "kodak"
    if "kodak4held" in name:
        return "kodak4", "kodak"
    if "clic" in name:
        return "clic_unknown", "clic"
    if "kodak" in name:
        return "kodak_unknown", "kodak"
    return path.stem, "unknown"


def enrich_row(row: dict[str, str], path: Path) -> dict[str, float | str]:
    source, domain = source_from_path(path)
    score = fval(row, "score")
    empirical_delta = fval(row, "active_replacement_delta_bpp")
    active_scalar = fval(row, "active_scalar_bpp")
    active_rvq_emp = fval(row, "active_rvq_extra_bpp")
    active_rvq_fixed = fval(row, "active_rvq_fixed_bpp", active_rvq_emp)
    fixed_delta = active_rvq_fixed - active_scalar
    fixed_score = score + (fixed_delta - empirical_delta)
    return {
        "domain": domain,
        "source": source,
        "artifact": str(path),
        "image": row.get("image", ""),
        "height": fval(row, "height", 0.0),
        "width": fval(row, "width", 0.0),
        "score": score,
        "fixed_replacement_score": fixed_score,
        "active_replacement_delta_bpp": empirical_delta,
        "fixed_replacement_delta_bpp": fixed_delta,
        "active_scalar_bpp": active_scalar,
        "active_rvq_extra_bpp": active_rvq_emp,
        "active_rvq_fixed_bpp": active_rvq_fixed,
        "delta_psnr": fval(row, "delta_psnr"),
        "delta_ms_ssim": fval(row, "delta_ms_ssim"),
        "delta_lpips": fval(row, "delta_lpips"),
        "delta_dists": fval(row, "delta_dists"),
        "index_entropy_mean": fval(row, "index_entropy_mean"),
        "index_dead_frac_mean": fval(row, "index_dead_frac_mean"),
    }


def read_rows(paths: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("label") == "trained_replacement_soft":
                    rows.append(enrich_row(row, path))
    return rows


def num(row: dict[str, float | str], key: str) -> float:
    value = row.get(key, math.nan)
    return float(value) if isinstance(value, (float, int)) else math.nan


def signal_bits(profile: str, row: dict[str, float | str], selected: bool) -> float:
    if profile == "none":
        return 0.0
    if profile == "image1_selected":
        return 1.0 if selected else 0.0
    if profile == "image1_all":
        return 1.0
    if profile == "image8_all":
        return 8.0
    if not selected:
        return 0.0

    height = max(1.0, num(row, "height"))
    width = max(1.0, num(row, "width"))
    if profile == "tile64_selected":
        return math.ceil(height / 64.0) * math.ceil(width / 64.0)
    if profile == "tile32_selected":
        return math.ceil(height / 32.0) * math.ceil(width / 32.0)
    raise ValueError(f"unknown signal profile: {profile}")


def signal_bpp(profile: str, row: dict[str, float | str], selected: bool) -> float:
    height = max(1.0, num(row, "height"))
    width = max(1.0, num(row, "width"))
    return signal_bits(profile, row, selected) / (height * width)


def eval_rows(
    rows: list[dict[str, float | str]],
    cap: float,
    profile: str,
    domain: str,
) -> dict[str, float | int | str]:
    subset = rows if domain == "all" else [row for row in rows if row["domain"] == domain]
    selected_flags = [num(row, "active_replacement_delta_bpp") <= cap for row in subset]
    emp_scores: list[float] = []
    fixed_scores: list[float] = []
    emp_dbpp: list[float] = []
    fixed_dbpp: list[float] = []
    overheads: list[float] = []

    for row, selected in zip(subset, selected_flags):
        overhead = signal_bpp(profile, row, selected)
        overheads.append(overhead)
        if selected:
            emp_scores.append(num(row, "score") + overhead)
            fixed_scores.append(num(row, "fixed_replacement_score") + overhead)
            emp_dbpp.append(num(row, "active_replacement_delta_bpp") + overhead)
            fixed_dbpp.append(num(row, "fixed_replacement_delta_bpp") + overhead)
        else:
            emp_scores.append(overhead)
            fixed_scores.append(overhead)
            emp_dbpp.append(overhead)
            fixed_dbpp.append(overhead)

    selected_rows = [row for row, selected in zip(subset, selected_flags) if selected]
    selected_emp_scores = [
        num(row, "score") + signal_bpp(profile, row, True) for row in selected_rows
    ]
    selected_fixed_scores = [
        num(row, "fixed_replacement_score") + signal_bpp(profile, row, True)
        for row in selected_rows
    ]

    return {
        "domain": domain,
        "cap": cap,
        "signal_profile": profile,
        "images": len(subset),
        "selected_frac": mean(1.0 if selected else 0.0 for selected in selected_flags),
        "empirical_score_with_signal": mean(emp_scores),
        "fixed_score_with_signal": mean(fixed_scores),
        "empirical_delta_bpp_with_signal": mean(emp_dbpp),
        "fixed_delta_bpp_with_signal": mean(fixed_dbpp),
        "mean_signal_bpp": mean(overheads),
        "max_signal_bpp": max(overheads) if overheads else math.nan,
        "empirical_win_frac_with_signal": mean(1.0 if score < 0.0 else 0.0 for score in emp_scores),
        "fixed_win_frac_with_signal": mean(1.0 if score < 0.0 else 0.0 for score in fixed_scores),
        "selected_empirical_win_frac_with_signal": mean(
            1.0 if score < 0.0 else 0.0 for score in selected_emp_scores
        ),
        "selected_fixed_win_frac_with_signal": mean(
            1.0 if score < 0.0 else 0.0 for score in selected_fixed_scores
        ),
    }


def audit(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    domains = sorted({str(row["domain"]) for row in rows}) + ["all"]
    out: list[dict[str, float | int | str]] = []
    for domain in domains:
        for cap in CAPS:
            for profile in PROFILES:
                out.append(eval_rows(rows, cap, profile, domain))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if abs(value) < 1 and value != 0:
            return f"{value:+.6f}"
        return f"{value:.6f}"
    return str(value)


def md_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    if not rows:
        raise SystemExit("no trained_replacement_soft rows found")

    summary = audit(rows)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".summary.csv"), summary)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "rows": len(rows),
                "caps": CAPS,
                "signal_profiles": PROFILES,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    focus_profiles = {"none", "image1_all", "image8_all", "tile64_selected", "tile32_selected"}
    focus_caps = {0.0030, 0.0035, 0.0040}
    focus = [
        row
        for row in summary
        if row["domain"] == "all"
        and row["cap"] in focus_caps
        and row["signal_profile"] in focus_profiles
    ]
    clic_focus = [
        row
        for row in summary
        if row["domain"] == "clic"
        and row["cap"] in focus_caps
        and row["signal_profile"] in {"none", "image1_all", "tile64_selected"}
    ]

    md: list[str] = []
    md.append("# GLC Replacement Signal Overhead Audit")
    md.append("")
    md.append(
        "This audit adds explicit selected-replacement signaling overhead to the "
        "E276/E277/E279 replacement rows. It separates cheap image-level mode "
        "signals from denser tile-map signals, because a paper-facing codec "
        "claim must charge the decoder-visible selection mechanism."
    )
    md.append("")
    md.append("## Pooled Focus")
    md.extend(
        md_table(
            focus,
            [
                "cap",
                "signal_profile",
                "selected_frac",
                "empirical_score_with_signal",
                "fixed_score_with_signal",
                "mean_signal_bpp",
                "max_signal_bpp",
                "selected_empirical_win_frac_with_signal",
                "selected_fixed_win_frac_with_signal",
            ],
        )
    )
    md.append("")
    md.append("## CLIC Focus")
    md.extend(
        md_table(
            clic_focus,
            [
                "cap",
                "signal_profile",
                "selected_frac",
                "empirical_score_with_signal",
                "fixed_score_with_signal",
                "mean_signal_bpp",
                "max_signal_bpp",
                "selected_empirical_win_frac_with_signal",
                "selected_fixed_win_frac_with_signal",
            ],
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "If image-level mode signaling barely changes the score, the selected "
        "replacement claim can be implemented as a coarse decoder-safe mode "
        "without hiding overhead. If tile-map signaling consumes noticeable "
        "margin, dense local selection should remain an ablation until the map "
        "itself is compressed or predicted from decoder-available state."
    )
    md.append("")
    prefix.with_suffix(".md").write_text("\n".join(md).rstrip() + "\n")


if __name__ == "__main__":
    main()
