#!/usr/bin/env python3
"""Perceptual-only q-index aggregation for GLC signal-accounted rows.

E371 extends the selected-replacement GLC audit across q indexes.  The older raw
codec-loop summary still carries legacy PSNR fields, so this tool rebuilds the
decision artifact with only perceptual deltas and explicit bpp/signal accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


FOCUS_LABELS = [
    "trained_soft_gate",
    "trained_replacement_soft",
    "trained_replacement_all_on",
    "trained_rate_cap_replacement_soft",
    "trained_rate_cap_replacement_soft_cap0p0035",
    "trained_rate_cap_replacement_soft_cap0p004",
]
DERIVED_CAPS = [0.0030]


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
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def cap_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def perceptual_score(row: dict[str, str] | dict[str, object]) -> float:
    if isinstance(next(iter(row.values()), ""), str):
        return fval(row, "delta_dists") + 3.0 * fval(row, "delta_lpips")
    return float(row["delta_dists"]) + 3.0 * float(row["delta_lpips"])


def fixed_delta_bpp(row: dict[str, str] | dict[str, object]) -> float:
    get = (lambda k, d=math.nan: fval(row, k, d)) if isinstance(next(iter(row.values()), ""), str) else (lambda k, d=math.nan: float(row.get(k, d)))
    if get("selected", 0.0) <= 0.0:
        return get("delta_bpp", 0.0)
    return get("active_rvq_fixed_bpp", 0.0) - get("active_scalar_bpp", 0.0) + get("selection_signal_bpp", 0.0)


def image_signal_bpp(bits: float, row: dict[str, object]) -> float:
    height = max(1.0, float(row.get("height", 1.0)))
    width = max(1.0, float(row.get("width", 1.0)))
    return max(0.0, bits) / (height * width)


def derived_label(cap: float) -> str:
    return f"derived_rate_cap_replacement_soft_cap{cap_token(cap)}"


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                label = raw.get("label", "")
                if label not in FOCUS_LABELS:
                    continue
                selected = fval(raw, "selected", 0.0)
                pscore = perceptual_score(raw)
                dbpp = fval(raw, "delta_bpp", 0.0)
                fixed_dbpp = fixed_delta_bpp(raw)
                rows.append({
                    "artifact": str(path),
                    "q_index": int(fval(raw, "q_index", -1)),
                    "label": label,
                    "image": raw.get("image", ""),
                    "height": fval(raw, "height", 0.0),
                    "width": fval(raw, "width", 0.0),
                    "score": pscore + dbpp,
                    "fixed_score": pscore + fixed_dbpp,
                    "perceptual_score": pscore,
                    "delta_bpp": dbpp,
                    "fixed_delta_bpp": fixed_dbpp,
                    "selected": selected,
                    "selection_signal_bpp": fval(raw, "selection_signal_bpp", 0.0),
                    "gate_mean": fval(raw, "gate_mean"),
                    "delta_ms_ssim": fval(raw, "delta_ms_ssim"),
                    "delta_lpips": fval(raw, "delta_lpips"),
                    "delta_dists": fval(raw, "delta_dists"),
                    "active_replacement_delta_bpp": fval(raw, "active_replacement_delta_bpp"),
                    "active_scalar_bpp": fval(raw, "active_scalar_bpp"),
                    "active_rvq_fixed_bpp": fval(raw, "active_rvq_fixed_bpp"),
                    "index_entropy_mean": fval(raw, "index_entropy_mean"),
                    "index_dead_frac_mean": fval(raw, "index_dead_frac_mean"),
                    "nonfinite": fval(raw, "nonfinite", 0.0),
                })
    return rows


def derive_cap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    derived: list[dict[str, object]] = []
    base_rows = [row for row in rows if row["label"] == "trained_replacement_soft"]
    for row in base_rows:
        replacement_dbpp = float(row["active_replacement_delta_bpp"])
        for cap in DERIVED_CAPS:
            selected = replacement_dbpp <= cap
            out = dict(row)
            out["label"] = derived_label(cap)
            out["selection_signal_bpp"] = 0.0
            out["selected"] = 1.0 if selected else 0.0
            if selected:
                out["score"] = float(row["score"])
                out["fixed_score"] = float(row["fixed_score"])
                out["delta_bpp"] = float(row["delta_bpp"])
                out["fixed_delta_bpp"] = float(row["fixed_delta_bpp"])
            else:
                out["score"] = 0.0
                out["fixed_score"] = 0.0
                out["perceptual_score"] = 0.0
                out["delta_bpp"] = 0.0
                out["fixed_delta_bpp"] = 0.0
                out["gate_mean"] = 0.0
                out["delta_ms_ssim"] = 0.0
                out["delta_lpips"] = 0.0
                out["delta_dists"] = 0.0
            derived.append(out)
    return derived


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = FOCUS_LABELS + [derived_label(cap) for cap in DERIVED_CAPS]
    q_groups: list[object] = sorted({int(row["q_index"]) for row in rows}) + ["all"]
    out: list[dict[str, object]] = []
    for q_group in q_groups:
        q_rows = rows if q_group == "all" else [row for row in rows if int(row["q_index"]) == q_group]
        for label in labels:
            subset = [row for row in q_rows if row["label"] == label]
            if not subset:
                continue
            scores = [float(row["score"]) for row in subset]
            fixed_scores = [float(row["fixed_score"]) for row in subset]
            selected_scores = [float(row["score"]) for row in subset if float(row["selected"]) > 0.0]
            selected_fixed_scores = [float(row["fixed_score"]) for row in subset if float(row["selected"]) > 0.0]
            out.append({
                "q_index": q_group,
                "label": label,
                "rows": len(subset),
                "score": mean(scores),
                "fixed_score": mean(fixed_scores),
                "perceptual_score": mean(float(row["perceptual_score"]) for row in subset),
                "delta_bpp": mean(float(row["delta_bpp"]) for row in subset),
                "fixed_delta_bpp": mean(float(row["fixed_delta_bpp"]) for row in subset),
                "selected_frac": mean(float(row["selected"]) for row in subset),
                "win_frac": mean(1.0 if score < 0.0 else 0.0 for score in scores),
                "fixed_win_frac": mean(1.0 if score < 0.0 else 0.0 for score in fixed_scores),
                "selected_win_frac": mean(1.0 if score < 0.0 else 0.0 for score in selected_scores),
                "selected_fixed_win_frac": mean(1.0 if score < 0.0 else 0.0 for score in selected_fixed_scores),
                "worst_score": max(scores),
                "worst_fixed_score": max(fixed_scores),
                "delta_ms_ssim": mean(float(row["delta_ms_ssim"]) for row in subset),
                "delta_lpips": mean(float(row["delta_lpips"]) for row in subset),
                "delta_dists": mean(float(row["delta_dists"]) for row in subset),
                "gate_mean": mean(float(row["gate_mean"]) for row in subset),
                "index_entropy_mean": mean(float(row["index_entropy_mean"]) for row in subset),
                "index_dead_frac_mean": mean(float(row["index_dead_frac_mean"]) for row in subset),
                "nonfinite_rows": sum(float(row["nonfinite"]) for row in subset),
            })
    return out


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
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
    rows.extend(derive_cap_rows(rows))
    summary = aggregate(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)
    with args.output_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"inputs": [str(path) for path in args.inputs], "summary": summary}, handle, indent=2)

    fields = [
        "q_index",
        "label",
        "rows",
        "score",
        "fixed_score",
        "selected_frac",
        "win_frac",
        "fixed_win_frac",
        "selected_win_frac",
        "selected_fixed_win_frac",
        "worst_score",
        "worst_fixed_score",
        "nonfinite_rows",
    ]
    focus = [
        "trained_soft_gate",
        "trained_replacement_soft",
        "trained_replacement_all_on",
        "trained_rate_cap_replacement_soft",
        derived_label(0.0030),
        "trained_rate_cap_replacement_soft_cap0p0035",
        "trained_rate_cap_replacement_soft_cap0p004",
    ]
    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC Q-Curve Perceptual Signal Pool\n\n")
        handle.write("PSNR is intentionally excluded. Score = delta_DISTS + 3 * delta_LPIPS + delta_bpp; fixed score replaces empirical index entropy with fixed-index cost where selected.\n\n")
        for q_group in sorted({row["q_index"] for row in summary}, key=lambda v: (v == "all", v)):
            subset = [row for row in summary if row["q_index"] == q_group and row["label"] in focus]
            if not subset:
                continue
            handle.write(f"## q_index {q_group}\n\n")
            handle.write("\n".join(table(subset, fields)))
            handle.write("\n\n")


if __name__ == "__main__":
    main()
