#!/usr/bin/env python3
"""Pool signal-accounted GLC codec-loop replacement rows.

This analyzer is for the current-code E285/E286 codec-loop outputs.  Unlike the
older E283 spreadsheet audit, these rows already contain the charged
decoder-visible selection signal.  The script therefore aggregates the emitted
CSV rows directly and also derives a fixed-index reinterpretation for selected
replacement rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FOCUS_LABELS = [
    "trained_soft_gate",
    "trained_replacement_soft",
    "trained_replacement_all_on",
    "trained_rate_cap_replacement_soft",
    "trained_rate_cap_replacement_soft_sig1b",
    "trained_rate_cap_replacement_soft_sig8b",
    "trained_rate_cap_replacement_soft_cap0p0035",
    "trained_rate_cap_replacement_soft_cap0p0035_sig1b",
    "trained_rate_cap_replacement_soft_cap0p0035_sig8b",
    "trained_rate_cap_replacement_soft_cap0p004",
    "trained_rate_cap_replacement_soft_cap0p004_sig1b",
    "trained_rate_cap_replacement_soft_cap0p004_sig8b",
]

DERIVED_CAPS = [0.0030]
DERIVED_SIGNAL_BITS = [0.0, 1.0, 8.0]


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def source_from_path(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "clicpro" in name:
        return "clic_tail9", "clic"
    if "kodak16held" in name:
        return "kodak16", "kodak"
    if "kodak" in name:
        return "kodak", "kodak"
    return path.stem, "unknown"


def cap_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def fixed_score(row: dict[str, str]) -> float:
    score = fval(row, "score")
    if fval(row, "selected", 0.0) <= 0.0:
        return score
    empirical_delta = fval(row, "active_replacement_delta_bpp")
    fixed_delta = fval(row, "active_rvq_fixed_bpp") - fval(row, "active_scalar_bpp")
    return score - empirical_delta + fixed_delta


def fixed_delta_bpp(row: dict[str, str]) -> float:
    if fval(row, "selected", 0.0) <= 0.0:
        return fval(row, "delta_bpp")
    signal_bpp = fval(row, "selection_signal_bpp", 0.0)
    return fval(row, "active_rvq_fixed_bpp") - fval(row, "active_scalar_bpp") + signal_bpp


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        source, domain = source_from_path(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                label = row.get("label", "")
                if label not in FOCUS_LABELS:
                    continue
                score = fval(row, "score")
                fixed = fixed_score(row)
                rows.append(
                    {
                        "artifact": str(path),
                        "source": source,
                        "domain": domain,
                        "label": label,
                        "image": row.get("image", ""),
                        "height": fval(row, "height", 0.0),
                        "width": fval(row, "width", 0.0),
                        "score": score,
                        "fixed_score": fixed,
                        "delta_bpp": fval(row, "delta_bpp"),
                        "fixed_delta_bpp": fixed_delta_bpp(row),
                        "selection_signal_bpp": fval(row, "selection_signal_bpp", 0.0),
                        "selected": fval(row, "selected", 0.0),
                        "gate_mean": fval(row, "gate_mean"),
                        "delta_psnr": fval(row, "delta_psnr"),
                        "delta_ms_ssim": fval(row, "delta_ms_ssim"),
                        "delta_lpips": fval(row, "delta_lpips"),
                        "delta_dists": fval(row, "delta_dists"),
                        "active_replacement_delta_bpp": fval(row, "active_replacement_delta_bpp"),
                        "active_scalar_bpp": fval(row, "active_scalar_bpp"),
                        "active_rvq_extra_bpp": fval(row, "active_rvq_extra_bpp"),
                        "active_rvq_fixed_bpp": fval(row, "active_rvq_fixed_bpp"),
                        "index_entropy_mean": fval(row, "index_entropy_mean"),
                        "index_dead_frac_mean": fval(row, "index_dead_frac_mean"),
                        "nonfinite": fval(row, "nonfinite", 0.0),
                    }
                )
    return rows


def image_signal_bpp(bits: float, row: dict[str, object]) -> float:
    height = max(1.0, float(row.get("height", 1.0)))
    width = max(1.0, float(row.get("width", 1.0)))
    return max(0.0, bits) / (height * width)


def derived_label(cap: float, signal_bits: float) -> str:
    base = f"derived_rate_cap_replacement_soft_cap{cap_token(cap)}"
    if signal_bits <= 0.0:
        return base
    return f"{base}_sig{cap_token(signal_bits)}b"


def derive_cap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    derived: list[dict[str, object]] = []
    replacements = [row for row in rows if row["label"] == "trained_replacement_soft"]
    for row in replacements:
        replacement_dbpp = float(row["active_replacement_delta_bpp"])
        for cap in DERIVED_CAPS:
            selected = replacement_dbpp <= cap
            for signal_bits in DERIVED_SIGNAL_BITS:
                signal = image_signal_bpp(signal_bits, row)
                out = dict(row)
                out["label"] = derived_label(cap, signal_bits)
                out["selection_signal_bpp"] = signal
                out["selected"] = 1.0 if selected else 0.0
                if selected:
                    out["score"] = float(row["score"]) + signal
                    out["fixed_score"] = float(row["fixed_score"]) + signal
                    out["delta_bpp"] = float(row["delta_bpp"]) + signal
                    out["fixed_delta_bpp"] = float(row["fixed_delta_bpp"]) + signal
                else:
                    out["score"] = signal
                    out["fixed_score"] = signal
                    out["delta_bpp"] = signal
                    out["fixed_delta_bpp"] = signal
                    out["gate_mean"] = 0.0
                    out["delta_psnr"] = 0.0
                    out["delta_ms_ssim"] = 0.0
                    out["delta_lpips"] = 0.0
                    out["delta_dists"] = 0.0
                derived.append(out)
    return derived


def group_key(row: dict[str, object], domain: str) -> tuple[str, str]:
    return (domain, str(row["label"]))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    label_order = FOCUS_LABELS + [
        derived_label(cap, bits) for cap in DERIVED_CAPS for bits in DERIVED_SIGNAL_BITS
    ]
    domains = sorted({str(row["domain"]) for row in rows}) + ["all"]
    for domain in domains:
        domain_rows = rows if domain == "all" else [row for row in rows if row["domain"] == domain]
        labels = [label for label in label_order if any(row["label"] == label for row in domain_rows)]
        for label in labels:
            subset = [row for row in domain_rows if row["label"] == label]
            scores = [float(row["score"]) for row in subset]
            fixed_scores = [float(row["fixed_score"]) for row in subset]
            selected_scores = [
                float(row["score"]) for row in subset if float(row["selected"]) > 0.0
            ]
            selected_fixed_scores = [
                float(row["fixed_score"]) for row in subset if float(row["selected"]) > 0.0
            ]
            out.append(
                {
                    "domain": domain,
                    "label": label,
                    "images": len(subset),
                    "score": mean(scores),
                    "fixed_score": mean(fixed_scores),
                    "delta_bpp": mean([float(row["delta_bpp"]) for row in subset]),
                    "fixed_delta_bpp": mean([float(row["fixed_delta_bpp"]) for row in subset]),
                    "selection_signal_bpp": mean(
                        [float(row["selection_signal_bpp"]) for row in subset]
                    ),
                    "selected_frac": mean([float(row["selected"]) for row in subset]),
                    "win_frac": mean([1.0 if score < 0.0 else 0.0 for score in scores]),
                    "fixed_win_frac": mean(
                        [1.0 if score < 0.0 else 0.0 for score in fixed_scores]
                    ),
                    "selected_win_frac": mean(
                        [1.0 if score < 0.0 else 0.0 for score in selected_scores]
                    ),
                    "selected_fixed_win_frac": mean(
                        [1.0 if score < 0.0 else 0.0 for score in selected_fixed_scores]
                    ),
                    "worst_score": max(scores) if scores else math.nan,
                    "worst_fixed_score": max(fixed_scores) if fixed_scores else math.nan,
                    "delta_psnr": mean([float(row["delta_psnr"]) for row in subset]),
                    "delta_ms_ssim": mean([float(row["delta_ms_ssim"]) for row in subset]),
                    "delta_lpips": mean([float(row["delta_lpips"]) for row in subset]),
                    "delta_dists": mean([float(row["delta_dists"]) for row in subset]),
                    "index_entropy_mean": mean(
                        [float(row["index_entropy_mean"]) for row in subset]
                    ),
                    "index_dead_frac_mean": mean(
                        [float(row["index_dead_frac_mean"]) for row in subset]
                    ),
                    "nonfinite_rows": sum(float(row["nonfinite"]) for row in subset),
                }
            )
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
    if not rows:
        raise SystemExit("no focus rows found")
    rows.extend(derive_cap_rows(rows))
    summary = aggregate(rows)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".summary.csv"), summary)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "rows": len(rows),
                "focus_labels": FOCUS_LABELS,
                "derived_caps": DERIVED_CAPS,
                "derived_signal_bits": DERIVED_SIGNAL_BITS,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    focus_labels = {
        "trained_soft_gate",
        "trained_replacement_soft",
        "trained_replacement_all_on",
        "trained_rate_cap_replacement_soft_cap0p0035",
        "trained_rate_cap_replacement_soft_cap0p0035_sig8b",
        "trained_rate_cap_replacement_soft_cap0p004",
        "trained_rate_cap_replacement_soft_cap0p004_sig8b",
        "derived_rate_cap_replacement_soft_cap0p003",
        "derived_rate_cap_replacement_soft_cap0p003_sig8b",
    }
    pooled = [row for row in summary if row["domain"] == "all" and row["label"] in focus_labels]
    by_domain = [
        row
        for row in summary
        if row["domain"] in {"clic", "kodak"}
        and row["label"]
        in {
            "trained_rate_cap_replacement_soft_cap0p0035",
            "trained_rate_cap_replacement_soft_cap0p0035_sig8b",
            "trained_rate_cap_replacement_soft_cap0p004",
            "trained_rate_cap_replacement_soft_cap0p004_sig8b",
            "derived_rate_cap_replacement_soft_cap0p003",
            "derived_rate_cap_replacement_soft_cap0p003_sig8b",
        }
    ]

    md: list[str] = []
    md.append("# GLC Signal-Accounted Current Subset Audit")
    md.append("")
    md.append(
        "This audit pools the E285 CLIC-tail and E286 Kodak-held codec-loop CSV "
        "rows. Signal-accounted rows are used as emitted by the pilot; fixed "
        "scores reinterpret selected replacement rows with fixed-length RVQ "
        "indices while preserving the charged image-level signal."
    )
    md.append("")
    md.append("## Pooled Focus")
    md.extend(
        table(
            pooled,
            [
                "label",
                "images",
                "score",
                "fixed_score",
                "delta_bpp",
                "fixed_delta_bpp",
                "selection_signal_bpp",
                "selected_frac",
                "win_frac",
                "fixed_win_frac",
                "selected_win_frac",
                "selected_fixed_win_frac",
                "worst_score",
                "worst_fixed_score",
            ],
        )
    )
    md.append("")
    md.append("## Domain Cap Rows")
    md.extend(
        table(
            by_domain,
            [
                "domain",
                "label",
                "images",
                "score",
                "fixed_score",
                "selected_frac",
                "win_frac",
                "fixed_win_frac",
                "selected_win_frac",
                "selected_fixed_win_frac",
                "worst_score",
                "worst_fixed_score",
            ],
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "The pooled table is still a short-cycle current-code subset, not a "
        "paper-final full-training curve. Its role is to decide which GLC "
        "replacement controller should be promoted into longer EF-LIC/GLC "
        "experiments. A good candidate should keep negative mean score, charged "
        "signal overhead, high selected win rate, and acceptable fixed-index "
        "reinterpretation."
    )
    md.append("")
    prefix.with_suffix(".md").write_text("\n".join(md).rstrip() + "\n")


if __name__ == "__main__":
    main()
