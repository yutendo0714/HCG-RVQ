#!/usr/bin/env python3
"""Build a PSNR-free parallel status report for EF-LIC and GLC HCG-RVQ tracks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def _fmt_plain(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _choices(choices: dict[str, int]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(choices.items()))


def _eflic_row(label: str, row: dict[str, Any], images: int) -> dict[str, Any]:
    return {
        "label": label,
        "mean_score": float(row["mean_score"]),
        "worst_score": float(row["worst_score"]),
        "score_wins": int(row["score_wins"]),
        "images": images,
        "choices": dict(row.get("choices", {})),
    }


def _load_eflic(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    images = int(data["images"])
    best_policy = data["best_policy"]
    rows = [
        _eflic_row("perceptual oracle", data["baselines"]["oracle_score"], images),
        _eflic_row("learned LOO ridge", data["policies"][best_policy], images),
        _eflic_row("fixed slice 1,2,3", data["baselines"]["fixed_1,2,3"], images),
        _eflic_row("fixed slice 2", data["baselines"]["fixed_2"], images),
        _eflic_row("all-on", data["baselines"]["fixed_all"], images),
        _eflic_row("no-op", data["baselines"]["fixed_noop"], images),
    ]
    return rows


def _load_glc(path: Path) -> list[dict[str, Any]]:
    wanted = {
        ("all", "derived_rate_cap_replacement_soft_cap0p003"): "strict cap 0.0030",
        ("all", "derived_rate_cap_replacement_soft_cap0p003_sig8b"): "strict cap 0.0030 + 8-bit signal",
        ("all", "trained_rate_cap_replacement_soft_cap0p0035"): "balanced cap 0.0035",
        ("all", "trained_rate_cap_replacement_soft_cap0p0035_sig8b"): "balanced cap 0.0035 + 8-bit signal",
        ("all", "trained_rate_cap_replacement_soft_cap0p004"): "aggressive cap 0.0040",
        ("all", "trained_replacement_all_on"): "dense all-on",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["domain"], row["label"])
            if key not in wanted:
                continue
            rows.append(
                {
                    "label": wanted[key],
                    "score": float(row["score"]),
                    "fixed_score": float(row["fixed_score"]),
                    "delta_bpp": float(row["delta_bpp"]),
                    "fixed_delta_bpp": float(row["fixed_delta_bpp"]),
                    "selection_signal_bpp": float(row["selection_signal_bpp"]),
                    "selected_frac": float(row["selected_frac"]),
                    "win_frac": float(row["win_frac"]),
                    "selected_win_frac": float(row["selected_win_frac"]),
                    "selected_fixed_win_frac": float(row["selected_fixed_win_frac"]),
                    "worst_score": float(row["worst_score"]),
                    "nonfinite_rows": int(float(row["nonfinite_rows"])),
                }
            )
    order = [
        "strict cap 0.0030",
        "strict cap 0.0030 + 8-bit signal",
        "balanced cap 0.0035",
        "balanced cap 0.0035 + 8-bit signal",
        "aggressive cap 0.0040",
        "dense all-on",
    ]
    return sorted(rows, key=lambda r: order.index(r["label"]))


def _write_markdown(path: Path, eflic: list[dict[str, Any]], glc: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# E367 Perceptual-Only Parallel Status")
    lines.append("")
    lines.append(
        "This report intentionally excludes PSNR from the decision tables. The "
        "paper-facing generative/low-bitrate branch is judged by perceptual score "
        "`delta_DISTS + 3 * delta_LPIPS` (lower is better), MS-SSIM/bpp-style "
        "accounting, decode/payload consistency, and nonfinite checks."
    )
    lines.append("")
    lines.append("## EF-LIC Track")
    lines.append("")
    lines.append(
        "EF-LIC remains an active parallel track, but it is not ready for all-on "
        "or final full training. The useful signal is selective, decoder-visible "
        "HCG geometry; the bottleneck is reliable local selection."
    )
    lines.append("")
    lines.append("| policy | mean score | worst score | score wins | choices |")
    lines.append("|---|---:|---:|---:|---|")
    for row in eflic:
        lines.append(
            "| {label} | {mean} | {worst} | {wins}/{images} | {choices} |".format(
                label=row["label"],
                mean=_fmt(row["mean_score"]),
                worst=_fmt(row["worst_score"]),
                wins=row["score_wins"],
                images=row["images"],
                choices=_choices(row["choices"]),
            )
        )
    lines.append("")
    lines.append("EF-LIC decision: keep the original decorrelation/fixed-payload path intact,")
    lines.append(
        "then train or freeze a tail-constrained local controller. Promote to "
        "full training only after a held-out controller beats fixed slice/risk "
        "baselines in mean score without positive tail expansion, while keeping "
        "bpp/decode exact and nonfinite rows at zero."
    )
    lines.append("")
    lines.append("## GLC Track")
    lines.append("")
    lines.append(
        "GLC is also active in parallel. Its current stronger direction is not "
        "dense all-on quantization, but selected replacement with explicit "
        "signal/index accounting. Cap settings can differ from EF-LIC because "
        "the codec contract and failure modes differ."
    )
    lines.append("")
    lines.append(
        "| controller | score | fixed score | delta bpp | fixed delta bpp | signal bpp | selected | wins | selected wins | selected fixed wins | worst score | nonfinite |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in glc:
        lines.append(
            "| {label} | {score} | {fixed_score} | {dbpp} | {fdbpp} | {sig} | {sel} | {win} | {swin} | {sfwin} | {worst} | {nonfinite} |".format(
                label=row["label"],
                score=_fmt(row["score"]),
                fixed_score=_fmt(row["fixed_score"]),
                dbpp=_fmt_plain(row["delta_bpp"]),
                fdbpp=_fmt_plain(row["fixed_delta_bpp"]),
                sig=_fmt_plain(row["selection_signal_bpp"], 8),
                sel=_fmt_plain(row["selected_frac"]),
                win=_fmt_plain(row["win_frac"]),
                swin=_fmt_plain(row["selected_win_frac"]),
                sfwin=_fmt_plain(row["selected_fixed_win_frac"]),
                worst=_fmt(row["worst_score"]),
                nonfinite=row["nonfinite_rows"],
            )
        )
    lines.append("")
    lines.append("GLC decision: cap 0.0035 is the balanced paper-facing controller,")
    lines.append(
        "cap 0.0030 is the stricter fixed-index/no-entropy candidate, and "
        "cap 0.0040 is an aggressive performance branch that must carry tail "
        "failure analysis. Dense all-on is a negative control, not a promotion path."
    )
    lines.append("")
    lines.append("## Parallel Plan")
    lines.append("")
    lines.append("- EF-LIC: build a local/perceptual reliability controller before full training.")
    lines.append("- GLC: promote selected replacement to larger perceptual full evaluation first.")
    lines.append("- Both: keep auxiliary losses minimal and account for any decoder-visible signal.")
    lines.append("- Both: report LPIPS/DISTS/MS-SSIM/bpp plus decode/nonfinite checks; PSNR is excluded from paper-facing decisions.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eflic-json",
        type=Path,
        default=Path("experiments/analysis/e366_eflic_perceptual_candidate_policy_loo_kodak24_riskm060.json"),
    )
    parser.add_argument(
        "--glc-summary",
        type=Path,
        default=Path("experiments/analysis/e287_glc_signal_accounted_clictail9_kodak16_current_subset.summary.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e367_perceptual_only_parallel_status"),
    )
    args = parser.parse_args()

    eflic = _load_eflic(args.eflic_json)
    glc = _load_glc(args.glc_summary)
    payload = {
        "metric_policy": "PSNR excluded; score = delta_DISTS + 3 * delta_LPIPS, lower is better",
        "eflic": eflic,
        "glc": glc,
    }

    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(args.output_prefix.with_suffix(".md"), eflic, glc)


if __name__ == "__main__":
    main()
