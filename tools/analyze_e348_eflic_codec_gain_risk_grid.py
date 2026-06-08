#!/usr/bin/env python3
"""Summarize E347 EF-LIC codec-gain controller risk grid results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_INPUTS = [
    ("risk0", 0.0, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_risk0_fixed.csv"),
    ("riskm020", -0.02, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm020_fixed.csv"),
    ("riskm040", -0.04, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm040_fixed.csv"),
    ("riskm060", -0.06, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm060_fixed.csv"),
    ("riskm080", -0.08, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm080_fixed.csv"),
    ("riskm100", -0.10, "experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm100_fixed.csv"),
]


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def finite_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = [f(row, key) for row in rows]
    return [v for v in values if math.isfinite(v)]


def summarize(label: str, max_risk: float, csv_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")
    rows = [row for row in rows if row.get("mode", "trained_hard") == "trained_hard"]
    if not rows:
        raise ValueError(f"no trained_hard rows in CSV: {csv_path}")

    dpsnr = finite_values(rows, "delta_psnr")
    dbpp = finite_values(rows, "delta_bpp")
    decode = finite_values(rows, "max_decode_diff") or finite_values(rows, "decode_max_abs_diff")
    y_gate = finite_values(rows, "y_gate_mean")
    y_alpha = finite_values(rows, "y_alpha_mean")
    y_risk = finite_values(rows, "y_risk_score_mean")
    y_index = finite_values(rows, "y_avg_index_entropy")
    y_used = finite_values(rows, "y_avg_index_used_frac")
    nonfinite = [int(f(row, "nonfinite", 0.0)) for row in rows]

    summary = {
        "label": label,
        "max_risk": max_risk,
        "csv": str(csv_path),
        "n": len(rows),
        "mean_delta_psnr": mean(dpsnr),
        "worst_delta_psnr": min(dpsnr),
        "best_delta_psnr": max(dpsnr),
        "negative_count": sum(v < 0 for v in dpsnr),
        "positive_count": sum(v > 0 for v in dpsnr),
        "mean_delta_bpp": mean(dbpp) if dbpp else float("nan"),
        "max_abs_delta_bpp": max(abs(v) for v in dbpp) if dbpp else float("nan"),
        "max_decode_diff": max(decode) if decode else float("nan"),
        "nonfinite_rows": sum(nonfinite),
        "payload_equal_all": all(str(row.get("payload_equal", "True")).lower() in {"true", "1"} for row in rows),
        "payload_len_equal_all": all(str(row.get("payload_len_equal", "True")).lower() in {"true", "1"} for row in rows),
        "mean_y_gate": mean(y_gate) if y_gate else float("nan"),
        "mean_y_alpha": mean(y_alpha) if y_alpha else float("nan"),
        "mean_y_risk_score": mean(y_risk) if y_risk else float("nan"),
        "mean_y_index_entropy": mean(y_index) if y_index else float("nan"),
        "mean_y_index_used_frac": mean(y_used) if y_used else float("nan"),
    }

    per_image = []
    for row in rows:
        per_image.append({
            "label": label,
            "max_risk": max_risk,
            "image": row.get("image", ""),
            "delta_psnr": f(row, "delta_psnr"),
            "delta_bpp": f(row, "delta_bpp"),
            "max_decode_diff": f(row, "max_decode_diff", f(row, "decode_max_abs_diff", float("nan"))),
            "nonfinite": int(f(row, "nonfinite", 0.0)),
            "y_gate_mean": f(row, "y_gate_mean"),
            "y_alpha_mean": f(row, "y_alpha_mean"),
            "y_risk_score_mean": f(row, "y_risk_score_mean"),
        })
    return summary, per_image


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="experiments/analysis/e348_eflic_codec_gain_risk_grid_kodak17_24")
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    per_image: list[dict[str, object]] = []
    for label, max_risk, csv_name in DEFAULT_INPUTS:
        csv_path = Path(csv_name)
        if not csv_path.exists():
            continue
        summary, rows = summarize(label, max_risk, csv_path)
        summaries.append(summary)
        per_image.extend(rows)

    summaries.sort(key=lambda row: float(row["max_risk"]), reverse=True)
    best_mean = max(summaries, key=lambda row: float(row["mean_delta_psnr"]))
    best_worst = max(summaries, key=lambda row: float(row["worst_delta_psnr"]))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".summary.csv"), summaries)
    write_csv(prefix.with_suffix(".per_image.csv"), per_image)
    with prefix.with_suffix(".json").open("w") as handle:
        json.dump({"summaries": summaries, "best_mean": best_mean, "best_worst": best_worst}, handle, indent=2)

    lines = [
        "# E348 EF-LIC Codec-gain Risk Grid",
        "",
        "Held-out Kodak17-24 codec-loop evaluation for the E347 balanced controller.",
        "All rows keep the EF-LIC fixed-payload contract when `max_abs_delta_bpp=0`, `max_decode_diff=0`, and `nonfinite_rows=0`.",
        "",
        "| label | max risk | mean dPSNR | worst dPSNR | negative | max abs dBPP | decode max | mean gate | mean alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {max_risk:.2f} | {mean_delta_psnr:+.6f} | {worst_delta_psnr:+.6f} | {negative_count} | {max_abs_delta_bpp:.6f} | {max_decode_diff:.3e} | {mean_y_gate:.6f} | {mean_y_alpha:.6f} |".format(**row)
        )
    lines.extend([
        "",
        "Key readings:",
        "",
        f"- Best mean row: `{best_mean['label']}` with {float(best_mean['mean_delta_psnr']):+.6f} dB mean and {float(best_mean['worst_delta_psnr']):+.6f} dB worst.",
        f"- Best tail row: `{best_worst['label']}` with {float(best_worst['mean_delta_psnr']):+.6f} dB mean and {float(best_worst['worst_delta_psnr']):+.6f} dB worst.",
        "- This grid uses a controller trained on the first 16 Kodak images and evaluated on the last 8, so it is controlled evidence, not final paper evidence.",
        "- The next paper-safe step is to freeze the codec-gain risk target and threshold selection on independent calibration data, then evaluate unchanged on Kodak/CLIC held-out splits.",
        "",
        "Artifacts:",
        "",
        f"- `{prefix.with_suffix('.summary.csv')}`",
        f"- `{prefix.with_suffix('.per_image.csv')}`",
        f"- `{prefix.with_suffix('.json')}`",
    ])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
