#!/usr/bin/env python3
"""Audit E263 fallback-gate pilots with conservative rate accounting.

E263 soft-gate rows use a diagnostic bpp estimate:
    base_bpp + gate_mean * empirical_branch_delta_bpp
This script keeps that score, but also asks what happens if a soft nonzero gate
requires transmitting the full branch index stream. The latter is a conservative
upper-bound check, not the final intended codec accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_smoke_t1_e1_s2.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak2_t2_e2_s4.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4held_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8_t8_e8_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8held_t8_e8_s8.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments/analysis/e264_e263_fallback_gate_rate_audit")
    return p.parse_args()


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, "")
        if val == "":
            return default
        out = float(val)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def dataset_from_path(path: Path) -> str:
    name = path.name
    if "clicpro8held" in name:
        return "clicpro8_held"
    if "clicpro8" in name:
        return "clicpro8_first"
    if "kodak4held" in name:
        return "kodak4_held"
    if "kodak4" in name:
        return "kodak4_first"
    if "kodak2" in name:
        return "kodak2_first"
    if "smoke" in name:
        return "kodak1_smoke"
    return path.stem


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        dataset = dataset_from_path(path)
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                row = dict(row)
                row["dataset"] = dataset
                row["source_csv"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def index_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    label = str(row["label"])
    phase = "trained" if label.startswith("trained_") else "init"
    return (str(row["dataset"]), phase, str(row["image"]), str(row.get("q_index", "0")))


def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "group": name,
        "rows": len(rows),
        "score_diag_mean": mean("score_diag"),
        "score_full_branch_bpp_mean": mean("score_full_branch_bpp"),
        "score_no_bpp_mean": mean("score_no_bpp"),
        "diag_win_rate": sum(float(r["score_diag"]) < 0.0 for r in rows) / len(rows) if rows else 0.0,
        "full_branch_bpp_win_rate": sum(float(r["score_full_branch_bpp"]) < 0.0 for r in rows) / len(rows) if rows else 0.0,
        "diagnostic_dbpp_mean": mean("diagnostic_dbpp"),
        "full_branch_dbpp_mean": mean("full_branch_dbpp"),
        "max_affordable_dbpp_mean": mean("max_affordable_dbpp"),
        "required_rate_fraction_mean": mean("required_rate_fraction"),
        "gate_mean": mean("gate_mean"),
        "nonfinite_rows": int(sum(float(r["nonfinite"]) for r in rows)),
    }


def main() -> None:
    args = parse_args()
    source_rows = read_rows(args.inputs)
    if not source_rows:
        raise SystemExit("no input rows found")

    all_on_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in source_rows:
        label = str(row["label"])
        if label.endswith("_all_on"):
            all_on_by_key[index_key(row)] = row

    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        label = str(row["label"])
        if not label.endswith("_soft_gate"):
            continue
        key = index_key(row)
        all_on = all_on_by_key.get(key)
        if all_on is None:
            continue
        no_bpp = f(row, "delta_dists") + 3.0 * f(row, "delta_lpips")
        diagnostic_dbpp = f(row, "delta_bpp")
        full_dbpp = f(all_on, "delta_bpp")
        max_affordable = max(0.0, -no_bpp)
        required_fraction = max_affordable / full_dbpp if full_dbpp > 0.0 else float("inf")
        phase = "trained" if label.startswith("trained_") else "init"
        audit_rows.append(
            {
                "dataset": key[0],
                "phase": phase,
                "image": key[2],
                "q_index": key[3],
                "label": label,
                "score_diag": f(row, "score"),
                "score_no_bpp": no_bpp,
                "score_full_branch_bpp": no_bpp + full_dbpp,
                "diagnostic_dbpp": diagnostic_dbpp,
                "full_branch_dbpp": full_dbpp,
                "max_affordable_dbpp": max_affordable,
                "required_rate_fraction": required_fraction,
                "gate_mean": f(row, "gate_mean"),
                "delta_psnr": f(row, "delta_psnr"),
                "delta_ms_ssim": f(row, "delta_ms_ssim"),
                "delta_lpips": f(row, "delta_lpips"),
                "delta_dists": f(row, "delta_dists"),
                "all_on_score": f(all_on, "score"),
                "all_on_delta_bpp": full_dbpp,
                "nonfinite": int(f(row, "nonfinite") + f(all_on, "nonfinite")),
                "source_csv": row["source_csv"],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(summarize_group("all_soft", audit_rows))
    for phase in ["init", "trained"]:
        subset = [r for r in audit_rows if r["phase"] == phase]
        if subset:
            summary_rows.append(summarize_group(f"phase:{phase}", subset))
    for dataset in sorted({r["dataset"] for r in audit_rows}):
        subset = [r for r in audit_rows if r["dataset"] == dataset]
        summary_rows.append(summarize_group(f"dataset:{dataset}", subset))
        trained = [r for r in subset if r["phase"] == "trained"]
        if trained:
            summary_rows.append(summarize_group(f"dataset:{dataset}:trained", trained))

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")

    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    with json_path.open("w") as fp:
        json.dump({"summary": summary_rows, "rows": audit_rows}, fp, indent=2)

    lines = [
        "# E264 E263 Fallback-Gate Rate Audit",
        "",
        "Purpose: convert E263 soft-gate rows into a promotion/readiness audit. The normal E263 score uses diagnostic gate-scaled bpp; this report also computes a conservative score that charges the full all-on branch bpp to every nonzero soft-gate row.",
        "",
        "## Summary",
        "",
        "| group | rows | diag score | full-branch-bpp score | no-bpp score | diag win | full-bpp win | diag dbpp | full dbpp | max affordable dbpp | required rate frac | gate | nonfinite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['rows']} | {row['score_diag_mean']:+.6f} | "
            f"{row['score_full_branch_bpp_mean']:+.6f} | {row['score_no_bpp_mean']:+.6f} | "
            f"{row['diag_win_rate']:.3f} | {row['full_branch_bpp_win_rate']:.3f} | "
            f"{row['diagnostic_dbpp_mean']:+.6f} | {row['full_branch_dbpp_mean']:+.6f} | "
            f"{row['max_affordable_dbpp_mean']:+.6f} | {row['required_rate_fraction_mean']:.3f} | "
            f"{row['gate_mean']:.6f} | {row['nonfinite_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A negative diagnostic score means the current soft blend is useful under the gate-scaled bpp proxy. A negative full-branch-bpp score means the image would remain useful even if any nonzero soft gate required transmitting the full branch index stream. The latter is intentionally conservative.",
            "",
            "If `required rate frac` is close to the observed gate, the current diagnostic accounting is plausible but needs a real entropy/index mechanism. If it is much smaller than full activation, dense all-on is expected to fail even when the residual correction improves perceptual metrics.",
            "",
            "The paper-main path should therefore either learn hard sparse activation with final bit accounting, or implement an entropy-coded / progressive branch where the paid index rate tracks gate strength. Until then, E263 is strong design evidence, not final benchmark accounting.",
            "",
            "## Artifacts",
            "",
            f"- `{csv_path.relative_to(ROOT)}`",
            f"- `{json_path.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
