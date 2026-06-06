#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_value(v: str) -> Any:
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [{k: parse_value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def metric_delta(row: dict[str, Any], use_active: bool, name: str) -> float:
    if not use_active:
        return 0.0
    return float(row[f"active_delta_{name}"])


def summarize(rows: list[dict[str, Any]], threshold: float, dists_w: float, lpips_w: float, psnr_w: float, pos_penalty: float) -> dict[str, Any]:
    decisions = [float(r["selector_probability_encoder"]) >= threshold for r in rows]
    n = max(1, len(rows))
    dd = float(np.mean([metric_delta(r, d, "dists") for r, d in zip(rows, decisions)]))
    dl = float(np.mean([metric_delta(r, d, "lpips") for r, d in zip(rows, decisions)]))
    dp = float(np.mean([metric_delta(r, d, "psnr") for r, d in zip(rows, decisions)]))
    score = dists_w * dd + lpips_w * dl - psnr_w * dp
    score += pos_penalty * max(dd, 0.0) + pos_penalty * max(dl, 0.0)
    return {
        "threshold": float(threshold),
        "images": len(rows),
        "branch_share": float(np.mean(decisions)) if rows else 0.0,
        "selected_delta_dists": dd,
        "selected_delta_lpips": dl,
        "selected_delta_psnr": dp,
        "multiobjective_score": float(score),
        "dists_wins": int(sum(metric_delta(r, d, "dists") < 0 for r, d in zip(rows, decisions))),
        "lpips_wins": int(sum(metric_delta(r, d, "lpips") < 0 for r, d in zip(rows, decisions))),
        "psnr_wins": int(sum(metric_delta(r, d, "psnr") > 0 for r, d in zip(rows, decisions))),
    }


def candidate_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    probs = sorted({float(r["selector_probability_encoder"]) for r in rows})
    candidates = [-1e-9, 0.0, 0.25, 0.5, 0.75, 1.0]
    if probs:
        candidates.append(probs[0] - 1e-9)
        candidates.append(probs[-1] + 1e-9)
        candidates.extend(probs)
        candidates.extend((a + b) * 0.5 for a, b in zip(probs, probs[1:]))
        if "selector_threshold" in rows[0]:
            candidates.append(float(rows[0]["selector_threshold"]))
    return sorted({float(x) for x in candidates})


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    fields = sorted({k for row in summaries for k in row})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summaries)
    best_score = min(summaries, key=lambda r: r["multiobjective_score"])
    best_dists = min(summaries, key=lambda r: r["selected_delta_dists"])
    current_threshold = float(rows[0].get("selector_threshold", 0.5)) if rows else 0.5
    current = min(summaries, key=lambda r: abs(r["threshold"] - current_threshold))
    payload = {
        "input_csv": str(args.input_csv),
        "rows": len(rows),
        "current_threshold": current_threshold,
        "current_summary": current,
        "best_score_summary": best_score,
        "best_dists_summary": best_dists,
        "summaries": summaries,
        "interpretation": "Eval-threshold sweep. Diagnostic calibration upper bound only; do not use as paper-facing validation without independent calibration split.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E201 EF-LIC Head Threshold Transfer Audit",
        "",
        f"Input: `{args.input_csv}`",
        "",
        "This sweeps the fitted head probability threshold on the evaluation rows. It is a diagnostic calibration upper bound, not a paper-facing selector unless the threshold is chosen on an independent calibration split.",
        "",
        "| selector | threshold | branch share | dDISTS | dLPIPS | dPSNR | score | DISTS wins | LPIPS wins | PSNR wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("current", current), ("best_score", best_score), ("best_dists", best_dists)]:
        threshold = row["threshold"]
        branch_share = row["branch_share"]
        dd = row["selected_delta_dists"]
        dl = row["selected_delta_lpips"]
        dp = row["selected_delta_psnr"]
        score = row["multiobjective_score"]
        images = row["images"]
        dists_wins = row["dists_wins"]
        lpips_wins = row["lpips_wins"]
        psnr_wins = row["psnr_wins"]
        lines.append(
            f"| {name} | {threshold:.9f} | {branch_share:.3f} | "
            f"{dd:+.6f} | {dl:+.6f} | {dp:+.6f} | "
            f"{score:+.6f} | {dists_wins}/{images} | {lpips_wins}/{images} | {psnr_wins}/{images} |"
        )
    lines.extend([
        "",
        "Next:",
        "",
        "- If best_score is much better than current, use a separate calibration split or margin/temperature calibration.",
        "- If no threshold improves both DISTS and LPIPS, the head score itself is not ranking useful active cases for this transfer.",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--positive-penalty", type=float, default=20.0)
    args = p.parse_args()
    rows = [r for r in read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    if not rows:
        raise SystemExit("no finite rows")
    summaries = [
        summarize(rows, t, args.dists_weight, args.lpips_weight, args.psnr_weight, args.positive_penalty)
        for t in candidate_thresholds(rows)
    ]
    write_outputs(args, rows, summaries)


if __name__ == "__main__":
    main()
