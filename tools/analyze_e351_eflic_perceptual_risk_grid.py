#!/usr/bin/env python3
"""Summarize EF-LIC HCG perceptual risk sweeps.

PSNR is reported as a diagnostic, while the selector score defaults to
DISTS + w * LPIPS (lower is better). No-op has zero deltas and score 0.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def f(row: dict, key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in (None, ""):
        return default
    return float(val)


def parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Run spec must be name=csv, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Run name is empty")
    return name, Path(path)


def read_rows(path: Path, lpips_weight: float) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out = dict(row)
            out["delta_psnr"] = f(row, "delta_psnr")
            out["delta_ms_ssim"] = f(row, "delta_ms_ssim")
            out["delta_lpips"] = f(row, "delta_lpips")
            out["delta_dists"] = f(row, "delta_dists")
            out["delta_bpp"] = f(row, "delta_bpp")
            out["max_decode_diff"] = f(row, "max_decode_diff")
            out["nonfinite"] = int(f(row, "nonfinite", 0.0))
            out["gate_mean"] = f(row, "y_gate_mean", f(row, "gate_mean"))
            out["alpha_mean"] = f(row, "y_alpha_mean", f(row, "alpha_mean"))
            out["geometry_delta_rms"] = f(row, "y_avg_geometry_delta_rms", f(row, "geometry_delta_rms"))
            out["score_dists_lpips"] = out["delta_dists"] + lpips_weight * out["delta_lpips"]
            rows.append(out)
    return rows


def run_summary(name: str, rows: List[dict]) -> dict:
    scores = [r["score_dists_lpips"] for r in rows]
    return {
        "risk": name,
        "n": len(rows),
        "mean_delta_psnr": mean(r["delta_psnr"] for r in rows),
        "median_delta_psnr": median(r["delta_psnr"] for r in rows),
        "worst_delta_psnr": min(r["delta_psnr"] for r in rows),
        "negative_psnr_count": sum(r["delta_psnr"] < 0 for r in rows),
        "mean_delta_ms_ssim": mean(r["delta_ms_ssim"] for r in rows),
        "mean_delta_lpips": mean(r["delta_lpips"] for r in rows),
        "mean_delta_dists": mean(r["delta_dists"] for r in rows),
        "mean_score_dists_lpips": mean(scores),
        "median_score_dists_lpips": median(scores),
        "worst_score_dists_lpips": max(scores),
        "score_win_count": sum(s < 0 for s in scores),
        "triple_perceptual_win_count": sum(
            r["delta_ms_ssim"] > 0 and r["delta_lpips"] < 0 and r["delta_dists"] < 0 for r in rows
        ),
        "psnr_bad_score_good_count": sum(r["delta_psnr"] < 0 and r["score_dists_lpips"] < 0 for r in rows),
        "psnr_good_score_bad_count": sum(r["delta_psnr"] > 0 and r["score_dists_lpips"] > 0 for r in rows),
        "mean_gate": mean(r["gate_mean"] for r in rows),
        "mean_alpha": mean(r["alpha_mean"] for r in rows),
        "max_abs_delta_bpp": max(abs(r["delta_bpp"]) for r in rows),
        "max_decode_diff": max(r["max_decode_diff"] for r in rows),
        "nonfinite_rows": sum(r["nonfinite"] for r in rows),
    }


def oracle_summary(runs: Dict[str, List[dict]]) -> Tuple[dict, List[dict]]:
    by_image: Dict[str, List[dict]] = defaultdict(list)
    for name, rows in runs.items():
        for row in rows:
            rr = dict(row)
            rr["risk"] = name
            by_image[rr["image"]].append(rr)

    chosen_rows: List[dict] = []
    for image, candidates in sorted(by_image.items()):
        noop = {
            "image": image,
            "risk": "noop",
            "delta_psnr": 0.0,
            "delta_ms_ssim": 0.0,
            "delta_lpips": 0.0,
            "delta_dists": 0.0,
            "score_dists_lpips": 0.0,
            "delta_bpp": 0.0,
            "max_decode_diff": 0.0,
            "nonfinite": 0,
            "gate_mean": 0.0,
            "alpha_mean": 0.0,
        }
        best = min([noop] + candidates, key=lambda r: (r["score_dists_lpips"], -r["delta_ms_ssim"]))
        chosen_rows.append(best)

    choices = Counter(r["risk"] for r in chosen_rows)
    summary = {
        "n": len(chosen_rows),
        "mean_delta_psnr": mean(r["delta_psnr"] for r in chosen_rows),
        "mean_delta_ms_ssim": mean(r["delta_ms_ssim"] for r in chosen_rows),
        "mean_delta_lpips": mean(r["delta_lpips"] for r in chosen_rows),
        "mean_delta_dists": mean(r["delta_dists"] for r in chosen_rows),
        "mean_score_dists_lpips": mean(r["score_dists_lpips"] for r in chosen_rows),
        "worst_score_dists_lpips": max(r["score_dists_lpips"] for r in chosen_rows),
        "worst_delta_psnr": min(r["delta_psnr"] for r in chosen_rows),
        "negative_psnr_count": sum(r["delta_psnr"] < 0 for r in chosen_rows),
        "choice_counts": dict(sorted(choices.items())),
    }
    return summary, chosen_rows


def write_csv(path: Path, rows: List[dict], fields: Iterable[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_run, required=True, help="riskName=path.csv")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--lpips-weight", type=float, default=3.0)
    args = ap.parse_args()

    runs = {name: read_rows(path, args.lpips_weight) for name, path in args.run}
    summaries = [run_summary(name, rows) for name, rows in runs.items()]
    oracle, oracle_rows = oracle_summary(runs)

    prefix = Path(args.output_prefix)
    payload = {
        "lpips_weight": args.lpips_weight,
        "score": f"delta_DISTS + {args.lpips_weight:g} * delta_LPIPS",
        "risk_summaries": summaries,
        "oracle": oracle,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_name(prefix.name + "_risk_summary.csv"), summaries, summaries[0].keys())
    write_csv(
        prefix.with_name(prefix.name + "_oracle_choices.csv"),
        oracle_rows,
        ["image", "risk", "score_dists_lpips", "delta_psnr", "delta_ms_ssim", "delta_lpips", "delta_dists", "gate_mean", "alpha_mean"],
    )

    lines = [
        "# E351 EF-LIC Perceptual Risk Grid", "",
        "PSNR is reported as a codec-health diagnostic. The selector score is "
        f"`delta_DISTS + {args.lpips_weight:g} * delta_LPIPS` (lower is better); no-op has score 0.",
        "",
        "| risk | n | dPSNR | dMS-SSIM | dLPIPS | dDISTS | score | score wins | triple wins | mean gate | mean alpha | PSNR bad/score good | PSNR good/score bad | worst PSNR | max dBPP | decode max | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {risk} | {n} | {dpsnr} | {dms} | {dlpips} | {ddists} | {score} | {score_wins} | {triple} | {gate} | {alpha} | {bad_good} | {good_bad} | {worst} | {dbpp} | {dec} | {nonfinite} |".format(
                risk=s["risk"], n=s["n"], dpsnr=fmt(s["mean_delta_psnr"]), dms=fmt(s["mean_delta_ms_ssim"]),
                dlpips=fmt(s["mean_delta_lpips"]), ddists=fmt(s["mean_delta_dists"]), score=fmt(s["mean_score_dists_lpips"]),
                score_wins=s["score_win_count"], triple=s["triple_perceptual_win_count"],
                gate=f"{s['mean_gate']:.6f}", alpha=f"{s['mean_alpha']:.6f}",
                bad_good=s["psnr_bad_score_good_count"], good_bad=s["psnr_good_score_bad_count"],
                worst=fmt(s["worst_delta_psnr"]), dbpp=fmt(s["max_abs_delta_bpp"]),
                dec=f"{s['max_decode_diff']:.3e}", nonfinite=s["nonfinite_rows"],
            )
        )
    lines += [
        "", "## No-op/Active Oracle", "",
        f"Oracle mean score: `{fmt(oracle['mean_score_dists_lpips'])}`; worst score `{fmt(oracle['worst_score_dists_lpips'])}`.",
        f"Oracle mean dPSNR: `{fmt(oracle['mean_delta_psnr'])}`; worst dPSNR `{fmt(oracle['worst_delta_psnr'])}`; negative PSNR images `{oracle['negative_psnr_count']}`.",
        f"Choices: `{oracle['choice_counts']}`.",
        "", "Interpretation:", "",
        "- A risk value should not be selected by dPSNR alone for this generative/perceptual branch.",
        "- If the no-op/active oracle is much better than any fixed risk, the next method component should be a reliability selector rather than a stronger always-on edit.",
        "- Exact delta bpp and decode checks must remain zero before moving this branch to full-training claims.",
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {prefix.with_suffix('.md')}, {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
