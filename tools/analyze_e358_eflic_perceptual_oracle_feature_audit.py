#!/usr/bin/env python3
"""Audit EF-LIC perceptual oracle choices against decoder-visible HCG features.

This is a selector-design diagnostic.  It reads candidate codec-loop CSVs for
fixed HCG risk strengths, adds a no-op candidate, defines the oracle by
perceptual score (DISTS + w * LPIPS), then reports dataset/split summaries and
feature correlations.  PSNR is intentionally not part of the selection score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GLOBAL_FEATURES = [
    "y_gate_mean",
    "y_gate_max",
    "y_alpha_active_frac",
    "y_alpha_mean",
    "y_alpha_max",
    "y_avg_geometry_delta_rms",
    "y_avg_index_entropy",
    "y_avg_index_used_frac",
    "y_avg_residual_error_rms",
    "y_family_zero_prob_mean",
    "y_local_score_mean",
    "y_mismatch",
    "y_risk_score_mean",
    "y_strength_mean",
    "z_hat_abs_mean",
    "z_hat_rms",
    "z_hat_std",
    "z_index_entropy",
    "z_index_perplexity",
    "z_index_used_frac",
    "z_mismatch",
]

SLICE_METRICS = [
    "gate_mean",
    "gate_max",
    "alpha_active_frac",
    "alpha_mean",
    "alpha_max",
    "avg_geometry_delta_rms",
    "avg_index_entropy",
    "avg_index_used_frac",
    "avg_residual_error_rms",
    "family_zero_prob_mean",
    "local_score_mean",
    "risk_score_mean",
    "strength_mean",
    "stage0_geometry_delta_rms",
    "stage0_index_entropy",
    "stage0_index_perplexity",
    "stage0_index_used_frac",
    "stage0_residual_error_rms",
]
FEATURES = GLOBAL_FEATURES + [f"slice{s}_{m}" for s in range(4) for m in SLICE_METRICS]


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("run must be name=csv")
    name, raw = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("empty run name")
    return name, Path(raw)


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            rows[row["image"]] = row
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def score(row: dict[str, Any], lpips_weight: float) -> float:
    return f(row, "delta_dists") + lpips_weight * f(row, "delta_lpips")


def candidate(image: str, risk: str, row: dict[str, str] | None, lpips_weight: float) -> dict[str, Any]:
    if row is None:
        return {
            "image": image,
            "risk": "noop",
            "source_dataset": "unknown",
            "source_split": "unknown",
            "score": 0.0,
            "delta_lpips": 0.0,
            "delta_dists": 0.0,
            "delta_ms_ssim": 0.0,
            "delta_bpp": 0.0,
            "delta_psnr": 0.0,
            "nonfinite": 0,
            "max_decode_diff": 0.0,
        }
    out = dict(row)
    out.update(
        {
            "risk": risk,
            "score": score(row, lpips_weight),
            "delta_lpips": f(row, "delta_lpips"),
            "delta_dists": f(row, "delta_dists"),
            "delta_ms_ssim": f(row, "delta_ms_ssim"),
            "delta_bpp": f(row, "delta_bpp"),
            "delta_psnr": f(row, "delta_psnr"),
            "nonfinite": int(f(row, "nonfinite")),
            "max_decode_diff": f(row, "max_decode_diff"),
        }
    )
    return out


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xarr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    yarr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(xarr.std()) <= 1e-12 or float(yarr.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xarr, yarr)[0, 1])


def auc(values: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(v, l) for v, l in zip(values, labels) if math.isfinite(v)]
    pos = [v for v, l in pairs if l]
    neg = [v for v, l in pairs if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "n": len(rows),
        "score": finite_mean(float(r["score"]) for r in rows),
        "worst_score": max((float(r["score"]) for r in rows), default=float("nan")),
        "delta_lpips": finite_mean(float(r["delta_lpips"]) for r in rows),
        "delta_dists": finite_mean(float(r["delta_dists"]) for r in rows),
        "delta_ms_ssim": finite_mean(float(r["delta_ms_ssim"]) for r in rows),
        "delta_bpp": finite_mean(float(r["delta_bpp"]) for r in rows),
        "delta_psnr_diag": finite_mean(float(r["delta_psnr"]) for r in rows),
        "wins": sum(float(r["score"]) < 0.0 for r in rows),
        "choices": dict(Counter(str(r["risk"]) for r in rows)),
        "nonfinite": sum(int(r["nonfinite"]) for r in rows),
        "max_decode_diff": max((float(r["max_decode_diff"]) for r in rows), default=float("nan")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, signed: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:+.6f}" if signed else f"{val:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_run, required=True)
    ap.add_argument("--reference-risk", default=None)
    ap.add_argument("--lpips-weight", type=float, default=3.0)
    ap.add_argument("--output-prefix", type=Path, required=True)
    args = ap.parse_args()

    runs = {name: read_csv(path) for name, path in args.run}
    risks = list(runs)
    ref_risk = args.reference_risk or risks[0]
    images = sorted(set.intersection(*(set(rows) for rows in runs.values())))

    per_image: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for image in images:
        candidates = [candidate(image, "noop", None, args.lpips_weight)]
        candidates += [candidate(image, risk, runs[risk][image], args.lpips_weight) for risk in risks]
        oracle = min(candidates, key=lambda r: (float(r["score"]), str(r["risk"])))
        ref = runs[ref_risk][image]
        source_dataset = str(ref.get("source_dataset", "unknown"))
        source_split = str(ref.get("source_split", "unknown"))
        row = {
            "image": image,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "oracle_risk": oracle["risk"],
            "oracle_score": oracle["score"],
            "oracle_active": int(str(oracle["risk"]) != "noop"),
            "oracle_delta_lpips": oracle["delta_lpips"],
            "oracle_delta_dists": oracle["delta_dists"],
            "oracle_delta_ms_ssim": oracle["delta_ms_ssim"],
            "oracle_delta_bpp": oracle["delta_bpp"],
            "oracle_delta_psnr_diag": oracle["delta_psnr"],
        }
        for risk in risks:
            cand = candidate(image, risk, runs[risk][image], args.lpips_weight)
            row[f"{risk}_score"] = cand["score"]
            row[f"{risk}_minus_oracle"] = cand["score"] - oracle["score"]
        per_image.append(row)
        feature_rows.append({**row, **{feat: f(ref, feat, float("nan")) for feat in FEATURES}})

    for group_name, pred in [
        ("all", lambda r: True),
        ("cal", lambda r: r["source_split"] == "cal"),
        ("eval", lambda r: r["source_split"] == "eval"),
        ("clic_pro", lambda r: r["source_dataset"] == "clic_pro"),
        ("kodak24", lambda r: r["source_dataset"] == "kodak24"),
        ("eval_clic_pro", lambda r: r["source_split"] == "eval" and r["source_dataset"] == "clic_pro"),
    ]:
        group_images = [r["image"] for r in per_image if pred(r)]
        if not group_images:
            continue
        oracle_rows = []
        fixed_rows_by_risk: dict[str, list[dict[str, Any]]] = {risk: [] for risk in risks}
        noop_rows = []
        for image in group_images:
            noop_rows.append(candidate(image, "noop", None, args.lpips_weight))
            cands = [candidate(image, "noop", None, args.lpips_weight)]
            cands += [candidate(image, risk, runs[risk][image], args.lpips_weight) for risk in risks]
            oracle_rows.append(min(cands, key=lambda r: (float(r["score"]), str(r["risk"]))))
            for risk in risks:
                fixed_rows_by_risk[risk].append(candidate(image, risk, runs[risk][image], args.lpips_weight))
        summaries.append(summarize(noop_rows, f"{group_name}:fixed_noop"))
        for risk, rows in fixed_rows_by_risk.items():
            summaries.append(summarize(rows, f"{group_name}:fixed_{risk}"))
        summaries.append(summarize(oracle_rows, f"{group_name}:oracle"))

    labels = [int(r["oracle_active"]) for r in feature_rows]
    oracle_scores = [float(r["oracle_score"]) for r in feature_rows]
    audit: list[dict[str, Any]] = []
    for feat in FEATURES:
        vals = [float(r.get(feat, float("nan"))) for r in feature_rows]
        corr_score = pearson(vals, oracle_scores)
        auc_active = auc(vals, labels)
        auc_active_inv = 1.0 - auc_active if math.isfinite(auc_active) else float("nan")
        audit.append(
            {
                "feature": feat,
                "pearson_with_oracle_score": corr_score,
                "auc_oracle_active_high": auc_active,
                "auc_oracle_active_low": auc_active_inv,
                "best_auc_abs": max(
                    abs(auc_active - 0.5) if math.isfinite(auc_active) else float("nan"),
                    abs(auc_active_inv - 0.5) if math.isfinite(auc_active_inv) else float("nan"),
                ),
            }
        )
    audit_sorted_corr = sorted(audit, key=lambda r: abs(float(r["pearson_with_oracle_score"])), reverse=True)
    audit_sorted_auc = sorted(audit, key=lambda r: float(r["best_auc_abs"]), reverse=True)

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".per_image.csv"), per_image)
    write_csv(prefix.with_suffix(".feature_rows.csv"), feature_rows)
    write_csv(prefix.with_suffix(".feature_audit.csv"), audit)
    write_csv(prefix.with_suffix(".summaries.csv"), summaries)
    payload = {
        "num_images": len(images),
        "risks": risks,
        "reference_risk": ref_risk,
        "lpips_weight": args.lpips_weight,
        "summaries": summaries,
        "top_abs_corr": audit_sorted_corr[:20],
        "top_active_auc": audit_sorted_auc[:20],
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# E358 EF-LIC Perceptual Oracle Feature Audit",
        "",
        "Score is `delta_DISTS + w * delta_LPIPS`; lower is better. PSNR is diagnostic only.",
        f"LPIPS weight: `{args.lpips_weight}`. Reference feature risk: `{ref_risk}`.",
        "",
        "## Summary",
        "",
        "| label | n | score | worst | dLPIPS | dDISTS | dMS-SSIM | dBPP | dPSNR diag | wins | choices | nonfinite | decode max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summaries:
        md.append(
            f"| {row['label']} | {row['n']} | {fmt(row['score'], True)} | {fmt(row['worst_score'], True)} | "
            f"{fmt(row['delta_lpips'], True)} | {fmt(row['delta_dists'], True)} | {fmt(row['delta_ms_ssim'], True)} | "
            f"{fmt(row['delta_bpp'], True)} | {fmt(row['delta_psnr_diag'], True)} | {row['wins']} | `{row['choices']}` | "
            f"{row['nonfinite']} | {float(row['max_decode_diff']):.3e} |"
        )
    md.extend([
        "",
        "## Top Feature Correlations With Oracle Score",
        "",
        "| feature | pearson |",
        "|---|---:|",
    ])
    for row in audit_sorted_corr[:15]:
        md.append(f"| {row['feature']} | {fmt(row['pearson_with_oracle_score'], True)} |")
    md.extend([
        "",
        "## Top Oracle-Active Feature AUCs",
        "",
        "| feature | AUC high-active | AUC low-active |",
        "|---|---:|---:|",
    ])
    for row in audit_sorted_auc[:15]:
        md.append(
            f"| {row['feature']} | {fmt(row['auc_oracle_active_high'])} | {fmt(row['auc_oracle_active_low'])} |"
        )
    md.extend([
        "",
        "Interpretation:",
        "",
        "- Oracle headroom is usable only if a deployable controller can approach it without using evaluation labels.",
        "- If global/slice summary features weakly separate oracle-active cases, the next design should be local or sequential rather than image-level.",
        "- PSNR is not used to select policies; it is retained only as a codec-health diagnostic.",
    ])
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {prefix.with_suffix('.md')}, {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
