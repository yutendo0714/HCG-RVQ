#!/usr/bin/env python3
"""Audit PSNR-era EF-LIC teacher signals under perceptual metrics.

E346/E347 were built around PSNR codec-gain margins.  For the generative
low-bitrate branch, PSNR is only a diagnostic, so this script checks two things:

1. how often fixed HCG risks look good by PSNR but bad by DISTS/LPIPS;
2. whether the E318/E346 PSNR teacher features explain the Kodak perceptual
   oracle well enough to reuse them for the next local controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_risk_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("risk spec must be name=csv")
    name, raw = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("empty risk name")
    return name, Path(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--e318-slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    p.add_argument(
        "--e346-manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e346_eflic_codec_gain_context_teacher_kodak24/manifest_kodak24_n24.csv",
    )
    p.add_argument(
        "--kodak-oracle",
        type=Path,
        default=ROOT / "experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_risk_grid_oracle_choices.csv",
    )
    p.add_argument(
        "--kodak-risk",
        action="append",
        type=parse_risk_spec,
        default=[
            (
                "riskm060",
                ROOT / "experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_riskm060.csv",
            ),
            (
                "riskm080",
                ROOT / "experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_riskm080.csv",
            ),
            (
                "riskm100",
                ROOT / "experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_riskm100.csv",
            ),
        ],
    )
    p.add_argument(
        "--clic-risk",
        action="append",
        type=parse_risk_spec,
        default=[
            (
                "riskm060",
                ROOT / "experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm060.csv",
            ),
            (
                "riskm080",
                ROOT / "experiments/analysis/e350_eflic_e347_perceptual_metric_clicpro41_riskm080.csv",
            ),
            (
                "riskm100",
                ROOT / "experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm100.csv",
            ),
        ],
    )
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--cal-count", type=int, default=16)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e361_eflic_psnr_teacher_perceptual_audit",
    )
    return p.parse_args()


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def read_by_image(path: Path) -> dict[str, dict[str, str]]:
    return {row["image"]: row for row in read_rows(path)}


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xarr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    yarr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(xarr.std()) <= 1e-12 or float(yarr.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xarr, yarr)[0, 1])


def auc(values: list[float], labels: list[int]) -> float:
    pairs = [(v, l) for v, l in zip(values, labels) if math.isfinite(v)]
    pos = [v for v, l in pairs if l]
    neg = [v for v, l in pairs if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for pval in pos:
        for nval in neg:
            if pval > nval:
                wins += 1.0
            elif pval == nval:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def score(row: dict[str, Any], lpips_weight: float) -> float:
    return f(row, "delta_dists", 0.0) + lpips_weight * f(row, "delta_lpips", 0.0)


def fixed_risk_summary(dataset: str, risk_name: str, rows: dict[str, dict[str, str]], lpips_weight: float) -> dict[str, Any]:
    items = []
    for row in rows.values():
        dpsnr = f(row, "delta_psnr")
        dms = f(row, "delta_ms_ssim")
        dlpips = f(row, "delta_lpips")
        ddists = f(row, "delta_dists")
        pscore = ddists + lpips_weight * dlpips
        items.append(
            {
                "image": row["image"],
                "score": pscore,
                "delta_psnr": dpsnr,
                "delta_ms_ssim": dms,
                "delta_lpips": dlpips,
                "delta_dists": ddists,
                "delta_bpp": f(row, "delta_bpp", 0.0),
                "max_decode_diff": f(row, "max_decode_diff", 0.0),
                "nonfinite": int(f(row, "nonfinite", 0.0)),
            }
        )
    return {
        "dataset": dataset,
        "risk": risk_name,
        "n": len(items),
        "mean_score": finite_mean([r["score"] for r in items]),
        "worst_score": max((r["score"] for r in items), default=float("nan")),
        "mean_delta_psnr": finite_mean([r["delta_psnr"] for r in items]),
        "worst_delta_psnr": min((r["delta_psnr"] for r in items), default=float("nan")),
        "mean_delta_ms_ssim": finite_mean([r["delta_ms_ssim"] for r in items]),
        "mean_delta_lpips": finite_mean([r["delta_lpips"] for r in items]),
        "mean_delta_dists": finite_mean([r["delta_dists"] for r in items]),
        "score_win_count": sum(r["score"] < 0.0 for r in items),
        "psnr_win_count": sum(r["delta_psnr"] > 0.0 for r in items),
        "psnr_win_score_loss_count": sum(r["delta_psnr"] > 0.0 and r["score"] > 0.0 for r in items),
        "psnr_loss_score_win_count": sum(r["delta_psnr"] < 0.0 and r["score"] < 0.0 for r in items),
        "triple_perceptual_win_count": sum(
            r["delta_ms_ssim"] > 0.0 and r["delta_lpips"] < 0.0 and r["delta_dists"] < 0.0 for r in items
        ),
        "max_abs_delta_bpp": max((abs(r["delta_bpp"]) for r in items), default=float("nan")),
        "max_decode_diff": max((r["max_decode_diff"] for r in items), default=float("nan")),
        "nonfinite_rows": sum(r["nonfinite"] for r in items),
    }


def aggregate_e318(slice_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in slice_rows:
        grouped[row["image"]].append(row)
    out: dict[str, dict[str, Any]] = {}
    for image, rows in grouped.items():
        margins = [f(row, "contextual_margin_psnr") for row in rows]
        single = [f(row, "single_delta_psnr") for row in rows]
        risk_scores = [f(row, "single_risk_score_mean") for row in rows]
        residual = [f(row, "single_avg_residual_error_rms") for row in rows]
        index_entropy = [f(row, "single_avg_index_entropy") for row in rows]
        out[image] = {
            "e318_contextual_margin_mean": finite_mean(margins),
            "e318_contextual_margin_max": max(margins),
            "e318_contextual_margin_min": min(margins),
            "e318_contextual_positive_frac": finite_mean([f(row, "contextual_positive", 0.0) for row in rows]),
            "e318_oracle_active_frac": finite_mean([f(row, "oracle_active", 0.0) for row in rows]),
            "e318_single_positive_frac": finite_mean([f(row, "single_positive", 0.0) for row in rows]),
            "e318_single_delta_psnr_mean": finite_mean(single),
            "e318_single_delta_psnr_max": max(single),
            "e318_best_gain_over_all": max(f(row, "best_gain_over_all", 0.0) for row in rows),
            "e318_single_risk_score_mean": finite_mean(risk_scores),
            "e318_residual_error_rms_mean": finite_mean(residual),
            "e318_index_entropy_mean": finite_mean(index_entropy),
        }
    return out


def build_joined_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    e318 = aggregate_e318(read_rows(args.e318_slice_labels))
    e346 = read_by_image(args.e346_manifest)
    oracle = read_by_image(args.kodak_oracle)
    risks = {name: read_by_image(path) for name, path in args.kodak_risk}
    images = sorted(set(e318) & set(e346) & set(oracle) & set.intersection(*(set(rows) for rows in risks.values())))
    rows: list[dict[str, Any]] = []
    for image in images:
        o = oracle[image]
        row: dict[str, Any] = {
            "image": image,
            **e318[image],
            "e346_active_frac": f(e346[image], "active_frac"),
            "e346_active_slice_count": f(e346[image], "active_slice_count"),
            "e346_alpha_mean": f(e346[image], "alpha_mean"),
            "e346_risk_mean": f(e346[image], "risk_mean"),
            "e346_risk_min": f(e346[image], "risk_min"),
            "e346_risk_max": f(e346[image], "risk_max"),
            "oracle_risk": o["risk"],
            "oracle_active": int(o["risk"] != "noop"),
            "oracle_score": f(o, "score_dists_lpips"),
            "oracle_delta_psnr_diag": f(o, "delta_psnr"),
            "oracle_delta_ms_ssim": f(o, "delta_ms_ssim"),
            "oracle_delta_lpips": f(o, "delta_lpips"),
            "oracle_delta_dists": f(o, "delta_dists"),
        }
        for name, by_image in risks.items():
            r = by_image[image]
            row[f"{name}_score"] = score(r, args.lpips_weight)
            row[f"{name}_delta_psnr"] = f(r, "delta_psnr")
            row[f"{name}_delta_ms_ssim"] = f(r, "delta_ms_ssim")
            row[f"{name}_delta_lpips"] = f(r, "delta_lpips")
            row[f"{name}_delta_dists"] = f(r, "delta_dists")
        rows.append(row)
    return rows


def feature_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "e318_contextual_margin_mean",
        "e318_contextual_margin_max",
        "e318_contextual_margin_min",
        "e318_contextual_positive_frac",
        "e318_oracle_active_frac",
        "e318_single_positive_frac",
        "e318_single_delta_psnr_mean",
        "e318_single_delta_psnr_max",
        "e318_best_gain_over_all",
        "e318_single_risk_score_mean",
        "e318_residual_error_rms_mean",
        "e318_index_entropy_mean",
        "e346_active_frac",
        "e346_active_slice_count",
        "e346_alpha_mean",
        "e346_risk_mean",
        "e346_risk_min",
        "e346_risk_max",
        "neg_e346_risk_mean",
        "neg_e346_risk_min",
    ]
    labels = [int(row["oracle_active"]) for row in rows]
    target_score = [float(row["oracle_score"]) for row in rows]
    out: list[dict[str, Any]] = []
    for feature in features:
        if feature == "neg_e346_risk_mean":
            vals = [-float(row["e346_risk_mean"]) for row in rows]
        elif feature == "neg_e346_risk_min":
            vals = [-float(row["e346_risk_min"]) for row in rows]
        else:
            vals = [float(row.get(feature, float("nan"))) for row in rows]
        out.append(
            {
                "feature": feature,
                "corr_with_oracle_score_lower_better": pearson(vals, target_score),
                "corr_with_oracle_gain_positive": pearson(vals, [-v for v in target_score]),
                "auc_oracle_active_high_feature": auc(vals, labels),
                "mean_active": finite_mean([v for v, label in zip(vals, labels) if label]),
                "mean_inactive": finite_mean([v for v, label in zip(vals, labels) if not label]),
            }
        )
    return sorted(out, key=lambda r: abs(float(r["corr_with_oracle_gain_positive"])), reverse=True)


def threshold_transfer(rows: list[dict[str, Any]], risk_names: list[str], cal_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = [
        "e318_contextual_margin_mean",
        "e318_contextual_margin_max",
        "e318_contextual_positive_frac",
        "e318_oracle_active_frac",
        "e318_single_positive_frac",
        "e318_best_gain_over_all",
        "e346_active_frac",
        "e346_risk_mean",
        "e346_risk_min",
        "neg_e346_risk_mean",
        "neg_e346_risk_min",
    ]
    sorted_rows = sorted(rows, key=lambda r: r["image"])
    cal = sorted_rows[:cal_count]
    eval_rows = sorted_rows[cal_count:]
    policies: list[dict[str, Any]] = []

    def value(row: dict[str, Any], feature: str) -> float:
        if feature == "neg_e346_risk_mean":
            return -float(row["e346_risk_mean"])
        if feature == "neg_e346_risk_min":
            return -float(row["e346_risk_min"])
        return float(row[feature])

    for feature in features:
        thresholds = sorted({value(row, feature) for row in cal if math.isfinite(value(row, feature))})
        if not thresholds:
            continue
        for direction in ("ge", "le"):
            for threshold in thresholds:
                for risk in risk_names:
                    def choose_score(row: dict[str, Any]) -> float:
                        val = value(row, feature)
                        active = val >= threshold if direction == "ge" else val <= threshold
                        return float(row[f"{risk}_score"]) if active else 0.0

                    cal_scores = [choose_score(row) for row in cal]
                    eval_scores = [choose_score(row) for row in eval_rows]
                    policies.append(
                        {
                            "policy": f"{risk}_if_{feature}_{direction}_{threshold:.8g}",
                            "risk": risk,
                            "feature": feature,
                            "direction": direction,
                            "threshold": threshold,
                            "cal_score": finite_mean(cal_scores),
                            "cal_worst_score": max(cal_scores) if cal_scores else float("nan"),
                            "cal_wins": sum(v < 0.0 for v in cal_scores),
                            "cal_active_frac": finite_mean([
                                1.0 if ((value(row, feature) >= threshold) if direction == "ge" else (value(row, feature) <= threshold)) else 0.0
                                for row in cal
                            ]),
                            "eval_score": finite_mean(eval_scores),
                            "eval_worst_score": max(eval_scores) if eval_scores else float("nan"),
                            "eval_wins": sum(v < 0.0 for v in eval_scores),
                            "eval_active_frac": finite_mean([
                                1.0 if ((value(row, feature) >= threshold) if direction == "ge" else (value(row, feature) <= threshold)) else 0.0
                                for row in eval_rows
                            ]),
                        }
                    )
    policies.sort(key=lambda r: (float(r["cal_score"]), float(r["cal_worst_score"])))
    fixed_eval = {
        risk: finite_mean([float(row[f"{risk}_score"]) for row in eval_rows]) for risk in risk_names
    }
    oracle_eval = finite_mean([float(row["oracle_score"]) for row in eval_rows])
    summary = {
        "cal_count": len(cal),
        "eval_count": len(eval_rows),
        "top_cal_policy": policies[0] if policies else {},
        "top_eval_diagnostic_policy": sorted(policies, key=lambda r: (float(r["eval_score"]), float(r["eval_worst_score"])))[0] if policies else {},
        "fixed_eval_scores": fixed_eval,
        "oracle_eval_score": oracle_eval,
    }
    return policies, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    kodak_risks = {name: read_by_image(path) for name, path in args.kodak_risk}
    clic_risks = {name: read_by_image(path) for name, path in args.clic_risk}

    fixed_summaries: list[dict[str, Any]] = []
    for name, rows in kodak_risks.items():
        fixed_summaries.append(fixed_risk_summary("kodak24", name, rows, args.lpips_weight))
    for name, rows in clic_risks.items():
        fixed_summaries.append(fixed_risk_summary("clicpro41", name, rows, args.lpips_weight))

    joined = build_joined_rows(args)
    audit = feature_audit(joined)
    policies, transfer_summary = threshold_transfer(joined, list(kodak_risks), args.cal_count)
    fixed_choice_counts = dict(Counter(row["oracle_risk"] for row in joined))
    payload = {
        "experiment": "E361 EF-LIC PSNR teacher perceptual audit",
        "purpose": "Check PSNR-era fixed-risk and local teacher signals under the perceptual objective.",
        "lpips_weight": args.lpips_weight,
        "fixed_summaries": fixed_summaries,
        "kodak_joined_rows": len(joined),
        "kodak_oracle_choice_counts": fixed_choice_counts,
        "feature_audit": audit,
        "threshold_transfer_summary": transfer_summary,
        "interpretation": {
            "psnr_conflict": "PSNR-positive fixed-risk rows can be perceptually harmful, so PSNR cannot select the paper method.",
            "teacher_reuse": "E318/E346 PSNR teacher signals are only reusable if they align with perceptual oracle features or threshold transfer; otherwise they should be initialization/diagnostic, not the final teacher.",
        },
    }
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    rows_path = args.output_prefix.with_suffix(".joined.csv")
    audit_path = args.output_prefix.with_suffix(".feature_audit.csv")
    policy_path = args.output_prefix.with_suffix(".threshold_policies.csv")
    summary_path = args.output_prefix.with_suffix(".fixed_summary.csv")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(rows_path, joined)
    write_csv(audit_path, audit)
    write_csv(policy_path, policies)
    write_csv(summary_path, fixed_summaries)

    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E361 EF-LIC PSNR Teacher Perceptual Audit\n\n")
        fobj.write("This audit checks whether PSNR-era EF-LIC HCG evidence survives the perceptual metric correction. The paper-facing score is `delta_DISTS + 3 * delta_LPIPS`; lower is better. PSNR is diagnostic only.\n\n")
        fobj.write("## Fixed-Risk Metric Conflicts\n\n")
        fobj.write("| dataset | risk | n | score | dPSNR diag | score wins | PSNR wins | PSNR win / score loss | PSNR loss / score win | triple perceptual wins |\n")
        fobj.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in fixed_summaries:
            fobj.write(
                "| {dataset} | {risk} | {n} | {score} | {dpsnr} | {sw} | {pw} | {conf1} | {conf2} | {triple} |\n".format(
                    dataset=row["dataset"],
                    risk=row["risk"],
                    n=row["n"],
                    score=fmt(row["mean_score"], signed=True),
                    dpsnr=fmt(row["mean_delta_psnr"], signed=True),
                    sw=row["score_win_count"],
                    pw=row["psnr_win_count"],
                    conf1=row["psnr_win_score_loss_count"],
                    conf2=row["psnr_loss_score_win_count"],
                    triple=row["triple_perceptual_win_count"],
                )
            )
        fobj.write("\n## Kodak PSNR-Teacher vs Perceptual Oracle\n\n")
        fobj.write(f"- Joined Kodak rows: `{len(joined)}`\n")
        fobj.write(f"- Oracle choices: `{fixed_choice_counts}`\n")
        top = transfer_summary.get("top_cal_policy", {})
        diag = transfer_summary.get("top_eval_diagnostic_policy", {})
        fobj.write(f"- Best calibration-selected threshold: `{top.get('policy', 'none')}` with cal score `{fmt(top.get('cal_score', float('nan')), signed=True)}` and eval score `{fmt(top.get('eval_score', float('nan')), signed=True)}`\n")
        fobj.write(f"- Best held diagnostic threshold: `{diag.get('policy', 'none')}` with eval score `{fmt(diag.get('eval_score', float('nan')), signed=True)}` (not selectable as evidence)\n")
        fobj.write(f"- Fixed eval scores after image {args.cal_count}: `{transfer_summary.get('fixed_eval_scores', {})}`\n")
        fobj.write(f"- Oracle eval score after image {args.cal_count}: `{fmt(transfer_summary.get('oracle_eval_score', float('nan')), signed=True)}`\n\n")
        fobj.write("Top feature alignments with perceptual oracle gain:\n\n")
        fobj.write("| feature | corr gain | corr score | AUC active | mean active | mean inactive |\n")
        fobj.write("|---|---:|---:|---:|---:|---:|\n")
        for row in audit[:12]:
            fobj.write(
                "| {feature} | {gain} | {score} | {aucv} | {active} | {inactive} |\n".format(
                    feature=row["feature"],
                    gain=fmt(row["corr_with_oracle_gain_positive"], signed=True),
                    score=fmt(row["corr_with_oracle_score_lower_better"], signed=True),
                    aucv=fmt(row["auc_oracle_active_high_feature"]),
                    active=fmt(row["mean_active"], signed=True),
                    inactive=fmt(row["mean_inactive"], signed=True),
                )
            )
        fobj.write("\n## Interpretation\n\n")
        fobj.write("- Some PSNR-positive rows are perceptually harmful, so the old PSNR-based story is not safe for the generative branch.\n")
        fobj.write("- E318/E346 PSNR teacher signals should be reused only as decoder-safe context initialization/diagnostics unless they transfer under the perceptual split.\n")
        fobj.write("- The next controller should be trained from perceptual codec-gain labels, using PSNR only as a tail-health constraint.\n")
    print(f"wrote {json_path}, {md_path}, {rows_path}, {audit_path}, {policy_path}, {summary_path}")


if __name__ == "__main__":
    main()
