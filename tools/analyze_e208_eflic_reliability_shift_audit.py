
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402
import analyze_e193_eflic_reliability_head as e193  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-csv", type=Path, required=True)
    p.add_argument("--fit-manifest-csv", type=Path, required=True)
    p.add_argument("--split", action="append", default=[], help="name=csv; can repeat")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--feature-set", default="global_predecision_context")
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--positive-penalty", type=float, default=20.0)
    p.add_argument("--l2", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_rows(rows: list[dict[str, Any]], force: int | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if int(float(r.get("nonfinite", 0.0))) == 0]
    if force is not None and out and "force_ind" in out[0]:
        out = [r for r in out if int(float(r["force_ind"])) == force]
    return out


def delta(row: dict[str, Any], name: str) -> float:
    return float(row[f"active_delta_{name}"]) if f"active_delta_{name}" in row else float(row[f"delta_{name}"])


def score(row: dict[str, Any], args: argparse.Namespace) -> float:
    return args.dists_weight * delta(row, "dists") + args.lpips_weight * delta(row, "lpips") - args.psnr_weight * delta(row, "psnr")


def labels(rows: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    return np.asarray([1.0 if score(r, args) < 0.0 else 0.0 for r in rows], dtype=np.float64)


def values(rows: list[dict[str, Any]], feature: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[feature]))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def fmean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def fstd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else float("nan")


def qtile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def auc_score(prob: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(prob) & np.isfinite(y)
    prob = prob[mask]
    y = y[mask]
    pos = prob[y > 0.5]
    neg = prob[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x_raw = e193.feature_matrix(rows, model["features"])
    x = (x_raw - model["mean"]) / model["std"]
    return e193.sigmoid(x @ model["weights"] + float(model["bias"]))


def policy_summary(split: str, rows: list[dict[str, Any]], prob: np.ndarray, threshold: float, args: argparse.Namespace) -> dict[str, Any]:
    decisions = prob >= threshold
    y = labels(rows, args)
    return {
        "split": split,
        "rows": len(rows),
        "threshold": float(threshold),
        "prob_mean": fmean(prob),
        "prob_std": fstd(prob),
        "prob_q10": qtile(prob, 0.10),
        "prob_q50": qtile(prob, 0.50),
        "prob_q90": qtile(prob, 0.90),
        "active_good_rate": float(np.mean(y)) if len(y) else float("nan"),
        "both_good_rate": float(np.mean([(delta(r, "dists") < 0 and delta(r, "lpips") < 0) for r in rows])) if rows else float("nan"),
        "mean_active_delta_dists": float(np.mean([delta(r, "dists") for r in rows])) if rows else float("nan"),
        "mean_active_delta_lpips": float(np.mean([delta(r, "lpips") for r in rows])) if rows else float("nan"),
        "prob_auc_good": auc_score(prob, y),
        "prob_corr_score": pearson(prob, np.asarray([score(r, args) for r in rows], dtype=np.float64)),
        "branch_share": float(np.mean(decisions)) if len(decisions) else 0.0,
        "selected_delta_dists": float(np.mean([delta(r, "dists") if d else 0.0 for r, d in zip(rows, decisions)])) if rows else float("nan"),
        "selected_delta_lpips": float(np.mean([delta(r, "lpips") if d else 0.0 for r, d in zip(rows, decisions)])) if rows else float("nan"),
        "selected_delta_psnr": float(np.mean([delta(r, "psnr") if d else 0.0 for r, d in zip(rows, decisions)])) if rows else float("nan"),
        "selected_good": int(sum(score(r, args) < 0 and bool(d) for r, d in zip(rows, decisions))),
        "selected_bad": int(sum(score(r, args) >= 0 and bool(d) for r, d in zip(rows, decisions))),
        "missed_good": int(sum(score(r, args) < 0 and not bool(d) for r, d in zip(rows, decisions))),
    }


def feature_shift_rows(splits: dict[str, list[dict[str, Any]]], features: list[str], coeffs: dict[str, float], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    fit_rows = splits["fit"]
    fit_stats = {f: (fmean(values(fit_rows, f)), fstd(values(fit_rows, f))) for f in features}
    for split, rows in splits.items():
        ys = np.asarray([score(r, args) for r in rows], dtype=np.float64)
        for f in features:
            x = values(rows, f)
            mean = fmean(x)
            std = fstd(x)
            fit_mean, fit_std = fit_stats[f]
            pooled = math.sqrt(max(1e-12, (std * std + fit_std * fit_std) * 0.5)) if math.isfinite(std) and math.isfinite(fit_std) else float("nan")
            shift = (mean - fit_mean) / pooled if math.isfinite(pooled) and pooled > 0 else float("nan")
            out.append({
                "split": split,
                "feature": f,
                "coefficient": float(coeffs.get(f, 0.0)),
                "abs_coefficient": abs(float(coeffs.get(f, 0.0))),
                "mean": mean,
                "std": std,
                "fit_mean": fit_mean,
                "fit_std": fit_std,
                "std_mean_shift_vs_fit": shift,
                "abs_std_mean_shift_vs_fit": abs(shift) if math.isfinite(shift) else float("nan"),
                "corr_with_active_score": pearson(x, ys),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    fit_rows = finite_rows(e184.read_rows(args.fit_csv), args.force)
    if not fit_rows:
        raise SystemExit("no finite fit rows")
    manifest = e184.read_manifest(args.fit_manifest_csv)
    feature_sets = e185.feature_sets(manifest)
    if args.feature_set not in feature_sets:
        raise SystemExit(f"unknown feature set {args.feature_set}")
    features = e184.valid_features(fit_rows, feature_sets[args.feature_set][0])
    if not features:
        raise SystemExit("no valid features")
    model = e193.fit_head(fit_rows, features, args)
    coeff_rows = e193.top_coefficients(model, max(args.topk, len(features)))
    coeffs = {r["feature"]: float(r["coefficient"]) for r in coeff_rows}
    splits: dict[str, list[dict[str, Any]]] = {"fit": fit_rows}
    for spec in args.split:
        if "=" not in spec:
            raise SystemExit(f"bad split spec {spec}")
        name, raw = spec.split("=", 1)
        rows = finite_rows(read_csv_rows(Path(raw)), None)
        if rows:
            missing = [f for f in model["features"] if f not in rows[0]]
            if missing:
                raise SystemExit(f"split {name} missing features: {missing[:5]}")
        splits[name] = rows
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, rows in splits.items():
        prob = predict(model, rows)
        summary.append(policy_summary(name, rows, prob, float(model["threshold"]), args))
    shift_all = feature_shift_rows(splits, model["features"], coeffs, args)
    top_features = {r["feature"] for r in coeff_rows[: args.topk]}
    shift_top = [r for r in shift_all if r["feature"] in top_features]
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)
    write_csv(args.output_prefix.with_suffix(".feature_shift.csv"), shift_all)
    write_csv(args.output_prefix.with_suffix(".top_feature_shift.csv"), shift_top)
    payload = {
        "fit_csv": str(args.fit_csv),
        "fit_rows": len(fit_rows),
        "feature_set": args.feature_set,
        "head_threshold": float(model["threshold"]),
        "summary": summary,
        "top_coefficients": coeff_rows[: args.topk],
        "interpretation": "Distribution-shift audit for EF-LIC reliability transfer.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    by_split_top_drift = {}
    for name in splits:
        rows = [r for r in shift_all if r["split"] == name and name != "fit"]
        rows.sort(key=lambda r: float(r["abs_std_mean_shift_vs_fit"]) if math.isfinite(float(r["abs_std_mean_shift_vs_fit"])) else -1.0, reverse=True)
        by_split_top_drift[name] = rows[: args.topk]
    lines = [
        "# E208 EF-LIC Reliability Transfer Shift Audit",
        "",
        f"Fit CSV: {args.fit_csv}",
        f"Feature set: {args.feature_set}",
        f"Head threshold: {float(model['threshold']):.6f}",
        "",
        "This audit compares head probabilities, active-good labels, and decoder-side feature distributions across fit/calibration/eval splits.",
        "",
        "| split | rows | active-good | both-good | prob mean | prob q10/q50/q90 | AUC good | corr(prob, score) | branch | dDISTS | dLPIPS | good/bad/missed |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['split']} | {r['rows']} | {r['active_good_rate']:.3f} | {r['both_good_rate']:.3f} | "
            f"{r['prob_mean']:.3f} | {r['prob_q10']:.3f}/{r['prob_q50']:.3f}/{r['prob_q90']:.3f} | "
            f"{r['prob_auc_good']:.3f} | {r['prob_corr_score']:.3f} | {r['branch_share']:.3f} | "
            f"{r['selected_delta_dists']:+.6f} | {r['selected_delta_lpips']:+.6f} | {r['selected_good']}/{r['selected_bad']}/{r['missed_good']} |"
        )
    lines.extend(["", "Top coefficients:", "", "| feature | coefficient |", "|---|---:|"])
    for r in coeff_rows[: args.topk]:
        lines.append(f"| {r['feature']} | {float(r['coefficient']):+.6f} |")
    for name, rows in by_split_top_drift.items():
        if name == "fit" or not rows:
            continue
        lines.extend(["", f"Top feature mean shifts vs fit: {name}", "", "| feature | coeff | mean | fit mean | std shift | corr(score) |", "|---|---:|---:|---:|---:|---:|"])
        for r in rows[:10]:
            lines.append(
                f"| {r['feature']} | {float(r['coefficient']):+.6f} | {float(r['mean']):+.6f} | "
                f"{float(r['fit_mean']):+.6f} | {float(r['std_mean_shift_vs_fit']):+.3f} | {float(r['corr_with_active_score']):+.3f} |"
            )
    lines.extend(["", "Guardrails:", "", "- AUC near 0.5 means the probability score is not ranking active-good rows in that split.", "- Large standardized feature shifts indicate domain or calibration mismatch even when bitstream checks pass.", "- This audit is diagnostic and does not choose a paper-facing threshold by itself."])
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
