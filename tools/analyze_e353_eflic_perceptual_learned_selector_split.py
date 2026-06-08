#!/usr/bin/env python3
"""Train/evaluate a small EF-LIC HCG perceptual risk selector on a split.

This is a selector-design audit, not a final paper claim.  It only uses
candidate-local features recorded by the decoder-side HCG branch, predicts the
perceptual score for each candidate, and then selects the lowest predicted score
among noop and the available HCG risk strengths.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


METRIC_KEYS = [
    "delta_psnr",
    "delta_ms_ssim",
    "delta_lpips",
    "delta_dists",
    "delta_bpp",
    "max_decode_diff",
    "nonfinite",
]

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

NOOP = {
    "risk": "noop",
    "delta_psnr": 0.0,
    "delta_ms_ssim": 0.0,
    "delta_lpips": 0.0,
    "delta_dists": 0.0,
    "delta_bpp": 0.0,
    "max_decode_diff": 0.0,
    "nonfinite": 0,
    "score": 0.0,
}


@dataclass(frozen=True)
class Model:
    name: str
    risk_names: List[str]
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    weights: List[List[float]]
    lambda_value: float
    train_mode: str


def f(row: dict, key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in (None, ""):
        return default
    return float(val)


def parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("run must be name=csv")
    name, path = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name is empty")
    return name, Path(path)


def risk_value(name: str) -> float:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return 0.0
    return -float(digits) / 1000.0


def read_rows(path: Path, risk: str, lpips_weight: float) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            r = dict(row)
            r["risk"] = risk
            for key in METRIC_KEYS:
                r[key] = int(f(row, key)) if key == "nonfinite" else f(row, key)
            r["score"] = r["delta_dists"] + lpips_weight * r["delta_lpips"]
            for key in FEATURES:
                r[key] = f(row, key)
            out[str(r["image"])] = r
    return out


def noop_row(image: str) -> dict:
    row = dict(NOOP)
    row["image"] = image
    return row


def candidate_rows(image: str, runs: Dict[str, Dict[str, dict]]) -> List[dict]:
    return [noop_row(image)] + [runs[risk][image] for risk in runs]


def choose_oracle(image: str, runs: Dict[str, Dict[str, dict]]) -> dict:
    return min(candidate_rows(image, runs), key=lambda r: (r["score"], -r["delta_ms_ssim"]))


def summarize(rows: Sequence[dict], policy: str) -> dict:
    choices = Counter(str(r["risk"]) for r in rows)
    return {
        "policy": policy,
        "n": len(rows),
        "mean_score": mean(float(r["score"]) for r in rows),
        "worst_score": max(float(r["score"]) for r in rows),
        "score_win_count": sum(float(r["score"]) < 0 for r in rows),
        "mean_delta_psnr": mean(float(r["delta_psnr"]) for r in rows),
        "worst_delta_psnr": min(float(r["delta_psnr"]) for r in rows),
        "negative_psnr_count": sum(float(r["delta_psnr"]) < 0 for r in rows),
        "mean_delta_ms_ssim": mean(float(r["delta_ms_ssim"]) for r in rows),
        "mean_delta_lpips": mean(float(r["delta_lpips"]) for r in rows),
        "mean_delta_dists": mean(float(r["delta_dists"]) for r in rows),
        "max_abs_delta_bpp": max(abs(float(r["delta_bpp"])) for r in rows),
        "max_decode_diff": max(float(r["max_decode_diff"]) for r in rows),
        "nonfinite_rows": sum(int(r["nonfinite"]) for r in rows),
        "choices": dict(sorted(choices.items())),
    }


def risk_list(runs: Dict[str, Dict[str, dict]]) -> List[str]:
    return list(runs.keys())


def model_feature_vector(row: dict, risk_names: List[str], feature_names: List[str]) -> List[float]:
    rv = risk_value(str(row["risk"]))
    vec = [1.0, rv, abs(rv), 1.0 if row["risk"] == "noop" else 0.0]
    vec.extend(1.0 if row["risk"] == risk else 0.0 for risk in risk_names)
    if row["risk"] == "noop":
        vec.extend(0.0 for _ in feature_names)
    else:
        vec.extend(float(row.get(k, 0.0)) for k in feature_names)
    return vec


def build_matrix(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    risk_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    xs: List[List[float]] = []
    ys: List[float] = []
    keys: List[Tuple[str, str]] = []
    for image in images:
        for row in candidate_rows(image, runs):
            xs.append(model_feature_vector(row, risk_names, feature_names))
            ys.append(float(row["score"]))
            keys.append((image, str(row["risk"])))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), keys


def fit_ridge(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    lambda_value: float,
) -> Tuple[Model, dict]:
    risk_names = risk_list(runs)
    x, y, _ = build_matrix(images, runs, feature_names, risk_names)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    mu[0] = 0.0
    sigma[0] = 1.0
    sigma[sigma < 1e-9] = 1.0
    xz = (x - mu) / sigma
    reg = np.eye(xz.shape[1], dtype=np.float64) * float(lambda_value)
    reg[0, 0] = 0.0
    lhs = xz.T @ xz + reg
    rhs = xz.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    model = Model(
        name=f"ridge_score_lam{lambda_value:g}",
        risk_names=risk_names,
        feature_names=feature_names,
        mean=mu.tolist(),
        scale=sigma.tolist(),
        weights=beta.reshape(-1, 1).tolist(),
        lambda_value=float(lambda_value),
        train_mode="ridge_candidate_score",
    )
    info = {
        "rank": int(np.linalg.matrix_rank(xz)),
        "condition": float(np.linalg.cond(lhs)),
        "n_train_candidates": int(xz.shape[0]),
        "n_features": int(xz.shape[1]),
    }
    return model, info


def predict_score(model: Model, row: dict) -> float:
    vec = np.asarray(model_feature_vector(row, model.risk_names, model.feature_names), dtype=np.float64)
    mu = np.asarray(model.mean, dtype=np.float64)
    sigma = np.asarray(model.scale, dtype=np.float64)
    beta = np.asarray(model.weights, dtype=np.float64).reshape(-1)
    return float(((vec - mu) / sigma) @ beta)


def select_with_model(image: str, runs: Dict[str, Dict[str, dict]], model: Model, margin: float = 0.0) -> dict:
    candidates = candidate_rows(image, runs)
    scored = []
    for row in candidates:
        pred = predict_score(model, row)
        if row["risk"] != "noop":
            pred += margin
        scored.append((pred, row))
    pred, chosen = min(scored, key=lambda x: (x[0], x[1]["risk"] != "noop"))
    out = dict(chosen)
    out["predicted_score"] = pred
    return out


def select_fixed(images: Sequence[str], runs: Dict[str, Dict[str, dict]], risk: str) -> List[dict]:
    if risk == "noop":
        return [noop_row(img) for img in images]
    return [runs[risk][img] for img in images]


def selected_rows(images: Sequence[str], runs: Dict[str, Dict[str, dict]], model: Model, margin: float) -> List[dict]:
    return [select_with_model(img, runs, model, margin) for img in images]


def loocv_for_lambda(
    images: Sequence[str],
    runs: Dict[str, Dict[str, dict]],
    feature_names: List[str],
    lambda_value: float,
    margin: float,
) -> dict:
    rows: List[dict] = []
    infos: List[dict] = []
    for held in images:
        train = [img for img in images if img != held]
        model, info = fit_ridge(train, runs, feature_names, lambda_value)
        infos.append(info)
        rows.append(select_with_model(held, runs, model, margin))
    s = summarize(rows, f"loocv_lam{lambda_value:g}_margin{margin:g}")
    s["lambda"] = float(lambda_value)
    s["margin"] = float(margin)
    s["mean_condition"] = mean(i["condition"] for i in infos)
    return s


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
    ap.add_argument("--cal-count", type=int, default=20)
    ap.add_argument("--lpips-weight", type=float, default=3.0)
    ap.add_argument("--lambda-grid", default="0,1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--margin-grid", default="0,0.0001,0.00025,0.0005")
    ap.add_argument("--feature-set", choices=["global", "global_slice"], default="global_slice")
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    runs = {name: read_rows(path, name, args.lpips_weight) for name, path in args.run}
    image_sets = [set(rows) for rows in runs.values()]
    images = sorted(set.intersection(*image_sets))
    cal_images = images[: args.cal_count]
    eval_images = images[args.cal_count :]
    if len(cal_images) < 4 or not eval_images:
        raise SystemExit("Need at least 4 calibration images and a non-empty eval split")

    feature_names = GLOBAL_FEATURES if args.feature_set == "global" else FEATURES
    lambdas = [float(x) for x in args.lambda_grid.split(",") if x]
    margins = [float(x) for x in args.margin_grid.split(",") if x]

    cv_rows: List[dict] = []
    for lam in lambdas:
        for margin in margins:
            cv_rows.append(loocv_for_lambda(cal_images, runs, feature_names, lam, margin))
    cv_rows.sort(key=lambda r: (r["mean_score"], r["worst_score"], -r["mean_delta_ms_ssim"], -r["mean_delta_psnr"]))
    selected_cv = cv_rows[0]

    model, fit_info = fit_ridge(cal_images, runs, feature_names, selected_cv["lambda"])
    cal_selected = selected_rows(cal_images, runs, model, selected_cv["margin"])
    eval_selected = selected_rows(eval_images, runs, model, selected_cv["margin"])
    cal_summary = summarize(cal_selected, "ridge_selected_cal")
    eval_summary = summarize(eval_selected, "ridge_selected_eval")

    fixed_summaries = []
    for risk in ["noop"] + risk_list(runs):
        fixed_summaries.append({
            "split": "cal",
            **summarize(select_fixed(cal_images, runs, risk), f"fixed_{risk}"),
        })
        fixed_summaries.append({
            "split": "eval",
            **summarize(select_fixed(eval_images, runs, risk), f"fixed_{risk}"),
        })

    cal_oracle = summarize([choose_oracle(img, runs) for img in cal_images], "oracle_cal")
    eval_oracle = summarize([choose_oracle(img, runs) for img in eval_images], "oracle_eval")

    per_image_rows: List[dict] = []
    for split, split_images in [("cal", cal_images), ("eval", eval_images)]:
        for img in split_images:
            chosen = select_with_model(img, runs, model, selected_cv["margin"])
            oracle = choose_oracle(img, runs)
            row = {
                "split": split,
                "image": img,
                "chosen_risk": chosen["risk"],
                "chosen_predicted_score": chosen.get("predicted_score", ""),
                "chosen_score": chosen["score"],
                "chosen_delta_psnr": chosen["delta_psnr"],
                "chosen_delta_ms_ssim": chosen["delta_ms_ssim"],
                "chosen_delta_lpips": chosen["delta_lpips"],
                "chosen_delta_dists": chosen["delta_dists"],
                "oracle_risk": oracle["risk"],
                "oracle_score": oracle["score"],
                "oracle_delta_psnr": oracle["delta_psnr"],
            }
            for risk in risk_list(runs):
                row[f"{risk}_score"] = runs[risk][img]["score"]
                row[f"{risk}_dpsnr"] = runs[risk][img]["delta_psnr"]
            per_image_rows.append(row)

    prefix = Path(args.output_prefix)
    payload = {
        "score": f"delta_DISTS + {args.lpips_weight:g} * delta_LPIPS",
        "psnr_role": "diagnostic_tail_metric_not_selector",
        "feature_set": args.feature_set,
        "feature_count": len(feature_names),
        "n_images": len(images),
        "cal_images": cal_images,
        "eval_images": eval_images,
        "selected_cv": selected_cv,
        "fit_info": fit_info,
        "cal_selected": cal_summary,
        "eval_selected": eval_summary,
        "cal_oracle": cal_oracle,
        "eval_oracle": eval_oracle,
        "fixed_summaries": fixed_summaries,
        "model": {
            "name": model.name,
            "lambda_value": model.lambda_value,
            "margin": selected_cv["margin"],
            "risk_names": model.risk_names,
            "feature_names": model.feature_names,
            "mean": model.mean,
            "scale": model.scale,
            "weights": model.weights,
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_name(prefix.name + "_cv_grid.csv"), cv_rows, cv_rows[0].keys())
    write_csv(prefix.with_name(prefix.name + "_fixed_summaries.csv"), fixed_summaries, fixed_summaries[0].keys())
    write_csv(prefix.with_name(prefix.name + "_per_image.csv"), per_image_rows, per_image_rows[0].keys())

    lines = [
        "# E353 EF-LIC Perceptual Learned Selector Split Audit",
        "",
        f"Score: `delta_DISTS + {args.lpips_weight:g} * delta_LPIPS` (lower is better). PSNR is diagnostic only.",
        f"Feature set: `{args.feature_set}` with {len(feature_names)} decoder-visible candidate features.",
        f"Calibration images: {len(cal_images)}; eval images: {len(eval_images)}.",
        "",
        "## Selected Ridge Policy",
        "",
        f"LOOCV selected lambda `{selected_cv['lambda']}` and active margin `{selected_cv['margin']}`.",
        f"LOOCV calibration score `{fmt(selected_cv['mean_score'])}`, worst `{fmt(selected_cv['worst_score'])}`, dPSNR `{fmt(selected_cv['mean_delta_psnr'])}`.",
        "",
        "| split | score | worst score | dPSNR | worst dPSNR | dMS-SSIM | dLPIPS | dDISTS | score wins | choices | max dBPP | decode max | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for split, s in [("cal selected", cal_summary), ("eval selected", eval_summary), ("cal oracle", cal_oracle), ("eval oracle", eval_oracle)]:
        lines.append(
            "| {split} | {score} | {worst} | {dpsnr} | {wdpsnr} | {dms} | {dlpips} | {ddists} | {wins}/{n} | `{choices}` | {dbpp} | {dec:.3e} | {nf} |".format(
                split=split,
                score=fmt(s["mean_score"]),
                worst=fmt(s["worst_score"]),
                dpsnr=fmt(s["mean_delta_psnr"]),
                wdpsnr=fmt(s["worst_delta_psnr"]),
                dms=fmt(s["mean_delta_ms_ssim"]),
                dlpips=fmt(s["mean_delta_lpips"]),
                ddists=fmt(s["mean_delta_dists"]),
                wins=s["score_win_count"],
                n=s["n"],
                choices=s["choices"],
                dbpp=fmt(s["max_abs_delta_bpp"]),
                dec=s["max_decode_diff"],
                nf=s["nonfinite_rows"],
            )
        )
    lines += [
        "",
        "## Fixed-Risk Baselines",
        "",
        "| split | policy | score | worst score | dPSNR | worst dPSNR | score wins | choices |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in fixed_summaries:
        lines.append(
            "| {split} | {policy} | {score} | {worst} | {dpsnr} | {wdpsnr} | {wins}/{n} | `{choices}` |".format(
                split=s["split"],
                policy=s["policy"],
                score=fmt(s["mean_score"]),
                worst=fmt(s["worst_score"]),
                dpsnr=fmt(s["mean_delta_psnr"]),
                wdpsnr=fmt(s["worst_delta_psnr"]),
                wins=s["score_win_count"],
                n=s["n"],
                choices=s["choices"],
            )
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- This is still a small split audit. It is meant to decide selector direction before EF-LIC full training, not to claim final performance.",
        "- A useful learned selector should beat simple calibration thresholds on held-out perceptual score without breaking bpp/decode exactness.",
        "- If it overfits or underperforms fixed risk, the next step is more independent teacher data or a more local/sequential controller, not immediate full training.",
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {prefix.with_suffix('.md')}, {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
