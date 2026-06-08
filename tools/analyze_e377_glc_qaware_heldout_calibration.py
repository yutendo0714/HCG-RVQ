#!/usr/bin/env python3
"""Held-out calibration audit for GLC q-aware HCG-RVQ reliability controllers.

This extends E376.  Policies are calibrated on image-disjoint folds and evaluated
on held-out images, so the result is closer to a paper-facing controller decision
than an in-sample threshold scan.  PSNR is intentionally ignored; the main scalar
score is delta_DISTS + 3 * delta_LPIPS + delta_bpp, with MS-SSIM side reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CLIC = Path(
    "experiments/analysis/"
    "e371_glc_signal_accounted_replacement_rows_clicpro41_qcurve_k4_parts01_t16_e41_s12.csv"
)
DEFAULT_KODAK = Path(
    "experiments/analysis/"
    "e373_glc_signal_accounted_replacement_rows_kodak24_qcurve_k4_parts01_t16_e24_s12.csv"
)


@dataclass(frozen=True)
class Policy:
    family: str
    feature: str
    direction: str
    mode: str
    thresholds: dict[int, float]
    profile: str

    @property
    def name(self) -> str:
        readable = ",".join(f"q{q}:{v:.6g}" for q, v in sorted(self.thresholds.items()))
        if self.mode == "global" and self.thresholds:
            readable = f"{next(iter(self.thresholds.values())):.6g}"
        return f"{self.profile} {self.mode} {self.feature} {self.direction} {readable}"


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def perceptual_score(raw: dict[str, str]) -> float:
    return fval(raw, "delta_dists") + 3.0 * fval(raw, "delta_lpips")


def read_rows(path: Path, dataset: str, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("label") != label:
                continue
            pscore = perceptual_score(raw)
            signal_bpp = fval(raw, "selection_signal_bpp", 0.0)
            fixed_dbpp = fval(raw, "active_rvq_fixed_bpp", 0.0) - fval(raw, "active_scalar_bpp", 0.0) + signal_bpp
            rows.append({
                "dataset": dataset,
                "image": raw.get("image", ""),
                "q_index": int(fval(raw, "q_index", -1)),
                "score": pscore + fval(raw, "delta_bpp", 0.0),
                "fixed_score": pscore + fixed_dbpp,
                "delta_lpips": fval(raw, "delta_lpips"),
                "delta_dists": fval(raw, "delta_dists"),
                "delta_ms_ssim": fval(raw, "delta_ms_ssim"),
                "delta_bpp": fval(raw, "delta_bpp", 0.0),
                "fixed_delta_bpp": fixed_dbpp,
                "nonfinite": fval(raw, "nonfinite", 0.0),
                "index_entropy_mean": fval(raw, "index_entropy_mean"),
                "active_mse_ratio": fval(raw, "active_mse_ratio"),
                "active_scalar_mse": fval(raw, "active_scalar_mse"),
                "active_rvq_mse": fval(raw, "active_rvq_mse"),
                "index_dead_frac_mean": fval(raw, "index_dead_frac_mean"),
                "index_used_frac_mean": fval(raw, "index_used_frac_mean"),
            })
    return rows


def assign_folds(rows: list[dict[str, object]], folds: int, seed: str) -> dict[tuple[str, str], int]:
    mapping: dict[tuple[str, str], int] = {}
    by_dataset: dict[str, list[str]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(str(row["image"]))
    for dataset, images in by_dataset.items():
        unique = sorted(set(images), key=lambda image: hashlib.sha1(f"{seed}:{dataset}:{image}".encode()).hexdigest())
        for idx, image in enumerate(unique):
            mapping[(dataset, image)] = idx % folds
    return mapping


def finite_values(rows: list[dict[str, object]], feature: str) -> list[float]:
    values = []
    for row in rows:
        value = float(row.get(feature, math.nan))
        if math.isfinite(value):
            values.append(value)
    return sorted(set(values))


def select_by_threshold(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        value = float(row.get(feature, math.nan))
        if not math.isfinite(value):
            continue
        if direction == ">=" and value >= threshold:
            selected.append(row)
        elif direction == "<=" and value <= threshold:
            selected.append(row)
    return selected


def summarize_selection(rows: list[dict[str, object]], selected: list[dict[str, object]]) -> dict[str, object]:
    total = max(1, len(rows))
    scores = [float(row["score"]) for row in selected]
    fixed_scores = [float(row["fixed_score"]) for row in selected]
    lpips = [float(row["delta_lpips"]) for row in selected]
    dists = [float(row["delta_dists"]) for row in selected]
    ms_ssim = [float(row["delta_ms_ssim"]) for row in selected]
    bpp = [float(row["delta_bpp"]) for row in selected]
    fixed_bpp = [float(row["fixed_delta_bpp"]) for row in selected]
    return {
        "rows": len(rows),
        "selected_rows": len(selected),
        "selected_frac": len(selected) / total,
        "score_sum": sum(scores),
        "fixed_score_sum": sum(fixed_scores),
        "score_all": sum(scores) / total if scores else 0.0,
        "fixed_score_all": sum(fixed_scores) / total if fixed_scores else 0.0,
        "selected_mean_score": mean(scores),
        "selected_mean_fixed_score": mean(fixed_scores),
        "selected_worst_score": max(scores) if scores else math.nan,
        "selected_worst_fixed_score": max(fixed_scores) if fixed_scores else math.nan,
        "selected_win_rows": sum(1 for v in scores if v < 0.0),
        "selected_fixed_win_rows": sum(1 for v in fixed_scores if v < 0.0),
        "selected_positive_rows": sum(1 for v in scores if v >= 0.0),
        "selected_fixed_positive_rows": sum(1 for v in fixed_scores if v >= 0.0),
        "delta_lpips_sum": sum(lpips),
        "delta_dists_sum": sum(dists),
        "delta_ms_ssim_sum": sum(ms_ssim),
        "delta_bpp_sum": sum(bpp),
        "fixed_delta_bpp_sum": sum(fixed_bpp),
        "nonfinite_rows": sum(float(row.get("nonfinite", 0.0)) for row in selected),
    }


def fit_threshold(
    rows: list[dict[str, object]],
    feature: str,
    direction: str,
    *,
    min_rows: int,
    max_worst: float,
    max_fixed_worst: float | None,
) -> tuple[float | None, dict[str, object] | None]:
    best_threshold: float | None = None
    best_summary: dict[str, object] | None = None
    for threshold in finite_values(rows, feature):
        selected = select_by_threshold(rows, feature, direction, threshold)
        if len(selected) < min_rows:
            continue
        summary = summarize_selection(rows, selected)
        worst = float(summary["selected_worst_score"])
        fixed_worst = float(summary["selected_worst_fixed_score"])
        if not math.isfinite(worst) or worst > max_worst:
            continue
        if max_fixed_worst is not None and (not math.isfinite(fixed_worst) or fixed_worst > max_fixed_worst):
            continue
        if best_summary is None:
            best_threshold = threshold
            best_summary = summary
            continue
        # More negative score_all is primary; coverage is secondary.
        key = (float(summary["score_all"]), -int(summary["selected_rows"]))
        best_key = (float(best_summary["score_all"]), -int(best_summary["selected_rows"]))
        if key < best_key:
            best_threshold = threshold
            best_summary = summary
    return best_threshold, best_summary


def fit_policy(
    rows: list[dict[str, object]],
    *,
    family: str,
    feature: str,
    direction: str,
    mode: str,
    profile: str,
    min_global_rows: int,
    min_q_rows: int,
    max_worst: float,
    max_fixed_worst: float | None,
) -> Policy | None:
    if mode == "global":
        threshold, summary = fit_threshold(
            rows,
            feature,
            direction,
            min_rows=min_global_rows,
            max_worst=max_worst,
            max_fixed_worst=max_fixed_worst,
        )
        if threshold is None or summary is None:
            return None
        q_indexes = sorted({int(row["q_index"]) for row in rows})
        return Policy(family, feature, direction, mode, {q: threshold for q in q_indexes}, profile)
    if mode == "q-aware":
        thresholds: dict[int, float] = {}
        for q_index in sorted({int(row["q_index"]) for row in rows}):
            q_rows = [row for row in rows if int(row["q_index"]) == q_index]
            threshold, summary = fit_threshold(
                q_rows,
                feature,
                direction,
                min_rows=min_q_rows,
                max_worst=max_worst,
                max_fixed_worst=max_fixed_worst,
            )
            if threshold is not None and summary is not None:
                thresholds[q_index] = threshold
        if not thresholds:
            return None
        return Policy(family, feature, direction, mode, thresholds, profile)
    raise ValueError(f"unsupported mode {mode}")


def row_matches(row: dict[str, object], policy: Policy) -> bool:
    threshold = policy.thresholds.get(int(row["q_index"]))
    if threshold is None:
        return False
    value = float(row.get(policy.feature, math.nan))
    if not math.isfinite(value):
        return False
    if policy.direction == ">=":
        return value >= threshold
    if policy.direction == "<=":
        return value <= threshold
    raise ValueError(f"unsupported direction {policy.direction}")


def apply_policy(rows: list[dict[str, object]], policy: Policy) -> list[dict[str, object]]:
    return [row for row in rows if row_matches(row, policy)]


def add_summary(prefix: str, row: dict[str, object], rows: list[dict[str, object]], selected: list[dict[str, object]]) -> None:
    summary = summarize_selection(rows, selected)
    for key, value in summary.items():
        if key.endswith("_sum") or key in {"selected_win_rows", "selected_fixed_win_rows"}:
            row[f"{prefix}_{key}"] = value
        else:
            row[f"{prefix}_{key}"] = value
    selected_rows = int(summary["selected_rows"])
    row[f"{prefix}_selected_win_frac"] = (float(summary["selected_win_rows"]) / selected_rows) if selected_rows else math.nan
    row[f"{prefix}_selected_fixed_win_frac"] = (float(summary["selected_fixed_win_rows"]) / selected_rows) if selected_rows else math.nan
    row[f"{prefix}_tail_safe"] = bool(selected_rows > 0 and float(summary["selected_worst_score"]) < 0.0)
    row[f"{prefix}_fixed_tail_safe"] = bool(selected_rows > 0 and float(summary["selected_worst_fixed_score"]) < 0.0)


def evaluate_fold(
    fold: int,
    policy: Policy,
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    dataset: str,
) -> dict[str, object]:
    if dataset == "pooled":
        eval_rows = test_rows
    else:
        eval_rows = [row for row in test_rows if row["dataset"] == dataset]
    selected_train = apply_policy(train_rows, policy)
    selected_eval = apply_policy(eval_rows, policy)
    out: dict[str, object] = {
        "fold": fold,
        "dataset": dataset,
        "family": policy.family,
        "policy": policy.name,
        "profile": policy.profile,
        "mode": policy.mode,
        "feature": policy.feature,
        "direction": policy.direction,
        "thresholds": json.dumps(policy.thresholds, sort_keys=True),
    }
    add_summary("calib", out, train_rows, selected_train)
    add_summary("heldout", out, eval_rows, selected_eval)
    return out


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total_rows = sum(int(row["heldout_rows"]) for row in rows)
    selected_rows = sum(int(row["heldout_selected_rows"]) for row in rows)
    score_sum = sum(float(row["heldout_score_sum"]) for row in rows)
    fixed_score_sum = sum(float(row["heldout_fixed_score_sum"]) for row in rows)
    lpips_sum = sum(float(row["heldout_delta_lpips_sum"]) for row in rows)
    dists_sum = sum(float(row["heldout_delta_dists_sum"]) for row in rows)
    ms_ssim_sum = sum(float(row["heldout_delta_ms_ssim_sum"]) for row in rows)
    bpp_sum = sum(float(row["heldout_delta_bpp_sum"]) for row in rows)
    fixed_bpp_sum = sum(float(row["heldout_fixed_delta_bpp_sum"]) for row in rows)
    win_rows = sum(int(row["heldout_selected_win_rows"]) for row in rows)
    fixed_win_rows = sum(int(row["heldout_selected_fixed_win_rows"]) for row in rows)
    positive_rows = sum(int(row["heldout_selected_positive_rows"]) for row in rows)
    fixed_positive_rows = sum(int(row["heldout_selected_fixed_positive_rows"]) for row in rows)
    worst_scores = [float(row["heldout_selected_worst_score"]) for row in rows if math.isfinite(float(row["heldout_selected_worst_score"]))]
    worst_fixed_scores = [float(row["heldout_selected_worst_fixed_score"]) for row in rows if math.isfinite(float(row["heldout_selected_worst_fixed_score"]))]
    first = rows[0]
    return {
        "dataset": first["dataset"],
        "family": first["family"],
        "profile": first["profile"],
        "mode": first["mode"],
        "feature": first["feature"],
        "direction": first["direction"],
        "folds": len(rows),
        "heldout_rows": total_rows,
        "heldout_selected_rows": selected_rows,
        "heldout_selected_frac": selected_rows / total_rows if total_rows else math.nan,
        "heldout_score_all": score_sum / total_rows if total_rows else math.nan,
        "heldout_fixed_score_all": fixed_score_sum / total_rows if total_rows else math.nan,
        "heldout_selected_win_frac": win_rows / selected_rows if selected_rows else math.nan,
        "heldout_selected_fixed_win_frac": fixed_win_rows / selected_rows if selected_rows else math.nan,
        "heldout_selected_worst_score": max(worst_scores) if worst_scores else math.nan,
        "heldout_selected_worst_fixed_score": max(worst_fixed_scores) if worst_fixed_scores else math.nan,
        "heldout_selected_positive_rows": positive_rows,
        "heldout_selected_fixed_positive_rows": fixed_positive_rows,
        "heldout_delta_lpips_all": lpips_sum / total_rows if total_rows else math.nan,
        "heldout_delta_dists_all": dists_sum / total_rows if total_rows else math.nan,
        "heldout_delta_ms_ssim_all": ms_ssim_sum / total_rows if total_rows else math.nan,
        "heldout_delta_bpp_all": bpp_sum / total_rows if total_rows else math.nan,
        "heldout_fixed_delta_bpp_all": fixed_bpp_sum / total_rows if total_rows else math.nan,
    }


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if value == 0.0:
            return "0.000000"
        if abs(value) < 1.0:
            return f"{value:+.6f}"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clic", type=Path, default=DEFAULT_CLIC)
    parser.add_argument("--kodak", type=Path, default=DEFAULT_KODAK)
    parser.add_argument("--label", default="trained_replacement_soft")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", default="e377-v1")
    parser.add_argument("--features", nargs="*", default=["index_entropy_mean", "active_mse_ratio"])
    parser.add_argument("--directions", nargs="*", default=[">="])
    parser.add_argument("--modes", nargs="*", default=["global", "q-aware"])
    parser.add_argument("--min-global-rows", type=int, default=10)
    parser.add_argument("--min-q-rows", type=int, default=2)
    parser.add_argument("--max-calib-worst", type=float, default=0.0)
    parser.add_argument("--max-calib-fixed-worst", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e377_glc_qaware_heldout_calibration"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    rows = read_rows(args.clic, "CLIC41", args.label) + read_rows(args.kodak, "Kodak24", args.label)
    if not rows:
        raise SystemExit("no input rows found")
    fold_map = assign_folds(rows, args.folds, args.seed)
    profiles = [("score-tail", None), ("score+fixed-tail", args.max_calib_fixed_worst)]
    fold_results: list[dict[str, object]] = []

    for fold in range(args.folds):
        train_rows = [row for row in rows if fold_map[(str(row["dataset"]), str(row["image"]))] != fold]
        test_rows = [row for row in rows if fold_map[(str(row["dataset"]), str(row["image"]))] == fold]
        for profile, max_fixed_worst in profiles:
            for feature in args.features:
                for direction in args.directions:
                    for mode in args.modes:
                        policy = fit_policy(
                            train_rows,
                            family=f"pooled-calib-{mode}-{feature}-{direction}",
                            feature=feature,
                            direction=direction,
                            mode=mode,
                            profile=profile,
                            min_global_rows=args.min_global_rows,
                            min_q_rows=args.min_q_rows,
                            max_worst=args.max_calib_worst,
                            max_fixed_worst=max_fixed_worst,
                        )
                        if policy is None:
                            continue
                        for dataset in ["pooled", "CLIC41", "Kodak24"]:
                            fold_results.append(evaluate_fold(fold, policy, train_rows, test_rows, dataset))

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in fold_results:
        key = (row["dataset"], row["family"], row["profile"], row["mode"], row["feature"], row["direction"])
        groups.setdefault(key, []).append(row)
    summary_rows = [aggregate_rows(group_rows) for group_rows in groups.values()]
    summary_rows.sort(key=lambda row: (
        row["dataset"] != "pooled",
        int(row["heldout_selected_positive_rows"]) > 0,
        int(row["heldout_selected_fixed_positive_rows"]) > 0,
        float(row["heldout_score_all"]),
        -int(row["heldout_selected_rows"]),
    ))

    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "clic": str(args.clic),
        "kodak": str(args.kodak),
        "folds": args.folds,
        "seed": args.seed,
        "score_definition": "delta_DISTS + 3 * delta_LPIPS + delta_bpp; PSNR ignored; MS-SSIM side reported",
        "fold_results": fold_results,
        "summary": summary_rows,
    }
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(out_prefix.with_suffix(".folds.csv"), fold_results)
    write_csv(out_prefix.with_suffix(".summary.csv"), summary_rows)

    fields = [
        "dataset",
        "profile",
        "mode",
        "feature",
        "heldout_rows",
        "heldout_selected_rows",
        "heldout_score_all",
        "heldout_fixed_score_all",
        "heldout_selected_win_frac",
        "heldout_selected_fixed_win_frac",
        "heldout_selected_worst_score",
        "heldout_selected_worst_fixed_score",
        "heldout_selected_positive_rows",
        "heldout_selected_fixed_positive_rows",
        "heldout_delta_lpips_all",
        "heldout_delta_dists_all",
        "heldout_delta_ms_ssim_all",
        "heldout_delta_bpp_all",
    ]
    with out_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC Q-Aware Held-Out Reliability Calibration\n\n")
        handle.write("PSNR is ignored. The scalar score is `delta_DISTS + 3 * delta_LPIPS + delta_bpp`; MS-SSIM is side reported. Thresholds are calibrated on image-disjoint folds from pooled CLIC41+Kodak24 and evaluated on held-out images. Unselected rows use exact fallback and contribute zero.\n\n")
        handle.write("## Summary\n\n")
        handle.write(markdown_table(summary_rows, fields))
        handle.write("\n\n## Interpretation\n\n")
        pooled = [row for row in summary_rows if row["dataset"] == "pooled"]
        strict_entropy = [row for row in pooled if row["feature"] == "index_entropy_mean" and row["profile"] == "score+fixed-tail"]
        if strict_entropy:
            best = min(strict_entropy, key=lambda row: (int(row["heldout_selected_positive_rows"]) > 0, int(row["heldout_selected_fixed_positive_rows"]) > 0, float(row["heldout_score_all"])))
            handle.write(
                "The held-out calibration result should be used as the promotion gate before long GLC fine-tuning. "
                f"The best strict entropy policy here is {best['mode']} `index_entropy_mean`, selecting {best['heldout_selected_rows']}/{best['heldout_rows']} held-out rows with score_all {float(best['heldout_score_all']):+.6f}, fixed_score_all {float(best['heldout_fixed_score_all']):+.6f}, worst {float(best['heldout_selected_worst_score']):+.6f}, and fixed worst {float(best['heldout_selected_worst_fixed_score']):+.6f}. "
            )
        handle.write(
            "If held-out empirical and fixed-index tails are both non-positive, q-aware entropy can be promoted to matched fine-tuning/full training. If fixed-index tails remain positive, use the conservative global high-entropy controller as the first full-training branch and keep q-aware thresholds as an ablation until a calibration split stabilizes them.\n"
        )


if __name__ == "__main__":
    main()
