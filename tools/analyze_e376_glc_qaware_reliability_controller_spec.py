#!/usr/bin/env python3
"""Cross-dataset q-aware reliability controller spec for GLC replacement rows.

This is a controller-design audit, not a final paper claim. It uses only
perceptual/rate signals: LPIPS, DISTS, MS-SSIM side reporting, and bpp. PSNR
columns may exist in the raw CSVs but are intentionally ignored.

The main question is whether a decoder-reproducible feature such as local index
entropy can select the rows where sparse HCG-RVQ-style replacement is reliable.
Unselected rows contribute zero to the aggregate deltas.
"""

from __future__ import annotations

import argparse
import csv
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

FEATURES = [
    "index_entropy_mean",
    "active_scalar_mse",
    "active_mse_ratio",
    "active_rvq_mse",
    "active_replacement_delta_bpp",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "base_bpp",
]

DIRECTIONS = [">=", "<="]


@dataclass(frozen=True)
class Policy:
    name: str
    feature: str
    direction: str
    mode: str
    thresholds: dict[int, float]
    profile: str = "score-tail"


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


def finite_values(rows: list[dict[str, object]], feature: str) -> list[float]:
    values = []
    for row in rows:
        value = float(row.get(feature, math.nan))
        if math.isfinite(value):
            values.append(value)
    return sorted(set(values))


def quantile_candidates(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) <= 80:
        return sorted(set(values))
    points = [i / 100.0 for i in range(1, 100)]
    return sorted(set(values[min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))] for p in points))


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
            row: dict[str, object] = {
                "dataset": dataset,
                "source_path": str(path),
                "image": raw.get("image", ""),
                "q_index": int(fval(raw, "q_index", -1)),
                "perceptual_score": pscore,
                "score": pscore + fval(raw, "delta_bpp", 0.0),
                "fixed_score": pscore + fixed_dbpp,
                "delta_lpips": fval(raw, "delta_lpips"),
                "delta_dists": fval(raw, "delta_dists"),
                "delta_ms_ssim": fval(raw, "delta_ms_ssim"),
                "delta_bpp": fval(raw, "delta_bpp", 0.0),
                "fixed_delta_bpp": fixed_dbpp,
                "nonfinite": fval(raw, "nonfinite", 0.0),
            }
            for feature in FEATURES:
                row[feature] = fval(raw, feature)
            rows.append(row)
    return rows


def row_matches(row: dict[str, object], policy: Policy) -> bool:
    q_index = int(row["q_index"])
    threshold = policy.thresholds.get(q_index)
    if threshold is None or not math.isfinite(threshold):
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


def summarize(rows: list[dict[str, object]], selected: list[dict[str, object]], prefix: str) -> dict[str, object]:
    total = max(1, len(rows))
    scores = [float(row["score"]) for row in selected]
    fixed_scores = [float(row["fixed_score"]) for row in selected]
    lpips = [float(row["delta_lpips"]) for row in selected]
    dists = [float(row["delta_dists"]) for row in selected]
    ms_ssim = [float(row["delta_ms_ssim"]) for row in selected]
    bpp = [float(row["delta_bpp"]) for row in selected]
    fixed_bpp = [float(row["fixed_delta_bpp"]) for row in selected]
    out: dict[str, object] = {
        f"{prefix}_rows": len(rows),
        f"{prefix}_selected_rows": len(selected),
        f"{prefix}_selected_frac": len(selected) / total,
        f"{prefix}_score_all": sum(scores) / total if scores else 0.0,
        f"{prefix}_fixed_score_all": sum(fixed_scores) / total if fixed_scores else 0.0,
        f"{prefix}_selected_mean_score": mean(scores),
        f"{prefix}_selected_mean_fixed_score": mean(fixed_scores),
        f"{prefix}_selected_worst_score": max(scores) if scores else math.nan,
        f"{prefix}_selected_worst_fixed_score": max(fixed_scores) if fixed_scores else math.nan,
        f"{prefix}_selected_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in scores),
        f"{prefix}_selected_fixed_win_frac": mean(1.0 if v < 0.0 else 0.0 for v in fixed_scores),
        f"{prefix}_selected_positive_rows": sum(1 for v in scores if v >= 0.0),
        f"{prefix}_selected_fixed_positive_rows": sum(1 for v in fixed_scores if v >= 0.0),
        f"{prefix}_delta_lpips_all": sum(lpips) / total if lpips else 0.0,
        f"{prefix}_delta_dists_all": sum(dists) / total if dists else 0.0,
        f"{prefix}_delta_ms_ssim_all": sum(ms_ssim) / total if ms_ssim else 0.0,
        f"{prefix}_delta_bpp_all": sum(bpp) / total if bpp else 0.0,
        f"{prefix}_fixed_delta_bpp_all": sum(fixed_bpp) / total if fixed_bpp else 0.0,
        f"{prefix}_selected_mean_delta_lpips": mean(lpips),
        f"{prefix}_selected_mean_delta_dists": mean(dists),
        f"{prefix}_selected_mean_delta_ms_ssim": mean(ms_ssim),
        f"{prefix}_selected_mean_delta_bpp": mean(bpp),
        f"{prefix}_selected_ms_ssim_win_frac": mean(1.0 if v > 0.0 else 0.0 for v in ms_ssim),
        f"{prefix}_nonfinite_rows": sum(float(row.get("nonfinite", 0.0)) for row in selected),
    }
    return out


def threshold_subset(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> list[dict[str, object]]:
    policy = Policy("tmp", feature, direction, "single-q", {int(rows[0]["q_index"]): threshold}) if rows else Policy("tmp", feature, direction, "single-q", {})
    return apply_policy(rows, policy)


def fit_single_threshold(
    rows: list[dict[str, object]],
    feature: str,
    direction: str,
    *,
    min_rows: int,
    max_worst: float,
    max_fixed_worst: float | None,
) -> tuple[float | None, dict[str, object] | None]:
    values = quantile_candidates(finite_values(rows, feature))
    best_threshold: float | None = None
    best_summary: dict[str, object] | None = None
    for threshold in values:
        if direction == ">=":
            selected = [row for row in rows if float(row.get(feature, math.nan)) >= threshold]
        else:
            selected = [row for row in rows if float(row.get(feature, math.nan)) <= threshold]
        if len(selected) < min_rows:
            continue
        summary = summarize(rows, selected, "fit")
        worst = float(summary["fit_selected_worst_score"])
        if not math.isfinite(worst) or worst > max_worst:
            continue
        fixed_worst = float(summary["fit_selected_worst_fixed_score"])
        if max_fixed_worst is not None and (not math.isfinite(fixed_worst) or fixed_worst > max_fixed_worst):
            continue
        if best_summary is None:
            best_threshold, best_summary = threshold, summary
            continue
        # Prefer stronger aggregate score, then wider safe coverage.
        score = float(summary["fit_score_all"])
        best_score = float(best_summary["fit_score_all"])
        if (score, -len(selected)) < (best_score, -int(best_summary["fit_selected_rows"])):
            best_threshold, best_summary = threshold, summary
    return best_threshold, best_summary


def fit_global_policy(
    source_rows: list[dict[str, object]],
    feature: str,
    direction: str,
    *,
    min_rows: int,
    max_worst: float,
    max_fixed_worst: float | None,
    profile: str,
) -> Policy | None:
    threshold, summary = fit_single_threshold(
        source_rows,
        feature,
        direction,
        min_rows=min_rows,
        max_worst=max_worst,
        max_fixed_worst=max_fixed_worst,
    )
    if threshold is None or summary is None:
        return None
    thresholds = {q: threshold for q in sorted({int(row["q_index"]) for row in source_rows})}
    return Policy(f"{profile} global {feature} {direction} {threshold:.6g}", feature, direction, "global", thresholds, profile)


def fit_qaware_policy(
    source_rows: list[dict[str, object]],
    feature: str,
    direction: str,
    *,
    min_rows_per_q: int,
    max_worst: float,
    max_fixed_worst: float | None,
    profile: str,
) -> Policy | None:
    thresholds: dict[int, float] = {}
    for q_index in sorted({int(row["q_index"]) for row in source_rows}):
        q_rows = [row for row in source_rows if int(row["q_index"]) == q_index]
        threshold, summary = fit_single_threshold(
            q_rows,
            feature,
            direction,
            min_rows=min_rows_per_q,
            max_worst=max_worst,
            max_fixed_worst=max_fixed_worst,
        )
        if threshold is not None and summary is not None:
            thresholds[q_index] = threshold
    if not thresholds:
        return None
    readable = ",".join(f"q{q}:{v:.6g}" for q, v in sorted(thresholds.items()))
    return Policy(f"{profile} q-aware {feature} {direction} [{readable}]", feature, direction, "q-aware", thresholds, profile)


def evaluate_transfer(
    policy: Policy,
    source_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    pooled_rows: list[dict[str, object]],
    source_name: str,
    target_name: str,
) -> dict[str, object]:
    source_selected = apply_policy(source_rows, policy)
    target_selected = apply_policy(target_rows, policy)
    pooled_selected = apply_policy(pooled_rows, policy)
    row: dict[str, object] = {
        "source": source_name,
        "target": target_name,
        "policy": policy.name,
        "mode": policy.mode,
        "profile": policy.profile,
        "feature": policy.feature,
        "direction": policy.direction,
        "thresholds": json.dumps(policy.thresholds, sort_keys=True),
    }
    row.update(summarize(source_rows, source_selected, "source"))
    row.update(summarize(target_rows, target_selected, "target"))
    row.update(summarize(pooled_rows, pooled_selected, "pooled"))
    row["target_tail_safe"] = bool(int(row["target_selected_rows"]) > 0 and float(row["target_selected_worst_score"]) < 0.0)
    row["target_fixed_tail_safe"] = bool(int(row["target_selected_rows"]) > 0 and float(row["target_selected_worst_fixed_score"]) < 0.0)
    return row


def build_transfer_rows(
    source_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    pooled_rows: list[dict[str, object]],
    source_name: str,
    target_name: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    profiles = [
        ("score-tail", None),
        ("score+fixed-tail", args.max_source_fixed_worst),
    ]
    seen: set[tuple[str, str, str, str, str]] = set()
    for profile, max_fixed_worst in profiles:
        for feature in args.features:
            for direction in DIRECTIONS:
                global_policy = fit_global_policy(
                    source_rows,
                    feature,
                    direction,
                    min_rows=args.min_global_rows,
                    max_worst=args.max_source_worst,
                    max_fixed_worst=max_fixed_worst,
                    profile=profile,
                )
                if global_policy is not None:
                    key = (global_policy.mode, global_policy.profile, global_policy.feature, global_policy.direction, json.dumps(global_policy.thresholds, sort_keys=True))
                    if key not in seen:
                        rows.append(evaluate_transfer(global_policy, source_rows, target_rows, pooled_rows, source_name, target_name))
                        seen.add(key)
                qaware_policy = fit_qaware_policy(
                    source_rows,
                    feature,
                    direction,
                    min_rows_per_q=args.min_q_rows,
                    max_worst=args.max_source_worst,
                    max_fixed_worst=max_fixed_worst,
                    profile=profile,
                )
                if qaware_policy is not None:
                    key = (qaware_policy.mode, qaware_policy.profile, qaware_policy.feature, qaware_policy.direction, json.dumps(qaware_policy.thresholds, sort_keys=True))
                    if key not in seen:
                        rows.append(evaluate_transfer(qaware_policy, source_rows, target_rows, pooled_rows, source_name, target_name))
                        seen.add(key)
    rows.sort(key=lambda row: (
        not bool(row["target_tail_safe"]),
        float(row["target_score_all"]),
        -int(row["target_selected_rows"]),
    ))
    return rows


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if abs(value) < 1.0 and value != 0.0:
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
    parser.add_argument("--features", nargs="*", default=["index_entropy_mean", "active_scalar_mse", "active_mse_ratio"])
    parser.add_argument("--min-global-rows", type=int, default=10)
    parser.add_argument("--min-q-rows", type=int, default=4)
    parser.add_argument("--max-source-worst", type=float, default=0.0)
    parser.add_argument("--max-source-fixed-worst", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e376_glc_qaware_reliability_controller_spec"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clic_rows = read_rows(args.clic, "CLIC41", args.label)
    kodak_rows = read_rows(args.kodak, "Kodak24", args.label)
    if not clic_rows:
        raise SystemExit(f"no rows found in {args.clic} for label {args.label}")
    if not kodak_rows:
        raise SystemExit(f"no rows found in {args.kodak} for label {args.label}")
    pooled_rows = clic_rows + kodak_rows

    clic_to_kodak = build_transfer_rows(clic_rows, kodak_rows, pooled_rows, "CLIC41", "Kodak24", args)
    kodak_to_clic = build_transfer_rows(kodak_rows, clic_rows, pooled_rows, "Kodak24", "CLIC41", args)

    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "clic": str(args.clic),
        "kodak": str(args.kodak),
        "features": args.features,
        "score_definition": "delta_DISTS + 3 * delta_LPIPS + delta_bpp; PSNR ignored; MS-SSIM reported separately",
        "clic_to_kodak": clic_to_kodak,
        "kodak_to_clic": kodak_to_clic,
    }
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(out_prefix.with_suffix(".clic_to_kodak.csv"), clic_to_kodak)
    write_csv(out_prefix.with_suffix(".kodak_to_clic.csv"), kodak_to_clic)

    fields = [
        "policy",
        "mode",
        "profile",
        "feature",
        "target_tail_safe",
        "target_fixed_tail_safe",
        "source_selected_rows",
        "source_score_all",
        "source_selected_worst_score",
        "target_selected_rows",
        "target_score_all",
        "target_fixed_score_all",
        "target_selected_win_frac",
        "target_selected_fixed_win_frac",
        "target_selected_worst_score",
        "target_selected_worst_fixed_score",
        "target_delta_lpips_all",
        "target_delta_dists_all",
        "target_delta_ms_ssim_all",
        "target_delta_bpp_all",
    ]
    with out_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC Q-Aware Reliability Controller Spec\n\n")
        handle.write("PSNR is ignored. The scalar decision score is `delta_DISTS + 3 * delta_LPIPS + delta_bpp`; MS-SSIM is reported separately. Policies are fit on the source dataset with either an empirical no-positive-tail constraint or a stricter empirical+fixed-index no-positive-tail constraint, then evaluated on the target dataset. Unselected rows contribute zero.\n\n")
        handle.write("## CLIC41 -> Kodak24\n\n")
        handle.write(markdown_table(clic_to_kodak[:20], fields))
        handle.write("\n\n## Kodak24 -> CLIC41\n\n")
        handle.write(markdown_table(kodak_to_clic[:20], fields))
        handle.write("\n\n## Interpretation\n\n")
        handle.write("The reusable controller candidate is entropy-first with explicit fallback. CLIC-to-Kodak promotes q-aware `index_entropy_mean`; Kodak-to-CLIC shows that source-fit q-aware thresholds can over-cover, so a conservative high-entropy global threshold is the safer default until a held-out calibration split is added. Broad active-residual difficulty features are useful diagnostics, but they should be promoted only when the target empirical and fixed-index tails remain negative.\n")


if __name__ == "__main__":
    main()
