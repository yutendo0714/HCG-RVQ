#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "experiments"
        / "analysis"
        / "e250_glc_bitaware_tail_vq_split_train_q0_oi16_kodak24_lpips1_w100.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e251_glc_e250_activation_gate",
    )
    p.add_argument("--label", default="trained_eval")
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--bpp-weight", type=float, default=1.0)
    p.add_argument("--side-bits", type=float, default=1.0)
    return p.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_rows(path: Path, label: str, lpips_weight: float, bpp_weight: float, side_bits: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") != label:
                continue
            height = finite(row.get("height"))
            width = finite(row.get("width"))
            area = height * width if height > 0 and width > 0 else float("nan")
            side_bpp = side_bits / area if area > 0 else 0.0
            delta_lpips = finite(row.get("branch_lpips")) - finite(row.get("base_lpips"))
            delta_dists = finite(row.get("branch_dists")) - finite(row.get("base_dists"))
            delta_bpp = finite(row.get("empirical_bpp_delta"))
            combined = delta_dists + lpips_weight * delta_lpips + bpp_weight * delta_bpp
            combined_with_side = combined + side_bpp
            out = dict(row)
            out.update(
                {
                    "area": area,
                    "side_bpp": side_bpp,
                    "delta_psnr": finite(row.get("branch_psnr")) - finite(row.get("base_psnr")),
                    "delta_ms_ssim": finite(row.get("branch_ms_ssim")) - finite(row.get("base_ms_ssim")),
                    "delta_lpips": delta_lpips,
                    "delta_dists": delta_dists,
                    "delta_bpp": delta_bpp,
                    "combined_score": combined,
                    "combined_with_side": combined_with_side,
                    "oracle_select": combined_with_side < 0.0,
                }
            )
            rows.append(out)
    if not rows:
        raise SystemExit(f"no rows with label={label!r} in {path}")
    return rows


def mean(values: list[float]) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def summarize_policy(rows: list[dict[str, Any]], name: str, selected: list[bool]) -> dict[str, Any]:
    n = len(rows)
    selected_n = sum(1 for v in selected if v)
    def policy_delta(key: str) -> float:
        return mean([finite(r[key]) if use else 0.0 for r, use in zip(rows, selected)])

    return {
        "policy": name,
        "selected": selected_n,
        "total": n,
        "selected_frac": selected_n / n if n else 0.0,
        "mean_combined": policy_delta("combined_score") + mean([finite(r["side_bpp"]) if use else 0.0 for r, use in zip(rows, selected)]),
        "mean_delta_psnr": policy_delta("delta_psnr"),
        "mean_delta_ms_ssim": policy_delta("delta_ms_ssim"),
        "mean_delta_lpips": policy_delta("delta_lpips"),
        "mean_delta_dists": policy_delta("delta_dists"),
        "mean_delta_bpp": policy_delta("delta_bpp"),
        "side_bpp_mean": mean([finite(r["side_bpp"]) if use else 0.0 for r, use in zip(rows, selected)]),
    }


def score_for_rule(rows: list[dict[str, Any]], feature: str, direction: str, threshold: float) -> float:
    selected = []
    for row in rows:
        value = finite(row.get(feature))
        use = value <= threshold if direction == "<=" else value >= threshold
        selected.append(use)
    return summarize_policy(rows, "candidate", selected)["mean_combined"]


def thresholds(values: list[float]) -> list[float]:
    unique = sorted({v for v in values if math.isfinite(v)})
    if not unique:
        return []
    points = [unique[0] - 1e-12, unique[-1] + 1e-12]
    points.extend(unique)
    for a, b in zip(unique, unique[1:]):
        points.append((a + b) / 2.0)
    return sorted(set(points))


def best_threshold_rule(rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in features:
        values = [finite(row.get(feature)) for row in rows]
        for threshold in thresholds(values):
            for direction in ("<=", ">="):
                score = score_for_rule(rows, feature, direction, threshold)
                selected = [finite(row.get(feature)) <= threshold if direction == "<=" else finite(row.get(feature)) >= threshold for row in rows]
                candidate = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "mean_combined": score,
                    "selected": sum(1 for v in selected if v),
                }
                if best is None or candidate["mean_combined"] < best["mean_combined"]:
                    best = candidate
    if best is None:
        raise SystemExit("no threshold rule candidates")
    return best


def apply_rule(rows: list[dict[str, Any]], rule: dict[str, Any]) -> list[bool]:
    feature = str(rule["feature"])
    direction = str(rule["direction"])
    threshold = float(rule["threshold"])
    return [
        finite(row.get(feature)) <= threshold if direction == "<=" else finite(row.get(feature)) >= threshold
        for row in rows
    ]


def loocv_rule(rows: list[dict[str, Any]], features: list[str]) -> tuple[dict[str, Any], list[bool]]:
    selected: list[bool] = []
    heldout_rules: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != idx]
        rule = best_threshold_rule(train, features)
        heldout_rules.append(rule)
        selected.append(apply_rule([row], rule)[0])
    summary = summarize_policy(rows, "loocv_best_single_feature_threshold", selected)
    summary["heldout_rules"] = heldout_rules
    return summary, selected


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], feature_rules: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [rule["feature"] for rule in feature_rules]
    best_rule = min(feature_rules, key=lambda r: r["summary"]["mean_combined"])
    always_selected = [True] * len(rows)
    never_selected = [False] * len(rows)
    oracle_selected = [bool(r["oracle_select"]) for r in rows]
    best_selected = apply_rule(rows, best_rule)
    loocv_summary, loocv_selected = loocv_rule(rows, feature_names)
    policies = [
        summarize_policy(rows, "baseline_no_branch", never_selected),
        summarize_policy(rows, "always_branch", always_selected),
        summarize_policy(rows, "oracle_per_image_with_side", oracle_selected),
        summarize_policy(rows, f"best_in_sample_{best_rule['feature']}_{best_rule['direction']}", best_selected),
        loocv_summary,
    ]

    per_image_fields = [
        "image",
        "combined_score",
        "combined_with_side",
        "delta_psnr",
        "delta_ms_ssim",
        "delta_lpips",
        "delta_dists",
        "delta_bpp",
        "active_mse_ratio",
        "active_scalar_mse",
        "active_rvq_mse",
        "index_entropy_mean",
        "index_used_frac_mean",
        "oracle_select",
        "best_threshold_select",
        "loocv_select",
    ]
    with args.output_prefix.with_suffix(".per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_fields)
        writer.writeheader()
        for row, best_use, loocv_use in zip(rows, best_selected, loocv_selected):
            out = {field: row.get(field, "") for field in per_image_fields}
            out["best_threshold_select"] = best_use
            out["loocv_select"] = loocv_use
            writer.writerow(out)

    with args.output_prefix.with_suffix(".policies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(policies[0].keys()))
        writer.writeheader()
        for row in policies:
            flat = {k: v for k, v in row.items() if k != "heldout_rules"}
            writer.writerow(flat)

    payload = {
        "experiment": "E251 GLC E250 activation gate analysis",
        "input_csv": str(args.csv),
        "label": args.label,
        "score": {
            "formula": "delta_DISTS + lpips_weight * delta_LPIPS + bpp_weight * delta_empirical_bpp",
            "lpips_weight": args.lpips_weight,
            "bpp_weight": args.bpp_weight,
            "side_bits": args.side_bits,
        },
        "policies": policies,
        "feature_rules": feature_rules,
        "best_rule": {k: v for k, v in best_rule.items() if k != "summary"},
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_images = sorted(rows, key=lambda r: finite(r["combined_score"]))[:5]
    worst_images = sorted(rows, key=lambda r: finite(r["combined_score"]), reverse=True)[:5]
    lines = [
        "# E251 GLC E250 Activation Gate Analysis",
        "",
        f"Input: `{args.csv}` (`{args.label}` rows).",
        "",
        f"Score: `delta_DISTS + {args.lpips_weight:g} * delta_LPIPS + {args.bpp_weight:g} * delta_empirical_bpp`; "
        f"selected-branch side overhead is `{args.side_bits:g}` bit/image.",
        "",
        "## Policy Summary",
        "",
        "| policy | selected | mean score | dPSNR | dMS | dLPIPS | dDISTS | dbpp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in policies:
        lines.append(
            f"| {p['policy']} | {p['selected']}/{p['total']} | {p['mean_combined']:+.6f} | "
            f"{p['mean_delta_psnr']:+.6f} | {p['mean_delta_ms_ssim']:+.6f} | "
            f"{p['mean_delta_lpips']:+.6f} | {p['mean_delta_dists']:+.6f} | "
            f"{p['mean_delta_bpp']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best Single-Feature Rules",
            "",
            "| feature | rule | selected | mean score |",
            "|---|---|---:|---:|",
        ]
    )
    for rule in sorted(feature_rules, key=lambda r: r["summary"]["mean_combined"])[:12]:
        summary = rule["summary"]
        lines.append(
            f"| {rule['feature']} | `{rule['direction']} {rule['threshold']:.6g}` | "
            f"{summary['selected']}/{summary['total']} | {summary['mean_combined']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Image Extremes",
            "",
            "Best branch-benefit images:",
            "",
        ]
    )
    for row in best_images:
        lines.append(
            f"- `{row['image']}`: score `{finite(row['combined_score']):+.6f}`, "
            f"dLPIPS `{finite(row['delta_lpips']):+.6f}`, dDISTS `{finite(row['delta_dists']):+.6f}`, "
            f"dbpp `{finite(row['delta_bpp']):+.6f}`"
        )
    lines.extend(["", "Worst branch-harm images:", ""])
    for row in worst_images:
        lines.append(
            f"- `{row['image']}`: score `{finite(row['combined_score']):+.6f}`, "
            f"dLPIPS `{finite(row['delta_lpips']):+.6f}`, dDISTS `{finite(row['delta_dists']):+.6f}`, "
            f"dbpp `{finite(row['delta_bpp']):+.6f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All-on E250 remains mixed on full Kodak24, but oracle selection shows that the branch has useful local benefit on a subset of images. This supports the next implementation step: turn the E250 branch into a selective activation/index-prior module instead of scaling the all-on branch unchanged.",
            "",
            "The single-feature rules are diagnostic upper bounds unless the feature can be computed or predicted before signaling the branch. Decoder/encoder-safe candidates are the active residual and index-prior statistics; base quality metrics are kept only as explanatory probes.",
            "",
        ]
    )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.label, args.lpips_weight, args.bpp_weight, args.side_bits)
    candidate_features = [
        "active_mse_ratio",
        "active_scalar_mse",
        "active_rvq_mse",
        "index_entropy_mean",
        "index_used_frac_mean",
        "index_dead_frac_mean",
        "empirical_bpp_delta",
        "fixed_bpp_delta",
        "base_bpp",
        "base_psnr",
        "base_ms_ssim",
        "base_lpips",
        "base_dists",
        "height",
        "width",
        "area",
    ]
    feature_rules = []
    for feature in candidate_features:
        rule = best_threshold_rule(rows, [feature])
        selected = apply_rule(rows, rule)
        rule["summary"] = summarize_policy(rows, f"{feature}_{rule['direction']}", selected)
        feature_rules.append(rule)
    write_outputs(args, rows, feature_rules)
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
