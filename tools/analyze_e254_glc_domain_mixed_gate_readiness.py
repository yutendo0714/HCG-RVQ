#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    (
        "kodak24",
        "e250_kodak24_lpips1",
        ROOT
        / "experiments"
        / "analysis"
        / "e250_glc_bitaware_tail_vq_split_train_q0_oi16_kodak24_lpips1_w100.csv",
    ),
    (
        "clicpro8",
        "e252_clicpro8_lpips1",
        ROOT / "experiments" / "analysis" / "e252_glc_e250_oi16_clicpro8_lpips1_w100.csv",
    ),
    (
        "clicpro8",
        "e253_clicpro8_dists2",
        ROOT / "experiments" / "analysis" / "e253_glc_e250_oi16_clicpro8_dists2_w100.csv",
    ),
]

BRANCH_INTERNAL_FEATURES = [
    "active_mse_ratio",
    "active_scalar_mse",
    "active_rvq_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "empirical_bpp_delta",
    "fixed_bpp_delta",
    "base_bpp",
]

ANALYSIS_ONLY_FEATURES = [
    "base_psnr",
    "base_ms_ssim",
    "base_lpips",
    "base_dists",
    "height",
    "width",
    "area",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec: domain,variant,path. Defaults to E250 Kodak24 and E252/E253 CLICPro8.",
    )
    p.add_argument(
        "--primary-variants",
        default="e250_kodak24_lpips1,e252_clicpro8_lpips1",
        help="Comma-separated variants used for the primary domain-mixed gate audit.",
    )
    p.add_argument("--label", default="trained_eval")
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--bpp-weight", type=float, default=1.0)
    p.add_argument("--side-bits", type=float, default=1.0)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e254_glc_domain_mixed_gate_readiness",
    )
    return p.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, str, Path]]:
    if not specs:
        return DEFAULT_DATASETS
    parsed: list[tuple[str, str, Path]] = []
    for spec in specs:
        parts = spec.split(",", 2)
        if len(parts) != 3:
            raise SystemExit(f"dataset spec must be domain,variant,path: {spec}")
        domain, variant, path = parts
        parsed.append((domain.strip(), variant.strip(), Path(path).expanduser()))
    return parsed


def load_rows(
    datasets: list[tuple[str, str, Path]],
    label: str,
    lpips_weight: float,
    bpp_weight: float,
    side_bits: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, variant, path in datasets:
        if not path.exists():
            raise SystemExit(f"missing input CSV: {path}")
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
                out = dict(row)
                out.update(
                    {
                        "domain": domain,
                        "variant": variant,
                        "source_csv": str(path),
                        "area": area,
                        "side_bpp": side_bpp,
                        "delta_psnr": finite(row.get("branch_psnr")) - finite(row.get("base_psnr")),
                        "delta_ms_ssim": finite(row.get("branch_ms_ssim")) - finite(row.get("base_ms_ssim")),
                        "delta_lpips": delta_lpips,
                        "delta_dists": delta_dists,
                        "delta_bpp": delta_bpp,
                        "combined_score": combined,
                        "combined_with_side": combined + side_bpp,
                        "oracle_select": combined + side_bpp < 0.0,
                        "nonfinite_bool": str(row.get("nonfinite", "")).lower() in {"1", "true", "yes"},
                    }
                )
                rows.append(out)
    if not rows:
        raise SystemExit("no matching rows loaded")
    return rows


def summarize_policy(rows: list[dict[str, Any]], name: str, selected: list[bool]) -> dict[str, Any]:
    n = len(rows)
    selected_n = sum(1 for v in selected if v)

    def delta(key: str) -> float:
        return mean([finite(row.get(key)) if use else 0.0 for row, use in zip(rows, selected)])

    side = mean([finite(row.get("side_bpp")) if use else 0.0 for row, use in zip(rows, selected)])
    return {
        "policy": name,
        "selected": selected_n,
        "total": n,
        "selected_frac": selected_n / n if n else 0.0,
        "mean_combined": delta("combined_score") + side,
        "mean_delta_psnr": delta("delta_psnr"),
        "mean_delta_ms_ssim": delta("delta_ms_ssim"),
        "mean_delta_lpips": delta("delta_lpips"),
        "mean_delta_dists": delta("delta_dists"),
        "mean_delta_bpp": delta("delta_bpp"),
        "side_bpp_mean": side,
    }


def thresholds(values: list[float]) -> list[float]:
    unique = sorted({v for v in values if math.isfinite(v)})
    if not unique:
        return []
    points = [unique[0] - 1e-12, unique[-1] + 1e-12]
    points.extend(unique)
    points.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    return sorted(set(points))


def apply_rule(rows: list[dict[str, Any]], rule: dict[str, Any]) -> list[bool]:
    feature = str(rule["feature"])
    threshold = float(rule["threshold"])
    direction = str(rule["direction"])
    out: list[bool] = []
    for row in rows:
        value = finite(row.get(feature))
        out.append(value <= threshold if direction == "<=" else value >= threshold)
    return out


def best_threshold_rule(rows: list[dict[str, Any]], features: list[str], prefix: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in features:
        values = [finite(row.get(feature)) for row in rows]
        for threshold in thresholds(values):
            for direction in ("<=", ">="):
                selected = [
                    finite(row.get(feature)) <= threshold
                    if direction == "<="
                    else finite(row.get(feature)) >= threshold
                    for row in rows
                ]
                summary = summarize_policy(rows, f"{prefix}_{feature}_{direction}", selected)
                candidate = {
                    "feature_group": prefix,
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "summary": summary,
                }
                if best is None or summary["mean_combined"] < best["summary"]["mean_combined"]:
                    best = candidate
    if best is None:
        raise SystemExit(f"no threshold candidates for {prefix}")
    return best


def leave_group_out(
    rows: list[dict[str, Any]],
    group_key: str,
    features: list[str],
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_summaries: list[dict[str, Any]] = []
    all_selected: list[bool] = [False] * len(rows)
    groups = sorted({str(row[group_key]) for row in rows})
    for group in groups:
        train = [row for row in rows if str(row[group_key]) != group]
        test_index = [idx for idx, row in enumerate(rows) if str(row[group_key]) == group]
        test = [rows[idx] for idx in test_index]
        rule = best_threshold_rule(train, features, prefix)
        selected = apply_rule(test, rule)
        for idx, use in zip(test_index, selected):
            all_selected[idx] = use
        fold = summarize_policy(test, f"heldout_{group}_{prefix}_{rule['feature']}_{rule['direction']}", selected)
        fold.update(
            {
                "heldout_group": group,
                "feature": rule["feature"],
                "direction": rule["direction"],
                "threshold": rule["threshold"],
            }
        )
        fold_summaries.append(fold)
    summary = summarize_policy(rows, f"leave_{group_key}_out_{prefix}_threshold", all_selected)
    return summary, fold_summaries


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        always = summarize_policy(subset, f"{key}:{value}:always", [True] * len(subset))
        oracle = summarize_policy(subset, f"{key}:{value}:oracle", [bool(row["oracle_select"]) for row in subset])
        out.append(
            {
                key: value,
                "rows": len(subset),
                "positives": sum(1 for row in subset if row["oracle_select"]),
                "always_score": always["mean_combined"],
                "oracle_score": oracle["mean_combined"],
                "always_dpsnr": always["mean_delta_psnr"],
                "always_dlpips": always["mean_delta_lpips"],
                "always_ddists": always["mean_delta_dists"],
                "always_dbpp": always["mean_delta_bpp"],
            }
        )
    return out


def feature_separation(rows: list[dict[str, Any]], features: list[str], group_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    positives = [row for row in rows if row["oracle_select"]]
    negatives = [row for row in rows if not row["oracle_select"]]
    for feature in features:
        pos_mean = mean([finite(row.get(feature)) for row in positives])
        neg_mean = mean([finite(row.get(feature)) for row in negatives])
        diff = pos_mean - neg_mean if math.isfinite(pos_mean) and math.isfinite(neg_mean) else float("nan")
        out.append(
            {
                "feature_group": group_name,
                "feature": feature,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "positive_minus_negative": diff,
            }
        )
    return sorted(out, key=lambda row: abs(finite(row["positive_minus_negative"], 0.0)), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_specs(args.dataset)
    rows = load_rows(datasets, args.label, args.lpips_weight, args.bpp_weight, args.side_bits)
    primary_variants = {item.strip() for item in args.primary_variants.split(",") if item.strip()}
    primary_rows = [row for row in rows if row["variant"] in primary_variants]
    if not primary_rows:
        raise SystemExit("primary variant filter removed all rows")

    internal_rule = best_threshold_rule(primary_rows, BRANCH_INTERNAL_FEATURES, "internal")
    analysis_rule = best_threshold_rule(primary_rows, ANALYSIS_ONLY_FEATURES, "analysis")
    internal_selected = apply_rule(primary_rows, internal_rule)
    analysis_selected = apply_rule(primary_rows, analysis_rule)
    leave_domain_summary, leave_domain_folds = leave_group_out(
        primary_rows, "domain", BRANCH_INTERNAL_FEATURES, "internal"
    )
    leave_variant_summary, leave_variant_folds = leave_group_out(
        primary_rows, "variant", BRANCH_INTERNAL_FEATURES, "internal"
    )

    policies = [
        summarize_policy(primary_rows, "primary_no_branch", [False] * len(primary_rows)),
        summarize_policy(primary_rows, "primary_all_on", [True] * len(primary_rows)),
        summarize_policy(primary_rows, "primary_oracle_with_side", [bool(row["oracle_select"]) for row in primary_rows]),
        summarize_policy(
            primary_rows,
            f"primary_best_internal_{internal_rule['feature']}_{internal_rule['direction']}",
            internal_selected,
        ),
        summarize_policy(
            primary_rows,
            f"primary_best_analysis_{analysis_rule['feature']}_{analysis_rule['direction']}",
            analysis_selected,
        ),
        leave_domain_summary,
        leave_variant_summary,
    ]

    primary_rows_csv: list[dict[str, Any]] = []
    for row, internal_use, analysis_use in zip(primary_rows, internal_selected, analysis_selected):
        primary_rows_csv.append(
            {
                "domain": row["domain"],
                "variant": row["variant"],
                "image": row["image"],
                "score": row["combined_score"],
                "score_with_side": row["combined_with_side"],
                "oracle_select": row["oracle_select"],
                "internal_select": internal_use,
                "analysis_select": analysis_use,
                "delta_psnr": row["delta_psnr"],
                "delta_ms_ssim": row["delta_ms_ssim"],
                "delta_lpips": row["delta_lpips"],
                "delta_dists": row["delta_dists"],
                "delta_bpp": row["delta_bpp"],
                **{feature: row.get(feature, "") for feature in BRANCH_INTERNAL_FEATURES + ANALYSIS_ONLY_FEATURES},
            }
        )

    group_summaries = group_summary(primary_rows, "domain") + group_summary(rows, "variant")
    separations = feature_separation(primary_rows, BRANCH_INTERNAL_FEATURES, "internal") + feature_separation(
        primary_rows, ANALYSIS_ONLY_FEATURES, "analysis"
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".policies.csv"), policies)
    write_csv(args.output_prefix.with_suffix(".folds.csv"), leave_domain_folds + leave_variant_folds)
    write_csv(args.output_prefix.with_suffix(".rows.csv"), primary_rows_csv)
    write_csv(args.output_prefix.with_suffix(".groups.csv"), group_summaries)
    write_csv(args.output_prefix.with_suffix(".feature_separation.csv"), separations)

    payload = {
        "experiment": "E254 GLC domain-mixed gate readiness",
        "datasets": [{"domain": d, "variant": v, "path": str(p)} for d, v, p in datasets],
        "primary_variants": sorted(primary_variants),
        "score": {
            "formula": "delta_DISTS + lpips_weight * delta_LPIPS + bpp_weight * delta_empirical_bpp",
            "lpips_weight": args.lpips_weight,
            "bpp_weight": args.bpp_weight,
            "side_bits": args.side_bits,
        },
        "policies": policies,
        "group_summaries": group_summaries,
        "best_internal_rule": {k: v for k, v in internal_rule.items() if k != "summary"},
        "best_analysis_rule": {k: v for k, v in analysis_rule.items() if k != "summary"},
        "leave_domain_folds": leave_domain_folds,
        "leave_variant_folds": leave_variant_folds,
        "feature_separation": separations,
        "nonfinite_rows": sum(1 for row in rows if row["nonfinite_bool"]),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_images = sorted(primary_rows, key=lambda r: finite(r["combined_score"]))[:8]
    worst_images = sorted(primary_rows, key=lambda r: finite(r["combined_score"]), reverse=True)[:8]

    lines = [
        "# E254 GLC Domain-Mixed Gate Readiness",
        "",
        "Purpose: test whether the E250 GLC local RVQ branch has a gate signal that survives the Kodak -> CLIC Professional shift.",
        "",
        f"Primary variants: `{', '.join(sorted(primary_variants))}`.",
        f"Score: `delta_DISTS + {args.lpips_weight:g} * delta_LPIPS + {args.bpp_weight:g} * delta_empirical_bpp`; selected branches pay `{args.side_bits:g}` bit/image side overhead.",
        "",
        "## Domain / Variant Summary",
        "",
        "| group | name | rows | positives | all-on score | oracle score | dPSNR | dLPIPS | dDISTS | dbpp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in group_summaries:
        group_key = "domain" if "domain" in item else "variant"
        lines.append(
            f"| {group_key} | {item[group_key]} | {item['rows']} | {item['positives']} | "
            f"{item['always_score']:+.6f} | {item['oracle_score']:+.6f} | "
            f"{item['always_dpsnr']:+.6f} | {item['always_dlpips']:+.6f} | "
            f"{item['always_ddists']:+.6f} | {item['always_dbpp']:+.6f} |"
        )

    lines += [
        "",
        "## Primary Policy Summary",
        "",
        "| policy | selected | mean score | dPSNR | dMS-SSIM | dLPIPS | dDISTS | dbpp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in policies:
        lines.append(
            f"| {item['policy']} | {item['selected']}/{item['total']} | {item['mean_combined']:+.6f} | "
            f"{item['mean_delta_psnr']:+.6f} | {item['mean_delta_ms_ssim']:+.6f} | "
            f"{item['mean_delta_lpips']:+.6f} | {item['mean_delta_dists']:+.6f} | "
            f"{item['mean_delta_bpp']:+.6f} |"
        )

    lines += [
        "",
        "## Best Rules",
        "",
        f"- Best branch-internal rule: `{internal_rule['feature']} {internal_rule['direction']} {internal_rule['threshold']:.6g}` with score `{internal_rule['summary']['mean_combined']:+.6f}`.",
        f"- Best analysis-only rule: `{analysis_rule['feature']} {analysis_rule['direction']} {analysis_rule['threshold']:.6g}` with score `{analysis_rule['summary']['mean_combined']:+.6f}`.",
        "",
        "## Leave-Domain / Leave-Variant Folds",
        "",
        "| heldout | feature | rule | threshold | selected | mean score |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in leave_domain_folds + leave_variant_folds:
        lines.append(
            f"| {item['heldout_group']} | {item['feature']} | {item['direction']} | "
            f"{item['threshold']:.6g} | {item['selected']}/{item['total']} | {item['mean_combined']:+.6f} |"
        )

    lines += [
        "",
        "## Strongest Beneficial Images",
        "",
        "| domain | variant | image | score | dPSNR | dLPIPS | dDISTS | dbpp |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_images:
        lines.append(
            f"| {row['domain']} | {row['variant']} | {row['image']} | {finite(row['combined_score']):+.6f} | "
            f"{finite(row['delta_psnr']):+.6f} | {finite(row['delta_lpips']):+.6f} | "
            f"{finite(row['delta_dists']):+.6f} | {finite(row['delta_bpp']):+.6f} |"
        )
    lines += [
        "",
        "## Worst Harmful Images",
        "",
        "| domain | variant | image | score | dPSNR | dLPIPS | dDISTS | dbpp |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst_images:
        lines.append(
            f"| {row['domain']} | {row['variant']} | {row['image']} | {finite(row['combined_score']):+.6f} | "
            f"{finite(row['delta_psnr']):+.6f} | {finite(row['delta_lpips']):+.6f} | "
            f"{finite(row['delta_dists']):+.6f} | {finite(row['delta_bpp']):+.6f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The primary question is not whether local RVQ can improve some images; E251 already showed that it can on Kodak. "
        "This audit asks whether a simple branch-internal gate can carry that decision across the CLIC Professional shift. "
        "If leave-domain-out remains weak while the oracle remains strong on Kodak, the next model should not be an all-on full-training run. "
        "It should be a hyperprior/index-prior reliability controller trained with domain-mixed calibration and a DISTS validation guard.",
        "",
    ]
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"output_prefix": str(args.output_prefix), "policies": policies}, indent=2))


if __name__ == "__main__":
    main()
