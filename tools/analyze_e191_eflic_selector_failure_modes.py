#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005.csv"),
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005_feature_manifest.csv"),
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e191_eflic_force0_e190_rule_failure_modes"),
    )
    p.add_argument("--force", type=int, default=0)
    p.add_argument("--selector-feature", default="slice0_mean_min")
    p.add_argument("--selector-op", choices=[">=", "<="], default=">=")
    p.add_argument("--selector-threshold", type=float, default=-10.7447786331)
    p.add_argument("--feature-set", default="global_predecision_context")
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=3.0)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--top-features", type=int, default=18)
    return p.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def decision(row: dict[str, Any], feature: str, op: str, threshold: float) -> bool:
    value = finite_float(row.get(feature))
    if value is None:
        return False
    return value >= threshold if op == ">=" else value <= threshold


def active_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    dd = float(row["delta_dists"])
    dl = float(row["delta_lpips"])
    dp = float(row["delta_psnr"])
    return args.dists_weight * dd + args.lpips_weight * dl - args.psnr_weight * dp


def selected_delta(row: dict[str, Any], selected: bool, key: str) -> float:
    return float(row[key]) if selected else 0.0


def summarize_group(name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "group": name,
            "images": 0,
            "branch_share": 0.0,
            "mean_delta_dists": 0.0,
            "mean_delta_lpips": 0.0,
            "mean_delta_psnr": 0.0,
            "mean_active_score": 0.0,
            "dists_wins": 0,
            "lpips_wins": 0,
            "both_wins": 0,
        }
    return {
        "group": name,
        "images": n,
        "branch_share": mean(float(r["selected"]) for r in rows),
        "mean_delta_dists": mean(float(r["selected_delta_dists"]) for r in rows),
        "mean_delta_lpips": mean(float(r["selected_delta_lpips"]) for r in rows),
        "mean_delta_psnr": mean(float(r["selected_delta_psnr"]) for r in rows),
        "mean_active_score": mean(float(r["active_score"]) for r in rows),
        "dists_wins": sum(float(r["selected_delta_dists"]) < 0 for r in rows),
        "lpips_wins": sum(float(r["selected_delta_lpips"]) < 0 for r in rows),
        "both_wins": sum(float(r["selected_delta_dists"]) < 0 and float(r["selected_delta_lpips"]) < 0 for r in rows),
    }


def numeric_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    vals = []
    for row in rows:
        value = finite_float(row.get(feature))
        if value is not None:
            vals.append(value)
    return vals


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def feature_separation(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    selected = [r for r in rows if r["selected"]]
    fallback = [r for r in rows if not r["selected"]]
    scored_pairs = []
    for feature in features:
        all_vals = numeric_values(rows, feature)
        sel_vals = numeric_values(selected, feature)
        fb_vals = numeric_values(fallback, feature)
        if len(all_vals) < len(rows) or len(sel_vals) < 2 or len(fb_vals) < 2:
            continue
        mu_all = mean(all_vals)
        var_all = mean((v - mu_all) ** 2 for v in all_vals)
        std_all = math.sqrt(var_all)
        if std_all <= 0:
            continue
        scores = [float(r["active_score"]) for r in rows]
        vals = [float(r[feature]) for r in rows]
        entry = {
            "feature": feature,
            "selected_mean": mean(sel_vals),
            "fallback_mean": mean(fb_vals),
            "mean_gap": mean(sel_vals) - mean(fb_vals),
            "std_gap": (mean(sel_vals) - mean(fb_vals)) / std_all,
            "corr_active_score": pearson(vals, scores),
        }
        scored_pairs.append(entry)
    scored_pairs.sort(key=lambda r: abs(float(r["std_gap"])) + abs(float(r["corr_active_score"])), reverse=True)
    return scored_pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def main() -> None:
    args = parse_args()
    rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in rows if int(float(r["force_ind"])) == args.force]
    if not rows:
        raise SystemExit(f"no finite rows for force{args.force}")

    manifest = e184.read_manifest(args.manifest_csv)
    feature_sets = e185.feature_sets(manifest)
    if args.feature_set not in feature_sets:
        raise SystemExit(f"unknown feature set: {args.feature_set}")
    features = feature_sets[args.feature_set][0]

    per_image = []
    for row in rows:
        selected = decision(row, args.selector_feature, args.selector_op, args.selector_threshold)
        score = active_score(row, args)
        active_beneficial = score < 0.0
        selected_dd = selected_delta(row, selected, "delta_dists")
        selected_dl = selected_delta(row, selected, "delta_lpips")
        selected_dp = selected_delta(row, selected, "delta_psnr")
        if selected and active_beneficial:
            category = "selected_good"
        elif selected and not active_beneficial:
            category = "selected_bad"
        elif (not selected) and active_beneficial:
            category = "missed_good"
        else:
            category = "fallback_ok"
        out = {
            "image": row["image"],
            "selected": int(selected),
            "category": category,
            "active_score": score,
            "delta_dists": float(row["delta_dists"]),
            "delta_lpips": float(row["delta_lpips"]),
            "delta_psnr": float(row["delta_psnr"]),
            "selected_delta_dists": selected_dd,
            "selected_delta_lpips": selected_dl,
            "selected_delta_psnr": selected_dp,
            "base_dists": float(row["base_dists"]),
            "active_dists": float(row["active_dists"]),
            "base_lpips": float(row["base_lpips"]),
            "active_lpips": float(row["active_lpips"]),
        }
        for feature in sorted(set(features + [args.selector_feature])):
            value = finite_float(row.get(feature))
            if value is not None:
                out[feature] = value
        per_image.append(out)

    per_image.sort(key=lambda r: (str(r["category"]), float(r["active_score"])))

    groups = {
        "all_selected_policy": per_image,
        "selected_branch": [r for r in per_image if r["selected"]],
        "fallback_branch": [r for r in per_image if not r["selected"]],
        "selected_good": [r for r in per_image if r["category"] == "selected_good"],
        "selected_bad": [r for r in per_image if r["category"] == "selected_bad"],
        "missed_good": [r for r in per_image if r["category"] == "missed_good"],
        "fallback_ok": [r for r in per_image if r["category"] == "fallback_ok"],
    }
    group_rows = [summarize_group(name, part, args) for name, part in groups.items()]
    sep_rows = feature_separation(per_image, features)[: args.top_features]

    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_per_image.csv"), per_image)
    write_csv(prefix.with_name(prefix.name + "_groups.csv"), group_rows)
    write_csv(prefix.with_name(prefix.name + "_feature_separation.csv"), sep_rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "groups": group_rows,
                "feature_separation": sep_rows,
                "per_image": per_image,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    selected = next(r for r in group_rows if r["group"] == "selected_branch")
    fallback = next(r for r in group_rows if r["group"] == "fallback_branch")
    all_policy = next(r for r in group_rows if r["group"] == "all_selected_policy")
    selected_bad = next(r for r in group_rows if r["group"] == "selected_bad")
    missed_good = next(r for r in group_rows if r["group"] == "missed_good")

    lines = [
        "# E191 EF-LIC Selector Rule Failure-Mode Analysis",
        "",
        "This diagnostic explains a no-side-bit global predecision rule at per-image and feature-distribution level. It is still a Kodak diagnostic, not a paper table.",
        "",
        f"Rule: `{args.selector_feature} {args.selector_op} {args.selector_threshold:.10g}`",
        f"Objective weights: DISTS `{args.dists_weight}`, LPIPS `{args.lpips_weight}`, PSNR `{args.psnr_weight}`",
        "",
        "| group | images | branch share | dDISTS | dLPIPS | dPSNR | DISTS wins | LPIPS wins | both wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {row['images']} | {row['branch_share']:.3f} | "
            f"{fmt(row['mean_delta_dists'])} | {fmt(row['mean_delta_lpips'])} | {fmt(row['mean_delta_psnr'])} | "
            f"{row['dists_wins']} | {row['lpips_wins']} | {row['both_wins']} |"
        )
    lines.extend(
        [
            "",
            "Key readout:",
            "",
            f"- The selected policy averages `dDISTS={fmt(all_policy['mean_delta_dists'])}` and `dLPIPS={fmt(all_policy['mean_delta_lpips'])}` over all `{all_policy['images']}` images.",
            f"- It activates on `{selected['images']}` images and falls back on `{fallback['images']}` images.",
            f"- Selected-but-bad cases under the weighted objective: `{selected_bad['images']}`.",
            f"- Missed-good cases under the weighted objective: `{missed_good['images']}`.",
            "",
            "Top feature separations between selected and fallback images:",
            "",
            "| feature | selected mean | fallback mean | std gap | corr(active score) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sep_rows[:12]:
        lines.append(
            f"| {row['feature']} | {row['selected_mean']:.6f} | {row['fallback_mean']:.6f} | "
            f"{row['std_gap']:+.3f} | {row['corr_active_score']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The rule is deployable because it uses `global_predecision_context`, available before the whole-image active/fallback decision.",
            "- `selected_bad` and `missed_good` rows are the immediate target for a learned decoder-side reliability head or a second scalar feature.",
            "- The feature-separation table identifies which hyperprior/early-slice statistics explain the reliability boundary and should be monitored on independent data.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
