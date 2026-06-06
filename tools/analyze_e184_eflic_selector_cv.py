#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


LOWER_BETTER = {"dists", "lpips", "bpp"}
HIGHER_BETTER = {"psnr"}


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
        default=Path("experiments/analysis/e184_eflic_projected_hcg_selector_cv_dists"),
    )
    p.add_argument("--target", choices=["dists", "lpips", "psnr"], default="dists")
    p.add_argument("--side-bits", type=float, default=1.0, help="Image-level signaled selector bits.")
    p.add_argument(
        "--loocv-feature-topk",
        type=int,
        default=32,
        help="Preselect this many in-sample threshold features before leave-one-out refitting.",
    )
    p.add_argument(
        "--include-all-forces",
        action="store_true",
        help="Also run the expensive aggregate all-forces selector audit.",
    )
    return p.parse_args()


def parse_value(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = [{k: parse_value(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for row in rows:
        row["force_ind"] = int(float(row["force_ind"]))
    return rows


def read_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["feature"]: r["class"] for r in csv.DictReader(f)}


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric(row: dict[str, Any], branch: str, name: str) -> float:
    if name == "bpp":
        return float(row["bpp"])
    prefix = "active" if branch == "active" else "base"
    return float(row[f"{prefix}_{name}"])


def selected_metric(row: dict[str, Any], use_active: bool, name: str, side_bits: float) -> float:
    if name == "bpp":
        return float(row["bpp"]) + side_bits * float(row["one_bit_bpp_cost_512x768"])
    return metric(row, "active" if use_active else "base", name)


def summarize_policy(
    group: str,
    selector: str,
    rows: list[dict[str, Any]],
    decisions: list[bool],
    side_bits: float,
    target: str,
    feature_set: str = "",
    rule: str = "",
) -> dict[str, Any]:
    n = max(1, len(rows))
    out: dict[str, Any] = {
        "group": group,
        "target": target,
        "selector": selector,
        "feature_set": feature_set,
        "rule": rule,
        "images": len(rows),
        "selector_side_bits": side_bits,
        "branch_share": sum(1 for x in decisions if x) / n,
    }
    for name in ("bpp", "psnr", "lpips", "dists"):
        base_vals = [metric(r, "base", name) for r in rows]
        active_vals = [metric(r, "active", name) if name != "bpp" else metric(r, "base", "bpp") for r in rows]
        sel_vals = [selected_metric(r, d, name, side_bits) for r, d in zip(rows, decisions)]
        out[f"base_{name}"] = sum(base_vals) / n
        out[f"active_{name}"] = sum(active_vals) / n
        out[f"selected_{name}"] = sum(sel_vals) / n
        out[f"selected_delta_{name}"] = out[f"selected_{name}"] - out[f"base_{name}"]
        out[f"active_delta_{name}"] = out[f"active_{name}"] - out[f"base_{name}"]
    out["active_win_dists"] = sum(1 for r in rows if metric(r, "active", "dists") < metric(r, "base", "dists"))
    out["active_win_lpips"] = sum(1 for r in rows if metric(r, "active", "lpips") < metric(r, "base", "lpips"))
    out["active_win_psnr"] = sum(1 for r in rows if metric(r, "active", "psnr") > metric(r, "base", "psnr"))
    out["selected_win_dists"] = sum(
        1 for r, d in zip(rows, decisions) if selected_metric(r, d, "dists", 0.0) < metric(r, "base", "dists")
    )
    return out


def oracle(rows: list[dict[str, Any]], target: str) -> list[bool]:
    decisions: list[bool] = []
    for row in rows:
        base = metric(row, "base", target)
        active = metric(row, "active", target)
        decisions.append(active < base if target in LOWER_BETTER else active > base)
    return decisions


def strict_oracle(rows: list[dict[str, Any]]) -> list[bool]:
    return [
        metric(row, "active", "dists") < metric(row, "base", "dists")
        and metric(row, "active", "lpips") < metric(row, "base", "lpips")
        for row in rows
    ]


def objective(rows: list[dict[str, Any]], decisions: list[bool], target: str, side_bits: float) -> float:
    mean = sum(selected_metric(r, d, target, side_bits) for r, d in zip(rows, decisions)) / max(1, len(rows))
    return mean if target in LOWER_BETTER else -mean


def candidate_thresholds(values: list[float]) -> list[float]:
    finite = sorted(set(v for v in values if math.isfinite(v)))
    if not finite:
        return []
    if len(finite) == 1:
        eps = max(1e-12, abs(finite[0]) * 1e-6)
        return [finite[0] - eps, finite[0] + eps]
    mids = [(a + b) * 0.5 for a, b in zip(finite, finite[1:])]
    eps = max(1e-12, (finite[-1] - finite[0]) * 1e-6)
    return [finite[0] - eps, *mids, finite[-1] + eps]


def threshold_decisions(rows: list[dict[str, Any]], feature: str, threshold: float, direction: str) -> list[bool]:
    decisions: list[bool] = []
    for row in rows:
        value = float(row[feature])
        decisions.append(value >= threshold if direction == ">=" else value <= threshold)
    return decisions


def valid_features(rows: list[dict[str, Any]], features: list[str]) -> list[str]:
    out: list[str] = []
    for feature in features:
        if feature not in rows[0]:
            continue
        vals = [float(r[feature]) for r in rows if finite_float(r.get(feature))]
        if len(vals) != len(rows):
            continue
        if len(set(vals)) <= 1:
            continue
        out.append(feature)
    return out


def best_threshold(
    rows: list[dict[str, Any]],
    features: list[str],
    target: str,
    side_bits: float,
) -> tuple[str, str, float, list[bool], float]:
    best: tuple[str, str, float, list[bool], float] | None = None
    for feature in valid_features(rows, features):
        values = [float(r[feature]) for r in rows]
        for threshold in candidate_thresholds(values):
            for direction in (">=", "<="):
                decisions = threshold_decisions(rows, feature, threshold, direction)
                score = objective(rows, decisions, target, side_bits)
                if best is None or score < best[4]:
                    best = (feature, direction, threshold, decisions, score)
    if best is None:
        return ("", ">=", float("nan"), [False] * len(rows), objective(rows, [False] * len(rows), target, side_bits))
    return best


def ranked_threshold_features(
    rows: list[dict[str, Any]],
    features: list[str],
    target: str,
    side_bits: float,
    topk: int,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for feature in valid_features(rows, features):
        best_score: float | None = None
        values = [float(r[feature]) for r in rows]
        for threshold in candidate_thresholds(values):
            for direction in (">=", "<="):
                decisions = threshold_decisions(rows, feature, threshold, direction)
                score = objective(rows, decisions, target, side_bits)
                if best_score is None or score < best_score:
                    best_score = score
        if best_score is not None:
            ranked.append((best_score, feature))
    ranked.sort(key=lambda x: x[0])
    return [feature for _, feature in ranked[: max(1, topk)]]


def loocv_threshold(
    rows: list[dict[str, Any]],
    features: list[str],
    target: str,
    side_bits: float,
    feature_topk: int,
) -> tuple[list[bool], str]:
    decisions: list[bool] = []
    rules: list[str] = []
    refit_features = ranked_threshold_features(rows, features, target, side_bits, feature_topk)
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1 :]
        feature, direction, threshold, _, _ = best_threshold(train, refit_features, target, side_bits)
        if not feature:
            decisions.append(False)
            rules.append("baseline")
            continue
        decision = threshold_decisions([row], feature, threshold, direction)[0]
        decisions.append(decision)
        rules.append(f"{feature} {direction} {threshold:.6g}")
    unique_rules = sorted(set(rules))
    if len(unique_rules) <= 5:
        return decisions, "; ".join(unique_rules)
    return decisions, f"{len(unique_rules)} fold-specific rules"


def make_feature_sets(manifest: dict[str, str]) -> dict[str, tuple[list[str], float, str]]:
    decoder_safe = [f for f, cls in manifest.items() if cls == "decoder_safe_context"]
    encoder_diag = [f for f, cls in manifest.items() if cls == "encoder_or_active_diagnostic"]
    return {
        "decoder_safe_context": (decoder_safe, 0.0, "decoder-reproducible no-side-bit threshold"),
        "encoder_active_diagnostic": (encoder_diag, 1.0, "encoder/active diagnostic threshold with one signaled bit"),
    }


def analyze_group(
    group: str,
    rows: list[dict[str, Any]],
    feature_sets: dict[str, tuple[list[str], float, str]],
    target: str,
    side_bits: float,
    loocv_feature_topk: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    results.append(summarize_policy(group, "baseline", rows, [False] * len(rows), 0.0, target))
    results.append(summarize_policy(group, "always_active", rows, [True] * len(rows), 0.0, target))
    results.append(summarize_policy(group, f"oracle_{target}", rows, oracle(rows, target), side_bits, target, "metric_oracle"))
    results.append(summarize_policy(group, "oracle_dists_and_lpips", rows, strict_oracle(rows), side_bits, target, "metric_oracle"))
    for feature_set, (features, feature_side_bits, description) in feature_sets.items():
        actual_side_bits = side_bits * feature_side_bits
        feature, direction, threshold, decisions, _ = best_threshold(rows, features, target, actual_side_bits)
        rule = f"{feature} {direction} {threshold:.6g}" if feature else "baseline"
        row = summarize_policy(
            group,
            f"best_threshold_{target}",
            rows,
            decisions,
            actual_side_bits,
            target,
            feature_set,
            rule,
        )
        row["feature_set_description"] = description
        row["candidate_features"] = len(valid_features(rows, features))
        row["loocv_feature_topk"] = loocv_feature_topk
        results.append(row)

        loocv_decisions, loocv_rule = loocv_threshold(
            rows, features, target, actual_side_bits, loocv_feature_topk
        )
        row = summarize_policy(
            group,
            f"loocv_threshold_{target}",
            rows,
            loocv_decisions,
            actual_side_bits,
            target,
            feature_set,
            loocv_rule,
        )
        row["feature_set_description"] = description
        row["candidate_features"] = len(valid_features(rows, features))
        row["loocv_feature_topk"] = loocv_feature_topk
        results.append(row)
    return results


def write_outputs(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E184 EF-LIC Projected-HCG Selector CV",
        "",
        "This audit reuses the E160/E161 EF-LIC projected-HCG active-branch labels. It separates deployable decoder-safe thresholding from encoder/active diagnostic upper bounds, so the result can guide the next reliability controller rather than become a paper-quality codec row by itself.",
        "",
        f"Target metric for threshold selection: `{args.target}`",
        f"Signaled selector cost for non-decoder-safe policies: `{args.side_bits}` bit per image.",
        "",
        "| group | selector | feature set | branch share | dbpp | dDISTS | dLPIPS | dPSNR | base DISTS | selected DISTS | active DISTS | DISTS wins | rule |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row.get('feature_set', '')} | "
            f"{row['branch_share']:.3f} | {row['selected_delta_bpp']:+.6f} | "
            f"{row['selected_delta_dists']:+.6f} | {row['selected_delta_lpips']:+.6f} | {row['selected_delta_psnr']:+.6f} | "
            f"{row['base_dists']:.5f} | {row['selected_dists']:.5f} | {row['active_dists']:.5f} | "
            f"{row['selected_win_dists']}/{row['images']} | {row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "",
            "- `oracle_*` rows are diagnostic upper bounds because they use evaluation metrics as labels.",
            "- `decoder_safe_context` thresholds are the only no-side-bit deployable scalar rules in this audit.",
            "- `encoder_active_diagnostic` thresholds are useful for failure analysis, but would need explicit signaling or a learned decoder-side proxy before a paper claim.",
            f"- LOOCV refits preselect the top `{args.loocv_feature_topk}` scalar-threshold features in-sample before fold refitting, to keep this audit tractable.",
            f"- Aggregate all-forces audit included: `{args.include_all_forces}`.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    rows = [r for r in read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    if not rows:
        raise SystemExit("no finite rows")
    manifest = read_manifest(args.manifest_csv)
    feature_sets = make_feature_sets(manifest)

    results: list[dict[str, Any]] = []
    for force in sorted({int(r["force_ind"]) for r in rows}):
        subset = [r for r in rows if int(r["force_ind"]) == force]
        results.extend(
            analyze_group(f"force{force}", subset, feature_sets, args.target, args.side_bits, args.loocv_feature_topk)
        )
    if args.include_all_forces:
        results.extend(
            analyze_group("all_forces", rows, feature_sets, args.target, args.side_bits, args.loocv_feature_topk)
        )
    write_outputs(args.output_prefix, results, args)


if __name__ == "__main__":
    main()
