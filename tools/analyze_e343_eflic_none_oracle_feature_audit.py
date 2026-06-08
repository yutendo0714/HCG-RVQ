#!/usr/bin/env python3
"""Audit decoder-safe features for EF-LIC HCG no-op/fallback control.

E317 gives a per-image oracle over EF-LIC HCG slice subsets.  E338/E335/etc.
show that the learned controller improves the mean but still has unsafe tails.
This script joins the oracle with codec-loop controller outputs and asks whether
decoder-available summary features can select a no-op fallback per image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONTROLLER_CSVS = [
    ROOT / "experiments/analysis/e338_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_risknone.csv",
    ROOT / "experiments/analysis/e339_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm005.csv",
    ROOT / "experiments/analysis/e340_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm010.csv",
    ROOT / "experiments/analysis/e335_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm015.csv",
    ROOT / "experiments/analysis/e337_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm020.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.by_image.csv",
    )
    parser.add_argument("--controller-csv", type=Path, nargs="+", default=DEFAULT_CONTROLLER_CSVS)
    parser.add_argument("--mode", default="trained_hard")
    parser.add_argument("--unsafe-threshold", type=float, default=0.0)
    parser.add_argument("--tail-floor", type=float, default=-0.02)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e343_eflic_none_oracle_feature_audit_kodak24",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy))


def auc_score(xs: list[float], labels: list[int]) -> float:
    pairs = [(x, y) for x, y in zip(xs, labels) if math.isfinite(x)]
    pos = [x for x, y in pairs if y]
    neg = [x for x, y in pairs if not y]
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


def count_csv_list(value: str) -> int:
    value = (value or "").strip()
    if not value:
        return 0
    if value == "all":
        return 4
    if value == "none":
        return 0
    return len([part for part in value.split(",") if part.strip()])


def read_oracle(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            image = row["image"]
            best = (row.get("best_slice_set") or "").strip()
            rows[image] = {
                "image": image,
                "oracle_best_slice_set": best,
                "oracle_none": int(best == "none"),
                "oracle_active_count": count_csv_list(best),
                "oracle_all_delta_psnr": safe_float(row.get("all_delta_psnr")),
                "oracle_best_delta_psnr": safe_float(row.get("best_delta_psnr")),
                "oracle_best_gain_over_all": safe_float(row.get("best_gain_over_all")),
                "oracle_worst_delta_psnr": safe_float(row.get("worst_delta_psnr")),
                "oracle_positive_single_count": count_csv_list(row.get("positive_single_slices", "")),
                "oracle_negative_single_count": count_csv_list(row.get("negative_single_slices", "")),
            }
    return rows


def run_name(path: Path) -> str:
    stem = path.stem
    if "risknone" in stem:
        return "risk_none"
    for token in ["riskm005", "riskm010", "riskm015", "riskm020"]:
        if token in stem:
            return token.replace("riskm", "risk_-0.")
    return stem


def slice_stat(row: dict[str, str], suffix: str, reducer: str) -> float:
    vals = [safe_float(row.get(f"slice{i}_{suffix}")) for i in range(4)]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    if reducer == "min":
        return min(vals)
    if reducer == "max":
        return max(vals)
    return mean(vals)


def extract_features(row: dict[str, str]) -> dict[str, float]:
    features: dict[str, float] = {}
    keep = [
        "bpp",
        "y_active_logit_mean",
        "y_alpha_active_frac",
        "y_alpha_max",
        "y_alpha_mean",
        "y_avg_geometry_delta_rms",
        "y_avg_index_entropy",
        "y_avg_index_used_frac",
        "y_avg_residual_error_rms",
        "y_family_zero_prob_mean",
        "y_gate_max",
        "y_gate_mean",
        "y_local_score_mean",
        "y_risk_score_mean",
        "y_strength_mean",
        "z_hat_rms",
        "z_index_entropy",
        "z_index_used_frac",
    ]
    for key in keep:
        if key in row:
            features[key] = safe_float(row.get(key))
    for suffix in [
        "risk_score_mean",
        "gate_mean",
        "alpha_mean",
        "avg_index_entropy",
        "avg_residual_error_rms",
        "avg_geometry_delta_rms",
    ]:
        features[f"slice_{suffix}_min"] = slice_stat(row, suffix, "min")
        features[f"slice_{suffix}_max"] = slice_stat(row, suffix, "max")
        features[f"slice_{suffix}_mean"] = slice_stat(row, suffix, "mean")
    return features


def read_controller(path: Path, *, mode: str, oracle: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            if row.get("mode") != mode:
                continue
            image = row["image"]
            if image not in oracle:
                continue
            features = extract_features(row)
            delta = safe_float(row.get("delta_psnr"))
            rows.append(
                {
                    **oracle[image],
                    "run": run_name(path),
                    "source_csv": str(path),
                    "active_slices": row.get("active_slices", ""),
                    "controller_delta_psnr": delta,
                    "controller_unsafe": int(math.isfinite(delta) and delta < 0.0),
                    "controller_tail_unsafe": int(math.isfinite(delta) and delta < -0.02),
                    **features,
                }
            )
    return rows


def evaluate_policy(rows: list[dict[str, Any]], feature: str, threshold: float, direction: str) -> dict[str, Any]:
    deltas: list[float] = []
    suppressed = 0
    suppressed_bad = 0
    suppressed_good = 0
    for row in rows:
        value = row.get(feature, float("nan"))
        if not math.isfinite(value):
            use_fallback = False
        elif direction == "<=":
            use_fallback = value <= threshold
        else:
            use_fallback = value >= threshold
        original = row["controller_delta_psnr"]
        if use_fallback:
            suppressed += 1
            if original < 0:
                suppressed_bad += 1
            elif original > 0:
                suppressed_good += 1
            deltas.append(0.0)
        else:
            deltas.append(original)
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "mean_delta_psnr": mean(deltas),
        "worst_delta_psnr": min(deltas) if deltas else float("nan"),
        "negative_count": sum(1 for v in deltas if v < 0.0),
        "positive_count": sum(1 for v in deltas if v > 0.0),
        "suppressed_count": suppressed,
        "suppressed_negative_count": suppressed_bad,
        "suppressed_positive_count": suppressed_good,
    }


def threshold_candidates(values: list[float]) -> list[float]:
    vals = sorted({v for v in values if math.isfinite(v)})
    if not vals:
        return []
    candidates = [vals[0] - 1e-9, vals[-1] + 1e-9]
    candidates.extend(vals)
    candidates.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(candidates))


def audit_features(rows: list[dict[str, Any]], *, tail_floor: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_names = sorted(
        key
        for key, value in rows[0].items()
        if isinstance(value, float)
        and key not in {"controller_delta_psnr"}
        and not key.startswith("oracle_")
    )
    feature_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = [row.get(feature, float("nan")) for row in rows]
        oracle_none = [int(row["oracle_none"]) for row in rows]
        negative = [int(row["controller_delta_psnr"] < 0.0) for row in rows]
        feature_rows.append(
            {
                "feature": feature,
                "corr_oracle_none": pearson(values, [float(x) for x in oracle_none]),
                "auc_oracle_none": auc_score(values, oracle_none),
                "corr_controller_negative": pearson(values, [float(x) for x in negative]),
                "auc_controller_negative": auc_score(values, negative),
                "mean_if_oracle_none": mean([v for v, y in zip(values, oracle_none) if y]),
                "mean_if_oracle_active": mean([v for v, y in zip(values, oracle_none) if not y]),
                "mean_if_negative": mean([v for v, y in zip(values, negative) if y]),
                "mean_if_nonnegative": mean([v for v, y in zip(values, negative) if not y]),
            }
        )
        best_by_mean: dict[str, Any] | None = None
        best_safe: dict[str, Any] | None = None
        for direction in ["<=", ">="]:
            for threshold in threshold_candidates(values):
                result = evaluate_policy(rows, feature, threshold, direction)
                if best_by_mean is None or result["mean_delta_psnr"] > best_by_mean["mean_delta_psnr"]:
                    best_by_mean = result
                if result["worst_delta_psnr"] >= tail_floor:
                    if best_safe is None or result["mean_delta_psnr"] > best_safe["mean_delta_psnr"]:
                        best_safe = result
        if best_by_mean is not None:
            policy_rows.append({"selection": "best_mean", **best_by_mean})
        if best_safe is not None:
            policy_rows.append({"selection": f"best_mean_with_worst_ge_{tail_floor}", **best_safe})
    feature_rows.sort(
        key=lambda row: max(abs(row.get("corr_oracle_none", 0.0) or 0.0), abs(row.get("corr_controller_negative", 0.0) or 0.0)),
        reverse=True,
    )
    policy_rows.sort(
        key=lambda row: (row["selection"], -(row["mean_delta_psnr"] if math.isfinite(row["mean_delta_psnr"]) else -999.0))
    )
    return feature_rows, policy_rows


def summarize_run(rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["controller_delta_psnr"] for row in rows]
    oracle_best = [row["oracle_best_delta_psnr"] for row in rows]
    safe_policies = [row for row in policies if row["selection"].startswith("best_mean_with")]
    mean_policies = [row for row in policies if row["selection"] == "best_mean"]
    best_safe = max(safe_policies, key=lambda row: row["mean_delta_psnr"], default=None)
    best_mean = max(mean_policies, key=lambda row: row["mean_delta_psnr"], default=None)
    return {
        "records": len(rows),
        "controller_mean_delta_psnr": mean(deltas),
        "controller_worst_delta_psnr": min(deltas),
        "controller_negative_count": sum(1 for value in deltas if value < 0.0),
        "oracle_mean_best_delta_psnr": mean(oracle_best),
        "oracle_none_count": sum(int(row["oracle_none"]) for row in rows),
        "best_mean_policy": best_mean,
        "best_safe_policy": best_safe,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    oracle = read_oracle(args.oracle_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_joined: list[dict[str, Any]] = []
    run_payloads: dict[str, dict[str, Any]] = {}
    for csv_path in args.controller_csv:
        rows = read_controller(csv_path, mode=args.mode, oracle=oracle)
        if not rows:
            continue
        run = rows[0]["run"]
        feature_rows, policy_rows = audit_features(rows, tail_floor=args.tail_floor)
        run_payloads[run] = {
            "summary": summarize_run(rows, policy_rows),
            "feature_audit": feature_rows,
            "policy_audit": policy_rows,
        }
        all_joined.extend(rows)

    joined_path = args.output_prefix.with_suffix(".joined.csv")
    feature_path = args.output_prefix.with_suffix(".features.csv")
    policy_path = args.output_prefix.with_suffix(".policies.csv")
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    write_csv(joined_path, all_joined)
    write_csv(
        feature_path,
        [
            {"run": run, **row}
            for run, payload in run_payloads.items()
            for row in payload["feature_audit"]
        ],
    )
    write_csv(
        policy_path,
        [
            {"run": run, **row}
            for run, payload in run_payloads.items()
            for row in payload["policy_audit"]
        ],
    )
    summary_rows = [{"run": run, **payload["summary"]} for run, payload in run_payloads.items()]
    write_csv(summary_path, summary_rows)

    payload = {
        "experiment": "E343 EF-LIC no-op oracle feature audit",
        "purpose": "Join E317 subset oracle with E329 codec-loop controller features to identify decoder-safe no-op/fallback control signals.",
        "args": {
            "oracle_csv": str(args.oracle_csv),
            "controller_csv": [str(path) for path in args.controller_csv],
            "mode": args.mode,
            "tail_floor": args.tail_floor,
        },
        "runs": run_payloads,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E343 EF-LIC No-op Oracle Feature Audit\n\n")
        fobj.write(
            "This audit is not final paper evidence. It checks whether decoder-safe controller features can "
            "recover a per-image no-op fallback suggested by the E317 slice-subset oracle.\n\n"
        )
        fobj.write("| run | mean ΔPSNR | worst ΔPSNR | neg | oracle none | best safe policy |\n")
        fobj.write("|---|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            safe = row.get("best_safe_policy") or {}
            safe_text = (
                f"{safe.get('feature')} {safe.get('direction')} {safe.get('threshold'):.6g}; "
                f"mean {safe.get('mean_delta_psnr'):.6f}, worst {safe.get('worst_delta_psnr'):.6f}, "
                f"suppress {safe.get('suppressed_count')}"
            ) if safe else "none"
            fobj.write(
                f"| {row['run']} | {row['controller_mean_delta_psnr']:.6f} | "
                f"{row['controller_worst_delta_psnr']:.6f} | {row['controller_negative_count']} | "
                f"{row['oracle_none_count']} | {safe_text} |\n"
            )
        fobj.write("\nTop feature correlations by run:\n\n")
        for run, payload_item in run_payloads.items():
            fobj.write(f"## {run}\n\n")
            fobj.write("| feature | corr oracle-none | auc oracle-none | corr negative | auc negative |\n")
            fobj.write("|---|---:|---:|---:|---:|\n")
            for row in payload_item["feature_audit"][:8]:
                fobj.write(
                    f"| {row['feature']} | {row['corr_oracle_none']:.6f} | {row['auc_oracle_none']:.6f} | "
                    f"{row['corr_controller_negative']:.6f} | {row['auc_controller_negative']:.6f} |\n"
                )
            fobj.write("\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write(
            "- A useful HCG fallback signal should suppress negative controller deltas while losing little positive headroom.\n"
        )
        fobj.write(
            "- If the best safe policy uses a single fragile threshold, the next step should be a learned no-op head trained on E318/E317-aligned labels rather than a paper claim.\n"
        )
    print(f"wrote {json_path}, {md_path}, {joined_path}, {feature_path}, {policy_path}, {summary_path}")


if __name__ == "__main__":
    main()
