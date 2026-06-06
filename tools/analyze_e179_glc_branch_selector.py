#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOWER_BETTER = {"dists", "lpips", "bpp"}
HIGHER_BETTER = {"psnr", "ms_ssim"}
FEATURES = (
    "active_mse_ratio",
    "empirical_bpp_delta",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "base_dists",
    "base_lpips",
    "base_psnr",
    "base_ms_ssim",
)


@dataclass
class EvalResult:
    name: str
    rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", type=Path, nargs="+")
    p.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e179_glc_branch_selector"))
    p.add_argument("--side-bits", type=float, default=1.0, help="Signaled selector bits per image.")
    p.add_argument("--target", default="dists", choices=["dists", "lpips", "psnr", "ms_ssim"])
    return p.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            out: dict[str, Any] = dict(row)
            for key, value in row.items():
                if key in {"image", "label"}:
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
            out["source"] = path.name
            out["side_bpp"] = 1.0 / max(1.0, float(out["height"]) * float(out["width"]))
            rows.append(out)
        return rows


def metric(row: dict[str, Any], prefix: str, name: str) -> float:
    if name == "bpp":
        if prefix == "base":
            return float(row["base_bpp"])
        return float(row["branch_hybrid_empirical_bpp"])
    if name == "ms_ssim":
        key = f"{prefix}_ms_ssim"
    else:
        key = f"{prefix}_{name}"
    return float(row[key])


def selected_row(row: dict[str, Any], use_branch: bool, side_bits: float) -> dict[str, Any]:
    side = float(row["side_bpp"]) * side_bits
    return {
        "bpp": metric(row, "branch" if use_branch else "base", "bpp") + side,
        "psnr": metric(row, "branch" if use_branch else "base", "psnr"),
        "ms_ssim": metric(row, "branch" if use_branch else "base", "ms_ssim"),
        "lpips": metric(row, "branch" if use_branch else "base", "lpips"),
        "dists": metric(row, "branch" if use_branch else "base", "dists"),
        "used_branch": 1.0 if use_branch else 0.0,
    }


def summarize_selected(name: str, rows: list[dict[str, Any]], decisions: list[bool], side_bits: float) -> EvalResult:
    selected = [selected_row(r, d, side_bits) for r, d in zip(rows, decisions)]
    n = max(1, len(selected))
    base = {
        "bpp": sum(metric(r, "base", "bpp") for r in rows) / n,
        "psnr": sum(metric(r, "base", "psnr") for r in rows) / n,
        "ms_ssim": sum(metric(r, "base", "ms_ssim") for r in rows) / n,
        "lpips": sum(metric(r, "base", "lpips") for r in rows) / n,
        "dists": sum(metric(r, "base", "dists") for r in rows) / n,
    }
    branch = {
        "bpp": sum(metric(r, "branch", "bpp") for r in rows) / n,
        "psnr": sum(metric(r, "branch", "psnr") for r in rows) / n,
        "ms_ssim": sum(metric(r, "branch", "ms_ssim") for r in rows) / n,
        "lpips": sum(metric(r, "branch", "lpips") for r in rows) / n,
        "dists": sum(metric(r, "branch", "dists") for r in rows) / n,
    }
    row = {
        "selector": name,
        "images": len(rows),
        "branch_share": sum(decisions) / n,
        "base_bpp": base["bpp"],
        "selected_bpp": sum(x["bpp"] for x in selected) / n,
        "always_branch_bpp": branch["bpp"],
        "selected_dbpp_vs_base": sum(x["bpp"] for x in selected) / n - base["bpp"],
        "always_branch_dbpp_vs_base": branch["bpp"] - base["bpp"],
        "base_psnr": base["psnr"],
        "selected_psnr": sum(x["psnr"] for x in selected) / n,
        "always_branch_psnr": branch["psnr"],
        "base_ms_ssim": base["ms_ssim"],
        "selected_ms_ssim": sum(x["ms_ssim"] for x in selected) / n,
        "always_branch_ms_ssim": branch["ms_ssim"],
        "base_lpips": base["lpips"],
        "selected_lpips": sum(x["lpips"] for x in selected) / n,
        "always_branch_lpips": branch["lpips"],
        "base_dists": base["dists"],
        "selected_dists": sum(x["dists"] for x in selected) / n,
        "always_branch_dists": branch["dists"],
        "selected_ddists_vs_base": sum(x["dists"] for x in selected) / n - base["dists"],
        "always_branch_ddists_vs_base": branch["dists"] - base["dists"],
    }
    return EvalResult(name=name, rows=[row])


def target_value(rows: list[dict[str, Any]], decisions: list[bool], target: str, side_bits: float) -> float:
    vals = [selected_row(r, d, side_bits)[target] for r, d in zip(rows, decisions)]
    mean = sum(vals) / max(1, len(vals))
    return mean if target in LOWER_BETTER else -mean


def metric_oracle(rows: list[dict[str, Any]], target: str) -> list[bool]:
    decisions: list[bool] = []
    for row in rows:
        base = metric(row, "base", target)
        branch = metric(row, "branch", target)
        decisions.append(branch < base if target in LOWER_BETTER else branch > base)
    return decisions


def strict_oracle(rows: list[dict[str, Any]]) -> list[bool]:
    decisions: list[bool] = []
    for row in rows:
        decisions.append(
            metric(row, "branch", "dists") < metric(row, "base", "dists")
            and metric(row, "branch", "lpips") < metric(row, "base", "lpips")
        )
    return decisions


def threshold_decisions(rows: list[dict[str, Any]], feature: str, threshold: float, direction: str) -> list[bool]:
    out = []
    for row in rows:
        value = float(row[feature])
        out.append(value >= threshold if direction == ">=" else value <= threshold)
    return out


def candidate_thresholds(values: list[float]) -> list[float]:
    finite = sorted(set(v for v in values if math.isfinite(v)))
    if not finite:
        return [float("inf")]
    mids = [(a + b) * 0.5 for a, b in zip(finite, finite[1:])]
    eps = max(1e-12, (finite[-1] - finite[0]) * 1e-6)
    return [finite[0] - eps, *mids, finite[-1] + eps]


def best_threshold(rows: list[dict[str, Any]], target: str, side_bits: float) -> tuple[str, str, float, float, list[bool]]:
    best: tuple[str, str, float, float, list[bool]] | None = None
    for feature in FEATURES:
        if feature not in rows[0]:
            continue
        values = [float(r[feature]) for r in rows]
        for threshold in candidate_thresholds(values):
            for direction in (">=", "<="):
                decisions = threshold_decisions(rows, feature, threshold, direction)
                score = target_value(rows, decisions, target, side_bits)
                if best is None or score < best[3]:
                    best = (feature, direction, threshold, score, decisions)
    assert best is not None
    return best


def loocv_threshold(rows: list[dict[str, Any]], target: str, side_bits: float) -> EvalResult:
    decisions: list[bool] = []
    selected_rules: list[str] = []
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1 :]
        if not train:
            decisions.append(False)
            selected_rules.append("none")
            continue
        feature, direction, threshold, _, _ = best_threshold(train, target, side_bits)
        decisions.extend(threshold_decisions([rows[i]], feature, threshold, direction))
        selected_rules.append(f"{feature} {direction} {threshold:.6g}")
    result = summarize_selected(f"loocv_threshold_{target}", rows, decisions, side_bits)
    result.rows[0]["rules"] = "; ".join(selected_rules)
    return result


def analyze_group(name: str, rows: list[dict[str, Any]], target: str, side_bits: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    policies = {
        "baseline": [False] * len(rows),
        "always_branch": [True] * len(rows),
        "oracle_dists": metric_oracle(rows, "dists"),
        "oracle_lpips": metric_oracle(rows, "lpips"),
        "oracle_psnr": metric_oracle(rows, "psnr"),
        "oracle_dists_and_lpips": strict_oracle(rows),
    }
    for policy_name, decisions in policies.items():
        policy_side_bits = 0.0 if policy_name in {"baseline", "always_branch"} else side_bits
        row = summarize_selected(policy_name, rows, decisions, policy_side_bits).rows[0]
        row["group"] = name
        row["selector_side_bits"] = policy_side_bits
        results.append(row)

    feature, direction, threshold, _, decisions = best_threshold(rows, target, side_bits)
    row = summarize_selected(f"best_threshold_{target}", rows, decisions, side_bits).rows[0]
    row["group"] = name
    row["rule"] = f"{feature} {direction} {threshold:.6g}"
    row["selector_side_bits"] = side_bits
    results.append(row)

    row = loocv_threshold(rows, target, side_bits).rows[0]
    row["group"] = name
    row["selector_side_bits"] = side_bits
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
        "# E179 GLC Branch Selector Audit",
        "",
        "This audit reuses E177 per-image rows and asks whether a signaled branch/fallback selector could make the decoder-aware active branch useful before scaling training. It is a diagnostic upper-bound and threshold check, not a new trained codec row.",
        "",
        f"Target metric for threshold selection: `{args.target}`",
        f"Selector signaling cost: `{args.side_bits}` bit(s) per image.",
        "",
        "| group | selector | branch share | dbpp | dDISTS | base DISTS | selected DISTS | always DISTS | base LPIPS | selected LPIPS | base PSNR | selected PSNR | rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row['branch_share']:.3f} | "
            f"{row['selected_dbpp_vs_base']:+.6f} | {row['selected_ddists_vs_base']:+.6f} | "
            f"{row['base_dists']:.5f} | {row['selected_dists']:.5f} | {row['always_branch_dists']:.5f} | "
            f"{row['base_lpips']:.5f} | {row['selected_lpips']:.5f} | "
            f"{row['base_psnr']:.4f} | {row['selected_psnr']:.4f} | {row.get('rule', row.get('rules', ''))} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main() -> None:
    args = parse_args()
    all_results: list[dict[str, Any]] = []
    for path in args.csvs:
        rows = read_rows(path)
        labels = sorted(set(str(r["label"]) for r in rows))
        for label in labels:
            subset = [r for r in rows if str(r["label"]) == label and int(r["nonfinite"]) == 0]
            if not subset:
                continue
            q_values = sorted(set(int(r["q_index"]) for r in subset))
            q_tag = ",".join(str(q) for q in q_values)
            group_name = f"{path.stem}:{label}:q{q_tag}"
            all_results.extend(analyze_group(group_name, subset, args.target, args.side_bits))
    write_outputs(args.output_prefix, all_results, args)


if __name__ == "__main__":
    main()
