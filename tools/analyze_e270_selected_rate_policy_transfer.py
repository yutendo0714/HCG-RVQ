#!/usr/bin/env python3
"""E270 selected-rate policy transfer audit over E269 margin rows.

E269 shows that the low-rate soft reconstruction usually has enough margin to
pay full branch bpp, but a small CLIC tail still needs fallback or rate savings.
This script tests simple one-feature hard-selection policies on the E269 row
CSV.  A selected row pays the full branch-bpp score from E269; an unselected row
falls back to the base and contributes zero score.

This is not a final codec.  It is a pre-implementation audit for deciding
whether a simple diagnostic threshold can be used by the next selected-index or
progressive-rate branch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments/analysis/e269_glc_lowrate_progressive_rate_margin.csv"
DEPLOYABLE_FEATURES = [
    "full_branch_dbpp",
    "diagnostic_dbpp",
    "gate_fraction_dbpp",
    "gate_mean",
    "active_mse_ratio",
    "active_rvq_mse",
    "active_scalar_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
]
ORACLE_FEATURES = ["score_no_bpp", "max_rate_fraction", "score_full_branch_bpp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e270_glc_lowrate_selected_rate_policy_transfer",
    )
    p.add_argument("--min-train-selected-frac", type=float, default=0.10)
    p.add_argument("--topk", type=int, default=8)
    return p.parse_args()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    raw = path if path.is_absolute() else ROOT / path
    with raw.open(newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    for row in rows:
        for key in DEPLOYABLE_FEATURES + ORACLE_FEATURES + [
            "score_full_branch_bpp",
            "score_diag",
            "full_branch_dbpp",
        ]:
            row[key] = finite(row.get(key))
    return rows


def subset(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "all":
        return rows
    if name == "trained":
        return [r for r in rows if r["phase"] == "trained"]
    if name == "trained_first":
        return [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).endswith("_first")]
    if name == "trained_held":
        return [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).endswith("_held")]
    if name == "trained_kodak":
        return [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).startswith("kodak")]
    if name == "trained_clic":
        return [r for r in rows if r["phase"] == "trained" and str(r["dataset"]).startswith("clic")]
    raise KeyError(name)


def policy_metrics(rows: list[dict[str, Any]], feature: str | None, op: str, threshold: float) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    selected = []
    scores = []
    for row in rows:
        if feature is None:
            keep = op == "all"
        elif op == "le":
            keep = float(row[feature]) <= threshold
        elif op == "ge":
            keep = float(row[feature]) >= threshold
        else:
            raise ValueError(op)
        selected.append(1.0 if keep else 0.0)
        scores.append(float(row["score_full_branch_bpp"]) if keep else 0.0)
    selected_scores = [float(r["score_full_branch_bpp"]) for r, keep in zip(rows, selected) if keep]
    return {
        "rows": len(rows),
        "selected": int(sum(selected)),
        "selected_frac": mean(selected),
        "score": mean(scores),
        "selected_score": mean(selected_scores) if selected_scores else 0.0,
        "selected_win_rate": mean([float(s < 0.0) for s in selected_scores]) if selected_scores else 0.0,
        "nonselected": len(rows) - int(sum(selected)),
    }


def candidate_thresholds(rows: list[dict[str, Any]], feature: str) -> list[float]:
    vals = sorted({float(r[feature]) for r in rows if math.isfinite(float(r[feature]))})
    if not vals:
        return []
    return vals


def fit_best(rows: list[dict[str, Any]], features: list[str], min_frac: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feature in features:
        for op in ("le", "ge"):
            for thr in candidate_thresholds(rows, feature):
                metrics = policy_metrics(rows, feature, op, thr)
                if metrics["selected_frac"] + 1e-12 < min_frac:
                    continue
                candidates.append({"feature": feature, "op": op, "threshold": thr, **metrics})
    candidates.append({"feature": "__all__", "op": "all", "threshold": float("nan"), **policy_metrics(rows, None, "all", 0.0)})
    candidates.append({"feature": "__none__", "op": "none", "threshold": float("nan"), **policy_metrics(rows, None, "none", 0.0)})
    return sorted(candidates, key=lambda r: (float(r["score"]), -float(r["selected_frac"])))


def eval_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    feature = None if str(policy["feature"]).startswith("__") else str(policy["feature"])
    metrics = policy_metrics(rows, feature, str(policy["op"]), finite(policy["threshold"], float("nan")))
    return {**policy, **{f"eval_{k}": v for k, v in metrics.items()}}


def policy_name(policy: dict[str, Any]) -> str:
    feature = str(policy["feature"])
    if feature.startswith("__"):
        return feature.replace("__", "")
    return f"{feature} {policy['op']} {float(policy['threshold']):.6g}"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    protocols = [
        ("resub_trained_deployable", "trained", "trained", DEPLOYABLE_FEATURES),
        ("first_to_held_deployable", "trained_first", "trained_held", DEPLOYABLE_FEATURES),
        ("kodak_to_clic_deployable", "trained_kodak", "trained_clic", DEPLOYABLE_FEATURES),
        ("clic_to_kodak_deployable", "trained_clic", "trained_kodak", DEPLOYABLE_FEATURES),
        ("resub_trained_oracle", "trained", "trained", ORACLE_FEATURES),
        ("first_to_held_oracle", "trained_first", "trained_held", ORACLE_FEATURES),
        ("kodak_to_clic_oracle", "trained_kodak", "trained_clic", ORACLE_FEATURES),
        ("clic_to_kodak_oracle", "trained_clic", "trained_kodak", ORACLE_FEATURES),
    ]
    results: list[dict[str, Any]] = []
    top_rows: dict[str, list[dict[str, Any]]] = {}
    for name, train_name, eval_name, features in protocols:
        train_rows = subset(rows, train_name)
        eval_rows = subset(rows, eval_name)
        fitted = fit_best(train_rows, features, args.min_train_selected_frac)
        top_rows[name] = fitted[: args.topk]
        best = fitted[0]
        results.append(
            {
                "protocol": name,
                "train_group": train_name,
                "eval_group": eval_name,
                "kind": "oracle" if features is ORACLE_FEATURES else "deployable",
                "policy": policy_name(best),
                "train_score": best["score"],
                "train_selected_frac": best["selected_frac"],
                "train_selected_win_rate": best["selected_win_rate"],
                **eval_policy(eval_rows, best),
            }
        )

    out_prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    with csv_path.open("w", newline="") as fp:
        fields = sorted({key for row in results for key in row})
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps({"results": results, "top_train_policies": top_rows}, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E270 Selected-Rate Policy Transfer Audit",
        "",
        "Purpose: test whether simple one-feature hard policies can drop the E269 low-rate CLIC tail before implementing selected-index/progressive coding.",
        "A selected row pays the E269 full-branch-bpp score; an unselected row falls back to the base and contributes zero score.",
        "",
        "## Best Policies",
        "",
        "| protocol | kind | train -> eval | policy | train score | train sel | eval score | eval sel | eval selected win |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['protocol']} | {row['kind']} | {row['train_group']} -> {row['eval_group']} | "
            f"{row['policy']} | {float(row['train_score']):+.6f} | {float(row['train_selected_frac']):.3f} | "
            f"{float(row['eval_score']):+.6f} | {float(row['eval_selected_frac']):.3f} | "
            f"{float(row['eval_selected_win_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Deployable policies use only branch diagnostics such as bpp, gate, entropy, and residual MSE. Oracle policies use quality-derived margin columns and are upper bounds for diagnosis only.",
            "",
            "A useful deployable policy should be negative on held-out/domain-transfer evaluation without selecting only a trivial single row. If deployable policies fail but oracle policies succeed, the next codec must learn a better reliability signal rather than rely on a hand threshold.",
            "",
            "## Artifacts",
            "",
            f"- `{csv_path.relative_to(ROOT)}`",
            f"- `{json_path.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
