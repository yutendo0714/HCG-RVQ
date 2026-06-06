#!/usr/bin/env python3
"""Build E239 training targets for the EF-LIC local HCG family head."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FAMILY_NAMES = (
    "zero",
    "constant",
    "guarded_constant",
    "guarded_support",
    "soft_blend",
    "sparse_union",
    "hybrid",
)
FAMILY_TO_INDEX = {name: idx for idx, name in enumerate(FAMILY_NAMES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e238_eflic_teacher_label_margins.labels.csv",
    )
    p.add_argument(
        "--family-costs",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e238_eflic_teacher_label_margins.family_costs.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan",
    )
    p.add_argument("--gain-threshold", type=float, default=5e-4)
    p.add_argument("--family-margin-threshold", type=float, default=5e-5)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=1.0)
    return p.parse_args()


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(rows: list[dict[str, str]], gain_t: float, margin_t: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        oracle_family = row["oracle_family"]
        gain = to_float(row["improvement_vs_zero"], 0.0)
        margin = to_float(row["family_margin"], 0.0)
        confident = int(gain >= gain_t and margin >= margin_t and oracle_family != "zero")
        target_family = oracle_family if confident else "zero"
        # Weight high-gain examples more, but keep fallback labels present.
        gain_weight = min(4.0, max(0.25, gain / max(gain_t, 1e-8)))
        margin_weight = min(2.0, max(0.25, margin / max(margin_t, 1e-8)))
        sample_weight = 0.5 if target_family == "zero" and oracle_family != "zero" else gain_weight * margin_weight
        out.append(
            {
                "dataset": row["dataset"],
                "image": row["image"],
                "oracle_family": oracle_family,
                "oracle_policy": row["oracle_policy"],
                "target_family": target_family,
                "target_index": FAMILY_TO_INDEX[target_family],
                "confident_nonzero": confident,
                "oracle_score": to_float(row["oracle_score"], 0.0),
                "target_score": to_float(row["oracle_score"], 0.0) if confident else 0.0,
                "improvement_vs_zero": gain,
                "family_margin": margin,
                "sample_weight": float(sample_weight),
                "alpha_mean": to_float(row.get("alpha_mean"), 0.0) if confident else 0.0,
                "alpha_active_frac": to_float(row.get("alpha_active_frac"), 0.0) if confident else 0.0,
                "geometry_delta_rms": to_float(row.get("geometry_delta_rms"), 0.0) if confident else 0.0,
            }
        )
    return out


def summarize_manifest(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    datasets = sorted({str(row["dataset"]) for row in rows}) + ["pooled"]
    for dataset in datasets:
        subset = rows if dataset == "pooled" else [row for row in rows if row["dataset"] == dataset]
        if not subset:
            continue
        target_score = mean([float(row["target_score"]) for row in subset])
        oracle_score = mean([float(row["oracle_score"]) for row in subset])
        out.append(
            {
                "dataset": dataset,
                "images": len(subset),
                "target_score": target_score,
                "oracle_score": oracle_score,
                "oracle_headroom_retained_frac": (-target_score / -oracle_score) if oracle_score < 0.0 else float("nan"),
                "confident_nonzero_frac": mean([float(row["confident_nonzero"]) for row in subset]),
                "target_family_counts": json.dumps(dict(Counter(str(row["target_family"]) for row in subset)), sort_keys=True),
                "oracle_family_counts": json.dumps(dict(Counter(str(row["oracle_family"]) for row in subset)), sort_keys=True),
                "mean_sample_weight": mean([float(row["sample_weight"]) for row in subset]),
            }
        )
    return out


def class_weights(manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(row["target_family"]) for row in manifest)
    total = sum(counts.values())
    out = []
    for family in FAMILY_NAMES:
        count = counts.get(family, 0)
        inv = total / max(1, len(FAMILY_NAMES) * count)
        out.append({"family": family, "index": FAMILY_TO_INDEX[family], "count": count, "class_weight": float(min(8.0, inv))})
    return out


def build_cost_matrix(rows: list[dict[str, str]], false_positive_weight: float) -> list[dict[str, object]]:
    costs = {(a, b): 0.0 for a in FAMILY_NAMES for b in FAMILY_NAMES}
    counts = {(a, b): 0 for a in FAMILY_NAMES for b in FAMILY_NAMES}
    for row in rows:
        if row.get("dataset") != "pooled":
            continue
        true_family = row["oracle_family"]
        cand_family = row["candidate_family"]
        if true_family not in FAMILY_TO_INDEX or cand_family not in FAMILY_TO_INDEX:
            continue
        regret = max(0.0, to_float(row.get("regret_vs_oracle"), 0.0))
        if true_family == "zero" and cand_family != "zero":
            regret *= false_positive_weight
        costs[(true_family, cand_family)] += regret
        counts[(true_family, cand_family)] += 1
    matrix_rows: list[dict[str, object]] = []
    for true_family in FAMILY_NAMES:
        for cand_family in FAMILY_NAMES:
            n = counts[(true_family, cand_family)]
            cost = costs[(true_family, cand_family)] / max(1, n)
            matrix_rows.append(
                {
                    "true_family": true_family,
                    "candidate_family": cand_family,
                    "true_index": FAMILY_TO_INDEX[true_family],
                    "candidate_index": FAMILY_TO_INDEX[cand_family],
                    "cost": float(cost),
                }
            )
    return matrix_rows


def write_markdown(
    path: Path,
    *,
    summary: list[dict[str, object]],
    weights: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if abs(value) < 10:
                return f"{value:+.6f}"
            return f"{value:.3f}"
        return str(value)

    with path.open("w") as fobj:
        fobj.write("# E239 EF-LIC Local HCG Head Training Plan\n\n")
        fobj.write(
            "E239 turns the E238 teacher-label margin audit into a concrete first "
            "training plan for a frozen EF-LIC local HCG family/strength head.\n\n"
        )
        fobj.write(f"- Gain threshold: `{args.gain_threshold}`\n")
        fobj.write(f"- Family-margin threshold: `{args.family_margin_threshold}`\n")
        fobj.write(f"- False-positive weight: `{args.false_positive_weight}`\n")
        fobj.write(f"- Missed-active weight: `{args.missed_active_weight}`\n\n")
        keys = [
            "dataset",
            "images",
            "target_score",
            "oracle_score",
            "oracle_headroom_retained_frac",
            "confident_nonzero_frac",
            "target_family_counts",
        ]
        fobj.write("## Manifest Summary\n\n")
        fobj.write("| " + " | ".join(keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for row in summary:
            fobj.write("| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |\n")
        fobj.write("\n## Class Weights\n\n")
        keys2 = ["family", "index", "count", "class_weight"]
        fobj.write("| " + " | ".join(keys2) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys2)) + "|\n")
        for row in weights:
            fobj.write("| " + " | ".join(fmt(row.get(key, "")) for key in keys2) + " |\n")
        fobj.write(
            "\nImplementation note: use `hcg_rvq.eflic_local_controller.LocalHCGFamilyHead` "
            "after EF-LIC `_mean_scale`, initialize with zero fallback bias, and "
            "train only the head first. The manifest is image-level supervision; "
            "the first pilot may broadcast labels spatially, then refine with "
            "slice/spatial teacher labels once the contract is stable.\n"
        )


def main() -> None:
    args = parse_args()
    labels = read_csv(args.labels)
    family_costs = read_csv(args.family_costs)
    manifest = build_manifest(labels, args.gain_threshold, args.family_margin_threshold)
    summary = summarize_manifest(manifest)
    weights = class_weights(manifest)
    cost_matrix = build_cost_matrix(family_costs, args.false_positive_weight)

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".manifest.csv"), manifest)
    write_csv(prefix.with_suffix(".summary.csv"), summary)
    write_csv(prefix.with_suffix(".class_weights.csv"), weights)
    write_csv(prefix.with_suffix(".cost_matrix.csv"), cost_matrix)
    payload = {
        "experiment": "E239 EF-LIC local HCG head training plan",
        "families": list(FAMILY_NAMES),
        "gain_threshold": args.gain_threshold,
        "family_margin_threshold": args.family_margin_threshold,
        "false_positive_weight": args.false_positive_weight,
        "missed_active_weight": args.missed_active_weight,
        "inputs": {"labels": str(args.labels), "family_costs": str(args.family_costs)},
        "summary": summary,
        "class_weights": weights,
        "recommended_first_pilot": {
            "module": "hcg_rvq.eflic_local_controller.LocalHCGFamilyHead",
            "freeze": "EF-LIC backbone and HCG/RVQ geometry path; train only local head",
            "loss": "asymmetric_family_loss with cost_matrix and false-positive penalty",
            "decode_contract": "same head inputs after _mean_scale in compress and decompress",
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(prefix.with_suffix(".md"), summary=summary, weights=weights, args=args)
    print(f"wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
