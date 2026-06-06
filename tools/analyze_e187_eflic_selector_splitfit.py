#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
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
        default=Path("experiments/analysis/e187_eflic_force0_global_selector_splitfit_dists"),
    )
    p.add_argument("--target", choices=["dists", "lpips", "psnr"], default="dists")
    p.add_argument("--force", type=int, default=0)
    return p.parse_args()


def image_number(row: dict[str, Any]) -> int:
    match = re.search(r"(\d+)", str(row["image"]))
    if not match:
        raise ValueError(f"cannot parse image number from {row['image']!r}")
    return int(match.group(1))


def split_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=image_number)
    if split == "first12_eval_last12":
        return ordered[:12], ordered[12:]
    if split == "last12_eval_first12":
        return ordered[12:], ordered[:12]
    if split == "odd_eval_even":
        train = [r for r in ordered if image_number(r) % 2 == 1]
        eval_rows = [r for r in ordered if image_number(r) % 2 == 0]
        return train, eval_rows
    if split == "even_eval_odd":
        train = [r for r in ordered if image_number(r) % 2 == 0]
        eval_rows = [r for r in ordered if image_number(r) % 2 == 1]
        return train, eval_rows
    raise ValueError(split)


def add_rule_stats(row: dict[str, Any], feature: str, direction: str, threshold: float, train_rows: list[dict[str, Any]]) -> None:
    row["fit_feature"] = feature
    row["fit_direction"] = direction
    row["fit_threshold"] = threshold
    row["fit_train_images"] = len(train_rows)
    row["fit_rule"] = f"{feature} {direction} {threshold:.9g}" if feature else "baseline"


def summarize_reference(
    split: str,
    phase: str,
    rows: list[dict[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    out = [
        e184.summarize_policy(split, f"{phase}_baseline", rows, [False] * len(rows), 0.0, target),
        e184.summarize_policy(split, f"{phase}_always_active", rows, [True] * len(rows), 0.0, target),
        e184.summarize_policy(split, f"{phase}_oracle_{target}", rows, e184.oracle(rows, target), 0.0, target),
        e184.summarize_policy(split, f"{phase}_oracle_dists_and_lpips", rows, e184.strict_oracle(rows), 0.0, target),
    ]
    for row in out:
        row["phase"] = phase
        row["feature_set"] = row.get("feature_set", "")
    return out


def analyze_split(
    split: str,
    rows: list[dict[str, Any]],
    features: list[str],
    target: str,
) -> list[dict[str, Any]]:
    train_rows, eval_rows = split_rows(rows, split)
    feature, direction, threshold, train_decisions, _ = e184.best_threshold(train_rows, features, target, 0.0)
    eval_decisions = e184.threshold_decisions(eval_rows, feature, threshold, direction) if feature else [False] * len(eval_rows)

    results = []
    results.extend(summarize_reference(split, "train", train_rows, target))
    train_summary = e184.summarize_policy(
        split,
        "train_fit_global_predecision_threshold",
        train_rows,
        train_decisions,
        0.0,
        target,
        "global_predecision_context",
    )
    train_summary["phase"] = "train"
    add_rule_stats(train_summary, feature, direction, threshold, train_rows)
    results.append(train_summary)

    results.extend(summarize_reference(split, "eval", eval_rows, target))
    eval_summary = e184.summarize_policy(
        split,
        "eval_apply_global_predecision_threshold",
        eval_rows,
        eval_decisions,
        0.0,
        target,
        "global_predecision_context",
    )
    eval_summary["phase"] = "eval"
    add_rule_stats(eval_summary, feature, direction, threshold, train_rows)
    results.append(eval_summary)
    return results


def write_outputs(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E187 EF-LIC Global Predecision Selector Split-Fit Audit",
        "",
        "This audit fits one scalar threshold on one Kodak split and evaluates it on the held-out split. It is still a small-sample Kodak audit, not a final paper row, but it checks whether the E186 no-side-bit selector is only a same-table artifact.",
        "",
        f"Target metric for fitting: `{args.target}`",
        f"Force index: `{args.force}`",
        "",
        "| split | phase | selector | branch share | dDISTS | dLPIPS | dPSNR | DISTS wins | rule |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["selector"].endswith("oracle_dists_and_lpips"):
            continue
        lines.append(
            f"| {row['group']} | {row.get('phase', '')} | {row['selector']} | "
            f"{row['branch_share']:.3f} | {row['selected_delta_dists']:+.6f} | "
            f"{row['selected_delta_lpips']:+.6f} | {row['selected_delta_psnr']:+.6f} | "
            f"{row['selected_win_dists']}/{row['images']} | {row.get('fit_rule', row.get('rule', ''))} |"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- Uses only `global_predecision_context`: z context plus slice0 mean/scale available before a whole-image active/fallback decision.",
            "- Fits only on the train half of Kodak24 and reports the held-out half separately.",
            "- This does not replace external validation or full EF-LIC training; it is a provenance check before spending larger GPU budget.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    all_rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in all_rows if int(r["force_ind"]) == args.force]
    if len(rows) != 24:
        raise SystemExit(f"expected 24 finite rows for force{args.force}, got {len(rows)}")
    manifest = e184.read_manifest(args.manifest_csv)
    feature_sets = e185.feature_sets(manifest)
    features = feature_sets["global_predecision_context"][0]

    results: list[dict[str, Any]] = []
    for split in ("first12_eval_last12", "last12_eval_first12", "odd_eval_even", "even_eval_odd"):
        results.extend(analyze_split(split, rows, features, args.target))
    write_outputs(args.output_prefix, results, args)


if __name__ == "__main__":
    main()
