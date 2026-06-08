#!/usr/bin/env python3
"""Held-out safety-margin sweep for GLC entropy reliability controllers.

E377 shows strong empirical held-out wins but a tiny fixed-index positive tail.
This sweep applies a conservative margin to calibrated high-entropy thresholds to
find a paper-safer controller before matched GLC fine-tuning/full training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_e377_glc_qaware_heldout_calibration as e377  # noqa: E402


def apply_margin(policy: e377.Policy, margin: float) -> e377.Policy:
    if margin <= 0.0:
        return policy
    if policy.direction == ">=":
        thresholds = {q: value + margin for q, value in policy.thresholds.items()}
    elif policy.direction == "<=":
        thresholds = {q: value - margin for q, value in policy.thresholds.items()}
    else:
        raise ValueError(f"unsupported direction {policy.direction}")
    return replace(policy, thresholds=thresholds, family=f"{policy.family}-margin{margin:g}")


def run_margin(args: argparse.Namespace, margin: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = e377.read_rows(args.clic, "CLIC41", args.label) + e377.read_rows(args.kodak, "Kodak24", args.label)
    fold_map = e377.assign_folds(rows, args.folds, args.seed)
    fold_results: list[dict[str, object]] = []
    for fold in range(args.folds):
        train_rows = [row for row in rows if fold_map[(str(row["dataset"]), str(row["image"]))] != fold]
        test_rows = [row for row in rows if fold_map[(str(row["dataset"]), str(row["image"]))] == fold]
        for mode in args.modes:
            policy = e377.fit_policy(
                train_rows,
                family=f"pooled-calib-{mode}-index_entropy_mean->=",
                feature="index_entropy_mean",
                direction=">=",
                mode=mode,
                profile="score+fixed-tail",
                min_global_rows=args.min_global_rows,
                min_q_rows=args.min_q_rows,
                max_worst=args.max_calib_worst,
                max_fixed_worst=args.max_calib_fixed_worst,
            )
            if policy is None:
                continue
            policy = apply_margin(policy, margin)
            for dataset in ["pooled", "CLIC41", "Kodak24"]:
                row = e377.evaluate_fold(fold, policy, train_rows, test_rows, dataset)
                row["threshold_margin"] = margin
                fold_results.append(row)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in fold_results:
        key = (row["threshold_margin"], row["dataset"], row["profile"], row["mode"], row["feature"], row["direction"])
        groups.setdefault(key, []).append(row)
    summary = []
    for group_rows in groups.values():
        out = e377.aggregate_rows(group_rows)
        out["threshold_margin"] = group_rows[0]["threshold_margin"]
        summary.append(out)
    return fold_results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clic", type=Path, default=e377.DEFAULT_CLIC)
    parser.add_argument("--kodak", type=Path, default=e377.DEFAULT_KODAK)
    parser.add_argument("--label", default="trained_replacement_soft")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", default="e377-v1")
    parser.add_argument("--modes", nargs="*", default=["global", "q-aware"])
    parser.add_argument("--margins", nargs="*", type=float, default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04])
    parser.add_argument("--min-global-rows", type=int, default=10)
    parser.add_argument("--min-q-rows", type=int, default=2)
    parser.add_argument("--max-calib-worst", type=float, default=0.0)
    parser.add_argument("--max-calib-fixed-worst", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e378_glc_entropy_margin_heldout"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_fold_results: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for margin in args.margins:
        fold_results, summary = run_margin(args, margin)
        all_fold_results.extend(fold_results)
        all_summary.extend(summary)
    all_summary.sort(key=lambda row: (
        row["dataset"] != "pooled",
        int(row["heldout_selected_positive_rows"]) > 0,
        int(row["heldout_selected_fixed_positive_rows"]) > 0,
        float(row["heldout_score_all"]),
        -int(row["heldout_selected_rows"]),
    ))
    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "folds": args.folds,
        "seed": args.seed,
        "margins": args.margins,
        "score_definition": "delta_DISTS + 3 * delta_LPIPS + delta_bpp; PSNR ignored; MS-SSIM side reported",
        "summary": all_summary,
        "fold_results": all_fold_results,
    }
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    e377.write_csv(out_prefix.with_suffix(".summary.csv"), all_summary)
    e377.write_csv(out_prefix.with_suffix(".folds.csv"), all_fold_results)

    fields = [
        "threshold_margin",
        "dataset",
        "mode",
        "heldout_rows",
        "heldout_selected_rows",
        "heldout_score_all",
        "heldout_fixed_score_all",
        "heldout_selected_win_frac",
        "heldout_selected_fixed_win_frac",
        "heldout_selected_worst_score",
        "heldout_selected_worst_fixed_score",
        "heldout_selected_positive_rows",
        "heldout_selected_fixed_positive_rows",
        "heldout_delta_lpips_all",
        "heldout_delta_dists_all",
        "heldout_delta_ms_ssim_all",
        "heldout_delta_bpp_all",
    ]
    pooled = [row for row in all_summary if row["dataset"] == "pooled"]
    safe = [row for row in pooled if int(row["heldout_selected_positive_rows"]) == 0 and int(row["heldout_selected_fixed_positive_rows"]) == 0]
    best_safe = min(safe, key=lambda row: float(row["heldout_score_all"])) if safe else None
    with out_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC Entropy Reliability Held-Out Safety-Margin Sweep\n\n")
        handle.write("PSNR is ignored. Strict score+fixed-tail high-entropy policies are calibrated on image-disjoint folds, then their thresholds are made more conservative by the listed margin before held-out evaluation.\n\n")
        handle.write("## Pooled Summary\n\n")
        handle.write(e377.markdown_table(pooled, fields))
        handle.write("\n\n## All Summary\n\n")
        handle.write(e377.markdown_table(all_summary, fields))
        handle.write("\n\n## Interpretation\n\n")
        if best_safe is not None:
            handle.write(
                f"The best held-out fixed-tail-safe pooled policy is {best_safe['mode']} with margin {float(best_safe['threshold_margin']):.4f}. "
                f"It selects {best_safe['heldout_selected_rows']}/{best_safe['heldout_rows']} rows, score_all {float(best_safe['heldout_score_all']):+.6f}, fixed_score_all {float(best_safe['heldout_fixed_score_all']):+.6f}, worst {float(best_safe['heldout_selected_worst_score']):+.6f}, and fixed worst {float(best_safe['heldout_selected_worst_fixed_score']):+.6f}. "
            )
        else:
            handle.write("No pooled policy removed the held-out fixed-index positive tail under this margin grid. ")
        handle.write("Use this as the promotion criterion: a GLC long-run branch should start from the simplest safe entropy controller, while stronger q-aware variants remain ablations if they keep more gain but leave any fixed-tail positives.\n")


if __name__ == "__main__":
    main()
