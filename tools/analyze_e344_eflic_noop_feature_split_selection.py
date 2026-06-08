#!/usr/bin/env python3
"""Split-select EF-LIC HCG no-op feature thresholds.

E343 finds optimistic per-feature fallback thresholds on all Kodak24 images.
This script selects thresholds on the first N images and evaluates the selected
policy on the remaining images.  It is still a small controlled diagnostic, but
it prevents the next controller design from relying on full-set post-hoc tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_e343_eflic_none_oracle_feature_audit import (  # noqa: E402
    DEFAULT_CONTROLLER_CSVS,
    evaluate_policy,
    mean,
    read_controller,
    read_oracle,
    threshold_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.by_image.csv",
    )
    parser.add_argument("--controller-csv", type=Path, nargs="+", default=DEFAULT_CONTROLLER_CSVS)
    parser.add_argument("--mode", default="trained_hard")
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--tail-floor", type=float, default=-0.02)
    parser.add_argument("--tail-weight", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e344_eflic_noop_feature_split_selection_kodak16_8",
    )
    return parser.parse_args()


def finite(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        key
        for key, value in rows[0].items()
        if finite(value)
        and key not in {"controller_delta_psnr"}
        and not key.startswith("oracle_")
    )


def policy_score(policy: dict[str, Any], *, tail_weight: float) -> float:
    mean_delta = policy["mean_delta_psnr"]
    worst = policy["worst_delta_psnr"]
    tail_penalty = min(0.0, worst)
    return float(mean_delta + tail_weight * tail_penalty)


def eval_selected(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    return evaluate_policy(
        rows,
        str(policy["feature"]),
        float(policy["threshold"]),
        str(policy["direction"]),
    )


def select_policies(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    tail_floor: float,
    tail_weights: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for feature in feature_names(train_rows):
        values = [row.get(feature, float("nan")) for row in train_rows]
        for direction in ["<=", ">="]:
            for threshold in threshold_candidates(values):
                train_policy = evaluate_policy(train_rows, feature, threshold, direction)
                candidates.append(train_policy)
    for tw in tail_weights:
        best = max(candidates, key=lambda policy: policy_score(policy, tail_weight=tw))
        eval_policy = eval_selected(eval_rows, best)
        out.append(
            {
                "selection": f"tail_weight_{tw:g}",
                "tail_weight": tw,
                **{f"train_{k}": v for k, v in best.items()},
                **{f"eval_{k}": v for k, v in eval_policy.items()},
            }
        )
    safe_candidates = [policy for policy in candidates if policy["worst_delta_psnr"] >= tail_floor]
    if safe_candidates:
        best_safe = max(safe_candidates, key=lambda policy: policy["mean_delta_psnr"])
        eval_policy = eval_selected(eval_rows, best_safe)
        out.append(
            {
                "selection": f"train_worst_ge_{tail_floor}",
                "tail_weight": float("nan"),
                **{f"train_{k}": v for k, v in best_safe.items()},
                **{f"eval_{k}": v for k, v in eval_policy.items()},
            }
        )
    return out


def summarize_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["controller_delta_psnr"] for row in rows]
    return {
        "records": len(rows),
        "mean_delta_psnr": mean(deltas),
        "worst_delta_psnr": min(deltas),
        "negative_count": sum(1 for value in deltas if value < 0.0),
        "positive_count": sum(1 for value in deltas if value > 0.0),
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

    rows_out: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "experiment": "E344 EF-LIC no-op feature split selection",
        "purpose": "Select decoder-safe fallback feature thresholds on first split and evaluate on held-out Kodak images.",
        "args": {
            "oracle_csv": str(args.oracle_csv),
            "controller_csv": [str(path) for path in args.controller_csv],
            "mode": args.mode,
            "train_count": args.train_count,
            "tail_floor": args.tail_floor,
            "tail_weight": args.tail_weight,
        },
        "runs": {},
    }
    for csv_path in args.controller_csv:
        rows = read_controller(csv_path, mode=args.mode, oracle=oracle)
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["image"])
        run = rows[0]["run"]
        train_rows = rows[: args.train_count]
        eval_rows = rows[args.train_count :]
        selected = select_policies(
            train_rows,
            eval_rows,
            tail_floor=args.tail_floor,
            tail_weights=[float(v) for v in args.tail_weight],
        )
        for row in selected:
            rows_out.append({"run": run, **row})
        payload["runs"][run] = {
            "train": summarize_raw(train_rows),
            "eval": summarize_raw(eval_rows),
            "selected": selected,
        }

    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(csv_path, rows_out)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E344 EF-LIC No-op Feature Split Selection\n\n")
        fobj.write(
            "Thresholds are selected on the first Kodak split and evaluated on the remaining images. "
            "This is a controlled diagnostic for a future learned no-op head, not final RD evidence.\n\n"
        )
        fobj.write("| run | selection | feature | dir | threshold | train mean | train worst | eval mean | eval worst | eval neg | eval suppressed |\n")
        fobj.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows_out:
            fobj.write(
                f"| {row['run']} | {row['selection']} | {row['train_feature']} | {row['train_direction']} | "
                f"{float(row['train_threshold']):.6g} | {float(row['train_mean_delta_psnr']):+.6f} | "
                f"{float(row['train_worst_delta_psnr']):+.6f} | {float(row['eval_mean_delta_psnr']):+.6f} | "
                f"{float(row['eval_worst_delta_psnr']):+.6f} | {int(row['eval_negative_count'])} | "
                f"{int(row['eval_suppressed_count'])} |\n"
            )
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- If split-selected thresholds keep the eval tail safe, a simple reliability rule may be enough for the next codec-loop pilot.\n")
        fobj.write("- If they overfit, the next step should train a no-op/fallback head on separate teacher images before full training.\n")
    print(f"wrote {csv_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
