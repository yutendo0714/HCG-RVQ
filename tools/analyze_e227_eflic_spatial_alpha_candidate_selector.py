#!/usr/bin/env python3
"""Analyze EF-LIC spatial alpha-map candidate selection headroom.

E225/E226 produced codec-valid candidate rows: zero fallback, scalar all-on
geometry, and deterministic local alpha maps at a few strengths. This script
does not claim a final method. It asks whether a strength/local selector is
worth implementing by measuring:

- fixed candidate performance,
- per-image candidate oracle headroom,
- leave-dataset-out fixed-candidate transfer,
- a tiny decoder-safe feature stump as a diagnostic selector.

Lower score = delta_dists + 3 * delta_lpips is better.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_INPUTS = [
    ("clicpro41", Path("experiments/analysis/e228_eflic_clicpro41_spatial_alpha_map_alpha0005_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e226_eflic_clicpro41_spatial_alpha_map_alpha001_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e225_eflic_clicpro41_spatial_alpha_map_keymodes.csv")),
    ("clicpro41", Path("experiments/analysis/e231_eflic_clicpro41_spatial_alpha_soft_alpha0005.csv")),
    ("clicpro41", Path("experiments/analysis/e230_eflic_clicpro41_spatial_alpha_soft_alpha001.csv")),
    ("clicpro41", Path("experiments/analysis/e229_eflic_clicpro41_spatial_alpha_soft_alpha002.csv")),
    ("kodak24", Path("experiments/analysis/e228_eflic_kodak24_spatial_alpha_map_alpha0005_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e226_eflic_kodak24_spatial_alpha_map_alpha001_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e225_eflic_kodak24_spatial_alpha_map_keymodes.csv")),
    ("kodak24", Path("experiments/analysis/e231_eflic_kodak24_spatial_alpha_soft_alpha0005.csv")),
    ("kodak24", Path("experiments/analysis/e230_eflic_kodak24_spatial_alpha_soft_alpha001.csv")),
    ("kodak24", Path("experiments/analysis/e229_eflic_kodak24_spatial_alpha_soft_alpha002.csv")),
]


METRIC_KEYS = {
    "active_dists",
    "active_frac_target",
    "active_lpips",
    "active_psnr",
    "base_dists",
    "base_lpips",
    "base_psnr",
    "bpp",
    "delta_bpp",
    "delta_dists",
    "delta_lpips",
    "delta_psnr",
    "max_decode_diff",
    "mean_decode_diff",
    "nonfinite",
    "payload_equal",
    "payload_len_equal",
    "y_mismatch",
    "y_total",
    "z_mismatch",
    "z_total",
}


def to_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def score(row: dict[str, str]) -> float:
    return to_float(row.get("delta_dists"), 0.0) + 3.0 * to_float(row.get("delta_lpips"), 0.0)


def candidate_key(row: dict[str, str]) -> str:
    mode = row["mode"]
    if mode == "zero":
        return "zero"
    return f"{mode}@a{to_float(row.get('alpha'), 0.0):.3f}"


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def summarize_records(records: list[dict[str, object]], label: str, kind: str) -> dict[str, object]:
    n = len(records)
    return {
        "label": label,
        "kind": kind,
        "images": n,
        "delta_psnr": mean(to_float(r.get("delta_psnr"), 0.0) for r in records),
        "delta_dists": mean(to_float(r.get("delta_dists"), 0.0) for r in records),
        "delta_lpips": mean(to_float(r.get("delta_lpips"), 0.0) for r in records),
        "score": mean(to_float(r.get("score"), 0.0) for r in records),
        "score_win_frac": mean(1.0 if to_float(r.get("score"), 0.0) < 0.0 else 0.0 for r in records),
        "max_decode_diff": max((to_float(r.get("max_decode_diff"), 0.0) for r in records), default=0.0),
        "nonfinite_rows": int(sum(to_float(r.get("nonfinite"), 0.0) for r in records)),
        "choice_counts": dict(Counter(str(r.get("choice", "")) for r in records)),
    }


def choose_fixed(images: list[tuple[str, str]], rows: dict[tuple[str, str], dict[str, dict[str, str]]], key: str) -> list[dict[str, object]]:
    out = []
    for dataset, image in images:
        row = rows[(dataset, image)][key]
        out.append(record_from_row(dataset, image, key, row))
    return out


def record_from_row(dataset: str, image: str, choice: str, row: dict[str, str]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "image": image,
        "choice": choice,
        "delta_psnr": to_float(row.get("delta_psnr"), 0.0),
        "delta_dists": to_float(row.get("delta_dists"), 0.0),
        "delta_lpips": to_float(row.get("delta_lpips"), 0.0),
        "score": score(row),
        "max_decode_diff": to_float(row.get("max_decode_diff"), 0.0),
        "nonfinite": to_float(row.get("nonfinite"), 0.0),
    }


def oracle_records(images: list[tuple[str, str]], rows: dict[tuple[str, str], dict[str, dict[str, str]]]) -> list[dict[str, object]]:
    out = []
    for dataset, image in images:
        candidates = rows[(dataset, image)]
        choice, row = min(candidates.items(), key=lambda kv: score(kv[1]))
        out.append(record_from_row(dataset, image, choice, row))
    return out


def fixed_best_key(images: list[tuple[str, str]], rows: dict[tuple[str, str], dict[str, dict[str, str]]], keys: list[str]) -> str:
    best_key = keys[0]
    best_score = float("inf")
    for key in keys:
        records = choose_fixed(images, rows, key)
        value = mean(to_float(r["score"], 0.0) for r in records)
        if value < best_score:
            best_score = value
            best_key = key
    return best_key


def is_feature_column(name: str) -> bool:
    if name in METRIC_KEYS:
        return False
    if name in {"dataset", "image", "mode", "alpha", "direction_source", "force_ind"}:
        return False
    if name.endswith("_alpha") or "_alpha_" in name:
        return False
    if name.startswith("active_") or name.startswith("base_") or name.startswith("delta_"):
        return False
    return name.startswith("z_") or name.startswith("slice")


def build_features(
    images: list[tuple[str, str]],
    rows: dict[tuple[str, str], dict[str, dict[str, str]]],
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    zero_rows = [rows[item]["zero"] for item in images]
    all_cols = sorted({col for row in zero_rows for col in row.keys() if is_feature_column(col)})
    features: dict[tuple[str, str], dict[str, float]] = {}
    keep: list[str] = []
    for col in all_cols:
        vals = [to_float(row.get(col)) for row in zero_rows]
        finite = [v for v in vals if math.isfinite(v)]
        if len(finite) != len(vals):
            continue
        if max(finite) - min(finite) <= 1e-12:
            continue
        keep.append(col)
    for item in images:
        zrow = rows[item]["zero"]
        features[item] = {col: to_float(zrow.get(col), 0.0) for col in keep}
    return keep, features


def eval_stump(
    images: list[tuple[str, str]],
    rows: dict[tuple[str, str], dict[str, dict[str, str]]],
    features: dict[tuple[str, str], dict[str, float]],
    stump: dict[str, object],
) -> list[dict[str, object]]:
    col = str(stump["feature"])
    thr = float(stump["threshold"])
    left = str(stump["left_choice"])
    right = str(stump["right_choice"])
    out = []
    for dataset, image in images:
        choice = left if features[(dataset, image)][col] <= thr else right
        out.append(record_from_row(dataset, image, choice, rows[(dataset, image)][choice]))
    return out


def train_stump(
    images: list[tuple[str, str]],
    rows: dict[tuple[str, str], dict[str, dict[str, str]]],
    features: dict[tuple[str, str], dict[str, float]],
    feature_cols: list[str],
    candidate_keys: list[str],
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_value = float("inf")
    for col in feature_cols:
        vals = sorted({features[item][col] for item in images})
        if len(vals) < 2:
            continue
        thresholds = [(a + b) * 0.5 for a, b in zip(vals, vals[1:])]
        thresholds = [vals[0] - 1e-9] + thresholds + [vals[-1] + 1e-9]
        for thr in thresholds:
            for left in candidate_keys:
                for right in candidate_keys:
                    stump = {
                        "feature": col,
                        "threshold": thr,
                        "left_choice": left,
                        "right_choice": right,
                    }
                    records = eval_stump(images, rows, features, stump)
                    value = mean(to_float(r["score"], 0.0) for r in records)
                    if value < best_value:
                        best_value = value
                        best = {**stump, "train_score": value}
    return best


def write_summary_csv(path: Path, rows_out: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "kind",
        "images",
        "delta_psnr",
        "delta_dists",
        "delta_lpips",
        "score",
        "score_win_frac",
        "max_decode_diff",
        "nonfinite_rows",
        "choice_counts",
    ]
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)


def fmt(value: object, digits: int = 8) -> str:
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def write_md(path: Path, summary_rows: list[dict[str, object]], stumps: list[dict[str, object]]) -> None:
    with path.open("w") as fobj:
        fobj.write("# E227 EF-LIC Spatial Alpha Candidate Selector\n\n")
        fobj.write("Lower `score = delta_dists + 3 * delta_lpips` is better. All candidates come from codec-valid E225/E226 rows.\n\n")
        headers = ["label", "kind", "images", "delta_psnr", "delta_dists", "delta_lpips", "score", "score_win_frac", "choice_counts"]
        fobj.write("| " + " | ".join(headers) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in summary_rows:
            cells = []
            for key in headers:
                val = row[key]
                if key in {"delta_psnr", "delta_dists", "delta_lpips", "score", "score_win_frac"}:
                    cells.append(fmt(val, 8))
                else:
                    cells.append(str(val))
            fobj.write("| " + " | ".join(cells) + " |\n")

        fobj.write("\n## Learned/Stump Diagnostics\n\n")
        if stumps:
            s_headers = ["label", "feature", "threshold", "left_choice", "right_choice", "train_score", "eval_score"]
            fobj.write("| " + " | ".join(s_headers) + " |\n")
            fobj.write("|" + "|".join(["---"] * len(s_headers)) + "|\n")
            for stump in stumps:
                cells = []
                for key in s_headers:
                    val = stump.get(key, "")
                    if isinstance(val, float):
                        cells.append(fmt(val, 8))
                    else:
                        cells.append(str(val))
                fobj.write("| " + " | ".join(cells) + " |\n")

        fobj.write("\nInterpretation:\n\n")
        fobj.write("- The per-image oracle measures the headroom for a future no-sidebit strength/local controller.\n")
        fobj.write("- Adding weak and strong soft-alpha branches increases oracle headroom, which means smooth local strength is useful as a controller branch even when no single soft rule is universal.\n")
        fobj.write("- Leave-dataset-out fixed and stump rows are diagnostics for transfer risk; they are not paper-main methods.\n")
        fobj.write("- If oracle headroom is much larger than fixed/stump transfer, the next step should be a codec-trained local head rather than another hand-coded rule.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e227_eflic_spatial_alpha_candidate_selector"))
    parser.add_argument("--skip-stump", action="store_true", help="Skip exhaustive stump diagnostics for large candidate sets.")
    args = parser.parse_args()

    rows: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for dataset, path in DEFAULT_INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            for row in csv.DictReader(fobj):
                image = row["image"]
                key = candidate_key(row)
                if key in rows[(dataset, image)]:
                    continue
                row = dict(row)
                row["dataset"] = dataset
                rows[(dataset, image)][key] = row

    images = sorted(rows.keys())
    datasets = sorted({dataset for dataset, _ in images})
    candidate_keys = sorted({key for item in images for key in rows[item].keys()}, key=lambda x: (x == "zero", x))
    candidate_keys = ["zero"] + [key for key in candidate_keys if key != "zero"]
    for item in images:
        missing = [key for key in candidate_keys if key not in rows[item]]
        if missing:
            raise RuntimeError(f"{item} missing candidates {missing}")

    feature_cols, features = build_features(images, rows)
    summary_rows: list[dict[str, object]] = []

    for dataset in ["pooled"] + datasets:
        subset = images if dataset == "pooled" else [item for item in images if item[0] == dataset]
        best_key = fixed_best_key(subset, rows, candidate_keys)
        summary_rows.append(summarize_records(choose_fixed(subset, rows, "zero"), dataset, "fixed:zero"))
        summary_rows.append(summarize_records(choose_fixed(subset, rows, best_key), dataset, f"best_fixed:{best_key}"))
        summary_rows.append(summarize_records(oracle_records(subset, rows), dataset, "candidate_oracle"))

    for held in datasets:
        train = [item for item in images if item[0] != held]
        test = [item for item in images if item[0] == held]
        best_key = fixed_best_key(train, rows, candidate_keys)
        summary_rows.append(summarize_records(choose_fixed(test, rows, best_key), f"lodo:{held}", f"train_best_fixed:{best_key}"))

    stumps: list[dict[str, object]] = []
    if not args.skip_stump:
        same_stump = train_stump(images, rows, features, feature_cols, candidate_keys)
        if same_stump:
            same_eval = summarize_records(eval_stump(images, rows, features, same_stump), "pooled", "same_table_stump")
            summary_rows.append(same_eval)
            stumps.append({**same_stump, "label": "pooled_same_table", "eval_score": same_eval["score"]})

        for held in datasets:
            train = [item for item in images if item[0] != held]
            test = [item for item in images if item[0] == held]
            stump = train_stump(train, rows, features, feature_cols, candidate_keys)
            if stump:
                eval_summary = summarize_records(eval_stump(test, rows, features, stump), f"lodo:{held}", "stump")
                summary_rows.append(eval_summary)
                stumps.append({**stump, "label": f"lodo:{held}", "eval_score": eval_summary["score"]})

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.output_prefix.with_suffix(".summary.csv"), summary_rows)
    write_md(args.output_prefix.with_suffix(".md"), summary_rows, stumps)
    with args.output_prefix.with_suffix(".json").open("w") as fobj:
        json.dump(
            {
                "candidate_keys": candidate_keys,
                "feature_cols": feature_cols,
                "summary": summary_rows,
                "stumps": stumps,
            },
            fobj,
            indent=2,
            sort_keys=True,
        )

    choice_path = args.output_prefix.with_suffix(".oracle_choices.csv")
    with choice_path.open("w", newline="") as fobj:
        fieldnames = ["dataset", "image", "choice", "delta_psnr", "delta_dists", "delta_lpips", "score"]
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for record in oracle_records(images, rows):
            writer.writerow({key: record[key] for key in fieldnames})

    print(
        f"wrote {args.output_prefix.with_suffix('.summary.csv')}, "
        f"{args.output_prefix.with_suffix('.oracle_choices.csv')}, "
        f"{args.output_prefix.with_suffix('.json')}, and {args.output_prefix.with_suffix('.md')}"
    )


if __name__ == "__main__":
    main()
