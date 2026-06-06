#!/usr/bin/env python3
"""Audit the EF-LIC spatial HCG branch library after E231.

E225-E231 established a codec-valid set of deterministic, decoder-reproducible
local geometry branches. This script turns those per-candidate CSVs into a
paper-planning audit:

- fixed candidate/family summaries,
- per-image oracle headroom,
- leave-one-family-out oracle ablations,
- greedy family-set construction,
- simple risk correlations for geometry/index perturbation.

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


def to_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def score(row: dict[str, str]) -> float:
    return to_float(row.get("delta_dists"), 0.0) + 3.0 * to_float(row.get("delta_lpips"), 0.0)


def candidate_key(row: dict[str, str]) -> str:
    mode = row["mode"]
    if mode == "zero":
        return "zero"
    return f"{mode}@a{to_float(row.get('alpha'), 0.0):.3f}"


def family_for_mode(mode: str) -> str:
    if mode == "zero":
        return "zero"
    if mode == "constant":
        return "constant"
    if mode == "prev_rms_top":
        return "sparse_prev"
    if mode == "support_rms_top":
        return "sparse_support"
    if mode in {"prev_rms_top_soft", "prev_over_scale_top_soft"}:
        return "soft_prev"
    if mode in {"support_rms_top_soft", "support_over_scale_top_soft"}:
        return "soft_support"
    return "other"


def family_for_key(key: str) -> str:
    mode = key.split("@", 1)[0]
    return family_for_mode(mode)


def load_rows() -> tuple[
    dict[tuple[str, str], dict[str, dict[str, str]]],
    dict[str, dict[str, str]],
    dict[str, list[tuple[str, str]]],
]:
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    candidate_meta: dict[str, dict[str, str]] = {}
    items_by_dataset: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for dataset, path in DEFAULT_INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as fobj:
            reader = csv.DictReader(fobj)
            for row in reader:
                image = row["image"]
                key = candidate_key(row)
                item = (dataset, image)
                items_by_dataset[dataset].add(item)
                if key in rows_by_item[item]:
                    # Zero and overlapping candidates appear in multiple runs.
                    continue
                rows_by_item[item][key] = row
                candidate_meta.setdefault(
                    key,
                    {
                        "candidate": key,
                        "mode": row["mode"],
                        "alpha": f"{to_float(row.get('alpha'), 0.0):.6f}",
                        "family": family_for_mode(row["mode"]),
                    },
                )
    ordered_items = {k: sorted(v) for k, v in items_by_dataset.items()}
    return dict(rows_by_item), candidate_meta, ordered_items


def all_items(items_by_dataset: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for dataset in sorted(items_by_dataset):
        out.extend(items_by_dataset[dataset])
    return out


def item_keys(items: list[tuple[str, str]], rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]]) -> set[str]:
    keys: set[str] | None = None
    for item in items:
        available = set(rows_by_item[item])
        keys = available if keys is None else keys & available
    return keys or set()


def record_from_row(dataset: str, image: str, choice: str, row: dict[str, str]) -> dict[str, object]:
    y_total = to_float(row.get("y_total"), 0.0)
    z_total = to_float(row.get("z_total"), 0.0)
    return {
        "dataset": dataset,
        "image": image,
        "choice": choice,
        "family": family_for_key(choice),
        "delta_psnr": to_float(row.get("delta_psnr"), 0.0),
        "delta_dists": to_float(row.get("delta_dists"), 0.0),
        "delta_lpips": to_float(row.get("delta_lpips"), 0.0),
        "delta_bpp": to_float(row.get("delta_bpp"), 0.0),
        "score": score(row),
        "y_mismatch_frac": safe_div(to_float(row.get("y_mismatch"), 0.0), y_total),
        "z_mismatch_frac": safe_div(to_float(row.get("z_mismatch"), 0.0), z_total),
        "alpha_active_frac": to_float(row.get("y_alpha_active_frac"), 0.0),
        "geometry_delta_rms": to_float(row.get("y_avg_geometry_delta_rms"), 0.0),
        "residual_mean_abs": to_float(row.get("y_avg_residual_mean_abs"), 0.0),
        "max_decode_diff": to_float(row.get("max_decode_diff"), 0.0),
        "nonfinite": to_float(row.get("nonfinite"), 0.0),
    }


def oracle_records(
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    allowed_keys: set[str],
) -> list[dict[str, object]]:
    out = []
    for dataset, image in items:
        candidates = [(key, row) for key, row in rows_by_item[(dataset, image)].items() if key in allowed_keys]
        if not candidates:
            raise RuntimeError(f"no candidates for {dataset}/{image}")
        choice, row = min(candidates, key=lambda kv: score(kv[1]))
        out.append(record_from_row(dataset, image, choice, row))
    return out


def summarize_records(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "label": label,
        "images": len(rows),
        "delta_psnr": mean(to_float(r.get("delta_psnr"), 0.0) for r in rows),
        "delta_dists": mean(to_float(r.get("delta_dists"), 0.0) for r in rows),
        "delta_lpips": mean(to_float(r.get("delta_lpips"), 0.0) for r in rows),
        "delta_bpp": mean(to_float(r.get("delta_bpp"), 0.0) for r in rows),
        "score": mean(to_float(r.get("score"), 0.0) for r in rows),
        "score_win_frac": mean(1.0 if to_float(r.get("score"), 0.0) < 0.0 else 0.0 for r in rows),
        "y_mismatch_frac": mean(to_float(r.get("y_mismatch_frac"), 0.0) for r in rows),
        "alpha_active_frac": mean(to_float(r.get("alpha_active_frac"), 0.0) for r in rows),
        "geometry_delta_rms": mean(to_float(r.get("geometry_delta_rms"), 0.0) for r in rows),
        "max_decode_diff": max((to_float(r.get("max_decode_diff"), 0.0) for r in rows), default=0.0),
        "nonfinite_rows": int(sum(to_float(r.get("nonfinite"), 0.0) for r in rows)),
        "choice_counts": dict(Counter(str(r.get("choice", "")) for r in rows)),
        "family_counts": dict(Counter(str(r.get("family", "")) for r in rows)),
    }


def fixed_candidate_summary(
    label: str,
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    candidate_meta: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows_out = []
    for key in sorted(item_keys(items, rows_by_item), key=lambda k: (family_for_key(k), k)):
        records = [record_from_row(ds, img, key, rows_by_item[(ds, img)][key]) for ds, img in items]
        summary = summarize_records(label, records)
        summary.update(candidate_meta[key])
        rows_out.append(summary)
    return rows_out


def best_fixed(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda r: to_float(r.get("score"), 0.0))


def leave_one_family_out(
    label: str,
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    candidate_keys: set[str],
) -> list[dict[str, object]]:
    all_oracle = summarize_records("oracle_all", oracle_records(items, rows_by_item, candidate_keys))
    families = sorted({family_for_key(k) for k in candidate_keys})
    out = []
    for family in families:
        if family == "zero":
            continue
        subset = {k for k in candidate_keys if family_for_key(k) != family}
        rec = summarize_records(f"without_{family}", oracle_records(items, rows_by_item, subset))
        out.append(
            {
                "dataset": label,
                "removed_family": family,
                "all_oracle_score": all_oracle["score"],
                "without_family_score": rec["score"],
                "oracle_loss_from_removal": to_float(rec["score"]) - to_float(all_oracle["score"]),
                "without_family_family_counts": json.dumps(rec["family_counts"], sort_keys=True),
            }
        )
    return out


def greedy_family_set(
    label: str,
    items: list[tuple[str, str]],
    rows_by_item: dict[tuple[str, str], dict[str, dict[str, str]]],
    candidate_keys: set[str],
) -> list[dict[str, object]]:
    families = sorted({family_for_key(k) for k in candidate_keys})
    selected = {"zero"}
    remaining = set(families) - selected
    out: list[dict[str, object]] = []
    step = 0
    last_added = ""
    while True:
        keys = {k for k in candidate_keys if family_for_key(k) in selected}
        summary = summarize_records("greedy", oracle_records(items, rows_by_item, keys))
        out.append(
            {
                "dataset": label,
                "step": step,
                "added_family": last_added,
                "selected_families": ",".join(sorted(selected)),
                "score": summary["score"],
                "delta_dists": summary["delta_dists"],
                "delta_lpips": summary["delta_lpips"],
                "family_counts": json.dumps(summary["family_counts"], sort_keys=True),
            }
        )
        if not remaining:
            break
        best_family = None
        best_score = float("inf")
        for family in sorted(remaining):
            test = selected | {family}
            test_keys = {k for k in candidate_keys if family_for_key(k) in test}
            test_summary = summarize_records("test", oracle_records(items, rows_by_item, test_keys))
            value = to_float(test_summary["score"])
            if value < best_score:
                best_score = value
                best_family = family
        if best_family is None:
            break
        selected.add(best_family)
        remaining.remove(best_family)
        last_added = best_family
        step += 1
    return out


def risk_correlations(label: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    active_rows = [r for r in rows if str(r.get("candidate")) != "zero"]
    score_vals = [to_float(r.get("score"), 0.0) for r in active_rows]
    return [
        {"dataset": label, "feature": "y_mismatch_frac", "pearson_to_score": pearson([to_float(r.get("y_mismatch_frac"), 0.0) for r in active_rows], score_vals)},
        {"dataset": label, "feature": "alpha_active_frac", "pearson_to_score": pearson([to_float(r.get("alpha_active_frac"), 0.0) for r in active_rows], score_vals)},
        {"dataset": label, "feature": "geometry_delta_rms", "pearson_to_score": pearson([to_float(r.get("geometry_delta_rms"), 0.0) for r in active_rows], score_vals)},
        {"dataset": label, "feature": "score_win_frac", "pearson_to_score": pearson([to_float(r.get("score_win_frac"), 0.0) for r in active_rows], score_vals)},
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 8) -> str:
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, object]], columns: list[str], digits: int = 8) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            cells.append(fmt(val, digits) if isinstance(val, float) else str(val))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e232_eflic_branch_library_audit"))
    args = parser.parse_args()

    rows_by_item, candidate_meta, items_by_dataset = load_rows()
    pooled_items = all_items(items_by_dataset)
    datasets = {**items_by_dataset, "pooled": pooled_items}

    fixed_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    leave_rows: list[dict[str, object]] = []
    greedy_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    summary_json: dict[str, object] = {"datasets": {}, "candidate_count": len(candidate_meta)}

    for label, items in datasets.items():
        keys = item_keys(items, rows_by_item)
        fixed = fixed_candidate_summary(label, items, rows_by_item, candidate_meta)
        fixed_rows.extend(fixed)
        best = best_fixed(fixed)
        oracle = oracle_records(items, rows_by_item, keys)
        oracle_summary = summarize_records("oracle_all", oracle)
        oracle_rows.extend(oracle)
        leave_rows.extend(leave_one_family_out(label, items, rows_by_item, keys))
        greedy_rows.extend(greedy_family_set(label, items, rows_by_item, keys))
        correlation_rows.extend(risk_correlations(label, fixed))
        top_rows.append(
            {
                "dataset": label,
                "images": len(items),
                "candidate_count": len(keys),
                "best_fixed_candidate": best["candidate"],
                "best_fixed_family": best["family"],
                "best_fixed_score": best["score"],
                "best_fixed_delta_dists": best["delta_dists"],
                "best_fixed_delta_lpips": best["delta_lpips"],
                "oracle_score": oracle_summary["score"],
                "oracle_delta_dists": oracle_summary["delta_dists"],
                "oracle_delta_lpips": oracle_summary["delta_lpips"],
                "oracle_gain_vs_best_fixed": to_float(oracle_summary["score"]) - to_float(best["score"]),
                "oracle_family_counts": json.dumps(oracle_summary["family_counts"], sort_keys=True),
            }
        )
        summary_json["datasets"][label] = {
            "images": len(items),
            "candidate_count": len(keys),
            "best_fixed": best,
            "oracle": oracle_summary,
        }

    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".fixed_summary.csv"), fixed_rows)
    write_csv(prefix.with_suffix(".oracle_choices.csv"), oracle_rows)
    write_csv(prefix.with_suffix(".leave_one_family_out.csv"), leave_rows)
    write_csv(prefix.with_suffix(".greedy_family_set.csv"), greedy_rows)
    write_csv(prefix.with_suffix(".risk_correlations.csv"), correlation_rows)
    write_csv(prefix.with_suffix(".summary.csv"), top_rows)
    prefix.with_suffix(".json").write_text(json.dumps(summary_json, indent=2, sort_keys=True))

    md_path = prefix.with_suffix(".md")
    with md_path.open("w") as fobj:
        fobj.write("# E232 EF-LIC Branch Library Audit\n\n")
        fobj.write("Lower `score = delta_dists + 3 * delta_lpips` is better. All candidates are codec-valid rows inherited from E225-E231.\n\n")
        fobj.write("## Top-Level Headroom\n\n")
        fobj.write(md_table(top_rows, [
            "dataset",
            "images",
            "candidate_count",
            "best_fixed_candidate",
            "best_fixed_score",
            "oracle_score",
            "oracle_gain_vs_best_fixed",
            "oracle_family_counts",
        ], digits=8))
        fobj.write("\n## Leave-One-Family-Out Oracle Loss\n\n")
        leave_sorted = sorted(leave_rows, key=lambda r: (str(r["dataset"]), -to_float(r["oracle_loss_from_removal"])))
        fobj.write(md_table(leave_sorted, [
            "dataset",
            "removed_family",
            "all_oracle_score",
            "without_family_score",
            "oracle_loss_from_removal",
        ], digits=8))
        fobj.write("\n## Greedy Family Set\n\n")
        fobj.write(md_table(greedy_rows, [
            "dataset",
            "step",
            "selected_families",
            "score",
            "delta_dists",
            "delta_lpips",
            "family_counts",
        ], digits=8))
        fobj.write("\n## Risk Correlations\n\n")
        fobj.write(md_table(correlation_rows, ["dataset", "feature", "pearson_to_score"], digits=6))
        fobj.write("\n## Top Fixed Candidates\n\n")
        top_fixed = []
        for label in datasets:
            subset = [r for r in fixed_rows if r["label"] == label]
            top_fixed.extend(sorted(subset, key=lambda r: to_float(r["score"]))[:8])
        fobj.write(md_table(top_fixed, [
            "label",
            "candidate",
            "family",
            "score",
            "delta_dists",
            "delta_lpips",
            "score_win_frac",
            "y_mismatch_frac",
            "alpha_active_frac",
            "geometry_delta_rms",
        ], digits=8))
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- This audit is still a design probe: it selects among codec-valid handcrafted branches, not a trained controller.\n")
        fobj.write("- A useful learned controller should preserve zero fallback, one conservative previous-context branch, one aggressive all-on branch, and the soft local branches that increase oracle headroom without changing bpp.\n")
        fobj.write("- The leave-one-family-out table is the safest guide for what to keep before full training: high removal loss means that family carries non-redundant per-image value.\n")

    print(f"wrote {prefix}.*")


if __name__ == "__main__":
    main()
