#!/usr/bin/env python3
"""Holdout4096 selector audit for low-rate HCG bias010 vs HCS."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e143_lowrate_bias010_holdout4096_selector"
DATA_ROOT = "/dpl/openimages/open-images-v6/train/data"
START_INDEX = 4096
MAX_IMAGES = 4096
SEEDS = ("1234", "2345", "3456")

RUNS = [
    {
        "seed": "1234",
        "method": "hcs",
        "config": "configs/pilot_hcs_rvq_frozen_lambda0018_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0018_seed1234/checkpoint_step_500.pth.tar",
        "step": 500,
    },
    {
        "seed": "1234",
        "method": "hcg_bias010",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed1234/checkpoint_step_250.pth.tar",
        "step": 250,
    },
    {
        "seed": "2345",
        "method": "hcs",
        "config": "configs/pilot_hcs_rvq_frozen_lambda0018_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0018_seed2345/checkpoint_step_250.pth.tar",
        "step": 250,
    },
    {
        "seed": "2345",
        "method": "hcg_bias010",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed2345/checkpoint_step_250.pth.tar",
        "step": 250,
    },
    {
        "seed": "3456",
        "method": "hcs",
        "config": "configs/pilot_hcs_rvq_frozen_lambda0018_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0018_seed3456/checkpoint_step_500.pth.tar",
        "step": 500,
    },
    {
        "seed": "3456",
        "method": "hcg_bias010",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed3456/checkpoint_step_500.pth.tar",
        "step": 500,
    },
]

FEATURES = [
    "hcs_rd",
    "hcs_bpp",
    "hcs_psnr",
    "hcs_ms_ssim",
    "hcg_rvq_s_q_mean",
    "hcg_rvq_householder_strength",
    "hcg_rvq_householder_delta_rms",
    "hcg_rvq_householder_delta_rms_local_mean",
    "hcg_rvq_householder_delta_rms_local_max",
    "hcg_rvq_latent_quant_mse",
    "hcg_rvq_dead_code_ratio",
    "hcg_rvq_perplexity",
    "hcg_rvq_index_empirical_bpp",
]


def mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def q95(values: list[float]) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_runs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows_all: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for run in RUNS:
        for key in ("config", "checkpoint"):
            path = ROOT / str(run[key])
            if not path.exists():
                raise FileNotFoundError(path)
        rows, summary = evaluate_mode(
            mode="exact",
            config_path=str(ROOT / str(run["config"])),
            checkpoint_path=str(ROOT / str(run["checkpoint"])),
            data_root=DATA_ROOT,
            device=device,
            max_images=MAX_IMAGES,
            start_index=START_INDEX,
            patch_size=None,
            reference={},
        )
        for row in rows:
            row.update(
                {
                    "seed": run["seed"],
                    "method": run["method"],
                    "selected_step": run["step"],
                    "config": run["config"],
                    "checkpoint": run["checkpoint"],
                }
            )
        summary.update(
            {
                "seed": run["seed"],
                "method": run["method"],
                "selected_step": run["step"],
                "config": run["config"],
                "checkpoint": run["checkpoint"],
            }
        )
        rows_all.extend(rows)
        summaries.append(summary)
    return rows_all, summaries


def align_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(row["seed"], row["method"], row["path"]): row for row in rows}
    pairs: list[dict[str, object]] = []
    for seed in SEEDS:
        paths = sorted({row["path"] for row in rows if row["seed"] == seed})
        for path in paths:
            hcs = by_key[(seed, "hcs", path)]
            hcg = by_key[(seed, "hcg_bias010", path)]
            item: dict[str, object] = {
                "seed": seed,
                "path": path,
                "hcs_rd": float(hcs["rd_score"]),
                "hcg_rd": float(hcg["rd_score"]),
                "hcs_bpp": float(hcs["bpp"]),
                "hcg_bpp": float(hcg["bpp"]),
                "hcs_psnr": float(hcs["psnr"]),
                "hcg_psnr": float(hcg["psnr"]),
                "hcs_ms_ssim": float(hcs["ms_ssim"]),
                "hcg_ms_ssim": float(hcg["ms_ssim"]),
                "hcg_minus_hcs": float(hcg["rd_score"]) - float(hcs["rd_score"]),
                "oracle_rd": min(float(hcs["rd_score"]), float(hcg["rd_score"])),
                "oracle_uses_hcg": float(hcg["rd_score"]) < float(hcs["rd_score"]),
                "hcs_nonfinite": int(hcs.get("has_nonfinite", 0)),
                "hcg_nonfinite": int(hcg.get("has_nonfinite", 0)),
            }
            for key, value in hcg.items():
                if key.startswith("rvq_") and isinstance(value, (float, int)):
                    item[f"hcg_{key}"] = float(value)
            pairs.append(item)
    return pairs


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    hcs = [float(row["hcs_rd"]) for row in rows]
    hcg = [float(row["hcg_rd"]) for row in rows]
    oracle = [float(row["oracle_rd"]) for row in rows]
    deltas = [float(row["hcg_minus_hcs"]) for row in rows]
    return {
        "num_images": len(rows),
        "hcs_rd": mean(hcs),
        "hcg_rd": mean(hcg),
        "hcg_minus_hcs": mean(deltas),
        "hcg_win_count": sum(delta < 0.0 for delta in deltas),
        "oracle_rd": mean(oracle),
        "oracle_minus_hcs": mean(oracle) - mean(hcs),
        "oracle_minus_hcg": mean(oracle) - mean(hcg),
        "oracle_hcg_count": sum(bool(row["oracle_uses_hcg"]) for row in rows),
        "q95_hcg_damage": q95([max(0.0, delta) for delta in deltas]),
        "nonfinite_rows": sum(int(row["hcs_nonfinite"]) + int(row["hcg_nonfinite"]) for row in rows),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    vals = sorted(set(value for value in values if math.isfinite(value)))
    if not vals:
        return []
    return [vals[0] - 1e-12, *[(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])], vals[-1] + 1e-12]


def mixed_summary(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> dict[str, object]:
    mixed = []
    selected = []
    for row in rows:
        use_hcg = float(row[feature]) <= threshold if direction == "low" else float(row[feature]) >= threshold
        selected.append(use_hcg)
        mixed.append(float(row["hcg_rd"]) if use_hcg else float(row["hcs_rd"]))
    hcs = mean([float(row["hcs_rd"]) for row in rows])
    hcg = mean([float(row["hcg_rd"]) for row in rows])
    rd = mean(mixed)
    return {
        "mixed_rd": rd,
        "mixed_minus_hcs": rd - hcs,
        "mixed_minus_hcg": rd - hcg,
        "selected_count": sum(selected),
        "selected_frac": mean([float(value) for value in selected]),
    }



def best_train_selector(rows: list[dict[str, object]], feature: str, direction: str) -> dict[str, object] | None:
    finite = []
    for row in rows:
        value = float(row[feature])
        if math.isfinite(value):
            finite.append((value, float(row["hcg_rd"]) - float(row["hcs_rd"])))
    if not finite:
        return None

    finite.sort(key=lambda item: item[0])
    groups: list[tuple[float, float, int]] = []
    for value, delta in finite:
        if groups and value == groups[-1][0]:
            old_value, old_delta, old_count = groups[-1]
            groups[-1] = (old_value, old_delta + delta, old_count + 1)
        else:
            groups.append((value, delta, 1))

    hcs_rd = mean([float(row["hcs_rd"]) for row in rows])
    hcg_rd = mean([float(row["hcg_rd"]) for row in rows])
    base_sum = sum(float(row["hcs_rd"]) for row in rows)
    total_delta = sum(delta for _, delta in finite)
    total_count = len(finite)
    num_rows = len(rows)

    candidates: list[tuple[float, float, int]] = []
    if direction == "low":
        candidates.append((groups[0][0] - 1e-12, 0.0, 0))
        prefix_delta = 0.0
        prefix_count = 0
        for index, (value, delta, count) in enumerate(groups):
            prefix_delta += delta
            prefix_count += count
            if index + 1 < len(groups):
                threshold = (value + groups[index + 1][0]) / 2.0
            else:
                threshold = value + 1e-12
            candidates.append((threshold, prefix_delta, prefix_count))
    else:
        candidates.append((groups[0][0] - 1e-12, total_delta, total_count))
        prefix_delta = 0.0
        prefix_count = 0
        for index, (value, delta, count) in enumerate(groups):
            prefix_delta += delta
            prefix_count += count
            if index + 1 < len(groups):
                threshold = (value + groups[index + 1][0]) / 2.0
                candidates.append((threshold, total_delta - prefix_delta, total_count - prefix_count))
        candidates.append((groups[-1][0] + 1e-12, 0.0, 0))

    best_threshold, best_delta_sum, best_count = min(candidates, key=lambda item: item[1] / num_rows)
    mixed_rd = (base_sum + best_delta_sum) / num_rows
    return {
        "threshold": best_threshold,
        "mixed_rd": mixed_rd,
        "mixed_minus_hcs": mixed_rd - hcs_rd,
        "mixed_minus_hcg": mixed_rd - hcg_rd,
        "selected_count": best_count,
        "selected_frac": best_count / num_rows,
    }


def selector_audit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for heldout_seed in SEEDS:
        train_seeds = tuple(seed for seed in SEEDS if seed != heldout_seed)
        train = [row for row in rows if row["seed"] in train_seeds]
        heldout = [row for row in rows if row["seed"] == heldout_seed]
        for feature in FEATURES:
            if any(feature not in row for row in train + heldout):
                continue
            for direction in ("low", "high"):
                train_summary = best_train_selector(train, feature, direction)
                if train_summary is None:
                    continue
                best = {
                    "heldout_seed": heldout_seed,
                    "train_seeds": "+".join(train_seeds),
                    "feature": feature,
                    "direction": direction,
                    "threshold": train_summary["threshold"],
                    "train_mixed_rd": train_summary["mixed_rd"],
                    "train_mixed_minus_hcs": train_summary["mixed_minus_hcs"],
                    "train_mixed_minus_hcg": train_summary["mixed_minus_hcg"],
                    "train_selected_count": train_summary["selected_count"],
                    "train_selected_frac": train_summary["selected_frac"],
                }
                heldout_summary = mixed_summary(heldout, feature, direction, float(train_summary["threshold"]))
                best.update(
                    {
                        "heldout_mixed_rd": heldout_summary["mixed_rd"],
                        "heldout_mixed_minus_hcs": heldout_summary["mixed_minus_hcs"],
                        "heldout_mixed_minus_hcg": heldout_summary["mixed_minus_hcg"],
                        "heldout_selected_count": heldout_summary["selected_count"],
                        "heldout_selected_frac": heldout_summary["selected_frac"],
                    }
                )
                output.append(best)
    return output


def selector_summary(selector_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in selector_rows:
        grouped.setdefault((str(row["feature"]), str(row["direction"])), []).append(row)
    output = []
    for (feature, direction), rows in grouped.items():
        if len(rows) != len(SEEDS):
            continue
        deltas = [float(row["heldout_mixed_minus_hcs"]) for row in rows]
        output.append(
            {
                "feature": feature,
                "direction": direction,
                "mean_heldout_mixed_minus_hcs": mean(deltas),
                "mean_heldout_mixed_rd": mean([float(row["heldout_mixed_rd"]) for row in rows]),
                "heldout_win_count": sum(delta < 0.0 for delta in deltas),
                "mean_heldout_selected_frac": mean([float(row["heldout_selected_frac"]) for row in rows]),
                "heldout_seed1234_delta": next(float(row["heldout_mixed_minus_hcs"]) for row in rows if row["heldout_seed"] == "1234"),
                "heldout_seed2345_delta": next(float(row["heldout_mixed_minus_hcs"]) for row in rows if row["heldout_seed"] == "2345"),
                "heldout_seed3456_delta": next(float(row["heldout_mixed_minus_hcs"]) for row in rows if row["heldout_seed"] == "3456"),
            }
        )
    return sorted(output, key=lambda row: float(row["mean_heldout_mixed_minus_hcs"]))


def quartiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["hcs_rd"]))
    qsize = len(ordered) // 4
    output = []
    for index in range(4):
        chunk = ordered[index * qsize : (index + 1) * qsize]
        deltas = [float(row["hcg_minus_hcs"]) for row in chunk]
        hcs = [float(row["hcs_rd"]) for row in chunk]
        output.append(
            {
                "quartile": f"Q{index + 1}",
                "num_images": len(chunk),
                "hcs_rd_min": min(hcs),
                "hcs_rd_max": max(hcs),
                "hcg_minus_hcs": mean(deltas),
                "hcg_win_count": sum(delta < 0.0 for delta in deltas),
                "q95_damage": q95([max(0.0, delta) for delta in deltas]),
            }
        )
    return output


def correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ys = [float(row["hcg_minus_hcs"]) for row in rows]
    output = []
    for feature in FEATURES:
        if any(feature not in row for row in rows):
            continue
        output.append({"feature": feature, "r_with_hcg_minus_hcs": pearson([float(row[feature]) for row in rows], ys)})
    return sorted(output, key=lambda row: abs(float(row["r_with_hcg_minus_hcs"])), reverse=True)


def write_markdown(summary: dict[str, object], per_seed: list[dict[str, object]], top_selectors: list[dict[str, object]]) -> None:
    best = top_selectors[0] if top_selectors else None
    lines = [
        "# E143 Low-Rate Bias010 Holdout4096 Selector Audit",
        "",
        f"Split: `{DATA_ROOT}`, start_index={START_INDEX}, max_images={MAX_IMAGES}.",
        "",
        "## Headline",
        "",
        f"- HCS mean RD: `{fmt(float(summary['hcs_rd']))}`",
        f"- HCG bias010 mean RD: `{fmt(float(summary['hcg_rd']))}` ({fmt(float(summary['hcg_minus_hcs']), True)} vs HCS)",
        f"- per-image oracle RD: `{fmt(float(summary['oracle_rd']))}` ({fmt(float(summary['oracle_minus_hcs']), True)} vs HCS)",
        f"- HCG wins: `{summary['hcg_win_count']}/{summary['num_images']}` images",
        f"- q95 positive HCG damage: `{fmt(float(summary['q95_hcg_damage']))}`",
        f"- nonfinite rows: `{summary['nonfinite_rows']}`",
        "",
        "## Per Seed",
        "",
        "| seed | HCS RD | HCG RD | HCG-HCS | wins | oracle-HCS | q95 damage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {hcs} | {hcg} | {delta} | {wins}/{n} | {oracle} | {q95v} |".format(
                seed=row["seed"],
                hcs=fmt(float(row["hcs_rd"])),
                hcg=fmt(float(row["hcg_rd"])),
                delta=fmt(float(row["hcg_minus_hcs"]), True),
                wins=row["hcg_win_count"],
                n=row["num_images"],
                oracle=fmt(float(row["oracle_minus_hcs"]), True),
                q95v=fmt(float(row["q95_hcg_damage"])),
            )
        )
    lines.extend(
        [
            "",
            "## Best Leave-One-Seed-Out Selector Families",
            "",
            "| feature | dir | mean heldout delta | wins | selected frac | seed1234 | seed2345 | seed3456 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_selectors[:10]:
        lines.append(
            "| `{feature}` | {direction} | {delta} | {wins}/3 | {selected} | {s1234} | {s2345} | {s3456} |".format(
                feature=row["feature"],
                direction=row["direction"],
                delta=fmt(float(row["mean_heldout_mixed_minus_hcs"]), True),
                wins=row["heldout_win_count"],
                selected=fmt(float(row["mean_heldout_selected_frac"])),
                s1234=fmt(float(row["heldout_seed1234_delta"]), True),
                s2345=fmt(float(row["heldout_seed2345_delta"]), True),
                s3456=fmt(float(row["heldout_seed3456_delta"]), True),
            )
        )
    if best is not None:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"The best holdout selector family is `{best['feature']}` / `{best['direction']}` with mean heldout delta {fmt(float(best['mean_heldout_mixed_minus_hcs']), True)} vs HCS. Treat this as diagnostic unless it is converted into one fixed controller and re-evaluated without validation-time switching.",
            ]
        )
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, run_summaries = evaluate_runs()
    pairs = align_pairs(rows)
    summary = summarize(pairs)
    per_seed = [{"seed": seed, **summarize([row for row in pairs if row["seed"] == seed])} for seed in SEEDS]
    write_csv(PREFIX.with_suffix(".all_rows.csv"), rows)
    write_csv(PREFIX.with_suffix(".pairs.csv"), pairs)
    write_csv(PREFIX.with_suffix(".per_seed.csv"), per_seed)
    selector_rows = selector_audit(pairs)
    selector_family_rows = selector_summary(selector_rows)
    quartile_rows = quartiles(pairs)
    correlation_rows = correlations(pairs)

    result = {
        "experiment": "E143 low-rate bias010 holdout4096 selector audit",
        "data_root": DATA_ROOT,
        "start_index": START_INDEX,
        "max_images": MAX_IMAGES,
        "runs": RUNS,
        "run_summaries": run_summaries,
        "summary": summary,
        "per_seed": per_seed,
        "selectors": selector_rows,
        "selector_summary": selector_family_rows,
        "quartiles": quartile_rows,
        "correlations": correlation_rows,
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(PREFIX.with_suffix(".all_rows.csv"), rows)
    write_csv(PREFIX.with_suffix(".pairs.csv"), pairs)
    write_csv(PREFIX.with_suffix(".per_seed.csv"), per_seed)
    write_csv(PREFIX.with_suffix(".selectors.csv"), selector_rows)
    write_csv(PREFIX.with_suffix(".selector_summary.csv"), selector_family_rows)
    write_csv(PREFIX.with_suffix(".quartiles.csv"), quartile_rows)
    write_csv(PREFIX.with_suffix(".correlations.csv"), correlation_rows)
    write_markdown(summary, per_seed, selector_family_rows)
    print(PREFIX.with_suffix(".md"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
