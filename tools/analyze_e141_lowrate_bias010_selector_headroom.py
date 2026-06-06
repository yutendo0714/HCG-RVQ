#!/usr/bin/env python3
"""Per-image headroom audit for low-rate HCG bias010 vs HCS."""

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
PREFIX = ANALYSIS / "e141_lowrate_bias010_selector_headroom"
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
    xbar = mean([x for x, _ in pairs])
    ybar = mean([y for _, y in pairs])
    vx = sum((x - xbar) ** 2 for x, _ in pairs)
    vy = sum((y - ybar) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - xbar) * (y - ybar) for x, y in pairs) / math.sqrt(vx * vy)


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


def verify_runs() -> None:
    for run in RUNS:
        for key in ("config", "checkpoint"):
            path = ROOT / str(run[key])
            if not path.exists():
                raise FileNotFoundError(path)


def evaluate_runs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for run in RUNS:
        rows, summary = evaluate_mode(
            mode="exact",
            config_path=str(ROOT / str(run["config"])),
            checkpoint_path=str(ROOT / str(run["checkpoint"])),
            data_root="/dpl/kodak",
            device=device,
            max_images=24,
            start_index=0,
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
        all_rows.extend(rows)
        summaries.append(summary)
    return all_rows, summaries


def aligned_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_key = {(row["seed"], row["method"], row["path"]): row for row in rows}
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
                "hcg_nonfinite": int(hcg.get("has_nonfinite", 0)),
                "hcs_nonfinite": int(hcs.get("has_nonfinite", 0)),
            }
            for key, value in hcg.items():
                if key.startswith("rvq_") and isinstance(value, (float, int)):
                    item[f"hcg_{key}"] = float(value)
            out.append(item)
    return out


def summarize_method(rows: list[dict[str, object]]) -> dict[str, object]:
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
        "nonfinite_rows": sum(int(row["hcg_nonfinite"]) + int(row["hcs_nonfinite"]) for row in rows),
    }


def split_rows(rows: list[dict[str, object]], train_seeds: tuple[str, ...]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [row for row in rows if row["seed"] in train_seeds]
    heldout = [row for row in rows if row["seed"] not in train_seeds]
    return train, heldout


def mixed_summary(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> dict[str, object]:
    mixed = []
    selected = []
    for row in rows:
        value = float(row[feature])
        use_hcg = value <= threshold if direction == "low" else value >= threshold
        selected.append(use_hcg)
        mixed.append(float(row["hcg_rd"]) if use_hcg else float(row["hcs_rd"]))
    hcs_rd = mean([float(row["hcs_rd"]) for row in rows])
    hcg_rd = mean([float(row["hcg_rd"]) for row in rows])
    mixed_rd = mean(mixed)
    return {
        "mixed_rd": mixed_rd,
        "mixed_minus_hcs": mixed_rd - hcs_rd,
        "mixed_minus_hcg": mixed_rd - hcg_rd,
        "selected_count": sum(selected),
        "selected_frac": mean([float(value) for value in selected]),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    vals = sorted(set(value for value in values if math.isfinite(value)))
    if not vals:
        return []
    mids = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-12, *mids, vals[-1] + 1e-12]


def selector_audit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    for heldout_seed in SEEDS:
        train_seeds = tuple(seed for seed in SEEDS if seed != heldout_seed)
        train, heldout = split_rows(rows, train_seeds)
        for feature in FEATURES:
            values = [float(row[feature]) for row in train if feature in row]
            if len(values) != len(train):
                continue
            for direction in ("low", "high"):
                best: dict[str, object] | None = None
                for threshold in candidate_thresholds(values):
                    train_summary = mixed_summary(train, feature, direction, threshold)
                    if best is None or float(train_summary["mixed_minus_hcs"]) < float(best["train_mixed_minus_hcs"]):
                        best = {
                            "heldout_seed": heldout_seed,
                            "train_seeds": "+".join(train_seeds),
                            "feature": feature,
                            "direction": direction,
                            "threshold": threshold,
                            "train_mixed_rd": train_summary["mixed_rd"],
                            "train_mixed_minus_hcs": train_summary["mixed_minus_hcs"],
                            "train_mixed_minus_hcg": train_summary["mixed_minus_hcg"],
                            "train_selected_count": train_summary["selected_count"],
                            "train_selected_frac": train_summary["selected_frac"],
                        }
                if best is None:
                    continue
                heldout_summary = mixed_summary(heldout, feature, direction, float(best["threshold"]))
                best.update(
                    {
                        "heldout_mixed_rd": heldout_summary["mixed_rd"],
                        "heldout_mixed_minus_hcs": heldout_summary["mixed_minus_hcs"],
                        "heldout_mixed_minus_hcg": heldout_summary["mixed_minus_hcg"],
                        "heldout_selected_count": heldout_summary["selected_count"],
                        "heldout_selected_frac": heldout_summary["selected_frac"],
                    }
                )
                audits.append(best)
    return audits


def quartiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["hcs_rd"]))
    out = []
    qsize = len(sorted_rows) // 4
    for index in range(4):
        chunk = sorted_rows[index * qsize : (index + 1) * qsize]
        hcs = [float(row["hcs_rd"]) for row in chunk]
        deltas = [float(row["hcg_minus_hcs"]) for row in chunk]
        out.append(
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
    return out


def correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ys = [float(row["hcg_minus_hcs"]) for row in rows]
    out = []
    for feature in FEATURES:
        xs = [float(row[feature]) for row in rows if feature in row]
        if len(xs) != len(rows):
            continue
        out.append({"feature": feature, "r_with_hcg_minus_hcs": pearson(xs, ys)})
    return sorted(out, key=lambda row: abs(float(row["r_with_hcg_minus_hcs"])), reverse=True)


def write_markdown(
    summary: dict[str, object],
    per_seed: list[dict[str, object]],
    quartile_rows: list[dict[str, object]],
    selector_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
) -> None:
    best_selectors = sorted(selector_rows, key=lambda row: float(row["heldout_mixed_minus_hcs"]))[:8]
    lines = [
        "# E141 Low-Rate Bias010 Selector Headroom",
        "",
        "This is a Kodak24 per-image audit for the E140 `lambda_rd=0.0018` active HCG `bias010` checkpoints. It tests whether the seed3456 failure is globally unavoidable or whether a per-image HCS/HCG reliability selector has measurable headroom.",
        "",
        "## Headline",
        "",
        f"- HCS mean RD: `{fmt(float(summary['hcs_rd']))}`",
        f"- HCG bias010 mean RD: `{fmt(float(summary['hcg_rd']))}` ({fmt(float(summary['hcg_minus_hcs']), True)} vs HCS)",
        f"- Per-image oracle switch RD: `{fmt(float(summary['oracle_rd']))}` ({fmt(float(summary['oracle_minus_hcs']), True)} vs HCS, {fmt(float(summary['oracle_minus_hcg']), True)} vs HCG)",
        f"- HCG wins: `{summary['hcg_win_count']}/{summary['num_images']}` images; oracle uses HCG on `{summary['oracle_hcg_count']}/{summary['num_images']}` images",
        f"- q95 positive HCG damage: `{fmt(float(summary['q95_hcg_damage']))}`",
        f"- nonfinite rows: `{summary['nonfinite_rows']}`",
        "",
        "## Per Seed",
        "",
        "| seed | HCS RD | HCG RD | HCG-HCS | HCG wins | oracle-HCS | q95 damage |",
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
            "## HCS-Difficulty Quartiles",
            "",
            "| quartile | HCS RD range | HCG-HCS | HCG wins | q95 damage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in quartile_rows:
        lines.append(
            "| {q} | {lo}-{hi} | {delta} | {wins}/{n} | {q95v} |".format(
                q=row["quartile"],
                lo=fmt(float(row["hcs_rd_min"])),
                hi=fmt(float(row["hcs_rd_max"])),
                delta=fmt(float(row["hcg_minus_hcs"]), True),
                wins=row["hcg_win_count"],
                n=row["num_images"],
                q95v=fmt(float(row["q95_damage"])),
            )
        )
    lines.extend(
        [
            "",
            "## Best Leave-One-Seed-Out Thresholds",
            "",
            "| heldout | train | feature | dir | threshold | train mix-HCS | heldout mix-HCS | selected heldout |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_selectors:
        lines.append(
            "| {heldout} | {train} | `{feature}` | {direction} | {thr} | {train_delta} | {held_delta} | {sel} |".format(
                heldout=row["heldout_seed"],
                train=row["train_seeds"],
                feature=row["feature"],
                direction=row["direction"],
                thr=fmt(float(row["threshold"])),
                train_delta=fmt(float(row["train_mixed_minus_hcs"]), True),
                held_delta=fmt(float(row["heldout_mixed_minus_hcs"]), True),
                sel=f"{row['heldout_selected_count']}/24",
            )
        )
    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| feature | r with HCG-HCS |",
            "|---|---:|",
        ]
    )
    for row in correlation_rows[:10]:
        lines.append(f"| `{row['feature']}` | {fmt(float(row['r_with_hcg_minus_hcs']), True)} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is a diagnostic selector/headroom audit, not a deployable codec yet. If the oracle gap is large but leave-one-seed thresholds are unstable, the next implementation target should be a reliability/usage-controlled low-rate checkpoint rather than a stronger raw Householder geometry.",
        ]
    )
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    verify_runs()
    rows, run_summaries = evaluate_runs()
    pairs = aligned_pairs(rows)
    summary = summarize_method(pairs)
    per_seed = [{"seed": seed, **summarize_method([row for row in pairs if row["seed"] == seed])} for seed in SEEDS]
    quartile_rows = quartiles(pairs)
    selector_rows = selector_audit(pairs)
    correlation_rows = correlations(pairs)

    result = {
        "experiment": "E141 low-rate bias010 selector headroom",
        "data_root": "/dpl/kodak",
        "runs": RUNS,
        "run_summaries": run_summaries,
        "summary": summary,
        "per_seed": per_seed,
        "quartiles": quartile_rows,
        "selectors": selector_rows,
        "correlations": correlation_rows,
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(PREFIX.with_suffix(".all_rows.csv"), rows)
    write_csv(PREFIX.with_suffix(".pairs.csv"), pairs)
    write_csv(PREFIX.with_suffix(".per_seed.csv"), per_seed)
    write_csv(PREFIX.with_suffix(".quartiles.csv"), quartile_rows)
    write_csv(PREFIX.with_suffix(".selectors.csv"), selector_rows)
    write_csv(PREFIX.with_suffix(".correlations.csv"), correlation_rows)
    write_markdown(summary, per_seed, quartile_rows, selector_rows, correlation_rows)

    print(PREFIX.with_suffix(".md"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
