#!/usr/bin/env python3
"""Analyze per-image selector-cap headroom for max500 dz018 transfer rows."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ANALYSIS = Path("experiments/analysis")
OUT_PREFIX = ANALYSIS / "e114_max500_per_image_cap_headroom"
SEEDS = (1234, 2345, 3456)
CAPS = (0.25, 0.35, 0.45, 0.50)
LOWER_CAPS = (0.25, 0.35, 0.45)
FEATURES = (
    "bpp",
    "bpp_y",
    "mse",
    "rvq_dead_code_ratio",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_max",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_delta_rms_local_std",
    "rvq_householder_residual_selector_multiplier",
    "rvq_householder_residual_selector_multiplier_min",
    "rvq_householder_residual_selector_multiplier_std",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_prob_max",
    "rvq_householder_residual_selector_prob_std",
    "rvq_householder_strength",
    "rvq_householder_strength_std",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_s_q_mean",
    "rvq_s_q_std",
    "rvq_stage_entropy",
    "rvq_y_norm_abs_mean",
)

POLICY_FEATURES = (
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_max",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_delta_rms_local_std",
    "rvq_householder_residual_selector_multiplier",
    "rvq_householder_residual_selector_multiplier_min",
    "rvq_householder_residual_selector_multiplier_std",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_prob_max",
    "rvq_householder_residual_selector_prob_std",
    "rvq_householder_strength",
    "rvq_latent_quant_mse",
    "rvq_s_q_mean",
)


def cap_tag(cap: float) -> str:
    return f"cap{int(round(cap * 100)):03d}"


def csv_path(seed: int, cap: float) -> Path:
    if cap == 0.50:
        return (
            ANALYSIS
            / f"e111_deadzone018_from_beta005_max500_seed{seed}_step500_fullimage_start8192_current.csv"
        )
    prefix = "e112" if seed == 3456 else "e113"
    return (
        ANALYSIS
        / f"{prefix}_deadzone018_{cap_tag(cap)}_max500_seed{seed}_step500_fullimage_transfer_current.csv"
    )


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def read_rows() -> list[dict[str, object]]:
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for seed in SEEDS:
        for cap in CAPS:
            path = csv_path(seed, cap)
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    key = (seed, row["path"])
                    item = by_key.setdefault(
                        key,
                        {
                            "seed": seed,
                            "path": row["path"],
                            "reference_rd": to_float(row["reference_rd_score"]),
                            "rd_by_cap": {},
                            "features": {},
                            "nonfinite": 0,
                        },
                    )
                    item["rd_by_cap"][cap] = to_float(row["rd_score"])  # type: ignore[index]
                    item["nonfinite"] = int(item["nonfinite"]) + int(float(row.get("has_nonfinite", 0)))  # type: ignore[arg-type]
                    if cap == 0.50:
                        item["features"] = {name: to_float(row.get(name)) for name in FEATURES}

    items: list[dict[str, object]] = []
    for item in by_key.values():
        rd_by_cap = item["rd_by_cap"]  # type: ignore[assignment]
        missing = [cap for cap in CAPS if cap not in rd_by_cap]
        if missing:
            raise ValueError(f"missing caps {missing} for {item['seed']} {item['path']}")
        items.append(item)
    return items


def summarize(items: list[dict[str, object]], chooser) -> dict[str, object]:
    rows = []
    by_seed: dict[int, list[float]] = defaultdict(list)
    best_caps: Counter[float] = Counter()
    for item in items:
        cap = float(chooser(item))
        rd = float(item["rd_by_cap"][cap])  # type: ignore[index]
        delta = rd - float(item["reference_rd"])
        rows.append(delta)
        by_seed[int(item["seed"])].append(delta)
        best_caps[cap] += 1
    return {
        "mean_delta": mean(rows),
        "win_rate_vs_reference": sum(v < 0.0 for v in rows) / len(rows),
        "num_images": len(rows),
        "per_seed_delta": {str(seed): mean(vals) for seed, vals in sorted(by_seed.items())},
        "cap_counts": {f"{cap:.2f}": count for cap, count in sorted(best_caps.items())},
    }


def baseline_chooser(_item: dict[str, object]) -> float:
    return 0.50


def oracle_chooser(item: dict[str, object]) -> float:
    rd_by_cap = item["rd_by_cap"]  # type: ignore[assignment]
    return min(CAPS, key=lambda cap: float(rd_by_cap[cap]))


def quantiles(values: list[float]) -> list[float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return []
    out = []
    for q in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        out.append(clean[min(len(clean) - 1, int(round(q * (len(clean) - 1))))])
    return sorted(set(out))


def policy_chooser(feature: str, direction: str, threshold: float, alt_cap: float):
    def choose(item: dict[str, object]) -> float:
        value = float(item["features"][feature])  # type: ignore[index]
        active = value <= threshold if direction == "<=" else value >= threshold
        return alt_cap if active else 0.50

    return choose


def policy_stats(items: list[dict[str, object]], feature: str, direction: str, threshold: float, alt_cap: float) -> dict[str, object]:
    active = 0
    deltas = []
    by_seed: dict[int, list[float]] = defaultdict(list)
    for item in items:
        value = float(item["features"][feature])  # type: ignore[index]
        is_active = value <= threshold if direction == "<=" else value >= threshold
        cap = alt_cap if is_active else 0.50
        if is_active:
            active += 1
        rd = float(item["rd_by_cap"][cap])  # type: ignore[index]
        delta = rd - float(item["reference_rd"])
        deltas.append(delta)
        by_seed[int(item["seed"])].append(delta)
    summary = {
        "mean_delta": mean(deltas),
        "win_rate_vs_reference": sum(v < 0.0 for v in deltas) / len(deltas),
        "num_images": len(deltas),
        "per_seed_delta": {str(seed): mean(vals) for seed, vals in sorted(by_seed.items())},
    }
    summary.update(
        {
            "feature": feature,
            "direction": direction,
            "threshold": threshold,
            "alt_cap": alt_cap,
            "action_rate": active / len(items),
        }
    )
    return summary


def find_policies(items: list[dict[str, object]]) -> list[dict[str, object]]:
    policies = []
    for feature in POLICY_FEATURES:
        values = [float(item["features"][feature]) for item in items]  # type: ignore[index]
        for threshold in quantiles(values):
            for direction in ("<=", ">="):
                for alt_cap in LOWER_CAPS:
                    policies.append(policy_stats(items, feature, direction, threshold, alt_cap))
    return sorted(policies, key=lambda row: float(row["mean_delta"]))


def leave_one_seed_cv(items: list[dict[str, object]]) -> dict[str, object]:
    held_rows = []
    held_deltas = []
    baseline_by_seed = summarize(items, baseline_chooser)["per_seed_delta"]  # type: ignore[index]
    for held_seed in SEEDS:
        train = [item for item in items if int(item["seed"]) != held_seed]
        test = [item for item in items if int(item["seed"]) == held_seed]
        best = find_policies(train)[0]
        test_stats = policy_stats(
            test,
            str(best["feature"]),
            str(best["direction"]),
            float(best["threshold"]),
            float(best["alt_cap"]),
        )
        delta = float(test_stats["mean_delta"])
        held_deltas.append(delta)
        held_rows.append(
            {
                "held_seed": held_seed,
                "feature": best["feature"],
                "direction": best["direction"],
                "threshold": best["threshold"],
                "alt_cap": best["alt_cap"],
                "train_delta": best["mean_delta"],
                "test_delta": delta,
                "test_gain_vs_cap050": delta - float(baseline_by_seed[str(held_seed)]),
                "test_action_rate": test_stats["action_rate"],
            }
        )
    baseline = summarize(items, baseline_chooser)
    return {
        "mean_delta": mean(held_deltas),
        "gain_vs_cap050": mean(held_deltas) - float(baseline["mean_delta"]),
        "held_rows": held_rows,
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return num / (den_x * den_y)


def feature_correlations(items: list[dict[str, object]]) -> list[dict[str, object]]:
    gains = []
    for item in items:
        rd_by_cap = item["rd_by_cap"]  # type: ignore[assignment]
        best_lower = min(float(rd_by_cap[cap]) for cap in LOWER_CAPS)
        gains.append(best_lower - float(rd_by_cap[0.50]))
    rows = []
    for feature in FEATURES:
        xs = [float(item["features"][feature]) for item in items]  # type: ignore[index]
        rows.append({"feature": feature, "pearson_with_lower_cap_gain": pearson(xs, gains)})
    return sorted(rows, key=lambda row: abs(float(row["pearson_with_lower_cap_gain"])), reverse=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    items = read_rows()
    baseline = summarize(items, baseline_chooser)
    oracle = summarize(items, oracle_chooser)
    policies = find_policies(items)
    cv = leave_one_seed_cv(items)
    correlations = feature_correlations(items)

    payload = {
        "baseline_cap050": baseline,
        "per_image_oracle": oracle,
        "oracle_gain_vs_cap050": float(oracle["mean_delta"]) - float(baseline["mean_delta"]),
        "best_single_threshold_policy": policies[0],
        "best_single_threshold_gain_vs_cap050": float(policies[0]["mean_delta"]) - float(baseline["mean_delta"]),
        "leave_one_seed_cv": cv,
        "feature_correlations": correlations,
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(OUT_PREFIX.with_suffix(".policies.csv"), policies[:50])
    write_csv(OUT_PREFIX.with_suffix(".cv.csv"), cv["held_rows"])  # type: ignore[arg-type]
    write_csv(OUT_PREFIX.with_suffix(".correlations.csv"), correlations)

    md = [
        "# E114 max500 per-image selector-cap headroom",
        "",
        "## Summary",
        "",
        f"- Baseline cap0.50 transfer delta: {float(baseline['mean_delta']):.6f}",
        f"- Per-image cap oracle transfer delta: {float(oracle['mean_delta']):.6f}",
        f"- Oracle gain vs cap0.50: {float(payload['oracle_gain_vs_cap050']):.6f}",
        f"- Oracle cap counts: {oracle['cap_counts']}",
        f"- Best single-threshold policy delta: {float(policies[0]['mean_delta']):.6f}",
        f"- Best single-threshold gain vs cap0.50: {float(payload['best_single_threshold_gain_vs_cap050']):.6f}",
        f"- Leave-one-seed CV policy delta: {float(cv['mean_delta']):.6f}",
        f"- Leave-one-seed CV gain vs cap0.50: {float(cv['gain_vs_cap050']):.6f}",
        "",
        "## Best Single-Threshold Policy",
        "",
        "| feature | direction | threshold | alt cap | action rate | mean delta |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {policies[0]['feature']} | {policies[0]['direction']} | "
            f"{float(policies[0]['threshold']):.6f} | {float(policies[0]['alt_cap']):.2f} | "
            f"{float(policies[0]['action_rate']):.6f} | {float(policies[0]['mean_delta']):.6f} |"
        ),
        "",
        "## Leave-One-Seed CV",
        "",
        "| held seed | feature | direction | threshold | alt cap | test delta | gain vs cap0.50 | action rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cv["held_rows"]:  # type: ignore[index]
        md.append(
            f"| {row['held_seed']} | {row['feature']} | {row['direction']} | "
            f"{float(row['threshold']):.6f} | {float(row['alt_cap']):.2f} | "
            f"{float(row['test_delta']):.6f} | {float(row['test_gain_vs_cap050']):.6f} | "
            f"{float(row['test_action_rate']):.6f} |"
        )
    md.extend(
        [
            "",
            "## Top Feature Correlations",
            "",
            "| feature | pearson with lower-cap gain |",
            "|---|---:|",
        ]
    )
    for row in correlations[:10]:
        md.append(f"| {row['feature']} | {float(row['pearson_with_lower_cap_gain']):.6f} |")
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
