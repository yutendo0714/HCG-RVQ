#!/usr/bin/env python3
"""Feature audit for usage-aware staged-geometry gating after E127."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
INPUT_CSV = ANALYSIS_DIR / "e127_staged_geometry_per_image_audit_per_image.csv"
OUT_PREFIX = ANALYSIS_DIR / "e128_usage_aware_gate_feature_audit"
BASELINE_CASE = "hcs_warmup_step30"
DEAD_BUDGETS = [0.025, 0.05, 0.075, 0.10]

FEATURE_SCOPES: dict[str, list[str]] = {
    "baseline_only": [
        "base_dead_code_ratio",
        "base_perplexity",
        "base_stage_entropy",
        "base_latent_quant_mse",
        "base_rd_score",
        "base_mse",
        "base_psnr",
    ],
    "candidate_forward": [
        "hcg_dead_code_ratio",
        "hcg_perplexity",
        "hcg_stage_entropy",
        "hcg_latent_quant_mse",
        "hcg_s_q_mean",
        "hcg_s_q_std",
        "hcg_mu_q_abs_mean",
        "hcg_householder_delta_rms",
        "hcg_householder_v_abs_mean",
    ],
    "posthoc_diagnostic": [
        "delta_dead_code_ratio",
        "delta_perplexity",
        "delta_stage_entropy",
        "delta_latent_quant_mse",
        "delta_s_q_mean",
        "delta_s_q_std",
    ],
}


@dataclass(frozen=True)
class Policy:
    case: str
    scope: str
    feature: str
    side: str
    threshold: float


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: list[float], q: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(lo)]
    frac = pos - lo
    return vals[int(lo)] * (1.0 - frac) + vals[int(hi)] * frac


def pearson(x_vals: list[float], y_vals: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(x_vals, y_vals) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def rank(values: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(values))
    ranks = [float("nan")] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        avg = (pos + end - 1) / 2.0
        for _, idx in indexed[pos:end]:
            ranks[idx] = avg
        pos = end
    return ranks


def spearman(x_vals: list[float], y_vals: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(x_vals, y_vals) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    return pearson(rank(xs), rank(ys))


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def load_records() -> list[dict[str, object]]:
    raw_rows: list[dict[str, str]] = []
    with INPUT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        raw_rows.extend(reader)

    by_image: dict[int, dict[str, dict[str, str]]] = {}
    for row in raw_rows:
        by_image.setdefault(int(row["image_index"]), {})[row["case"]] = row

    records: list[dict[str, object]] = []
    for image_index, rows in sorted(by_image.items()):
        base = rows[BASELINE_CASE]
        for case, row in sorted(rows.items()):
            if case == BASELINE_CASE:
                continue
            record: dict[str, object] = {
                "case": case,
                "image_index": image_index,
                "path": row["path"],
                "baseline_rd": parse_float(base["rd_score"]),
                "candidate_rd": parse_float(row["rd_score"]),
                "delta_rd_score": parse_float(row["delta_rd_score"]),
                "delta_dead_code_ratio": parse_float(row["delta_dead_code_ratio"]),
                "delta_perplexity": parse_float(row["delta_perplexity"]),
                "delta_stage_entropy": parse_float(row["delta_stage_entropy"]),
                "delta_latent_quant_mse": parse_float(row["delta_latent_quant_mse"]),
                "delta_s_q_mean": parse_float(row.get("delta_s_q_mean")),
                "delta_s_q_std": parse_float(row.get("delta_s_q_std")),
            }
            for key in ("dead_code_ratio", "perplexity", "stage_entropy", "latent_quant_mse", "rd_score", "mse", "psnr"):
                record[f"base_{key}"] = parse_float(base.get(key))
            for key in (
                "dead_code_ratio",
                "perplexity",
                "stage_entropy",
                "latent_quant_mse",
                "s_q_mean",
                "s_q_std",
                "mu_q_abs_mean",
                "householder_delta_rms",
                "householder_v_abs_mean",
            ):
                record[f"hcg_{key}"] = parse_float(row.get(key))
            record["rd_win"] = float(record["delta_rd_score"]) < 0.0
            for budget in DEAD_BUDGETS:
                record[f"safe_dead_le_{budget:.3f}"] = float(record["delta_dead_code_ratio"]) <= budget
                record[f"safe_win_dead_le_{budget:.3f}"] = (
                    bool(record["rd_win"]) and float(record["delta_dead_code_ratio"]) <= budget
                )
            records.append(record)
    return records


def thresholds(values: list[float]) -> list[float]:
    vals = sorted(set(finite(values)))
    if not vals:
        return []
    out: list[float] = [vals[0] - 1e-12, vals[-1] + 1e-12]
    out.extend(vals)
    out.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(out))


def selected_by(policy: Policy, record: dict[str, object]) -> bool:
    value = float(record[policy.feature])
    if not math.isfinite(value):
        return False
    if policy.side == "<=":
        return value <= policy.threshold
    return value >= policy.threshold


def evaluate_policy(records: list[dict[str, object]], policy: Policy) -> dict[str, object]:
    selected: list[dict[str, object]] = []
    all_rows = [r for r in records if r["case"] == policy.case]
    for record in all_rows:
        if selected_by(policy, record):
            selected.append(record)

    candidate_delta_rd = [float(r["delta_rd_score"]) if selected_by(policy, r) else 0.0 for r in all_rows]
    candidate_delta_dead = [float(r["delta_dead_code_ratio"]) if selected_by(policy, r) else 0.0 for r in all_rows]
    candidate_delta_qmse = [float(r["delta_latent_quant_mse"]) if selected_by(policy, r) else 0.0 for r in all_rows]
    selected_delta_dead = [float(r["delta_dead_code_ratio"]) for r in selected]
    safe05 = [r for r in selected if bool(r["safe_win_dead_le_0.050"])]
    safe075 = [r for r in selected if bool(r["safe_win_dead_le_0.075"])]
    return {
        "case": policy.case,
        "scope": policy.scope,
        "feature": policy.feature,
        "side": policy.side,
        "threshold": policy.threshold,
        "num_images": len(all_rows),
        "selected": len(selected),
        "mean_delta_rd": mean(candidate_delta_rd),
        "mean_delta_dead": mean(candidate_delta_dead),
        "mean_delta_qmse": mean(candidate_delta_qmse),
        "selected_max_delta_dead": max(selected_delta_dead) if selected_delta_dead else 0.0,
        "selected_q95_delta_dead": percentile(selected_delta_dead, 0.95) if selected_delta_dead else 0.0,
        "q95_damage_rd": percentile([max(0.0, v) for v in candidate_delta_rd], 0.95),
        "max_damage_rd": max([max(0.0, v) for v in candidate_delta_rd]) if candidate_delta_rd else float("nan"),
        "selected_rd_win_rate": sum(1 for r in selected if bool(r["rd_win"])) / len(selected) if selected else float("nan"),
        "selected_safe_win_dead005_rate": len(safe05) / len(selected) if selected else float("nan"),
        "selected_safe_win_dead0075_rate": len(safe075) / len(selected) if selected else float("nan"),
    }


def all_threshold_policies(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = sorted({str(r["case"]) for r in records})
    for case in cases:
        case_records = [r for r in records if r["case"] == case]
        for scope, features in FEATURE_SCOPES.items():
            for feature in features:
                vals = [float(r[feature]) for r in case_records]
                for threshold in thresholds(vals):
                    for side in ("<=", ">="):
                        rows.append(evaluate_policy(records, Policy(case, scope, feature, side, threshold)))
    return rows


def best_by_budget(policy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(r["case"]), str(r["scope"])) for r in policy_rows})
    for case, scope in keys:
        subset = [r for r in policy_rows if r["case"] == case and r["scope"] == scope]
        for budget in DEAD_BUDGETS:
            valid = [r for r in subset if float(r["mean_delta_dead"]) <= budget and int(r["selected"]) > 0]
            if not valid:
                valid = [r for r in subset if float(r["mean_delta_dead"]) <= budget]
            if not valid:
                continue
            best = min(valid, key=lambda r: (float(r["mean_delta_rd"]), float(r["q95_damage_rd"]), -int(r["selected"])))
            row = dict(best)
            row["dead_budget"] = budget
            out.append(row)
    return out


def best_by_strict_cap(policy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(r["case"]), str(r["scope"])) for r in policy_rows})
    for case, scope in keys:
        subset = [r for r in policy_rows if r["case"] == case and r["scope"] == scope]
        for cap in DEAD_BUDGETS:
            valid = [
                r
                for r in subset
                if int(r["selected"]) > 0 and float(r["selected_max_delta_dead"]) <= cap
            ]
            if not valid:
                continue
            best = min(valid, key=lambda r: (float(r["mean_delta_rd"]), float(r["q95_damage_rd"]), -int(r["selected"])))
            row = dict(best)
            row["dead_cap"] = cap
            out.append(row)
    return out


def leave_one_out(records: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    image_indices = sorted({int(r["image_index"]) for r in records})
    cases = sorted({str(r["case"]) for r in records})
    for case in cases:
        for scope, features in FEATURE_SCOPES.items():
            scoped = [r for r in records if r["case"] == case]
            for budget in DEAD_BUDGETS:
                held_deltas_rd: list[float] = []
                held_deltas_dead: list[float] = []
                selected_count = 0
                chosen_features: list[str] = []
                for held in image_indices:
                    train = [r for r in scoped if int(r["image_index"]) != held]
                    held_records = [r for r in scoped if int(r["image_index"]) == held]
                    if not train or not held_records:
                        continue
                    train_policy_rows: list[dict[str, object]] = []
                    for feature in features:
                        vals = [float(r[feature]) for r in train]
                        for threshold in thresholds(vals):
                            for side in ("<=", ">="):
                                train_policy_rows.append(evaluate_policy(train, Policy(case, scope, feature, side, threshold)))
                    valid = [r for r in train_policy_rows if float(r["mean_delta_dead"]) <= budget and int(r["selected"]) > 0]
                    if not valid:
                        valid = [r for r in train_policy_rows if float(r["mean_delta_dead"]) <= budget]
                    if not valid:
                        held_deltas_rd.append(0.0)
                        held_deltas_dead.append(0.0)
                        chosen_features.append("none")
                        continue
                    best = min(valid, key=lambda r: (float(r["mean_delta_rd"]), float(r["q95_damage_rd"]), -int(r["selected"])))
                    policy = Policy(
                        case=case,
                        scope=scope,
                        feature=str(best["feature"]),
                        side=str(best["side"]),
                        threshold=float(best["threshold"]),
                    )
                    held_record = held_records[0]
                    if selected_by(policy, held_record):
                        held_deltas_rd.append(float(held_record["delta_rd_score"]))
                        held_deltas_dead.append(float(held_record["delta_dead_code_ratio"]))
                        selected_count += 1
                    else:
                        held_deltas_rd.append(0.0)
                        held_deltas_dead.append(0.0)
                    chosen_features.append(policy.feature)
                out.append(
                    {
                        "case": case,
                        "scope": scope,
                        "dead_budget": budget,
                        "num_images": len(held_deltas_rd),
                        "selected": selected_count,
                        "mean_delta_rd": mean(held_deltas_rd),
                        "mean_delta_dead": mean(held_deltas_dead),
                        "q95_damage_rd": percentile([max(0.0, v) for v in held_deltas_rd], 0.95),
                        "chosen_features": ";".join(chosen_features),
                    }
                )
    return out


def correlation_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for case in sorted({str(r["case"]) for r in records}):
        rows = [r for r in records if r["case"] == case]
        for scope, features in FEATURE_SCOPES.items():
            for feature in features:
                xs = [float(r[feature]) for r in rows]
                out.append(
                    {
                        "case": case,
                        "scope": scope,
                        "feature": feature,
                        "pearson_delta_dead": pearson(xs, [float(r["delta_dead_code_ratio"]) for r in rows]),
                        "spearman_delta_dead": spearman(xs, [float(r["delta_dead_code_ratio"]) for r in rows]),
                        "pearson_delta_rd": pearson(xs, [float(r["delta_rd_score"]) for r in rows]),
                        "spearman_delta_rd": spearman(xs, [float(r["delta_rd_score"]) for r in rows]),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    records: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    strict_cap_rows: list[dict[str, object]],
    loo_rows: list[dict[str, object]],
    corr_rows: list[dict[str, object]],
) -> None:
    top_best = sorted(best_rows, key=lambda r: (float(r["dead_budget"]), str(r["case"]), str(r["scope"])))
    top_strict = sorted(strict_cap_rows, key=lambda r: (float(r["dead_cap"]), str(r["case"]), str(r["scope"])))
    top_loo = sorted(loo_rows, key=lambda r: (float(r["dead_budget"]), str(r["case"]), str(r["scope"])))
    corr_sorted = sorted(
        corr_rows,
        key=lambda r: (
            str(r["case"]),
            str(r["scope"]),
            -abs(float(r["spearman_delta_dead"])) if math.isfinite(float(r["spearman_delta_dead"])) else -1.0,
        ),
    )
    lines = [
        "# E128 Usage-Aware Gate Feature Audit",
        "",
        "This audit uses the corrected E127 per-image checkpoint results to estimate whether staged geometry can be controlled by simple reliability or usage features. It is an analysis step, not a new training result.",
        "",
        f"- Input: `{INPUT_CSV}`",
        f"- Baseline: `{BASELINE_CASE}`",
        f"- Candidate rows: `{len(records)}`",
        "",
        "## Best Single-Feature Policies Under Mean Dead-Code Budget",
        "",
        "| case | scope | dead budget | feature | rule | selected | delta RD | delta dead | selected max dead | q95 damage |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_best:
        lines.append(
            "| {case} | {scope} | {budget} | {feature} | {side} {threshold} | {selected} | {mean_delta_rd} | {mean_delta_dead} | {selected_max_delta_dead} | {q95_damage_rd} |".format(
                case=row["case"],
                scope=row["scope"],
                budget=fmt(float(row["dead_budget"]), 3),
                feature=row["feature"],
                side=row["side"],
                threshold=fmt(float(row["threshold"])),
                selected=row["selected"],
                mean_delta_rd=fmt(float(row["mean_delta_rd"])),
                mean_delta_dead=fmt(float(row["mean_delta_dead"])),
                selected_max_delta_dead=fmt(float(row["selected_max_delta_dead"])),
                q95_damage_rd=fmt(float(row["q95_damage_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Best Single-Feature Policies Under Per-Selected Dead-Code Cap",
            "",
            "| case | scope | dead cap | feature | rule | selected | delta RD | delta dead | selected max dead | q95 damage |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_strict:
        lines.append(
            "| {case} | {scope} | {cap} | {feature} | {side} {threshold} | {selected} | {mean_delta_rd} | {mean_delta_dead} | {selected_max_delta_dead} | {q95_damage_rd} |".format(
                case=row["case"],
                scope=row["scope"],
                cap=fmt(float(row["dead_cap"]), 3),
                feature=row["feature"],
                side=row["side"],
                threshold=fmt(float(row["threshold"])),
                selected=row["selected"],
                mean_delta_rd=fmt(float(row["mean_delta_rd"])),
                mean_delta_dead=fmt(float(row["mean_delta_dead"])),
                selected_max_delta_dead=fmt(float(row["selected_max_delta_dead"])),
                q95_damage_rd=fmt(float(row["q95_damage_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Leave-One-Image Check",
            "",
            "| case | scope | dead budget | selected | delta RD | delta dead | q95 damage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_loo:
        lines.append(
            "| {case} | {scope} | {budget} | {selected} | {mean_delta_rd} | {mean_delta_dead} | {q95_damage_rd} |".format(
                case=row["case"],
                scope=row["scope"],
                budget=fmt(float(row["dead_budget"]), 3),
                selected=row["selected"],
                mean_delta_rd=fmt(float(row["mean_delta_rd"])),
                mean_delta_dead=fmt(float(row["mean_delta_dead"])),
                q95_damage_rd=fmt(float(row["q95_damage_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Strongest Dead-Code Correlations",
            "",
            "| case | scope | feature | Spearman dead | Pearson dead | Spearman RD |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    shown = 0
    for row in corr_sorted:
        if shown >= 24:
            break
        shown += 1
        lines.append(
            "| {case} | {scope} | {feature} | {spearman_delta_dead} | {pearson_delta_dead} | {spearman_delta_rd} |".format(
                case=row["case"],
                scope=row["scope"],
                feature=row["feature"],
                spearman_delta_dead=fmt(float(row["spearman_delta_dead"])),
                pearson_delta_dead=fmt(float(row["pearson_delta_dead"])),
                spearman_delta_rd=fmt(float(row["spearman_delta_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `posthoc_diagnostic` features are upper-bound diagnostics. They are useful for proving that usage-safe selection has headroom, but they are not paper-main deployable unless the corresponding signal can be made deterministic and available to the decoder or signaled.",
            "- `candidate_forward` features are closer to an implementable guard because they can be measured from the candidate quantization path, but this still needs protocol care if the decoder must reproduce the same decision.",
            "- `baseline_only` features are the safest deployment proxy, but they may be too weak on this 8-image smoke split. A weak baseline-only result should not be read as a failure of the HCG idea; it means the controller likely needs candidate usage statistics or a learned reliability head.",
            "- The mean-budget table is useful for expected RD/usage trade-off. The strict-cap table is safer for a manuscript reliability claim because it disallows selecting images whose individual dead-code increase exceeds the cap.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_features.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_policies.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_best.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_loo.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_correlations.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    records = load_records()
    policy_rows = all_threshold_policies(records)
    best_rows = best_by_budget(policy_rows)
    strict_cap_rows = best_by_strict_cap(policy_rows)
    loo_rows = leave_one_out(records)
    corr_rows = correlation_rows(records)
    payload = {
        "experiment": "E128 usage-aware gate feature audit",
        "input": str(INPUT_CSV),
        "baseline": BASELINE_CASE,
        "dead_budgets": DEAD_BUDGETS,
        "feature_scopes": FEATURE_SCOPES,
        "num_records": len(records),
        "best": best_rows,
        "strict_cap": strict_cap_rows,
        "loo": loo_rows,
        "correlations": corr_rows,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_features.csv"), records)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_policies.csv"), policy_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_best.csv"), best_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_strict_cap.csv"), strict_cap_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_loo.csv"), loo_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_correlations.csv"), corr_rows)
    write_markdown(records, best_rows, strict_cap_rows, loo_rows, corr_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
