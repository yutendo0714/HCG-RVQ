#!/usr/bin/env python3
"""Split-protocol audit for usage-aware staged-geometry controllers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
DEFAULT_INPUT = ANALYSIS_DIR / "e129_staged_geometry_kodak24_audit_per_image.csv"
DEFAULT_OUT_PREFIX = ANALYSIS_DIR / "e130_usage_controller_split_protocol"
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
    objective: str
    budget: float


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
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def thresholds(values: list[float]) -> list[float]:
    vals = sorted(set(finite(values)))
    if not vals:
        return []
    out = [vals[0] - 1e-12, vals[-1] + 1e-12]
    out.extend(vals)
    out.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(out))


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


def load_records(path: Path) -> list[dict[str, object]]:
    raw_rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
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
            records.append(record)
    return records


def split_protocols(image_indices: list[int]) -> dict[str, tuple[set[int], set[int]]]:
    indices = sorted(image_indices)
    half = len(indices) // 2
    first = set(indices[:half])
    last = set(indices[half:])
    even = {idx for idx in indices if idx % 2 == 0}
    odd = set(indices) - even
    return {
        "first_half_to_second_half": (first, last),
        "second_half_to_first_half": (last, first),
        "even_to_odd": (even, odd),
        "odd_to_even": (odd, even),
    }


def is_selected(policy: Policy | None, record: dict[str, object]) -> bool:
    if policy is None:
        return False
    if record["case"] != policy.case:
        return False
    value = float(record[policy.feature])
    if not math.isfinite(value):
        return False
    return value <= policy.threshold if policy.side == "<=" else value >= policy.threshold


def eval_policy(records: list[dict[str, object]], policy: Policy | None) -> dict[str, object]:
    if policy is None:
        case = "noop"
        scope = "noop"
        feature = "noop"
        side = "noop"
        threshold = float("nan")
        objective = "noop"
        budget = float("nan")
    else:
        case = policy.case
        scope = policy.scope
        feature = policy.feature
        side = policy.side
        threshold = policy.threshold
        objective = policy.objective
        budget = policy.budget

    deltas_rd: list[float] = []
    deltas_dead: list[float] = []
    deltas_qmse: list[float] = []
    selected_dead: list[float] = []
    selected_rd: list[float] = []
    selected = 0
    for record in records:
        if is_selected(policy, record):
            selected += 1
            delta_rd = float(record["delta_rd_score"])
            delta_dead = float(record["delta_dead_code_ratio"])
            delta_qmse = float(record["delta_latent_quant_mse"])
            selected_dead.append(delta_dead)
            selected_rd.append(delta_rd)
            deltas_rd.append(delta_rd)
            deltas_dead.append(delta_dead)
            deltas_qmse.append(delta_qmse)
        else:
            deltas_rd.append(0.0)
            deltas_dead.append(0.0)
            deltas_qmse.append(0.0)
    return {
        "case": case,
        "scope": scope,
        "feature": feature,
        "side": side,
        "threshold": threshold,
        "objective": objective,
        "budget": budget,
        "num_images": len(records),
        "selected": selected,
        "mean_delta_rd": mean(deltas_rd),
        "mean_delta_dead": mean(deltas_dead),
        "mean_delta_qmse": mean(deltas_qmse),
        "q95_damage_rd": percentile([max(0.0, v) for v in deltas_rd], 0.95),
        "max_damage_rd": max([max(0.0, v) for v in deltas_rd]) if deltas_rd else float("nan"),
        "selected_max_delta_dead": max(selected_dead) if selected_dead else 0.0,
        "selected_q95_delta_dead": percentile(selected_dead, 0.95) if selected_dead else 0.0,
        "selected_win_rate": sum(1 for v in selected_rd if v < 0.0) / len(selected_rd) if selected_rd else float("nan"),
    }


def candidate_policies(train_records: list[dict[str, object]], case: str, scope: str, objective: str, budget: float) -> list[Policy]:
    rows = [r for r in train_records if r["case"] == case]
    policies: list[Policy] = []
    for feature in FEATURE_SCOPES[scope]:
        vals = [float(r[feature]) for r in rows]
        for threshold in thresholds(vals):
            for side in ("<=", ">="):
                policies.append(Policy(case=case, scope=scope, feature=feature, side=side, threshold=threshold, objective=objective, budget=budget))
    return policies


def select_policy(train_records: list[dict[str, object]], case: str, scope: str, objective: str, budget: float) -> tuple[Policy | None, dict[str, object]]:
    rows = [r for r in train_records if r["case"] == case]
    best_policy: Policy | None = None
    best_eval: dict[str, object] | None = None
    for policy in candidate_policies(rows, case, scope, objective, budget):
        metrics = eval_policy(rows, policy)
        if int(metrics["selected"]) <= 0:
            continue
        if objective == "mean_dead_budget":
            if float(metrics["mean_delta_dead"]) > budget:
                continue
        elif objective == "strict_selected_dead_cap":
            if float(metrics["selected_max_delta_dead"]) > budget:
                continue
        else:
            raise ValueError(f"unknown objective {objective}")
        key = (
            float(metrics["mean_delta_rd"]),
            float(metrics["q95_damage_rd"]),
            float(metrics["mean_delta_dead"]),
            -int(metrics["selected"]),
        )
        if best_eval is None:
            best_policy = policy
            best_eval = metrics
            best_key = key
        elif key < best_key:
            best_policy = policy
            best_eval = metrics
            best_key = key
    if best_eval is None:
        best_eval = eval_policy(rows, None)
    return best_policy, best_eval


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_test_rows(test_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(r["case"]), str(r["scope"]), str(r["objective"]), float(r["budget"])) for r in test_rows})
    for case, scope, objective, budget in keys:
        rows = [r for r in test_rows if r["case"] == case and r["scope"] == scope and r["objective"] == objective and float(r["budget"]) == budget]
        out.append(
            {
                "case": case,
                "scope": scope,
                "objective": objective,
                "budget": budget,
                "num_protocols": len(rows),
                "mean_selected": mean([float(r["selected"]) for r in rows]),
                "mean_delta_rd": mean([float(r["mean_delta_rd"]) for r in rows]),
                "mean_delta_dead": mean([float(r["mean_delta_dead"]) for r in rows]),
                "mean_q95_damage_rd": mean([float(r["q95_damage_rd"]) for r in rows]),
                "worst_delta_rd": max(float(r["mean_delta_rd"]) for r in rows) if rows else float("nan"),
                "worst_q95_damage_rd": max(float(r["q95_damage_rd"]) for r in rows) if rows else float("nan"),
                "positive_protocols": sum(1 for r in rows if float(r["mean_delta_rd"]) < 0.0),
            }
        )
    return out


def markdown(out_prefix: Path, input_csv: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    summary_sorted = sorted(summary, key=lambda r: (str(r["objective"]), float(r["budget"]), str(r["case"]), str(r["scope"])))
    test_sorted = sorted(rows, key=lambda r: (str(r["protocol"]), str(r["objective"]), float(r["budget"]), str(r["case"]), str(r["scope"])))
    lines = [
        "# E130 Usage Controller Split Protocol",
        "",
        "This audit selects single-feature geometry-use policies on one image split and evaluates the chosen policy on a disjoint split. It is still small-sample, but it is stricter than choosing thresholds on the same images being reported.",
        "",
        f"- Input: `{input_csv}`",
        f"- Baseline: `{BASELINE_CASE}`",
        "",
        "## Cross-Split Summary",
        "",
        "| objective | budget | case | scope | protocols won | selected | delta RD | delta dead | q95 damage | worst delta RD |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_sorted:
        lines.append(
            "| {objective} | {budget} | {case} | {scope} | {positive}/{total} | {selected} | {delta_rd} | {delta_dead} | {q95} | {worst_rd} |".format(
                objective=row["objective"],
                budget=fmt(float(row["budget"]), 3),
                case=row["case"],
                scope=row["scope"],
                positive=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(float(row["mean_selected"]), 2),
                delta_rd=fmt(float(row["mean_delta_rd"])),
                delta_dead=fmt(float(row["mean_delta_dead"])),
                q95=fmt(float(row["mean_q95_damage_rd"])),
                worst_rd=fmt(float(row["worst_delta_rd"])),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Policies On Test Splits",
            "",
            "| protocol | objective | budget | case | scope | feature | rule | selected | delta RD | delta dead | q95 damage | selected max dead |",
            "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in test_sorted:
        lines.append(
            "| {protocol} | {objective} | {budget} | {case} | {scope} | {feature} | {side} {threshold} | {selected} | {delta_rd} | {delta_dead} | {q95} | {max_dead} |".format(
                protocol=row["protocol"],
                objective=row["objective"],
                budget=fmt(float(row["budget"]), 3),
                case=row["case"],
                scope=row["scope"],
                feature=row["feature"],
                side=row["side"],
                threshold=fmt(float(row["threshold"])),
                selected=row["selected"],
                delta_rd=fmt(float(row["mean_delta_rd"])),
                delta_dead=fmt(float(row["mean_delta_dead"])),
                q95=fmt(float(row["q95_damage_rd"])),
                max_dead=fmt(float(row["selected_max_delta_dead"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `mean_dead_budget` is the expected-usage controller. It is useful for average RD trade-offs but can still allow individual images with larger dead-code increases.",
            "- `strict_selected_dead_cap` is the safer reliability view. It is much harder for simple one-feature rules, so positive results here are stronger but smaller.",
            "- `posthoc_diagnostic` rows are upper bounds. The method-relevant rows are `baseline_only` and especially `candidate_forward`, provided the decision can be made reproducible for decoding.",
            "",
            "## Artifacts",
            "",
            f"- `{out_prefix.with_suffix('.json')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_test.csv')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_train.csv')}`",
            f"- `{out_prefix.with_name(out_prefix.name + '_summary.csv')}`",
        ]
    )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-prefix", default=str(DEFAULT_OUT_PREFIX))
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = ROOT / input_csv
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = ROOT / out_prefix

    records = load_records(input_csv)
    image_indices = sorted({int(r["image_index"]) for r in records})
    protocols = split_protocols(image_indices)
    cases = sorted({str(r["case"]) for r in records})

    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    objectives = ["mean_dead_budget", "strict_selected_dead_cap"]
    for protocol_name, (train_ids, test_ids) in protocols.items():
        for case in cases:
            for scope in FEATURE_SCOPES:
                for objective in objectives:
                    for budget in DEAD_BUDGETS:
                        train_records = [r for r in records if int(r["image_index"]) in train_ids and r["case"] == case]
                        test_records = [r for r in records if int(r["image_index"]) in test_ids and r["case"] == case]
                        policy, train_metrics = select_policy(train_records, case, scope, objective, budget)
                        test_metrics = eval_policy(test_records, policy)
                        train_row = dict(train_metrics)
                        test_row = dict(test_metrics)
                        for row, split_name in ((train_row, "train"), (test_row, "test")):
                            if row["case"] == "noop":
                                row["case"] = case
                                row["scope"] = scope
                                row["objective"] = objective
                                row["budget"] = budget
                            row["protocol"] = protocol_name
                            row["split"] = split_name
                            row["train_images"] = ",".join(str(i) for i in sorted(train_ids))
                            row["test_images"] = ",".join(str(i) for i in sorted(test_ids))
                        train_rows.append(train_row)
                        test_rows.append(test_row)

    summary = summarize_test_rows(test_rows)
    payload = {
        "experiment": "E130 usage controller split protocol",
        "input": str(input_csv),
        "baseline": BASELINE_CASE,
        "dead_budgets": DEAD_BUDGETS,
        "protocols": {name: {"train": sorted(train), "test": sorted(test)} for name, (train, test) in protocols.items()},
        "summary": summary,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(out_prefix.with_name(out_prefix.name + "_train.csv"), train_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_test.csv"), test_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary)
    markdown(out_prefix, input_csv, test_rows, summary)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
