#!/usr/bin/env python3
"""Build a compact decision package from E129/E130 usage-controller audits."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
E129_SUMMARY = ANALYSIS_DIR / "e129_staged_geometry_kodak24_audit_summary.csv"
E129_SELECTORS = ANALYSIS_DIR / "e129_staged_geometry_kodak24_audit_selectors.csv"
E130_SUMMARY = ANALYSIS_DIR / "e130_usage_controller_split_protocol_summary.csv"
E130_TEST = ANALYSIS_DIR / "e130_usage_controller_split_protocol_test.csv"
OUT_PREFIX = ANALYSIS_DIR / "e131_usage_controller_decision_package"


def f(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def e129_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in read_csv(E129_SUMMARY):
        out.append(
            {
                "source": "e129_checkpoint",
                "case": row["case"],
                "mean_delta_rd": f(row["mean_delta_rd"]),
                "win_rate_rd": f(row["win_rate_rd"]),
                "q95_damage_rd": f(row["q95_damage_rd"]),
                "max_damage_rd": f(row["max_damage_rd"]),
                "mean_delta_dead": f(row["mean_delta_dead"]),
                "mean_delta_perplexity": f(row["mean_delta_perplexity"]),
                "mean_delta_qmse": f(row["mean_delta_qmse"]),
                "nonfinite_sum": f(row["nonfinite_sum"]),
            }
        )
    for row in read_csv(E129_SELECTORS):
        out.append(
            {
                "source": "e129_oracle_selector",
                "case": row["policy"],
                "mean_delta_rd": f(row["delta_vs_baseline"]),
                "win_rate_rd": float("nan"),
                "q95_damage_rd": float("nan"),
                "max_damage_rd": float("nan"),
                "mean_delta_dead": f(row["mean_delta_dead"]),
                "mean_delta_perplexity": float("nan"),
                "mean_delta_qmse": f(row["mean_delta_qmse"]),
                "nonfinite_sum": 0.0,
                "selected": f(row["hcg_selected"]),
            }
        )
    return out


def e130_candidate_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in read_csv(E130_SUMMARY):
        case = row["case"]
        scope = row["scope"]
        objective = row["objective"]
        if case != "staged_gate001_step30":
            continue
        if scope not in {"candidate_forward", "baseline_only", "posthoc_diagnostic"}:
            continue
        out.append(
            {
                "case": case,
                "scope": scope,
                "objective": objective,
                "budget": f(row["budget"]),
                "positive_protocols": int(float(row["positive_protocols"])),
                "num_protocols": int(float(row["num_protocols"])),
                "mean_selected": f(row["mean_selected"]),
                "mean_delta_rd": f(row["mean_delta_rd"]),
                "mean_delta_dead": f(row["mean_delta_dead"]),
                "mean_q95_damage_rd": f(row["mean_q95_damage_rd"]),
                "worst_delta_rd": f(row["worst_delta_rd"]),
                "worst_q95_damage_rd": f(row["worst_q95_damage_rd"]),
            }
        )
    return out


def policy_stability_rows() -> list[dict[str, object]]:
    counters: dict[tuple[str, str, float], Counter[tuple[str, str]]] = defaultdict(Counter)
    thresholds: dict[tuple[str, str, float, str, str], list[float]] = defaultdict(list)
    for row in read_csv(E130_TEST):
        if row["case"] != "staged_gate001_step30" or row["scope"] != "candidate_forward":
            continue
        key = (row["objective"], row["scope"], f(row["budget"]))
        feature_key = (row["feature"], row["side"])
        counters[key][feature_key] += 1
        thresholds[(row["objective"], row["scope"], f(row["budget"]), row["feature"], row["side"])].append(f(row["threshold"]))

    out: list[dict[str, object]] = []
    for (objective, scope, budget), counter in sorted(counters.items()):
        for (feature, side), count in counter.most_common():
            vals = [v for v in thresholds[(objective, scope, budget, feature, side)] if math.isfinite(v)]
            out.append(
                {
                    "case": "staged_gate001_step30",
                    "scope": scope,
                    "objective": objective,
                    "budget": budget,
                    "feature": feature,
                    "side": side,
                    "count": count,
                    "threshold_mean": sum(vals) / len(vals) if vals else float("nan"),
                    "threshold_min": min(vals) if vals else float("nan"),
                    "threshold_max": max(vals) if vals else float("nan"),
                }
            )
    return out


def recommendation_rows(e130_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [r for r in e130_rows if r["scope"] == "candidate_forward"]
    def find(objective: str, budget: float) -> dict[str, object]:
        for row in rows:
            if row["objective"] == objective and abs(float(row["budget"]) - budget) < 1e-12:
                return row
        raise KeyError((objective, budget))

    mean005 = find("mean_dead_budget", 0.05)
    mean0075 = find("mean_dead_budget", 0.075)
    strict0075 = find("strict_selected_dead_cap", 0.075)
    strict010 = find("strict_selected_dead_cap", 0.10)
    return [
        {
            "rank": 1,
            "role": "next_implementation_target",
            "objective": mean005["objective"],
            "budget": mean005["budget"],
            "reason": "best balance near the observed usage cost: all split protocols win, mean RD gain is large, and mean dead-code delta stays near the budget.",
            **{f"metric_{k}": v for k, v in mean005.items() if k not in {"case", "scope", "objective", "budget"}},
        },
        {
            "rank": 2,
            "role": "mean_rd_ablation",
            "objective": mean0075["objective"],
            "budget": mean0075["budget"],
            "reason": "stronger mean RD gain with all protocols winning, useful as an ablation if the paper can tolerate a looser expected usage budget.",
            **{f"metric_{k}": v for k, v in mean0075.items() if k not in {"case", "scope", "objective", "budget"}},
        },
        {
            "rank": 3,
            "role": "strict_safety_ablation",
            "objective": strict0075["objective"],
            "budget": strict0075["budget"],
            "reason": "safe-subset evidence: all protocols win under per-selected dead cap, but selected coverage is smaller.",
            **{f"metric_{k}": v for k, v in strict0075.items() if k not in {"case", "scope", "objective", "budget"}},
        },
        {
            "rank": 4,
            "role": "strict_safety_looser",
            "objective": strict010["objective"],
            "budget": strict010["budget"],
            "reason": "looser strict-cap backup with more selected images, useful if cap0.075 is too conservative in future holdout.",
            **{f"metric_{k}": v for k, v in strict010.items() if k not in {"case", "scope", "objective", "budget"}},
        },
    ]


def write_markdown(
    e129: list[dict[str, object]],
    e130: list[dict[str, object]],
    stability: list[dict[str, object]],
    recommendations: list[dict[str, object]],
) -> None:
    lines = [
        "# E131 Usage Controller Decision Package",
        "",
        "This package turns E129/E130 into a concrete next implementation decision. It does not introduce new GPU results; it organizes the checkpoint, feature-distribution, and split-protocol evidence for the usage-aware HCG geometry controller.",
        "",
        "## E129 Checkpoint And Oracle Context",
        "",
        "| source | case/policy | delta RD | win rate | q95 damage | delta dead | delta perplexity | selected | nonfinite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in e129:
        if row["source"] == "e129_checkpoint" and row["case"] == "hcs_warmup_step30":
            continue
        lines.append(
            "| {source} | {case} | {rd} | {win} | {q95} | {dead} | {perp} | {selected} | {nonfinite} |".format(
                source=row["source"],
                case=row["case"],
                rd=fmt(float(row["mean_delta_rd"])),
                win=fmt(float(row["win_rate_rd"])),
                q95=fmt(float(row["q95_damage_rd"])),
                dead=fmt(float(row["mean_delta_dead"])),
                perp=fmt(float(row["mean_delta_perplexity"])),
                selected=fmt(row.get("selected", "")),
                nonfinite=fmt(float(row["nonfinite_sum"])),
            )
        )

    lines.extend(
        [
            "",
            "## E130 Candidate-Forward Controller Options",
            "",
            "| objective | budget | protocols won | selected | delta RD | delta dead | q95 damage | worst delta RD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in e130:
        if row["scope"] != "candidate_forward":
            continue
        lines.append(
            "| {objective} | {budget} | {wins}/{total} | {selected} | {rd} | {dead} | {q95} | {worst} |".format(
                objective=row["objective"],
                budget=fmt(float(row["budget"]), 3),
                wins=row["positive_protocols"],
                total=row["num_protocols"],
                selected=fmt(float(row["mean_selected"]), 2),
                rd=fmt(float(row["mean_delta_rd"])),
                dead=fmt(float(row["mean_delta_dead"])),
                q95=fmt(float(row["mean_q95_damage_rd"])),
                worst=fmt(float(row["worst_delta_rd"])),
            )
        )

    lines.extend(
        [
            "",
            "## Selected Feature Stability",
            "",
            "| objective | budget | feature | side | count | threshold mean | threshold range |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in stability:
        if row["count"] <= 0:
            continue
        lines.append(
            "| {objective} | {budget} | {feature} | {side} | {count} | {mean} | {lo}..{hi} |".format(
                objective=row["objective"],
                budget=fmt(float(row["budget"]), 3),
                feature=row["feature"],
                side=row["side"],
                count=row["count"],
                mean=fmt(float(row["threshold_mean"])),
                lo=fmt(float(row["threshold_min"])),
                hi=fmt(float(row["threshold_max"])),
            )
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "| rank | role | objective | budget | selected | delta RD | delta dead | q95 damage | reason |",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in recommendations:
        lines.append(
            "| {rank} | {role} | {objective} | {budget} | {selected} | {rd} | {dead} | {q95} | {reason} |".format(
                rank=row["rank"],
                role=row["role"],
                objective=row["objective"],
                budget=fmt(float(row["budget"]), 3),
                selected=fmt(float(row["metric_mean_selected"]), 2),
                rd=fmt(float(row["metric_mean_delta_rd"])),
                dead=fmt(float(row["metric_mean_delta_dead"])),
                q95=fmt(float(row["metric_mean_q95_damage_rd"])),
                reason=row["reason"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The next implementation target should be the full-gate `staged_gate001` branch with a candidate-forward expected-usage controller, not the half-gate branch.",
            "- Budget `0.05` is the best default target because it wins all split protocols and keeps mean dead-code increase close to the selected budget while retaining most of the RD gain.",
            "- Strict-cap `0.075` should be carried as the conservative safety ablation; it wins all split protocols but selects fewer images.",
            "- The selected features are not one single scalar yet. This argues for either a tiny learned reliability head or a small hand-designed candidate-forward feature set, selected on an independent split and confirmed on holdout.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_checkpoint_context.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_controller_options.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_feature_stability.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_recommendations.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    e129 = e129_rows()
    e130 = e130_candidate_rows()
    stability = policy_stability_rows()
    recommendations = recommendation_rows(e130)
    payload = {
        "experiment": "E131 usage controller decision package",
        "checkpoint_context": e129,
        "controller_options": e130,
        "feature_stability": stability,
        "recommendations": recommendations,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_checkpoint_context.csv"), e129)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_controller_options.csv"), e130)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_feature_stability.csv"), stability)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_recommendations.csv"), recommendations)
    write_markdown(e129, e130, stability, recommendations)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
