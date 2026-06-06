#!/usr/bin/env python3
"""Materialize cross-fit usage-guard decisions from E130 policies."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
PER_IMAGE_CSV = ANALYSIS_DIR / "e129_staged_geometry_kodak24_audit_per_image.csv"
E130_TEST_CSV = ANALYSIS_DIR / "e130_usage_controller_split_protocol_test.csv"
E132_LABELS_CSV = ANALYSIS_DIR / "e132_usage_controller_teacher_labels_labels.csv"
OUT_PREFIX = ANALYSIS_DIR / "e134_usage_guard_crossfit_package"

BASELINE_CASE = "hcs_warmup_step30"
TARGET_CASE = "staged_gate001_step30"
TARGET_SCOPE = "candidate_forward"
ROLES = [
    ("next_implementation_target", "mean_dead_budget", 0.05),
    ("mean_rd_ablation", "mean_dead_budget", 0.075),
    ("strict_safety_ablation", "strict_selected_dead_cap", 0.075),
    ("strict_safety_looser", "strict_selected_dead_cap", 0.10),
]

FEATURES = [
    "base_dead_code_ratio",
    "base_perplexity",
    "base_stage_entropy",
    "base_latent_quant_mse",
    "base_rd_score",
    "hcg_dead_code_ratio",
    "hcg_perplexity",
    "hcg_stage_entropy",
    "hcg_latent_quant_mse",
    "hcg_householder_delta_rms",
    "hcg_householder_v_abs_mean",
    "delta_rd_score",
    "delta_dead_code_ratio",
    "delta_latent_quant_mse",
    "delta_perplexity",
]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


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


def load_records() -> dict[int, dict[str, object]]:
    raw = read_csv(PER_IMAGE_CSV)
    by_image: dict[int, dict[str, dict[str, str]]] = {}
    for row in raw:
        by_image.setdefault(int(row["image_index"]), {})[row["case"]] = row

    label_rows = {int(row["image_index"]): row for row in read_csv(E132_LABELS_CSV)}
    records: dict[int, dict[str, object]] = {}
    for image_index, rows in sorted(by_image.items()):
        base = rows[BASELINE_CASE]
        hcg = rows[TARGET_CASE]
        record: dict[str, object] = {
            "image_index": image_index,
            "path": hcg["path"],
            "baseline_rd": as_float(base["rd_score"]),
            "candidate_rd": as_float(hcg["rd_score"]),
            "delta_rd_score": as_float(hcg["delta_rd_score"]),
            "delta_dead_code_ratio": as_float(hcg["delta_dead_code_ratio"]),
            "delta_perplexity": as_float(hcg["delta_perplexity"]),
            "delta_stage_entropy": as_float(hcg["delta_stage_entropy"]),
            "delta_latent_quant_mse": as_float(hcg["delta_latent_quant_mse"]),
            "delta_s_q_mean": as_float(hcg.get("delta_s_q_mean")),
            "delta_s_q_std": as_float(hcg.get("delta_s_q_std")),
        }
        for key in ("dead_code_ratio", "perplexity", "stage_entropy", "latent_quant_mse", "rd_score", "mse", "psnr"):
            record[f"base_{key}"] = as_float(base.get(key))
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
            record[f"hcg_{key}"] = as_float(hcg.get(key))
        labels = label_rows.get(image_index, {})
        for label in ("rd_win", "safe_win_dead_le_0.050", "safe_win_dead_le_0.075", "safe_win_dead_le_0.100"):
            record[label] = int(as_float(labels.get(label, 0.0)))
        records[image_index] = record
    return records


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part != ""]


def select(policy: dict[str, str], record: dict[str, object]) -> bool:
    side = policy["side"]
    feature = policy["feature"]
    if side == "noop" or feature == "noop":
        return False
    value = as_float(record.get(feature))
    threshold = as_float(policy["threshold"])
    if not math.isfinite(value) or not math.isfinite(threshold):
        return False
    return value <= threshold if side == "<=" else value >= threshold


def materialize_detail(records: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    policies = [
        row
        for row in read_csv(E130_TEST_CSV)
        if row["case"] == TARGET_CASE
        and row["scope"] == TARGET_SCOPE
        and any(row["objective"] == objective and abs(as_float(row["budget"]) - budget) < 1e-12 for _, objective, budget in ROLES)
    ]
    detail: list[dict[str, object]] = []
    role_by_key = {(objective, budget): role for role, objective, budget in ROLES}
    for policy in policies:
        budget = as_float(policy["budget"])
        role = role_by_key[(policy["objective"], budget)]
        for image_index in parse_indices(policy["test_images"]):
            record = records[image_index]
            selected = select(policy, record)
            chosen_delta_rd = as_float(record["delta_rd_score"]) if selected else 0.0
            chosen_delta_dead = as_float(record["delta_dead_code_ratio"]) if selected else 0.0
            chosen_delta_qmse = as_float(record["delta_latent_quant_mse"]) if selected else 0.0
            out = {
                "role": role,
                "protocol": policy["protocol"],
                "objective": policy["objective"],
                "budget": budget,
                "case": policy["case"],
                "scope": policy["scope"],
                "feature": policy["feature"],
                "side": policy["side"],
                "threshold": as_float(policy["threshold"]),
                "image_index": image_index,
                "path": record["path"],
                "feature_value": as_float(record.get(policy["feature"])),
                "selected": int(selected),
                "chosen_case": TARGET_CASE if selected else BASELINE_CASE,
                "chosen_delta_rd": chosen_delta_rd,
                "chosen_delta_dead": chosen_delta_dead,
                "chosen_delta_qmse": chosen_delta_qmse,
                "damage_rd": max(0.0, chosen_delta_rd),
                "selected_rd_win": int(selected and as_float(record["delta_rd_score"]) < 0.0),
            }
            for key in FEATURES:
                out[key] = record.get(key, float("nan"))
            for label in ("rd_win", "safe_win_dead_le_0.050", "safe_win_dead_le_0.075", "safe_win_dead_le_0.100"):
                out[label] = record.get(label, 0)
            detail.append(out)
    return detail


def summarize(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, objective, budget in ROLES:
        role_rows = [r for r in detail if r["role"] == role]
        by_protocol: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in role_rows:
            by_protocol[str(row["protocol"])].append(row)
        protocol_summaries: list[dict[str, float]] = []
        for rows_in_protocol in by_protocol.values():
            selected_rows = [r for r in rows_in_protocol if int(r["selected"]) == 1]
            protocol_summaries.append(
                {
                    "selected": float(sum(int(r["selected"]) for r in rows_in_protocol)),
                    "delta_rd": mean([as_float(r["chosen_delta_rd"]) for r in rows_in_protocol]),
                    "delta_dead": mean([as_float(r["chosen_delta_dead"]) for r in rows_in_protocol]),
                    "delta_qmse": mean([as_float(r["chosen_delta_qmse"]) for r in rows_in_protocol]),
                    "q95_damage_rd": percentile([as_float(r["damage_rd"]) for r in rows_in_protocol], 0.95),
                    "max_damage_rd": max([as_float(r["damage_rd"]) for r in rows_in_protocol]) if rows_in_protocol else float("nan"),
                    "selected_win_rate": mean([as_float(r["selected_rd_win"]) for r in selected_rows]),
                    "selected_max_dead": max([as_float(r["delta_dead_code_ratio"]) for r in selected_rows]) if selected_rows else 0.0,
                    "label_safe075_rate": mean([as_float(r["safe_win_dead_le_0.075"]) for r in selected_rows]),
                }
            )
        rows.append(
            {
                "role": role,
                "objective": objective,
                "budget": budget,
                "num_protocols": len(protocol_summaries),
                "mean_selected": mean([r["selected"] for r in protocol_summaries]),
                "mean_delta_rd": mean([r["delta_rd"] for r in protocol_summaries]),
                "mean_delta_dead": mean([r["delta_dead"] for r in protocol_summaries]),
                "mean_delta_qmse": mean([r["delta_qmse"] for r in protocol_summaries]),
                "mean_q95_damage_rd": mean([r["q95_damage_rd"] for r in protocol_summaries]),
                "worst_delta_rd": max([r["delta_rd"] for r in protocol_summaries]) if protocol_summaries else float("nan"),
                "worst_q95_damage_rd": max([r["q95_damage_rd"] for r in protocol_summaries]) if protocol_summaries else float("nan"),
                "mean_selected_win_rate": mean([r["selected_win_rate"] for r in protocol_summaries]),
                "mean_selected_max_dead": mean([r["selected_max_dead"] for r in protocol_summaries]),
                "mean_selected_safe075_rate": mean([r["label_safe075_rate"] for r in protocol_summaries]),
            }
        )
    return rows


def vote_summary(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, objective, budget in ROLES:
        role_rows = [r for r in detail if r["role"] == role]
        by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in role_rows:
            by_image[int(row["image_index"])].append(row)
        counts = Counter(sum(int(r["selected"]) for r in image_rows) for image_rows in by_image.values())
        for image_index, image_rows in sorted(by_image.items()):
            votes = sum(int(r["selected"]) for r in image_rows)
            first = image_rows[0]
            rows.append(
                {
                    "role": role,
                    "objective": objective,
                    "budget": budget,
                    "image_index": image_index,
                    "path": first["path"],
                    "heldout_votes": len(image_rows),
                    "selected_votes": votes,
                    "selected_vote_rate": votes / len(image_rows) if image_rows else float("nan"),
                    "delta_rd_score": first["delta_rd_score"],
                    "delta_dead_code_ratio": first["delta_dead_code_ratio"],
                    "hcg_latent_quant_mse": first["hcg_latent_quant_mse"],
                    "hcg_householder_delta_rms": first["hcg_householder_delta_rms"],
                    "hcg_dead_code_ratio": first["hcg_dead_code_ratio"],
                    "rd_win": first["rd_win"],
                    "safe_win_dead_le_0.075": first["safe_win_dead_le_0.075"],
                    "role_vote_histogram": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                }
            )
    return rows


def feature_groups(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, _, _ in ROLES:
        role_rows = [r for r in detail if r["role"] == role]
        for selected in (0, 1):
            subset = [r for r in role_rows if int(r["selected"]) == selected]
            out: dict[str, object] = {
                "role": role,
                "selected": selected,
                "num_decisions": len(subset),
                "mean_chosen_delta_rd": mean([as_float(r["chosen_delta_rd"]) for r in subset]),
                "mean_chosen_delta_dead": mean([as_float(r["chosen_delta_dead"]) for r in subset]),
            }
            for feature in FEATURES:
                out[f"mean_{feature}"] = mean([as_float(r[feature]) for r in subset])
            rows.append(out)
    return rows


def write_markdown(summary_rows: list[dict[str, object]], vote_rows: list[dict[str, object]], feature_rows: list[dict[str, object]]) -> None:
    lines = [
        "# E134 Usage Guard Cross-Fit Package",
        "",
        "This package materializes the E130 split-selected candidate-forward policies into per-image decisions. It is analysis-only and adds no new GPU evaluation.",
        "",
        f"- Per-image input: `{PER_IMAGE_CSV}`",
        f"- Policy input: `{E130_TEST_CSV}`",
        f"- Case: `{TARGET_CASE}`",
        "",
        "## Cross-Fit Guard Summary",
        "",
        "| role | objective | budget | protocols | selected | delta RD | delta dead | q95 damage | selected win | selected max dead | safe075 among selected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {role} | {objective} | {budget} | {protocols} | {selected} | {rd} | {dead} | {q95} | {win} | {maxdead} | {safe} |".format(
                role=row["role"],
                objective=row["objective"],
                budget=fmt(float(row["budget"])),
                protocols=row["num_protocols"],
                selected=fmt(float(row["mean_selected"]), 2),
                rd=fmt(float(row["mean_delta_rd"])),
                dead=fmt(float(row["mean_delta_dead"])),
                q95=fmt(float(row["mean_q95_damage_rd"])),
                win=fmt(float(row["mean_selected_win_rate"])),
                maxdead=fmt(float(row["mean_selected_max_dead"])),
                safe=fmt(float(row["mean_selected_safe075_rate"])),
            )
        )

    lines.extend(
        [
            "",
            "## Vote Stability",
            "",
            "| role | vote histogram | both-vote selected | one-vote selected | never selected |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for role, _, _ in ROLES:
        rows = [r for r in vote_rows if r["role"] == role]
        hist = Counter(int(r["selected_votes"]) for r in rows)
        lines.append(
            f"| {role} | {rows[0]['role_vote_histogram'] if rows else ''} | {hist.get(2, 0)} | {hist.get(1, 0)} | {hist.get(0, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Main Guard Feature Contrast",
            "",
            "| selected | decisions | delta RD | delta dead | hcg qMSE | H-delta | hcg dead | hcg perplexity | base RD |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    main_features = [r for r in feature_rows if r["role"] == "next_implementation_target"]
    for row in main_features:
        lines.append(
            "| {selected} | {n} | {rd} | {dead} | {qmse} | {hdelta} | {hcgdead} | {perp} | {baserd} |".format(
                selected=row["selected"],
                n=row["num_decisions"],
                rd=fmt(float(row["mean_chosen_delta_rd"])),
                dead=fmt(float(row["mean_chosen_delta_dead"])),
                qmse=fmt(float(row["mean_hcg_latent_quant_mse"])),
                hdelta=fmt(float(row["mean_hcg_householder_delta_rms"])),
                hcgdead=fmt(float(row["mean_hcg_dead_code_ratio"])),
                perp=fmt(float(row["mean_hcg_perplexity"])),
                baserd=fmt(float(row["mean_base_rd_score"])),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main guard remains the E131 recommendation: candidate-forward mean dead-code budget `0.05`.",
            "- Materializing decisions confirms the guard is not merely a summary-table artifact; it selects concrete held-out images and keeps the split-protocol mean RD gain.",
            "- Mean-budget guards control expected usage cost, not individual-image safety. The selected max-dead columns show that tail usage damage can still exceed the nominal budget.",
            "- Strict-cap guards are also train-split constraints, so they should be reported as held-out safety probes rather than guaranteed hard caps until the decision rule is fixed and revalidated on a new split.",
            "- Vote stability is imperfect, which supports the earlier decision not to claim a universal one-scalar rule. The guard should be treated as a reproducible controller prototype, while the learned reliability head remains a parallel upside path.",
            "- The selected/rejected feature contrast is the next diagnostic bridge for an actual implementation: selected decisions should be checked for lower candidate qMSE, controlled Householder delta, and acceptable codebook usage.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_detail.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_summary.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_votes.csv')}`",
            f"- `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_feature_groups.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    records = load_records()
    detail = materialize_detail(records)
    summary_rows = summarize(detail)
    votes = vote_summary(detail)
    features = feature_groups(detail)
    payload = {
        "experiment": "E134 usage guard cross-fit package",
        "per_image_csv": str(PER_IMAGE_CSV),
        "policy_csv": str(E130_TEST_CSV),
        "target_case": TARGET_CASE,
        "roles": [
            {"role": role, "objective": objective, "budget": budget}
            for role, objective, budget in ROLES
        ],
        "summary": summary_rows,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_detail.csv"), detail)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), summary_rows)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_votes.csv"), votes)
    write_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_feature_groups.csv"), features)
    write_markdown(summary_rows, votes, features)
    print(json.dumps({"summary": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
