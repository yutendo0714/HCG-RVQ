#!/usr/bin/env python3
"""Choose transfer-split feature thresholds for E086 reliability selection."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
TEACHER_CSV = ANALYSIS / "beta005_previous_local_teacher_labels_transfer8192_margin_weighted.csv"

FEATURES = {
    "rvq_householder_delta_rms": "high",
    "rvq_householder_gate_raw": "high",
    "rvq_householder_strength": "high",
    "rvq_latent_quant_mse": "high",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return mean(values) if values else math.nan


def threshold_candidates(values: list[float]) -> list[float]:
    values = sorted(v for v in values if math.isfinite(v))
    return [values[min(len(values) - 1, int(len(values) * q / 100))] for q in range(5, 100, 5)]


def summarize_threshold(rows: list[dict[str, str]], feature: str, threshold: float, direction: str) -> dict:
    selected = [
        (f(row, feature) >= threshold if direction == "high" else f(row, feature) <= threshold)
        for row in rows
    ]
    margins = [f(row, "margin_beta005_minus_previous_local") for row in rows]
    wins = [bool(int(float(row["previous_local_wins"]))) for row in rows]
    selected_margins = [m for m, flag in zip(margins, selected) if flag]
    selected_wins = [w for w, flag in zip(wins, selected) if flag]
    captured_margin = [max(m, 0.0) if flag else 0.0 for m, flag in zip(margins, selected)]
    total_positive_margin = sum(max(m, 0.0) for m in margins)
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "selected_fraction": sum(selected) / len(selected),
        "previous_local_win_precision": sum(selected_wins) / len(selected_wins) if selected_wins else math.nan,
        "previous_local_win_recall": sum(1 for w, flag in zip(wins, selected) if w and flag) / max(1, sum(wins)),
        "selected_mean_margin_beta_minus_previous": safe_mean(selected_margins),
        "captured_positive_margin_per_row": safe_mean(captured_margin),
        "captured_positive_margin_fraction": sum(captured_margin) / total_positive_margin if total_positive_margin else math.nan,
    }


def score(record: dict) -> float:
    # Favor high-margin, not-too-broad selectors. This is for choosing an independent
    # training prior, not for claiming holdout performance.
    selected = record["selected_fraction"]
    if selected < 0.05 or selected > 0.35:
        return -math.inf
    return record["captured_positive_margin_per_row"] + 0.01 * record["previous_local_win_precision"]


def analyze_feature(rows: list[dict[str, str]], feature: str, preferred: str) -> dict:
    records = []
    for threshold in threshold_candidates([f(row, feature) for row in rows]):
        for direction in ("high", "low"):
            records.append(summarize_threshold(rows, feature, threshold, direction))
    best = max(records, key=score)
    preferred_records = [record for record in records if record["direction"] == preferred]
    best_preferred = max(preferred_records, key=score)
    q85 = sorted(f(row, feature) for row in rows)[int(len(rows) * 0.85)]
    q90 = sorted(f(row, feature) for row in rows)[int(len(rows) * 0.90)]
    return {
        "feature": feature,
        "preferred_direction": preferred,
        "best_any": best,
        "best_preferred": best_preferred,
        "q85": q85,
        "q90": q90,
        "all_records": records,
    }


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# E086 Transfer Selector Threshold Audit",
        "",
        "This audit chooses selector thresholds only from transfer8192 teacher-label rows, before holdout4096 E086 evaluation.",
        "",
        "| feature | best direction | threshold | selected | precision | recall | selected margin | captured margin/row | captured margin frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in payload["features"]:
        row = feature["best_preferred"]
        lines.append(
            f"| {feature['feature']} | {row['direction']} | {fmt(row['threshold'])} | "
            f"{fmt(row['selected_fraction'])} | {fmt(row['previous_local_win_precision'])} | "
            f"{fmt(row['previous_local_win_recall'])} | "
            f"{fmt(row['selected_mean_margin_beta_minus_previous'], signed=True)} | "
            f"{fmt(row['captured_positive_margin_per_row'])} | "
            f"{fmt(row['captured_positive_margin_fraction'])} |"
        )
    selected = payload["selected_threshold"]
    lines += [
        "",
        "## Selected E086 Prior",
        "",
        f"Use `{selected['feature']}` {selected['direction']} `{fmt(selected['threshold'])}`.",
        "",
        payload["decision"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    rows = read_csv(TEACHER_CSV)
    features = [analyze_feature(rows, feature, direction) for feature, direction in FEATURES.items()]
    delta = next(feature for feature in features if feature["feature"] == "rvq_householder_delta_rms")
    selected = delta["best_preferred"]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_csv": str(TEACHER_CSV.relative_to(ROOT)),
        "rows": len(rows),
        "positive_fraction": safe_mean([f(row, "previous_local_wins") for row in rows]),
        "features": features,
        "selected_threshold": selected,
        "decision": (
            "Use the transfer-derived delta-RMS threshold for E086 local keep-target selection. "
            "The threshold is not tuned on holdout4096; holdout remains reserved for evaluation."
        ),
    }
    out_json = ANALYSIS / "e086_transfer_selector_thresholds.json"
    out_md = ANALYSIS / "e086_transfer_selector_thresholds.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, out_md)
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "selected_threshold": selected}, indent=2))


if __name__ == "__main__":
    main()
