#!/usr/bin/env python3
"""Summarize branch/fallback headroom against continuous suppression failures."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - analysis can still run without sizes.
    Image = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_e143_lowrate_bias010_holdout_selector as e143

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e150_branch_vs_continuous_controller_audit"
PAIRS = ANALYSIS / "e143_lowrate_bias010_holdout4096_selector.pairs.csv"
E144 = ANALYSIS / "e144_lowrate_bias010_transfer_to_holdout_controller.json"
E145 = ANALYSIS / "e145_lowrate_strength_backoff_single_checkpoint.json"
E147 = ANALYSIS / "e147_lowrate_bias010_teacher_headonly_rho20_lr005_transfer8192_fit.json"
E148 = ANALYSIS / "e148_lowrate_bias010_residualselector_transfer8192_suppress_rho100_yhatanchor50_deadzone014_fit.json"
E149 = ANALYSIS / "e149_lowrate_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_fit.json"


def parse_value(value: str) -> object:
    if value == "":
        return ""
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def q95(values: list[float]) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]


def actual_side_bit_bpp(rows: list[dict[str, object]]) -> float:
    if Image is None:
        return float("nan")
    cache: dict[str, float] = {}
    for row in rows:
        path = str(row["path"])
        if path in cache:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError:
            continue
        if width > 0 and height > 0:
            cache[path] = 1.0 / float(width * height)
    return mean([cache.get(str(row["path"]), float("nan")) for row in rows])


def summarize_rd(rows: list[dict[str, object]], rd_key: str, ref_hcs: float, ref_hcg: float) -> dict[str, object]:
    vals = [float(row[rd_key]) for row in rows]
    rd = mean(vals)
    deltas = [float(row[rd_key]) - float(row["hcs_rd"]) for row in rows]
    return {
        "rd": rd,
        "minus_hcs": rd - ref_hcs,
        "minus_fixed_hcg": rd - ref_hcg,
        "win_hcs_count": sum(delta < 0.0 for delta in deltas),
        "q95_damage_vs_hcs": q95([max(0.0, delta) for delta in deltas]),
    }


def branch_rows(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> list[dict[str, object]]:
    out = []
    for row in rows:
        value = float(row[feature])
        use_hcg = math.isfinite(value) and (value <= threshold if direction == "low" else value >= threshold)
        mixed = float(row["hcg_rd"]) if use_hcg else float(row["hcs_rd"])
        out.append({**row, "branch_rd": mixed, "branch_use_hcg": float(use_hcg)})
    return out


def method_row(
    *,
    method: str,
    split: str,
    rd: float,
    hcs_rd: float,
    fixed_hcg_rd: float,
    win_hcs_count: int | float = float("nan"),
    q95_damage_vs_hcs: float = float("nan"),
    selected_frac: float = float("nan"),
    nonfinite_rows: int | float = 0,
    note: str,
) -> dict[str, object]:
    return {
        "method": method,
        "split": split,
        "rd": rd,
        "minus_hcs": rd - hcs_rd,
        "minus_fixed_hcg": rd - fixed_hcg_rd,
        "win_hcs_count": win_hcs_count,
        "q95_damage_vs_hcs": q95_damage_vs_hcs,
        "selected_frac": selected_frac,
        "nonfinite_rows": nonfinite_rows,
        "note": note,
    }


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def main() -> None:
    rows = load_csv(PAIRS)
    e144 = load_json(E144)
    hcs_rd = float(e144["holdout_summary"]["hcs_rd"])
    fixed_hcg_rd = float(e144["holdout_summary"]["hcg_rd"])
    side_bit_actual = actual_side_bit_bpp(rows)
    side_bit_patch256 = 1.0 / float(256 * 256)

    oracle_stats = summarize_rd(rows, "oracle_rd", hcs_rd, fixed_hcg_rd)
    hcg_deltas = [float(row["hcg_minus_hcs"]) for row in rows]
    methods: list[dict[str, object]] = [
        method_row(
            method="HCS",
            split="holdout4096",
            rd=hcs_rd,
            hcs_rd=hcs_rd,
            fixed_hcg_rd=fixed_hcg_rd,
            win_hcs_count=0,
            q95_damage_vs_hcs=0.0,
            selected_frac=0.0,
            note="reference fallback state",
        ),
        method_row(
            method="fixed HCG bias010",
            split="holdout4096",
            rd=fixed_hcg_rd,
            hcs_rd=hcs_rd,
            fixed_hcg_rd=fixed_hcg_rd,
            win_hcs_count=sum(delta < 0.0 for delta in hcg_deltas),
            q95_damage_vs_hcs=q95([max(0.0, delta) for delta in hcg_deltas]),
            selected_frac=1.0,
            note="raw geometry state",
        ),
        method_row(
            method="per-image oracle HCS/HCG",
            split="holdout4096",
            rd=float(oracle_stats["rd"]),
            hcs_rd=hcs_rd,
            fixed_hcg_rd=fixed_hcg_rd,
            win_hcs_count=int(oracle_stats["win_hcs_count"]),
            q95_damage_vs_hcs=float(oracle_stats["q95_damage_vs_hcs"]),
            selected_frac=float(e144["holdout_summary"]["oracle_hcg_count"]) / float(e144["holdout_summary"]["num_images"]),
            note="upper bound, uses holdout labels",
        ),
    ]

    preset = e144["preset_householder_strength_low"]
    branched = branch_rows(
        rows,
        str(preset["feature"]),
        str(preset["direction"]),
        float(preset["threshold"]),
    )
    branch_stats = summarize_rd(branched, "branch_rd", hcs_rd, fixed_hcg_rd)
    branch_rd = float(branch_stats["rd"])
    methods.extend(
        [
            method_row(
                method="transfer-trained strength branch",
                split="holdout4096",
                rd=branch_rd,
                hcs_rd=hcs_rd,
                fixed_hcg_rd=fixed_hcg_rd,
                win_hcs_count=int(branch_stats["win_hcs_count"]),
                q95_damage_vs_hcs=float(branch_stats["q95_damage_vs_hcs"]),
                selected_frac=float(preset["holdout_selected_frac"]),
                note="protocol-clean diagnostic branch, no side-bit penalty",
            ),
            method_row(
                method="strength branch + actual 1-bit signal",
                split="holdout4096",
                rd=branch_rd + side_bit_actual,
                hcs_rd=hcs_rd,
                fixed_hcg_rd=fixed_hcg_rd,
                win_hcs_count=int(branch_stats["win_hcs_count"]),
                q95_damage_vs_hcs=float(branch_stats["q95_damage_vs_hcs"]) + side_bit_actual,
                selected_frac=float(preset["holdout_selected_frac"]),
                note="same branch with one image-level signaled bit using actual image sizes",
            ),
            method_row(
                method="strength branch + 256x256 1-bit signal",
                split="holdout4096",
                rd=branch_rd + side_bit_patch256,
                hcs_rd=hcs_rd,
                fixed_hcg_rd=fixed_hcg_rd,
                win_hcs_count=int(branch_stats["win_hcs_count"]),
                q95_damage_vs_hcs=float(branch_stats["q95_damage_vs_hcs"]) + side_bit_patch256,
                selected_frac=float(preset["holdout_selected_frac"]),
                note="conservative patch-size side-bit penalty",
            ),
        ]
    )

    e145 = load_json(E145)["summary"]
    methods.append(
        method_row(
            method="E145 strength backoff",
            split="holdout4096",
            rd=float(e145["backoff_rd"]),
            hcs_rd=hcs_rd,
            fixed_hcg_rd=fixed_hcg_rd,
            win_hcs_count=int(e145["backoff_win_hcs_count"]),
            q95_damage_vs_hcs=float(e145["q95_backoff_damage_vs_hcs"]),
            selected_frac=float("nan"),
            nonfinite_rows=int(e145["nonfinite_rows"]),
            note="single-checkpoint continuous suppression",
        )
    )

    transfer_hcs = float(e144["transfer_per_seed"][2]["hcs_rd"])
    transfer_hcg = float(e144["transfer_per_seed"][2]["hcg_rd"])
    for path, metric_key, delta_key, q95_key, label in (
        (E147, "teacher_head_rd", "teacher_head_minus_hcs", "q95_teacher_head_damage_vs_hcs", "E147 strong reliability head"),
        (E148, "selector_rd", "selector_minus_hcs", "q95_selector_damage_vs_hcs", "E148 conservative residual selector"),
        (E149, "selector_rd", "selector_minus_hcs", "q95_selector_damage_vs_hcs", "E149 strong residual selector"),
    ):
        payload = load_json(path)
        for row in payload["by_step"]:
            rd = float(row[metric_key])
            methods.append(
                method_row(
                    method=f"{label} step{int(row['step'])}",
                    split="transfer8192 seed3456",
                    rd=rd,
                    hcs_rd=transfer_hcs,
                    fixed_hcg_rd=transfer_hcg,
                    win_hcs_count=int(row.get("teacher_head_win_hcs_count", row.get("selector_win_hcs_count", -1))),
                    q95_damage_vs_hcs=float(row[q95_key]),
                    selected_frac=float(row.get("pred_keep_fraction_rel05", row.get("pred_suppress_fraction_p050", float("nan")))),
                    nonfinite_rows=int(row["nonfinite_rows"]),
                    note="continuous learned deployment stress test",
                )
            )

    branch_margin_vs_continuous = float(e145["backoff_rd"]) - branch_rd
    result = {
        "experiment": "E150 branch vs continuous controller audit",
        "hcs_rd": hcs_rd,
        "fixed_hcg_rd": fixed_hcg_rd,
        "transfer_trained_branch_rd": branch_rd,
        "transfer_trained_branch_minus_hcs": branch_rd - hcs_rd,
        "transfer_trained_branch_minus_fixed_hcg": branch_rd - fixed_hcg_rd,
        "oracle_rd": float(oracle_stats["rd"]),
        "oracle_minus_hcs": float(oracle_stats["minus_hcs"]),
        "actual_one_bit_side_bpp": side_bit_actual,
        "patch256_one_bit_side_bpp": side_bit_patch256,
        "branch_margin_vs_e145_backoff_rd": branch_margin_vs_continuous,
        "branch_keeps_fixed_hcg_gain_fraction": (
            (hcs_rd - branch_rd) / (hcs_rd - fixed_hcg_rd)
            if hcs_rd != fixed_hcg_rd
            else float("nan")
        ),
        "methods": methods,
        "decision": (
            "The next deployable method should preserve explicit HCS-like and HCG-like states. "
            "The transfer-trained branch remains strong even with a one-bit image-level signal penalty, "
            "whereas continuous suppression variants either become no-ops or badly damage RD."
        ),
    }

    write_csv(PREFIX.with_suffix(".methods.csv"), methods)
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E150 Branch vs Continuous Controller Audit",
        "",
        "## Main Numbers",
        "",
        f"- HCS RD: `{fmt(hcs_rd)}`",
        f"- fixed HCG bias010 RD: `{fmt(fixed_hcg_rd)}` ({fmt(fixed_hcg_rd - hcs_rd, True)} vs HCS)",
        f"- transfer-trained strength branch RD: `{fmt(branch_rd)}` ({fmt(branch_rd - hcs_rd, True)} vs HCS, `{fmt(branch_rd - fixed_hcg_rd, True)}` vs fixed HCG)",
        f"- per-image oracle RD: `{fmt(float(oracle_stats['rd']))}` ({fmt(float(oracle_stats['minus_hcs']), True)} vs HCS)",
        f"- actual one-bit side signal bpp: `{side_bit_actual:.9f}`",
        f"- conservative 256x256 one-bit bpp: `{side_bit_patch256:.9f}`",
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "## Method Table",
        "",
        "| method | split | RD | vs HCS | vs fixed HCG | q95 damage | selected/active frac | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in methods:
        lines.append(
            "| {method} | {split} | {rd} | {hcs} | {hcg} | {q95} | {sel} | {note} |".format(
                method=row["method"],
                split=row["split"],
                rd=fmt(float(row["rd"])),
                hcs=fmt(float(row["minus_hcs"]), True),
                hcg=fmt(float(row["minus_fixed_hcg"]), True),
                q95=fmt(float(row["q95_damage_vs_hcs"])),
                sel=fmt(float(row["selected_frac"])),
                note=row["note"],
            )
        )
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
