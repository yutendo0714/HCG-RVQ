#!/usr/bin/env python3
"""Summarize E141 low-rate selector results without rerunning GPU eval."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ANALYSIS = Path("experiments/analysis")
PREFIX = ANALYSIS / "e141_lowrate_bias010_selector_headroom"
OUT = ANALYSIS / "e141_lowrate_bias010_selector_summary"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


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


def main() -> None:
    pairs = read_csv(PREFIX.with_suffix(".pairs.csv"))
    selectors = read_csv(PREFIX.with_suffix(".selectors.csv"))

    hcs_rd = mean([float(row["hcs_rd"]) for row in pairs])
    hcg_rd = mean([float(row["hcg_rd"]) for row in pairs])
    oracle_rd = mean([float(row["oracle_rd"]) for row in pairs])

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selectors:
        groups[(row["feature"], row["direction"])].append(row)

    selector_summary: list[dict[str, object]] = []
    for (feature, direction), rows in groups.items():
        if len(rows) != 3:
            continue
        heldout_deltas = [float(row["heldout_mixed_minus_hcs"]) for row in rows]
        train_deltas = [float(row["train_mixed_minus_hcs"]) for row in rows]
        selected = [float(row["heldout_selected_frac"]) for row in rows]
        mixed_rd = [float(row["heldout_mixed_rd"]) for row in rows]
        selector_summary.append(
            {
                "feature": feature,
                "direction": direction,
                "mean_train_mixed_minus_hcs": mean(train_deltas),
                "mean_heldout_mixed_rd": mean(mixed_rd),
                "mean_heldout_mixed_minus_hcs": mean(heldout_deltas),
                "heldout_win_count": sum(delta < 0.0 for delta in heldout_deltas),
                "mean_heldout_selected_frac": mean(selected),
                "heldout_seed1234_delta": heldout_deltas[0],
                "heldout_seed2345_delta": heldout_deltas[1],
                "heldout_seed3456_delta": heldout_deltas[2],
            }
        )
    selector_summary.sort(key=lambda row: float(row["mean_heldout_mixed_minus_hcs"]))

    best = selector_summary[0]
    result = {
        "baseline": {
            "hcs_rd": hcs_rd,
            "hcg_bias010_rd": hcg_rd,
            "hcg_minus_hcs": hcg_rd - hcs_rd,
            "oracle_rd": oracle_rd,
            "oracle_minus_hcs": oracle_rd - hcs_rd,
            "oracle_minus_hcg": oracle_rd - hcg_rd,
        },
        "best_selector": best,
        "selector_summary": selector_summary,
        "decision": (
            "Use the s_q high-threshold selector as the next low-rate diagnostic/control target. "
            "It is decoder-known/hyperprior-side, improves the leave-one-seed-out mean over fixed HCG, "
            "and nearly neutralizes the fragile seed3456 damage. It is still posthoc and must be "
            "implemented or trained as a single fixed checkpoint before becoming paper-main."
        ),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(OUT.with_suffix(".csv"), selector_summary)

    lines = [
        "# E141 Low-Rate Selector Summary",
        "",
        "This summary aggregates the leave-one-seed-out selector rows from `e141_lowrate_bias010_selector_headroom` without rerunning GPU evaluation.",
        "",
        "## Baselines",
        "",
        f"- HCS RD: `{fmt(hcs_rd)}`",
        f"- fixed HCG bias010 RD: `{fmt(hcg_rd)}` ({fmt(hcg_rd - hcs_rd, True)} vs HCS)",
        f"- per-image oracle RD: `{fmt(oracle_rd)}` ({fmt(oracle_rd - hcs_rd, True)} vs HCS)",
        "",
        "## Best Selector",
        "",
        f"- feature: `{best['feature']}`",
        f"- direction: `{best['direction']}`",
        f"- leave-one-seed-out mixed RD: `{fmt(float(best['mean_heldout_mixed_rd']))}`",
        f"- leave-one-seed-out delta vs HCS: `{fmt(float(best['mean_heldout_mixed_minus_hcs']), True)}`",
        f"- held-out wins: `{best['heldout_win_count']}/3`",
        f"- selected fraction: `{fmt(float(best['mean_heldout_selected_frac']))}`",
        f"- per-heldout deltas: seed1234 `{fmt(float(best['heldout_seed1234_delta']), True)}`, seed2345 `{fmt(float(best['heldout_seed2345_delta']), True)}`, seed3456 `{fmt(float(best['heldout_seed3456_delta']), True)}`",
        "",
        "## Top Selectors",
        "",
        "| feature | dir | mean heldout delta | wins | selected frac | seed1234 | seed2345 | seed3456 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selector_summary[:8]:
        lines.append(
            "| `{feature}` | {direction} | {mean_delta} | {wins}/3 | {selected} | {s1234} | {s2345} | {s3456} |".format(
                feature=row["feature"],
                direction=row["direction"],
                mean_delta=fmt(float(row["mean_heldout_mixed_minus_hcs"]), True),
                wins=row["heldout_win_count"],
                selected=fmt(float(row["mean_heldout_selected_frac"])),
                s1234=fmt(float(row["heldout_seed1234_delta"]), True),
                s2345=fmt(float(row["heldout_seed2345_delta"]), True),
                s3456=fmt(float(row["heldout_seed3456_delta"]), True),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(result["decision"]),
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
