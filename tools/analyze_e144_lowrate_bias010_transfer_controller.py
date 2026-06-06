#!/usr/bin/env python3
"""Train a low-rate HCG reliability selector on start8192 and apply it to holdout4096."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_e143_lowrate_bias010_holdout_selector as e143

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e144_lowrate_bias010_transfer_to_holdout_controller"
TRANSFER_START_INDEX = 8192
HOLDOUT_PAIRS = ANALYSIS / "e143_lowrate_bias010_holdout4096_selector.pairs.csv"
TRANSFER_PAIRS = PREFIX.with_suffix(".transfer_start8192_pairs.csv")
TRANSFER_ALL_ROWS = PREFIX.with_suffix(".transfer_start8192_all_rows.csv")
TRANSFER_PER_SEED = PREFIX.with_suffix(".transfer_start8192_per_seed.csv")


def parse_value(value: str) -> object:
    if value == "":
        return ""
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    e143.write_csv(path, rows)


def ensure_transfer_pairs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if TRANSFER_PAIRS.exists():
        pairs = load_csv(TRANSFER_PAIRS)
        per_seed = load_csv(TRANSFER_PER_SEED) if TRANSFER_PER_SEED.exists() else []
        return pairs, per_seed

    original_start = e143.START_INDEX
    original_prefix = e143.PREFIX
    try:
        e143.START_INDEX = TRANSFER_START_INDEX
        e143.PREFIX = PREFIX
        rows, _run_summaries = e143.evaluate_runs()
    finally:
        e143.START_INDEX = original_start
        e143.PREFIX = original_prefix

    pairs = e143.align_pairs(rows)
    per_seed = [{"seed": seed, **e143.summarize([row for row in pairs if row["seed"] == seed])} for seed in e143.SEEDS]
    write_csv(TRANSFER_ALL_ROWS, rows)
    write_csv(TRANSFER_PAIRS, pairs)
    write_csv(TRANSFER_PER_SEED, per_seed)
    return pairs, per_seed


def selector_summary(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> dict[str, object]:
    mixed = []
    selected = []
    selected_deltas = []
    for row in rows:
        value = float(row[feature])
        use_hcg = math.isfinite(value) and (value <= threshold if direction == "low" else value >= threshold)
        selected.append(use_hcg)
        delta = float(row["hcg_minus_hcs"])
        if use_hcg:
            selected_deltas.append(delta)
        mixed.append(float(row["hcg_rd"]) if use_hcg else float(row["hcs_rd"]))
    hcs = e143.mean([float(row["hcs_rd"]) for row in rows])
    hcg = e143.mean([float(row["hcg_rd"]) for row in rows])
    rd = e143.mean(mixed)
    return {
        "mixed_rd": rd,
        "mixed_minus_hcs": rd - hcs,
        "mixed_minus_hcg": rd - hcg,
        "selected_count": sum(selected),
        "selected_frac": e143.mean([float(value) for value in selected]),
        "selected_win_count": sum(delta < 0.0 for delta in selected_deltas),
        "selected_mean_delta": e143.mean(selected_deltas),
        "q95_selected_damage": e143.q95([max(0.0, delta) for delta in selected_deltas]),
        "q95_mixed_damage": e143.q95([max(0.0, float(row["hcg_minus_hcs"])) for row, use in zip(rows, selected) if use]),
    }


def train_apply_rows(train: list[dict[str, object]], holdout: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for feature in e143.FEATURES:
        if any(feature not in row for row in train + holdout):
            continue
        for direction in ("low", "high"):
            train_best = e143.best_train_selector(train, feature, direction)
            if train_best is None:
                continue
            threshold = float(train_best["threshold"])
            holdout_summary = selector_summary(holdout, feature, direction, threshold)
            row = {
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "transfer_mixed_rd": train_best["mixed_rd"],
                "transfer_mixed_minus_hcs": train_best["mixed_minus_hcs"],
                "transfer_mixed_minus_hcg": train_best["mixed_minus_hcg"],
                "transfer_selected_count": train_best["selected_count"],
                "transfer_selected_frac": train_best["selected_frac"],
                "holdout_mixed_rd": holdout_summary["mixed_rd"],
                "holdout_mixed_minus_hcs": holdout_summary["mixed_minus_hcs"],
                "holdout_mixed_minus_hcg": holdout_summary["mixed_minus_hcg"],
                "holdout_selected_count": holdout_summary["selected_count"],
                "holdout_selected_frac": holdout_summary["selected_frac"],
                "holdout_selected_win_count": holdout_summary["selected_win_count"],
                "holdout_selected_mean_delta": holdout_summary["selected_mean_delta"],
                "holdout_q95_selected_damage": holdout_summary["q95_selected_damage"],
            }
            for seed in e143.SEEDS:
                seed_rows = [item for item in holdout if str(item["seed"]) == seed]
                seed_summary = selector_summary(seed_rows, feature, direction, threshold)
                row[f"holdout_seed{seed}_mixed_minus_hcs"] = seed_summary["mixed_minus_hcs"]
                row[f"holdout_seed{seed}_selected_frac"] = seed_summary["selected_frac"]
            output.append(row)
    return sorted(output, key=lambda row: float(row["holdout_mixed_minus_hcs"]))


def feature_stats(rows: list[dict[str, object]], feature: str, direction: str, threshold: float) -> list[dict[str, object]]:
    selected = []
    rejected = []
    for row in rows:
        value = float(row[feature])
        use_hcg = math.isfinite(value) and (value <= threshold if direction == "low" else value >= threshold)
        (selected if use_hcg else rejected).append(row)
    output = []
    for name, chunk in (("selected", selected), ("rejected", rejected)):
        if not chunk:
            continue
        output.append(
            {
                "group": name,
                "count": len(chunk),
                "hcs_rd": e143.mean([float(row["hcs_rd"]) for row in chunk]),
                "hcg_minus_hcs": e143.mean([float(row["hcg_minus_hcs"]) for row in chunk]),
                "hcg_win_frac": e143.mean([float(float(row["hcg_minus_hcs"]) < 0.0) for row in chunk]),
                "householder_strength": e143.mean([float(row["hcg_rvq_householder_strength"]) for row in chunk]),
                "s_q_mean": e143.mean([float(row["hcg_rvq_s_q_mean"]) for row in chunk]),
                "latent_quant_mse": e143.mean([float(row["hcg_rvq_latent_quant_mse"]) for row in chunk]),
                "dead_code_ratio": e143.mean([float(row["hcg_rvq_dead_code_ratio"]) for row in chunk]),
                "perplexity": e143.mean([float(row["hcg_rvq_perplexity"]) for row in chunk]),
            }
        )
    return output


def write_markdown(result: dict[str, object]) -> None:
    transfer = result["transfer_summary"]
    holdout = result["holdout_summary"]
    best = result["best_transfer_to_holdout"]
    preset = result["preset_householder_strength_low"]
    lines = [
        "# E144 Low-Rate Bias010 Transfer Controller",
        "",
        f"Transfer split: start_index={TRANSFER_START_INDEX}, max_images={e143.MAX_IMAGES}. Holdout split: E143 holdout4096.",
        "",
        "## Fixed HCG References",
        "",
        f"- transfer HCS RD: `{e143.fmt(float(transfer['hcs_rd']))}`",
        f"- transfer HCG RD: `{e143.fmt(float(transfer['hcg_rd']))}` ({e143.fmt(float(transfer['hcg_minus_hcs']), True)} vs HCS)",
        f"- holdout HCS RD: `{e143.fmt(float(holdout['hcs_rd']))}`",
        f"- holdout HCG RD: `{e143.fmt(float(holdout['hcg_rd']))}` ({e143.fmt(float(holdout['hcg_minus_hcs']), True)} vs HCS)",
        "",
        "## Best Transfer-Trained Controller On Holdout",
        "",
        f"- feature: `{best['feature']}` / `{best['direction']}`",
        f"- threshold: `{float(best['threshold']):.9f}`",
        f"- transfer mixed delta: `{e143.fmt(float(best['transfer_mixed_minus_hcs']), True)}`",
        f"- holdout mixed delta: `{e143.fmt(float(best['holdout_mixed_minus_hcs']), True)}`",
        f"- holdout selected frac: `{e143.fmt(float(best['holdout_selected_frac']))}`",
        f"- holdout seed deltas: seed1234 `{e143.fmt(float(best['holdout_seed1234_mixed_minus_hcs']), True)}`, seed2345 `{e143.fmt(float(best['holdout_seed2345_mixed_minus_hcs']), True)}`, seed3456 `{e143.fmt(float(best['holdout_seed3456_mixed_minus_hcs']), True)}`",
        "",
        "## Prespecified E143 Family",
        "",
        f"- feature: `hcg_rvq_householder_strength` / `low`",
        f"- threshold: `{float(preset['threshold']):.9f}`",
        f"- transfer mixed delta: `{e143.fmt(float(preset['transfer_mixed_minus_hcs']), True)}`",
        f"- holdout mixed delta: `{e143.fmt(float(preset['holdout_mixed_minus_hcs']), True)}`",
        f"- holdout selected frac: `{e143.fmt(float(preset['holdout_selected_frac']))}`",
        f"- holdout seed deltas: seed1234 `{e143.fmt(float(preset['holdout_seed1234_mixed_minus_hcs']), True)}`, seed2345 `{e143.fmt(float(preset['holdout_seed2345_mixed_minus_hcs']), True)}`, seed3456 `{e143.fmt(float(preset['holdout_seed3456_mixed_minus_hcs']), True)}`",
        "",
        "## Decision",
        "",
        str(result["decision"]),
    ]
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not HOLDOUT_PAIRS.exists():
        raise FileNotFoundError(HOLDOUT_PAIRS)
    transfer_pairs, transfer_per_seed = ensure_transfer_pairs()
    holdout_pairs = load_csv(HOLDOUT_PAIRS)
    transfer_summary = e143.summarize(transfer_pairs)
    holdout_summary = e143.summarize(holdout_pairs)
    controller_rows = train_apply_rows(transfer_pairs, holdout_pairs)
    if not controller_rows:
        raise RuntimeError("no controller rows")
    best = controller_rows[0]
    preset = next(
        row
        for row in controller_rows
        if row["feature"] == "hcg_rvq_householder_strength" and row["direction"] == "low"
    )
    best_stats = feature_stats(holdout_pairs, str(best["feature"]), str(best["direction"]), float(best["threshold"]))
    preset_stats = feature_stats(holdout_pairs, "hcg_rvq_householder_strength", "low", float(preset["threshold"]))
    decision = (
        "Treat transfer-trained selectors as protocol-clean diagnostics, not final codec rows. "
        "If the prespecified householder-strength controller keeps a holdout gain while reducing seed3456 damage, "
        "convert it into a single decoder-reproducible reliability gate; otherwise keep fixed bias010 as the stronger "
        "method lane and use controller evidence only to motivate richer reliability heads."
    )
    result = {
        "experiment": "E144 low-rate bias010 transfer-to-holdout controller",
        "transfer_start_index": TRANSFER_START_INDEX,
        "max_images": e143.MAX_IMAGES,
        "transfer_summary": transfer_summary,
        "transfer_per_seed": transfer_per_seed,
        "holdout_summary": holdout_summary,
        "controller_rows": controller_rows,
        "best_transfer_to_holdout": best,
        "preset_householder_strength_low": preset,
        "best_holdout_feature_stats": best_stats,
        "preset_holdout_feature_stats": preset_stats,
        "decision": decision,
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(PREFIX.with_suffix(".controllers.csv"), controller_rows)
    write_csv(PREFIX.with_suffix(".best_feature_stats.csv"), best_stats)
    write_csv(PREFIX.with_suffix(".preset_feature_stats.csv"), preset_stats)
    write_markdown(result)
    print(PREFIX.with_suffix(".md"))
    print(json.dumps({"best": best, "preset": preset, "transfer_summary": transfer_summary, "holdout_summary": holdout_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
