#!/usr/bin/env python3
"""Decompose E088 selector headroom into rate and distortion terms."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from analyze_beta005_decoder_safe_selector import (
    DECODER_SAFE_FEATURES,
    SEEDS,
    f as strict_float,
    read_csv,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "e091_selector_gain_decomposition"

BETA_HOLDOUT = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.csv"
TRANSFER_BETA = ANALYSIS / "beta005_transfer_openimages_start8192_n4096.csv"
TRANSFER_PREVIOUS = ANALYSIS / "local_cap080_rho1_transfer8192_checkpoint_sweep.csv"
E088_TEACHER = ANALYSIS / "e088_decoder_safe_selector_teacher_labels_transfer8192.json"

LAMBDA = 0.0035
MSE_SCALE = 255.0 * 255.0


def finite_flag(row: dict[str, str]) -> bool:
    return str(row.get("has_nonfinite", "0")).lower() not in {"1", "true", "yes"}


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    if key not in row or row[key] == "":
        return default
    try:
        value = float(row[key])
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def pearson(a: Iterable[float], b: Iterable[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(a, b, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_csv(path) if finite_flag(row)}


def load_beta_holdout_step500() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["seed"], row["path"]): row
        for row in read_csv(BETA_HOLDOUT)
        if row.get("step") == "500" and finite_flag(row)
    }


def load_e088_model() -> dict[str, object]:
    payload = json.loads(E088_TEACHER.read_text(encoding="utf-8"))
    model = payload["model"]
    return {
        "features": list(model["features"]),
        "mean": np.asarray(model["mean"], dtype=np.float64),
        "scale": np.asarray(model["scale"], dtype=np.float64),
        "weight": np.asarray(model["weight"], dtype=np.float64),
        "bias": float(model["bias"]),
        "threshold": float(payload["threshold"]),
    }


def score_e088(row: dict[str, str], model: dict[str, object]) -> float:
    features = model["features"]
    loc = model["mean"]
    scale = model["scale"]
    weight = model["weight"]
    x = np.asarray([f(row, feature) for feature in features], dtype=np.float64)
    z = float(((x - loc) / scale) @ weight + float(model["bias"]))
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def make_pair(
    *,
    split: str,
    seed: str,
    path: str,
    beta: dict[str, str],
    previous: dict[str, str],
    hcs_rd: float,
    e088_score: float,
    e088_threshold: float,
) -> dict[str, float | str | bool]:
    beta_bpp = f(beta, "bpp")
    previous_bpp = f(previous, "bpp")
    beta_mse = f(beta, "mse")
    previous_mse = f(previous, "mse")
    beta_dist = LAMBDA * MSE_SCALE * beta_mse
    previous_dist = LAMBDA * MSE_SCALE * previous_mse
    beta_rd = f(beta, "rd_score")
    previous_rd = f(previous, "rd_score")
    rd_delta = previous_rd - beta_rd
    bpp_delta = previous_bpp - beta_bpp
    dist_delta = previous_dist - beta_dist
    selected = e088_score >= e088_threshold
    return {
        "split": split,
        "seed": seed,
        "path": path,
        "hcs_rd": hcs_rd,
        "beta_rd": beta_rd,
        "previous_rd": previous_rd,
        "beta_bpp": beta_bpp,
        "previous_bpp": previous_bpp,
        "beta_mse": beta_mse,
        "previous_mse": previous_mse,
        "beta_distortion_term": beta_dist,
        "previous_distortion_term": previous_dist,
        "rd_delta_previous_minus_beta": rd_delta,
        "bpp_delta_previous_minus_beta": bpp_delta,
        "distortion_delta_previous_minus_beta": dist_delta,
        "rd_recomposition_residual": rd_delta - bpp_delta - dist_delta,
        "previous_local_wins": previous_rd < beta_rd,
        "e088_score": e088_score,
        "e088_selected_previous_local": selected,
        "e088_threshold": e088_threshold,
        "mixed_rd": previous_rd if selected else beta_rd,
        "mixed_delta_vs_beta": rd_delta if selected else 0.0,
        "mixed_bpp_delta_vs_beta": bpp_delta if selected else 0.0,
        "mixed_distortion_delta_vs_beta": dist_delta if selected else 0.0,
    }


def load_holdout_pairs(model: dict[str, object]) -> list[dict[str, float | str | bool]]:
    beta_rows = load_beta_holdout_step500()
    rows: list[dict[str, float | str | bool]] = []
    for seed, cfg in SEEDS.items():
        previous = by_path(ANALYSIS / cfg["previous_local"])
        hcs_old = by_path(ANALYSIS / cfg["hcs_old"])
        paths = sorted(set(previous) & set(hcs_old) & {path for s, path in beta_rows if s == seed})
        if len(paths) != 4096:
            raise RuntimeError(f"holdout seed {seed}: expected 4096 aligned rows, got {len(paths)}")
        for path in paths:
            beta = beta_rows[(seed, path)]
            rows.append(
                make_pair(
                    split="holdout4096",
                    seed=seed,
                    path=path,
                    beta=beta,
                    previous=previous[path],
                    hcs_rd=strict_float(hcs_old[path], "HCS_rd_score"),
                    e088_score=score_e088(beta, model),
                    e088_threshold=float(model["threshold"]),
                )
            )
    return rows


def load_transfer_pairs(model: dict[str, object]) -> list[dict[str, float | str | bool]]:
    beta_rows = [
        row
        for row in read_csv(TRANSFER_BETA)
        if row.get("method") == "beta005 guard" and finite_flag(row)
    ]
    previous_by_key = {
        (row["seed"], row["path"]): row
        for row in read_csv(TRANSFER_PREVIOUS)
        if finite_flag(row)
    }
    rows: list[dict[str, float | str | bool]] = []
    for beta in beta_rows:
        key = (beta["seed"], beta["path"])
        previous = previous_by_key.get(key)
        if previous is None:
            continue
        rows.append(
            make_pair(
                split="transfer8192",
                seed=key[0],
                path=key[1],
                beta=beta,
                previous=previous,
                hcs_rd=float("nan"),
                e088_score=score_e088(beta, model),
                e088_threshold=float(model["threshold"]),
            )
        )
    if len(rows) != 12288:
        raise RuntimeError(f"transfer: expected 12288 aligned rows, got {len(rows)}")
    return rows


def summary_for_mask(
    rows: list[dict[str, float | str | bool]],
    label: str,
    mask: Iterable[bool],
) -> dict[str, float | str]:
    selected = [row for row, flag in zip(rows, mask, strict=True) if flag]
    if not selected:
        return {"label": label, "rows": 0.0, "fraction": 0.0}
    rd_delta = mean(float(row["rd_delta_previous_minus_beta"]) for row in selected)
    bpp_delta = mean(float(row["bpp_delta_previous_minus_beta"]) for row in selected)
    dist_delta = mean(float(row["distortion_delta_previous_minus_beta"]) for row in selected)
    gain = -rd_delta
    return {
        "label": label,
        "rows": float(len(selected)),
        "fraction": len(selected) / len(rows),
        "beta_rd": mean(float(row["beta_rd"]) for row in selected),
        "previous_rd": mean(float(row["previous_rd"]) for row in selected),
        "rd_delta_previous_minus_beta": rd_delta,
        "bpp_delta_previous_minus_beta": bpp_delta,
        "distortion_delta_previous_minus_beta": dist_delta,
        "rd_recomposition_residual": mean(float(row["rd_recomposition_residual"]) for row in selected),
        "previous_local_win_fraction": mean(1.0 if row["previous_local_wins"] else 0.0 for row in selected),
        "e088_selected_fraction": mean(1.0 if row["e088_selected_previous_local"] else 0.0 for row in selected),
        "e088_score_mean": mean(float(row["e088_score"]) for row in selected),
        "rate_share_of_gain": (-bpp_delta / gain) if abs(gain) > 1e-12 else float("nan"),
        "distortion_share_of_gain": (-dist_delta / gain) if abs(gain) > 1e-12 else float("nan"),
    }


def summarize_split(rows: list[dict[str, float | str | bool]]) -> dict[str, object]:
    selected = [bool(row["e088_selected_previous_local"]) for row in rows]
    wins = [bool(row["previous_local_wins"]) for row in rows]
    true_selected = [s and w for s, w in zip(selected, wins, strict=True)]
    false_selected = [s and not w for s, w in zip(selected, wins, strict=True)]
    missed_wins = [(not s) and w for s, w in zip(selected, wins, strict=True)]
    keep_correct = [(not s) and (not w) for s, w in zip(selected, wins, strict=True)]
    beta_rd = mean(float(row["beta_rd"]) for row in rows)
    previous_rd = mean(float(row["previous_rd"]) for row in rows)
    mixed_rd = mean(float(row["mixed_rd"]) for row in rows)
    oracle_rd = mean(min(float(row["beta_rd"]), float(row["previous_rd"])) for row in rows)
    base = {
        "rows": float(len(rows)),
        "beta005_rd": beta_rd,
        "previous_local_rd": previous_rd,
        "previous_local_delta_vs_beta005": previous_rd - beta_rd,
        "oracle_rd": oracle_rd,
        "oracle_delta_vs_beta005": oracle_rd - beta_rd,
        "e088_mixed_rd": mixed_rd,
        "e088_mixed_delta_vs_beta005": mixed_rd - beta_rd,
        "e088_selected_fraction": mean(1.0 if flag else 0.0 for flag in selected),
        "previous_local_win_fraction": mean(1.0 if flag else 0.0 for flag in wins),
        "mixed_bpp_delta_vs_beta005": mean(float(row["mixed_bpp_delta_vs_beta"]) for row in rows),
        "mixed_distortion_delta_vs_beta005": mean(float(row["mixed_distortion_delta_vs_beta"]) for row in rows),
    }
    groups = [
        summary_for_mask(rows, "all rows", [True] * len(rows)),
        summary_for_mask(rows, "previous-local wins", wins),
        summary_for_mask(rows, "E088 selected", selected),
        summary_for_mask(rows, "E088 true positives", true_selected),
        summary_for_mask(rows, "E088 false positives", false_selected),
        summary_for_mask(rows, "E088 missed winners", missed_wins),
        summary_for_mask(rows, "E088 correct keeps", keep_correct),
    ]
    correlations = {
        "score_vs_margin_beta_minus_previous": pearson(
            [float(row["e088_score"]) for row in rows],
            [-float(row["rd_delta_previous_minus_beta"]) for row in rows],
        ),
        "score_vs_rate_gain": pearson(
            [float(row["e088_score"]) for row in rows],
            [-float(row["bpp_delta_previous_minus_beta"]) for row in rows],
        ),
        "score_vs_distortion_gain": pearson(
            [float(row["e088_score"]) for row in rows],
            [-float(row["distortion_delta_previous_minus_beta"]) for row in rows],
        ),
    }
    for feature in DECODER_SAFE_FEATURES:
        correlations[f"score_vs_{feature}"] = pearson(
            [float(row["e088_score"]) for row in rows],
            [float(row.get(feature, float("nan"))) for row in rows],
        )
    return {"base": base, "groups": groups, "correlations": correlations}


def add_beta_features(rows: list[dict[str, float | str | bool]], beta_feature_rows: dict[tuple[str, str], dict[str, str]]) -> None:
    for row in rows:
        beta = beta_feature_rows.get((str(row["seed"]), str(row["path"])))
        if beta is None:
            continue
        for feature in DECODER_SAFE_FEATURES:
            row[feature] = f(beta, feature)


def hcs_quartiles(rows: list[dict[str, float | str | bool]]) -> list[dict[str, object]]:
    valid = [row for row in rows if math.isfinite(float(row["hcs_rd"]))]
    if not valid:
        return []
    ordered = sorted(valid, key=lambda row: float(row["hcs_rd"]))
    chunks = np.array_split(np.arange(len(ordered)), 4)
    out = []
    for index, chunk in enumerate(chunks, start=1):
        subset = [ordered[int(i)] for i in chunk]
        out.append(
            {
                "quartile": float(index),
                "summary": summarize_split(subset)["base"],
                "selected_group": summary_for_mask(
                    subset,
                    f"Q{index} E088 selected",
                    [bool(row["e088_selected_previous_local"]) for row in subset],
                ),
            }
        )
    return out


def write_csv(rows: list[dict[str, float | str | bool]]) -> None:
    fieldnames = [
        "split",
        "seed",
        "path",
        "beta_rd",
        "previous_rd",
        "rd_delta_previous_minus_beta",
        "bpp_delta_previous_minus_beta",
        "distortion_delta_previous_minus_beta",
        "rd_recomposition_residual",
        "previous_local_wins",
        "e088_score",
        "e088_selected_previous_local",
        "mixed_rd",
        "mixed_delta_vs_beta",
    ]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def group_table(groups: list[dict[str, float | str]]) -> list[str]:
    lines = [
        "| group | rows | frac | prev win frac | E088 sel frac | beta RD | previous RD | ΔRD prev-beta | Δbpp | Δdist | rate/gain | dist/gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['label']} | {int(float(row['rows']))} | {fmt(float(row['fraction']))} | "
            f"{fmt(float(row.get('previous_local_win_fraction', float('nan'))))} | "
            f"{fmt(float(row.get('e088_selected_fraction', float('nan'))))} | "
            f"{fmt(float(row.get('beta_rd', float('nan'))))} | {fmt(float(row.get('previous_rd', float('nan'))))} | "
            f"{fmt(float(row.get('rd_delta_previous_minus_beta', float('nan'))), signed=True)} | "
            f"{fmt(float(row.get('bpp_delta_previous_minus_beta', float('nan'))), signed=True)} | "
            f"{fmt(float(row.get('distortion_delta_previous_minus_beta', float('nan'))), signed=True)} | "
            f"{fmt(float(row.get('rate_share_of_gain', float('nan'))), signed=True)} | "
            f"{fmt(float(row.get('distortion_share_of_gain', float('nan'))), signed=True)} |"
        )
    return lines


def base_table(label: str, base: dict[str, float]) -> list[str]:
    return [
        f"### {label}",
        "",
        "| item | value |",
        "|---|---:|",
        f"| rows | {int(base['rows'])} |",
        f"| beta005 RD | {fmt(base['beta005_rd'])} |",
        f"| previous-local RD | {fmt(base['previous_local_rd'])} |",
        f"| previous-local vs beta005 | {fmt(base['previous_local_delta_vs_beta005'], signed=True)} |",
        f"| oracle RD | {fmt(base['oracle_rd'])} |",
        f"| oracle vs beta005 | {fmt(base['oracle_delta_vs_beta005'], signed=True)} |",
        f"| E088 mixed RD | {fmt(base['e088_mixed_rd'])} |",
        f"| E088 mixed vs beta005 | {fmt(base['e088_mixed_delta_vs_beta005'], signed=True)} |",
        f"| E088 selected fraction | {fmt(base['e088_selected_fraction'])} |",
        f"| previous-local win fraction | {fmt(base['previous_local_win_fraction'])} |",
        f"| E088 mixed Δbpp | {fmt(base['mixed_bpp_delta_vs_beta005'], signed=True)} |",
        f"| E088 mixed Δdistortion | {fmt(base['mixed_distortion_delta_vs_beta005'], signed=True)} |",
    ]


def write_markdown(result: dict[str, object]) -> None:
    holdout = result["holdout4096"]
    transfer = result["transfer8192"]
    lines = [
        "# E091 Selector Gain Decomposition",
        "",
        "This audit decomposes the beta005-vs-previous-local headroom used by the E088 selector into rate (`bpp`) and distortion (`lambda * 255^2 * mse`) terms. It is a design audit for the next single-checkpoint objective, not a new proposed-method result.",
        "",
    ]
    lines.extend(base_table("Transfer8192 teacher split", transfer["base"]))
    lines.extend(["", "#### Transfer groups", ""])
    lines.extend(group_table(transfer["groups"]))
    lines.extend(["", "#### Transfer correlations", ""])
    lines.extend(
        [
            "| correlation | value |",
            "|---|---:|",
            f"| E088 score vs RD gain | {fmt(transfer['correlations']['score_vs_margin_beta_minus_previous'], signed=True)} |",
            f"| E088 score vs rate gain | {fmt(transfer['correlations']['score_vs_rate_gain'], signed=True)} |",
            f"| E088 score vs distortion gain | {fmt(transfer['correlations']['score_vs_distortion_gain'], signed=True)} |",
        ]
    )
    lines.extend(["", ""])
    lines.extend(base_table("Holdout4096 paper-facing split", holdout["base"]))
    lines.extend(["", "#### Holdout groups", ""])
    lines.extend(group_table(holdout["groups"]))
    lines.extend(["", "#### Holdout correlations", ""])
    lines.extend(
        [
            "| correlation | value |",
            "|---|---:|",
            f"| E088 score vs RD gain | {fmt(holdout['correlations']['score_vs_margin_beta_minus_previous'], signed=True)} |",
            f"| E088 score vs rate gain | {fmt(holdout['correlations']['score_vs_rate_gain'], signed=True)} |",
            f"| E088 score vs distortion gain | {fmt(holdout['correlations']['score_vs_distortion_gain'], signed=True)} |",
        ]
    )
    lines.extend(["", "#### Holdout HCS quartiles", ""])
    lines.extend(
        [
            "| quartile | E088 mixed vs beta005 | selected frac | prev win frac | selected ΔRD | selected Δbpp | selected Δdist |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["holdout_hcs_quartiles"]:
        base = item["summary"]
        selected = item["selected_group"]
        lines.append(
            f"| Q{int(item['quartile'])} | {fmt(base['e088_mixed_delta_vs_beta005'], signed=True)} | "
            f"{fmt(base['e088_selected_fraction'])} | {fmt(base['previous_local_win_fraction'])} | "
            f"{fmt(float(selected.get('rd_delta_previous_minus_beta', float('nan'))), signed=True)} | "
            f"{fmt(float(selected.get('bpp_delta_previous_minus_beta', float('nan'))), signed=True)} | "
            f"{fmt(float(selected.get('distortion_delta_previous_minus_beta', float('nan'))), signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "E088's selected-image gains are decomposed here before adding another GPU branch. If the selected gains are mostly distortion-side, a y-hat-preserving distortion/ranking objective can be a useful minimal next step. If a large part of the selected gain is rate-side, the next implementation needs per-image or local index-rate supervision rather than only an image distortion margin.",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    model = load_e088_model()
    holdout = load_holdout_pairs(model)
    transfer = load_transfer_pairs(model)
    add_beta_features(holdout, load_beta_holdout_step500())
    transfer_beta_features = {
        (row["seed"], row["path"]): row
        for row in read_csv(TRANSFER_BETA)
        if row.get("method") == "beta005 guard" and finite_flag(row)
    }
    add_beta_features(transfer, transfer_beta_features)
    result = {
        "lambda": LAMBDA,
        "mse_scale": MSE_SCALE,
        "transfer8192": summarize_split(transfer),
        "holdout4096": summarize_split(holdout),
        "holdout_hcs_quartiles": hcs_quartiles(holdout),
        "artifacts": {
            "transfer_beta": str(TRANSFER_BETA.relative_to(ROOT)),
            "transfer_previous": str(TRANSFER_PREVIOUS.relative_to(ROOT)),
            "holdout_beta": str(BETA_HOLDOUT.relative_to(ROOT)),
            "e088_teacher": str(E088_TEACHER.relative_to(ROOT)),
        },
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(transfer + holdout)
    write_markdown(result)
    print(f"wrote {OUT.with_suffix('.json')}")
    print(f"wrote {OUT.with_suffix('.md')}")
    print(f"wrote {OUT.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
