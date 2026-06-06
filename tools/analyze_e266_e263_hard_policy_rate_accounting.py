#!/usr/bin/env python3
"""E266 hard-policy audit for E263 GLC fallback-gate rows.

E264 showed that diagnostic soft-gate bpp is not final paper accounting.  This
audit asks a narrower question: if we keep the E263 soft-gated reconstruction
but select rows with a hard sparse policy and charge selected rows the full
branch bpp, can simple pre-outcome diagnostics recover enough useful rows?

This is still an audit, not the intended final codec.  The intended final path
may instead use progressive/entropy-coded branch bits whose paid rate tracks the
gate.  E266 tells us whether hard selection alone is promising enough to pursue
first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_smoke_t1_e1_s2.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak2_t2_e2_s4.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4held_t4_e4_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8_t8_e8_s8.csv",
    ROOT / "experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8held_t8_e8_s8.csv",
]
PREDICTORS = [
    "gate_mean",
    "active_mse_ratio",
    "active_rvq_mse",
    "active_scalar_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments/analysis/e266_e263_hard_policy_rate_accounting")
    return p.parse_args()


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, "")
        if val == "":
            return default
        out = float(val)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def dataset_from_path(path: Path) -> str:
    name = path.name
    if "clicpro8held" in name:
        return "clicpro8_held"
    if "clicpro8" in name:
        return "clicpro8_first"
    if "kodak4held" in name:
        return "kodak4_held"
    if "kodak4" in name:
        return "kodak4_first"
    if "kodak2" in name:
        return "kodak2_first"
    if "smoke" in name:
        return "kodak1_smoke"
    return path.stem


def index_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    label = str(row["label"])
    phase = "trained" if label.startswith("trained_") else "init"
    return (str(row["dataset"]), phase, str(row["image"]), str(row.get("q_index", "0")))


def read_source(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_path in paths:
        path = input_path if input_path.is_absolute() else ROOT / input_path
        path = path.resolve()
        if not path.exists():
            continue
        dataset = dataset_from_path(path)
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                item = dict(row)
                item["dataset"] = dataset
                try:
                    item["source_csv"] = str(path.relative_to(ROOT))
                except ValueError:
                    item["source_csv"] = str(path)
                rows.append(item)
    return rows


def build_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_on_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in source_rows:
        if str(row["label"]).endswith("_all_on"):
            all_on_by_key[index_key(row)] = row

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        label = str(row["label"])
        if not label.endswith("_soft_gate"):
            continue
        key = index_key(row)
        all_on = all_on_by_key.get(key)
        if all_on is None:
            continue
        no_bpp_score = f(row, "delta_dists") + 3.0 * f(row, "delta_lpips")
        diag_score = f(row, "score")
        diag_dbpp = f(row, "delta_bpp")
        full_dbpp = f(all_on, "delta_bpp")
        full_score = no_bpp_score + full_dbpp
        out: dict[str, Any] = {
            "dataset": key[0],
            "phase": key[1],
            "image": key[2],
            "q_index": key[3],
            "label": label,
            "score_diag": diag_score,
            "score_no_bpp": no_bpp_score,
            "score_full_branch_bpp": full_score,
            "diagnostic_dbpp": diag_dbpp,
            "full_branch_dbpp": full_dbpp,
            "full_win": int(full_score < 0.0),
            "source_csv": row["source_csv"],
        }
        for key_name in PREDICTORS:
            out[key_name] = f(row, key_name)
        rows.append(out)
    return rows


def summarize_policy(rows: list[dict[str, Any]], selected: list[bool], name: str) -> dict[str, Any]:
    scores = [float(row["score_full_branch_bpp"]) if use else 0.0 for row, use in zip(rows, selected)]
    selected_scores = [float(row["score_full_branch_bpp"]) for row, use in zip(rows, selected) if use]
    return {
        "policy": name,
        "rows": len(rows),
        "selected": int(sum(selected)),
        "selected_frac": sum(selected) / len(rows) if rows else 0.0,
        "mean_score": mean(scores),
        "selected_mean_score": mean(selected_scores),
        "selected_win_rate": (sum(v < 0.0 for v in selected_scores) / len(selected_scores)) if selected_scores else 0.0,
        "diag_mean_score": mean([float(row["score_diag"]) for row in rows]),
        "oracle_mean_score": mean([min(float(row["score_full_branch_bpp"]), 0.0) for row in rows]),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    vals = sorted({v for v in values if math.isfinite(v)})
    if not vals:
        return [0.0]
    out = [vals[0] - 1e-9, vals[-1] + 1e-9]
    out.extend(vals)
    out.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(out))


def select(rows: list[dict[str, Any]], feature: str, direction: str, threshold: float) -> list[bool]:
    if direction == "<=":
        return [float(row[feature]) <= threshold for row in rows]
    return [float(row[feature]) >= threshold for row in rows]


def fit_threshold(train: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in PREDICTORS:
        values = [float(row[feature]) for row in train]
        for threshold in candidate_thresholds(values):
            for direction in ["<=", ">="]:
                selected = select(train, feature, direction, threshold)
                summary = summarize_policy(train, selected, "train")
                # Tie-break toward fewer selected rows, then interpretable gate feature.
                key = (summary["mean_score"], summary["selected"], 0 if feature == "gate_mean" else 1)
                if best is None or key < best["key"]:
                    best = {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "train_summary": summary,
                        "key": key,
                    }
    assert best is not None
    best.pop("key")
    return best


def protocol_rows(rows: list[dict[str, Any]], protocol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if protocol == "all_resub":
        return rows, rows
    if protocol == "trained_resub":
        subset = [r for r in rows if r["phase"] == "trained"]
        return subset, subset
    if protocol == "first_to_held_trained":
        train = [r for r in rows if r["phase"] == "trained" and r["dataset"].endswith("_first")]
        eval_rows = [r for r in rows if r["phase"] == "trained" and r["dataset"].endswith("_held")]
        return train, eval_rows
    if protocol == "kodak_to_clic_trained":
        train = [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("kodak")]
        eval_rows = [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("clic")]
        return train, eval_rows
    if protocol == "clic_to_kodak_trained":
        train = [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("clic")]
        eval_rows = [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("kodak")]
        return train, eval_rows
    raise ValueError(protocol)


def run_protocol(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    train, eval_rows = protocol_rows(rows, protocol)
    if not train or not eval_rows:
        return {"protocol": protocol, "train_rows": len(train), "eval_rows": len(eval_rows), "skipped": True}
    fit = fit_threshold(train)
    eval_selected = select(eval_rows, fit["feature"], fit["direction"], fit["threshold"])
    return {
        "protocol": protocol,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "skipped": False,
        "feature": fit["feature"],
        "direction": fit["direction"],
        "threshold": fit["threshold"],
        "train": fit["train_summary"],
        "eval": summarize_policy(eval_rows, eval_selected, "eval"),
        "eval_all_on": summarize_policy(eval_rows, [True] * len(eval_rows), "eval_all_on"),
        "eval_none": summarize_policy(eval_rows, [False] * len(eval_rows), "eval_none"),
    }


def main() -> None:
    args = parse_args()
    rows = build_rows(read_source(args.inputs))
    if not rows:
        raise SystemExit("no E263 soft/all-on row pairs found")

    protocols = [
        "all_resub",
        "trained_resub",
        "first_to_held_trained",
        "kodak_to_clic_trained",
        "clic_to_kodak_trained",
    ]
    protocol_results = [run_protocol(rows, p) for p in protocols]
    groups = []
    for name, subset in [
        ("all", rows),
        ("trained", [r for r in rows if r["phase"] == "trained"]),
        ("trained_kodak", [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("kodak")]),
        ("trained_clic", [r for r in rows if r["phase"] == "trained" and r["dataset"].startswith("clic")]),
    ]:
        if subset:
            groups.append({"group": name, **summarize_policy(subset, [True] * len(subset), "all_hard"), "oracle_mean_score": mean([min(float(r["score_full_branch_bpp"]), 0.0) for r in subset])})

    out_prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    json_path.write_text(json.dumps({"groups": groups, "protocols": protocol_results, "rows": rows}, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Hard-Policy Rate-Accounting Audit",
        "",
        "Purpose: test whether the current E263 soft-gated reconstructions remain useful when selected rows are charged full branch bpp instead of diagnostic gate-scaled bpp.",
        "",
        "## Group Baselines",
        "",
        "| group | rows | selected-soft/full-bpp score | selected frac | selected win | oracle score | diagnostic soft score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        lines.append(
            f"| {row['group']} | {row['rows']} | {row['mean_score']:+.6f} | {row['selected_frac']:.3f} | "
            f"{row['selected_win_rate']:.3f} | {row['oracle_mean_score']:+.6f} | {row['diag_mean_score']:+.6f} |"
        )
    lines.extend([
        "",
        "## Threshold Protocols",
        "",
        "| protocol | feature | rule | train score | eval score | eval selected | eval selected win | eval oracle | eval all-on |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in protocol_results:
        if result.get("skipped"):
            lines.append(f"| {result['protocol']} | skipped | - | nan | nan | 0.000 | 0.000 | nan | nan |")
            continue
        lines.append(
            f"| {result['protocol']} | {result['feature']} | {result['direction']} {result['threshold']:.6g} | "
            f"{result['train']['mean_score']:+.6f} | {result['eval']['mean_score']:+.6f} | "
            f"{result['eval']['selected_frac']:.3f} | {result['eval']['selected_win_rate']:.3f} | "
            f"{result['eval']['oracle_mean_score']:+.6f} | {result['eval_all_on']['mean_score']:+.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A negative eval score means the soft-gated reconstruction remains useful even after selected rows are charged full branch bpp. This is not the same as all-on branch output; all-on remains a separate ablation.",
        "",
        "If held-out protocols stay near zero while oracle remains negative, the next implementation should prioritize lower-rate/progressive branch bits and calibrated selection rather than relying only on the original dense branch.",
        "",
        "## Artifacts",
        "",
        f"- `{display_path(csv_path)}`",
        f"- `{display_path(json_path)}`",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {display_path(md_path)}")


if __name__ == "__main__":
    main()
