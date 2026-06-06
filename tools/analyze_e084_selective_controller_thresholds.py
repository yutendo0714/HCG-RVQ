#!/usr/bin/env python3
"""Analyze posthoc feature thresholds for selective controller use."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

REFERENCE_CSV = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv"
BETA_CSV = ANALYSIS / "beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.csv"

CONTROLLERS = {
    "E076_rel075_rho005_step500": ANALYSIS
    / "teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
    "E077_rel075_rho050_step500": ANALYSIS
    / "teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.csv",
    "E078_marginw_rho050_step500": ANALYSIS
    / "teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    "E079_rel095_marginw_lrm025_step500": ANALYSIS
    / "teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
    "E080_yhat_anchor_step500": ANALYSIS
    / "teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.csv",
}

FEATURES = {
    "hcs_rd": "high",
    "rvq_householder_delta_rms": "high",
    "rvq_householder_delta_rms_local_mean": "high",
    "rvq_householder_delta_rms_local_max": "high",
    "rvq_householder_gate_raw": "high",
    "rvq_householder_strength": "high",
    "rvq_s_q_mean": "low",
    "rvq_latent_quant_mse": "high",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else default


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return mean(values) if values else math.nan


def load_reference() -> dict[str, dict[str, float]]:
    refs = {}
    for row in read_csv(REFERENCE_CSV):
        if row["seed"] != "3456":
            continue
        refs[row["path"]] = {
            "hcs_rd": float(row["hcs_rd"]),
            "beta005_rd": float(row["variant500_rd"]),
            "previous_local_rd": float(row["previous_local_rd"]),
        }
    return refs


def threshold_candidates(values: list[float]) -> list[float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return []
    qs = [i / 100 for i in range(5, 100, 5)]
    return [values[min(len(values) - 1, int(len(values) * q))] for q in qs]


def quartile_summary(rows: list[dict[str, float]], selected: set[int]) -> dict[str, float]:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["hcs_rd"])
    out = {}
    n = len(ordered)
    for qi in range(4):
        chunk = ordered[qi * n // 4 : (qi + 1) * n // 4]
        deltas = [row["controller_delta"] if idx in selected else 0.0 for idx, row in chunk]
        out[f"Q{qi + 1}"] = safe_mean(deltas)
    return out


def best_threshold(rows: list[dict[str, float]], feature: str, preferred_direction: str) -> dict:
    best = None
    for threshold in threshold_candidates([row[feature] for row in rows]):
        for direction in ("high", "low"):
            selected = {
                idx
                for idx, row in enumerate(rows)
                if (row[feature] >= threshold if direction == "high" else row[feature] <= threshold)
            }
            if not selected:
                continue
            deltas = [row["controller_delta"] if idx in selected else 0.0 for idx, row in enumerate(rows)]
            selected_deltas = [row["controller_delta"] for idx, row in enumerate(rows) if idx in selected]
            record = {
                "feature": feature,
                "direction": direction,
                "preferred_direction": preferred_direction,
                "threshold": threshold,
                "selected_fraction": len(selected) / len(rows),
                "mean_delta_vs_beta005": safe_mean(deltas),
                "selected_mean_delta_vs_beta005": safe_mean(selected_deltas),
                "quartile_delta_vs_beta005": quartile_summary(rows, selected),
            }
            if best is None or record["mean_delta_vs_beta005"] < best["mean_delta_vs_beta005"]:
                best = record
    return best or {}


def summarize_controller(name: str, csv_path: Path, refs: dict[str, dict[str, float]], beta_rows: dict[str, dict[str, str]]) -> dict:
    rows = []
    for row in read_csv(csv_path):
        path = row["path"]
        if path not in refs or path not in beta_rows:
            continue
        beta = beta_rows[path]
        ref = refs[path]
        item = {
            "path": path,
            "hcs_rd": ref["hcs_rd"],
            "controller_delta": float(row["rd_score"]) - ref["beta005_rd"],
        }
        for feature in FEATURES:
            if feature == "hcs_rd":
                continue
            item[feature] = f(beta, feature)
        rows.append(item)

    all_delta = safe_mean([row["controller_delta"] for row in rows])
    oracle_selected = {idx for idx, row in enumerate(rows) if row["controller_delta"] < 0.0}
    feature_best = [best_threshold(rows, feature, direction) for feature, direction in FEATURES.items()]
    feature_best = [record for record in feature_best if record]
    best = min(feature_best, key=lambda record: record["mean_delta_vs_beta005"])
    deployable_best = min(
        [record for record in feature_best if record["feature"] != "hcs_rd"],
        key=lambda record: record["mean_delta_vs_beta005"],
    )
    return {
        "controller": name,
        "csv": str(csv_path.relative_to(ROOT)),
        "rows": len(rows),
        "mean_delta_vs_beta005": all_delta,
        "win_fraction_vs_beta005": len(oracle_selected) / len(rows),
        "oracle_select_delta_vs_beta005": safe_mean(
            [row["controller_delta"] if idx in oracle_selected else 0.0 for idx, row in enumerate(rows)]
        ),
        "oracle_selected_fraction": len(oracle_selected) / len(rows),
        "best_any_feature_threshold": best,
        "best_deployable_feature_threshold": deployable_best,
        "feature_thresholds": feature_best,
    }


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# E084 Selective Controller Threshold Headroom",
        "",
        "This audit asks whether the E076-E080 reliability controllers are harmful everywhere or only too broadly applied.",
        "The posthoc selector keeps beta005 for unselected images and uses a controller row only when a beta005-side feature crosses a threshold.",
        "",
        "| controller | full-use delta | oracle delta | best deployable feature | direction | threshold | selected | selected delta | mixed delta |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["controllers"]:
        best = row["best_deployable_feature_threshold"]
        lines.append(
            f"| {row['controller']} | {fmt(row['mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(row['oracle_select_delta_vs_beta005'], signed=True)} | {best['feature']} | "
            f"{best['direction']} | {fmt(best['threshold'])} | {fmt(best['selected_fraction'])} | "
            f"{fmt(best['selected_mean_delta_vs_beta005'], signed=True)} | "
            f"{fmt(best['mean_delta_vs_beta005'], signed=True)} |"
        )
    lines += [
        "",
        "## Best Deployable Threshold Quartiles",
        "",
        "| controller | feature | Q1 easy | Q2 | Q3 | Q4 hard |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["controllers"]:
        best = row["best_deployable_feature_threshold"]
        q = best["quartile_delta_vs_beta005"]
        lines.append(
            f"| {row['controller']} | {best['feature']} {best['direction']} | "
            f"{fmt(q['Q1'], signed=True)} | {fmt(q['Q2'], signed=True)} | "
            f"{fmt(q['Q3'], signed=True)} | {fmt(q['Q4'], signed=True)} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Next Action",
        "",
        payload["next_action"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    refs = load_reference()
    beta_rows = {row["path"]: row for row in read_csv(BETA_CSV)}
    summaries = [
        summarize_controller(name, path, refs, beta_rows)
        for name, path in CONTROLLERS.items()
    ]
    best = min(
        (row["best_deployable_feature_threshold"] | {"controller": row["controller"]} for row in summaries),
        key=lambda record: record["mean_delta_vs_beta005"],
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Quantify whether beta005-side features can select reliability-controller outputs only on hard/risky images.",
        "reference": {
            "split": "OpenImages holdout4096 seed3456 full-image",
            "beta_csv": str(BETA_CSV.relative_to(ROOT)),
            "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
            "note": "Posthoc threshold tuning is diagnostic, not a paper-valid validation protocol.",
        },
        "controllers": summaries,
        "best_deployable_threshold": best,
        "decision": (
            "The controller rows are not uniformly bad. Full-use E078/E080 lose on mean RD, but selecting them only "
            "on high beta005 raw-gate or high beta005 delta-RMS images gives a diagnostic mixed-checkpoint gain. "
            f"The best deployable feature threshold is {best['controller']} with {best['feature']} {best['direction']} "
            f"{best['threshold']:.6f}, selecting {best['selected_fraction']:.4f} of images and reaching "
            f"{best['mean_delta_vs_beta005']:+.6f} RD vs beta005."
        ),
        "next_action": (
            "Do not resurrect broad rawbackoff. Instead, train a beta005-initialized reliability head with a local "
            "raw-gate/delta-RMS weighted teacher objective so fallback supervision is concentrated in high-risk regions "
            "while low-risk regions remain anchored to beta005."
        ),
    }
    out_json = ANALYSIS / "e084_selective_controller_threshold_headroom.json"
    out_md = ANALYSIS / "e084_selective_controller_threshold_headroom.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(payload, out_md)
    print(out_json)
    print(out_md)
    print(json.dumps(payload["best_deployable_threshold"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
