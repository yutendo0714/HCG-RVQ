#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_PREFIX = ANALYSIS / "excessrisk090_local_cap080_rho1_tail_holdout4096"

SEEDS = {
    "1234": {
        "hcs_old": "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "old_features": "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed1234_step250_val4096_holdout4096_current.csv",
    },
    "2345": {
        "hcs_old": "per_image_seed2345_hcs250_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "old_features": "per_image_features_hcg_h_gate025_seed2345_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed2345_step250_val4096_holdout4096_current.csv",
    },
    "3456": {
        "hcs_old": "per_image_seed3456_hcs250_vs_hcgh_gate025_step500_val4096_holdout4096_current.csv",
        "old_features": "per_image_features_hcg_h_gate025_seed3456_step500_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed3456_step250_val4096_holdout4096_current.csv",
    },
}

FEATURE_KEYS = [
    "rvq_s_q_mean",
    "rvq_householder_strength",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
    "rvq_householder_risk_multiplier",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_csv(path)}


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}: {value}")
    return value


def maybe_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        return float("nan")
    value = float(raw)
    return value if math.isfinite(value) else float("nan")


def mean(values) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def pearson(xs, ys) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def summarize_method(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    result = {"rd": mean(row[f"{key}_rd"] for row in rows)}
    for ref in ["hcs", "old", "min090", "previous_local", "excess250", "excess500"]:
        if ref != key:
            result[f"delta_vs_{ref}"] = result["rd"] - mean(row[f"{ref}_rd"] for row in rows)
    result["win_vs_hcs"] = mean(1.0 if row[f"{key}_rd"] < row["hcs_rd"] else 0.0 for row in rows)
    result["win_vs_old"] = mean(1.0 if row[f"{key}_rd"] < row["old_rd"] else 0.0 for row in rows)
    result["win_vs_min090"] = mean(1.0 if row[f"{key}_rd"] < row["min090_rd"] else 0.0 for row in rows)
    result["win_vs_previous_local"] = mean(1.0 if row[f"{key}_rd"] < row["previous_local_rd"] else 0.0 for row in rows)
    return result


def feature_summary(rows: list[dict[str, str]], name: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {"method": name}
    for key in FEATURE_KEYS:
        out[key] = mean(maybe_float(row, key) for row in rows)
    return out


def load_excess() -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(ANALYSIS / "excessrisk090_local_cap080_rho1_holdout4096_checkpoint_sweep.csv")
    return {(row["seed"], row["step"], row["path"]): row for row in rows}


def main() -> None:
    excess = load_excess()
    image_rows: list[dict[str, float | str]] = []
    feature_rows = {"old": [], "min090": [], "previous_local": [], "excess250": [], "excess500": []}

    for seed, cfg in SEEDS.items():
        hcs_old = by_path(ANALYSIS / cfg["hcs_old"])
        old_features = by_path(ANALYSIS / cfg["old_features"])
        min090 = by_path(ANALYSIS / cfg["min090"])
        previous_local = by_path(ANALYSIS / cfg["previous_local"])
        paths = sorted(set(hcs_old) & set(old_features) & set(min090) & set(previous_local))
        if len(paths) != 4096:
            raise RuntimeError(f"seed {seed}: expected 4096 aligned paths, got {len(paths)}")
        for path in paths:
            ex250 = excess[(seed, "250", path)]
            ex500 = excess[(seed, "500", path)]
            row = {
                "seed": seed,
                "path": path,
                "hcs_rd": as_float(hcs_old[path], "HCS_rd_score"),
                "old_rd": as_float(hcs_old[path], "old_gate025_rd_score"),
                "min090_rd": as_float(min090[path], "rd_score"),
                "previous_local_rd": as_float(previous_local[path], "rd_score"),
                "excess250_rd": as_float(ex250, "rd_score"),
                "excess500_rd": as_float(ex500, "rd_score"),
            }
            for method in ["old", "min090", "previous_local", "excess250", "excess500"]:
                row[f"{method}_minus_hcs"] = row[f"{method}_rd"] - row["hcs_rd"]
            row["excess500_minus_old"] = row["excess500_rd"] - row["old_rd"]
            row["excess500_minus_min090"] = row["excess500_rd"] - row["min090_rd"]
            row["excess500_minus_previous_local"] = row["excess500_rd"] - row["previous_local_rd"]
            image_rows.append(row)
            feature_rows["old"].append(old_features[path])
            feature_rows["min090"].append(min090[path])
            feature_rows["previous_local"].append(previous_local[path])
            feature_rows["excess250"].append(ex250)
            feature_rows["excess500"].append(ex500)

    method_keys = ["hcs", "old", "min090", "previous_local", "excess250", "excess500"]
    summaries = {key: summarize_method(image_rows, key) for key in method_keys}

    seed_summaries = []
    for seed in SEEDS:
        subset = [row for row in image_rows if row["seed"] == seed]
        seed_summaries.append({"seed": seed, **{key: summarize_method(subset, key)["rd"] for key in method_keys}})

    sorted_rows = sorted(image_rows, key=lambda row: row["hcs_rd"])
    qsize = len(sorted_rows) // 4
    quartiles = []
    for idx in range(4):
        subset = sorted_rows[idx * qsize : (idx + 1) * qsize]
        quartiles.append({
            "quartile": f"Q{idx + 1}",
            "num_images": len(subset),
            "hcs_rd_min": subset[0]["hcs_rd"],
            "hcs_rd_max": subset[-1]["hcs_rd"],
            "old_minus_hcs": mean(row["old_minus_hcs"] for row in subset),
            "min090_minus_hcs": mean(row["min090_minus_hcs"] for row in subset),
            "previous_local_minus_hcs": mean(row["previous_local_minus_hcs"] for row in subset),
            "excess250_minus_hcs": mean(row["excess250_minus_hcs"] for row in subset),
            "excess500_minus_hcs": mean(row["excess500_minus_hcs"] for row in subset),
            "excess500_minus_old": mean(row["excess500_minus_old"] for row in subset),
            "excess500_minus_min090": mean(row["excess500_minus_min090"] for row in subset),
            "excess500_minus_previous_local": mean(row["excess500_minus_previous_local"] for row in subset),
            "excess500_win_vs_hcs": mean(1.0 if row["excess500_rd"] < row["hcs_rd"] else 0.0 for row in subset),
        })

    features = {name: feature_summary(rows, name) for name, rows in feature_rows.items()}
    correlations = []
    y_hcs = [row["excess500_minus_hcs"] for row in image_rows]
    y_prev = [row["excess500_minus_previous_local"] for row in image_rows]
    for label, values in {
        "HCS RD difficulty": [row["hcs_rd"] for row in image_rows],
        "excess500 s_q": [maybe_float(row, "rvq_s_q_mean") for row in feature_rows["excess500"]],
        "excess500 strength": [maybe_float(row, "rvq_householder_strength") for row in feature_rows["excess500"]],
        "excess500 delta RMS": [maybe_float(row, "rvq_householder_delta_rms") for row in feature_rows["excess500"]],
        "excess500 local delta mean": [maybe_float(row, "rvq_householder_delta_rms_local_mean") for row in feature_rows["excess500"]],
        "excess500 qMSE": [maybe_float(row, "rvq_latent_quant_mse") for row in feature_rows["excess500"]],
        "excess500 risk multiplier": [maybe_float(row, "rvq_householder_risk_multiplier") for row in feature_rows["excess500"]],
    }.items():
        correlations.append({
            "feature": label,
            "r_with_excess500_minus_hcs": pearson(values, y_hcs),
            "r_with_excess500_minus_previous_local": pearson(values, y_prev),
        })

    result = {
        "num_images": len(image_rows),
        "summaries": summaries,
        "seed_summaries": seed_summaries,
        "quartiles_by_hcs_difficulty": quartiles,
        "feature_summaries": features,
        "correlations": correlations,
    }

    csv_path = OUT_PREFIX.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Excess-Risk Local Cap080/rho1 Tail Analysis",
        "",
        "OpenImages holdout4096, 3 seeds, path-matched against trusted HCS/old gate0.25/min090 and previous local cap080/rho1 direct probes. Delta is method minus reference, so negative RD is better.",
        "",
        "## Overall RD",
        "",
        "| method | RD | vs HCS | vs old | vs min090 | vs previous local | win vs HCS | win vs previous local |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "hcs": "HCS",
        "old": "old gate0.25",
        "min090": "min090",
        "previous_local": "local cap080/rho1 step250",
        "excess250": "excess-risk step250",
        "excess500": "excess-risk step500 fixed",
    }
    for key in method_keys:
        item = summaries[key]
        lines.append(
            "| {label} | {rd} | {dh} | {do} | {dm} | {dl} | {wh} | {wl} |".format(
                label=labels[key],
                rd=fmt(item["rd"]),
                dh=fmt(item.get("delta_vs_hcs", 0.0), signed=True),
                do=fmt(item.get("delta_vs_old", 0.0), signed=True),
                dm=fmt(item.get("delta_vs_min090", 0.0), signed=True),
                dl=fmt(item.get("delta_vs_previous_local", 0.0), signed=True),
                wh=fmt(item["win_vs_hcs"]),
                wl=fmt(item["win_vs_previous_local"]),
            )
        )

    lines.extend([
        "",
        "## Per-Seed RD",
        "",
        "| seed | HCS | old | min090 | previous local | excess step250 | excess step500 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in seed_summaries:
        lines.append("| {seed} | {hcs} | {old} | {min090} | {prev} | {e250} | {e500} |".format(
            seed=row["seed"], hcs=fmt(row["hcs"]), old=fmt(row["old"]), min090=fmt(row["min090"]), prev=fmt(row["previous_local"]), e250=fmt(row["excess250"]), e500=fmt(row["excess500"])))

    lines.extend([
        "",
        "## HCS-Difficulty Quartiles",
        "",
        "| quartile | HCS range | old-HCS | min090-HCS | previous local-HCS | excess250-HCS | excess500-HCS | excess500-old | excess500-prev local | excess500 win vs HCS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in quartiles:
        lines.append("| {q} | {lo:.3f}-{hi:.3f} | {old} | {min090} | {prev} | {e250} | {e500} | {eo} | {ep} | {win} |".format(
            q=row["quartile"], lo=row["hcs_rd_min"], hi=row["hcs_rd_max"], old=fmt(row["old_minus_hcs"], signed=True), min090=fmt(row["min090_minus_hcs"], signed=True), prev=fmt(row["previous_local_minus_hcs"], signed=True), e250=fmt(row["excess250_minus_hcs"], signed=True), e500=fmt(row["excess500_minus_hcs"], signed=True), eo=fmt(row["excess500_minus_old"], signed=True), ep=fmt(row["excess500_minus_previous_local"], signed=True), win=fmt(row["excess500_win_vs_hcs"])))

    lines.extend([
        "",
        "## Feature Means",
        "",
        "| method | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | risk mult |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in ["old", "min090", "previous_local", "excess250", "excess500"]:
        row = features[key]
        lines.append("| {label} | {sq} | {st} | {dr} | {ld} | {qm} | {pp} | {dead} | {risk} |".format(
            label=labels[key], sq=fmt(row["rvq_s_q_mean"]), st=fmt(row["rvq_householder_strength"]), dr=fmt(row["rvq_householder_delta_rms"]), ld=fmt(row["rvq_householder_delta_rms_local_mean"]), qm=fmt(row["rvq_latent_quant_mse"]), pp=fmt(row["rvq_perplexity"]), dead=fmt(row["rvq_dead_code_ratio"]), risk=fmt(row["rvq_householder_risk_multiplier"])))

    lines.extend([
        "",
        "## Correlations",
        "",
        "| feature | r with excess500-HCS | r with excess500-previous local |",
        "|---|---:|---:|",
    ])
    for row in correlations:
        lines.append("| {feature} | {rh} | {rp} |".format(feature=row["feature"], rh=fmt(row["r_with_excess500_minus_hcs"], signed=True), rp=fmt(row["r_with_excess500_minus_previous_local"], signed=True)))

    lines.extend([
        "",
        "Conclusion:",
        "",
        "- Fixed step500 is the clean paper-facing candidate for average RD and secondary-split transfer: it improves the 3-seed mean over all current trusted references and over previous local cap080/rho1.",
        "- The mechanism is not the same as the previous local step250 hard-tail story. Fixed step500 strongly repairs Q1/Q2 and remains slightly better than HCS in Q4, but it gives back much of the previous local cap Q4 hard-tail gain.",
        "- Paper positioning should therefore separate two claims: previous/local step250 demonstrates hard-tail reliability control, while excess-risk fixed step500 is the stronger fixed-checkpoint average/transfer candidate.",
        "- The next paper check is a separate checkpoint-selection protocol or another held-out split, since the per-seed-best number is diagnostic rather than a final test result.",
    ])
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT_PREFIX.with_suffix(".md")), "output_json": str(OUT_PREFIX.with_suffix(".json")), "output_csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
