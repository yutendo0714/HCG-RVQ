#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"

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
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


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


def load_variant(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(path)
    return {(row["seed"], row["step"], row["path"]): row for row in rows}


def summarize_method(rows: list[dict[str, float | str]], key: str, method_keys: list[str]) -> dict[str, float]:
    rd = mean(row[f"{key}_rd"] for row in rows)
    result = {"rd": rd}
    for ref in method_keys:
        if ref != key:
            result[f"delta_vs_{ref}"] = rd - mean(row[f"{ref}_rd"] for row in rows)
    for ref in ["hcs", "old", "min090", "previous_local", "variant250", "variant500"]:
        result[f"win_vs_{ref}"] = mean(1.0 if row[f"{key}_rd"] < row[f"{ref}_rd"] else 0.0 for row in rows)
    return result


def feature_summary(rows: list[dict[str, str]], name: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {"method": name}
    for key in FEATURE_KEYS:
        out[key] = mean(maybe_float(row, key) for row in rows)
    return out


def build_rows(variant: dict[tuple[str, str, str], dict[str, str]]) -> tuple[list[dict[str, float | str]], dict[str, list[dict[str, str]]]]:
    image_rows: list[dict[str, float | str]] = []
    feature_rows = {"old": [], "min090": [], "previous_local": [], "variant250": [], "variant500": []}

    for seed, cfg in SEEDS.items():
        hcs_old = by_path(ANALYSIS / cfg["hcs_old"])
        old_features = by_path(ANALYSIS / cfg["old_features"])
        min090 = by_path(ANALYSIS / cfg["min090"])
        previous_local = by_path(ANALYSIS / cfg["previous_local"])
        paths = sorted(set(hcs_old) & set(old_features) & set(min090) & set(previous_local))
        if len(paths) != 4096:
            raise RuntimeError(f"seed {seed}: expected 4096 aligned paths, got {len(paths)}")

        for path in paths:
            var250 = variant[(seed, "250", path)]
            var500 = variant[(seed, "500", path)]
            row = {
                "seed": seed,
                "path": path,
                "hcs_rd": as_float(hcs_old[path], "HCS_rd_score"),
                "old_rd": as_float(hcs_old[path], "old_gate025_rd_score"),
                "min090_rd": as_float(min090[path], "rd_score"),
                "previous_local_rd": as_float(previous_local[path], "rd_score"),
                "variant250_rd": as_float(var250, "rd_score"),
                "variant500_rd": as_float(var500, "rd_score"),
            }
            for method in ["old", "min090", "previous_local", "variant250", "variant500"]:
                row[f"{method}_minus_hcs"] = float(row[f"{method}_rd"]) - float(row["hcs_rd"])
                row[f"{method}_minus_previous_local"] = float(row[f"{method}_rd"]) - float(row["previous_local_rd"])
            row["variant500_minus_variant250"] = float(row["variant500_rd"]) - float(row["variant250_rd"])
            image_rows.append(row)
            feature_rows["old"].append(old_features[path])
            feature_rows["min090"].append(min090[path])
            feature_rows["previous_local"].append(previous_local[path])
            feature_rows["variant250"].append(var250)
            feature_rows["variant500"].append(var500)

    return image_rows, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail and feature analysis for a holdout4096 variant CSV.")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--variant-csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    variant_csv = args.variant_csv if args.variant_csv.is_absolute() else ROOT / args.variant_csv
    output_prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    variant = load_variant(variant_csv)
    image_rows, feature_rows = build_rows(variant)

    method_keys = ["hcs", "old", "min090", "previous_local", "variant250", "variant500"]
    summaries = {key: summarize_method(image_rows, key, method_keys) for key in method_keys}

    seed_summaries = []
    for seed in SEEDS:
        subset = [row for row in image_rows if row["seed"] == seed]
        seed_summaries.append({"seed": seed, **{key: summarize_method(subset, key, method_keys)["rd"] for key in method_keys}})

    sorted_rows = sorted(image_rows, key=lambda row: float(row["hcs_rd"]))
    qsize = len(sorted_rows) // 4
    quartiles = []
    for idx in range(4):
        subset = sorted_rows[idx * qsize : (idx + 1) * qsize]
        quartiles.append({
            "quartile": f"Q{idx + 1}",
            "num_images": len(subset),
            "hcs_rd_min": float(subset[0]["hcs_rd"]),
            "hcs_rd_max": float(subset[-1]["hcs_rd"]),
            "old_minus_hcs": mean(row["old_minus_hcs"] for row in subset),
            "min090_minus_hcs": mean(row["min090_minus_hcs"] for row in subset),
            "previous_local_minus_hcs": mean(row["previous_local_minus_hcs"] for row in subset),
            "variant250_minus_hcs": mean(row["variant250_minus_hcs"] for row in subset),
            "variant500_minus_hcs": mean(row["variant500_minus_hcs"] for row in subset),
            "variant500_minus_variant250": mean(row["variant500_minus_variant250"] for row in subset),
            "variant500_minus_previous_local": mean(row["variant500_minus_previous_local"] for row in subset),
            "variant500_win_vs_hcs": mean(1.0 if row["variant500_rd"] < row["hcs_rd"] else 0.0 for row in subset),
            "variant250_win_vs_hcs": mean(1.0 if row["variant250_rd"] < row["hcs_rd"] else 0.0 for row in subset),
        })

    features = {name: feature_summary(rows, name) for name, rows in feature_rows.items()}
    correlations = []
    y_500_hcs = [row["variant500_minus_hcs"] for row in image_rows]
    y_500_prev = [row["variant500_minus_previous_local"] for row in image_rows]
    y_500_250 = [row["variant500_minus_variant250"] for row in image_rows]
    for label, values in {
        "HCS RD difficulty": [row["hcs_rd"] for row in image_rows],
        "variant500 s_q": [maybe_float(row, "rvq_s_q_mean") for row in feature_rows["variant500"]],
        "variant500 strength": [maybe_float(row, "rvq_householder_strength") for row in feature_rows["variant500"]],
        "variant500 delta RMS": [maybe_float(row, "rvq_householder_delta_rms") for row in feature_rows["variant500"]],
        "variant500 local delta mean": [maybe_float(row, "rvq_householder_delta_rms_local_mean") for row in feature_rows["variant500"]],
        "variant500 qMSE": [maybe_float(row, "rvq_latent_quant_mse") for row in feature_rows["variant500"]],
        "variant500 risk multiplier": [maybe_float(row, "rvq_householder_risk_multiplier") for row in feature_rows["variant500"]],
    }.items():
        correlations.append({
            "feature": label,
            "r_with_variant500_minus_hcs": pearson(values, y_500_hcs),
            "r_with_variant500_minus_previous_local": pearson(values, y_500_prev),
            "r_with_variant500_minus_variant250": pearson(values, y_500_250),
        })

    result = {
        "variant": args.variant_name,
        "num_images": len(image_rows),
        "summaries": summaries,
        "seed_summaries": seed_summaries,
        "quartiles_by_hcs_difficulty": quartiles,
        "feature_summaries": features,
        "correlations": correlations,
        "input_csv": str(variant_csv.relative_to(ROOT)),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    labels = {
        "hcs": "HCS",
        "old": "old gate0.25",
        "min090": "min090",
        "previous_local": "local cap080/rho1 step250",
        "variant250": f"{args.variant_name} step250",
        "variant500": f"{args.variant_name} step500",
    }
    lines = [
        f"# {args.variant_name} Tail and Feature Analysis",
        "",
        "OpenImages holdout4096, 3 seeds, path-matched against trusted HCS/old gate0.25/min090 and previous local cap080/rho1 direct probes. Delta is method minus reference, so negative RD is better.",
        "",
        "## Overall RD",
        "",
        "| method | RD | vs HCS | vs old | vs min090 | vs previous local | win vs HCS | win vs previous local |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
        "| seed | HCS | old | min090 | previous local | variant step250 | variant step500 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in seed_summaries:
        lines.append(
            "| {seed} | {hcs} | {old} | {min090} | {prev} | {v250} | {v500} |".format(
                seed=row["seed"],
                hcs=fmt(row["hcs"]),
                old=fmt(row["old"]),
                min090=fmt(row["min090"]),
                prev=fmt(row["previous_local"]),
                v250=fmt(row["variant250"]),
                v500=fmt(row["variant500"]),
            )
        )

    lines.extend([
        "",
        "## HCS-Difficulty Quartiles",
        "",
        "| quartile | HCS range | old-HCS | min090-HCS | previous local-HCS | variant250-HCS | variant500-HCS | variant500-variant250 | variant500-prev local | variant500 win vs HCS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in quartiles:
        lines.append(
            "| {q} | {lo:.3f}-{hi:.3f} | {old} | {min090} | {prev} | {v250} | {v500} | {vstep} | {vprev} | {win} |".format(
                q=row["quartile"],
                lo=row["hcs_rd_min"],
                hi=row["hcs_rd_max"],
                old=fmt(row["old_minus_hcs"], signed=True),
                min090=fmt(row["min090_minus_hcs"], signed=True),
                prev=fmt(row["previous_local_minus_hcs"], signed=True),
                v250=fmt(row["variant250_minus_hcs"], signed=True),
                v500=fmt(row["variant500_minus_hcs"], signed=True),
                vstep=fmt(row["variant500_minus_variant250"], signed=True),
                vprev=fmt(row["variant500_minus_previous_local"], signed=True),
                win=fmt(row["variant500_win_vs_hcs"]),
            )
        )

    lines.extend([
        "",
        "## Feature Means",
        "",
        "| method | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | risk mult |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in ["old", "min090", "previous_local", "variant250", "variant500"]:
        row = features[key]
        lines.append(
            "| {label} | {sq} | {st} | {dr} | {ld} | {qm} | {pp} | {dead} | {risk} |".format(
                label=labels[key],
                sq=fmt(row["rvq_s_q_mean"]),
                st=fmt(row["rvq_householder_strength"]),
                dr=fmt(row["rvq_householder_delta_rms"]),
                ld=fmt(row["rvq_householder_delta_rms_local_mean"]),
                qm=fmt(row["rvq_latent_quant_mse"]),
                pp=fmt(row["rvq_perplexity"]),
                dead=fmt(row["rvq_dead_code_ratio"]),
                risk=fmt(row["rvq_householder_risk_multiplier"]),
            )
        )

    lines.extend([
        "",
        "## Correlations",
        "",
        "| feature | r with variant500-HCS | r with variant500-previous local | r with variant500-variant250 |",
        "|---|---:|---:|---:|",
    ])
    for row in correlations:
        lines.append(
            "| {feature} | {rh} | {rp} | {rs} |".format(
                feature=row["feature"],
                rh=fmt(row["r_with_variant500_minus_hcs"], signed=True),
                rp=fmt(row["r_with_variant500_minus_previous_local"], signed=True),
                rs=fmt(row["r_with_variant500_minus_variant250"], signed=True),
            )
        )

    lines.extend([
        "",
        "Interpretation guardrail:",
        "",
        "- This is a diagnostic analysis, not a new test split. Use it to decide the next single-checkpoint training design and to explain failure modes.",
        "- A per-seed best checkpoint is useful for mechanism diagnosis, but a paper claim should prefer a fixed checkpoint selection rule or an independent validation split.",
    ])
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(output_prefix.with_suffix(".md")), "output_json": str(output_prefix.with_suffix(".json")), "output_csv": str(csv_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
