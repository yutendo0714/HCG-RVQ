from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def as_float_any(row: dict[str, str], keys: list[str], default: float = math.nan) -> float:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return float(value)
    return default


def prefixed_value(row: dict[str, str], suffix: str, preferred_prefixes: list[str]) -> float:
    keys = [f"{prefix}_{suffix}" for prefix in preferred_prefixes]
    keys.extend(key for key in row if key.endswith("_" + suffix))
    return as_float_any(row, keys)


def mean(values) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else math.nan


def corr(xs, ys) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def build_items(
    old_compare: Path,
    risk_compare: Path,
    old_features: Path,
    risk_features: Path,
) -> list[dict[str, float | int | str]]:
    old_compare_rows = read_rows(old_compare)
    risk_compare_rows = read_rows(risk_compare)
    old_feature_rows = read_rows(old_features)
    risk_feature_rows = read_rows(risk_features)
    assert len(old_compare_rows) == len(risk_compare_rows) == len(old_feature_rows) == len(risk_feature_rows)

    items: list[dict[str, float | int | str]] = []
    for old_cmp, risk_cmp, old_feat, risk_feat in zip(
        old_compare_rows,
        risk_compare_rows,
        old_feature_rows,
        risk_feature_rows,
    ):
        assert old_cmp["index"] == risk_cmp["index"] == old_feat["index"] == risk_feat["index"]
        assert old_cmp["path"] == risk_cmp["path"] == old_feat["path"] == risk_feat["path"]
        old_delta = as_float(old_cmp, "delta_rd_score")
        risk_delta = as_float(risk_cmp, "delta_rd_score")
        hcs_rd = prefixed_value(risk_cmp, "rd_score", ["HCS", "hcs"])
        items.append(
            {
                "index": int(old_cmp["index"]),
                "path": old_cmp["path"],
                "hcs_rd": hcs_rd,
                "old_delta_rd": old_delta,
                "risk_delta_rd": risk_delta,
                "risk_minus_old_delta_rd": risk_delta - old_delta,
                "old_s_q_mean": as_float(old_feat, "s_q_mean"),
                "risk_s_q_mean": as_float(risk_feat, "s_q_mean"),
                "risk_minus_old_s_q_mean": as_float(risk_feat, "s_q_mean") - as_float(old_feat, "s_q_mean"),
                "old_y_norm_abs_mean": as_float(old_feat, "y_norm_abs_mean"),
                "risk_y_norm_abs_mean": as_float(risk_feat, "y_norm_abs_mean"),
                "old_raw_gate_mean": as_float(old_feat, "householder_gate_raw_mean"),
                "risk_raw_gate_mean": as_float(risk_feat, "householder_gate_raw_mean"),
                "old_strength_mean": as_float(old_feat, "householder_strength_mean"),
                "risk_strength_mean": as_float(risk_feat, "householder_strength_mean"),
                "risk_multiplier_mean": as_float(risk_feat, "householder_risk_multiplier_mean"),
                "old_delta_rms": as_float(old_feat, "householder_delta_rms"),
                "risk_delta_rms": as_float(risk_feat, "householder_delta_rms"),
                "old_y_error_rms": as_float(old_feat, "y_error_rms"),
                "risk_y_error_rms": as_float(risk_feat, "y_error_rms"),
                "old_latent_quant_mse": as_float(old_feat, "rvq_latent_quant_mse"),
                "risk_latent_quant_mse": as_float(risk_feat, "rvq_latent_quant_mse"),
                "risk_commit_loss": as_float(risk_feat, "commit_loss"),
                "risk_bpp_y": as_float(risk_feat, "bpp_y"),
                "risk_ms_ssim": as_float(risk_feat, "ms_ssim"),
                "risk_psnr": as_float(risk_feat, "psnr"),
            }
        )
    return items


def summarize_bucket(name: str, subset: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    feature_keys = [
        "old_delta_rd",
        "risk_delta_rd",
        "risk_minus_old_delta_rd",
        "old_s_q_mean",
        "risk_s_q_mean",
        "risk_minus_old_s_q_mean",
        "old_raw_gate_mean",
        "risk_raw_gate_mean",
        "old_strength_mean",
        "risk_strength_mean",
        "risk_multiplier_mean",
        "old_delta_rms",
        "risk_delta_rms",
        "old_y_error_rms",
        "risk_y_error_rms",
        "old_latent_quant_mse",
        "risk_latent_quant_mse",
        "risk_commit_loss",
        "risk_bpp_y",
        "risk_psnr",
        "risk_ms_ssim",
    ]
    row: dict[str, float | int | str] = {
        "bucket": name,
        "n": len(subset),
        "hcs_rd_mean": mean(float(item["hcs_rd"]) for item in subset),
    }
    for key in feature_keys:
        row[f"{key}_mean"] = mean(float(item[key]) for item in subset)
    row["old_win_rate"] = mean(1.0 if float(item["old_delta_rd"]) < 0 else 0.0 for item in subset)
    row["risk_win_rate"] = mean(1.0 if float(item["risk_delta_rd"]) < 0 else 0.0 for item in subset)
    row["risk_beats_old_rate"] = mean(
        1.0 if float(item["risk_minus_old_delta_rd"]) < 0 else 0.0 for item in subset
    )
    return row


def profile(subset: list[dict[str, float | int | str]]) -> dict[str, float]:
    keys = [
        "risk_delta_rd",
        "risk_s_q_mean",
        "risk_multiplier_mean",
        "risk_raw_gate_mean",
        "risk_strength_mean",
        "risk_delta_rms",
        "risk_latent_quant_mse",
        "risk_commit_loss",
        "risk_psnr",
        "risk_ms_ssim",
    ]
    return {key: mean(float(item[key]) for item in subset) for key in keys}


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    correlations: dict[str, dict[str, float]],
    profiles: dict[str, dict[str, float]],
) -> None:
    lines: list[str] = []
    lines.append("# Gate0.25 risk-aware per-image feature tail analysis")
    lines.append("")
    lines.append("Quartiles are sorted by HCS per-image RD. Delta is method minus HCS, so negative RD is better.")
    lines.append("")
    lines.append("## Quartile Feature Means")
    lines.append("")
    cols = [
        "bucket",
        "old_delta_rd_mean",
        "risk_delta_rd_mean",
        "risk_minus_old_delta_rd_mean",
        "old_s_q_mean_mean",
        "risk_s_q_mean_mean",
        "risk_multiplier_mean_mean",
        "old_strength_mean_mean",
        "risk_strength_mean_mean",
        "old_delta_rms_mean",
        "risk_delta_rms_mean",
        "old_latent_quant_mse_mean",
        "risk_latent_quant_mse_mean",
        "risk_commit_loss_mean",
        "risk_win_rate",
        "risk_beats_old_rate",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
    for row in summary_rows:
        values = []
        for col in cols:
            value = row[col]
            values.append(str(value) if isinstance(value, str) else f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Correlations")
    lines.append("")
    lines.append("| scope | feature | corr(HCS RD, feature) | corr(risk delta RD, feature) |")
    lines.append("|---|---|---:|---:|")
    for scope in ["all", "Q4_high_HCS_RD"]:
        for key in [
            "risk_s_q_mean",
            "risk_multiplier_mean",
            "risk_strength_mean",
            "risk_delta_rms",
            "risk_latent_quant_mse",
            "risk_commit_loss",
        ]:
            lines.append(
                f"| {scope} | {key} | "
                f"{correlations[scope]['corr_hcs_rd__' + key]:+.6f} | "
                f"{correlations[scope]['corr_risk_delta_rd__' + key]:+.6f} |"
            )

    lines.append("")
    lines.append("## Q4 Extreme Profiles")
    lines.append("")
    lines.append(
        "| group | risk delta RD | s_q | risk multiplier | raw gate | effective gate | "
        "H delta RMS | latent quant MSE | commit loss | PSNR | MS-SSIM |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, prof in profiles.items():
        lines.append(
            f"| {name} | {prof['risk_delta_rd']:+.6f} | {prof['risk_s_q_mean']:.6f} | "
            f"{prof['risk_multiplier_mean']:.6f} | {prof['risk_raw_gate_mean']:.6f} | "
            f"{prof['risk_strength_mean']:.6f} | {prof['risk_delta_rms']:.6f} | "
            f"{prof['risk_latent_quant_mse']:.6f} | {prof['risk_commit_loss']:.6f} | "
            f"{prof['risk_psnr']:.6f} | {prof['risk_ms_ssim']:.6f} |"
        )

    q1 = summary_rows[1]
    q4 = summary_rows[4]
    q1_multiplier = float(q1["risk_multiplier_mean_mean"])
    q4_multiplier = float(q4["risk_multiplier_mean_mean"])
    multiplier_direction = "increases" if q4_multiplier > q1_multiplier else "decreases"
    multiplier_effect = (
        "so the calibrated/inverted signal keeps more geometry on hard images"
        if q4_multiplier > q1_multiplier
        else "so this risk signal suppresses geometry more on hard images"
    )
    old_q4_mse = float(q4["old_latent_quant_mse_mean"])
    risk_q4_mse = float(q4["risk_latent_quant_mse_mean"])
    mse_direction = "rises" if risk_q4_mse > old_q4_mse else "falls"
    corr_hcs_sq = correlations["all"]["corr_hcs_rd__risk_s_q_mean"]
    corr_hcs_multiplier = correlations["all"]["corr_hcs_rd__risk_multiplier_mean"]
    corr_hcs_strength = correlations["all"]["corr_hcs_rd__risk_strength_mean"]
    corr_delta_mse = correlations["all"]["corr_risk_delta_rd__risk_latent_quant_mse"]
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "- The tail response is quantified with per-image features: "
        f"Q1 changes from old delta {float(q1['old_delta_rd_mean']):+.6f} "
        f"to risk delta {float(q1['risk_delta_rd_mean']):+.6f}, while Q4 changes from "
        f"{float(q4['old_delta_rd_mean']):+.6f} to {float(q4['risk_delta_rd_mean']):+.6f}. "
        "Delta is method minus HCS, so negative RD is better."
    )
    lines.append(
        f"- Risk multiplier {multiplier_direction} from Q1 {q1_multiplier:.6f} "
        f"to Q4 {q4_multiplier:.6f}; {multiplier_effect}."
    )
    lines.append(
        "- Effective gate drops relative to old gate. On Q4, old strength "
        f"{float(summary_rows[4]['old_strength_mean_mean']):.6f} becomes risk strength "
        f"{float(summary_rows[4]['risk_strength_mean_mean']):.6f}, while latent quant MSE {mse_direction} from "
        f"{old_q4_mse:.6f} to {risk_q4_mse:.6f}."
    )
    lines.append(
        "- Across all images, the correlations with HCS RD are "
        f"`s_q` {corr_hcs_sq:+.6f}, risk multiplier {corr_hcs_multiplier:+.6f}, "
        f"and effective gate {corr_hcs_strength:+.6f}. "
        f"Risk delta RD correlates with latent quantization MSE at {corr_delta_mse:+.6f}. "
        "This separates whether the risk signal is aligned with image hardness from whether the "
        "resulting coordinate frame improves RD."
    )
    lines.append(
        "- The co-adaptation concern should remain a hypothesis rather than a proof. "
        "Detached risk prevents direct gate-gradient pressure on `s_q`, but training can still change "
        "`s_q`, raw gate, effective gate, and RVQ/codebook fit jointly through the RD objective."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join gate risk per-image RD and feature diagnostics by HCS-RD quartile.")
    parser.add_argument("--old-compare", required=True)
    parser.add_argument("--risk-compare", required=True)
    parser.add_argument("--old-features", required=True)
    parser.add_argument("--risk-features", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    items = build_items(
        Path(args.old_compare),
        Path(args.risk_compare),
        Path(args.old_features),
        Path(args.risk_features),
    )
    by_rd = sorted(items, key=lambda item: float(item["hcs_rd"]))
    quartiles = [by_rd[i * 1024:(i + 1) * 1024] for i in range(4)]
    buckets = [
        ("all", items),
        ("Q1_low_HCS_RD", quartiles[0]),
        ("Q2", quartiles[1]),
        ("Q3", quartiles[2]),
        ("Q4_high_HCS_RD", quartiles[3]),
    ]
    summary_rows = [summarize_bucket(name, subset) for name, subset in buckets]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    corr_keys = [
        "old_delta_rd",
        "risk_delta_rd",
        "risk_minus_old_delta_rd",
        "old_s_q_mean",
        "risk_s_q_mean",
        "risk_multiplier_mean",
        "old_strength_mean",
        "risk_strength_mean",
        "old_delta_rms",
        "risk_delta_rms",
        "old_y_error_rms",
        "risk_y_error_rms",
        "old_latent_quant_mse",
        "risk_latent_quant_mse",
        "risk_commit_loss",
    ]
    correlations = {"all": {}, "Q4_high_HCS_RD": {}}
    for name, subset in [("all", items), ("Q4_high_HCS_RD", quartiles[3])]:
        for key in corr_keys:
            correlations[name]["corr_hcs_rd__" + key] = corr(
                [float(item["hcs_rd"]) for item in subset],
                [float(item[key]) for item in subset],
            )
            correlations[name]["corr_risk_delta_rd__" + key] = corr(
                [float(item["risk_delta_rd"]) for item in subset],
                [float(item[key]) for item in subset],
            )

    q4 = quartiles[3]
    profiles = {
        "q4_worst20": profile(sorted(q4, key=lambda item: float(item["risk_delta_rd"]), reverse=True)[:20]),
        "q4_best20": profile(sorted(q4, key=lambda item: float(item["risk_delta_rd"]))[:20]),
    }
    write_markdown(Path(args.output_md), summary_rows, correlations, profiles)
    Path(args.output_json).write_text(
        json.dumps({"summary_rows": summary_rows, "correlations": correlations, "profiles": profiles}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputs": [args.output_csv, args.output_md, args.output_json], "profiles": profiles}, indent=2))


if __name__ == "__main__":
    main()
