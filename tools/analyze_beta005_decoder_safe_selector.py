from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT = ANALYSIS / "beta005_previous_local_decoder_safe_selector"
VARIANT_CSV = ANALYSIS / "excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.csv"

SEEDS = {
    "1234": {
        "hcs_old": "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed1234_step250_val4096_holdout4096_current.csv",
    },
    "2345": {
        "hcs_old": "per_image_seed2345_hcs250_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed2345_step250_val4096_holdout4096_current.csv",
    },
    "3456": {
        "hcs_old": "per_image_seed3456_hcs250_vs_hcgh_gate025_step500_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_holdout4096_current.csv",
        "previous_local": "direct_local_cap080_rho1_seed3456_step250_val4096_holdout4096_current.csv",
    },
}

DECODER_SAFE_FEATURES = [
    "rvq_s_q_mean",
    "rvq_s_q_min",
    "rvq_s_q_max",
    "rvq_s_q_std",
    "rvq_mu_q_abs_mean",
    "rvq_mu_q_std",
    "rvq_householder_gate_raw",
    "rvq_householder_risk_multiplier",
    "rvq_householder_strength",
    "rvq_householder_strength_min",
    "rvq_householder_strength_max",
    "rvq_householder_strength_std",
    "rvq_householder_v_abs_mean",
]

DIAGNOSTIC_FEATURES = [
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_delta_rms_local_max",
    "rvq_householder_delta_rms_local_std",
    "rvq_latent_quant_mse",
    "rvq_y_norm_abs_mean",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_csv(path)}


def f(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}: {value}")
    return value


def mean(values) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def percentile_thresholds(values: list[float], bins: int = 101) -> list[float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return []
    thresholds = []
    n = len(values) - 1
    for i in range(bins):
        pos = n * i / (bins - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            thresholds.append(values[lo])
        else:
            alpha = pos - lo
            thresholds.append(values[lo] * (1.0 - alpha) + values[hi] * alpha)
    return sorted(set(thresholds))


def load_variant_step500() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(VARIANT_CSV)
    return {(row["seed"], row["path"]): row for row in rows if row["step"] == "500"}


def build_rows() -> list[dict[str, float | str]]:
    variant = load_variant_step500()
    rows: list[dict[str, float | str]] = []
    for seed, cfg in SEEDS.items():
        hcs_old = by_path(ANALYSIS / cfg["hcs_old"])
        min090 = by_path(ANALYSIS / cfg["min090"])
        previous = by_path(ANALYSIS / cfg["previous_local"])
        paths = sorted(set(hcs_old) & set(min090) & set(previous))
        if len(paths) != 4096:
            raise RuntimeError(f"seed {seed}: expected 4096 aligned paths, got {len(paths)}")
        for path in paths:
            beta = variant[(seed, path)]
            row: dict[str, float | str] = {
                "seed": seed,
                "path": path,
                "hcs_rd": f(hcs_old[path], "HCS_rd_score"),
                "old_rd": f(hcs_old[path], "old_gate025_rd_score"),
                "min090_rd": f(min090[path], "rd_score"),
                "previous_local_rd": f(previous[path], "rd_score"),
                "beta005_rd": f(beta, "rd_score"),
            }
            for feature in DECODER_SAFE_FEATURES + DIAGNOSTIC_FEATURES:
                row[feature] = f(beta, feature)
            rows.append(row)
    return rows


def summarize_rd(rows: list[dict[str, float | str]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def evaluate_policy(rows: list[dict[str, float | str]], feature: str, threshold: float, direction: str) -> dict[str, float | str]:
    use_beta = []
    selected_rd = []
    for row in rows:
        value = float(row[feature])
        selected = value <= threshold if direction == "le" else value >= threshold
        use_beta.append(1.0 if selected else 0.0)
        selected_rd.append(float(row["beta005_rd"] if selected else row["previous_local_rd"]))
    rd = mean(selected_rd)
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "rd": rd,
        "delta_vs_hcs": rd - summarize_rd(rows, "hcs_rd"),
        "delta_vs_previous_local": rd - summarize_rd(rows, "previous_local_rd"),
        "delta_vs_beta005": rd - summarize_rd(rows, "beta005_rd"),
        "beta_fraction": mean(use_beta),
    }


def best_thresholds(rows: list[dict[str, float | str]], features: list[str]) -> list[dict[str, float | str]]:
    out = []
    for feature in features:
        values = [float(row[feature]) for row in rows]
        for threshold in percentile_thresholds(values):
            for direction in ["le", "ge"]:
                out.append(evaluate_policy(rows, feature, threshold, direction))
    return sorted(out, key=lambda row: float(row["rd"]))


def oracle(rows: list[dict[str, float | str]]) -> dict[str, float]:
    rd_values = [min(float(row["beta005_rd"]), float(row["previous_local_rd"])) for row in rows]
    rd = mean(rd_values)
    return {
        "rd": rd,
        "delta_vs_hcs": rd - summarize_rd(rows, "hcs_rd"),
        "delta_vs_previous_local": rd - summarize_rd(rows, "previous_local_rd"),
        "delta_vs_beta005": rd - summarize_rd(rows, "beta005_rd"),
    }


def main() -> None:
    rows = build_rows()
    base = {
        "hcs": summarize_rd(rows, "hcs_rd"),
        "old_gate025": summarize_rd(rows, "old_rd"),
        "min090": summarize_rd(rows, "min090_rd"),
        "previous_local": summarize_rd(rows, "previous_local_rd"),
        "beta005": summarize_rd(rows, "beta005_rd"),
    }
    safe_thresholds = best_thresholds(rows, DECODER_SAFE_FEATURES)
    diagnostic_thresholds = best_thresholds(rows, DIAGNOSTIC_FEATURES)
    result = {
        "num_images": len(rows),
        "base": base,
        "oracle_previous_local_or_beta005": oracle(rows),
        "decoder_safe_features": DECODER_SAFE_FEATURES,
        "diagnostic_features": DIAGNOSTIC_FEATURES,
        "top_decoder_safe_thresholds": safe_thresholds[:20],
        "top_diagnostic_thresholds": diagnostic_thresholds[:20],
        "note": (
            "Decoder-safe features depend only on hyperprior-generated conditioning available at encoder and decoder. "
            "Diagnostic features may depend on y/u or codebook outcomes and are used only as headroom guidance."
        ),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["group", "feature", "direction", "threshold", "rd", "delta_vs_hcs", "delta_vs_previous_local", "delta_vs_beta005", "beta_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, items in [("decoder_safe", safe_thresholds), ("diagnostic", diagnostic_thresholds)]:
            for item in items:
                writer.writerow({"group": group, **{key: item[key] for key in fieldnames if key != "group"}})

    lines = [
        "# Beta005 vs Previous-Local Decoder-Safe Selector Audit",
        "",
        "This is a diagnostic audit over existing OpenImages holdout4096 per-image rows. It asks whether the beta005/previous-local complementarity can be approximated by a single feature that is available from the hyperprior-generated conditioning, without switching checkpoints in a paper-facing method.",
        "",
        "## Base RD",
        "",
        "| method | RD | vs HCS |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("hcs", "HCS"),
        ("old_gate025", "old gate0.25"),
        ("min090", "min090"),
        ("previous_local", "previous local step250"),
        ("beta005", "beta005 step500"),
    ]:
        lines.append(f"| {label} | {fmt(base[key])} | {fmt(base[key] - base['hcs'], signed=True)} |")
    ora = result["oracle_previous_local_or_beta005"]
    lines.extend(
        [
            "",
            "## Headroom",
            "",
            "| policy | RD | vs HCS | vs previous local | vs beta005 |",
            "|---|---:|---:|---:|---:|",
            f"| oracle min(previous local, beta005) | {fmt(ora['rd'])} | {fmt(ora['delta_vs_hcs'], signed=True)} | {fmt(ora['delta_vs_previous_local'], signed=True)} | {fmt(ora['delta_vs_beta005'], signed=True)} |",
            "",
            "## Best Decoder-Safe Thresholds",
            "",
            "| feature | dir | threshold | RD | vs HCS | vs previous local | vs beta005 | beta fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in safe_thresholds[:10]:
        lines.append(
            f"| {row['feature']} | {row['direction']} | {fmt(float(row['threshold']))} | {fmt(float(row['rd']))} | "
            f"{fmt(float(row['delta_vs_hcs']), signed=True)} | {fmt(float(row['delta_vs_previous_local']), signed=True)} | "
            f"{fmt(float(row['delta_vs_beta005']), signed=True)} | {fmt(float(row['beta_fraction']))} |"
        )
    lines.extend(
        [
            "",
            "## Best Diagnostic Thresholds",
            "",
            "| feature | dir | threshold | RD | vs HCS | vs previous local | vs beta005 | beta fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_thresholds[:10]:
        lines.append(
            f"| {row['feature']} | {row['direction']} | {fmt(float(row['threshold']))} | {fmt(float(row['rd']))} | "
            f"{fmt(float(row['delta_vs_hcs']), signed=True)} | {fmt(float(row['delta_vs_previous_local']), signed=True)} | "
            f"{fmt(float(row['delta_vs_beta005']), signed=True)} | {fmt(float(row['beta_fraction']))} |"
        )
    best_safe = safe_thresholds[0]
    best_diag = diagnostic_thresholds[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The best decoder-safe one-feature threshold reaches RD `{fmt(float(best_safe['rd']))}`, "
                f"which is `{fmt(float(best_safe['delta_vs_beta005']), signed=True)}` vs beta005. "
                "If this is much weaker than the diagnostic-delta threshold and the oracle, the next controller needs a learned/teacher signal rather than a hand threshold."
            ),
            (
                f"The best diagnostic threshold reaches RD `{fmt(float(best_diag['rd']))}`. "
                "Diagnostic features are useful for target construction, but not directly safe as a deployable decoder-side gate if they depend on the unquantized latent or quantization outcome."
            ),
            "",
            "Artifacts:",
            "",
            f"- `{OUT.with_suffix('.json').relative_to(ROOT)}`",
            f"- `{OUT.with_suffix('.csv').relative_to(ROOT)}`",
        ]
    )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT.with_suffix(".md")), "best_safe": safe_thresholds[0]}, indent=2))


if __name__ == "__main__":
    main()
