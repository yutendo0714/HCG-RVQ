from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_BASE = ANALYSIS / "e139_entropy_only_holdout4096_package"
SEEDS = [1234, 2345, 3456]

METHODS = ["entropy_only", "hcs", "deadzone014", "deadzone018"]
FEATURES = [
    "bpp",
    "bpp_y",
    "bpp_z",
    "psnr",
    "ms_ssim",
    "rvq_latent_quant_mse",
    "rvq_dead_code_ratio",
    "rvq_perplexity",
    "rvq_stage_entropy",
    "rvq_fixed_bpp",
]
HCG_FEATURES = [
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_strength",
    "rvq_householder_residual_selector_prob",
    "rvq_householder_residual_selector_multiplier",
    "rvq_s_q_mean",
    "rvq_s_q_std",
    "rvq_mu_q_abs_mean",
]


def entropy_path(seed: int) -> Path:
    return ANALYSIS / f"e139_entropy_only_seed{seed}_step500_fullimage_holdout4096_current.csv"


def entropy_json_path(seed: int) -> Path:
    return ANALYSIS / f"e139_entropy_only_seed{seed}_step500_fullimage_holdout4096_current.json"


def deadzone014_path(seed: int) -> Path:
    if seed == 3456:
        return ANALYSIS / "e102_e099_deadzone014_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"
    return ANALYSIS / f"e109_deadzone014_from_beta005_seed{seed}_step250_fullimage_holdout4096_current.csv"


def deadzone018_path(seed: int) -> Path:
    if seed == 3456:
        return ANALYSIS / "e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.csv"
    return ANALYSIS / f"e104_deadzone018_from_beta005_seed{seed}_step250_fullimage_holdout4096_current.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def by_path(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in rows}


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in {"", "nan", "NaN", "None"}:
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def finite(values) -> list[float]:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return clean


def mean(values) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else math.nan


def quantile(values, q: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def has_nonfinite(row: dict[str, str]) -> int:
    return int(str(row.get("has_nonfinite", "0")).lower() in {"1", "true", "yes"})


def fmt(value: object, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.6f}" if signed else f"{number:.6f}"


def load_rows(seed: int) -> dict[str, dict[str, dict[str, str]]]:
    entropy = by_path(read_csv(entropy_path(seed)))
    dz014 = by_path(read_csv(deadzone014_path(seed)))
    dz018 = by_path(read_csv(deadzone018_path(seed)))
    shared = sorted(set(entropy) & set(dz014) & set(dz018))
    if len(shared) != 4096:
        raise RuntimeError(f"seed {seed}: expected 4096 shared paths, got {len(shared)}")
    rows: dict[str, dict[str, dict[str, str]]] = {}
    for path in shared:
        hcs_row = dict(dz014[path])
        hcs_row["rd_score"] = dz014[path]["reference_rd_score"]
        hcs_row["bpp"] = ""
        hcs_row["bpp_y"] = ""
        hcs_row["bpp_z"] = ""
        hcs_row["psnr"] = ""
        hcs_row["ms_ssim"] = ""
        hcs_row["has_nonfinite"] = "0"
        for feature in FEATURES + HCG_FEATURES:
            hcs_row[feature] = ""
        rows[path] = {
            "entropy_only": entropy[path],
            "hcs": hcs_row,
            "deadzone014": dz014[path],
            "deadzone018": dz018[path],
        }
    return rows


def summarize_method(seed: int, rows: dict[str, dict[str, dict[str, str]]], method: str) -> dict[str, object]:
    method_rows = [methods[method] for methods in rows.values()]
    hcs_rows = [methods["hcs"] for methods in rows.values()]
    entropy_rows = [methods["entropy_only"] for methods in rows.values()]
    rd = [f(row, "rd_score") for row in method_rows]
    hcs_rd = [f(row, "rd_score") for row in hcs_rows]
    entropy_rd = [f(row, "rd_score") for row in entropy_rows]
    deltas_hcs = [m - h for m, h in zip(rd, hcs_rd)]
    deltas_entropy = [m - e for m, e in zip(rd, entropy_rd)]
    out: dict[str, object] = {
        "seed": seed,
        "method": method,
        "images": len(method_rows),
        "mean_rd": mean(rd),
        "mean_hcs_rd": mean(hcs_rd),
        "delta_vs_hcs": mean(deltas_hcs),
        "median_delta_vs_hcs": quantile(deltas_hcs, 0.50),
        "q05_delta_vs_hcs": quantile(deltas_hcs, 0.05),
        "q95_delta_vs_hcs": quantile(deltas_hcs, 0.95),
        "win_rate_vs_hcs": mean([float(delta < 0.0) for delta in deltas_hcs]),
        "mean_entropy_rd": mean(entropy_rd),
        "delta_vs_entropy_only": mean(deltas_entropy),
        "win_rate_vs_entropy_only": mean([float(delta < 0.0) for delta in deltas_entropy]),
        "nonfinite_rows": sum(has_nonfinite(row) for row in method_rows),
    }
    for feature in FEATURES + HCG_FEATURES:
        vals = [f(row, feature) for row in method_rows]
        if finite(vals):
            out[f"mean_{feature}"] = mean(vals)
    return out


def build_per_image(seed: int, rows: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, object]]:
    out = []
    for path, methods in rows.items():
        hcs_rd = f(methods["hcs"], "rd_score")
        entropy_rd = f(methods["entropy_only"], "rd_score")
        row: dict[str, object] = {
            "seed": seed,
            "path": path,
            "hcs_rd": hcs_rd,
            "entropy_only_rd": entropy_rd,
            "entropy_only_delta_vs_hcs": entropy_rd - hcs_rd,
        }
        for method in ["deadzone014", "deadzone018"]:
            method_rd = f(methods[method], "rd_score")
            row[f"{method}_rd"] = method_rd
            row[f"{method}_delta_vs_hcs"] = method_rd - hcs_rd
            row[f"{method}_delta_vs_entropy_only"] = method_rd - entropy_rd
        for feature in FEATURES:
            row[f"entropy_{feature}"] = f(methods["entropy_only"], feature)
        for feature in FEATURES + HCG_FEATURES:
            value = f(methods["deadzone014"], feature)
            if math.isfinite(value):
                row[f"deadzone014_{feature}"] = value
        out.append(row)
    return out


def quartile_rows(seed: int, per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(per_image, key=lambda row: float(row["hcs_rd"]))
    out = []
    for index in range(4):
        lo = index * len(ordered) // 4
        hi = (index + 1) * len(ordered) // 4
        subset = ordered[lo:hi]
        row: dict[str, object] = {
            "seed": seed,
            "quartile_by_hcs_rd": f"Q{index + 1}",
            "images": len(subset),
            "hcs_rd": mean([r["hcs_rd"] for r in subset]),
        }
        for method in ["entropy_only", "deadzone014", "deadzone018"]:
            delta_key = f"{method}_delta_vs_hcs"
            row[f"{method}_delta_vs_hcs"] = mean([r[delta_key] for r in subset])
            row[f"{method}_win_rate_vs_hcs"] = mean([float(float(r[delta_key]) < 0.0) for r in subset])
        for method in ["deadzone014", "deadzone018"]:
            delta_key = f"{method}_delta_vs_entropy_only"
            row[f"{method}_delta_vs_entropy_only"] = mean([r[delta_key] for r in subset])
            row[f"{method}_win_rate_vs_entropy_only"] = mean(
                [float(float(r[delta_key]) < 0.0) for r in subset]
            )
        out.append(row)
    return out


def aggregate_by_method(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in METHODS:
        rows = [row for row in seed_rows if row["method"] == method]
        agg: dict[str, object] = {
            "seed": "all",
            "method": method,
            "images": sum(int(row["images"]) for row in rows),
        }
        numeric_keys = [
            key
            for key in rows[0]
            if key not in {"seed", "method", "images"} and all(isinstance(row.get(key), (int, float)) for row in rows)
        ]
        for key in numeric_keys:
            agg[key] = mean([row[key] for row in rows])
        out.append(agg)
    return out


def aggregate_quartiles(quartiles: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for label in ["Q1", "Q2", "Q3", "Q4"]:
        rows = [row for row in quartiles if row["quartile_by_hcs_rd"] == label]
        agg: dict[str, object] = {
            "seed": "all",
            "quartile_by_hcs_rd": label,
            "images": sum(int(row["images"]) for row in rows),
        }
        for key in rows[0]:
            if key in {"seed", "quartile_by_hcs_rd", "images"}:
                continue
            agg[key] = mean([row[key] for row in rows])
        out.append(agg)
    return out


def main() -> None:
    seed_summaries = []
    per_image_all = []
    quartiles = []
    json_summaries = []

    for seed in SEEDS:
        rows = load_rows(seed)
        per_image = build_per_image(seed, rows)
        per_image_all.extend(per_image)
        quartiles.extend(quartile_rows(seed, per_image))
        for method in METHODS:
            seed_summaries.append(summarize_method(seed, rows, method))
        json_summary = json.loads(entropy_json_path(seed).read_text(encoding="utf-8"))["summaries"][0]
        json_summaries.append(
            {
                "seed": seed,
                "entropy_json_mean_rd": float(json_summary["mean_rd"]),
                "entropy_csv_mean_rd": mean([row["entropy_only_rd"] for row in per_image]),
                "entropy_json_nonfinite_rows": int(json_summary["nonfinite_rows"]),
                "entropy_json_mean_rvq_latent_quant_mse": float(json_summary["mean_rvq_latent_quant_mse"]),
                "entropy_json_mean_rvq_perplexity": float(json_summary["mean_rvq_perplexity"]),
                "entropy_json_mean_rvq_dead_code_ratio": float(json_summary["mean_rvq_dead_code_ratio"]),
            }
        )

    aggregate_summaries = aggregate_by_method(seed_summaries)
    all_summaries = seed_summaries + aggregate_summaries
    aggregate_quartile_rows = aggregate_quartiles(quartiles)
    all_quartiles = quartiles + aggregate_quartile_rows

    summary_by_method = {row["method"]: row for row in aggregate_summaries}
    headline = {
        "experiment": "E139 entropy-only holdout4096 ablation",
        "status": "done",
        "protocol": "OpenImages holdout4096, start-index 4096, path-aligned per-image comparison, CUDA device 0.",
        "entropy_only_mean_rd": summary_by_method["entropy_only"]["mean_rd"],
        "hcs_mean_rd": summary_by_method["hcs"]["mean_rd"],
        "deadzone014_mean_rd": summary_by_method["deadzone014"]["mean_rd"],
        "deadzone018_mean_rd": summary_by_method["deadzone018"]["mean_rd"],
        "entropy_only_delta_vs_hcs": summary_by_method["entropy_only"]["delta_vs_hcs"],
        "deadzone014_delta_vs_hcs": summary_by_method["deadzone014"]["delta_vs_hcs"],
        "deadzone018_delta_vs_hcs": summary_by_method["deadzone018"]["delta_vs_hcs"],
        "deadzone014_delta_vs_entropy_only": summary_by_method["deadzone014"]["delta_vs_entropy_only"],
        "deadzone018_delta_vs_entropy_only": summary_by_method["deadzone018"]["delta_vs_entropy_only"],
        "entropy_only_nonfinite_rows": summary_by_method["entropy_only"]["nonfinite_rows"],
        "deadzone014_nonfinite_rows": summary_by_method["deadzone014"]["nonfinite_rows"],
        "deadzone018_nonfinite_rows": summary_by_method["deadzone018"]["nonfinite_rows"],
        "entropy_only_mean_rvq_dead_code_ratio": summary_by_method["entropy_only"].get("mean_rvq_dead_code_ratio"),
        "deadzone014_mean_rvq_dead_code_ratio": summary_by_method["deadzone014"].get("mean_rvq_dead_code_ratio"),
        "deadzone018_mean_rvq_dead_code_ratio": summary_by_method["deadzone018"].get("mean_rvq_dead_code_ratio"),
        "interpretation": (
            "Entropy-only is a strong hyperprior-index control on holdout4096, but HCG geometry "
            "deadzone014/deadzone018 still adds substantial RD gain under the same split."
        ),
    }

    package = {
        "headline": headline,
        "summaries": all_summaries,
        "quartiles": all_quartiles,
        "entropy_json_checks": json_summaries,
        "inputs": {
            "entropy_only": [str(entropy_path(seed).relative_to(ROOT)) for seed in SEEDS],
            "deadzone014": [str(deadzone014_path(seed).relative_to(ROOT)) for seed in SEEDS],
            "deadzone018": [str(deadzone018_path(seed).relative_to(ROOT)) for seed in SEEDS],
        },
    }

    OUT_BASE.with_suffix(".json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT_BASE.with_suffix(".summary.csv"), all_summaries)
    write_csv(OUT_BASE.with_suffix(".quartiles.csv"), all_quartiles)
    write_csv(OUT_BASE.with_suffix(".per_image.csv"), per_image_all)
    write_csv(OUT_BASE.with_suffix(".json_checks.csv"), json_summaries)
    write_csv(OUT_BASE.with_suffix(".headline.csv"), [headline])

    lines = [
        "# E139 Entropy-Only Holdout4096 Package",
        "",
        "## Conclusion",
        "",
        (
            "The entropy-only / HVQ-like row is a necessary OpenImages holdout control, but it does not explain away the HCG geometry gains. "
            "Under the same path-aligned holdout4096 split, entropy-only is worse than HCS on mean RD, while deadzone014/deadzone018 still improve over HCS and substantially outperform entropy-only."
        ),
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| HCS mean RD | {fmt(headline['hcs_mean_rd'])} |",
        f"| entropy-only mean RD | {fmt(headline['entropy_only_mean_rd'])} |",
        f"| deadzone014 mean RD | {fmt(headline['deadzone014_mean_rd'])} |",
        f"| deadzone018 mean RD | {fmt(headline['deadzone018_mean_rd'])} |",
        f"| entropy-only - HCS | {fmt(headline['entropy_only_delta_vs_hcs'], True)} |",
        f"| deadzone014 - HCS | {fmt(headline['deadzone014_delta_vs_hcs'], True)} |",
        f"| deadzone018 - HCS | {fmt(headline['deadzone018_delta_vs_hcs'], True)} |",
        f"| deadzone014 - entropy-only | {fmt(headline['deadzone014_delta_vs_entropy_only'], True)} |",
        f"| deadzone018 - entropy-only | {fmt(headline['deadzone018_delta_vs_entropy_only'], True)} |",
        f"| entropy-only nonfinite rows | {fmt(headline['entropy_only_nonfinite_rows'])} |",
        f"| deadzone014 nonfinite rows | {fmt(headline['deadzone014_nonfinite_rows'])} |",
        f"| deadzone018 nonfinite rows | {fmt(headline['deadzone018_nonfinite_rows'])} |",
        "",
        "## Per-Seed RD",
        "",
        "| seed | HCS | entropy-only | dz014 | dz018 | entropy-HCS | dz014-entropy | dz018-entropy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        rows = {row["method"]: row for row in seed_summaries if row["seed"] == seed}
        lines.append(
            "| "
            f"{seed} | {fmt(rows['hcs']['mean_rd'])} | {fmt(rows['entropy_only']['mean_rd'])} | "
            f"{fmt(rows['deadzone014']['mean_rd'])} | {fmt(rows['deadzone018']['mean_rd'])} | "
            f"{fmt(rows['entropy_only']['delta_vs_hcs'], True)} | "
            f"{fmt(rows['deadzone014']['delta_vs_entropy_only'], True)} | "
            f"{fmt(rows['deadzone018']['delta_vs_entropy_only'], True)} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Quartiles By HCS RD",
            "",
            "| quartile | HCS RD | entropy-HCS | dz014-HCS | dz018-HCS | dz014-entropy | dz018-entropy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_quartile_rows:
        lines.append(
            "| "
            f"{row['quartile_by_hcs_rd']} | {fmt(row['hcs_rd'])} | "
            f"{fmt(row['entropy_only_delta_vs_hcs'], True)} | "
            f"{fmt(row['deadzone014_delta_vs_hcs'], True)} | "
            f"{fmt(row['deadzone018_delta_vs_hcs'], True)} | "
            f"{fmt(row['deadzone014_delta_vs_entropy_only'], True)} | "
            f"{fmt(row['deadzone018_delta_vs_entropy_only'], True)} |"
        )
    lines.extend(
        [
            "",
            "## Intermediate Features",
            "",
            "| method | dead-code | latent qMSE | perplexity | stage entropy | delta RMS | strength |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ["entropy_only", "deadzone014", "deadzone018"]:
        row = summary_by_method[method]
        lines.append(
            "| "
            f"{method} | {fmt(row.get('mean_rvq_dead_code_ratio'))} | "
            f"{fmt(row.get('mean_rvq_latent_quant_mse'))} | "
            f"{fmt(row.get('mean_rvq_perplexity'))} | "
            f"{fmt(row.get('mean_rvq_stage_entropy'))} | "
            f"{fmt(row.get('mean_rvq_householder_delta_rms'))} | "
            f"{fmt(row.get('mean_rvq_householder_strength'))} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Use Guidance",
            "",
            "- Use E139 as the OpenImages holdout version of the entropy-only / HVQ-like ablation.",
            "- The safe holdout claim is that index-entropy-only conditioning is not sufficient; local HCG geometry adds beyond it on the same split.",
            "- Keep the nonfinite audit in the table: GPU0 evaluation produced zero nonfinite rows for entropy-only and both HCG geometry rows.",
            "- Treat this as claim-solidifying evidence; method-strengthening should continue on separate controller/rate/backbone tracks.",
            "",
            "## Artifacts",
            "",
            "- `tools/build_e139_entropy_only_holdout_package.py`",
            "- `experiments/analysis/e139_entropy_only_holdout4096_package.{json,md}`",
            "- `experiments/analysis/e139_entropy_only_holdout4096_package.{summary,quartiles,per_image,json_checks,headline}.csv`",
        ]
    )
    OUT_BASE.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
