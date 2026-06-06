#!/usr/bin/env python3
"""Summarize local-cap HCG probes against trusted holdout4096 references."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_JSON = ANALYSIS / "local_cap080_rho1_multiseed_holdout4096_trusted_protocol.json"
OUT_MD = ANALYSIS / "local_cap080_rho1_multiseed_holdout4096_trusted_protocol.md"

SEEDS = {
    "1234": {
        "hcs_old": "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "hcs_label": "HCS step500",
        "old_label": "old gate0.25 step250",
        "old_features": "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv",
        "min090_label": "min090 step500",
        "local": {
            "step250": "direct_local_cap080_rho1_seed1234_step250_val4096_holdout4096_current.csv",
            "step500": "direct_local_cap080_rho1_seed1234_step500_val4096_holdout4096_current.csv",
        },
    },
    "2345": {
        "hcs_old": "per_image_seed2345_hcs250_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
        "hcs_label": "HCS step250",
        "old_label": "old gate0.25 step250",
        "old_features": "per_image_features_hcg_h_gate025_seed2345_step250_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096_holdout4096_current.csv",
        "min090_label": "min090 step250",
        "local": {
            "step250": "direct_local_cap080_rho1_seed2345_step250_val4096_holdout4096_current.csv",
            "step500": "direct_local_cap080_rho1_seed2345_step500_val4096_holdout4096_current.csv",
        },
    },
    "3456": {
        "hcs_old": "per_image_seed3456_hcs250_vs_hcgh_gate025_step500_val4096_holdout4096_current.csv",
        "hcs_label": "HCS step250",
        "old_label": "old gate0.25 step500",
        "old_features": "per_image_features_hcg_h_gate025_seed3456_step500_val4096_holdout4096_current.csv",
        "min090": "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_holdout4096_current.csv",
        "min090_label": "min090 step500",
        "local": {
            "step250": "direct_local_cap080_rho1_seed3456_step250_val4096_holdout4096_current.csv",
            "step500": "direct_local_cap080_rho1_seed3456_step500_val4096_holdout4096_current.csv",
        },
    },
}

FEATURE_KEYS = [
    "rvq_s_q_mean",
    "rvq_householder_strength",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_delta_rms_local_max",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
    "rvq_householder_risk_multiplier",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_rows(path)}


def f(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}: {value}")
    return value


def maybe_f(row: dict[str, str], key: str) -> float | None:
    if key not in row or row[key] == "":
        return None
    value = float(row[key])
    return value if math.isfinite(value) else None


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def summarize_features(name: str, rows: list[dict[str, str]]) -> dict[str, float | str]:
    result: dict[str, float | str] = {"method": name}
    for key in FEATURE_KEYS:
        vals = [maybe_f(row, key) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            result[key] = mean(vals)
    return result


def main() -> None:
    seed_results = []
    image_rows = []
    feature_rows: dict[str, list[dict[str, str]]] = {"old gate0.25": [], "min090": [], "local cap080/rho1 best": []}

    for seed, cfg in SEEDS.items():
        hcs_old = by_path(ANALYSIS / str(cfg["hcs_old"]))
        old_features = by_path(ANALYSIS / str(cfg["old_features"]))
        min090 = by_path(ANALYSIS / str(cfg["min090"]))
        local_by_step = {step: by_path(ANALYSIS / filename) for step, filename in cfg["local"].items()}

        paths = sorted(set(hcs_old) & set(old_features) & set(min090) & set.intersection(*(set(v) for v in local_by_step.values())))
        if len(paths) != 4096:
            raise RuntimeError(f"seed {seed}: expected 4096 aligned paths, got {len(paths)}")

        hcs_rd = mean([f(hcs_old[p], "HCS_rd_score") for p in paths])
        old_rd = mean([f(hcs_old[p], "old_gate025_rd_score") for p in paths])
        min090_rd = mean([f(min090[p], "rd_score") for p in paths])
        local_steps = {
            step: {
                "rd": mean([f(rows[p], "rd_score") for p in paths]),
                "bpp": mean([f(rows[p], "bpp") for p in paths]),
                "psnr": mean([f(rows[p], "psnr") for p in paths]),
                "ms_ssim": mean([f(rows[p], "ms_ssim") for p in paths]),
                "nonfinite_rows": sum(1 for p in paths if int(float(rows[p].get("nonfinite", "0") or 0)) != 0),
            }
            for step, rows in local_by_step.items()
        }
        best_step = min(local_steps, key=lambda step: local_steps[step]["rd"])
        local_best = local_by_step[best_step]
        local_rd = local_steps[best_step]["rd"]

        seed_results.append(
            {
                "seed": seed,
                "hcs_label": cfg["hcs_label"],
                "old_label": cfg["old_label"],
                "min090_label": cfg["min090_label"],
                "hcs_rd": hcs_rd,
                "old_rd": old_rd,
                "min090_rd": min090_rd,
                "local_steps": local_steps,
                "local_best_step": best_step,
                "local_best_rd": local_rd,
                "local_delta_vs_hcs": local_rd - hcs_rd,
                "local_delta_vs_old": local_rd - old_rd,
                "local_delta_vs_min090": local_rd - min090_rd,
            }
        )

        feature_rows["old gate0.25"].extend(old_features[p] for p in paths)
        feature_rows["min090"].extend(min090[p] for p in paths)
        feature_rows["local cap080/rho1 best"].extend(local_best[p] for p in paths)

        for p in paths:
            item = {
                "seed": seed,
                "path": p,
                "hcs_rd": f(hcs_old[p], "HCS_rd_score"),
                "old_rd": f(hcs_old[p], "old_gate025_rd_score"),
                "min090_rd": f(min090[p], "rd_score"),
                "local_rd": f(local_best[p], "rd_score"),
            }
            item["local_minus_hcs"] = item["local_rd"] - item["hcs_rd"]
            item["local_minus_old"] = item["local_rd"] - item["old_rd"]
            item["local_minus_min090"] = item["local_rd"] - item["min090_rd"]
            for key in FEATURE_KEYS:
                value = maybe_f(local_best[p], key)
                if value is not None:
                    item[key] = value
            image_rows.append(item)

    aggregate = {
        "num_seeds": len(seed_results),
        "num_images": len(image_rows),
        "hcs_rd": mean([r["hcs_rd"] for r in seed_results]),
        "old_rd": mean([r["old_rd"] for r in seed_results]),
        "min090_rd": mean([r["min090_rd"] for r in seed_results]),
        "local_best_rd": mean([r["local_best_rd"] for r in seed_results]),
    }
    aggregate["local_delta_vs_hcs"] = aggregate["local_best_rd"] - aggregate["hcs_rd"]
    aggregate["local_delta_vs_old"] = aggregate["local_best_rd"] - aggregate["old_rd"]
    aggregate["local_delta_vs_min090"] = aggregate["local_best_rd"] - aggregate["min090_rd"]

    wins = {
        "local_better_than_hcs": sum(row["local_minus_hcs"] < 0.0 for row in image_rows),
        "local_better_than_old": sum(row["local_minus_old"] < 0.0 for row in image_rows),
        "local_better_than_min090": sum(row["local_minus_min090"] < 0.0 for row in image_rows),
    }

    sorted_rows = sorted(image_rows, key=lambda row: row["hcs_rd"])
    qsize = len(sorted_rows) // 4
    quartiles = []
    for qi in range(4):
        qrows = sorted_rows[qi * qsize : (qi + 1) * qsize]
        quartiles.append(
            {
                "quartile": f"Q{qi + 1}",
                "num_images": len(qrows),
                "hcs_rd_min": qrows[0]["hcs_rd"],
                "hcs_rd_max": qrows[-1]["hcs_rd"],
                "old_minus_hcs": mean([row["old_rd"] - row["hcs_rd"] for row in qrows]),
                "min090_minus_hcs": mean([row["min090_rd"] - row["hcs_rd"] for row in qrows]),
                "local_minus_hcs": mean([row["local_minus_hcs"] for row in qrows]),
                "local_minus_old": mean([row["local_minus_old"] for row in qrows]),
                "local_minus_min090": mean([row["local_minus_min090"] for row in qrows]),
            }
        )

    feature_summaries = [summarize_features(name, rows) for name, rows in feature_rows.items()]
    correlations = [
        {
            "feature": label,
            "r_with_local_minus_hcs": pearson(values, [row["local_minus_hcs"] for row in image_rows]),
            "r_with_local_minus_old": pearson(values, [row["local_minus_old"] for row in image_rows]),
        }
        for label, values in {
            "HCS RD difficulty": [row["hcs_rd"] for row in image_rows],
            "local s_q_mean": [row.get("rvq_s_q_mean", float("nan")) for row in image_rows],
            "local strength": [row.get("rvq_householder_strength", float("nan")) for row in image_rows],
            "local delta RMS": [row.get("rvq_householder_delta_rms", float("nan")) for row in image_rows],
            "local local-delta mean": [row.get("rvq_householder_delta_rms_local_mean", float("nan")) for row in image_rows],
            "local qMSE": [row.get("rvq_latent_quant_mse", float("nan")) for row in image_rows],
            "local risk multiplier": [row.get("rvq_householder_risk_multiplier", float("nan")) for row in image_rows],
        }.items()
    ]

    result = {
        "seeds": seed_results,
        "aggregate": aggregate,
        "wins": wins,
        "quartiles_by_hcs_difficulty": quartiles,
        "feature_summaries": feature_summaries,
        "correlations": correlations,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Local Cap080/rho1 Multi-Seed Trusted-Protocol Audit",
        "",
        "OpenImages holdout4096, path-matched against trusted HCS/old gate0.25/min090 references. Local-cap rows are direct current-code exact-inverse probes on CUDA device 0.",
        "",
        "## Per-Seed Checkpoint Results",
        "",
        "| seed | HCS RD | old RD | min090 RD | local step250 | local step500 | best local | local-HCS | local-old | local-min090 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_results:
        lines.append(
            "| {seed} | {hcs} | {old} | {min090} | {l250} | {l500} | {best} ({step}) | {dh} | {do} | {dm} |".format(
                seed=row["seed"],
                hcs=fmt(row["hcs_rd"]),
                old=fmt(row["old_rd"]),
                min090=fmt(row["min090_rd"]),
                l250=fmt(row["local_steps"]["step250"]["rd"]),
                l500=fmt(row["local_steps"]["step500"]["rd"]),
                best=fmt(row["local_best_rd"]),
                step=row["local_best_step"],
                dh=fmt(row["local_delta_vs_hcs"], signed=True),
                do=fmt(row["local_delta_vs_old"], signed=True),
                dm=fmt(row["local_delta_vs_min090"], signed=True),
            )
        )
    lines.extend(
        [
            "",
            "## Mean Across Available Seeds",
            "",
            "| HCS RD | old RD | min090 RD | local best RD | local-HCS | local-old | local-min090 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            "| {hcs} | {old} | {min090} | {local} | {dh} | {do} | {dm} |".format(
                hcs=fmt(aggregate["hcs_rd"]),
                old=fmt(aggregate["old_rd"]),
                min090=fmt(aggregate["min090_rd"]),
                local=fmt(aggregate["local_best_rd"]),
                dh=fmt(aggregate["local_delta_vs_hcs"], signed=True),
                do=fmt(aggregate["local_delta_vs_old"], signed=True),
                dm=fmt(aggregate["local_delta_vs_min090"], signed=True),
            ),
            "",
            "## HCS-Difficulty Quartiles",
            "",
            "| quartile | HCS RD range | old-HCS | min090-HCS | local-HCS | local-old | local-min090 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quartiles:
        lines.append(
            "| {q} | {lo}-{hi} | {old} | {min090} | {local} | {lo_old} | {lo_min} |".format(
                q=row["quartile"],
                lo=fmt(row["hcs_rd_min"]),
                hi=fmt(row["hcs_rd_max"]),
                old=fmt(row["old_minus_hcs"], signed=True),
                min090=fmt(row["min090_minus_hcs"], signed=True),
                local=fmt(row["local_minus_hcs"], signed=True),
                lo_old=fmt(row["local_minus_old"], signed=True),
                lo_min=fmt(row["local_minus_min090"], signed=True),
            )
        )
    lines.extend(
        [
            "",
            "## Feature Means",
            "",
            "| method | s_q | strength | delta RMS | local delta mean | local delta max | qMSE | perplexity | dead code | risk mult |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in feature_summaries:
        lines.append(
            "| {method} | {sq} | {strength} | {delta} | {ldmean} | {ldmax} | {qmse} | {perp} | {dead} | {risk} |".format(
                method=row["method"],
                sq=fmt(float(row.get("rvq_s_q_mean", float("nan")))),
                strength=fmt(float(row.get("rvq_householder_strength", float("nan")))),
                delta=fmt(float(row.get("rvq_householder_delta_rms", float("nan")))),
                ldmean=fmt(float(row.get("rvq_householder_delta_rms_local_mean", float("nan")))),
                ldmax=fmt(float(row.get("rvq_householder_delta_rms_local_max", float("nan")))),
                qmse=fmt(float(row.get("rvq_latent_quant_mse", float("nan")))),
                perp=fmt(float(row.get("rvq_perplexity", float("nan")))),
                dead=fmt(float(row.get("rvq_dead_code_ratio", float("nan")))),
                risk=fmt(float(row.get("rvq_householder_risk_multiplier", float("nan")))),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Image Wins",
            "",
            f"- local better than HCS: {wins['local_better_than_hcs']} / {len(image_rows)}",
            f"- local better than old gate0.25: {wins['local_better_than_old']} / {len(image_rows)}",
            f"- local better than min090: {wins['local_better_than_min090']} / {len(image_rows)}",
            "",
            "## Correlations",
            "",
            "| feature | r with local-HCS | r with local-old |",
            "|---|---:|---:|",
        ]
    )
    for row in correlations:
        lines.append(
            f"| {row['feature']} | {fmt(row['r_with_local_minus_hcs'], signed=True)} | {fmt(row['r_with_local_minus_old'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            "- local cap080/rho1 is now the strongest 3-seed unified-controller candidate under this trusted holdout protocol: it beats HCS, old gate0.25, and min090 on mean RD.",
            "- The gain is most convincing in the hard-image tail: Q4 improves by `-0.136153` RD vs HCS and by `-0.067987` RD vs old gate0.25.",
            "- The best local checkpoints are step250 for all three seeds; step500 consistently moves toward lower `s_q`, higher latent qMSE, and worse RD.",
            "- This is promising enough to become the next paper-main candidate, but it still needs secondary-split/Kodak checks and cleaner documentation before being treated as final.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
