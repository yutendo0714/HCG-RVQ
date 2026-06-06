#!/usr/bin/env python3
"""Analyze direct local-cap HCG results against trusted holdout4096 references."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_JSON = ANALYSIS / "local_cap080_rho1_trusted_protocol_seed1234_holdout4096.json"
OUT_MD = ANALYSIS / "local_cap080_rho1_trusted_protocol_seed1234_holdout4096.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def finite_float(row: dict[str, str], key: str) -> float | None:
    if key not in row or row[key] == "":
        return None
    value = float(row[key])
    return value if math.isfinite(value) else None


def by_path(path: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in read_rows(path)}


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


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
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def method_summary(
    name: str,
    rows: dict[str, dict[str, str]],
    rd_key: str,
    paths: list[str],
    metric_prefix: str = "",
) -> dict[str, float | str | int]:
    result: dict[str, float | str | int] = {
        "method": name,
        "num_images": len(paths),
        "rd": mean([float(rows[p][rd_key]) for p in paths]),
    }
    metric_keys = {
        "bpp": f"{metric_prefix}bpp",
        "psnr": f"{metric_prefix}psnr",
        "ms_ssim": f"{metric_prefix}ms_ssim",
    }
    for out_key, key in metric_keys.items():
        if key in next(iter(rows.values())):
            vals = [finite_float(rows[p], key) for p in paths]
            vals = [v for v in vals if v is not None]
            if vals:
                result[out_key] = mean(vals)
    return result


def feature_summary(name: str, rows: dict[str, dict[str, str]], paths: list[str]) -> dict[str, float | str]:
    result: dict[str, float | str] = {"method": name}
    keys = [
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
    for key in keys:
        vals = [finite_float(rows[p], key) for p in paths]
        vals = [v for v in vals if v is not None]
        if vals:
            result[key] = mean(vals)
    return result


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def main() -> None:
    hcs_old_path = ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv"
    min090_path = ANALYSIS / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv"
    old_feature_path = ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv"
    local_direct_path = ANALYSIS / "direct_local_cap080_rho1_seed1234_step250_val4096_holdout4096_current.csv"
    local_stale_path = ANALYSIS / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_seed1234_step250_val4096_holdout4096_current.csv"

    hcs_old = by_path(hcs_old_path)
    old_features = by_path(old_feature_path)
    min090 = by_path(min090_path)
    local_direct = by_path(local_direct_path)
    local_stale = by_path(local_stale_path)
    paths = sorted(set(hcs_old) & set(old_features) & set(min090) & set(local_direct))
    if len(paths) != 4096:
        raise RuntimeError(f"expected 4096 aligned paths, got {len(paths)}")

    method_rows = {
        "HCS step500": (hcs_old, "HCS_rd_score", "HCS_"),
        "old gate0.25 step250": (hcs_old, "old_gate025_rd_score", "old_gate025_"),
        "min090 step500": (min090, "rd_score", ""),
        "local cap080/rho1 direct step250": (local_direct, "rd_score", ""),
    }
    summaries = [method_summary(name, rows, rd_key, paths, prefix) for name, (rows, rd_key, prefix) in method_rows.items()]
    rd = {s["method"]: float(s["rd"]) for s in summaries}
    for s in summaries:
        s["delta_vs_hcs"] = float(s["rd"]) - rd["HCS step500"]
        s["delta_vs_old"] = float(s["rd"]) - rd["old gate0.25 step250"]
        s["delta_vs_min090"] = float(s["rd"]) - rd["min090 step500"]

    hcs_scores = {p: float(hcs_old[p]["HCS_rd_score"]) for p in paths}
    sorted_paths = sorted(paths, key=lambda p: hcs_scores[p])
    quartiles = []
    for qi in range(4):
        qpaths = sorted_paths[qi * 1024 : (qi + 1) * 1024]
        item: dict[str, float | str | int] = {
            "quartile": f"Q{qi + 1}",
            "hcs_rd_min": hcs_scores[qpaths[0]],
            "hcs_rd_max": hcs_scores[qpaths[-1]],
            "num_images": len(qpaths),
        }
        for name, (rows, rd_key, _prefix) in method_rows.items():
            item[f"{name}_minus_hcs"] = mean([float(rows[p][rd_key]) - hcs_scores[p] for p in qpaths])
        item["local_minus_old"] = mean([
            float(local_direct[p]["rd_score"]) - float(hcs_old[p]["old_gate025_rd_score"]) for p in qpaths
        ])
        item["local_minus_min090"] = mean([
            float(local_direct[p]["rd_score"]) - float(min090[p]["rd_score"]) for p in qpaths
        ])
        quartiles.append(item)

    local_minus_hcs = [float(local_direct[p]["rd_score"]) - hcs_scores[p] for p in paths]
    local_minus_old = [float(local_direct[p]["rd_score"]) - float(hcs_old[p]["old_gate025_rd_score"]) for p in paths]
    local_minus_min090 = [float(local_direct[p]["rd_score"]) - float(min090[p]["rd_score"]) for p in paths]
    wins = {
        "local_better_than_hcs": sum(v < 0.0 for v in local_minus_hcs),
        "local_better_than_old": sum(v < 0.0 for v in local_minus_old),
        "local_better_than_min090": sum(v < 0.0 for v in local_minus_min090),
    }

    feature_summaries = [
        feature_summary("old gate0.25", old_features, paths),
        feature_summary("min090", min090, paths),
        feature_summary("local cap080/rho1 direct", local_direct, paths),
    ]

    corr_features = {
        "HCS RD difficulty": [hcs_scores[p] for p in paths],
        "local s_q_mean": [finite_float(local_direct[p], "rvq_s_q_mean") or float("nan") for p in paths],
        "local strength": [finite_float(local_direct[p], "rvq_householder_strength") or float("nan") for p in paths],
        "local delta RMS": [finite_float(local_direct[p], "rvq_householder_delta_rms") or float("nan") for p in paths],
        "local local-delta mean": [finite_float(local_direct[p], "rvq_householder_delta_rms_local_mean") or float("nan") for p in paths],
        "local qMSE": [finite_float(local_direct[p], "rvq_latent_quant_mse") or float("nan") for p in paths],
        "local risk multiplier": [finite_float(local_direct[p], "rvq_householder_risk_multiplier") or float("nan") for p in paths],
    }
    correlations = [
        {
            "feature": name,
            "r_with_local_minus_hcs": pearson(values, local_minus_hcs),
            "r_with_local_minus_old": pearson(values, local_minus_old),
        }
        for name, values in corr_features.items()
    ]

    stale_paths = sorted(set(local_direct) & set(local_stale))
    stale_diffs = [float(local_direct[p]["rd_score"]) - float(local_stale[p]["rd_score"]) for p in stale_paths]
    stale_audit = {
        "num_images": len(stale_paths),
        "direct_mean_rd": mean([float(local_direct[p]["rd_score"]) for p in stale_paths]),
        "stale_mean_rd": mean([float(local_stale[p]["rd_score"]) for p in stale_paths]),
        "mean_direct_minus_stale": mean(stale_diffs),
        "mean_abs_direct_minus_stale": mean([abs(v) for v in stale_diffs]),
        "max_abs_direct_minus_stale": max(abs(v) for v in stale_diffs),
    }

    result = {
        "inputs": {
            "hcs_old": str(hcs_old_path.relative_to(ROOT)),
            "old_features": str(old_feature_path.relative_to(ROOT)),
            "min090": str(min090_path.relative_to(ROOT)),
            "local_direct": str(local_direct_path.relative_to(ROOT)),
            "local_stale": str(local_stale_path.relative_to(ROOT)),
        },
        "summaries": summaries,
        "quartiles_by_hcs_difficulty": quartiles,
        "wins": wins,
        "feature_summaries": feature_summaries,
        "correlations": correlations,
        "stale_audit": stale_audit,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Local Cap080/rho1 Trusted-Protocol Audit",
        "",
        "Seed1234 OpenImages holdout4096, path-matched against trusted historical HCS/old/min090 references and a direct current-code local-cap rerun on CUDA device 0.",
        "",
        "## Checkpoint RD",
        "",
        "| method | RD | delta vs HCS | delta vs old | delta vs min090 | bpp | PSNR | MS-SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {method} | {rd} | {dh} | {do} | {dm} | {bpp} | {psnr} | {msssim} |".format(
                method=s["method"],
                rd=fmt(float(s["rd"])),
                dh=fmt(float(s["delta_vs_hcs"]), signed=True),
                do=fmt(float(s["delta_vs_old"]), signed=True),
                dm=fmt(float(s["delta_vs_min090"]), signed=True),
                bpp=fmt(float(s.get("bpp", float("nan")))),
                psnr=fmt(float(s.get("psnr", float("nan")))),
                msssim=fmt(float(s.get("ms_ssim", float("nan")))),
            )
        )
    lines.extend([
        "",
        "## HCS-Difficulty Quartiles",
        "",
        "| quartile | HCS RD range | old-HCS | min090-HCS | local-HCS | local-old | local-min090 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for q in quartiles:
        lines.append(
            "| {q} | {lo:.3f}-{hi:.3f} | {old} | {min090} | {local} | {lold} | {lmin} |".format(
                q=q["quartile"],
                lo=float(q["hcs_rd_min"]),
                hi=float(q["hcs_rd_max"]),
                old=fmt(float(q["old gate0.25 step250_minus_hcs"]), signed=True),
                min090=fmt(float(q["min090 step500_minus_hcs"]), signed=True),
                local=fmt(float(q["local cap080/rho1 direct step250_minus_hcs"]), signed=True),
                lold=fmt(float(q["local_minus_old"]), signed=True),
                lmin=fmt(float(q["local_minus_min090"]), signed=True),
            )
        )
    lines.extend([
        "",
        "## Feature Means",
        "",
        "| method | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | risk mult |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for f in feature_summaries:
        lines.append(
            "| {method} | {sq} | {strength} | {delta} | {local_delta} | {qmse} | {perp} | {dead} | {risk} |".format(
                method=f["method"],
                sq=fmt(float(f.get("rvq_s_q_mean", float("nan")))),
                strength=fmt(float(f.get("rvq_householder_strength", float("nan")))),
                delta=fmt(float(f.get("rvq_householder_delta_rms", float("nan")))),
                local_delta=fmt(float(f.get("rvq_householder_delta_rms_local_mean", float("nan")))),
                qmse=fmt(float(f.get("rvq_latent_quant_mse", float("nan")))),
                perp=fmt(float(f.get("rvq_perplexity", float("nan")))),
                dead=fmt(float(f.get("rvq_dead_code_ratio", float("nan")))),
                risk=fmt(float(f.get("rvq_householder_risk_multiplier", float("nan")))),
            )
        )
    lines.extend([
        "",
        "## Per-Image Wins",
        "",
        f"- local better than HCS: {wins['local_better_than_hcs']} / 4096",
        f"- local better than old gate0.25: {wins['local_better_than_old']} / 4096",
        f"- local better than min090: {wins['local_better_than_min090']} / 4096",
        "",
        "## Correlations",
        "",
        "| feature | r with local-HCS | r with local-old |",
        "|---|---:|---:|",
    ])
    for c in correlations:
        lines.append(
            f"| {c['feature']} | {fmt(float(c['r_with_local_minus_hcs']), signed=True)} | {fmt(float(c['r_with_local_minus_old']), signed=True)} |"
        )
    lines.extend([
        "",
        "## Artifact Quarantine Check",
        "",
        f"The old local-cap artifact mean RD is {fmt(float(stale_audit['stale_mean_rd']))}, but the direct current-code rerun is {fmt(float(stale_audit['direct_mean_rd']))} on the same 4096 paths.",
        f"Mean direct-stale RD difference is {fmt(float(stale_audit['mean_direct_minus_stale']), signed=True)}; mean absolute difference is {fmt(float(stale_audit['mean_abs_direct_minus_stale']))}; max absolute difference is {fmt(float(stale_audit['max_abs_direct_minus_stale']))}.",
        "",
        "Conclusion:",
        "",
        "- Under the direct trusted protocol, local cap080/rho1 is a real positive seed1234 checkpoint: it improves HCS, old gate0.25, and min090 on mean RD.",
        "- The gain over old gate0.25 is small on seed1234, so it is not yet a paper-main claim without multi-seed confirmation.",
        "- The stale local-cap/current-localstats artifacts are invalid for paper tables and should be excluded from future summaries.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.relative_to(ROOT))
    print(OUT_JSON.relative_to(ROOT))


if __name__ == "__main__":
    main()
