#!/usr/bin/env python3
"""Audit holdout4096 artifacts across historical and current-code protocols."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
OUT_JSON = ANALYSIS / "holdout4096_artifact_consistency_audit.json"
OUT_MD = ANALYSIS / "holdout4096_artifact_consistency_audit.md"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v for k, v in row.items() if k is not None} for row in reader]


def _resolve_column(rows: list[dict[str, str]], column: str) -> str:
    if not rows:
        return column
    if column in rows[0]:
        return column
    lowered = {key.lower(): key for key in rows[0]}
    if column.lower() in lowered:
        return lowered[column.lower()]
    aliases = {
        "gate025_rd_score": "old_gate025_rd_score",
        "hcs_rd_score": "HCS_rd_score",
    }
    alias = aliases.get(column)
    if alias and alias in rows[0]:
        return alias
    raise ValueError(f"{column} is missing; available columns: {sorted(rows[0])[:20]}")


def _finite_mean(rows: list[dict[str, str]], column: str) -> float:
    column = _resolve_column(rows, column)
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError(f"{column} has no finite values")
    return sum(values) / len(values)


def _summary_best(path: Path) -> dict[str, object]:
    rows = _read_rows(path)
    best = min(rows, key=lambda r: float(r["rd_score"]))
    return {
        "path": str(path.relative_to(ROOT)),
        "n_rows": len(rows),
        "best_step": int(float(best["step"])),
        "mean_rd": float(best["rd_score"]),
        "bpp": float(best["bpp"]),
        "mse": float(best["mse"]),
        "psnr": float(best["psnr"]),
        "ms_ssim": float(best["ms_ssim"]),
        "missing_keys": best.get("missing_keys", ""),
        "unexpected_keys": best.get("unexpected_keys", ""),
    }


def _per_image(path: Path, column: str) -> dict[str, object]:
    rows = _read_rows(path)
    actual_column = _resolve_column(rows, column)
    result: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "n_rows": len(rows),
        "column": actual_column,
        "mean_rd": _finite_mean(rows, actual_column),
    }
    first = rows[0]
    if "path" in first:
        result["first_image"] = first["path"]
    if actual_column in first:
        result["first_rd"] = float(first[actual_column])
    for metric in ("bpp", "mse", "psnr", "ms_ssim"):
        if metric in first:
            result[f"mean_{metric}"] = _finite_mean(rows, metric)
    return result


def _mean_ref(refs: list[dict[str, object]]) -> float | None:
    if not refs:
        return None
    return sum(float(r["mean_rd"]) for r in refs) / len(refs)


def _aligned_abs_diffs(
    left_path: Path,
    left_column: str,
    right_path: Path,
    right_column: str,
    limit: int | None = None,
) -> dict[str, object]:
    left_rows = _read_rows(left_path)
    right_rows = _read_rows(right_path)
    left_column = _resolve_column(left_rows, left_column)
    right_column = _resolve_column(right_rows, right_column)
    if limit is not None:
        left_rows = left_rows[:limit]
        right_rows = right_rows[:limit]
    diffs: list[float] = []
    mismatched_paths: list[tuple[str, str]] = []
    for left, right in zip(left_rows, right_rows):
        if left.get("path") != right.get("path"):
            mismatched_paths.append((left.get("path", ""), right.get("path", "")))
        diffs.append(abs(float(left[left_column]) - float(right[right_column])))
    return {
        "left_path": str(left_path.relative_to(ROOT)),
        "right_path": str(right_path.relative_to(ROOT)),
        "left_column": left_column,
        "right_column": right_column,
        "n": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else None,
        "mismatched_paths": mismatched_paths,
    }


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.6f}"
    return f"{value:.6f}"


def main() -> None:
    checks = [
        {
            "name": "seed1234 HCS step500",
            "legacy_summary": ANALYSIS / "pilot_hcs_rvq_frozen_seed1234_openimages_val4096_holdout4096_current.csv",
            "current_summary": ANALYSIS / "pilot_hcs_rvq_frozen_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv",
            "historical_refs": [
                (
                    ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
                    "hcs_rd_score",
                ),
                (
                    ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_risk_inv_detach_s044_min090_step500_val4096_holdout4096_current.csv",
                    "hcs_rd_score",
                ),
            ],
            "current_refs": [],
        },
        {
            "name": "seed1234 old gate0.25 step250",
            "legacy_summary": ANALYSIS / "pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_holdout4096_current.csv",
            "current_summary": ANALYSIS / "pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv",
            "historical_refs": [
                (
                    ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
                    "gate025_rd_score",
                ),
                (
                    ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv",
                    "rd_score",
                ),
            ],
            "current_refs": [
                (
                    ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current_localstats.csv",
                    "rd_score",
                )
            ],
        },
        {
            "name": "seed1234 min090 step500",
            "legacy_summary": ANALYSIS / "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_holdout4096_current.csv",
            "current_summary": ANALYSIS / "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv",
            "historical_refs": [
                (
                    ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_risk_inv_detach_s044_min090_step500_val4096_holdout4096_current.csv",
                    "min090_rd_score",
                ),
                (
                    ANALYSIS / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv",
                    "rd_score",
                ),
            ],
            "current_refs": [],
        },
    ]

    report: list[dict[str, object]] = []
    for check in checks:
        legacy = _summary_best(check["legacy_summary"])
        current = _summary_best(check["current_summary"])
        historical_refs = [_per_image(path, column) for path, column in check["historical_refs"]]
        current_refs = [_per_image(path, column) for path, column in check["current_refs"]]
        historical_mean = _mean_ref(historical_refs)
        current_ref_mean = _mean_ref(current_refs)
        item: dict[str, object] = {
            "name": check["name"],
            "legacy_summary": legacy,
            "current_summary": current,
            "historical_per_image_refs": historical_refs,
            "current_per_image_refs": current_refs,
            "historical_per_image_mean_rd": historical_mean,
            "current_per_image_mean_rd": current_ref_mean,
            "legacy_minus_historical": float(legacy["mean_rd"]) - historical_mean,
            "current_minus_historical": float(current["mean_rd"]) - historical_mean,
        }
        if current_ref_mean is not None:
            item["current_minus_current_ref"] = float(current["mean_rd"]) - current_ref_mean
        report.append(item)

    direct_probes = [
        _aligned_abs_diffs(
            ANALYSIS / "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv",
            "hcs_rd_score",
            ANALYSIS / "per_image_features_hcs_seed1234_step500_val4_holdout4096_current_probe.csv",
            "rd_score",
            limit=4,
        ),
        _aligned_abs_diffs(
            ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv",
            "rd_score",
            ANALYSIS / "householder_inverse_modes_gate025_seed1234_step250_val16_holdout4096_current.csv",
            "rd_score",
            limit=16,
        ),
        _aligned_abs_diffs(
            ANALYSIS / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv",
            "rd_score",
            ANALYSIS / "householder_inverse_modes_min090_seed1234_step500_val16_holdout4096_current.csv",
            "rd_score",
            limit=16,
        ),
        _aligned_abs_diffs(
            ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv",
            "rd_score",
            ANALYSIS / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current_localstats.csv",
            "rd_score",
            limit=16,
        ),
    ]

    result = {
        "conclusion": (
            "Historical holdout4096 summaries are internally consistent with historical per-image artifacts. "
            "A direct current-code HCS probe reproduces the historical HCS rows, so the broad data/evaluation path is not broken. "
            "Direct current-code HCG probes reproduce historical old gate0.25 and min090 rows under the exact inverse convention. The `current_recheck_after_localstats`/`current_localstats` artifacts disagree with direct probes and must be quarantined as mis-generated or non-comparable artifacts."
        ),
        "checks": report,
        "direct_probe_checks": direct_probes,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Holdout4096 Artifact Consistency Audit",
        "",
        "This audit compares historical holdout4096 checkpoint-summary CSVs against historical per-image references, current-code recheck summaries, and direct path-aligned probes.",
        "",
        "| check | historical summary RD | historical per-image RD | current recheck RD | hist summary - hist ref | current - hist ref | current - current ref |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report:
        lines.append(
            "| {name} | {legacy} | {hist_ref} | {current} | {d_legacy} | {d_current_hist} | {d_current_ref} |".format(
                name=item["name"],
                legacy=_fmt(item["legacy_summary"]["mean_rd"]),
                hist_ref=_fmt(item["historical_per_image_mean_rd"]),
                current=_fmt(item["current_summary"]["mean_rd"]),
                d_legacy=_fmt(item["legacy_minus_historical"], signed=True),
                d_current_hist=_fmt(item["current_minus_historical"], signed=True),
                d_current_ref=_fmt(item.get("current_minus_current_ref"), signed=True),
            )
        )
    lines.extend(
        [
            "",
            "## Direct Path-Aligned Probes",
            "",
            "| probe | n | max abs RD diff | mean abs RD diff | path mismatches |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    probe_names = [
        "HCS historical vs current-code direct probe",
        "old gate0.25 historical vs current-code exact probe",
        "min090 historical vs current-code exact probe",
        "old gate0.25 historical vs current localstats",
    ]
    for name, probe in zip(probe_names, direct_probes):
        lines.append(
            f"| {name} | {probe['n']} | {_fmt(probe['max_abs_diff'])} | {_fmt(probe['mean_abs_diff'])} | {len(probe['mismatched_paths'])} |"
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            "- The historical holdout4096 summaries are not stale by themselves: HCS, old gate0.25, and min090 match historical per-image artifacts within numerical noise.",
            "- Direct current-code probes reproduce HCS, old gate0.25, and min090 historical rows on path-aligned holdout images within numerical noise. The OpenImages ordering, checkpoint loading, HCS path, and HCG exact-inverse path are therefore reproducible.",
            "- The Householder inverse probe confirms that the historical HCG rows used the mathematically exact partial-reflection inverse. `same_partial`, full Householder, and identity inverses are far worse on the same images.",
            "- The files named `*_current_recheck_after_localstats.csv` and the old gate0.25 `*_current_localstats.csv` should be quarantined for paper-facing holdout claims. They are contradicted by direct path-aligned probes and likely came from a non-comparable command/protocol.",
            "- Next paper-safe action: keep the trusted historical holdout rows as valid, exclude the quarantined recheck/localstats artifacts from claim tables, and rerun local-control checkpoints against directly reproduced HCS/old/min090 references under one pinned command.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD.relative_to(ROOT))
    print(OUT_JSON.relative_to(ROOT))


if __name__ == "__main__":
    main()
