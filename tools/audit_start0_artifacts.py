import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactCheck:
    family: str
    seed: int
    step: int
    legacy_summary: str
    reference_json: str
    reference_csv: str | None = None


CHECKS = [
    ArtifactCheck(
        "hcs",
        1234,
        500,
        "pilot_hcs_rvq_frozen_seed1234_openimages_val4096_reeval_current.csv",
        "per_image_features_hcs_seed1234_step500_val4096_start0_current_recheck.json",
        "debug_hcs_seed1234_step500_start0_current_recheck.csv",
    ),
    ArtifactCheck(
        "hcs",
        2345,
        250,
        "pilot_hcs_rvq_frozen_seed2345_openimages_val4096_reeval_current.csv",
        "per_image_features_hcs_seed2345_step250_val4096_start0_current_recheck.json",
        "debug_hcs_seed2345_step250_start0_current_recheck.csv",
    ),
    ArtifactCheck(
        "hcs",
        3456,
        250,
        "pilot_hcs_rvq_frozen_seed3456_openimages_val4096_reeval_current.csv",
        "per_image_features_hcs_seed3456_step250_val4096_start0_current_recheck.json",
        "debug_hcs_seed3456_step250_start0_current_recheck.csv",
    ),
    ArtifactCheck(
        "old_gate025",
        1234,
        250,
        "pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_seed1234_step250_val4096_reeval_current.json",
        "debug_gate025_seed1234_step250_start0_current_recheck.csv",
    ),
    ArtifactCheck(
        "old_gate025",
        2345,
        250,
        "pilot_hcg_rvq_h_gate025_seed2345_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_seed2345_step250_val4096_reeval_current.json",
    ),
    ArtifactCheck(
        "old_gate025",
        3456,
        250,
        "pilot_hcg_rvq_h_gate025_seed3456_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_seed3456_step250_val4096_reeval_current.json",
    ),
    ArtifactCheck(
        "risk_inv_detach_min090",
        1234,
        500,
        "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_reeval_current.json",
    ),
    ArtifactCheck(
        "risk_inv_detach_min090",
        2345,
        250,
        "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed2345_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096_reeval_current.json",
    ),
    ArtifactCheck(
        "risk_inv_detach_min090",
        3456,
        250,
        "pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed3456_openimages_val4096_reeval_current.csv",
        "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step250_val4096_reeval_current.json",
    ),
]


TRUSTED_EXTRA_REFERENCES = [
    "per_image_features_hcg_h_gate025_seed3456_step500_val4096_start0_current_recheck.json",
    "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_start0_current_recheck.json",
    "gate025_min090_selector_start0_current_recheck_transfer.json",
    "gate025_min090_selector_start0_current_recheck_slice_best.json",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary_rd(path: Path, step: int) -> float:
    rows = read_csv_rows(path)
    matches = [row for row in rows if int(row["step"]) == step]
    if not matches:
        raise ValueError(f"{path} has no step {step}")
    if len(matches) > 1:
        raise ValueError(f"{path} has multiple rows for step {step}")
    return float(matches[0]["rd_score"])


def read_reference_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_reference_rd(path: Path) -> float:
    payload = read_reference_json(path)
    return float(payload["rd_score_mean"])


def is_finite_tree(value: Any, path: str = "$") -> list[str]:
    problems: list[str] = []
    if isinstance(value, float):
        if not math.isfinite(value):
            problems.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            problems.extend(is_finite_tree(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems.extend(is_finite_tree(item, f"{path}[{index}]"))
    return problems


def audit(analysis_dir: Path, tolerance: float) -> dict[str, Any]:
    records = []
    reference_checks = []
    nonfinite_files = {}

    for item in CHECKS:
        legacy_path = analysis_dir / item.legacy_summary
        reference_path = analysis_dir / item.reference_json
        legacy_rd = read_summary_rd(legacy_path, item.step)
        reference_payload = read_reference_json(reference_path)
        reference_rd = float(reference_payload["rd_score_mean"])
        delta = legacy_rd - reference_rd
        status = "trusted" if abs(delta) <= tolerance else "exclude_legacy_summary"
        records.append(
            {
                "family": item.family,
                "seed": item.seed,
                "step": item.step,
                "legacy_summary": item.legacy_summary,
                "reference_json": item.reference_json,
                "legacy_rd": legacy_rd,
                "reference_rd": reference_rd,
                "legacy_minus_reference": delta,
                "status": status,
            }
        )

        problems = is_finite_tree(reference_payload)
        if problems:
            nonfinite_files[item.reference_json] = problems

        if item.reference_csv:
            reference_csv_path = analysis_dir / item.reference_csv
            reference_csv_rd = read_summary_rd(reference_csv_path, item.step)
            csv_delta = reference_csv_rd - reference_rd
            reference_checks.append(
                {
                    "family": item.family,
                    "seed": item.seed,
                    "step": item.step,
                    "reference_csv": item.reference_csv,
                    "reference_json": item.reference_json,
                    "reference_csv_rd": reference_csv_rd,
                    "reference_json_rd": reference_rd,
                    "reference_csv_minus_json": csv_delta,
                    "status": "trusted" if abs(csv_delta) <= tolerance else "mismatch",
                }
            )

    for name in TRUSTED_EXTRA_REFERENCES:
        path = analysis_dir / name
        if not path.exists():
            nonfinite_files[name] = ["missing"]
            continue
        payload = read_reference_json(path)
        problems = is_finite_tree(payload)
        if problems:
            nonfinite_files[name] = problems

    family_summary = {}
    for family in sorted({record["family"] for record in records}):
        group = [record for record in records if record["family"] == family]
        family_summary[family] = {
            "checked": len(group),
            "excluded_legacy_summary": sum(record["status"] == "exclude_legacy_summary" for record in group),
            "legacy_rd_mean": sum(record["legacy_rd"] for record in group) / len(group),
            "reference_rd_mean": sum(record["reference_rd"] for record in group) / len(group),
            "legacy_minus_reference_mean": sum(record["legacy_minus_reference"] for record in group) / len(group),
        }

    return {
        "tolerance": tolerance,
        "legacy_summary_checks": records,
        "reference_csv_json_checks": reference_checks,
        "family_summary": family_summary,
        "nonfinite_files": nonfinite_files,
        "trusted_extra_references": TRUSTED_EXTRA_REFERENCES,
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Start0 Artifact Consistency Audit",
        "",
        "This audit compares legacy checkpoint-summary CSV rows against the current per-image/debug start0 references.",
        "Rows marked `exclude_legacy_summary` should not be used for start0 paper claims.",
        "",
        "## Family Summary",
        "",
        "| family | checked | excluded legacy | legacy RD mean | reference RD mean | legacy-reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, summary in payload["family_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(summary["checked"]),
                    str(summary["excluded_legacy_summary"]),
                    fmt(summary["legacy_rd_mean"]),
                    fmt(summary["reference_rd_mean"]),
                    fmt(summary["legacy_minus_reference_mean"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Legacy Summary Checks",
            "",
            "| family | seed | step | legacy RD | reference RD | legacy-reference | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in payload["legacy_summary_checks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["family"],
                    str(record["seed"]),
                    str(record["step"]),
                    fmt(record["legacy_rd"]),
                    fmt(record["reference_rd"]),
                    fmt(record["legacy_minus_reference"]),
                    record["status"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reference CSV/JSON Checks",
            "",
            "| family | seed | step | csv RD | json RD | csv-json | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in payload["reference_csv_json_checks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["family"],
                    str(record["seed"]),
                    str(record["step"]),
                    fmt(record["reference_csv_rd"]),
                    fmt(record["reference_json_rd"]),
                    fmt(record["reference_csv_minus_json"]),
                    record["status"],
                ]
            )
            + " |"
        )

    nonfinite_files = payload["nonfinite_files"]
    lines.extend(["", "## Nonfinite Check", ""])
    if nonfinite_files:
        lines.append("Non-finite values or missing trusted references were found:")
        for name, problems in nonfinite_files.items():
            lines.append(f"- `{name}`: {', '.join(problems[:8])}")
    else:
        lines.append("All checked trusted JSON references contain finite numeric values.")

    lines.extend(
        [
            "",
            "## Trusted Extra References",
            "",
        ]
    )
    for name in payload["trusted_extra_references"]:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--output-prefix", default="start0_artifact_consistency_audit")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    payload = audit(analysis_dir, args.tolerance)

    json_path = analysis_dir / f"{args.output_prefix}.json"
    md_path = analysis_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
