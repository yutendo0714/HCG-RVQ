#!/usr/bin/env python3
"""Build the E152 branch-manifest and SOTA/backbone scouting package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e152_branch_manifest_sota_package"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repo_exists(path: str) -> bool:
    return (ROOT / path).exists()


def main() -> None:
    e151 = read_json(ANALYSIS / "e151_signaled_branch_direct_eval.json")
    smoke = read_json(ANALYSIS / "e152_lowrate_hcs_hcg_signaled_branch_smoke.json")
    repos = [
        {
            "name": "DCAE",
            "path": "third_party/DCAE",
            "paper_role": "dictionary-based entropy model and strong CompressAI-style LIC baseline",
            "implementation_status": "code, train/eval, compress/decompress, pretrained links",
            "plug_in_difficulty": "medium",
            "first_probe": "forward-only adapter around y/y_hat slices before any real bitstream changes",
            "why": "DCAE exposes y=g_a(x), z_hat hyper features, sliced y_hat construction, and g_s(y_hat).",
            "risk": "dictionary entropy path may absorb quantizer effects; compare against DCAE-over-itself.",
        },
        {
            "name": "MambaIC",
            "path": "third_party/MambaIC",
            "paper_role": "strong SSM/context backbone",
            "implementation_status": "code, train/eval scripts, HuggingFace checkpoint link",
            "plug_in_difficulty": "medium-high",
            "first_probe": "dependency/import smoke, then adapter around y_hat_slices_for_gs",
            "why": "MambaIC has a CompressAI-like g_a/h_a/h_s/g_s path and explicit y_hat slice assembly.",
            "risk": "VMamba/selective_scan dependencies can dominate setup; avoid long training before import smoke.",
        },
        {
            "name": "HPCM",
            "path": "third_party/LIC-HPCM",
            "paper_role": "hierarchical progressive context SOTA and RVQ stage-context design reference",
            "implementation_status": "code, train/test, pretrained links, arithmetic coder path",
            "plug_in_difficulty": "high",
            "first_probe": "use as stage-context design reference before modifying forward_hpcm/compress_hpcm",
            "why": "HPCM directly matches future progressive RVQ-stage context but has deep multi-step coding logic.",
            "risk": "raw plug-in would touch forward_hpcm, compress_hpcm, and decompress_hpcm at once.",
        },
        {
            "name": "RDVQ",
            "path": "third_party/RDVQ",
            "paper_role": "closest differentiable VQ-RD optimization competitor",
            "implementation_status": "official repo cloned, but README says code will come soon",
            "plug_in_difficulty": "not executable yet",
            "first_probe": "paper comparison and design reading only until code is released",
            "why": "RDVQ informs entropy-aware VQ training, not hyperprior-generated quantizer geometry.",
            "risk": "waiting for code would stall HCG-RVQ controlled evidence.",
        },
    ]
    for row in repos:
        row["cloned"] = repo_exists(str(row["path"]))

    summary = {
        "experiment": "E152 branch manifest and SOTA/backbone scouting",
        "branch_manifest": "configs/e152_lowrate_hcs_hcg_signaled_branch_manifest.yaml",
        "branch_smoke": {
            "num_rows": smoke["num_rows"],
            "base_rd": smoke["base_rd"],
            "candidate_rd": smoke["candidate_rd"],
            "branch_rd": smoke["branch_rd"],
            "branch_minus_base": smoke["branch_minus_base"],
            "branch_minus_candidate": smoke["branch_minus_candidate"],
            "selected_frac": smoke["selected_frac"],
            "nonfinite_rows": smoke["nonfinite_rows"],
        },
        "full_e151": {
            "num_rows": e151["num_rows"],
            "hcs_rd": e151["hcs_rd"],
            "hcg_rd": e151["hcg_rd"],
            "branch_rd": e151["branch_rd"],
            "branch_signaled_rd": e151["branch_signaled_rd"],
            "branch_minus_hcs": e151["branch_minus_hcs"],
            "branch_minus_hcg": e151["branch_minus_hcg"],
            "selected_frac": e151["selected_frac"],
            "nonfinite_rows": e151["nonfinite_rows"],
        },
        "sota_repos": repos,
        "decision": (
            "Run controlled-evidence and method-strengthening in parallel. "
            "The next SOTA/backbone probe should port the state-preserving branch interface, "
            "not raw continuous HCG gate shrinkage."
        ),
        "next_actions": [
            "Use the E152 branch manifest for any new split or rate-point branch evaluation.",
            "Prepare lambda0067 HCS/HCG active-geometry configs after scalar initialization is available.",
            "Start SOTA plug-in with DCAE forward-only adapter smoke because it has the cleanest y/y_hat slice boundary.",
            "Use MambaIC after dependency/import smoke; use HPCM first as stage-context design guidance.",
            "Keep RDVQ as a primary paper comparison, but do not block on its unreleased code.",
        ],
    }

    PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(PREFIX.with_suffix(".sota_repos.csv"), repos)
    write_csv(PREFIX.with_suffix(".next_actions.csv"), [{"action": action} for action in summary["next_actions"]])

    lines = [
        "# E152 Branch Manifest and SOTA/Backbone Scouting",
        "",
        "## Branch Manifest",
        "",
        f"- Manifest: `{summary['branch_manifest']}`",
        f"- Smoke rows: `{summary['branch_smoke']['num_rows']}`",
        f"- Smoke branch RD: `{summary['branch_smoke']['branch_rd']:.6f}` ({summary['branch_smoke']['branch_minus_base']:+.6f} vs base)",
        f"- Smoke nonfinite rows: `{summary['branch_smoke']['nonfinite_rows']}`",
        "",
        "Full E151 is still the main low-rate branch evidence:",
        "",
        f"- HCS RD: `{summary['full_e151']['hcs_rd']:.6f}`",
        f"- fixed HCG RD: `{summary['full_e151']['hcg_rd']:.6f}`",
        f"- branch RD: `{summary['full_e151']['branch_rd']:.6f}` ({summary['full_e151']['branch_minus_hcs']:+.6f} vs HCS, {summary['full_e151']['branch_minus_hcg']:+.6f} vs fixed HCG)",
        f"- signaled branch RD: `{summary['full_e151']['branch_signaled_rd']:.6f}`",
        f"- nonfinite rows: `{summary['full_e151']['nonfinite_rows']}`",
        "",
        "## SOTA/Backbone Scout",
        "",
        "| repo | cloned | difficulty | first probe | risk |",
        "|---|---:|---|---|---|",
    ]
    for row in repos:
        lines.append(
            f"| {row['name']} | {row['cloned']} | {row['plug_in_difficulty']} | {row['first_probe']} | {row['risk']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(summary["decision"]),
            "",
            "## Next Actions",
            "",
        ]
    )
    lines.extend(f"- {action}" for action in summary["next_actions"])
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
