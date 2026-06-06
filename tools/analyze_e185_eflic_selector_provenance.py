#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_e184_eflic_selector_cv as e184  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005.csv"),
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005_feature_manifest.csv"),
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/analysis/e185_eflic_selector_provenance_dists"),
    )
    p.add_argument("--target", choices=["dists", "lpips", "psnr"], default="dists")
    p.add_argument("--side-bits", type=float, default=1.0)
    p.add_argument("--loocv-feature-topk", type=int, default=32)
    return p.parse_args()


def feature_sets(manifest: dict[str, str]) -> dict[str, tuple[list[str], float, str]]:
    legacy_decoder_safe = [f for f, cls in manifest.items() if cls == "decoder_safe_context"]
    global_predecision = [
        f
        for f in legacy_decoder_safe
        if f.startswith(("z_hat_", "z_index_", "slice0_mean_", "slice0_scale_"))
    ]
    sequential_context = [
        f
        for f in legacy_decoder_safe
        if f.startswith(("z_hat_", "z_index_"))
        or any(f.startswith(f"slice{i}_mean_") or f.startswith(f"slice{i}_scale_") for i in range(4))
    ]
    encoder_diag = [f for f, cls in manifest.items() if cls == "encoder_or_active_diagnostic"]
    return {
        "global_predecision_context": (
            global_predecision,
            0.0,
            "Available before choosing a whole-image active/fallback branch: z context plus slice0 mean/scale.",
        ),
        "sequential_context": (
            sequential_context,
            0.0,
            "Decoder-reproducible during sequential slice decoding, but not necessarily available before a whole-image branch decision.",
        ),
        "legacy_decoder_safe_context": (
            legacy_decoder_safe,
            0.0,
            "Original E161/E184 decoder-safe manifest class; useful for comparison but too broad for whole-image predecision.",
        ),
        "encoder_active_diagnostic": (
            encoder_diag,
            1.0,
            "Encoder/active diagnostics; needs signaling or a learned decoder-side proxy before paper use.",
        ),
    }


def write_outputs(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# E185 EF-LIC Selector Provenance Audit",
        "",
        "This audit tightens E184 by separating when each selector feature is actually available.",
        "",
        f"Target metric: `{args.target}`",
        "",
        "| group | selector | feature set | candidates | branch share | dbpp | dDISTS | dLPIPS | dPSNR | DISTS wins | rule |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['selector']} | {row.get('feature_set', '')} | "
            f"{row.get('candidate_features', '')} | {row['branch_share']:.3f} | "
            f"{row['selected_delta_bpp']:+.6f} | {row['selected_delta_dists']:+.6f} | "
            f"{row['selected_delta_lpips']:+.6f} | {row['selected_delta_psnr']:+.6f} | "
            f"{row['selected_win_dists']}/{row['images']} | {row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Feature provenance:",
            "",
            "- `global_predecision_context`: usable for a no-side-bit whole-image branch decision.",
            "- `sequential_context`: usable only if the codec implements a sequential per-slice decision policy.",
            "- `legacy_decoder_safe_context`: the broader E184 set; kept to expose possible optimistic leakage from later-slice context.",
            "- `encoder_active_diagnostic`: not deployable without a side bit or a decoder-side proxy.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    manifest = e184.read_manifest(args.manifest_csv)
    sets = feature_sets(manifest)

    results: list[dict[str, Any]] = []
    for force in sorted({int(r["force_ind"]) for r in rows}):
        subset = [r for r in rows if int(r["force_ind"]) == force]
        results.extend(
            e184.analyze_group(
                f"force{force}",
                subset,
                sets,
                args.target,
                args.side_bits,
                args.loocv_feature_topk,
            )
        )
    write_outputs(args.output_prefix, results, args)


if __name__ == "__main__":
    main()
