#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

METRIC_OR_ID = {
    "image", "direction_source", "alpha", "force_ind", "bpp", "payload_bytes", "payload_equal", "payload_len_equal",
    "base_psnr", "active_psnr", "delta_psnr", "base_lpips", "active_lpips", "delta_lpips",
    "base_dists", "active_dists", "delta_dists", "max_decode_diff", "mean_decode_diff", "nonfinite",
    "z_mismatch", "z_total", "y_mismatch", "y_total",
}

DECODER_SAFE_PREFIXES = (
    "z_hat_", "z_index_", "slice0_mean_", "slice1_mean_", "slice2_mean_", "slice3_mean_",
    "slice0_scale_", "slice1_scale_", "slice2_scale_", "slice3_scale_",
)

ENCODER_OR_ACTIVE_PREFIXES = (
    "y_", "slice0_y_norm_", "slice1_y_norm_", "slice2_y_norm_", "slice3_y_norm_",
    "slice0_stage", "slice1_stage", "slice2_stage", "slice3_stage",
    "slice0_avg_", "slice1_avg_", "slice2_avg_", "slice3_avg_",
)


def parse_float(v: str) -> Any:
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [{k: parse_float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def classify_feature(name: str) -> str:
    if name in METRIC_OR_ID:
        return "id_or_metric"
    if name.startswith(DECODER_SAFE_PREFIXES):
        return "decoder_safe_context"
    if name.startswith(ENCODER_OR_ACTIVE_PREFIXES):
        return "encoder_or_active_diagnostic"
    return "other"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for force in sorted({int(r["force_ind"]) for r in rows}):
        s = [r for r in rows if int(r["force_ind"]) == force]
        dd = np.array([float(r["delta_dists"]) for r in s])
        dl = np.array([float(r["delta_lpips"]) for r in s])
        dp = np.array([float(r["delta_psnr"]) for r in s])
        out.append(
            {
                "force_ind": force,
                "images": len(s),
                "dists_positive_labels": int((dd < 0).sum()),
                "lpips_positive_labels": int((dl < 0).sum()),
                "both_positive_labels": int(((dd < 0) & (dl < 0)).sum()),
                "psnr_positive_labels": int((dp > 0).sum()),
                "dists_oracle_delta": float(np.minimum(dd, 0).mean()),
                "lpips_oracle_delta": float(np.minimum(dl, 0).mean()),
                "mean_delta_dists": float(dd.mean()),
                "mean_delta_lpips": float(dl.mean()),
                "mean_delta_psnr": float(dp.mean()),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, default=Path("experiments/analysis/e160_eflic_projected_hcg_kodak24_alpha005.csv"))
    p.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005"))
    args = p.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit("no rows")

    feature_names = sorted(rows[0])
    manifest = [{"feature": f, "class": classify_feature(f)} for f in feature_names]

    label_rows: list[dict[str, Any]] = []
    for r in rows:
        y_mismatch_frac = float(r["y_mismatch"]) / max(1.0, float(r["y_total"]))
        rr = dict(r)
        rr.update(
            {
                "label_dists_active": int(float(r["delta_dists"]) < 0),
                "label_lpips_active": int(float(r["delta_lpips"]) < 0),
                "label_both_active": int(float(r["delta_dists"]) < 0 and float(r["delta_lpips"]) < 0),
                "label_psnr_active": int(float(r["delta_psnr"]) > 0),
                "y_mismatch_frac": y_mismatch_frac,
                "one_bit_bpp_cost_512x768": 1.0 / (512.0 * 768.0),
            }
        )
        label_rows.append(rr)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    manifest_path = args.output_prefix.with_name(args.output_prefix.name + "_feature_manifest.csv")

    fields = sorted({k for r in label_rows for k in r})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(label_rows)
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "class"])
        w.writeheader(); w.writerows(manifest)

    summary = summarize(label_rows)
    decoder_safe = sum(1 for m in manifest if m["class"] == "decoder_safe_context")
    encoder_diag = sum(1 for m in manifest if m["class"] == "encoder_or_active_diagnostic")
    payload = {
        "source_csv": str(args.input_csv),
        "rows": len(label_rows),
        "summary": summary,
        "feature_manifest": str(manifest_path),
        "decoder_safe_context_features": decoder_safe,
        "encoder_or_active_diagnostic_features": encoder_diag,
        "interpretation": "Teacher labels for future reliability controller; not a trained selector.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E161 EF-LIC Projected-HCG Selector Labels",
        "",
        f"Source: `{args.input_csv}`",
        "",
        "This package turns E160 per-image active-vs-baseline deltas into teacher labels for a future reliability controller. It is not a trained selector.",
        "",
        f"Decoder-safe context feature columns: `{decoder_safe}`",
        f"Encoder/active diagnostic feature columns: `{encoder_diag}`",
        "",
        "| force | images | DISTS labels | LPIPS labels | both labels | PSNR labels | DISTS oracle d | LPIPS oracle d | mean dDISTS | mean dLPIPS | mean dPSNR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['force_ind']} | {s['images']} | {s['dists_positive_labels']} | {s['lpips_positive_labels']} | {s['both_positive_labels']} | {s['psnr_positive_labels']} | "
            f"{s['dists_oracle_delta']:+.6f} | {s['lpips_oracle_delta']:+.6f} | {s['mean_delta_dists']:+.6f} | {s['mean_delta_lpips']:+.6f} | {s['mean_delta_psnr']:+.6f} |"
        )
    lines.extend([
        "",
        "Next:",
        "",
        "- Use this as a label source for a tiny reliability-controller probe.",
        "- Keep decoder-safe features separate from encoder/active diagnostics. A no-side-bit controller may only use decoder-safe features; a signaled image-level controller may use richer encoder-side diagnostics with explicit side-bit accounting.",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


if __name__ == "__main__":
    main()
