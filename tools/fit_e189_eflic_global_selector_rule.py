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
import analyze_e185_eflic_selector_provenance as e185  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="E161-style selector label CSV produced from an EF-LIC active-branch evaluation.",
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="Feature manifest emitted next to the E161-style label CSV.",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix for the fitted selector rule.",
    )
    p.add_argument("--target", choices=["dists", "lpips", "psnr"], default="lpips")
    p.add_argument("--force", type=int, default=0)
    p.add_argument(
        "--feature-set",
        choices=["global_predecision_context", "sequential_context", "legacy_decoder_safe_context"],
        default="global_predecision_context",
    )
    p.add_argument(
        "--eval-dir-placeholder",
        default="PATH_TO_HELDOUT_IMAGES",
        help="Placeholder shown in the generated direct-eval command.",
    )
    return p.parse_args()


def write_outputs(prefix: Path, rows: list[dict[str, Any]], rule: dict[str, Any], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "args": vars(args),
        "rule": rule,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    command = (
        "env CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
        "tools/run_e186_eflic_global_predecision_selector_probe.py "
        f"--device cuda:0 --kodak-dir {args.eval_dir_placeholder} "
        f"--force-ind {args.force} --alpha 0.05 --direction-source mean "
        f"--selector-feature {rule['feature']} --selector-op '{rule['op']}' "
        f"--selector-threshold {rule['threshold']:.12g} "
        "--output-prefix experiments/analysis/EVAL_OUTPUT_PREFIX"
    )

    lines = [
        "# E189 EF-LIC Global Selector Rule Fit",
        "",
        "This file fits one scalar selector rule from an E161-style EF-LIC active-branch label table. It is intended to fit on an independent validation split and then evaluate on a held-out image directory with the direct EF-LIC forward probe.",
        "",
        f"Input labels: `{args.input_csv}`",
        f"Feature manifest: `{args.manifest_csv}`",
        f"Force index: `{args.force}`",
        f"Target: `{args.target}`",
        f"Feature set: `{args.feature_set}`",
        "",
        "Fitted rule:",
        "",
        f"- feature: `{rule['feature']}`",
        f"- op: `{rule['op']}`",
        f"- threshold: `{rule['threshold']:.12g}`",
        f"- train images: `{rule['train_images']}`",
        f"- candidate features: `{rule['candidate_features']}`",
        "",
        "Direct held-out evaluation command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "| selector | branch share | dDISTS | dLPIPS | dPSNR | DISTS wins | rule |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['selector']} | {row['branch_share']:.3f} | "
            f"{row['selected_delta_dists']:+.6f} | {row['selected_delta_lpips']:+.6f} | "
            f"{row['selected_delta_psnr']:+.6f} | {row['selected_win_dists']}/{row['images']} | "
            f"{row.get('rule', '')} |"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- `global_predecision_context` is the paper-safe no-side-bit whole-image branch feature set.",
            "- `sequential_context` can be valid only for a sequential per-slice policy.",
            "- A rule fitted and evaluated on the same image set is diagnostic only; final claims need independent fit/eval or full matched training.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    args = parse_args()
    rows = [r for r in e184.read_rows(args.input_csv) if int(float(r.get("nonfinite", 0.0))) == 0]
    rows = [r for r in rows if int(r["force_ind"]) == args.force]
    if not rows:
        raise SystemExit(f"no finite rows for force{args.force}")

    manifest = e184.read_manifest(args.manifest_csv)
    feature_sets = e185.feature_sets(manifest)
    features = feature_sets[args.feature_set][0]
    valid_features = e184.valid_features(rows, features)
    feature, op, threshold, decisions, _ = e184.best_threshold(rows, valid_features, args.target, 0.0)
    if not feature:
        raise SystemExit("no valid scalar selector rule found")

    results = [
        e184.summarize_policy(f"force{args.force}", "baseline", rows, [False] * len(rows), 0.0, args.target),
        e184.summarize_policy(f"force{args.force}", "always_active", rows, [True] * len(rows), 0.0, args.target),
        e184.summarize_policy(
            f"force{args.force}",
            f"oracle_{args.target}",
            rows,
            e184.oracle(rows, args.target),
            0.0,
            args.target,
            "metric_oracle",
        ),
        e184.summarize_policy(
            f"force{args.force}",
            f"best_threshold_{args.target}",
            rows,
            decisions,
            0.0,
            args.target,
            args.feature_set,
            f"{feature} {op} {threshold:.9g}",
        ),
    ]
    rule = {
        "feature": feature,
        "op": op,
        "threshold": float(threshold),
        "target": args.target,
        "force": args.force,
        "feature_set": args.feature_set,
        "train_images": len(rows),
        "candidate_features": len(valid_features),
    }
    write_outputs(args.output_prefix, results, rule, args)


if __name__ == "__main__":
    main()
