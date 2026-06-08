#!/usr/bin/env python3
"""Build the GLC q-aware entropy-margin deployment controller spec.

E377/E378 are held-out audits. This script fits the chosen controller on the
available calibration rows and exports the deterministic spec to use for the
next matched GLC fine-tuning/full-training branch. It is not a new validation
result; validation remains the image-disjoint E378 margin sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import analyze_e377_glc_qaware_heldout_calibration as e377  # noqa: E402
from hcg_rvq.reliability_index_controller import QAwareThresholdControllerSpec  # noqa: E402


def policy_with_spec(policy: e377.Policy, spec: QAwareThresholdControllerSpec, family_suffix: str) -> e377.Policy:
    return replace(policy, thresholds=dict(spec.thresholds), family=f"{policy.family}{family_suffix}")


def eval_policy(rows: list[dict[str, object]], policy: e377.Policy, dataset: str) -> dict[str, object]:
    eval_rows = rows if dataset == "pooled" else [row for row in rows if row["dataset"] == dataset]
    selected = e377.apply_policy(eval_rows, policy)
    summary = e377.summarize_selection(eval_rows, selected)
    selected_rows = int(summary["selected_rows"])
    row = {
        "dataset": dataset,
        "family": policy.family,
        "policy": policy.name,
        "mode": policy.mode,
        "feature": policy.feature,
        "direction": policy.direction,
        "thresholds": json.dumps(policy.thresholds, sort_keys=True),
    }
    row.update(summary)
    row["selected_win_frac"] = (float(summary["selected_win_rows"]) / selected_rows) if selected_rows else math.nan
    row["selected_fixed_win_frac"] = (float(summary["selected_fixed_win_rows"]) / selected_rows) if selected_rows else math.nan
    return row


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if value == 0.0:
            return "0.000000"
        if abs(value) < 1.0:
            return f"{value:+.6f}"
        return f"{value:.6f}"
    return str(value)


def table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clic", type=Path, default=e377.DEFAULT_CLIC)
    parser.add_argument("--kodak", type=Path, default=e377.DEFAULT_KODAK)
    parser.add_argument("--label", default="trained_replacement_soft")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--feature", default="index_entropy_mean")
    parser.add_argument("--direction", default=">=", choices=[">=", "<="])
    parser.add_argument("--min-global-rows", type=int, default=10)
    parser.add_argument("--min-q-rows", type=int, default=2)
    parser.add_argument("--max-calib-worst", type=float, default=0.0)
    parser.add_argument("--max-calib-fixed-worst", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = e377.read_rows(args.clic, "CLIC41", args.label) + e377.read_rows(args.kodak, "Kodak24", args.label)
    if not rows:
        raise SystemExit("no input rows found")

    policies: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for mode in ["q-aware", "global"]:
        base = e377.fit_policy(
            rows,
            family=f"deployment-{mode}-{args.feature}-{args.direction}",
            feature=args.feature,
            direction=args.direction,
            mode=mode,
            profile="score+fixed-tail",
            min_global_rows=args.min_global_rows,
            min_q_rows=args.min_q_rows,
            max_worst=args.max_calib_worst,
            max_fixed_worst=args.max_calib_fixed_worst,
        )
        if base is None:
            continue
        base_spec = QAwareThresholdControllerSpec(dict(base.thresholds), direction=base.direction)
        margin_spec = base_spec.with_margin(args.margin)
        active = policy_with_spec(base, margin_spec, f"-margin{args.margin:g}")
        policy_payload = {
            "mode": mode,
            "profile": active.profile,
            "feature": active.feature,
            "direction": active.direction,
            "threshold_margin": args.margin,
            "base_thresholds": base.thresholds,
            "deployment_thresholds": active.thresholds,
            "controller_spec": {
                "thresholds": active.thresholds,
                "direction": active.direction,
                "soft_width": 0.0,
            },
            "python_literal": f"QAwareThresholdControllerSpec(thresholds={active.thresholds!r}, direction={active.direction!r}, soft_width=0.0)",
        }
        policies.append(policy_payload)
        for dataset in ["pooled", "CLIC41", "Kodak24"]:
            row = eval_policy(rows, active, dataset)
            row["threshold_margin"] = args.margin
            eval_rows.append(row)

    if not policies:
        raise SystemExit("no deployable policy fitted")

    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E379 GLC q-aware entropy-margin deployment spec",
        "validation_source": "E378 image-disjoint held-out safety-margin sweep",
        "score_definition": "delta_DISTS + 3 * delta_LPIPS + delta_bpp; PSNR ignored; MS-SSIM side reported",
        "input_label": args.label,
        "margin": args.margin,
        "policies": policies,
        "diagnostic_fit_on_all_rows": eval_rows,
        "main_policy": next((p for p in policies if p["mode"] == "q-aware"), policies[0]),
        "simple_ablation_policy": next((p for p in policies if p["mode"] == "global"), None),
    }
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(out_prefix.with_suffix(".csv"), eval_rows)

    fields = [
        "dataset",
        "mode",
        "selected_rows",
        "selected_frac",
        "score_all",
        "fixed_score_all",
        "selected_win_frac",
        "selected_fixed_win_frac",
        "selected_worst_score",
        "selected_worst_fixed_score",
        "selected_positive_rows",
        "selected_fixed_positive_rows",
        "delta_lpips_sum",
        "delta_dists_sum",
        "delta_ms_ssim_sum",
        "delta_bpp_sum",
    ]
    with out_prefix.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# GLC q-Aware Entropy-Margin Deployment Spec\n\n")
        handle.write(
            "This file exports the controller spec to use in the next GLC matched "
            "fine-tuning/full-training branch. It is fitted on all currently available "
            "calibration rows; validation evidence comes from E378, not from this "
            "same-row diagnostic table. PSNR is ignored.\n\n"
        )
        handle.write("## Main Spec\n\n")
        main_policy = payload["main_policy"]
        handle.write(f"- mode: `{main_policy['mode']}`\n")
        handle.write(f"- feature: `{main_policy['feature']}`\n")
        handle.write(f"- direction: `{main_policy['direction']}`\n")
        handle.write(f"- threshold_margin: `{main_policy['threshold_margin']}`\n")
        handle.write(f"- deployment_thresholds: `{main_policy['deployment_thresholds']}`\n")
        handle.write(f"- python_literal: `{main_policy['python_literal']}`\n\n")
        handle.write("## Same-Row Diagnostic Fit\n\n")
        handle.write(table(eval_rows, fields))
        handle.write("\n\n## Use\n\n")
        handle.write(
            "Use the q-aware spec as the first GLC long-run controller. Keep the global "
            "spec as the simple entropy ablation. Any paper claim should cite E378 "
            "held-out safety rather than this fit-on-all deployment diagnostic.\n"
        )



if __name__ == "__main__":
    main()
