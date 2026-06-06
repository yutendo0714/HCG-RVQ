import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from analyze_gate_selector import SEEDS, fmt, load_rows as load_holdout_rows, mean, summarize
from analyze_gate_selector_cv import DEPLOYABLE_FEATURES, choose_policy, policy_delta
from analyze_gate_selector_start0_recheck import load_rows as load_start0_rows


def evaluate_rows(rows: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    baseline = summarize(rows)
    hcs_mean = mean(float(row["hcs_rd"]) for row in rows)
    old_mean = mean(float(row["hcs_rd"]) + float(row["old_delta_rd"]) for row in rows)
    risk_mean = mean(float(row["hcs_rd"]) + float(row["risk_delta_rd"]) for row in rows)
    oracle_delta = baseline["oracle_old_or_min090_delta"]

    result: dict[str, Any] = {
        "n": len(rows),
        "hcs_mean_rd": hcs_mean,
        "old_mean_rd": old_mean,
        "risk_mean_rd": risk_mean,
        "oracle_old_or_min090_mean_rd": hcs_mean + oracle_delta,
        "baseline": baseline,
    }

    if policy is None:
        return result

    deltas = []
    risk_flags = []
    for row in rows:
        delta, use_risk = policy_delta(
            row,
            str(policy["feature"]),
            float(policy["threshold"]),
            str(policy["direction"]),
        )
        deltas.append(delta)
        risk_flags.append(1.0 if use_risk else 0.0)

    selected_delta = mean(deltas)
    selected_mean = hcs_mean + selected_delta
    old_delta = baseline["old_gate_mean_delta"]
    oracle_gap = old_delta - oracle_delta
    improvement_vs_old = old_delta - selected_delta
    result["selected_policy"] = {
        "mean_delta": selected_delta,
        "mean_rd": selected_mean,
        "vs_old_gate025": selected_delta - old_delta,
        "risk_fraction": mean(risk_flags),
        "hcs_win_rate": mean(1.0 if delta < 0.0 else 0.0 for delta in deltas),
        "oracle_gap_closed_fraction": improvement_vs_old / oracle_gap if abs(oracle_gap) > 1e-12 else math.nan,
    }
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_pairs = [
            policy_delta(
                row,
                str(policy["feature"]),
                float(policy["threshold"]),
                str(policy["direction"]),
            )
            for row in seed_rows
        ]
        result["selected_policy"][f"seed{seed}_delta"] = mean(delta for delta, _ in seed_pairs)
        result["selected_policy"][f"seed{seed}_risk_fraction"] = mean(1.0 if use_risk else 0.0 for _, use_risk in seed_pairs)
    return result


def compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "feature",
        "direction",
        "threshold",
        "mean_delta",
        "risk_fraction",
        "hcs_win_rate",
        "seed1234_delta",
        "seed2345_delta",
        "seed3456_delta",
    ]
    return {key: policy[key] for key in keys if key in policy}


def read_threshold_policy(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    deployable = [row for row in rows if row["feature"] in DEPLOYABLE_FEATURES]
    if not deployable:
        return None
    deployable.sort(key=lambda row: float(row["mean_delta"]))
    selected = deployable[0]
    policy: dict[str, Any] = {
        "feature": selected["feature"],
        "direction": selected["direction"],
        "threshold": float(selected["threshold"]),
        "mean_delta": float(selected["mean_delta"]),
        "risk_fraction": float(selected["risk_fraction"]),
        "hcs_win_rate": float(selected["hcs_win_rate"]),
    }
    for seed in SEEDS:
        key = f"seed{seed}_delta"
        if key in selected:
            policy[key] = float(selected[key])
    return policy


def select_policy(rows: list[dict[str, Any]], threshold_path: Path | None = None) -> dict[str, Any]:
    if threshold_path is not None:
        policy = read_threshold_policy(threshold_path)
        if policy is not None:
            return compact_policy(policy)
    return compact_policy(choose_policy(rows, DEPLOYABLE_FEATURES))


def policy_name(policy: dict[str, Any]) -> str:
    return "{} {} {:.6f}".format(policy["feature"], policy["direction"], float(policy["threshold"]))


def render_table_row(name: str, result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    selected = result["selected_policy"]
    return (
        "| {} | {} | {} ({}) | {} ({}) | {} ({}) | {} ({}) | {} | {} | {} |".format(
            name,
            fmt(result["hcs_mean_rd"]),
            fmt(result["old_mean_rd"]),
            fmt(baseline["old_gate_mean_delta"], True),
            fmt(result["risk_mean_rd"]),
            fmt(baseline["min090_mean_delta"], True),
            fmt(result["oracle_old_or_min090_mean_rd"]),
            fmt(baseline["oracle_old_or_min090_delta"], True),
            fmt(selected["mean_rd"]),
            fmt(selected["mean_delta"], True),
            fmt(selected["vs_old_gate025"], True),
            fmt(selected["risk_fraction"]),
            fmt(selected["oracle_gap_closed_fraction"]),
        )
    )


def render_markdown(payload: dict[str, Any]) -> str:
    policy = payload["calibrated_policy"]
    validation = payload["splits"]["validation_holdout4096"]
    transfer = payload["splits"]["reporting_start0_transfer"]
    slice_best = payload["splits"]["reporting_start0_slice_best"]
    validation_best = payload["split_best_policies"]["validation_holdout4096"]
    transfer_best = payload["split_best_policies"]["reporting_start0_transfer"]
    slice_best_best = payload["split_best_policies"]["reporting_start0_slice_best"]

    lines = [
        "# Gate0.25/min090 Reporting-Protocol Selector",
        "",
        "Calibration split: OpenImages `start_index=4096` current holdout.",
        "Reporting split: OpenImages `start_index=0` current recheck.",
        "The calibrated policy is selected once on the calibration split and then applied unchanged to reporting rows.",
        "",
        "## Calibrated Policy",
        "",
        "- policy: `{}`".format(policy_name(policy)),
        "- calibration mean delta: `{}`".format(fmt(float(policy["mean_delta"]), True)),
        "- calibration min090 fraction: `{}`".format(fmt(float(policy["risk_fraction"]))),
        "",
        "## Split Results",
        "",
        "| split | HCS RD | old gate RD (delta) | min090 RD (delta) | oracle old/min090 RD (delta) | calibrated policy RD (delta) | calibrated vs old | min090 fraction | oracle gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        render_table_row("validation holdout4096", validation),
        render_table_row("reporting start0 transfer", transfer),
        render_table_row("reporting start0 slice_best", slice_best),
        "",
        "## Best Same-Split Deployable Policies",
        "",
        "| split | policy | mean delta | min090 fraction |",
        "| --- | --- | ---: | ---: |",
        "| validation holdout4096 | `{}` | {} | {} |".format(policy_name(validation_best), fmt(float(validation_best["mean_delta"]), True), fmt(float(validation_best["risk_fraction"]))),
        "| reporting start0 transfer | `{}` | {} | {} |".format(policy_name(transfer_best), fmt(float(transfer_best["mean_delta"]), True), fmt(float(transfer_best["risk_fraction"]))),
        "| reporting start0 slice_best | `{}` | {} | {} |".format(policy_name(slice_best_best), fmt(float(slice_best_best["mean_delta"]), True), fmt(float(slice_best_best["risk_fraction"]))),
        "",
        "## Interpretation",
        "",
        "- A validation-calibrated raw-gate feature transfers to the reporting transfer protocol, improving old gate0.25 in the current multi-checkpoint analysis.",
        "- The same policy does not improve the slice-best checkpoint protocol, where old gate0.25 is already much stronger and the best same-split feature changes.",
        "- Therefore the paper-safe claim should couple reliability selection with an explicit validation-selected checkpoint protocol. The selector evidence is real, but the current old/min090 switch is not yet a single-checkpoint codec.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--output-prefix", default="gate025_min090_selector_reporting_protocol")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    validation_rows = load_holdout_rows(analysis_dir, "current_holdout")
    transfer_rows = load_start0_rows(analysis_dir, "transfer")
    slice_best_rows = load_start0_rows(analysis_dir, "slice_best")

    calibrated_policy = select_policy(
        validation_rows,
        analysis_dir / "gate025_min090_selector_val4096_holdout4096_current_thresholds.csv",
    )
    payload = {
        "calibration_split": "openimages_holdout4096_current",
        "reporting_split": "openimages_start0_current_recheck",
        "calibrated_policy": calibrated_policy,
        "splits": {
            "validation_holdout4096": evaluate_rows(validation_rows, calibrated_policy),
            "reporting_start0_transfer": evaluate_rows(transfer_rows, calibrated_policy),
            "reporting_start0_slice_best": evaluate_rows(slice_best_rows, calibrated_policy),
        },
        "split_best_policies": {
            "validation_holdout4096": select_policy(
                validation_rows,
                analysis_dir / "gate025_min090_selector_val4096_holdout4096_current_thresholds.csv",
            ),
            "reporting_start0_transfer": select_policy(
                transfer_rows,
                analysis_dir / "gate025_min090_selector_start0_current_recheck_transfer_thresholds.csv",
            ),
            "reporting_start0_slice_best": select_policy(
                slice_best_rows,
                analysis_dir / "gate025_min090_selector_start0_current_recheck_slice_best_thresholds.csv",
            ),
        },
    }

    json_path = analysis_dir / f"{args.output_prefix}.json"
    md_path = analysis_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
