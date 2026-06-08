#!/usr/bin/env python3
"""Summarize E349 EF-LIC CLIC transfer risk-grid results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("mode") == "trained_hard"]


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _summarize_rows(name: str, rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    vals = [_safe_float(row, "delta_psnr") for row in rows]
    bpps = [_safe_float(row, "delta_bpp") for row in rows]
    dec = [_safe_float(row, "max_decode_diff") for row in rows]
    nonfinite = [int(_safe_float(row, "nonfinite")) for row in rows]
    gates = [_safe_float(row, "y_gate_mean") for row in rows]
    alphas = [_safe_float(row, "y_alpha_mean") for row in rows]
    if not vals:
        raise ValueError(f"no trained_hard rows for {name}")
    return {
        "name": name,
        "n": len(vals),
        "mean_delta_psnr": sum(vals) / len(vals),
        "median_delta_psnr": statistics.median(vals),
        "worst_delta_psnr": min(vals),
        "best_delta_psnr": max(vals),
        "negative_count": sum(v < 0.0 for v in vals),
        "max_abs_delta_bpp": max(abs(v) for v in bpps),
        "max_decode_diff": max(dec),
        "nonfinite_sum": sum(nonfinite),
        "mean_gate": sum(gates) / len(gates),
        "mean_alpha": sum(alphas) / len(alphas),
    }


def _oracle(
    image_order: list[str],
    by_risk: dict[str, dict[str, float]],
    split_name: str,
) -> dict[str, float | int | str]:
    vals: list[float] = []
    choices: dict[str, int] = {}
    for image in image_order:
        candidates = {"noop": 0.0}
        candidates.update({risk: table[image] for risk, table in by_risk.items()})
        best = max(candidates, key=candidates.get)
        vals.append(candidates[best])
        choices[best] = choices.get(best, 0) + 1
    return {
        "split": split_name,
        "n": len(vals),
        "oracle_mean_delta_psnr": sum(vals) / len(vals),
        "oracle_worst_delta_psnr": min(vals),
        "oracle_negative_count": sum(v < 0.0 for v in vals),
        "oracle_choices_json": json.dumps(choices, sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    risk_rows = {name: _read_rows(Path(path)) for name, path in args.risk}
    image_order = [row["image"] for row in next(iter(risk_rows.values()))]
    for name, rows in risk_rows.items():
        order = [row["image"] for row in rows]
        if order != image_order:
            raise ValueError(f"image order mismatch for {name}")

    risk_summary = [_summarize_rows(name, rows) for name, rows in risk_rows.items()]

    split_summaries: list[dict[str, object]] = []
    splits = {
        "full41": image_order,
        "calib_first21": image_order[:21],
        "eval_last20": image_order[21:],
        "first16": image_order[:16],
        "last25": image_order[16:],
    }
    row_by_image = {
        risk: {row["image"]: row for row in rows}
        for risk, rows in risk_rows.items()
    }
    by_risk_delta = {
        risk: {row["image"]: _safe_float(row, "delta_psnr") for row in rows}
        for risk, rows in risk_rows.items()
    }
    for split_name, images in splits.items():
        for risk, table in row_by_image.items():
            split_summaries.append(
                {
                    "split": split_name,
                    **_summarize_rows(risk, [table[image] for image in images]),
                }
            )

    oracle_rows = [_oracle(images, by_risk_delta, split_name) for split_name, images in splits.items()]

    failures: list[dict[str, object]] = []
    for risk, rows in risk_rows.items():
        for row in rows:
            delta = _safe_float(row, "delta_psnr")
            if delta < 0.0:
                failures.append(
                    {
                        "risk": risk,
                        "image": row["image"],
                        "delta_psnr": delta,
                        "gate_mean": _safe_float(row, "y_gate_mean"),
                        "alpha_mean": _safe_float(row, "y_alpha_mean"),
                        "risk_score_mean": _safe_float(row, "y_risk_score_mean"),
                    }
                )
    failures.sort(key=lambda row: (row["delta_psnr"], row["risk"]))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(prefix.with_suffix(".risk_summary.csv"), risk_summary)
    _write_csv(prefix.with_suffix(".split_summary.csv"), split_summaries)
    _write_csv(prefix.with_suffix(".oracle_summary.csv"), oracle_rows)
    _write_csv(prefix.with_suffix(".failures.csv"), failures)
    with prefix.with_suffix(".json").open("w") as f:
        json.dump(
            {
                "risk_summary": risk_summary,
                "split_summary": split_summaries,
                "oracle_summary": oracle_rows,
                "failure_count": len(failures),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    lines = [
        "# E349 EF-LIC CLIC Transfer Grid",
        "",
        "## Risk Summary",
        "",
        "| risk | n | mean dPSNR | median | worst | best | neg | max abs dbpp | decode max | nonfinite | gate | alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in risk_summary:
        lines.append(
            "| {name} | {n} | {mean_delta_psnr} | {median_delta_psnr} | {worst_delta_psnr} | "
            "{best_delta_psnr} | {negative_count} | {max_abs_delta_bpp} | {max_decode_diff} | "
            "{nonfinite_sum} | {mean_gate} | {mean_alpha} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    lines += [
        "",
        "## Oracle",
        "",
        "| split | n | oracle mean dPSNR | oracle worst | neg | choices |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in oracle_rows:
        lines.append(
            "| {split} | {n} | {oracle_mean_delta_psnr} | {oracle_worst_delta_psnr} | "
            "{oracle_negative_count} | `{oracle_choices_json}` |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    lines += [
        "",
        "## Worst Failures",
        "",
        "| risk | image | dPSNR | gate | alpha | risk score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in failures[:12]:
        lines.append(
            "| {risk} | {image} | {delta_psnr} | {gate_mean} | {alpha_mean} | {risk_score_mean} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
