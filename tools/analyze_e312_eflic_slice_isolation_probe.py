#!/usr/bin/env python3
"""Aggregate EF-LIC HCG slice-isolation probe CSVs.

E312 is a diagnostic bridge from image-level HCG policy selection to local/slice
controller labels. It reads E295 outputs produced with different
``--active-slices`` values and summarizes single-slice, leave-one-out, and
all-slice interactions. It is not a final RD claim.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ALL_SLICES = {0, 1, 2, 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-glob",
        default=str(ROOT / "experiments/analysis/e312_eflic_slice_isolation_kodak1_*.csv"),
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e312_eflic_slice_isolation_probe_summary",
    )
    return p.parse_args()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_slices(value: str) -> set[int]:
    text = str(value).strip().lower()
    if text in {"", "all"}:
        return set(ALL_SLICES)
    if text in {"none", "off", "zero"}:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def slice_label(slices: set[int]) -> str:
    if slices == ALL_SLICES:
        return "all"
    if not slices:
        return "none"
    return ",".join(str(i) for i in sorted(slices))


def read_rows(pattern: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in sorted(glob.glob(pattern)):
        path = Path(raw)
        with path.open(newline="") as fobj:
            for row in csv.DictReader(fobj):
                slices = parse_slices(row.get("active_slices", "all"))
                item: dict[str, Any] = dict(row)
                item["source_csv"] = str(path)
                item["slice_set"] = slice_label(slices)
                item["slice_count"] = len(slices)
                item["omitted_slice"] = next(iter(ALL_SLICES - slices)) if len(slices) == 3 else ""
                item["single_slice"] = next(iter(slices)) if len(slices) == 1 else ""
                for key in [
                    "delta_psnr",
                    "delta_bpp",
                    "max_decode_diff",
                    "max_baseline_diff",
                    "nonfinite",
                    "payload_equal",
                    "payload_len_equal",
                    "y_alpha_mean",
                    "y_avg_geometry_delta_rms",
                    "y_avg_index_entropy",
                    "y_slice_enabled",
                ]:
                    item[key] = safe_float(row.get(key), 0.0)
                item["contract_ok"] = int(
                    abs(item["delta_bpp"]) <= 1e-12
                    and item["max_decode_diff"] <= 1e-10
                    and int(item["nonfinite"]) == 0
                    and int(item["payload_len_equal"]) == 1
                )
                out.append(item)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(str(row["image"]), []).append(row)
    out: list[dict[str, Any]] = []
    for image, subset in sorted(by_image.items()):
        valid = [r for r in subset if int(r["contract_ok"]) == 1]
        if not valid:
            continue
        all_rows = [r for r in valid if r["slice_set"] == "all"]
        all_delta = safe_float(all_rows[0]["delta_psnr"]) if all_rows else float("nan")
        best = max(valid, key=lambda r: safe_float(r["delta_psnr"], -1e9))
        worst = min(valid, key=lambda r: safe_float(r["delta_psnr"], 1e9))
        singles = [r for r in valid if r["slice_count"] == 1]
        leaves = [r for r in valid if r["slice_count"] == 3]
        leave_marginals = {}
        if math.isfinite(all_delta):
            for r in leaves:
                leave_marginals[str(r["omitted_slice"])] = all_delta - safe_float(r["delta_psnr"])
        out.append(
            {
                "image": image,
                "rows": len(subset),
                "valid_rows": len(valid),
                "all_delta_psnr": all_delta,
                "best_slice_set": best["slice_set"],
                "best_delta_psnr": safe_float(best["delta_psnr"]),
                "best_gain_over_all": safe_float(best["delta_psnr"]) - all_delta if math.isfinite(all_delta) else float("nan"),
                "worst_slice_set": worst["slice_set"],
                "worst_delta_psnr": safe_float(worst["delta_psnr"]),
                "positive_single_slices": ",".join(str(r["single_slice"]) for r in singles if safe_float(r["delta_psnr"]) > 0),
                "negative_single_slices": ",".join(str(r["single_slice"]) for r in singles if safe_float(r["delta_psnr"]) < 0),
                "leave_one_out_marginal_json": json.dumps(leave_marginals, sort_keys=True),
                "contract_ok_frac": sum(int(r["contract_ok"]) for r in subset) / max(1, len(subset)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_glob)
    if not rows:
        raise SystemExit(f"no rows matched {args.input_glob}")
    summary = summarize(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_csv = args.output_prefix.with_suffix(".rows.csv")
    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(rows_csv, rows)
    write_csv(summary_csv, summary)
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True))
    with md_path.open("w") as fobj:
        fobj.write("# E312 EF-LIC Slice-Isolation Probe Summary\n\n")
        fobj.write("This aggregates E295 runs with different `--active-slices` values. It is a diagnostic artifact for local/slice label design, not final RD evidence.\n\n")
        fobj.write(f"- Input glob: `{args.input_glob}`\n")
        fobj.write(f"- Rows: `{len(rows)}`\n")
        fobj.write(f"- Images: `{len(summary)}`\n\n")
        keys = ["image", "all_delta_psnr", "best_slice_set", "best_delta_psnr", "best_gain_over_all", "worst_slice_set", "worst_delta_psnr", "positive_single_slices", "negative_single_slices", "leave_one_out_marginal_json", "contract_ok_frac"]
        fobj.write("| " + " | ".join(keys) + " |\n")
        fobj.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for item in summary:
            fobj.write("| " + " | ".join(fmt(item.get(k, "")) for k in keys) + " |\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- A useful single slice can become harmful in the all-slice context, and vice versa. Use leave-one-out marginals plus local residual/headroom labels rather than additive single-slice deltas alone.\n")
        fobj.write("- `contract_ok_frac` should stay 1.0 before using these rows for controller-label design.\n")
    print(f"wrote {rows_csv}, {summary_csv}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
