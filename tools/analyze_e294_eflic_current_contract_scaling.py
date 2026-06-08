#!/usr/bin/env python3
"""Aggregate EF-LIC current-code HCG branch smokes.

E292/E293 are codec-path contract runs, not final paper evidence. This analyzer
turns their per-image CSVs into a promotion audit: which fixed branches are safe
enough to keep as ablations, which are only oracle/controller candidates, and
whether the zero fallback preserves the EF-LIC no-entropy path.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "kodak24": ROOT / "experiments/analysis/e292_eflic_branch_controller_current_kodak24_contract.csv",
    "clicpro16": ROOT / "experiments/analysis/e293_eflic_branch_controller_current_clicpro16_contract.csv",
}
OUT_PREFIX = ROOT / "experiments/analysis/e294_eflic_current_contract_scaling"


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, path in INPUTS.items():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["split"] = split
                rows.append(row)
    return rows


def summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [as_float(r, "delta_psnr") for r in rows]  # type: ignore[arg-type]
    dbpps = [as_float(r, "delta_bpp") for r in rows]  # type: ignore[arg-type]
    decode = [as_float(r, "max_decode_diff") for r in rows]  # type: ignore[arg-type]
    nonfinite = [as_int(r, "nonfinite") for r in rows]  # type: ignore[arg-type]
    geom = [as_float(r, "y_avg_geometry_delta_rms") for r in rows]  # type: ignore[arg-type]
    ym = [as_float(r, "y_mismatch") / max(as_float(r, "y_total", 1.0), 1.0) for r in rows]  # type: ignore[arg-type]
    entropy = [as_float(r, "y_avg_index_entropy") for r in rows]  # type: ignore[arg-type]
    payload_equal = [as_int(r, "payload_equal") for r in rows]  # type: ignore[arg-type]
    payload_len_equal = [as_int(r, "payload_len_equal") for r in rows]  # type: ignore[arg-type]
    worst_row = min(rows, key=lambda r: as_float(r, "delta_psnr"))  # type: ignore[arg-type]
    best_row = max(rows, key=lambda r: as_float(r, "delta_psnr"))  # type: ignore[arg-type]
    return {
        "images": len(rows),
        "delta_bpp": mean(dbpps),
        "delta_psnr": mean(deltas),
        "win_frac": sum(1 for v in deltas if v > 0.0) / len(deltas),
        "nonharm_frac": sum(1 for v in deltas if v >= 0.0) / len(deltas),
        "worst_delta_psnr": min(deltas),
        "worst_image": str(worst_row["image"]),
        "best_delta_psnr": max(deltas),
        "best_image": str(best_row["image"]),
        "max_decode_diff": max(decode),
        "nonfinite_rows": sum(nonfinite),
        "payload_equal_frac": mean(payload_equal),
        "payload_len_equal_frac": mean(payload_len_equal),
        "y_mismatch_frac": mean(ym),
        "geometry_delta_rms": mean(geom),
        "index_entropy": mean(entropy),
    }


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    pooled: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        preset = str(row["preset"])
        grouped[(split, preset)].append(row)
        pooled[preset].append(row)

    split_rows = []
    for (split, preset), group_rows in sorted(grouped.items()):
        item = summarize_group(group_rows)
        item.update({"split": split, "preset": preset, "family": str(group_rows[0]["family"])})
        split_rows.append(item)

    pooled_rows = []
    for preset, group_rows in sorted(pooled.items()):
        item = summarize_group(group_rows)
        item.update({"split": "pooled", "preset": preset, "family": str(group_rows[0]["family"])})
        pooled_rows.append(item)

    decisions = {
        "zero_contract_ok": all(
            r["preset"] != "zero"
            or (r["max_decode_diff"] == 0.0 and r["nonfinite_rows"] == 0 and r["payload_equal_frac"] == 1.0)
            for r in split_rows
        ),
        "fixed_branch_caveat": (
            "Fixed nonzero branches are codec-valid but not reliable enough as paper-main policies; "
            "the Kodak/CLIC disagreement supports a learned decoder-safe fallback controller."
        ),
        "main_next_step": (
            "Train the EF-LIC HCGBranchController in-codec with dominant original RD/loss terms, "
            "false-positive/fallback regularization, and Kodak24+CLIC Professional validation."
        ),
    }
    return {"inputs": {k: str(v) for k, v in INPUTS.items()}, "split_rows": split_rows, "pooled_rows": pooled_rows, "decisions": decisions}


def fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:+.6f}"
    return str(x)


def write_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# E294 EF-LIC Current Contract Scaling Audit",
        "",
        "E292/E293 extend the current EF-LIC decoder-safe branch vocabulary from tiny four-image smokes to Kodak24 and CLIC Professional 16. This is still a short-cycle codec-path audit, not final full-training evidence.",
        "",
        "## Pooled Rows",
        "",
        "| preset | family | images | delta bpp | delta PSNR | win | nonharm | worst PSNR | worst image | best PSNR | max decode | nonfinite | y mismatch | geom RMS | index entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pooled_rows"]:  # type: ignore[index]
        lines.append(
            "| {preset} | {family} | {images} | {delta_bpp} | {delta_psnr} | {win_frac} | {nonharm_frac} | {worst_delta_psnr} | {worst_image} | {best_delta_psnr} | {max_decode_diff} | {nonfinite_rows} | {y_mismatch_frac} | {geometry_delta_rms} | {index_entropy} |".format(
                **{k: fmt(v) for k, v in row.items()}  # type: ignore[union-attr]
            )
        )

    lines += [
        "",
        "## Split Rows",
        "",
        "| split | preset | images | delta PSNR | win | worst PSNR | worst image | best PSNR | best image | y mismatch | geom RMS |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in summary["split_rows"]:  # type: ignore[index]
        lines.append(
            "| {split} | {preset} | {images} | {delta_psnr} | {win_frac} | {worst_delta_psnr} | {worst_image} | {best_delta_psnr} | {best_image} | {y_mismatch_frac} | {geometry_delta_rms} |".format(
                **{k: fmt(v) for k, v in row.items()}  # type: ignore[union-attr]
            )
        )

    decisions = summary["decisions"]  # type: ignore[index]
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Zero fallback contract ok: `{decisions['zero_contract_ok']}`.",
        f"- {decisions['fixed_branch_caveat']}",
        "- Kodak24 and CLIC Professional disagree on aggressive all-position geometry: this is direct evidence against making `constant020` or dense all-on HCG the main policy.",
        "- Sparse previous-context geometry is safer but too weak/mixed to be the final method by itself.",
        "- `soft_support020` has useful best cases but still needs a learned fallback gate, because unconditional use remains harmful on CLIC16.",
        "",
        "## Next Step",
        "",
        str(decisions["main_next_step"]),
        "",
        "## Inputs",
        "",
    ]
    for key, path in summary["inputs"].items():  # type: ignore[index]
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    summary = build_summary(rows)
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    OUT_PREFIX.with_suffix(".md").write_text(write_markdown(summary))
    print(f"wrote {OUT_PREFIX.with_suffix('.md')} and {OUT_PREFIX.with_suffix('.json')}")


if __name__ == "__main__":
    main()
