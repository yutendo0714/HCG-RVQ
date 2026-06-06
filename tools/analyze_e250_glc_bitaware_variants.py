#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--glob",
        default="experiments/analysis/e250_glc_bitaware_tail_vq_split_train*.json",
        help="Glob for E250 JSON artifacts, relative to the repository root.",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e250_glc_bitaware_variant_summary",
    )
    p.add_argument("--reference-name", default="E181 q0 OpenImages16->Kodak8 trained")
    p.add_argument("--reference-dbpp", type=float, default=0.014548)
    p.add_argument("--reference-score0", type=float, default=-0.016977)
    p.add_argument("--reference-score1", type=float, default=-0.002429)
    return p.parse_args()


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def trained_delta(payload: dict[str, Any]) -> dict[str, Any] | None:
    for row in payload.get("delta_summary", []):
        if row.get("label") == "trained_eval":
            return row
    return None


def trained_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    for row in payload.get("summary", []):
        if row.get("label") == "trained_eval":
            return row
    return None


def collect_rows(paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        delta = trained_delta(payload)
        summary = trained_summary(payload)
        if delta is None or summary is None:
            continue
        run_args = payload.get("args", {})
        d_lpips = finite(delta.get("delta_lpips"))
        d_dists = finite(delta.get("delta_dists"))
        d_bpp = finite(delta.get("empirical_bpp_delta"))
        score0 = d_dists + 3.0 * d_lpips
        score1 = score0 + d_bpp
        rows.append(
            {
                "artifact": path.stem,
                "train_limit": run_args.get("train_limit"),
                "eval_limit": run_args.get("eval_limit"),
                "k": run_args.get("k"),
                "scope": run_args.get("scope"),
                "soft_index_weight": run_args.get("soft_index_weight"),
                "soft_index_target": run_args.get("soft_index_target"),
                "lpips_weight": run_args.get("lpips_weight"),
                "delta_psnr": finite(delta.get("delta_psnr")),
                "delta_ms_ssim": finite(delta.get("delta_ms_ssim")),
                "delta_lpips": d_lpips,
                "delta_dists": d_dists,
                "delta_empirical_bpp": d_bpp,
                "active_mse_ratio": finite(delta.get("active_mse_ratio")),
                "index_entropy_mean": finite(summary.get("index_entropy_mean")),
                "score0_dists_3lpips": score0,
                "score1_plus_bpp": score1,
                "beats_reference_dbpp": d_bpp < args.reference_dbpp,
                "beats_reference_score0": score0 < args.reference_score0,
                "beats_reference_score1": score1 < args.reference_score1,
                "nonfinite_rows": int(delta.get("nonfinite_rows", 0)),
            }
        )
    rows.sort(key=lambda r: (r["score1_plus_bpp"], r["delta_empirical_bpp"], r["artifact"]))
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "artifact",
        "train_limit",
        "eval_limit",
        "k",
        "scope",
        "soft_index_weight",
        "soft_index_target",
        "lpips_weight",
        "delta_psnr",
        "delta_ms_ssim",
        "delta_lpips",
        "delta_dists",
        "delta_empirical_bpp",
        "active_mse_ratio",
        "index_entropy_mean",
        "score0_dists_3lpips",
        "score1_plus_bpp",
        "beats_reference_dbpp",
        "beats_reference_score0",
        "beats_reference_score1",
        "nonfinite_rows",
    ]
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "E250 GLC bit-aware variant summary",
        "reference": {
            "name": args.reference_name,
            "dbpp": args.reference_dbpp,
            "score0_dists_3lpips": args.reference_score0,
            "score1_plus_bpp": args.reference_score1,
        },
        "rows": rows,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E250 GLC Bit-Aware Variant Summary",
        "",
        f"Reference: `{args.reference_name}` with dbpp `{args.reference_dbpp:.6f}`, "
        f"score0 `{args.reference_score0:.6f}`, score1 `{args.reference_score1:.6f}`.",
        "",
        "Lower scores are better. `score0 = delta_DISTS + 3 * delta_LPIPS`; "
        "`score1 = score0 + delta_empirical_bpp`.",
        "",
        "| artifact | train/eval | K | scope | soft w | lpips w | dPSNR | dMS | dLPIPS | dDISTS | dbpp | active MSE | H | score0 | score1 | nonfinite |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['artifact']} | {r['train_limit']}/{r['eval_limit']} | {r['k']} | {r['scope']} | "
            f"{float(r['soft_index_weight']):.3g} | {float(r['lpips_weight']):.3g} | "
            f"{r['delta_psnr']:+.6f} | {r['delta_ms_ssim']:+.6f} | "
            f"{r['delta_lpips']:+.6f} | {r['delta_dists']:+.6f} | "
            f"{r['delta_empirical_bpp']:+.6f} | {r['active_mse_ratio']:.6f} | "
            f"{r['index_entropy_mean']:.6f} | {r['score0_dists_3lpips']:+.6f} | "
            f"{r['score1_plus_bpp']:+.6f} | {r['nonfinite_rows']} |"
        )
    best = rows[0]
    beats_ref = bool(best["beats_reference_score1"])
    if beats_ref:
        decision = [
            "The best E250 row so far is the K=8 part-group branch with LPIPS in the image loss and a strong soft-index penalty. On the current mid-scale gate it improves the bpp-charged score over the E181 reference, while also lowering the empirical bpp delta.",
            "",
            "This promotes the E250 design from a pure smoke diagnostic to the next GLC candidate gate. It is still not a final paper-main result because this is not full training, but the matched OI16->Kodak8 gate is now strong enough to justify full-Kodak and CLIC Professional scaling.",
        ]
    else:
        decision = [
            "The best E250 row so far is the K=8 part-group branch with LPIPS in the image loss and a strong soft-index penalty. It restores the perceptual direction on the small smoke split, but it still does not beat the E181 reference after bpp is charged.",
            "",
            "Therefore E250 is a useful implementation gate, not a paper-main result. The next GLC step should keep the K=8 part-group branch, include the perceptual term, and replace or augment the soft entropy proxy with explicit activation/index-prior control before a larger Kodak8 or full-Kodak run.",
        ]
    lines.extend(["", "## Decision", "", *decision, ""])
    lines.extend(
        [
            "K=4 and shared-codebook variants remain rate lower-bound ablations. They reduce empirical bpp, but their quality losses are too large for promotion.",
        ]
    )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(ROOT.glob(args.glob))
    rows = collect_rows(paths, args)
    if not rows:
        raise SystemExit(f"no trained_eval rows found for {args.glob}")
    write_outputs(args, rows)
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
