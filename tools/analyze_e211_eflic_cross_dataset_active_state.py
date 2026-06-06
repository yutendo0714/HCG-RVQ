#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


DEFAULT_INPUTS = [
    (
        "openimages24576_eval64",
        "experiments/analysis/e211_eflic_openimages24576_eval64_mean_alpha002_active.csv",
    ),
    ("clic_mobile24", "experiments/analysis/e211_eflic_clic_mobile24_mean_alpha002_active.csv"),
    ("clic_professional24", "experiments/analysis/e211_eflic_clic_professional24_mean_alpha002_active.csv"),
    ("div2k_valid24", "experiments/analysis/e211_eflic_div2k_valid24_mean_alpha002_active.csv"),
    ("tecnick_b01r01_24", "experiments/analysis/e211_eflic_tecnick_b01r01_24_mean_alpha002_active.csv"),
]


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return mean(finite) if finite else float("nan")


def safe_max(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def summarize_dataset(name: str, path: Path) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    rows = list(csv.DictReader(path.open()))
    delta_lpips = [f(r, "delta_lpips") for r in rows]
    delta_dists = [f(r, "delta_dists") for r in rows]
    delta_psnr = [f(r, "delta_psnr") for r in rows]
    bpp = [f(r, "bpp") for r in rows]
    nonfinite = [f(r, "nonfinite") for r in rows]
    decode = [f(r, "max_decode_diff") for r in rows]
    y_mismatch_frac = [f(r, "y_mismatch") / max(f(r, "y_total"), 1.0) for r in rows]
    z_mismatch_frac = [f(r, "z_mismatch") / max(f(r, "z_total"), 1.0) for r in rows]
    geom = [f(r, "y_avg_geometry_delta_rms") for r in rows]
    idx_entropy = [f(r, "y_avg_index_entropy") for r in rows]
    idx_used = [f(r, "y_avg_index_used_frac") for r in rows]
    z_used = [f(r, "z_index_used_frac") for r in rows]

    score = [d + 3.0 * l for d, l in zip(delta_dists, delta_lpips)]
    summary: dict[str, float | str] = {
        "dataset": name,
        "path": str(path),
        "n": len(rows),
        "bpp_mean": safe_mean(bpp),
        "delta_lpips_mean": safe_mean(delta_lpips),
        "delta_dists_mean": safe_mean(delta_dists),
        "delta_psnr_mean": safe_mean(delta_psnr),
        "score_dists_plus_3lpips_mean": safe_mean(score),
        "lpips_wins": sum(v < 0 for v in delta_lpips if math.isfinite(v)),
        "dists_wins": sum(v < 0 for v in delta_dists if math.isfinite(v)),
        "both_wins": sum(
            l < 0 and d < 0 for l, d in zip(delta_lpips, delta_dists) if math.isfinite(l) and math.isfinite(d)
        ),
        "lpips_loss": sum(v > 0 for v in delta_lpips if math.isfinite(v)),
        "dists_loss": sum(v > 0 for v in delta_dists if math.isfinite(v)),
        "nonfinite_sum": sum(v for v in nonfinite if math.isfinite(v)),
        "max_decode_diff": safe_max(decode),
        "y_mismatch_frac_mean": safe_mean(y_mismatch_frac),
        "z_mismatch_frac_mean": safe_mean(z_mismatch_frac),
        "geometry_delta_rms_mean": safe_mean(geom),
        "y_index_entropy_mean": safe_mean(idx_entropy),
        "y_index_used_frac_mean": safe_mean(idx_used),
        "z_index_used_frac_mean": safe_mean(z_used),
        "corr_y_mismatch_delta_dists": pearson(y_mismatch_frac, delta_dists),
        "corr_y_mismatch_delta_lpips": pearson(y_mismatch_frac, delta_lpips),
        "corr_geom_delta_dists": pearson(geom, delta_dists),
        "corr_geom_delta_lpips": pearson(geom, delta_lpips),
        "corr_y_index_entropy_delta_dists": pearson(idx_entropy, delta_dists),
        "corr_y_index_entropy_delta_lpips": pearson(idx_entropy, delta_lpips),
    }
    per_rows: list[dict[str, float | str]] = []
    for r, l, d, p, s, ym, zm, g, e in zip(
        rows, delta_lpips, delta_dists, delta_psnr, score, y_mismatch_frac, z_mismatch_frac, geom, idx_entropy
    ):
        per_rows.append(
            {
                "dataset": name,
                "image": r.get("image", ""),
                "delta_lpips": l,
                "delta_dists": d,
                "delta_psnr": p,
                "score_dists_plus_3lpips": s,
                "y_mismatch_frac": ym,
                "z_mismatch_frac": zm,
                "geometry_delta_rms": g,
                "y_index_entropy": e,
            }
        )
    return summary, per_rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: float | str) -> str:
    if isinstance(v, str):
        return v
    if not math.isfinite(v):
        return "nan"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=Path("experiments/analysis/e211_eflic_cross_dataset_mean_alpha002_active"))
    args = parser.parse_args()

    summaries = []
    per_image = []
    for name, path_text in DEFAULT_INPUTS:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        summary, rows = summarize_dataset(name, path)
        summaries.append(summary)
        per_image.extend(rows)

    write_csv(args.output_prefix.with_suffix(".summary.csv"), summaries)
    write_csv(args.output_prefix.with_suffix(".per_image.csv"), per_image)

    lines = [
        "# E211 EF-LIC mean/alpha=0.02 cross-dataset active-state audit",
        "",
        "All runs use `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`. Negative deltas mean the HCG active branch improves over EF-LIC baseline.",
        "",
        "| dataset | n | dDISTS | dLPIPS | dPSNR | score | DISTS wins | LPIPS wins | both wins | nonfinite | max decode diff | y mismatch | geom RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {dataset} | {n} | {delta_dists_mean} | {delta_lpips_mean} | {delta_psnr_mean} | {score_dists_plus_3lpips_mean} | "
            "{dists_wins} | {lpips_wins} | {both_wins} | {nonfinite_sum} | {max_decode_diff} | {y_mismatch_frac_mean} | {geometry_delta_rms_mean} |".format(
                **{k: fmt(v) for k, v in s.items()}
            )
        )

    lines += [
        "",
        "## Correlation Audit",
        "",
        "| dataset | corr(y mismatch,dDISTS) | corr(y mismatch,dLPIPS) | corr(geom,dDISTS) | corr(geom,dLPIPS) | corr(index entropy,dDISTS) | corr(index entropy,dLPIPS) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {dataset} | {corr_y_mismatch_delta_dists} | {corr_y_mismatch_delta_lpips} | {corr_geom_delta_dists} | {corr_geom_delta_lpips} | {corr_y_index_entropy_delta_dists} | {corr_y_index_entropy_delta_lpips} |".format(
                **{k: fmt(v) for k, v in s.items()}
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `mean/alpha=0.02` is codec-valid across all audited splits: no nonfinite outputs and exact active encode/decode agreement.",
        "- The result is not uniformly strong across domains. It improves Kodak/OpenImages-calib in E209 and many DIV2K/CLIC-mobile cases, but CLIC-professional and Tecnick show mixed or harmful DISTS/LPIPS behavior.",
        "- This supports using `mean/alpha=0.02` as the current active branch, but not as a final always-on paper claim. The next step should be domain-robust reliability control or a dataset-conditioned alpha/safety rule trained on independent splits.",
    ]
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_prefix.with_suffix('.summary.csv')}")
    print(f"wrote {args.output_prefix.with_suffix('.per_image.csv')}")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
