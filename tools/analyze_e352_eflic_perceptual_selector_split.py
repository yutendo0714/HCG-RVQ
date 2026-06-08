#!/usr/bin/env python3
"""Split-audit simple EF-LIC HCG perceptual reliability selectors."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def f(row: dict, key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in (None, ""):
        return default
    return float(val)


def parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("run must be name=csv")
    name, path = spec.split("=", 1)
    return name, Path(path)


def read_rows(path: Path, risk: str, lpips_weight: float) -> Dict[str, dict]:
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            r = dict(row)
            r["risk"] = risk
            r["delta_psnr"] = f(row, "delta_psnr")
            r["delta_ms_ssim"] = f(row, "delta_ms_ssim")
            r["delta_lpips"] = f(row, "delta_lpips")
            r["delta_dists"] = f(row, "delta_dists")
            r["delta_bpp"] = f(row, "delta_bpp")
            r["max_decode_diff"] = f(row, "max_decode_diff")
            r["nonfinite"] = int(f(row, "nonfinite"))
            r["score"] = r["delta_dists"] + lpips_weight * r["delta_lpips"]
            for key in FEATURES:
                r[key] = f(row, key)
            out[r["image"]] = r
    return out


FEATURES = [
    "y_gate_mean",
    "y_alpha_mean",
    "y_alpha_max",
    "y_avg_geometry_delta_rms",
    "y_avg_index_entropy",
    "y_avg_residual_error_rms",
    "y_mismatch",
    "y_risk_score_mean",
    "y_strength_mean",
]

NOOP = {
    "risk": "noop",
    "delta_psnr": 0.0,
    "delta_ms_ssim": 0.0,
    "delta_lpips": 0.0,
    "delta_dists": 0.0,
    "delta_bpp": 0.0,
    "max_decode_diff": 0.0,
    "nonfinite": 0,
    "score": 0.0,
}


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    risk: str = ""
    feature: str = ""
    direction: str = "ge"
    threshold: float = 0.0

    def select(self, image: str, runs: Dict[str, Dict[str, dict]]) -> dict:
        if self.kind == "noop":
            row = dict(NOOP)
            row["image"] = image
            return row
        if self.kind == "fixed":
            return runs[self.risk][image]
        if self.kind == "threshold":
            cand = runs[self.risk][image]
            val = cand[self.feature]
            ok = val >= self.threshold if self.direction == "ge" else val <= self.threshold
            if ok:
                return cand
            row = dict(NOOP)
            row["image"] = image
            return row
        if self.kind == "maxfeature":
            cands = [runs[r][image] for r in runs]
            best = max(cands, key=lambda r: r[self.feature])
            if best[self.feature] >= self.threshold:
                return best
            row = dict(NOOP)
            row["image"] = image
            return row
        raise ValueError(self.kind)


def summarize(policy: Policy, images: List[str], runs: Dict[str, Dict[str, dict]]) -> dict:
    rows = [policy.select(img, runs) for img in images]
    choices: Dict[str, int] = {}
    for r in rows:
        choices[r["risk"]] = choices.get(r["risk"], 0) + 1
    return {
        "policy": policy.name,
        "n": len(rows),
        "mean_score": mean(r["score"] for r in rows),
        "worst_score": max(r["score"] for r in rows),
        "score_win_count": sum(r["score"] < 0 for r in rows),
        "mean_delta_psnr": mean(r["delta_psnr"] for r in rows),
        "worst_delta_psnr": min(r["delta_psnr"] for r in rows),
        "negative_psnr_count": sum(r["delta_psnr"] < 0 for r in rows),
        "mean_delta_ms_ssim": mean(r["delta_ms_ssim"] for r in rows),
        "mean_delta_lpips": mean(r["delta_lpips"] for r in rows),
        "mean_delta_dists": mean(r["delta_dists"] for r in rows),
        "max_abs_delta_bpp": max(abs(r["delta_bpp"]) for r in rows),
        "max_decode_diff": max(r["max_decode_diff"] for r in rows),
        "nonfinite_rows": sum(r["nonfinite"] for r in rows),
        "choices": choices,
    }


def oracle_policy(images: List[str], runs: Dict[str, Dict[str, dict]]) -> dict:
    rows = []
    for img in images:
        candidates = [dict(NOOP, image=img)] + [runs[r][img] for r in runs]
        rows.append(min(candidates, key=lambda r: (r["score"], -r["delta_ms_ssim"])))
    choices: Dict[str, int] = {}
    for r in rows:
        choices[r["risk"]] = choices.get(r["risk"], 0) + 1
    return {
        "policy": "oracle_noop_risk",
        "n": len(rows),
        "mean_score": mean(r["score"] for r in rows),
        "worst_score": max(r["score"] for r in rows),
        "score_win_count": sum(r["score"] < 0 for r in rows),
        "mean_delta_psnr": mean(r["delta_psnr"] for r in rows),
        "worst_delta_psnr": min(r["delta_psnr"] for r in rows),
        "negative_psnr_count": sum(r["delta_psnr"] < 0 for r in rows),
        "mean_delta_ms_ssim": mean(r["delta_ms_ssim"] for r in rows),
        "mean_delta_lpips": mean(r["delta_lpips"] for r in rows),
        "mean_delta_dists": mean(r["delta_dists"] for r in rows),
        "max_abs_delta_bpp": max(abs(r["delta_bpp"]) for r in rows),
        "max_decode_diff": max(r["max_decode_diff"] for r in rows),
        "nonfinite_rows": sum(r["nonfinite"] for r in rows),
        "choices": choices,
    }


def thresholds(values: List[float]) -> List[float]:
    vals = sorted(set(values))
    if not vals:
        return [0.0]
    mids = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
    return [vals[0] - 1e-12] + mids + [vals[-1] + 1e-12]


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def write_csv(path: Path, rows: List[dict], fields: Iterable[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_run, required=True)
    ap.add_argument("--cal-count", type=int, default=20)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--lpips-weight", type=float, default=3.0)
    args = ap.parse_args()

    runs = {name: read_rows(path, name, args.lpips_weight) for name, path in args.run}
    image_lists = [list(rows) for rows in runs.values()]
    images = sorted(set(image_lists[0]).intersection(*map(set, image_lists[1:])))
    cal_images = images[: args.cal_count]
    eval_images = images[args.cal_count :]

    policies: List[Policy] = [Policy("noop", "noop")]
    policies += [Policy(f"fixed_{risk}", "fixed", risk=risk) for risk in runs]
    for risk, rows in runs.items():
        cal_rows = [rows[img] for img in cal_images]
        for feature in FEATURES:
            vals = [r[feature] for r in cal_rows]
            for thr in thresholds(vals):
                policies.append(Policy(f"thr_{risk}_{feature}_ge_{thr:.6g}", "threshold", risk=risk, feature=feature, direction="ge", threshold=thr))
                policies.append(Policy(f"thr_{risk}_{feature}_le_{thr:.6g}", "threshold", risk=risk, feature=feature, direction="le", threshold=thr))
    for feature in FEATURES:
        vals = [runs[r][img][feature] for r in runs for img in cal_images]
        for thr in thresholds(vals):
            policies.append(Policy(f"maxfeature_{feature}_ge_{thr:.6g}", "maxfeature", feature=feature, threshold=thr))

    rows = []
    for p in policies:
        cal = summarize(p, cal_images, runs)
        ev = summarize(p, eval_images, runs)
        rows.append({
            "policy": p.name,
            "cal_mean_score": cal["mean_score"],
            "cal_worst_score": cal["worst_score"],
            "cal_score_wins": cal["score_win_count"],
            "cal_mean_dpsnr": cal["mean_delta_psnr"],
            "cal_worst_dpsnr": cal["worst_delta_psnr"],
            "eval_mean_score": ev["mean_score"],
            "eval_worst_score": ev["worst_score"],
            "eval_score_wins": ev["score_win_count"],
            "eval_mean_dpsnr": ev["mean_delta_psnr"],
            "eval_worst_dpsnr": ev["worst_delta_psnr"],
            "eval_max_abs_delta_bpp": ev["max_abs_delta_bpp"],
            "eval_max_decode_diff": ev["max_decode_diff"],
            "eval_nonfinite_rows": ev["nonfinite_rows"],
            "eval_choices": ev["choices"],
        })

    rows.sort(key=lambda r: (r["cal_mean_score"], r["cal_worst_score"], -r["cal_mean_dpsnr"]))
    selected = rows[0]
    eval_ranked = sorted(rows, key=lambda r: (r["eval_mean_score"], r["eval_worst_score"], -r["eval_mean_dpsnr"]))[:20]
    cal_oracle = oracle_policy(cal_images, runs)
    eval_oracle = oracle_policy(eval_images, runs)

    prefix = Path(args.output_prefix)
    payload = {
        "lpips_weight": args.lpips_weight,
        "score": f"delta_DISTS + {args.lpips_weight:g} * delta_LPIPS",
        "n_images": len(images),
        "cal_images": cal_images,
        "eval_images": eval_images,
        "selected_by_calibration": selected,
        "top_eval_policies_diagnostic": eval_ranked[:10],
        "cal_oracle": cal_oracle,
        "eval_oracle": eval_oracle,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(prefix.with_name(prefix.name + "_all_policies.csv"), rows, rows[0].keys())
    write_csv(prefix.with_name(prefix.name + "_top_eval_diagnostic.csv"), eval_ranked, eval_ranked[0].keys())

    base_fields = ["policy", "cal_mean_score", "cal_worst_score", "cal_mean_dpsnr", "cal_worst_dpsnr", "eval_mean_score", "eval_worst_score", "eval_mean_dpsnr", "eval_worst_dpsnr", "eval_score_wins", "eval_choices"]
    lines = [
        "# E352 EF-LIC Perceptual Selector Split Audit", "",
        f"Score: `delta_DISTS + {args.lpips_weight:g} * delta_LPIPS` (lower is better).",
        f"Calibration images: {len(cal_images)}; eval images: {len(eval_images)}.", "",
        "## Selected by Calibration", "",
        "| policy | cal score | cal worst | cal dPSNR | cal worst dPSNR | eval score | eval worst | eval dPSNR | eval worst dPSNR | eval score wins | eval choices |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        "| {policy} | {cal_s} | {cal_w} | {cal_p} | {cal_wp} | {eval_s} | {eval_w} | {eval_p} | {eval_wp} | {wins} | `{choices}` |".format(
            policy=selected["policy"], cal_s=fmt(selected["cal_mean_score"]), cal_w=fmt(selected["cal_worst_score"]),
            cal_p=fmt(selected["cal_mean_dpsnr"]), cal_wp=fmt(selected["cal_worst_dpsnr"]),
            eval_s=fmt(selected["eval_mean_score"]), eval_w=fmt(selected["eval_worst_score"]),
            eval_p=fmt(selected["eval_mean_dpsnr"]), eval_wp=fmt(selected["eval_worst_dpsnr"]),
            wins=selected["eval_score_wins"], choices=selected["eval_choices"],
        ),
        "", "## Eval Oracle Upper Bound", "",
        "| split | score | worst score | dPSNR | worst dPSNR | score wins | choices |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| cal oracle | {s} | {w} | {p} | {wp} | {wins} | `{choices}` |".format(
            s=fmt(cal_oracle["mean_score"]), w=fmt(cal_oracle["worst_score"]), p=fmt(cal_oracle["mean_delta_psnr"]),
            wp=fmt(cal_oracle["worst_delta_psnr"]), wins=cal_oracle["score_win_count"], choices=cal_oracle["choices"]),
        "| eval oracle | {s} | {w} | {p} | {wp} | {wins} | `{choices}` |".format(
            s=fmt(eval_oracle["mean_score"]), w=fmt(eval_oracle["worst_score"]), p=fmt(eval_oracle["mean_delta_psnr"]),
            wp=fmt(eval_oracle["worst_delta_psnr"]), wins=eval_oracle["score_win_count"], choices=eval_oracle["choices"]),
        "", "## Top Eval Policies (diagnostic only)", "",
        "| policy | eval score | eval worst | eval dPSNR | eval worst dPSNR | choices |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in eval_ranked[:10]:
        lines.append("| {policy} | {s} | {w} | {p} | {wp} | `{choices}` |".format(
            policy=r["policy"], s=fmt(r["eval_mean_score"]), w=fmt(r["eval_worst_score"]),
            p=fmt(r["eval_mean_dpsnr"]), wp=fmt(r["eval_worst_dpsnr"]), choices=r["eval_choices"]
        ))
    lines += [
        "", "Interpretation:", "",
        "- This is a selector-design audit, not a final performance claim, because the policies are selected on a small split.",
        "- If the calibration-selected policy fails to transfer while the eval oracle is strong, the next step is a learned local/sequential selector with more independent calibration data.",
        "- If a simple threshold transfers, EF-LIC can move toward longer/full training with that selector frozen.",
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {prefix.with_suffix('.md')}, {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
