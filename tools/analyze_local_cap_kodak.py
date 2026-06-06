#!/usr/bin/env python3
"""Evaluate local-cap HCG-RVQ checkpoints on Kodak with per-image features."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe_householder_inverse_modes import evaluate_mode

ANALYSIS = ROOT / "experiments" / "analysis"
OUT_CSV = ANALYSIS / "local_cap080_rho1_multiseed_kodak_trusted_protocol.csv"
OUT_JSON = ANALYSIS / "local_cap080_rho1_multiseed_kodak_trusted_protocol.json"
OUT_MD = ANALYSIS / "local_cap080_rho1_multiseed_kodak_trusted_protocol.md"

RUNS = [
    {
        "method": "HCS",
        "seed": "1234",
        "step": "500",
        "config": "configs/pilot_hcs_rvq_frozen.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0035/checkpoint_step_500.pth.tar",
    },
    {
        "method": "HCS",
        "seed": "2345",
        "step": "250",
        "config": "configs/pilot_hcs_rvq_frozen_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0035_seed2345_seed2345/checkpoint_step_250.pth.tar",
    },
    {
        "method": "HCS",
        "seed": "3456",
        "step": "250",
        "config": "configs/pilot_hcs_rvq_frozen_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcs_rvq_frozen_g64_l1_k128_lambda0035_seed3456_seed3456/checkpoint_step_250.pth.tar",
    },
    {
        "method": "old gate0.25",
        "seed": "1234",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_frozen_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_250.pth.tar",
    },
    {
        "method": "old gate0.25",
        "seed": "2345",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_frozen_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_frozen_g64_l1_k128_lambda0035_seed2345/checkpoint_step_250.pth.tar",
    },
    {
        "method": "old gate0.25",
        "seed": "3456",
        "step": "500",
        "config": "configs/pilot_hcg_rvq_h_gate025_frozen_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar",
    },
    {
        "method": "min090",
        "seed": "1234",
        "step": "500",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_500.pth.tar",
    },
    {
        "method": "min090",
        "seed": "2345",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_g64_l1_k128_lambda0035_seed2345/checkpoint_step_250.pth.tar",
    },
    {
        "method": "min090",
        "seed": "3456",
        "step": "500",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar",
    },
    {
        "method": "local cap080/rho1",
        "seed": "1234",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_250.pth.tar",
    },
    {
        "method": "local cap080/rho1",
        "seed": "2345",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_g64_l1_k128_lambda0035_seed2345/checkpoint_step_250.pth.tar",
    },
    {
        "method": "local cap080/rho1",
        "seed": "3456",
        "step": "250",
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_250.pth.tar",
    },
]

FEATURE_KEYS = [
    "rvq_s_q_mean",
    "rvq_householder_strength",
    "rvq_householder_delta_rms",
    "rvq_householder_delta_rms_local_mean",
    "rvq_householder_delta_rms_local_max",
    "rvq_latent_quant_mse",
    "rvq_perplexity",
    "rvq_dead_code_ratio",
    "rvq_householder_risk_multiplier",
]


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = finite_values(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / (len(vals) - 1))


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def maybe_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def summarize_rows(method: str, rows: list[dict[str, object]]) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "method": method,
        "num_rows": len(rows),
        "mean_rd": mean([float(row["rd_score"]) for row in rows]),
        "mean_bpp": mean([float(row["bpp"]) for row in rows]),
        "mean_psnr": mean([float(row["psnr"]) for row in rows]),
        "mean_ms_ssim": mean([float(row["ms_ssim"]) for row in rows]),
        "nonfinite_rows": sum(int(row.get("has_nonfinite", 0)) for row in rows),
    }
    for key in FEATURE_KEYS:
        values = [maybe_float(row, key) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            result[f"mean_{key}"] = mean(values)
    return result


def main() -> None:
    data_root = "/dpl/kodak"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for run in RUNS:
        for key in ("config", "checkpoint"):
            path = ROOT / str(run[key])
            if not path.exists():
                raise FileNotFoundError(path)
        rows, summary = evaluate_mode(
            mode="exact",
            config_path=str(ROOT / str(run["config"])),
            checkpoint_path=str(ROOT / str(run["checkpoint"])),
            data_root=data_root,
            device=device,
            max_images=24,
            start_index=0,
            patch_size=None,
            reference={},
        )
        for row in rows:
            row.update(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "selected_step": run["step"],
                    "config": run["config"],
                    "checkpoint": run["checkpoint"],
                }
            )
        summary.update(
            {
                "method": run["method"],
                "seed": run["seed"],
                "selected_step": run["step"],
                "config": run["config"],
                "checkpoint": run["checkpoint"],
            }
        )
        all_rows.extend(rows)
        run_summaries.append(summary)

    fieldnames = sorted({key for row in all_rows for key in row})
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    method_order = ["HCS", "old gate0.25", "min090", "local cap080/rho1"]
    seed_order = ["1234", "2345", "3456"]
    rows_by_method = {method: [row for row in all_rows if row["method"] == method] for method in method_order}
    aggregate = [summarize_rows(method, rows_by_method[method]) for method in method_order]
    agg_by_method = {row["method"]: row for row in aggregate}

    per_seed = []
    for seed in seed_order:
        seed_items = {row["method"]: row for row in run_summaries if row["seed"] == seed}
        item: dict[str, float | int | str] = {"seed": seed}
        for method in method_order:
            summary = seed_items[method]
            item[f"{method}_step"] = str(summary["selected_step"])
            item[f"{method}_rd"] = float(summary["mean_rd"])
            item[f"{method}_nonfinite_rows"] = int(summary["nonfinite_rows"])
        item["local_minus_hcs"] = float(item["local cap080/rho1_rd"]) - float(item["HCS_rd"])
        item["local_minus_old"] = float(item["local cap080/rho1_rd"]) - float(item["old gate0.25_rd"])
        item["local_minus_min090"] = float(item["local cap080/rho1_rd"]) - float(item["min090_rd"])
        per_seed.append(item)

    deltas = {
        "local_minus_hcs": float(agg_by_method["local cap080/rho1"]["mean_rd"]) - float(agg_by_method["HCS"]["mean_rd"]),
        "local_minus_old": float(agg_by_method["local cap080/rho1"]["mean_rd"]) - float(agg_by_method["old gate0.25"]["mean_rd"]),
        "local_minus_min090": float(agg_by_method["local cap080/rho1"]["mean_rd"]) - float(agg_by_method["min090"]["mean_rd"]),
        "old_minus_hcs": float(agg_by_method["old gate0.25"]["mean_rd"]) - float(agg_by_method["HCS"]["mean_rd"]),
        "min090_minus_hcs": float(agg_by_method["min090"]["mean_rd"]) - float(agg_by_method["HCS"]["mean_rd"]),
    }

    aligned = []
    for seed in seed_order:
        seed_rows = [row for row in all_rows if row["seed"] == seed]
        by_key = {(row["method"], row["path"]): row for row in seed_rows}
        paths = sorted({row["path"] for row in seed_rows})
        for path in paths:
            hcs = by_key[("HCS", path)]
            old = by_key[("old gate0.25", path)]
            min090 = by_key[("min090", path)]
            local = by_key[("local cap080/rho1", path)]
            item = {
                "seed": seed,
                "path": path,
                "hcs_rd": float(hcs["rd_score"]),
                "old_rd": float(old["rd_score"]),
                "min090_rd": float(min090["rd_score"]),
                "local_rd": float(local["rd_score"]),
            }
            item["local_minus_hcs"] = item["local_rd"] - item["hcs_rd"]
            item["local_minus_old"] = item["local_rd"] - item["old_rd"]
            item["local_minus_min090"] = item["local_rd"] - item["min090_rd"]
            for key in FEATURE_KEYS:
                value = maybe_float(local, key)
                if value is not None:
                    item[key] = value
            aligned.append(item)

    wins = {
        "local_better_than_hcs": sum(row["local_minus_hcs"] < 0.0 for row in aligned),
        "local_better_than_old": sum(row["local_minus_old"] < 0.0 for row in aligned),
        "local_better_than_min090": sum(row["local_minus_min090"] < 0.0 for row in aligned),
        "total": len(aligned),
    }

    sorted_rows = sorted(aligned, key=lambda row: row["hcs_rd"])
    quartiles = []
    qsize = len(sorted_rows) // 4
    for qi in range(4):
        qrows = sorted_rows[qi * qsize : (qi + 1) * qsize]
        quartiles.append(
            {
                "quartile": f"Q{qi + 1}",
                "num_images": len(qrows),
                "hcs_rd_min": qrows[0]["hcs_rd"],
                "hcs_rd_max": qrows[-1]["hcs_rd"],
                "old_minus_hcs": mean([row["old_rd"] - row["hcs_rd"] for row in qrows]),
                "min090_minus_hcs": mean([row["min090_rd"] - row["hcs_rd"] for row in qrows]),
                "local_minus_hcs": mean([row["local_minus_hcs"] for row in qrows]),
                "local_minus_old": mean([row["local_minus_old"] for row in qrows]),
                "local_minus_min090": mean([row["local_minus_min090"] for row in qrows]),
            }
        )

    correlations = [
        {
            "feature": feature,
            "r_with_local_minus_hcs": pearson(values, [row["local_minus_hcs"] for row in aligned]),
            "r_with_local_minus_old": pearson(values, [row["local_minus_old"] for row in aligned]),
        }
        for feature, values in {
            "HCS RD difficulty": [row["hcs_rd"] for row in aligned],
            "local s_q_mean": [row.get("rvq_s_q_mean", float("nan")) for row in aligned],
            "local strength": [row.get("rvq_householder_strength", float("nan")) for row in aligned],
            "local delta RMS": [row.get("rvq_householder_delta_rms", float("nan")) for row in aligned],
            "local local-delta mean": [row.get("rvq_householder_delta_rms_local_mean", float("nan")) for row in aligned],
            "local qMSE": [row.get("rvq_latent_quant_mse", float("nan")) for row in aligned],
            "local risk multiplier": [row.get("rvq_householder_risk_multiplier", float("nan")) for row in aligned],
        }.items()
    ]

    result = {
        "split": "Kodak",
        "data_root": data_root,
        "device": str(device),
        "runs": RUNS,
        "run_summaries": run_summaries,
        "aggregate": aggregate,
        "deltas": deltas,
        "per_seed": per_seed,
        "wins": wins,
        "quartiles_by_hcs_difficulty": quartiles,
        "correlations": correlations,
        "rd_std_across_seeds": {
            method: std([float(row[f"{method}_rd"]) for row in per_seed]) for method in method_order
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Local Cap080/rho1 Multi-Seed Kodak Audit",
        "",
        "Kodak 24 images, same checkpoint selections as the trusted OpenImages holdout4096 protocol. All rows are exact-inverse direct probes.",
        "",
        "## Mean Across Seeds",
        "",
        "| method | RD | RD std(seed) | bpp | PSNR | MS-SSIM | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        method = str(row["method"])
        lines.append(
            "| {method} | {rd} | {stdv} | {bpp} | {psnr} | {msssim} | {sq} | {strength} | {delta} | {ldmean} | {qmse} | {perp} | {dead} | {nonfinite} |".format(
                method=method,
                rd=fmt(float(row["mean_rd"])),
                stdv=fmt(float(result["rd_std_across_seeds"][method])),
                bpp=fmt(float(row["mean_bpp"])),
                psnr=fmt(float(row["mean_psnr"])),
                msssim=fmt(float(row["mean_ms_ssim"])),
                sq=fmt(float(row.get("mean_rvq_s_q_mean", float("nan")))),
                strength=fmt(float(row.get("mean_rvq_householder_strength", float("nan")))),
                delta=fmt(float(row.get("mean_rvq_householder_delta_rms", float("nan")))),
                ldmean=fmt(float(row.get("mean_rvq_householder_delta_rms_local_mean", float("nan")))),
                qmse=fmt(float(row.get("mean_rvq_latent_quant_mse", float("nan")))),
                perp=fmt(float(row.get("mean_rvq_perplexity", float("nan")))),
                dead=fmt(float(row.get("mean_rvq_dead_code_ratio", float("nan")))),
                nonfinite=int(row["nonfinite_rows"]),
            )
        )

    lines.extend(
        [
            "",
            "## Local-Cap Deltas",
            "",
            "| comparison | delta RD |",
            "|---|---:|",
            f"| local - HCS | {fmt(deltas['local_minus_hcs'], signed=True)} |",
            f"| local - old gate0.25 | {fmt(deltas['local_minus_old'], signed=True)} |",
            f"| local - min090 | {fmt(deltas['local_minus_min090'], signed=True)} |",
            f"| old gate0.25 - HCS | {fmt(deltas['old_minus_hcs'], signed=True)} |",
            f"| min090 - HCS | {fmt(deltas['min090_minus_hcs'], signed=True)} |",
            "",
            "## Per-Seed Checkpoint Results",
            "",
            "| seed | HCS RD | old RD | min090 RD | local RD | local-HCS | local-old | local-min090 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        lines.append(
            "| {seed} | {hcs} | {old} | {min090} | {local} | {dh} | {do} | {dm} |".format(
                seed=row["seed"],
                hcs=fmt(float(row["HCS_rd"])),
                old=fmt(float(row["old gate0.25_rd"])),
                min090=fmt(float(row["min090_rd"])),
                local=fmt(float(row["local cap080/rho1_rd"])),
                dh=fmt(float(row["local_minus_hcs"]), signed=True),
                do=fmt(float(row["local_minus_old"]), signed=True),
                dm=fmt(float(row["local_minus_min090"]), signed=True),
            )
        )

    lines.extend(
        [
            "",
            "## HCS-Difficulty Quartiles",
            "",
            "| quartile | HCS RD range | old-HCS | min090-HCS | local-HCS | local-old | local-min090 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quartiles:
        lines.append(
            "| {q} | {lo}-{hi} | {old} | {min090} | {local} | {lo_old} | {lo_min} |".format(
                q=row["quartile"],
                lo=fmt(float(row["hcs_rd_min"])),
                hi=fmt(float(row["hcs_rd_max"])),
                old=fmt(float(row["old_minus_hcs"]), signed=True),
                min090=fmt(float(row["min090_minus_hcs"]), signed=True),
                local=fmt(float(row["local_minus_hcs"]), signed=True),
                lo_old=fmt(float(row["local_minus_old"]), signed=True),
                lo_min=fmt(float(row["local_minus_min090"]), signed=True),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Image Wins",
            "",
            f"- local better than HCS: {wins['local_better_than_hcs']} / {wins['total']}",
            f"- local better than old gate0.25: {wins['local_better_than_old']} / {wins['total']}",
            f"- local better than min090: {wins['local_better_than_min090']} / {wins['total']}",
            "",
            "## Correlations",
            "",
            "| feature | r with local-HCS | r with local-old |",
            "|---|---:|---:|",
        ]
    )
    for row in correlations:
        lines.append(
            f"| {row['feature']} | {fmt(float(row['r_with_local_minus_hcs']), signed=True)} | {fmt(float(row['r_with_local_minus_old']), signed=True)} |"
        )

    if deltas["local_minus_hcs"] < 0.0 and deltas["local_minus_old"] < 0.0 and deltas["local_minus_min090"] < 0.0:
        conclusion = "Local cap080/rho1 transfers positively to Kodak under the same checkpoint protocol."
    else:
        conclusion = "Local cap080/rho1 does not cleanly dominate all Kodak baselines under this checkpoint protocol."
    lines.extend(["", "Conclusion:", "", f"- {conclusion}"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(OUT_MD), "deltas": deltas, "wins": wins}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
