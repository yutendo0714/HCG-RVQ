#!/usr/bin/env python3
"""Posthoc s_q risk-multiplier sweep for low-rate HCG bias010 checkpoints."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_msssim, compute_psnr
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config

ANALYSIS = ROOT / "experiments" / "analysis"
PREFIX = ANALYSIS / "e142_lowrate_sq_risk_posthoc"

RUNS = [
    {
        "seed": "1234",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed1234/checkpoint_step_250.pth.tar",
        "hcs_rd": 1.196004474941292,
    },
    {
        "seed": "2345",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed2345.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed2345/checkpoint_step_250.pth.tar",
        "hcs_rd": 1.2044472785776208,
    },
    {
        "seed": "3456",
        "config": "configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed3456.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_bias010_frozen_g64_l1_k128_lambda0018_seed3456/checkpoint_step_500.pth.tar",
        "hcs_rd": 1.1782728698712404,
    },
]

CENTERS = (0.65, 0.70, 0.75, 0.80, 0.85)
MINS = (0.0, 0.25, 0.50, 0.75)
SHARPNESS = 16.0


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def tensor_scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_with_sq_risk(config_path: str, center: float, minimum: float) -> dict:
    config = load_config(ROOT / config_path)
    qcfg = dict(config.get("quantizer", {}))
    qcfg.update(
        {
            "householder_gate_risk_enabled": True,
            "householder_gate_risk_center": center,
            "householder_gate_risk_sharpness": SHARPNESS,
            "householder_gate_risk_min": minimum,
            "householder_gate_risk_invert": False,
            "householder_gate_risk_detach": True,
        }
    )
    config["quantizer"] = qcfg
    return config


def evaluate_run(run: dict[str, object], center: float, minimum: float, device: torch.device) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = config_with_sq_risk(str(run["config"]), center, minimum)
    model = build_model(config).to(device)
    checkpoint = torch.load(ROOT / str(run["checkpoint"]), map_location=device)
    load_info = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    loss_cfg = dict(config.get("loss", {}))
    loss_cfg["rho_householder_reliability_teacher"] = 0.0
    loss_cfg["rho_householder_residual_selector_teacher"] = 0.0
    loss_cfg["rho_householder_residual_selector_noop"] = 0.0
    criterion = RateDistortionLoss(**loss_cfg)
    dataset = ImageFolderDataset(["/dpl/kodak"], training=False, max_images=24, start_index=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.get("eval", {}).get("num_workers", 2))

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for index, x in enumerate(tqdm(loader, desc=f"{run['seed']} c{center:.2f} m{minimum:.2f}", dynamic_ncols=True)):
            x = x.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            mse = float(losses["mse"].cpu())
            bpp = float(losses["bpp_total"].cpu())
            rd = bpp + float(loss_cfg["lambda_rd"]) * float(loss_cfg.get("mse_scale", 255.0 * 255.0)) * mse
            row: dict[str, object] = {
                "seed": run["seed"],
                "index": index,
                "path": str(dataset.paths[index]),
                "risk_center": center,
                "risk_min": minimum,
                "rd_score": rd,
                "rd_minus_hcs": rd - float(run["hcs_rd"]),
                "bpp": bpp,
                "bpp_y": float(losses["bpp_y"].cpu()),
                "bpp_z": float(losses["bpp_z"].cpu()),
                "mse": mse,
                "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
                "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
            }
            for key, value in output.get("rvq_stats", {}).items():
                scalar = tensor_scalar(value)
                if scalar is not None:
                    row[f"rvq_{key}"] = scalar
            row["has_nonfinite"] = int(
                not all(math.isfinite(float(v)) for k, v in row.items() if isinstance(v, (float, int)) and k != "index")
            )
            rows.append(row)

    summary: dict[str, object] = {
        "seed": run["seed"],
        "risk_center": center,
        "risk_min": minimum,
        "rd_score": mean([float(row["rd_score"]) for row in rows]),
        "rd_minus_hcs": mean([float(row["rd_minus_hcs"]) for row in rows]),
        "bpp": mean([float(row["bpp"]) for row in rows]),
        "psnr": mean([float(row["psnr"]) for row in rows]),
        "ms_ssim": mean([float(row["ms_ssim"]) for row in rows]),
        "nonfinite_rows": sum(int(row["has_nonfinite"]) for row in rows),
        "missing_keys": ";".join(load_info.missing_keys),
        "unexpected_keys": ";".join(load_info.unexpected_keys),
    }
    for key in (
        "rvq_householder_risk_multiplier",
        "rvq_householder_strength",
        "rvq_householder_delta_rms",
        "rvq_householder_delta_rms_local_mean",
        "rvq_latent_quant_mse",
        "rvq_dead_code_ratio",
        "rvq_perplexity",
        "rvq_index_empirical_bpp",
        "rvq_s_q_mean",
    ):
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            summary[key] = mean(vals)
    return rows, summary


def aggregate(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        groups[(float(row["risk_center"]), float(row["risk_min"]))].append(row)

    out: list[dict[str, object]] = []
    for (center, minimum), rows in groups.items():
        out.append(
            {
                "risk_center": center,
                "risk_min": minimum,
                "mean_rd": mean([float(row["rd_score"]) for row in rows]),
                "mean_rd_minus_hcs": mean([float(row["rd_minus_hcs"]) for row in rows]),
                "win_seed_count": sum(float(row["rd_minus_hcs"]) < 0.0 for row in rows),
                "seed1234_delta": next(float(row["rd_minus_hcs"]) for row in rows if row["seed"] == "1234"),
                "seed2345_delta": next(float(row["rd_minus_hcs"]) for row in rows if row["seed"] == "2345"),
                "seed3456_delta": next(float(row["rd_minus_hcs"]) for row in rows if row["seed"] == "3456"),
                "mean_risk_multiplier": mean([float(row.get("rvq_householder_risk_multiplier", float("nan"))) for row in rows]),
                "mean_delta_rms": mean([float(row.get("rvq_householder_delta_rms", float("nan"))) for row in rows]),
                "mean_dead_code": mean([float(row.get("rvq_dead_code_ratio", float("nan"))) for row in rows]),
                "nonfinite_rows": sum(int(row["nonfinite_rows"]) for row in rows),
            }
        )
    return sorted(out, key=lambda row: float(row["mean_rd_minus_hcs"]))


def write_markdown(agg_rows: list[dict[str, object]]) -> None:
    best = agg_rows[0]
    lines = [
        "# E142 Low-Rate s_q Risk Posthoc Sweep",
        "",
        "This diagnostic applies a decoder-known local `s_q` risk multiplier to the already-trained E140 HCG `bias010` checkpoints. It is posthoc and not a paper-main codec result.",
        "",
        "## Best Setting",
        "",
        f"- center: `{fmt(float(best['risk_center']))}`",
        f"- min multiplier: `{fmt(float(best['risk_min']))}`",
        f"- mean delta vs HCS: `{fmt(float(best['mean_rd_minus_hcs']), True)}`",
        f"- seed deltas: 1234 `{fmt(float(best['seed1234_delta']), True)}`, 2345 `{fmt(float(best['seed2345_delta']), True)}`, 3456 `{fmt(float(best['seed3456_delta']), True)}`",
        f"- mean risk multiplier: `{fmt(float(best['mean_risk_multiplier']))}`",
        f"- nonfinite rows: `{best['nonfinite_rows']}`",
        "",
        "## Top Settings",
        "",
        "| center | min | mean delta | wins | seed1234 | seed2345 | seed3456 | risk mult | delta RMS | dead code | nonfinite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows[:12]:
        lines.append(
            "| {center} | {minimum} | {delta} | {wins}/3 | {s1234} | {s2345} | {s3456} | {risk} | {drms} | {dead} | {nf} |".format(
                center=fmt(float(row["risk_center"])),
                minimum=fmt(float(row["risk_min"])),
                delta=fmt(float(row["mean_rd_minus_hcs"]), True),
                wins=row["win_seed_count"],
                s1234=fmt(float(row["seed1234_delta"]), True),
                s2345=fmt(float(row["seed2345_delta"]), True),
                s3456=fmt(float(row["seed3456_delta"]), True),
                risk=fmt(float(row["mean_risk_multiplier"])),
                drms=fmt(float(row["mean_delta_rms"])),
                dead=fmt(float(row["mean_dead_code"])),
                nf=row["nonfinite_rows"],
            )
        )
    if float(best["mean_rd_minus_hcs"]) < -0.001171:
        decision = (
            "The best posthoc row beats fixed HCG on the three-seed mean. The next real experiment should train this "
            "control as a fixed checkpoint and then evaluate holdout4096 plus feature distributions."
        )
    else:
        decision = (
            "The posthoc local multiplier does not beat fixed HCG and worsens the fragile seed3456. Keep the image-level "
            "`s_q` selector as headroom evidence, but do not promote this continuous local multiplier. The next controller "
            "should use image-level or learned reliability control rather than directly shrinking local geometry from `s_q`."
        )
    lines.extend(["", "## Decision", "", decision])
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    for run in RUNS:
        for key in ("config", "checkpoint"):
            path = ROOT / str(run[key])
            if not path.exists():
                raise FileNotFoundError(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for center in CENTERS:
        for minimum in MINS:
            for run in RUNS:
                rows, summary = evaluate_run(run, center, minimum, device)
                all_rows.extend(rows)
                summaries.append(summary)
    agg_rows = aggregate(summaries)

    result = {
        "experiment": "E142 low-rate s_q risk posthoc sweep",
        "centers": CENTERS,
        "mins": MINS,
        "sharpness": SHARPNESS,
        "runs": RUNS,
        "aggregate": agg_rows,
        "summaries": summaries,
        "decision": "posthoc diagnostic only; promote only after fixed-checkpoint training and holdout confirmation",
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(PREFIX.with_suffix(".all_rows.csv"), all_rows)
    write_csv(PREFIX.with_suffix(".summary.csv"), summaries)
    write_csv(PREFIX.with_suffix(".aggregate.csv"), agg_rows)
    write_markdown(agg_rows)
    print(PREFIX.with_suffix(".md"))
    print(json.dumps(agg_rows[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
