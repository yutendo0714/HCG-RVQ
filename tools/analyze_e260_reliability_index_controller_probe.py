#!/usr/bin/env python3
"""E260 reliability/index controller probe.

This is a compact module-level experiment after E259.  It verifies that the new
HCG-RVQ reliability/index controller has conservative fallback initialization
and tests the same controller contract on the existing GLC E257 per-image rows.
The experiment is intentionally small: it screens the controller design before
any codec-loop/full-training promotion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.reliability_index_controller import (  # noqa: E402
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    SpatialReliabilityIndexConfig,
    SpatialReliabilityIndexHead,
    reliability_index_loss,
    select_with_fallback,
)


FEATURES = [
    "active_mse_ratio",
    "active_scalar_mse",
    "active_rvq_mse",
    "index_entropy_mean",
    "index_used_frac_mean",
    "index_dead_frac_mean",
    "empirical_bpp_delta",
    "fixed_bpp_delta",
    "base_bpp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "analysis"
        / "e257_glc_domain_mixed_with_cliccalib_gate_readiness.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "e260_reliability_index_controller_probe",
    )
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=260)
    return p.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = dict(row)
            item["score_with_side"] = finite(row.get("score_with_side"))
            item["oracle_select"] = str(row.get("oracle_select", "")).lower() == "true"
            for feature in FEATURES:
                item[feature] = finite(row.get(feature))
            rows.append(item)
    if not rows:
        raise SystemExit(f"no rows loaded from {path}")
    return rows


def standardize(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    stds = []
    for feature in FEATURES:
        vals = [finite(row.get(feature), 0.0) for row in train]
        mu = mean(vals)
        var = mean([(v - mu) ** 2 for v in vals])
        means.append(mu)
        stds.append(math.sqrt(var) if math.isfinite(var) and var > 1e-12 else 1.0)

    def build(rows: list[dict[str, Any]]) -> torch.Tensor:
        matrix = []
        for row in rows:
            matrix.append([(finite(row.get(feature), means[i]) - means[i]) / stds[i] for i, feature in enumerate(FEATURES)])
        return torch.tensor(matrix, dtype=torch.float32)

    return build(train), build(eval_rows)


def summarize_policy(rows: list[dict[str, Any]], selected: list[bool], policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "selected": sum(1 for v in selected if v),
        "total": len(rows),
        "selected_frac": sum(1 for v in selected if v) / len(rows) if rows else 0.0,
        "mean_score": mean([finite(row["score_with_side"]) if use else 0.0 for row, use in zip(rows, selected)]),
        "mean_positive_rate": mean([1.0 if finite(row["score_with_side"]) < 0 else 0.0 for row in rows]),
    }


def thresholds(values: list[float]) -> list[float]:
    vals = sorted({v for v in values if math.isfinite(v)})
    if not vals:
        return [0.0]
    points = [vals[0] - 1e-9, vals[-1] + 1e-9]
    points.extend(vals)
    points.extend((a + b) / 2.0 for a, b in zip(vals, vals[1:]))
    return sorted(set(points))


def train_predict(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[float], list[float], float]:
    torch.manual_seed(seed)
    x_train, x_eval = standardize(train_rows, eval_rows)
    y_active = torch.tensor([[1.0 if finite(row["score_with_side"]) < 0 else 0.0] for row in train_rows])
    y_score = torch.tensor([[finite(row["score_with_side"], 0.0)] for row in train_rows])

    model = ReliabilityIndexMLP(ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=24))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x_train)
        loss = reliability_index_loss(
            out["active_logit"],
            y_active,
            risk_score=out["risk_score"],
            target_score=y_score,
            false_positive_weight=4.0,
            missed_active_weight=1.0,
            score_weight=0.5,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_out = model(x_train)
        eval_out = model(x_eval)
    train_risk = [float(v) for v in train_out["risk_score"].flatten().tolist()]
    eval_risk = [float(v) for v in eval_out["risk_score"].flatten().tolist()]
    eval_prob = [float(v) for v in torch.sigmoid(eval_out["active_logit"]).flatten().tolist()]

    best_threshold = 0.0
    best_score = float("inf")
    for threshold in thresholds(train_risk):
        selected = [risk <= threshold for risk in train_risk]
        score = summarize_policy(train_rows, selected, "train_threshold")["mean_score"]
        if score < best_score:
            best_score = score
            best_threshold = threshold
    return eval_risk, eval_prob, best_threshold


def smoke_heads() -> dict[str, Any]:
    torch.manual_seed(260)
    spatial = SpatialReliabilityIndexHead(SpatialReliabilityIndexConfig(input_channels=11, hidden_channels=8))
    mlp = ReliabilityIndexMLP(ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=8))

    context = torch.randn(2, 11, 4, 5)
    features = torch.randn(3, len(FEATURES))
    spatial_out = spatial(context)
    mlp_out = mlp(features)

    spatial_mask = select_with_fallback(spatial_out["active_logit"], risk_score=spatial_out["risk_score"])
    mlp_mask = select_with_fallback(mlp_out["active_logit"], risk_score=mlp_out["risk_score"])
    target_map = torch.zeros_like(spatial_out["active_logit"])
    loss = reliability_index_loss(
        spatial_out["active_logit"],
        target_map,
        risk_score=spatial_out["risk_score"],
        target_score=torch.zeros_like(spatial_out["risk_score"]),
    )
    loss.backward()
    grads_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item())
        for p in list(spatial.parameters()) + list(mlp.parameters())
    )
    return {
        "spatial_active_prob_mean": float(torch.sigmoid(spatial_out["active_logit"]).mean().item()),
        "spatial_selected_frac_init": float(spatial_mask.float().mean().item()),
        "mlp_active_prob_mean": float(torch.sigmoid(mlp_out["active_logit"]).mean().item()),
        "mlp_selected_frac_init": float(mlp_mask.float().mean().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grads_finite": grads_finite,
    }


def run_protocols(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = [
        summarize_policy(rows, [False] * len(rows), "no_branch"),
        summarize_policy(rows, [True] * len(rows), "all_on"),
        summarize_policy(rows, [bool(row["oracle_select"]) for row in rows], "oracle"),
    ]

    risk, prob, threshold = train_predict(
        rows,
        rows,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    summaries.append(
        summarize_policy(rows, [r <= threshold for r in risk], "resub_reliability_index_mlp")
        | {"threshold": threshold, "mean_prob": mean(prob)}
    )

    loocv_selected: list[bool] = []
    loocv_prob: list[float] = []
    for idx, row in enumerate(rows):
        train = rows[:idx] + rows[idx + 1 :]
        risk_i, prob_i, threshold_i = train_predict(
            train,
            [row],
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + idx + 1,
        )
        loocv_selected.append(risk_i[0] <= threshold_i)
        loocv_prob.append(prob_i[0])
    summaries.append(
        summarize_policy(rows, loocv_selected, "loocv_reliability_index_mlp")
        | {"threshold": float("nan"), "mean_prob": mean(loocv_prob)}
    )

    domain_selected = [False] * len(rows)
    domain_prob = [float("nan")] * len(rows)
    for domain in sorted({str(row.get("domain")) for row in rows}):
        eval_idx = [i for i, row in enumerate(rows) if str(row.get("domain")) == domain]
        train = [row for row in rows if str(row.get("domain")) != domain]
        eval_rows = [rows[i] for i in eval_idx]
        risk_d, prob_d, threshold_d = train_predict(
            train,
            eval_rows,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + 1000 + len(domain),
        )
        for offset, idx in enumerate(eval_idx):
            domain_selected[idx] = risk_d[offset] <= threshold_d
            domain_prob[idx] = prob_d[offset]
    summaries.append(
        summarize_policy(rows, domain_selected, "leave_domain_reliability_index_mlp")
        | {"threshold": float("nan"), "mean_prob": mean(domain_prob)}
    )
    return summaries


def write_outputs(args: argparse.Namespace, smoke: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    csv_path = args.output_prefix.with_suffix(".summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["policy", "selected", "total", "selected_frac", "mean_score", "mean_positive_rate", "threshold", "mean_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    payload = {
        "rows": str(args.rows),
        "features": FEATURES,
        "smoke": smoke,
        "summaries": summaries,
        "note": "Small controller probe only; not a full-training claim.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(value: Any) -> str:
        val = finite(value)
        return "NA" if not math.isfinite(val) else f"{val:.6f}"

    lines = [
        "# E260 Reliability/Index Controller Probe",
        "",
        "This is a module-level screen for the E259 full-training target: compact "
        "local HCG-RVQ branch + reliability/index control + no-branch fallback.",
        "",
        "## Smoke",
        "",
        f"- Spatial init active probability mean: `{fmt(smoke['spatial_active_prob_mean'])}`.",
        f"- Spatial init selected fraction: `{fmt(smoke['spatial_selected_frac_init'])}`.",
        f"- MLP init active probability mean: `{fmt(smoke['mlp_active_prob_mean'])}`.",
        f"- MLP init selected fraction: `{fmt(smoke['mlp_selected_frac_init'])}`.",
        f"- Loss finite: `{smoke['loss_finite']}`; gradients finite: `{smoke['grads_finite']}`.",
        "",
        "## GLC E257 Row Probe",
        "",
        "| policy | selected | mean score | mean prob |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['policy']} | {int(row['selected'])}/{int(row['total'])} | "
            f"{fmt(row['mean_score'])} | {fmt(row.get('mean_prob'))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this probe tests the controller contract and fallback "
            "initialization. A positive held-out or leave-domain score still blocks "
            "full-training promotion; a negative score only means this controller is "
            "worth moving into a mid-scale codec-loop run.",
            "",
        ]
    )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.rows)
    smoke = smoke_heads()
    summaries = run_protocols(rows, args)
    write_outputs(args, smoke, summaries)
    print(json.dumps({"output_prefix": str(args.output_prefix), "smoke": smoke, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
