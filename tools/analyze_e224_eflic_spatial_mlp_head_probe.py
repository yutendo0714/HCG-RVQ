#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e221_eflic_spatial_quant_mse_probe import safe_float, summarize_local, write_csv  # noqa: E402
from analyze_e222_eflic_spatial_linear_controller import read_rows  # noqa: E402
from analyze_e223_eflic_spatial_normalized_controller import best_threshold, infer_feature_names, make_design  # noqa: E402


@dataclass(frozen=True)
class HeadPolicy:
    feature_mode: str
    threshold: float
    train_delta_mse: float
    target_scale: float
    x_mean: list[float]
    x_std: list[float]
    state_dict: dict[str, torch.Tensor]
    train_loss: float


class SpatialMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(x), x, mean[None, :])
    std = filled.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (filled - mean[None, :]) / std[None, :], mean, std


def standardize_eval(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, mean[None, :])
    return (filled - mean[None, :]) / std[None, :]


def rows_delta(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([safe_float(row["delta_mse"]) for row in rows], dtype=float)


def fit_head(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    feature_mode: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    target_scale: float,
    harmful_weight: float,
    seed: int,
    device: str,
) -> HeadPolicy:
    set_seed(seed)
    x, _ = make_design(rows, feature_names, feature_mode)
    y = rows_delta(rows)
    x_std, mean, std = standardize_train(x)
    finite_y = np.isfinite(y)
    if not finite_y.all():
        x_std = x_std[finite_y]
        y = y[finite_y]
        rows = [row for row, ok in zip(rows, finite_y) if ok]

    x_t = torch.from_numpy(x_std.astype(np.float32)).to(device)
    y_t = torch.from_numpy((y * target_scale).astype(np.float32)).to(device)
    w = np.where(y > 0, harmful_weight, 1.0).astype(np.float32)
    w_t = torch.from_numpy(w).to(device)
    model = SpatialMLPHead(x_t.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_t.shape[0]
    last_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_t[idx])
            loss = (F.smooth_l1_loss(pred, y_t[idx], reduction="none") * w_t[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(idx.numel())
            count += int(idx.numel())
        last_loss = total / max(1, count)

    model.eval()
    with torch.no_grad():
        pred = (model(x_t).detach().cpu().numpy().astype(float) / target_scale)
    threshold, train_score, _ = best_threshold(rows, pred, "pooled")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return HeadPolicy(
        feature_mode=feature_mode,
        threshold=float(threshold),
        train_delta_mse=float(train_score),
        target_scale=float(target_scale),
        x_mean=[float(v) for v in mean],
        x_std=[float(v) for v in std],
        state_dict=state,
        train_loss=float(last_loss),
    )


def eval_head(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    policy: HeadPolicy,
    hidden: int,
    dropout: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    x, _ = make_design(rows, feature_names, policy.feature_mode)
    x_std = standardize_eval(x, np.array(policy.x_mean, dtype=float), np.array(policy.x_std, dtype=float))
    model = SpatialMLPHead(x_std.shape[1], hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(policy.state_dict)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_std.astype(np.float32)).to(device)).detach().cpu().numpy().astype(float)
    pred = pred / policy.target_scale
    active = np.isfinite(pred) & (pred <= policy.threshold)
    summary = summarize_local(rows, active)
    summary.update(
        {
            "feature_mode": policy.feature_mode,
            "threshold": policy.threshold,
            "train_delta_mse": policy.train_delta_mse,
            "train_loss": policy.train_loss,
            "pred_mean": float(np.nanmean(pred)),
            "pred_std": float(np.nanstd(pred)),
        }
    )
    return active, summary, pred


def add_summary(summaries: list[dict[str, Any]], group: str, policy: str, rows: list[dict[str, Any]], active: np.ndarray, **extra: Any) -> None:
    summary = summarize_local(rows, active)
    summary.update({"group": group, "policy": policy})
    summary.update(extra)
    summaries.append(summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--feature-modes", default="raw_plus_image_rel,raw_plus_image_slice_rel")
    p.add_argument("--min-finite-frac", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-scale", type=float, default=1000.0)
    p.add_argument("--harmful-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=224)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rows = read_rows(args.samples)
    if not rows:
        raise SystemExit("no samples")
    feature_names = infer_feature_names(rows, args.min_finite_frac)
    feature_modes = [x for x in args.feature_modes.split(",") if x]
    datasets = sorted({row["dataset"] for row in rows})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested {args.device}, but CUDA is unavailable")

    summaries: list[dict[str, Any]] = []
    delta = rows_delta(rows)
    add_summary(summaries, "pooled", "all_off", rows, np.zeros(len(rows), dtype=bool))
    add_summary(summaries, "pooled", "all_on", rows, np.ones(len(rows), dtype=bool))
    add_summary(summaries, "pooled", "oracle_local", rows, delta < 0.0)

    payload_policies: dict[str, dict[str, Any]] = {}
    for feature_mode in feature_modes:
        policy = fit_head(
            rows,
            feature_names,
            feature_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
            target_scale=args.target_scale,
            harmful_weight=args.harmful_weight,
            seed=args.seed,
            device=args.device,
        )
        _, summary, _ = eval_head(rows, feature_names, policy, args.hidden, args.dropout, args.device)
        summary.update({"group": "pooled", "policy": f"same_table_mlp_{feature_mode}"})
        summaries.append(summary)
        payload_policies[f"same_table_{feature_mode}"] = {
            "feature_mode": feature_mode,
            "threshold": policy.threshold,
            "train_delta_mse": policy.train_delta_mse,
            "train_loss": policy.train_loss,
        }

        for held in datasets:
            train_rows = [row for row in rows if row["dataset"] != held]
            eval_rows = [row for row in rows if row["dataset"] == held]
            held_policy = fit_head(
                train_rows,
                feature_names,
                feature_mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden,
                dropout=args.dropout,
                target_scale=args.target_scale,
                harmful_weight=args.harmful_weight,
                seed=args.seed + 17 + datasets.index(held),
                device=args.device,
            )
            _, held_summary, _ = eval_head(eval_rows, feature_names, held_policy, args.hidden, args.dropout, args.device)
            held_summary.update(
                {
                    "group": held,
                    "policy": f"leave_dataset_out_mlp_{feature_mode}",
                    "train_group": "+".join(sorted({row["dataset"] for row in train_rows})),
                }
            )
            summaries.append(held_summary)

    summary_csv = args.output_prefix.with_suffix(".summary.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(summary_csv, summaries)
    payload = {
        "samples": len(rows),
        "base_features": feature_names,
        "feature_modes": feature_modes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "target_scale": args.target_scale,
        "harmful_weight": args.harmful_weight,
        "seed": args.seed,
        "device": args.device,
        "summaries": summaries,
        "policies": payload_policies,
        "interpretation": "Teacher-label MLP diagnostic for EF-LIC local HCG head transfer; not a final codec method.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E224 EF-LIC Spatial MLP Head Probe",
        "",
        "This trains a small local MLP head on E221 spatial quant-MSE teacher labels.",
        "It is a bridge toward an in-codec learned HCG head, not a final paper metric row.",
        "",
        f"Samples: `{len(rows)}`",
        f"Base features: `{len(feature_names)}`",
        f"Epochs: `{args.epochs}`",
        f"Device: `{args.device}`",
        "",
        "| group | policy | samples | dMSE | all-on dMSE | oracle dMSE | active | helpful | precision | recall | threshold | train dMSE | train loss | train |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {group} | {policy} | {samples} | {delta_mse:+.8f} | {all_on_delta_mse:+.8f} | {oracle_delta_mse:+.8f} | {active_frac:.3f} | {helpful_frac:.3f} | {precision:.3f} | {recall:.3f} | {threshold} | {train_delta_mse} | {train_loss} | {train_group} |".format(
                group=row.get("group", ""),
                policy=row.get("policy", ""),
                samples=int(row.get("samples", 0)),
                delta_mse=float(row.get("delta_mse", 0.0)),
                all_on_delta_mse=float(row.get("all_on_delta_mse", 0.0)),
                oracle_delta_mse=float(row.get("oracle_delta_mse", 0.0)),
                active_frac=float(row.get("active_frac", 0.0)),
                helpful_frac=float(row.get("helpful_frac", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                threshold=row.get("threshold", ""),
                train_delta_mse=row.get("train_delta_mse", ""),
                train_loss=row.get("train_loss", ""),
                train_group=row.get("train_group", ""),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Same-table MLP rows test whether the feature/label pair has enough capacity for a learned head.",
            "- Leave-dataset-out rows decide whether teacher-label training already transfers.",
            "- If LODO remains weak, move the learned local head into codec-aware training rather than relying on teacher labels alone.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
