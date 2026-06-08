#!/usr/bin/env python3
"""Train the EF-LIC HCG branch controller on saved decoder-safe context tensors.

E295 verifies that `EFLICHCGBranchController` can sit in the EF-LIC codec loop.
This script verifies the next link: the same controller can be optimized on the
E242 teacher-context tensors, saved, and later loaded by E295 for codec-loop
evaluation. The labels are still teacher/smoke labels, not final paper evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import (  # noqa: E402
    EFLICHCGBranchController,
    EFLICHCGBranchControllerConfig,
    asymmetric_family_loss,
)
from hcg_rvq.reliability_index_controller import reliability_index_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kodak-manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e242_eflic_spatial_teacher_contexts_kodak24/manifest_kodak24_n24.csv",
    )
    p.add_argument(
        "--clicpro-manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e242_eflic_spatial_teacher_contexts_clicpro41/manifest_clicpro41_n41.csv",
    )
    p.add_argument("--train-per-domain", type=int, default=8)
    p.add_argument("--eval-per-domain", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hidden-channels", type=int, default=24)
    p.add_argument("--max-alpha", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=296)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=1.0)
    p.add_argument("--score-weight", type=float, default=0.1)
    p.add_argument("--alpha-weight", type=float, default=20.0)
    p.add_argument("--family-weight", type=float, default=0.1)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e296_eflic_hcg_controller_context_train_smoke",
    )
    return p.parse_args()


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def load_manifest(path: Path, *, train_count: int, eval_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            tensor_path = Path(row["tensor_path"])
            if not tensor_path.is_absolute():
                tensor_path = ROOT / tensor_path
            rows.append({"dataset": row["dataset"], "image": row["image"], "tensor_path": tensor_path})
    return rows[:train_count], rows[train_count : train_count + eval_count]


def load_tensor(record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    obj = torch.load(record["tensor_path"], map_location=device)
    context = obj["context_maps"].float()
    alpha = obj["alpha_target"].float()
    target_map = obj["target_map"].long()
    risk_target = obj.get("risk_target")
    if risk_target is not None:
        risk_target = risk_target.float()
    if context.ndim != 4 or context.shape[1] != 11:
        raise RuntimeError(f"unexpected context shape {tuple(context.shape)} in {record['tensor_path']}")
    if alpha.shape != (context.shape[0], 1, context.shape[2], context.shape[3]):
        raise RuntimeError(f"unexpected alpha shape {tuple(alpha.shape)} in {record['tensor_path']}")
    if target_map.shape != (context.shape[0], context.shape[2], context.shape[3]):
        raise RuntimeError(f"unexpected target shape {tuple(target_map.shape)} in {record['tensor_path']}")
    if not torch.isfinite(context).all().item() or not torch.isfinite(alpha).all().item():
        raise RuntimeError(f"nonfinite tensor in {record['tensor_path']}")
    if risk_target is not None:
        if risk_target.shape != alpha.shape:
            raise RuntimeError(f"unexpected risk_target shape {tuple(risk_target.shape)} in {record['tensor_path']}")
        if not torch.isfinite(risk_target).all().item():
            raise RuntimeError(f"nonfinite risk_target in {record['tensor_path']}")
    return {"context": context, "alpha": alpha, "target_map": target_map, "risk_target": risk_target}


def controller_objective(
    controller: EFLICHCGBranchController,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = batch["context"]
    alpha = batch["alpha"]
    target_map = batch["target_map"]
    active = (target_map > 0).unsqueeze(1).to(dtype=context.dtype)
    target_score = batch.get("risk_target")
    if target_score is None:
        target_score = torch.where(active > 0, -alpha.abs(), torch.zeros_like(alpha))
    out = controller(context, hard=False, force_zero=False, risk_temperature=0.5)
    rel_loss = reliability_index_loss(
        out["active_logit"],
        active,
        risk_score=out["risk_score"],
        target_score=target_score,
        false_positive_weight=args.false_positive_weight,
        missed_active_weight=args.missed_active_weight,
        score_weight=args.score_weight,
    )
    alpha_loss = F.smooth_l1_loss(out["alpha_map"], alpha)
    family_loss = asymmetric_family_loss(
        out["family_logits"],
        target_map,
        false_positive_weight=args.false_positive_weight,
        missed_active_weight=args.missed_active_weight,
    )
    loss = rel_loss + float(args.alpha_weight) * alpha_loss + float(args.family_weight) * family_loss
    hard = controller(context, hard=True, force_zero=False)
    force_zero = controller(context, hard=True, force_zero=True)
    metrics = {
        "loss": float(loss.detach().item()),
        "rel_loss": float(rel_loss.detach().item()),
        "alpha_loss": float(alpha_loss.detach().item()),
        "family_loss": float(family_loss.detach().item()),
        "target_active_frac": float(active.mean().item()),
        "target_alpha_mean": float(alpha.mean().item()),
        "pred_alpha_mean": float(out["alpha_map"].detach().mean().item()),
        "pred_alpha_max": float(out["alpha_map"].detach().max().item()),
        "soft_gate_mean": float(out["gate"].detach().float().mean().item()),
        "hard_gate_mean": float(hard["gate"].detach().float().mean().item()),
        "hard_alpha_mean": float(hard["alpha_map"].detach().float().mean().item()),
        "force_zero_alpha_max": float(force_zero["alpha_map"].detach().abs().max().item()),
        "family_zero_prob_mean": float(out["family_logits"].detach().softmax(dim=1)[:, 0:1].mean().item()),
    }
    return loss, metrics


def evaluate(
    controller: EFLICHCGBranchController,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    by_dataset: dict[str, dict[str, list[float]]] = {}
    nonfinite = 0
    with torch.no_grad():
        for record in records:
            batch = load_tensor(record, device)
            loss, metrics = controller_objective(controller, batch, args)
            metrics = dict(metrics)
            metrics["loss"] = float(loss.item())
            if not all(math.isfinite(v) for v in metrics.values()):
                nonfinite += 1
            for key, value in metrics.items():
                values.setdefault(key, []).append(value)
            domain = by_dataset.setdefault(record["dataset"], {})
            for key, value in metrics.items():
                domain.setdefault(key, []).append(value)
    summary = {key: _mean(vals) for key, vals in values.items()}
    summary["records"] = len(records)
    summary["nonfinite_records"] = nonfinite
    summary["by_dataset"] = {
        name: {key: _mean(vals) for key, vals in domain.items()} | {"records": len(next(iter(domain.values()), []))}
        for name, domain in by_dataset.items()
    }
    return summary


def write_outputs(
    *,
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    train_trace: list[dict[str, float]],
    state_path: Path,
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    trace_path = args.output_prefix.with_suffix(".trace.csv")
    with trace_path.open("w", newline="") as fobj:
        fieldnames = sorted({key for row in train_trace for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_trace)
    arg_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "experiment": "E296 EF-LIC HCG controller context train smoke",
        "purpose": "Verify that the integrated EF-LIC HCG controller trains on E242 teacher contexts and can be saved for E295 codec-loop loading.",
        "args": arg_payload | {"output_prefix": str(args.output_prefix), "device": str(args.device)},
        "state_path": str(state_path),
        "train_records": [{"dataset": r["dataset"], "image": r["image"]} for r in train_records],
        "eval_records": [{"dataset": r["dataset"], "image": r["image"]} for r in eval_records],
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    keys = [
        "records",
        "loss",
        "rel_loss",
        "alpha_loss",
        "family_loss",
        "target_active_frac",
        "target_alpha_mean",
        "pred_alpha_mean",
        "soft_gate_mean",
        "hard_gate_mean",
        "force_zero_alpha_max",
        "nonfinite_records",
    ]
    with md_path.open("w") as fobj:
        fobj.write("# E296 EF-LIC HCG Controller Context Train Smoke\n\n")
        fobj.write("This is a controller trainability and artifact handoff smoke, not final codec RD evidence.\n\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Steps: `{args.steps}`\n")
        fobj.write(f"- State path: `{state_path}`\n\n")
        fobj.write("| split | " + " | ".join(keys) + " |\n")
        fobj.write("|---|" + "|".join(["---"] * len(keys)) + "|\n")
        for name, summary in [("train", train_summary), ("eval", eval_summary)]:
            vals = []
            for key in keys:
                value = summary.get(key, "")
                if isinstance(value, float):
                    vals.append(f"{value:.8f}")
                else:
                    vals.append(str(value))
            fobj.write("| " + name + " | " + " | ".join(vals) + " |\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- `force_zero_alpha_max` should remain 0 because fallback is hard-coded independent of learned weights.\n")
        fobj.write("- This checkpoint is suitable for E295 codec-loop loading only as a smoke artifact; it is not paper evidence.\n")
    print(f"wrote {json_path}, {md_path}, {trace_path}, {state_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_k, eval_k = load_manifest(args.kodak_manifest, train_count=args.train_per_domain, eval_count=args.eval_per_domain)
    train_c, eval_c = load_manifest(args.clicpro_manifest, train_count=args.train_per_domain, eval_count=args.eval_per_domain)
    train_records = train_k + train_c
    eval_records = eval_k + eval_c
    if not train_records or not eval_records:
        raise SystemExit("empty train/eval split")

    controller = EFLICHCGBranchController(
        EFLICHCGBranchControllerConfig(hidden_channels=args.hidden_channels, max_alpha=args.max_alpha)
    ).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr)
    trace: list[dict[str, float]] = []
    for step in range(args.steps):
        controller.train()
        record = train_records[step % len(train_records)]
        batch = load_tensor(record, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = controller_objective(controller, batch, args)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
        optimizer.step()
        trace_row = {"step": float(step), "grad_norm": float(grad_norm.item())}
        trace_row.update(metrics)
        trace.append(trace_row)
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            print(
                f"step={step+1}/{args.steps} loss={metrics['loss']:.4f} "
                f"alpha={metrics['pred_alpha_mean']:.6f} soft_gate={metrics['soft_gate_mean']:.4f} "
                f"hard_gate={metrics['hard_gate_mean']:.4f} grad={float(grad_norm.item()):.4f}"
            )

    controller.eval()
    train_summary = evaluate(controller, train_records, args, device)
    eval_summary = evaluate(controller, eval_records, args, device)
    state_path = args.output_prefix.with_suffix(".pth")
    torch.save({"model": controller.state_dict(), "config": asdict(controller.config), "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, state_path)
    write_outputs(
        args=args,
        train_records=train_records,
        eval_records=eval_records,
        train_summary=train_summary,
        eval_summary=eval_summary,
        train_trace=trace,
        state_path=state_path,
    )


if __name__ == "__main__":
    main()
