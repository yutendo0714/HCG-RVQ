#!/usr/bin/env python3
"""Train EF-LIC HCG controller on E324 E318-aligned teacher tensors.

E324 provides a cleaner bridge than E242: decoder-safe context maps with labels
aligned to the latest fallback-aware E317/E318 oracle. This script trains the
existing controller on one manifest with an explicit train/eval split, avoiding
the older E296 two-domain smoke assumptions.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from hcg_rvq.eflic_local_controller import EFLICHCGBranchController, EFLICHCGBranchControllerConfig  # noqa: E402
from run_e296_eflic_hcg_controller_context_train_smoke import controller_objective, evaluate, load_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e324_eflic_e318_aligned_context_teacher_kodak24/manifest_kodak24_n24.csv",
    )
    p.add_argument("--train-count", type=int, default=16)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-channels", type=int, default=24)
    p.add_argument("--max-alpha", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=325)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=1.0)
    p.add_argument("--score-weight", type=float, default=0.1)
    p.add_argument("--alpha-weight", type=float, default=20.0)
    p.add_argument("--family-weight", type=float, default=0.1)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e325_eflic_hcg_controller_e318_teacher_train_kodak24_t16_e8_s96",
    )
    return p.parse_args()


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            tensor_path = Path(row["tensor_path"])
            if not tensor_path.is_absolute():
                tensor_path = ROOT / tensor_path
            rows.append({"dataset": row["dataset"], "image": row["image"], "tensor_path": tensor_path})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    trace_path = args.output_prefix.with_suffix(".trace.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(trace_path, train_trace)
    arg_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "experiment": "E325 EF-LIC HCG controller training on E318-aligned teacher",
        "purpose": "Trainability check for the E324 slice-dense fallback-aware teacher bridge.",
        "args": arg_payload,
        "state_path": str(state_path),
        "train_records": [{"dataset": r["dataset"], "image": r["image"]} for r in train_records],
        "eval_records": [{"dataset": r["dataset"], "image": r["image"]} for r in eval_records],
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    keys = [
        "records",
        "loss",
        "rel_loss",
        "alpha_loss",
        "family_loss",
        "target_active_frac",
        "target_alpha_mean",
        "pred_alpha_mean",
        "pred_alpha_max",
        "soft_gate_mean",
        "hard_gate_mean",
        "hard_alpha_mean",
        "force_zero_alpha_max",
        "nonfinite_records",
    ]
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E325 EF-LIC HCG Controller E318-Teacher Train\n\n")
        fobj.write("This is a trainability check for E324's E318-aligned slice-dense teacher, not final codec RD evidence.\n\n")
        fobj.write(f"- Manifest: `{args.manifest}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Train/eval images: `{len(train_records)}` / `{len(eval_records)}`\n")
        fobj.write(f"- Steps: `{args.steps}`\n")
        fobj.write(f"- State path: `{state_path}`\n\n")
        fobj.write("| split | " + " | ".join(keys) + " |\n")
        fobj.write("|---|" + "|".join(["---"] * len(keys)) + "|\n")
        for name, summary in [("train", train_summary), ("eval", eval_summary)]:
            vals: list[str] = []
            for key in keys:
                value = summary.get(key, "")
                vals.append(f"{value:.8f}" if isinstance(value, float) else str(value))
            fobj.write("| " + name + " | " + " | ".join(vals) + " |\n")
        final = train_trace[-1] if train_trace else {}
        fobj.write("\nFinal training trace:\n\n")
        for key in ("loss", "rel_loss", "alpha_loss", "family_loss", "pred_alpha_mean", "soft_gate_mean", "hard_gate_mean", "grad_norm"):
            if key in final:
                fobj.write(f"- {key}: `{final[key]:.8f}`\n")
        fobj.write("\nInterpretation:\n\n")
        fobj.write("- A useful bridge should keep `force_zero_alpha_max` at zero while learning nonzero gates on E324 active slices.\n")
        fobj.write("- Codec-loop value still requires E295 evaluation on the held-out image range.\n")
    print(f"wrote {json_path}, {md_path}, {trace_path}, {state_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.manifest)
    train_records = records[: args.train_count]
    eval_records = records[args.train_count : args.train_count + args.eval_count]
    if not train_records or not eval_records:
        raise SystemExit("empty train/eval split")
    device = torch.device(args.device)
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
        trace_row = {"step": float(step + 1), "grad_norm": float(grad_norm.item())}
        trace_row.update(metrics)
        trace.append(trace_row)
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            print(
                f"step={step + 1}/{args.steps} loss={metrics['loss']:.4f} "
                f"alpha={metrics['pred_alpha_mean']:.6f} soft_gate={metrics['soft_gate_mean']:.4f} "
                f"hard_gate={metrics['hard_gate_mean']:.4f} grad={float(grad_norm.item()):.4f}"
            )

    controller.eval()
    train_summary = evaluate(controller, train_records, args, device)
    eval_summary = evaluate(controller, eval_records, args, device)
    state_path = args.output_prefix.with_suffix(".pth")
    torch.save(
        {
            "model": controller.state_dict(),
            "config": asdict(controller.config),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        state_path,
    )
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
