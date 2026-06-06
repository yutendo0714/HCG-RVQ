#!/usr/bin/env python3
"""E265 EF-LIC fallback-gated controller context smoke.

This script does not claim EF-LIC codec performance.  It verifies the next
integration unit after E263/E264: decoder-safe EF-LIC context tensors from E242
can drive the shared E260/E262 reliability/index controller, preserve exact
no-branch fallback under a hard zero gate, and produce finite losses/gradients.

The synthetic branch target is the E242 teacher alpha map.  That makes this a
controller wiring and artifact-provenance check, not final RD evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import (  # noqa: E402
    LocalHCGFamilyHead,
    asymmetric_family_loss,
)
from hcg_rvq.reliability_index_controller import (  # noqa: E402
    SpatialReliabilityIndexConfig,
    SpatialReliabilityIndexHead,
    mix_with_fallback,
    reliability_index_loss,
)


@dataclass
class ContextRecord:
    dataset: str
    image: str
    tensor_path: Path


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
    p.add_argument("--train-per-domain", type=int, default=4)
    p.add_argument("--eval-per-domain", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden-channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=265)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e265_eflic_fallback_gate_context_smoke",
    )
    return p.parse_args()


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def tensor_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.float()).all().item())


def load_manifest(path: Path, limit: int, offset: int = 0) -> list[ContextRecord]:
    rows: list[ContextRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < offset:
                continue
            if len(rows) >= limit:
                break
            tensor_path = Path(row["tensor_path"])
            if not tensor_path.is_absolute():
                tensor_path = ROOT / tensor_path
            rows.append(ContextRecord(dataset=row["dataset"], image=row["image"], tensor_path=tensor_path))
    return rows


def load_tensor(record: ContextRecord) -> dict[str, Any]:
    obj = torch.load(record.tensor_path, map_location="cpu")
    required = {"context_maps", "alpha_target", "target_map"}
    missing = required.difference(obj)
    if missing:
        raise RuntimeError(f"{record.tensor_path} is missing {sorted(missing)}")
    context = obj["context_maps"].float()
    alpha = obj["alpha_target"].float()
    target_map = obj["target_map"].long()
    if context.ndim != 4 or context.shape[1] != 11:
        raise RuntimeError(f"unexpected context shape {tuple(context.shape)} in {record.tensor_path}")
    if alpha.shape != (context.shape[0], 1, context.shape[2], context.shape[3]):
        raise RuntimeError(f"unexpected alpha shape {tuple(alpha.shape)} in {record.tensor_path}")
    if target_map.shape != (context.shape[0], context.shape[2], context.shape[3]):
        raise RuntimeError(f"unexpected target_map shape {tuple(target_map.shape)} in {record.tensor_path}")
    if not (tensor_finite(context) and tensor_finite(alpha)):
        raise RuntimeError(f"nonfinite tensor in {record.tensor_path}")
    return {"context": context, "alpha": alpha, "target_map": target_map, "meta": obj}


def controller_objective(
    controller: SpatialReliabilityIndexHead,
    family_head: LocalHCGFamilyHead,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    context = batch["context"]
    alpha = batch["alpha"]
    target_map = batch["target_map"]
    active = (target_map > 0).unsqueeze(1).to(dtype=context.dtype)
    target_score = torch.where(active > 0, -alpha.abs(), torch.zeros_like(alpha))

    out = controller(context)
    family_logits = family_head(context)
    base = torch.zeros_like(alpha)
    branch = alpha
    mixed, gate = mix_with_fallback(
        base,
        branch,
        out["active_logit"],
        risk_score=out["risk_score"],
        risk_temperature=0.5,
        hard=False,
        max_gate=1.0,
    )
    rel_loss = reliability_index_loss(
        out["active_logit"],
        active,
        risk_score=out["risk_score"],
        target_score=target_score,
        false_positive_weight=4.0,
        missed_active_weight=1.0,
        score_weight=0.1,
    )
    alpha_loss = F.smooth_l1_loss(mixed, alpha)
    fam_loss = asymmetric_family_loss(
        family_logits,
        target_map,
        false_positive_weight=4.0,
        missed_active_weight=1.0,
    )
    loss = rel_loss + 0.5 * alpha_loss + 0.1 * fam_loss
    metrics = {
        "loss": float(loss.detach().item()),
        "rel_loss": float(rel_loss.detach().item()),
        "alpha_loss": float(alpha_loss.detach().item()),
        "family_loss": float(fam_loss.detach().item()),
        "active_frac": float(active.mean().item()),
        "alpha_mean": float(alpha.mean().item()),
        "gate_mean": float(gate.detach().mean().item()),
        "gate_max": float(gate.detach().max().item()),
    }
    return loss, metrics


def evaluate(
    controller: SpatialReliabilityIndexHead,
    family_head: LocalHCGFamilyHead,
    records: list[ContextRecord],
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "loss": [],
        "rel_loss": [],
        "alpha_loss": [],
        "family_loss": [],
        "active_frac": [],
        "alpha_mean": [],
        "gate_mean": [],
        "gate_max": [],
        "hard_selected_frac": [],
        "hard_base_exact": [],
    }
    by_dataset: dict[str, dict[str, list[float]]] = {}
    nonfinite = 0
    with torch.no_grad():
        for record in records:
            batch = load_tensor(record)
            loss, metrics = controller_objective(controller, family_head, batch)
            out = controller(batch["context"])
            base = torch.zeros_like(batch["alpha"])
            branch = batch["alpha"]
            hard_mixed, hard_gate = mix_with_fallback(
                base,
                branch,
                torch.full_like(out["active_logit"], -12.0),
                risk_score=torch.ones_like(out["risk_score"]),
                hard=True,
            )
            hard_exact = 1.0 if torch.allclose(hard_mixed, base, atol=0.0, rtol=0.0) else 0.0
            metrics = dict(metrics)
            metrics["loss"] = float(loss.item())
            metrics["hard_selected_frac"] = float(hard_gate.mean().item())
            metrics["hard_base_exact"] = hard_exact
            if not all(math.isfinite(v) for v in metrics.values()):
                nonfinite += 1
            for key, value in metrics.items():
                values.setdefault(key, []).append(value)
            ds = by_dataset.setdefault(record.dataset, {key: [] for key in values})
            for key, value in metrics.items():
                ds.setdefault(key, []).append(value)
    summary = {key: mean(vals) for key, vals in values.items()}
    summary["images"] = len(records)
    summary["nonfinite_records"] = nonfinite
    summary["by_dataset"] = {
        ds: {key: mean(vals) for key, vals in metrics.items()} | {"images": len(next(iter(metrics.values()), []))}
        for ds, metrics in by_dataset.items()
    }
    return summary


def render_float(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    train_records = load_manifest(args.kodak_manifest, args.train_per_domain, 0) + load_manifest(
        args.clicpro_manifest, args.train_per_domain, 0
    )
    eval_records = load_manifest(args.kodak_manifest, args.eval_per_domain, args.train_per_domain) + load_manifest(
        args.clicpro_manifest, args.eval_per_domain, args.train_per_domain
    )
    if not train_records or not eval_records:
        raise SystemExit("no train/eval records loaded")

    controller = SpatialReliabilityIndexHead(
        SpatialReliabilityIndexConfig(input_channels=11, hidden_channels=args.hidden_channels, zero_bias=-2.0)
    )
    family_head = LocalHCGFamilyHead()
    opt = torch.optim.AdamW(list(controller.parameters()) + list(family_head.parameters()), lr=args.lr, weight_decay=1e-4)

    init_train = evaluate(controller, family_head, train_records)
    init_eval = evaluate(controller, family_head, eval_records)
    trace = []
    for step in range(1, args.steps + 1):
        step_metrics: list[dict[str, float]] = []
        for record in train_records:
            batch = load_tensor(record)
            opt.zero_grad(set_to_none=True)
            loss, metrics = controller_objective(controller, family_head, batch)
            loss.backward()
            grads_finite = all(
                p.grad is None or tensor_finite(p.grad)
                for p in list(controller.parameters()) + list(family_head.parameters())
            )
            if not grads_finite or not tensor_finite(loss.detach()):
                raise RuntimeError(f"nonfinite loss/grad at step {step} on {record.tensor_path}")
            opt.step()
            step_metrics.append(metrics)
        trace.append({"step": step, **{key: mean([m[key] for m in step_metrics]) for key in step_metrics[0]}})

    trained_train = evaluate(controller, family_head, train_records)
    trained_eval = evaluate(controller, family_head, eval_records)
    all_checks_passed = (
        init_train["nonfinite_records"] == 0
        and init_eval["nonfinite_records"] == 0
        and trained_train["nonfinite_records"] == 0
        and trained_eval["nonfinite_records"] == 0
        and init_eval["hard_base_exact"] == 1.0
        and trained_eval["hard_base_exact"] == 1.0
    )
    result = {
        "purpose": "EF-LIC decoder-safe context smoke for fallback-gated HCG-RVQ controller; not codec RD evidence.",
        "train_records": [str(r.tensor_path.relative_to(ROOT)) for r in train_records],
        "eval_records": [str(r.tensor_path.relative_to(ROOT)) for r in eval_records],
        "settings": {
            "train_per_domain": args.train_per_domain,
            "eval_per_domain": args.eval_per_domain,
            "steps": args.steps,
            "lr": args.lr,
            "hidden_channels": args.hidden_channels,
            "seed": args.seed,
        },
        "init_train": init_train,
        "init_eval": init_eval,
        "trained_train": trained_train,
        "trained_eval": trained_eval,
        "trace": trace,
        "all_checks_passed": all_checks_passed,
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E265 EF-LIC Fallback-Gate Context Smoke",
        "",
        "This is a controller wiring smoke, not EF-LIC codec RD evidence. It uses E242 decoder-safe context tensors and teacher alpha maps to verify finite fallback-gated training and exact hard no-branch fallback.",
        "",
        f"Settings: train/domain `{args.train_per_domain}`, eval/domain `{args.eval_per_domain}`, steps `{args.steps}`, hidden `{args.hidden_channels}`, seed `{args.seed}`.",
        "",
        "| split | images | loss | rel loss | alpha loss | family loss | active frac | alpha mean | gate mean | gate max | hard selected | hard base exact | nonfinite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in [
        ("init_train", init_train),
        ("init_eval", init_eval),
        ("trained_train", trained_train),
        ("trained_eval", trained_eval),
    ]:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary["images"]),
                    render_float(summary["loss"]),
                    render_float(summary["rel_loss"]),
                    render_float(summary["alpha_loss"]),
                    render_float(summary["family_loss"]),
                    render_float(summary["active_frac"]),
                    render_float(summary["alpha_mean"]),
                    render_float(summary["gate_mean"]),
                    render_float(summary["gate_max"]),
                    render_float(summary["hard_selected_frac"]),
                    render_float(summary["hard_base_exact"]),
                    str(summary["nonfinite_records"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass means EF-LIC decoder-safe context tensors can drive the shared reliability/index controller and preserve exact no-branch fallback. It does not prove final EF-LIC RD performance.",
            "",
            "Next EF-LIC work should insert this controller next to the real EF-LIC local branch and report selected-index/rate accounting, matching the E264 promotion rule.",
            "",
            f"All checks passed: `{all_checks_passed}`.",
            "",
            "## Artifacts",
            "",
            f"- `{out_prefix.with_suffix('.json').relative_to(ROOT)}`",
        ]
    )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_prefix": str(out_prefix), "all_checks_passed": all_checks_passed}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
