#!/usr/bin/env python3
"""Train/evaluate the E239 EF-LIC local HCG family head from E240 contexts.

This is a frozen-head smoke/audit script. It trains only `LocalHCGFamilyHead`
from exported decoder-safe local context maps and image-level E239 labels. The
labels are broadcast over EF-LIC slices and spatial positions; later experiments
should replace them with slice/spatial labels before any paper performance claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import (  # noqa: E402
    FAMILY_NAMES,
    FAMILY_TO_INDEX,
    LocalHCGFamilyHead,
    LocalHCGHeadConfig,
    asymmetric_family_loss,
)


@dataclass(frozen=True)
class Sample:
    index: int
    dataset: str
    image: str
    tensor_path: Path
    target_index: int
    target_family: str
    sample_weight: float
    confident_nonzero: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context-manifest", type=Path, default=ROOT / "experiments" / "analysis" / "e240_eflic_local_head_contexts_kodak24" / "manifest_kodak24_n24.csv")
    p.add_argument("--class-weights-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan.class_weights.csv")
    p.add_argument("--cost-matrix-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan.cost_matrix.csv")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e241_eflic_local_head_kodak24_split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=48)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-mod", type=int, default=4)
    p.add_argument("--holdout-rem", type=int, default=0)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=1.0)
    p.add_argument("--zero-weight-multiplier", type=float, default=1.0)
    p.add_argument("--use-family-cost", action="store_true")
    p.add_argument("--disable-class-weights", action="store_true")
    p.add_argument("--image-level-loss", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def read_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="") as fobj:
        for idx, row in enumerate(csv.DictReader(fobj)):
            if int(row.get("finite_context", 1)) != 1:
                continue
            samples.append(
                Sample(
                    index=idx,
                    dataset=row["dataset"],
                    image=row["image"],
                    tensor_path=Path(row["tensor_path"]),
                    target_index=int(row["target_index"]),
                    target_family=row["target_family"],
                    sample_weight=float(row["sample_weight"]),
                    confident_nonzero=int(row.get("confident_nonzero", int(row["target_index"]) != 0)),
                )
            )
    if not samples:
        raise SystemExit(f"no finite samples found in {path}")
    return samples


def read_class_weights(path: Path) -> torch.Tensor:
    weights = torch.ones(len(FAMILY_NAMES), dtype=torch.float32)
    if not path.exists():
        return weights
    with path.open(newline="") as fobj:
        for row in csv.DictReader(fobj):
            idx = int(row["index"])
            if 0 <= idx < len(FAMILY_NAMES):
                weights[idx] = float(row["class_weight"])
    return weights


def read_cost_matrix(path: Path) -> torch.Tensor:
    mat = torch.zeros((len(FAMILY_NAMES), len(FAMILY_NAMES)), dtype=torch.float32)
    if not path.exists():
        return mat
    with path.open(newline="") as fobj:
        for row in csv.DictReader(fobj):
            i = int(row["true_index"])
            j = int(row["candidate_index"])
            if 0 <= i < len(FAMILY_NAMES) and 0 <= j < len(FAMILY_NAMES):
                mat[i, j] = float(row["cost"])
    return mat


def load_maps(sample: Sample, device: torch.device) -> torch.Tensor:
    obj = torch.load(sample.tensor_path, map_location="cpu")
    maps = obj["context_maps"].float()
    if maps.ndim != 4 or maps.shape[0] != 4 or maps.shape[1] != 11:
        raise RuntimeError(f"bad context shape for {sample.tensor_path}: {tuple(maps.shape)}")
    if not torch.isfinite(maps).all().item():
        raise RuntimeError(f"nonfinite context in {sample.tensor_path}")
    return maps.to(device=device, non_blocking=True)


def split_samples(samples: list[Sample], holdout_mod: int, holdout_rem: int) -> tuple[list[Sample], list[Sample]]:
    if holdout_mod <= 1:
        return samples, []
    train = [s for s in samples if s.index % holdout_mod != holdout_rem]
    val = [s for s in samples if s.index % holdout_mod == holdout_rem]
    if not train or not val:
        raise SystemExit("empty train/val split; adjust holdout settings")
    return train, val


def family_counts(samples: list[Sample]) -> dict[str, int]:
    counts = {name: 0 for name in FAMILY_NAMES}
    for sample in samples:
        counts[sample.target_family] += 1
    return {key: value for key, value in counts.items() if value}


def make_weight(
    sample: Sample,
    class_weights: torch.Tensor,
    disable_class_weights: bool,
    zero_weight_multiplier: float = 1.0,
) -> float:
    weight = float(sample.sample_weight)
    if sample.target_index == FAMILY_TO_INDEX["zero"]:
        weight *= float(zero_weight_multiplier)
    if not disable_class_weights:
        weight *= float(class_weights[sample.target_index].item())
    return weight


def train_one_epoch(
    *,
    head: LocalHCGFamilyHead,
    samples: list[Sample],
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    cost_matrix: torch.Tensor | None,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    head.train()
    order = list(samples)
    random.shuffle(order)
    losses: list[float] = []
    for sample in order:
        maps = load_maps(sample, device)
        weight_value = make_weight(
            sample,
            class_weights,
            args.disable_class_weights,
            args.zero_weight_multiplier,
        )
        raw_logits = head(maps)
        if args.image_level_loss:
            logits = raw_logits.mean(dim=(0, 2, 3), keepdim=False)[None, :, None, None]
            target = torch.tensor([sample.target_index], dtype=torch.long, device=device)
            weights = torch.tensor([weight_value], dtype=torch.float32, device=device)
        else:
            logits = raw_logits
            target = torch.full((maps.shape[0],), sample.target_index, dtype=torch.long, device=device)
            weights = torch.full((maps.shape[0],), weight_value, dtype=torch.float32, device=device)
        loss = asymmetric_family_loss(
            logits,
            target,
            sample_weight=weights,
            false_positive_weight=args.false_positive_weight,
            missed_active_weight=args.missed_active_weight,
            cost_matrix=cost_matrix,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


@torch.inference_mode()
def evaluate(
    *,
    head: LocalHCGFamilyHead,
    samples: list[Sample],
    class_weights: torch.Tensor,
    cost_matrix: torch.Tensor,
    device: torch.device,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head.eval()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        maps = load_maps(sample, device)
        logits = head(maps)
        image_logits = logits.mean(dim=(0, 2, 3))
        probs = image_logits.softmax(dim=0)
        pred_idx = int(probs.argmax().item())
        target_idx = int(sample.target_index)
        row = {
            "split": split,
            "index": sample.index,
            "dataset": sample.dataset,
            "image": sample.image,
            "target_family": sample.target_family,
            "target_index": target_idx,
            "pred_family": FAMILY_NAMES[pred_idx],
            "pred_index": pred_idx,
            "correct": int(pred_idx == target_idx),
            "target_zero": int(target_idx == FAMILY_TO_INDEX["zero"]),
            "pred_zero": int(pred_idx == FAMILY_TO_INDEX["zero"]),
            "false_positive_nonzero": int(target_idx == FAMILY_TO_INDEX["zero"] and pred_idx != FAMILY_TO_INDEX["zero"]),
            "missed_active": int(target_idx != FAMILY_TO_INDEX["zero"] and pred_idx == FAMILY_TO_INDEX["zero"]),
            "pred_nonzero_prob": float((1.0 - probs[FAMILY_TO_INDEX["zero"]]).item()),
            "target_prob": float(probs[target_idx].item()),
            "pred_prob": float(probs[pred_idx].item()),
            "family_cost": float(cost_matrix[target_idx, pred_idx].item()),
            "sample_weight": float(sample.sample_weight),
            "effective_weight": make_weight(sample, class_weights, False, 1.0),
        }
        rows.append(row)

    n = max(1, len(rows))
    zero_rows = [r for r in rows if r["target_zero"]]
    active_rows = [r for r in rows if not r["target_zero"]]
    summary = {
        "split": split,
        "images": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "avg_family_cost": sum(r["family_cost"] for r in rows) / n,
        "avg_pred_nonzero_prob": sum(r["pred_nonzero_prob"] for r in rows) / n,
        "pred_nonzero_frac": sum(1 - r["pred_zero"] for r in rows) / n,
        "target_nonzero_frac": sum(1 - r["target_zero"] for r in rows) / n,
        "false_positive_nonzero_frac": (sum(r["false_positive_nonzero"] for r in zero_rows) / max(1, len(zero_rows))),
        "missed_active_frac": (sum(r["missed_active"] for r in active_rows) / max(1, len(active_rows))),
        "target_family_counts": family_counts(samples),
        "pred_family_counts": {name: sum(1 for r in rows if r["pred_family"] == name) for name in FAMILY_NAMES if any(r["pred_family"] == name for r in rows)},
    }
    return summary, rows


def write_outputs(
    *,
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    train_samples: list[Sample],
    val_samples: list[Sample],
    final_loss: float,
    checkpoint_path: Path | None,
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    pred_path = args.output_prefix.with_suffix(".predictions.csv")

    with summary_path.open("w", newline="") as fobj:
        fields = ["split", "images", "accuracy", "avg_family_cost", "avg_pred_nonzero_prob", "pred_nonzero_frac", "target_nonzero_frac", "false_positive_nonzero_frac", "missed_active_frac", "target_family_counts", "pred_family_counts"]
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    with pred_path.open("w", newline="") as fobj:
        fields = sorted({key for row in predictions for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    payload = {
        "experiment": "E241 EF-LIC frozen local HCG head supervised smoke",
        "context_manifest": str(args.context_manifest),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "seed": args.seed,
        "holdout_mod": args.holdout_mod,
        "holdout_rem": args.holdout_rem,
        "use_family_cost": bool(args.use_family_cost),
        "disable_class_weights": bool(args.disable_class_weights),
        "zero_weight_multiplier": args.zero_weight_multiplier,
        "image_level_loss": bool(args.image_level_loss),
        "final_train_loss": final_loss,
        "train_images": [s.image for s in train_samples],
        "val_images": [s.image for s in val_samples],
        "summaries": summaries,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with md_path.open("w") as fobj:
        fobj.write("# E241 EF-LIC Frozen Local HCG Head Supervised Smoke\n\n")
        fobj.write("This trains only `LocalHCGFamilyHead` from E240 decoder-safe context tensors. ")
        fobj.write("Image-level E239 labels are broadcast over slices/spatial positions, so this is a training-contract smoke, not a final paper result.\n\n")
        fobj.write(f"- Context manifest: `{args.context_manifest}`\n")
        fobj.write(f"- Epochs: `{args.epochs}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Split: holdout index `% {args.holdout_mod} == {args.holdout_rem}`\n")
        fobj.write(f"- Zero weight multiplier: `{args.zero_weight_multiplier}`\n")
        fobj.write(f"- Image-level loss: `{bool(args.image_level_loss)}`\n")
        fobj.write(f"- Final train loss: `{final_loss:.6f}`\n")
        if checkpoint_path:
            fobj.write(f"- Checkpoint: `{checkpoint_path}`\n")
        fobj.write("\n| split | images | accuracy | avg_family_cost | pred_nonzero_frac | false_positive_nonzero_frac | missed_active_frac | pred_family_counts |\n")
        fobj.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for item in summaries:
            fobj.write(
                f"| {item['split']} | {item['images']} | {item['accuracy']:.6f} | {item['avg_family_cost']:.6f} | "
                f"{item['pred_nonzero_frac']:.6f} | {item['false_positive_nonzero_frac']:.6f} | {item['missed_active_frac']:.6f} | "
                f"`{json.dumps(item['pred_family_counts'], sort_keys=True)}` |\n"
            )
        fobj.write("\nInterpretation guardrail:\n\n")
        fobj.write("- Good train accuracy alone is not enough; held-out false-positive nonzero rate is the key safety metric.\n")
        fobj.write("- If validation collapses to a frequent nonzero family, the next step should improve labels/features, not promote this as performance.\n")

    print(f"wrote {md_path}, {json_path}, {summary_path}, {pred_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = read_samples(args.context_manifest)
    train_samples, val_samples = split_samples(samples, args.holdout_mod, args.holdout_rem)
    class_weights = read_class_weights(args.class_weights_csv)
    cost_matrix_cpu = read_cost_matrix(args.cost_matrix_csv)
    cost_matrix = cost_matrix_cpu.to(device) if args.use_family_cost else None

    head = LocalHCGFamilyHead(LocalHCGHeadConfig(hidden_channels=args.hidden_channels)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    final_loss = math.nan
    for _epoch in range(args.epochs):
        final_loss = train_one_epoch(
            head=head,
            samples=train_samples,
            optimizer=optimizer,
            class_weights=class_weights,
            cost_matrix=cost_matrix,
            device=device,
            args=args,
        )

    eval_cost_matrix = cost_matrix_cpu.to(device)
    train_summary, train_rows = evaluate(
        head=head,
        samples=train_samples,
        class_weights=class_weights,
        cost_matrix=eval_cost_matrix,
        device=device,
        split="train",
    )
    val_summary, val_rows = evaluate(
        head=head,
        samples=val_samples,
        class_weights=class_weights,
        cost_matrix=eval_cost_matrix,
        device=device,
        split="val",
    )

    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = args.output_prefix.with_suffix(".pth")
        torch.save({"state_dict": head.state_dict(), "family_names": FAMILY_NAMES, "args": vars(args)}, checkpoint_path)

    write_outputs(
        args=args,
        summaries=[train_summary, val_summary],
        predictions=train_rows + val_rows,
        train_samples=train_samples,
        val_samples=val_samples,
        final_loss=float(final_loss),
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
