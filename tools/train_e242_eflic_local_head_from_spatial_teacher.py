#!/usr/bin/env python3
"""Train/evaluate LocalHCGFamilyHead from E242 spatial teacher maps.

E241 used image-level labels and failed held-out zero/nonzero calibration. This
script consumes E242 tensors with `target_map` so the local head is supervised at
slice/spatial resolution. It is still a frozen-head audit, not a full codec
performance experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
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
    teacher_policy: str
    sample_weight: float
    active_frac: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context-manifest", type=Path, default=ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts_kodak24" / "manifest_kodak24_n24.csv")
    p.add_argument("--cost-matrix-csv", type=Path, default=ROOT / "experiments" / "analysis" / "e239_eflic_local_head_training_plan.cost_matrix.csv")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e242_eflic_local_head_kodak24_map_split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=48)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-mod", type=int, default=4)
    p.add_argument("--holdout-rem", type=int, default=0)
    p.add_argument("--false-positive-weight", type=float, default=8.0)
    p.add_argument("--missed-active-weight", type=float, default=1.0)
    p.add_argument("--zero-weight-multiplier", type=float, default=4.0)
    p.add_argument("--active-weight-multiplier", type=float, default=1.0)
    p.add_argument("--use-family-cost", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def read_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="") as fobj:
        for idx, row in enumerate(csv.DictReader(fobj)):
            if int(row.get("finite_context", 1)) != 1 or int(row.get("finite_alpha", 1)) != 1:
                continue
            samples.append(
                Sample(
                    index=idx,
                    dataset=row["dataset"],
                    image=row["image"],
                    tensor_path=Path(row["tensor_path"]),
                    target_index=int(row["target_index"]),
                    target_family=row["target_family"],
                    teacher_policy=row["teacher_policy"],
                    sample_weight=float(row.get("sample_weight", 1.0)),
                    active_frac=float(row.get("active_frac", 0.0)),
                )
            )
    if not samples:
        raise SystemExit(f"no finite samples found in {path}")
    return samples


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


def split_samples(samples: list[Sample], holdout_mod: int, holdout_rem: int) -> tuple[list[Sample], list[Sample]]:
    if holdout_mod <= 1:
        return samples, []
    train = [s for s in samples if s.index % holdout_mod != holdout_rem]
    val = [s for s in samples if s.index % holdout_mod == holdout_rem]
    if not train or not val:
        raise SystemExit("empty train/val split; adjust holdout settings")
    return train, val


def load_item(sample: Sample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(sample.tensor_path, map_location="cpu")
    maps = obj["context_maps"].float()
    target = obj["target_map"].long()
    if maps.ndim != 4 or maps.shape[0] != 4 or maps.shape[1] != 11:
        raise RuntimeError(f"bad context shape for {sample.tensor_path}: {tuple(maps.shape)}")
    if target.shape != (maps.shape[0], maps.shape[2], maps.shape[3]):
        raise RuntimeError(f"bad target shape for {sample.tensor_path}: {tuple(target.shape)} vs {tuple(maps.shape)}")
    if not torch.isfinite(maps).all().item():
        raise RuntimeError(f"nonfinite context in {sample.tensor_path}")
    return maps.to(device=device), target.to(device=device)


def make_weight(target: torch.Tensor, sample: Sample, args: argparse.Namespace) -> torch.Tensor:
    weight = torch.full_like(target, float(sample.sample_weight), dtype=torch.float32)
    zero = target == FAMILY_TO_INDEX["zero"]
    active = ~zero
    weight[zero] *= float(args.zero_weight_multiplier)
    weight[active] *= float(args.active_weight_multiplier)
    return weight


def train_one_epoch(
    *,
    head: LocalHCGFamilyHead,
    samples: list[Sample],
    optimizer: torch.optim.Optimizer,
    cost_matrix: torch.Tensor | None,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    head.train()
    order = list(samples)
    random.shuffle(order)
    losses: list[float] = []
    for sample in order:
        maps, target = load_item(sample, device)
        logits = head(maps)
        weights = make_weight(target, sample, args).to(device=device)
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


def _counts(tensor: torch.Tensor) -> dict[str, int]:
    flat = tensor.detach().cpu().reshape(-1).tolist()
    count = Counter(int(v) for v in flat)
    return {FAMILY_NAMES[i]: count[i] for i in range(len(FAMILY_NAMES)) if count[i]}


@torch.inference_mode()
def evaluate(
    *,
    head: LocalHCGFamilyHead,
    samples: list[Sample],
    cost_matrix: torch.Tensor,
    device: torch.device,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head.eval()
    rows: list[dict[str, Any]] = []
    total_pixels = 0
    correct_pixels = 0
    total_cost = 0.0
    zero_pixels = 0
    active_pixels = 0
    false_positive = 0
    missed_active = 0
    pred_active = 0
    target_active = 0
    target_counter: Counter[int] = Counter()
    pred_counter: Counter[int] = Counter()

    for sample in samples:
        maps, target = load_item(sample, device)
        logits = head(maps)
        pred = logits.argmax(dim=1)
        n = int(target.numel())
        correct = int((pred == target).sum().item())
        zero = target == FAMILY_TO_INDEX["zero"]
        active = ~zero
        pred_zero = pred == FAMILY_TO_INDEX["zero"]
        pred_nonzero = ~pred_zero
        fp = int((zero & pred_nonzero).sum().item())
        miss = int((active & pred_zero).sum().item())
        costs = cost_matrix.to(device=device)[target, pred]

        total_pixels += n
        correct_pixels += correct
        total_cost += float(costs.float().sum().item())
        zero_pixels += int(zero.sum().item())
        active_pixels += int(active.sum().item())
        false_positive += fp
        missed_active += miss
        pred_active += int(pred_nonzero.sum().item())
        target_active += int(active.sum().item())
        target_counter.update(int(v) for v in target.detach().cpu().reshape(-1).tolist())
        pred_counter.update(int(v) for v in pred.detach().cpu().reshape(-1).tolist())

        image_logits = logits.mean(dim=(0, 2, 3))
        image_pred = int(image_logits.argmax().item())
        rows.append(
            {
                "split": split,
                "index": sample.index,
                "dataset": sample.dataset,
                "image": sample.image,
                "teacher_policy": sample.teacher_policy,
                "image_target_family": sample.target_family,
                "image_pred_family": FAMILY_NAMES[image_pred],
                "map_accuracy": correct / max(1, n),
                "map_avg_family_cost": float(costs.float().mean().item()),
                "target_active_frac": int(active.sum().item()) / max(1, n),
                "pred_active_frac": int(pred_nonzero.sum().item()) / max(1, n),
                "false_positive_nonzero_frac": fp / max(1, int(zero.sum().item())),
                "missed_active_frac": miss / max(1, int(active.sum().item())),
                "target_family_counts": json.dumps(_counts(target), sort_keys=True),
                "pred_family_counts": json.dumps(_counts(pred), sort_keys=True),
            }
        )

    summary = {
        "split": split,
        "images": len(samples),
        "pixels": total_pixels,
        "map_accuracy": correct_pixels / max(1, total_pixels),
        "map_avg_family_cost": total_cost / max(1, total_pixels),
        "target_active_frac": target_active / max(1, total_pixels),
        "pred_active_frac": pred_active / max(1, total_pixels),
        "false_positive_nonzero_frac": false_positive / max(1, zero_pixels),
        "missed_active_frac": missed_active / max(1, active_pixels),
        "target_family_counts": {FAMILY_NAMES[i]: target_counter[i] for i in range(len(FAMILY_NAMES)) if target_counter[i]},
        "pred_family_counts": {FAMILY_NAMES[i]: pred_counter[i] for i in range(len(FAMILY_NAMES)) if pred_counter[i]},
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
        fields = ["split", "images", "pixels", "map_accuracy", "map_avg_family_cost", "target_active_frac", "pred_active_frac", "false_positive_nonzero_frac", "missed_active_frac", "target_family_counts", "pred_family_counts"]
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    with pred_path.open("w", newline="") as fobj:
        fields = sorted({key for row in predictions for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    payload = {
        "experiment": "E242 EF-LIC frozen local HCG head map-level supervised audit",
        "context_manifest": str(args.context_manifest),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "seed": args.seed,
        "holdout_mod": args.holdout_mod,
        "holdout_rem": args.holdout_rem,
        "use_family_cost": bool(args.use_family_cost),
        "zero_weight_multiplier": args.zero_weight_multiplier,
        "active_weight_multiplier": args.active_weight_multiplier,
        "final_train_loss": final_loss,
        "train_images": [s.image for s in train_samples],
        "val_images": [s.image for s in val_samples],
        "summaries": summaries,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with md_path.open("w") as fobj:
        fobj.write("# E242 EF-LIC Frozen Local HCG Head Map-Level Supervision Audit\n\n")
        fobj.write("This trains only `LocalHCGFamilyHead` from E242 decoder-safe contexts and spatial `target_map` labels. ")
        fobj.write("It checks whether map-level supervision fixes the E241 image-label collapse.\n\n")
        fobj.write(f"- Context manifest: `{args.context_manifest}`\n")
        fobj.write(f"- Epochs: `{args.epochs}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Split: holdout index `% {args.holdout_mod} == {args.holdout_rem}`\n")
        fobj.write(f"- Final train loss: `{final_loss:.6f}`\n")
        if checkpoint_path:
            fobj.write(f"- Checkpoint: `{checkpoint_path}`\n")
        fobj.write("\n| split | images | map_accuracy | map_avg_family_cost | target_active_frac | pred_active_frac | false_positive_nonzero_frac | missed_active_frac | pred_family_counts |\n")
        fobj.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for item in summaries:
            fobj.write(
                f"| {item['split']} | {item['images']} | {item['map_accuracy']:.6f} | {item['map_avg_family_cost']:.6f} | "
                f"{item['target_active_frac']:.6f} | {item['pred_active_frac']:.6f} | "
                f"{item['false_positive_nonzero_frac']:.6f} | {item['missed_active_frac']:.6f} | "
                f"`{json.dumps(item['pred_family_counts'], sort_keys=True)}` |\n"
            )
        fobj.write("\nInterpretation guardrail:\n\n")
        fobj.write("- This is still a frozen-head supervision audit. Promote only if held-out false-positive and missed-active rates are controlled, then verify codec-loop RD.\n")

    print(f"wrote {md_path}, {json_path}, {summary_path}, {pred_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = read_samples(args.context_manifest)
    train_samples, val_samples = split_samples(samples, args.holdout_mod, args.holdout_rem)
    cost_matrix = read_cost_matrix(args.cost_matrix_csv)
    train_cost = cost_matrix if args.use_family_cost else None

    head = LocalHCGFamilyHead(LocalHCGHeadConfig(hidden_channels=args.hidden_channels)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_loss = 0.0
    for epoch in range(args.epochs):
        final_loss = train_one_epoch(
            head=head,
            samples=train_samples,
            optimizer=optimizer,
            cost_matrix=train_cost,
            device=device,
            args=args,
        )
        if (epoch + 1) in {1, args.epochs} or (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch + 1:04d}/{args.epochs} loss={final_loss:.6f}")

    train_summary, train_rows = evaluate(head=head, samples=train_samples, cost_matrix=cost_matrix, device=device, split="train")
    val_summary, val_rows = evaluate(head=head, samples=val_samples, cost_matrix=cost_matrix, device=device, split="val")
    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = args.output_prefix.with_suffix(".pth")
        torch.save({"state_dict": head.state_dict(), "args": vars(args), "summaries": [train_summary, val_summary]}, checkpoint_path)
    write_outputs(
        args=args,
        summaries=[train_summary, val_summary],
        predictions=train_rows + val_rows,
        train_samples=train_samples,
        val_samples=val_samples,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
