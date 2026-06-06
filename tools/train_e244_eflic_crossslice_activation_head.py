#!/usr/bin/env python3
"""Train/calibrate an EF-LIC cross-slice HCG activation head.

E243 trained a tiny per-slice binary activation head. It exposed the right
problem, but four-fold calibration was too weak. E244 keeps the same frozen
teacher setup while giving the head decoder-safe cross-slice/global context:
all four EF-LIC slice context maps are stacked as one `[44, H, W]` input and
the head predicts four activation maps jointly.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcg_rvq.eflic_local_controller import FAMILY_TO_INDEX  # noqa: E402


@dataclass(frozen=True)
class Sample:
    index: int
    dataset: str
    image: str
    tensor_path: Path
    sample_weight: float
    active_frac: float
    target_family: str
    teacher_policy: str


class CrossSliceActivationHead(nn.Module):
    """Decoder-safe joint four-slice activation head.

    The global branch is deliberately small. Its role is not to memorize Kodak
    images, but to expose whether image/slice-level state missing from E243 is
    useful before touching EF-LIC's R-D training loop.
    """

    def __init__(self, input_channels: int = 44, hidden_channels: int = 64, zero_bias: float = -2.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.out = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.zero_bias = float(zero_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.out.bias, self.zero_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.local(x)
        global_feat = self.global_mlp(F.adaptive_avg_pool2d(feat, 1))
        return self.out(F.silu(feat + global_feat))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context-manifest", type=Path, default=ROOT / "experiments" / "analysis" / "e242_eflic_spatial_teacher_contexts_kodak24" / "manifest_kodak24_n24.csv")
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "analysis" / "e244_eflic_crossslice_activation_head_kodak24_split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-mod", type=int, default=4)
    p.add_argument("--holdout-rem", type=int, default=0)
    p.add_argument("--false-positive-weight", type=float, default=4.0)
    p.add_argument("--missed-active-weight", type=float, default=2.0)
    p.add_argument("--zero-bias", type=float, default=-2.0)
    p.add_argument("--use-sample-weight", action="store_true")
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
                    sample_weight=float(row.get("sample_weight", 1.0)),
                    active_frac=float(row.get("active_frac", 0.0)),
                    target_family=row["target_family"],
                    teacher_policy=row["teacher_policy"],
                )
            )
    if not samples:
        raise SystemExit(f"no finite samples found in {path}")
    return samples


def split_samples(samples: list[Sample], holdout_mod: int, holdout_rem: int) -> tuple[list[Sample], list[Sample]]:
    if holdout_mod <= 1:
        return samples, []
    train = [s for s in samples if s.index % holdout_mod != holdout_rem]
    val = [s for s in samples if s.index % holdout_mod == holdout_rem]
    if not train or not val:
        raise SystemExit("empty train/val split; adjust holdout settings")
    return train, val


def load_item(sample: Sample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    obj = torch.load(sample.tensor_path, map_location="cpu")
    maps = obj["context_maps"].float()
    target_map = obj["target_map"].long()
    if maps.ndim != 4 or maps.shape[0] != 4 or maps.shape[1] != 11:
        raise RuntimeError(f"bad context shape for {sample.tensor_path}: {tuple(maps.shape)}")
    if target_map.shape != (maps.shape[0], maps.shape[2], maps.shape[3]):
        raise RuntimeError(f"bad target shape for {sample.tensor_path}: {tuple(target_map.shape)}")
    if not torch.isfinite(maps).all().item():
        raise RuntimeError(f"nonfinite context in {sample.tensor_path}")

    x = maps.reshape(1, maps.shape[0] * maps.shape[1], maps.shape[2], maps.shape[3])
    active = (target_map != FAMILY_TO_INDEX["zero"]).float()[None]
    sample_weight = None
    if sample.sample_weight > 0:
        sample_weight = torch.full_like(active, float(sample.sample_weight))
    return x.to(device=device), active.to(device=device), sample_weight.to(device=device) if sample_weight is not None else None


def binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None,
    false_positive_weight: float,
    missed_active_weight: float,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"target shape {tuple(target.shape)} is incompatible with logits {tuple(logits.shape)}")
    target = (target > 0).to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(
        target > 0.5,
        torch.full_like(loss, float(missed_active_weight)),
        torch.full_like(loss, float(false_positive_weight)),
    )
    if sample_weight is not None:
        weight = weight * sample_weight.to(device=logits.device, dtype=logits.dtype)
    return (loss * weight).mean()


def train_one_epoch(
    *,
    head: CrossSliceActivationHead,
    samples: list[Sample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    head.train()
    order = list(samples)
    random.shuffle(order)
    losses: list[float] = []
    for sample in order:
        x, active, sample_weight = load_item(sample, device)
        logits = head(x)
        loss = binary_loss(
            logits,
            active,
            sample_weight=sample_weight if args.use_sample_weight else None,
            false_positive_weight=args.false_positive_weight,
            missed_active_weight=args.missed_active_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


@torch.inference_mode()
def collect_scores(head: CrossSliceActivationHead, samples: list[Sample], device: torch.device, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    head.eval()
    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    image_rows: list[dict[str, Any]] = []
    for sample in samples:
        x, active, _ = load_item(sample, device)
        probs = torch.sigmoid(head(x)).detach().cpu().float()
        target = active.detach().cpu().float()
        all_scores.append(probs.reshape(-1).numpy())
        all_targets.append(target.reshape(-1).numpy())
        per_slice_score = probs.mean(dim=(0, 2, 3)).tolist()
        per_slice_target = target.mean(dim=(0, 2, 3)).tolist()
        row = {
            "split": split,
            "index": sample.index,
            "dataset": sample.dataset,
            "image": sample.image,
            "target_family": sample.target_family,
            "teacher_policy": sample.teacher_policy,
            "target_active_frac": float(target.mean().item()),
            "score_mean": float(probs.mean().item()),
            "score_std": float(probs.std(unbiased=False).item()),
            "score_min": float(probs.min().item()),
            "score_max": float(probs.max().item()),
        }
        for idx, value in enumerate(per_slice_score):
            row[f"score_mean_slice{idx}"] = float(value)
        for idx, value in enumerate(per_slice_target):
            row[f"target_active_slice{idx}"] = float(value)
        image_rows.append(row)
    return np.concatenate(all_scores), np.concatenate(all_targets).astype(bool), image_rows


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    pos = labels
    neg = ~labels
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    miss = fn / max(1, tp + fn)
    pred_active = int(pred.sum()) / max(1, pred.size)
    accuracy = (tp + tn) / max(1, pred.size)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "missed_active_rate": float(miss),
        "pred_active_frac": float(pred_active),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def auc_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool)
    pos_n = int(labels.sum())
    neg_n = int((~labels).sum())
    if pos_n == 0 or neg_n == 0:
        return {"auroc": float("nan"), "auprc": float("nan")}
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    tpr = np.concatenate([[0.0], tp / pos_n, [1.0]])
    fpr = np.concatenate([[0.0], fp / neg_n, [1.0]])
    auroc = float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / pos_n
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    auprc = float(np.trapz(precision, recall))
    return {"auroc": auroc, "auprc": auprc}


def choose_thresholds(train_scores: np.ndarray, train_labels: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 501), np.quantile(train_scores, np.linspace(0.0, 1.0, 101))]))
    metrics = [threshold_metrics(train_scores, train_labels, float(t)) for t in grid]
    selected: list[dict[str, Any]] = []

    def add(name: str, item: dict[str, float]) -> None:
        payload = dict(item)
        payload["threshold_name"] = name
        selected.append(payload)

    for fixed in [0.25, 0.50, 0.75]:
        add(f"fixed_{fixed:.2f}", threshold_metrics(train_scores, train_labels, fixed))

    best_f1 = max(metrics, key=lambda m: (m["f1"], m["recall"], -m["false_positive_rate"]))
    add("train_best_f1", best_f1)

    for cap in [0.01, 0.05, 0.10, 0.20]:
        feasible = [m for m in metrics if m["false_positive_rate"] <= cap]
        if feasible:
            chosen = max(feasible, key=lambda m: (m["recall"], m["precision"], -m["threshold"]))
        else:
            chosen = min(metrics, key=lambda m: (m["false_positive_rate"], -m["recall"]))
        add(f"train_fpr_le_{cap:.2f}", chosen)

    fp_w = float(args.false_positive_weight)
    miss_w = float(args.missed_active_weight)
    min_risk = min(metrics, key=lambda m: fp_w * m["false_positive_rate"] + miss_w * m["missed_active_rate"])
    add("train_min_weighted_risk", min_risk)
    return selected


def write_outputs(
    *,
    args: argparse.Namespace,
    final_loss: float,
    train_samples: list[Sample],
    val_samples: list[Sample],
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    image_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    checkpoint_path: Path | None,
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    thresholds_path = args.output_prefix.with_suffix(".thresholds.csv")
    images_path = args.output_prefix.with_suffix(".images.csv")

    with thresholds_path.open("w", newline="") as fobj:
        fields = sorted({key for row in threshold_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(threshold_rows)

    with images_path.open("w", newline="") as fobj:
        fields = sorted({key for row in image_rows for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(image_rows)

    train_auc = auc_metrics(train_scores, train_labels)
    val_auc = auc_metrics(val_scores, val_labels)
    payload = {
        "experiment": "E244 EF-LIC cross-slice binary activation head calibration audit",
        "context_manifest": str(args.context_manifest),
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_channels": args.hidden_channels,
        "seed": args.seed,
        "holdout_mod": args.holdout_mod,
        "holdout_rem": args.holdout_rem,
        "false_positive_weight": args.false_positive_weight,
        "missed_active_weight": args.missed_active_weight,
        "zero_bias": args.zero_bias,
        "use_sample_weight": bool(args.use_sample_weight),
        "final_train_loss": final_loss,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_images": [s.image for s in train_samples],
        "val_images": [s.image for s in val_samples],
        "threshold_rows": threshold_rows,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    best_val_f1 = max([r for r in threshold_rows if r["split"] == "val"], key=lambda r: r["f1"])
    conservative = [r for r in threshold_rows if r["split"] == "val" and r["threshold_name"] == "train_fpr_le_0.10"]
    with md_path.open("w") as fobj:
        fobj.write("# E244 EF-LIC Cross-Slice Activation Head Calibration Audit\n\n")
        fobj.write("This trains a joint four-slice zero-vs-active HCG activation head from E242 spatial teachers. It is a frozen-head calibration audit, not codec RD evidence.\n\n")
        fobj.write(f"- Context manifest: `{args.context_manifest}`\n")
        fobj.write(f"- Epochs: `{args.epochs}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Split: holdout index `% {args.holdout_mod} == {args.holdout_rem}`\n")
        fobj.write(f"- Input: all four EF-LIC slice contexts stacked as `[44, H, W]`\n")
        fobj.write(f"- Final train loss: `{final_loss:.6f}`\n")
        fobj.write(f"- Train AUROC/AUPRC: `{train_auc['auroc']:.6f}` / `{train_auc['auprc']:.6f}`\n")
        fobj.write(f"- Val AUROC/AUPRC: `{val_auc['auroc']:.6f}` / `{val_auc['auprc']:.6f}`\n")
        fobj.write(f"- Best val F1 row: `{best_val_f1['threshold_name']}` threshold `{best_val_f1['threshold']:.6f}` F1 `{best_val_f1['f1']:.6f}` FPR `{best_val_f1['false_positive_rate']:.6f}` recall `{best_val_f1['recall']:.6f}`\n")
        if conservative:
            row = conservative[0]
            fobj.write(f"- Train-FPR<=0.10 transfer: val recall `{row['recall']:.6f}` at FPR `{row['false_positive_rate']:.6f}`\n")
        if checkpoint_path:
            fobj.write(f"- Checkpoint: `{checkpoint_path}`\n")
        fobj.write("\n| threshold | split | thr | precision | recall | fpr | missed | pred_active | f1 |\n")
        fobj.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in threshold_rows:
            fobj.write(
                f"| {row['threshold_name']} | {row['split']} | {row['threshold']:.6f} | {row['precision']:.6f} | "
                f"{row['recall']:.6f} | {row['false_positive_rate']:.6f} | {row['missed_active_rate']:.6f} | "
                f"{row['pred_active_frac']:.6f} | {row['f1']:.6f} |\n"
            )
        fobj.write("\nInterpretation guardrail:\n\n")
        fobj.write("- This is not full-training evidence. It decides whether stronger decoder-safe activation conditioning is worth codec-loop insertion.\n")

    print(f"wrote {md_path}, {json_path}, {thresholds_path}, {images_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = read_samples(args.context_manifest)
    train_samples, val_samples = split_samples(samples, args.holdout_mod, args.holdout_rem)
    head = CrossSliceActivationHead(hidden_channels=args.hidden_channels, zero_bias=args.zero_bias).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    final_loss = 0.0
    for epoch in range(args.epochs):
        final_loss = train_one_epoch(head=head, samples=train_samples, optimizer=optimizer, device=device, args=args)
        if (epoch + 1) in {1, args.epochs} or (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch + 1:04d}/{args.epochs} loss={final_loss:.6f}")

    train_scores, train_labels, train_image_rows = collect_scores(head, train_samples, device, "train")
    val_scores, val_labels, val_image_rows = collect_scores(head, val_samples, device, "val")
    selected = choose_thresholds(train_scores, train_labels, args)
    threshold_rows: list[dict[str, Any]] = []
    for item in selected:
        name = item["threshold_name"]
        thr = item["threshold"]
        train_row = threshold_metrics(train_scores, train_labels, thr)
        val_row = threshold_metrics(val_scores, val_labels, thr)
        train_row.update({"threshold_name": name, "split": "train", "selected_on": "train"})
        val_row.update({"threshold_name": name, "split": "val", "selected_on": "train"})
        threshold_rows.extend([train_row, val_row])

    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = args.output_prefix.with_suffix(".pth")
        torch.save({"state_dict": head.state_dict(), "args": vars(args), "threshold_rows": threshold_rows}, checkpoint_path)

    write_outputs(
        args=args,
        final_loss=final_loss,
        train_samples=train_samples,
        val_samples=val_samples,
        train_scores=train_scores,
        train_labels=train_labels,
        val_scores=val_scores,
        val_labels=val_labels,
        image_rows=train_image_rows + val_image_rows,
        threshold_rows=threshold_rows,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
