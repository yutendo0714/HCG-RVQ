#!/usr/bin/env python3
"""Train a tiny EF-LIC HCG direction/fallback selector from decoder-safe contexts.

E306 showed oracle headroom for choosing among fallback/mean/logscale/fixed
geometry directions. E307 showed that a single gate/alpha threshold cannot
recover that oracle. This script connects those labels to the E242 local context
tensors and tests whether a small decoder-safe head can learn the image-level
direction/fallback decision.

This is a trainability/proxy audit, not final codec RD evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CHOICES = ("fallback", "mean", "logscale", "fixed")
CHOICE_TO_INDEX = {name: idx for idx, name in enumerate(CHOICES)}


@dataclass(frozen=True)
class Sample:
    index: int
    dataset: str
    image: str
    tensor_path: Path
    target_choice: str
    target_index: int
    best_delta: float
    deltas: dict[str, float]


class DirectionSelectorHead(nn.Module):
    def __init__(self, input_channels: int = 11, hidden_channels: int = 32, zero_bias: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, len(CHOICES), kernel_size=1),
        )
        self.reset_parameters(zero_bias)

    def reset_parameters(self, zero_bias: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Conv2d) and last.bias is not None:
            nn.init.zeros_(last.bias)
            last.bias.data[CHOICE_TO_INDEX["fallback"]] = float(zero_bias)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        logits = self.net(maps)
        return logits.mean(dim=(0, 2, 3))


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
    p.add_argument(
        "--oracle-csv",
        type=Path,
        default=ROOT / "experiments/analysis/e306_eflic_direction_oracle.csv",
    )
    p.add_argument("--output-prefix", type=Path, default=ROOT / "experiments/analysis/e308_eflic_direction_selector_context_train")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=308)
    p.add_argument("--holdout-mod", type=int, default=4)
    p.add_argument("--holdout-rem", type=int, default=0)
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--fallback-fp-weight", type=float, default=2.0)
    p.add_argument("--delta-weight", type=float, default=100.0)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_oracle(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            choice = row["best_with_fallback"]
            if choice not in CHOICE_TO_INDEX:
                raise RuntimeError(f"unknown oracle choice {choice!r}")
            deltas = {
                "fallback": 0.0,
                "mean": safe_float(row["mean_delta_psnr"]),
                "logscale": safe_float(row["logscale_delta_psnr"]),
                "fixed": safe_float(row["fixed_delta_psnr"]),
            }
            rows[(row["dataset"], row["image"])] = {
                "target_choice": choice,
                "best_delta": safe_float(row["best_with_fallback_delta_psnr"]),
                "deltas": deltas,
            }
    return rows


def read_manifest(path: Path, dataset_name: str, oracle: dict[tuple[str, str], dict[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            image = row["image"]
            key = (dataset_name, image)
            if key not in oracle:
                continue
            item = oracle[key]
            tensor_path = Path(row["tensor_path"])
            if not tensor_path.is_absolute():
                tensor_path = ROOT / tensor_path
            target_choice = item["target_choice"]
            samples.append(
                Sample(
                    index=len(samples),
                    dataset=dataset_name,
                    image=image,
                    tensor_path=tensor_path,
                    target_choice=target_choice,
                    target_index=CHOICE_TO_INDEX[target_choice],
                    best_delta=float(item["best_delta"]),
                    deltas=dict(item["deltas"]),
                )
            )
    return samples


def load_maps(sample: Sample, device: torch.device) -> torch.Tensor:
    obj = torch.load(sample.tensor_path, map_location="cpu")
    maps = obj["context_maps"].float()
    if maps.ndim != 4 or maps.shape[1] != 11:
        raise RuntimeError(f"bad context shape {tuple(maps.shape)} for {sample.tensor_path}")
    if not torch.isfinite(maps).all().item():
        raise RuntimeError(f"nonfinite context in {sample.tensor_path}")
    return maps.to(device=device, non_blocking=True)


def class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = Counter(sample.target_index for sample in samples)
    weights = torch.ones(len(CHOICES), dtype=torch.float32)
    for idx in range(len(CHOICES)):
        weights[idx] = len(samples) / max(1, len(CHOICES) * counts.get(idx, 0))
    return weights.to(device)


def split_mod(samples: list[Sample], holdout_mod: int, holdout_rem: int) -> tuple[list[Sample], list[Sample]]:
    if holdout_mod <= 1:
        return samples, []
    train = [s for idx, s in enumerate(samples) if idx % holdout_mod != holdout_rem]
    val = [s for idx, s in enumerate(samples) if idx % holdout_mod == holdout_rem]
    if not train or not val:
        raise SystemExit("empty train/val split")
    return train, val


def train_head(
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
) -> tuple[DirectionSelectorHead, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    head = DirectionSelectorHead(hidden_channels=args.hidden_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(samples, device) if args.use_class_weights else None
    losses: list[float] = []
    for _ in range(args.epochs):
        order = list(samples)
        random.shuffle(order)
        epoch_losses: list[float] = []
        head.train()
        for sample in order:
            maps = load_maps(sample, device)
            logits = head(maps)[None]
            target = torch.tensor([sample.target_index], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target, weight=weights)
            probs = logits.softmax(dim=1)
            if sample.target_choice == "fallback":
                nonfallback_prob = 1.0 - probs[:, CHOICE_TO_INDEX["fallback"]]
                loss = loss + float(args.fallback_fp_weight) * nonfallback_prob.mean()
            if sample.best_delta > 0:
                target_prob = probs[:, sample.target_index]
                loss = loss + float(args.delta_weight) * float(sample.best_delta) * (1.0 - target_prob).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
    return head, float(losses[-1]) if losses else float("nan")


@torch.inference_mode()
def evaluate(head: DirectionSelectorHead, samples: list[Sample], device: torch.device, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head.eval()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        maps = load_maps(sample, device)
        logits = head(maps)
        probs = logits.softmax(dim=0)
        pred_idx = int(probs.argmax().item())
        pred_choice = CHOICES[pred_idx]
        score = float(sample.deltas[pred_choice])
        rows.append(
            {
                "split": split,
                "dataset": sample.dataset,
                "image": sample.image,
                "target_choice": sample.target_choice,
                "pred_choice": pred_choice,
                "correct": int(pred_choice == sample.target_choice),
                "target_nonfallback": int(sample.target_choice != "fallback"),
                "pred_nonfallback": int(pred_choice != "fallback"),
                "false_positive_nonfallback": int(sample.target_choice == "fallback" and pred_choice != "fallback"),
                "missed_nonfallback": int(sample.target_choice != "fallback" and pred_choice == "fallback"),
                "pred_delta_psnr": score,
                "oracle_delta_psnr": sample.best_delta,
                "regret": sample.best_delta - score,
                **{f"prob_{name}": float(probs[idx].item()) for idx, name in enumerate(CHOICES)},
                **{f"{name}_delta_psnr": sample.deltas[name] for name in CHOICES},
            }
        )
    summary = summarize_rows(split, rows)
    return summary, rows


def summarize_rows(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    scores = [float(r["pred_delta_psnr"]) for r in rows]
    regrets = [float(r["regret"]) for r in rows]
    fallback_targets = [r for r in rows if r["target_choice"] == "fallback"]
    active_targets = [r for r in rows if r["target_choice"] != "fallback"]
    return {
        "split": split,
        "images": len(rows),
        "accuracy": sum(int(r["correct"]) for r in rows) / n,
        "mean_delta_psnr": sum(scores) / n,
        "worst_delta_psnr": min(scores) if scores else 0.0,
        "win_frac": sum(1 for s in scores if s > 0.0) / n,
        "nonnegative_frac": sum(1 for s in scores if s >= 0.0) / n,
        "mean_regret": sum(regrets) / n,
        "pred_nonfallback_frac": sum(int(r["pred_nonfallback"]) for r in rows) / n,
        "target_nonfallback_frac": sum(int(r["target_nonfallback"]) for r in rows) / n,
        "false_positive_nonfallback_frac": sum(int(r["false_positive_nonfallback"]) for r in fallback_targets) / max(1, len(fallback_targets)),
        "missed_nonfallback_frac": sum(int(r["missed_nonfallback"]) for r in active_targets) / max(1, len(active_targets)),
        "target_counts": dict(Counter(str(r["target_choice"]) for r in rows)),
        "pred_counts": dict(Counter(str(r["pred_choice"]) for r in rows)),
    }


def add_baselines(summaries: list[dict[str, Any]], samples: list[Sample], split: str) -> None:
    for policy in CHOICES:
        rows = []
        for sample in samples:
            score = float(sample.deltas[policy])
            rows.append(
                {
                    "target_choice": sample.target_choice,
                    "pred_choice": policy,
                    "correct": int(policy == sample.target_choice),
                    "target_nonfallback": int(sample.target_choice != "fallback"),
                    "pred_nonfallback": int(policy != "fallback"),
                    "false_positive_nonfallback": int(sample.target_choice == "fallback" and policy != "fallback"),
                    "missed_nonfallback": int(sample.target_choice != "fallback" and policy == "fallback"),
                    "pred_delta_psnr": score,
                    "regret": sample.best_delta - score,
                }
            )
        summary = summarize_rows(split, rows)
        summary["policy"] = f"always_{policy}"
        summaries.append(summary)
    oracle_rows = []
    for sample in samples:
        oracle_rows.append(
            {
                "target_choice": sample.target_choice,
                "pred_choice": sample.target_choice,
                "correct": 1,
                "target_nonfallback": int(sample.target_choice != "fallback"),
                "pred_nonfallback": int(sample.target_choice != "fallback"),
                "false_positive_nonfallback": 0,
                "missed_nonfallback": 0,
                "pred_delta_psnr": sample.best_delta,
                "regret": 0.0,
            }
        )
    summary = summarize_rows(split, oracle_rows)
    summary["policy"] = "oracle_label"
    summaries.append(summary)


def write_outputs(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    samples: list[Sample],
    checkpoint_path: Path | None,
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    pred_path = args.output_prefix.with_suffix(".predictions.csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    with summary_path.open("w", newline="", encoding="utf-8") as fobj:
        preferred_fields = [
            "split",
            "policy",
            "train_loss",
            "images",
            "accuracy",
            "mean_delta_psnr",
            "worst_delta_psnr",
            "win_frac",
            "nonnegative_frac",
            "mean_regret",
            "pred_nonfallback_frac",
            "target_nonfallback_frac",
            "false_positive_nonfallback_frac",
            "missed_nonfallback_frac",
            "target_counts",
            "pred_counts",
        ]
        extra_fields = sorted({key for row in summaries for key in row} - set(preferred_fields))
        fields = preferred_fields + extra_fields
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with pred_path.open("w", newline="", encoding="utf-8") as fobj:
        fields = sorted({key for row in predictions for key in row})
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    payload = {
        "experiment": "E308 EF-LIC direction selector context train",
        "purpose": "Test whether E242 decoder-safe context tensors can learn E306 fallback/mean/logscale/fixed oracle labels.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "choices": CHOICES,
        "samples": [{"dataset": s.dataset, "image": s.image, "target_choice": s.target_choice, "best_delta": s.best_delta} for s in samples],
        "summaries": summaries,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as fobj:
        fobj.write("# E308 EF-LIC Direction Selector Context Train\n\n")
        fobj.write("This trains a tiny decoder-safe context head to predict the E306 image-level oracle choice among fallback/mean/logscale/fixed. It is a trainability/proxy audit, not final codec RD evidence.\n\n")
        fobj.write(f"- Images: `{len(samples)}`\n")
        fobj.write(f"- Device: `{args.device}`\n")
        fobj.write(f"- Epochs: `{args.epochs}`\n")
        fobj.write(f"- Use class weights: `{bool(args.use_class_weights)}`\n")
        if checkpoint_path:
            fobj.write(f"- Checkpoint: `{checkpoint_path}`\n")
        fobj.write("\n| split | policy | images | accuracy | mean dPSNR | worst | win | nonnegative | FP nonfallback | missed nonfallback | pred counts |\n")
        fobj.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for item in summaries:
            fobj.write(
                f"| {item['split']} | {item.get('policy', 'trained_head')} | {item['images']} | "
                f"{item['accuracy']:.6f} | {item['mean_delta_psnr']:+.6f} | {item['worst_delta_psnr']:+.6f} | "
                f"{item['win_frac']:.6f} | {item['nonnegative_frac']:.6f} | "
                f"{item['false_positive_nonfallback_frac']:.6f} | {item['missed_nonfallback_frac']:.6f} | "
                f"`{json.dumps(item['pred_counts'], sort_keys=True)}` |\n"
            )
        fobj.write("\nInterpretation guardrails:\n\n")
        fobj.write("- Same-table performance is only a capacity check; held-out/domain rows determine whether this selector is promising.\n")
        fobj.write("- If the trained head collapses to fallback, context labels are too sparse for direct deployment and we need slice/local labels or more data.\n")
        fobj.write("- If it beats simple thresholds while keeping worst rows bounded, it justifies adding a real direction/fallback head to the EF-LIC codec path.\n")
    print(f"wrote {summary_path}, {pred_path}, {json_path}, {md_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    oracle = read_oracle(args.oracle_csv)
    samples = []
    samples.extend(read_manifest(args.kodak_manifest, "kodak24", oracle))
    samples.extend(read_manifest(args.clicpro_manifest, "clicpro16", oracle))
    if not samples:
        raise SystemExit("no matched samples")
    train_samples, val_samples = split_mod(samples, args.holdout_mod, args.holdout_rem)
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    add_baselines(summaries, samples, "pooled")
    add_baselines(summaries, val_samples, "mod_holdout")

    head, train_loss = train_head(train_samples, args, device, seed=args.seed)
    for split, split_samples in [("train", train_samples), ("mod_holdout", val_samples), ("pooled", samples)]:
        summary, rows = evaluate(head, split_samples, device, split)
        summary["policy"] = "trained_head"
        summary["train_loss"] = train_loss
        summaries.append(summary)
        predictions.extend(rows)

    datasets = sorted({s.dataset for s in samples})
    for held in datasets:
        held_train = [s for s in samples if s.dataset != held]
        held_eval = [s for s in samples if s.dataset == held]
        held_head, held_loss = train_head(held_train, args, device, seed=args.seed + 100 + datasets.index(held))
        summary, rows = evaluate(held_head, held_eval, device, f"leave_dataset_out:{held}")
        summary["policy"] = "trained_head"
        summary["train_loss"] = held_loss
        summaries.append(summary)
        predictions.extend(rows)

    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = args.output_prefix.with_suffix(".pth")
        torch.save(
            {
                "model": head.state_dict(),
                "choices": CHOICES,
                "hidden_channels": args.hidden_channels,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            checkpoint_path,
        )
    write_outputs(args, summaries, predictions, samples, checkpoint_path)


if __name__ == "__main__":
    main()
