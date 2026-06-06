#!/usr/bin/env python3
"""Tiny trainability pilot for an HCG adapter on frozen local CompressAI mbt2018_mean."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from compressai.zoo import mbt2018_mean
from torch.utils.data import DataLoader

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_psnr
from hcg_rvq.quantizers import HCGQuantizerAdapter


OUT_DIR = ROOT / "experiments" / "e125_mbt2018_hcg_adapter_trainability_pilot"
ANALYSIS_PREFIX = ROOT / "experiments" / "analysis" / "e125_mbt2018_hcg_adapter_trainability_pilot"


def tagged_paths(tag: str) -> tuple[Path, Path]:
    if not tag:
        return OUT_DIR, ANALYSIS_PREFIX
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag).strip("_")
    if not safe:
        return OUT_DIR, ANALYSIS_PREFIX
    return OUT_DIR.with_name(f"{OUT_DIR.name}_{safe}"), ANALYSIS_PREFIX.with_name(f"{ANALYSIS_PREFIX.name}_{safe}")


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def scalar(value: object) -> float | None:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


class FrozenMbt2018HCG(torch.nn.Module):
    def __init__(self, adapter: HCGQuantizerAdapter) -> None:
        super().__init__()
        self.backbone = mbt2018_mean(quality=1, pretrained=False)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        image_hw = (int(x.shape[-2]), int(x.shape[-1]))
        with torch.no_grad():
            y = self.backbone.g_a(x)
            z = self.backbone.h_a(y)
            z_hat, z_likelihoods = self.backbone.entropy_bottleneck(z)
            hyper_features = self.backbone.h_s(z_hat)
        y_hat, indices, commit_loss, rvq_stats, conditioning_tensors = self.adapter(y, hyper_features, image_hw)
        x_hat = self.backbone.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"z": z_likelihoods},
            "y": y,
            "y_hat": y_hat,
            "hyper_features": hyper_features,
            "indices": indices,
            "commit_loss": commit_loss,
            "rvq_stats": rvq_stats,
            "conditioning_tensors": conditioning_tensors,
            "bpp_y_index": rvq_stats["fixed_bpp"],
        }


def inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def make_model(args: argparse.Namespace, device: torch.device) -> FrozenMbt2018HCG:
    adapter = HCGQuantizerAdapter(
        latent_channels=192,
        hyper_channels=384,
        variant=args.variant,
        group_size=args.group_size,
        num_stages=args.num_stages,
        codebook_size=args.codebook_size,
        householder_bias_init_scale=args.householder_bias_init_scale,
        householder_gate_enabled=args.householder_gate_enabled,
        householder_gate_max=args.householder_gate_max,
        householder_gate_init=args.householder_gate_init,
    )
    return FrozenMbt2018HCG(adapter).to(device)


def nonfinite_output(output: dict[str, object], losses: dict[str, torch.Tensor]) -> int:
    count = 0
    for value in [output["x_hat"], output["y_hat"], output["bpp_y_index"], output["commit_loss"]]:
        if torch.is_tensor(value):
            count += int((~torch.isfinite(value)).sum().detach().cpu())
    for value in output.get("rvq_stats", {}).values():
        if torch.is_tensor(value):
            count += int((~torch.isfinite(value)).sum().detach().cpu())
    for value in losses.values():
        if torch.is_tensor(value):
            count += int((~torch.isfinite(value)).sum().detach().cpu())
    return count


def evaluate(
    model: FrozenMbt2018HCG,
    loader: DataLoader,
    criterion: RateDistortionLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            x_pad, hw = pad_to_multiple(x)
            output = model(x_pad)
            output["x_hat"] = crop_to_hw(output["x_hat"], hw)
            losses = criterion(output, x)
            totals["loss"].append(float(losses["loss"].detach().cpu()))
            totals["bpp"].append(float(losses["bpp_total"].detach().cpu()))
            totals["bpp_y"].append(float(losses["bpp_y"].detach().cpu()))
            totals["bpp_z"].append(float(losses["bpp_z"].detach().cpu()))
            totals["mse"].append(float(losses["mse"].detach().cpu()))
            totals["psnr"].append(float(compute_psnr(x, output["x_hat"]).detach().cpu()))
            totals["nonfinite"].append(float(nonfinite_output(output, losses)))
            rvq_stats = output["rvq_stats"]
            for key in (
                "latent_quant_mse",
                "dead_code_ratio",
                "perplexity",
                "stage_entropy",
                "s_q_mean",
                "s_q_std",
                "mu_q_abs_mean",
                "householder_delta_rms",
                "householder_v_abs_mean",
            ):
                value = scalar(rvq_stats.get(key))
                if value is not None:
                    totals[key].append(value)
    lambda_rd = float(criterion.lambda_rd)
    mse_scale = float(criterion.mse_scale)
    out = {key: sum(values) / len(values) for key, values in totals.items() if values}
    out["rd_score"] = out["bpp"] + lambda_rd * mse_scale * out["mse"]
    return out


def load_adapter_checkpoint(model: FrozenMbt2018HCG, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.init_adapter_checkpoint:
        return {"loaded": False}
    checkpoint_path = Path(args.init_adapter_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("adapter_state_dict", checkpoint)
    load_result = model.adapter.load_state_dict(state_dict, strict=True)
    postload_actions: list[str] = []
    with torch.no_grad():
        if args.postload_householder_bias_init_scale > 0.0:
            model.adapter.householder_head.weight.zero_()
            model.adapter.householder_head.bias.normal_(mean=0.0, std=args.postload_householder_bias_init_scale)
            postload_actions.append("reset_householder_head_bias")
        if args.postload_householder_gate_init is not None:
            if not 0.0 < args.postload_householder_gate_init < args.householder_gate_max:
                raise ValueError("postload householder gate init must stay in (0, householder_gate_max)")
            model.adapter.householder_gate_head.weight.zero_()
            gate_ratio = args.postload_householder_gate_init / args.householder_gate_max
            model.adapter.householder_gate_head.bias.fill_(inverse_sigmoid(gate_ratio))
            postload_actions.append("reset_householder_gate_bias")
    return {
        "loaded": True,
        "path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "postload_actions": postload_actions,
    }


def save_checkpoint(model: FrozenMbt2018HCG, step: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_step_{step}.pth.tar"
    torch.save({"step": step, "adapter_state_dict": model.adapter.state_dict()}, path)
    return path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="/dpl/openimages")
    parser.add_argument("--eval-root", default="/dpl/kodak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument(
        "--variant",
        default="hcg_rvq_h",
        choices=["global_rvq", "hcs_rvq", "hcg_rvq_h", "hcg_rvq_h_no_transform"],
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--householder-bias-init-scale", type=float, default=0.0)
    parser.add_argument("--householder-gate-enabled", action="store_true")
    parser.add_argument("--householder-gate-max", type=float, default=0.45)
    parser.add_argument("--householder-gate-init", type=float, default=0.25)
    parser.add_argument("--init-adapter-checkpoint", default="")
    parser.add_argument("--postload-householder-bias-init-scale", type=float, default=0.0)
    parser.add_argument("--postload-householder-gate-init", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rd", type=float, default=0.0035)
    parser.add_argument("--beta-commit", type=float, default=0.05)
    parser.add_argument("--continue-on-nonfinite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir, analysis_prefix = tagged_paths(args.tag)
    torch.manual_seed(1234)
    train_dataset = ImageFolderDataset(
        [args.train_root],
        patch_size=args.patch_size,
        training=True,
        max_images=args.train_images,
    )
    eval_dataset = ImageFolderDataset(
        [args.eval_root],
        patch_size=None,
        training=False,
        max_images=args.eval_images,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = make_model(args, device)
    load_info = load_adapter_checkpoint(model, args, device)
    criterion = RateDistortionLoss(lambda_rd=args.lambda_rd, beta_commit=args.beta_commit)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=args.lr)

    checkpoints: list[Path] = [save_checkpoint(model, 0, out_dir)]
    eval_rows: list[dict[str, object]] = [{"step": 0, "checkpoint": str(checkpoints[-1]), **evaluate(model, eval_loader, criterion, device)}]
    train_rows: list[dict[str, object]] = []

    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        x = batch.to(device, non_blocking=True)
        x_pad, hw = pad_to_multiple(x)
        optimizer.zero_grad(set_to_none=True)
        output = model(x_pad)
        output["x_hat"] = crop_to_hw(output["x_hat"], hw)
        losses = criterion(output, x)
        losses["loss"].backward()
        grad_nonfinite = 0
        for param in model.adapter.parameters():
            if param.grad is not None:
                grad_nonfinite += int((~torch.isfinite(param.grad)).sum().detach().cpu())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=10.0)
        grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        row_nonfinite = nonfinite_output(output, losses)
        skipped_step = int(row_nonfinite > 0 or grad_nonfinite > 0 or not math.isfinite(grad_norm_value))
        if not skipped_step:
            optimizer.step()
        rvq_stats = output["rvq_stats"]
        row = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu()),
            "bpp": float(losses["bpp_total"].detach().cpu()),
            "bpp_y": float(losses["bpp_y"].detach().cpu()),
            "bpp_z": float(losses["bpp_z"].detach().cpu()),
            "mse": float(losses["mse"].detach().cpu()),
            "commit_loss": float(losses["commit_loss"].detach().cpu()),
            "grad_norm": grad_norm_value,
            "grad_nonfinite": grad_nonfinite,
            "nonfinite": row_nonfinite,
            "skipped_step": skipped_step,
        }
        for key in ("latent_quant_mse", "dead_code_ratio", "perplexity", "s_q_mean", "s_q_std", "householder_delta_rms"):
            value = scalar(rvq_stats.get(key))
            if value is not None:
                row[key] = value
        train_rows.append(row)
        if skipped_step and not args.continue_on_nonfinite:
            break
        if step in {args.steps // 2, args.steps}:
            checkpoint = save_checkpoint(model, step, out_dir)
            eval_rows.append({"step": step, "checkpoint": str(checkpoint), **evaluate(model, eval_loader, criterion, device)})
            model.train()

    completed_steps = int(train_rows[-1]["step"]) if train_rows else 0
    all_finite = all(
        row.get("nonfinite", 0) == 0 and row.get("grad_nonfinite", 0) == 0 and row.get("skipped_step", 0) == 0
        for row in train_rows + eval_rows
    )
    result = {
        "experiment": "E125 mbt2018 HCG adapter trainability pilot",
        "status": "pass" if completed_steps == args.steps and all_finite else "fail",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": vars(args),
        "completed_steps": completed_steps,
        "out_dir": str(out_dir),
        "analysis_prefix": str(analysis_prefix),
        "load_info": load_info,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "interpretation": (
            "This is a trainability and checkpoint/feature smoke for a random local CompressAI backbone, "
            "not a quality or SOTA result."
        ),
    }
    analysis_prefix.parent.mkdir(parents=True, exist_ok=True)
    analysis_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(analysis_prefix.with_name(analysis_prefix.name + "_train.csv"), train_rows)
    write_csv(analysis_prefix.with_name(analysis_prefix.name + "_eval.csv"), eval_rows)
    lines = [
        "# E125 mbt2018 HCG Adapter Trainability Pilot",
        "",
        f"- Status: `{result['status']}`",
        f"- Device: `{result['device']}`, CUDA_VISIBLE_DEVICES=`{result['cuda_visible_devices']}`",
        f"- Variant: `{args.variant}`",
        f"- Tag: `{args.tag or 'none'}`",
        f"- Steps: `{args.steps}`",
        f"- Completed steps: `{completed_steps}`",
        f"- Init checkpoint: `{load_info.get('path', 'none')}`",
        "",
        "## Evaluation Checkpoints",
        "",
    ]
    for row in eval_rows:
        lines.append(
            "- step `{step}`: rd `{rd:.6f}`, bpp `{bpp:.6f}`, mse `{mse:.6f}`, psnr `{psnr:.3f}`, qMSE `{qmse:.6f}`, dead `{dead:.6f}`, nonfinite `{nonfinite}`".format(
                step=row["step"],
                rd=row["rd_score"],
                bpp=row["bpp"],
                mse=row["mse"],
                psnr=row["psnr"],
                qmse=row.get("latent_quant_mse", float("nan")),
                dead=row.get("dead_code_ratio", float("nan")),
                nonfinite=row.get("nonfinite", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The pilot checks whether the standalone adapter can be optimized, checkpointed, and evaluated on a frozen local CompressAI backbone with finite RD and feature statistics. Because the local backbone is not a pretrained quality baseline in this run, these numbers are not paper quality claims.",
            "",
            "## Artifacts",
            "",
            f"- `{analysis_prefix.with_suffix('.json')}`",
            f"- `{analysis_prefix.with_name(analysis_prefix.name + '_train.csv')}`",
            f"- `{analysis_prefix.with_name(analysis_prefix.name + '_eval.csv')}`",
            f"- `{out_dir}`",
        ]
    )
    analysis_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
