#!/usr/bin/env python3
"""Evaluate a saved GLC/HCG-RVQ branch checkpoint without further training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import get_state_dict  # noqa: E402
from hcg_rvq.reliability_index_controller import (  # noqa: E402
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    list_images,
    prepare_images,
)
from tools.run_e175_glc_decoder_aware_tail_vq_train import TrainableRVQCodebooks  # noqa: E402
from tools.run_e263_glc_fallback_gate_codec_loop_pilot import (  # noqa: E402
    FEATURES,
    evaluate_policies,
    summarize,
    wandb_log_summary,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch-checkpoint", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="*", default=None)
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=24)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--active-threshold", type=float, default=0.5)
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--rate-cap-dbpp", type=float, default=-1.0)
    p.add_argument("--emit-progressive-extra-rows", action="store_true")
    p.add_argument("--progressive-extra-cap-bpp", type=float, default=-1.0)
    p.add_argument("--emit-replacement-rows", action="store_true")
    p.add_argument("--replacement-cap-dbpp", type=float, default=-1.0)
    p.add_argument("--replacement-cap-dbpp-values", type=float, nargs="*", default=[])
    p.add_argument("--replacement-signal-bits", type=float, nargs="*", default=[])
    p.add_argument("--qaware-controller-json", type=Path, default=None)
    p.add_argument("--qaware-policy-modes", nargs="*", default=["q-aware", "global"])
    p.add_argument("--controller-hidden", type=int, default=None)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--mse-weight", type=float, default=0.0)
    p.add_argument("--lpips-weight", type=float, default=0.30)
    p.add_argument("--dists-weight", type=float, default=1.0)
    p.add_argument("--soft-index-weight", type=float, default=0.005)
    p.add_argument("--gate-rate-weight", type=float, default=1.0)
    p.add_argument("--gate-l1-weight", type=float, default=0.01)
    p.add_argument("--train-dir", type=Path, default=Path("."))
    p.add_argument("--train-start-index", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--train-crop-size", type=int, default=256)
    return p.parse_args()


def key_from_param_name(name: str) -> tuple[int, int] | None:
    if not name.startswith("params."):
        return None
    token = name.split(".", 1)[1]
    key_token, stage_token = token.rsplit("_s", 1)
    if key_token.startswith("m"):
        key = -int(key_token[1:])
    elif key_token.startswith("p"):
        key = int(key_token[1:])
    else:
        return None
    return key, int(stage_token)


def codebooks_from_state_dict(state_dict: dict[str, torch.Tensor], device: torch.device) -> TrainableRVQCodebooks:
    staged: dict[int, dict[int, torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        parsed = key_from_param_name(str(name))
        if parsed is None:
            continue
        key, stage = parsed
        staged.setdefault(key, {})[stage] = tensor.detach().cpu()
    initial = {key: [stages[i] for i in sorted(stages)] for key, stages in staged.items()}
    module = TrainableRVQCodebooks(initial, device).to(device)
    module.load_state_dict(state_dict, strict=True)
    return module


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload: dict[str, Any] = torch.load(args.branch_checkpoint, map_location="cpu")
    ckpt_args = payload.get("args", {})
    if args.eval_dir is None:
        args.eval_dir = Path(str(ckpt_args.get("eval_dir", ROOT / "experiments" / "data" / "kodak24")))
    if args.q_indexes is None:
        args.q_indexes = [int(q) for q in payload["codebooks_by_q"].keys()]
    if args.controller_hidden is None:
        args.controller_hidden = int(ckpt_args.get("controller_hidden", 16))

    import DISTS_pytorch as dists
    import lpips

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for param in lpips_fn.parameters():
        param.requires_grad_(False)
    dists_fn = dists.DISTS().to(device).eval()
    for param in dists_fn.parameters():
        param.requires_grad_(False)

    net = GLC_Image(inplace=False).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for param in net.parameters():
        param.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    net.masks = {}

    codebooks_by_q = {
        int(q): codebooks_from_state_dict(state, device)
        for q, state in payload["codebooks_by_q"].items()
        if int(q) in set(args.q_indexes)
    }
    controller = ReliabilityIndexMLP(
        ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=args.controller_hidden, zero_bias=-2.0)
    ).to(device)
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval()

    feature_mu = {str(k): float(v) for k, v in payload["feature_mu"].items()}
    feature_std = {str(k): float(v) for k, v in payload["feature_std"].items()}
    eval_paths = list_images(args.eval_dir, args.eval_start_index, args.eval_limit)
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.eval_dir}")
    eval_prepared = prepare_images(eval_paths, device, args.padding_size, args.eval_crop_size)

    label = f"ckpt_step{int(payload.get('step', -1)):04d}"
    rows = evaluate_policies(
        net,
        official_forward,
        controller,
        codebooks_by_q,
        eval_prepared,
        args.q_indexes,
        args,
        lpips_fn,
        dists_fn,
        feature_mu,
        feature_std,
        label,
    )
    summary = summarize(rows)
    write_outputs(args, [], eval_paths, rows, summary, [], feature_mu, feature_std)
    print(json.dumps({"checkpoint": str(args.branch_checkpoint), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
