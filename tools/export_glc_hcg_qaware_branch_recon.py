#!/usr/bin/env python3
"""Export GLC/HCG-RVQ reconstructions for official GLC metric evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
GLC_DIR = ROOT / "third_party" / "GLC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLC_DIR))

from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.metric_image import evaluate_quality  # noqa: E402
from src.utils.test_utils import get_state_dict, init_func, write_image  # noqa: E402
from hcg_rvq.reliability_index_controller import (  # noqa: E402
    qaware_threshold_gate,
    ReliabilityIndexMLP,
    ReliabilityIndexMLPConfig,
    mix_with_fallback,
)
from tools.eval_glc_qaware_branch_checkpoint import codebooks_from_state_dict  # noqa: E402
from tools.run_e175_glc_decoder_aware_tail_vq_train import (  # noqa: E402
    crop_to_image,
    install_trainable_branch,
    run_instrumented,
)
from tools.run_e177_glc_decoder_aware_tail_vq_split_train import (  # noqa: E402
    list_images,
    prepare_images,
)
from tools.run_e263_glc_fallback_gate_codec_loop_pilot import (  # noqa: E402
    FEATURES,
    branch_feature_dict,
    cap_token,
    feature_tensor,
    image_signal_bpp,
    load_qaware_specs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch-checkpoint", type=Path, required=True)
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, default=GLC_DIR / "checkpoints" / "GLC_image.pth.tar")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--q-indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--eval-start-index", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=100000)
    p.add_argument("--padding-size", type=int, default=64)
    p.add_argument("--eval-crop-size", type=int, default=0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--active-groups", type=int, nargs="*", default=[1, 7, 10, 15])
    p.add_argument("--active-parts", type=int, nargs="*", default=[0, 1])
    p.add_argument("--scope", default="part_group", choices=["part_group", "shared"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--active-threshold", type=float, default=0.5)
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--controller-hidden", type=int, default=None)
    p.add_argument("--labels", nargs="+", default=["base", "hard_gate"])
    p.add_argument("--replacement-signal-bits", type=float, nargs="*", default=[], help="Optional image-level selection/mode signal costs for deployable selected/fallback rows.")
    p.add_argument("--qaware-controller-json", type=Path, default=None, help="Optional E379-style q-aware deployment JSON. Adds q-aware hard replacement rows.")
    p.add_argument("--qaware-policy-modes", nargs="*", default=["q-aware", "global"], help="Policy modes to load from --qaware-controller-json. Use an empty list to load all modes.")
    p.add_argument("--fid-patch-size", type=int, default=256)
    p.add_argument("--skip-quality", action="store_true")
    return p.parse_args()


def load_branch(args: argparse.Namespace, device: torch.device):
    payload: dict[str, Any] = torch.load(args.branch_checkpoint, map_location="cpu")
    hidden = args.controller_hidden or int(payload.get("args", {}).get("controller_hidden", 16))
    codebooks_by_q = {
        int(q): codebooks_from_state_dict(state, device)
        for q, state in payload["codebooks_by_q"].items()
        if int(q) in set(args.q_indexes)
    }
    controller = ReliabilityIndexMLP(
        ReliabilityIndexMLPConfig(input_dim=len(FEATURES), hidden_dim=hidden, zero_bias=-2.0)
    ).to(device)
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval()
    feature_mu = {str(k): float(v) for k, v in payload["feature_mu"].items()}
    feature_std = {str(k): float(v) for k, v in payload["feature_std"].items()}
    return payload, codebooks_by_q, controller, feature_mu, feature_std


def export_one(tensor: torch.Tensor, out_dir: Path, image_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(str(out_dir / image_name), tensor.clamp(-1, 1))


def main() -> None:
    init_func()
    args = parse_args()
    device = torch.device(args.device)
    payload, codebooks_by_q, controller, feature_mu, feature_std = load_branch(args, device)

    net = GLC_Image(inplace=True).to(device).eval()
    net.load_state_dict(get_state_dict(str(args.ckpt_path)), strict=True)
    for param in net.parameters():
        param.requires_grad_(False)
    official_forward = net.forward_four_part_prior
    net.masks = {}

    branch_args = SimpleNamespace(
        group_size=args.group_size,
        active_groups=args.active_groups,
        active_parts=args.active_parts,
        scope=args.scope,
        k=args.k,
        stages=args.stages,
    )

    eval_paths = list_images(args.input_path, args.eval_start_index, args.eval_limit)
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.input_path}")

    qaware_specs = load_qaware_specs(args.qaware_controller_json, args.qaware_policy_modes)
    dynamic_labels: list[str] = list(args.labels)
    for spec_row in qaware_specs:
        base_label = f"{spec_row['tag']}_replacement_hard"
        dynamic_labels.append(base_label)
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{base_label}_sig{cap_token(float(signal_bits))}b")
    dynamic_labels = list(dict.fromkeys(dynamic_labels))

    bpps: dict[tuple[str, int], list[float]] = {(label, q): [] for label in dynamic_labels for q in args.q_indexes}
    rows = []
    with torch.no_grad():
        for q in args.q_indexes:
            if q not in codebooks_by_q:
                raise SystemExit(f"q={q} missing from branch checkpoint")
            for idx, path in enumerate(eval_paths):
                prepared = prepare_images([path], device, args.padding_size, args.eval_crop_size)
                item = prepared[0]
                pixels = float(item.height * item.width)
                image_name = path.with_suffix(".png").name

                net.forward_four_part_prior = official_forward
                base_pad, base_stats = run_instrumented(net, item.x_pad, q)
                install_trainable_branch(net, codebooks_by_q[q], branch_args)
                branch_pad, branch_stats = run_instrumented(net, item.x_pad, q)
                net.forward_four_part_prior = official_forward

                base = crop_to_image(base_pad, item)
                branch = crop_to_image(branch_pad, item)
                feature_row = branch_feature_dict(base_stats, branch_stats, pixels)
                base_bpp = float(feature_row["base_bpp"])
                branch_bpp = base_bpp + float(feature_row["empirical_bpp_delta"])
                replacement_bpp = base_bpp + float(feature_row["active_replacement_delta_bpp"])
                features = feature_tensor(feature_row, feature_mu, feature_std, base.device)
                ctrl = controller(features)
                soft_mixed, soft_gate = mix_with_fallback(
                    base,
                    branch,
                    ctrl["active_logit"],
                    active_threshold=args.active_threshold,
                    hard=False,
                    max_gate=args.max_gate,
                )
                hard_mixed, hard_gate = mix_with_fallback(
                    base,
                    branch,
                    ctrl["active_logit"],
                    active_threshold=args.active_threshold,
                    hard=True,
                    max_gate=args.max_gate,
                )
                soft_gate_mean = float(soft_gate.mean().item())
                hard_gate_mean = float(hard_gate.mean().item())

                tensors = {
                    "base": base,
                    "all_on": branch,
                    "soft_gate": soft_mixed,
                    "hard_gate": hard_mixed,
                    "replacement_soft": soft_mixed,
                    "replacement_hard": hard_mixed,
                }
                label_bpp = {
                    "base": base_bpp,
                    "all_on": branch_bpp,
                    "soft_gate": base_bpp + soft_gate_mean * float(feature_row["empirical_bpp_delta"]),
                    "hard_gate": base_bpp + hard_gate_mean * float(feature_row["empirical_bpp_delta"]),
                    "replacement_soft": replacement_bpp,
                    "replacement_hard": base_bpp + hard_gate_mean * float(feature_row["active_replacement_delta_bpp"]),
                }
                for label in args.labels:
                    out_dir = args.output_root / label / f"q{q}"
                    export_one(tensors[label], out_dir, image_name)
                    bpps[(label, q)].append(float(label_bpp[label]))
                for spec_row in qaware_specs:
                    feature_name = str(spec_row["feature"])
                    feature_value = float(feature_row.get(feature_name, float("nan")))
                    selected = False
                    if torch.isfinite(torch.tensor(feature_value)):
                        gate_tensor = qaware_threshold_gate(
                            torch.tensor([feature_value], dtype=torch.float32, device=base.device),
                            int(q),
                            spec_row["spec"],
                            hard=True,
                        )
                        selected = bool(float(gate_tensor.item()) > 0.5)
                    selected_tensor = branch if selected else base
                    selected_bpp = replacement_bpp if selected else base_bpp
                    base_label = f"{spec_row['tag']}_replacement_hard"
                    export_one(selected_tensor, args.output_root / base_label / f"q{q}", image_name)
                    bpps[(base_label, q)].append(float(selected_bpp))
                    for signal_bits in args.replacement_signal_bits:
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_label = f"{base_label}_sig{cap_token(float(signal_bits))}b"
                        export_one(selected_tensor, args.output_root / signal_label / f"q{q}", image_name)
                        bpps[(signal_label, q)].append(float(selected_bpp + signal_bpp))
                rows.append({
                    "q_index": q,
                    "image": image_name,
                    "base_bpp": base_bpp,
                    "branch_bpp": branch_bpp,
                    "replacement_bpp": replacement_bpp,
                    "soft_gate_mean": soft_gate_mean,
                    "hard_gate_mean": hard_gate_mean,
                    "active_mse_ratio": float(feature_row["active_mse_ratio"]),
                    "index_entropy_mean": float(feature_row["index_entropy_mean"]),
                })
                print(f"[export] q={q} {idx + 1}/{len(eval_paths)} {image_name} base={base_bpp:.6f} repl={replacement_bpp:.6f} soft_gate={soft_gate_mean:.4f} hard_gate={hard_gate_mean:.4f}")
                del prepared, item, base_pad, branch_pad, base, branch, soft_mixed, soft_gate, hard_mixed, hard_gate
                torch.cuda.empty_cache()

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "export_rows.json").open("w") as f:
        json.dump({"args": vars(args), "checkpoint_step": payload.get("step"), "rows": rows}, f, indent=2, default=str)

    if not args.skip_quality:
        for label in dynamic_labels:
            for q in args.q_indexes:
                out_dir = args.output_root / label / f"q{q}"
                evaluate_quality(
                    bpps[(label, q)],
                    input_path=str(args.input_path),
                    output_path=str(out_dir),
                    log_path=str(out_dir),
                    patch_size=args.fid_patch_size,
                )


if __name__ == "__main__":
    main()
