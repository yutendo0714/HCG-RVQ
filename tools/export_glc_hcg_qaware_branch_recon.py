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
    p.add_argument("--context-from-scalar", action="store_true", help="Override/export with scalar autoregressive context and HCG-RVQ final latent correction.")
    p.add_argument("--max-gate", type=float, default=1.0)
    p.add_argument("--controller-hidden", type=int, default=None)
    p.add_argument("--labels", nargs="+", default=["base", "hard_gate"])
    p.add_argument("--replacement-signal-bits", type=float, nargs="*", default=[], help="Optional image-level selection/mode signal costs for deployable selected/fallback rows.")
    p.add_argument("--quantize-soft-gate-bits", type=int, default=0, help="If >0, export replacement_soft with an image-level quantized soft gate and account for transmitting that gate.")
    p.add_argument("--learned-hard-min-q", type=int, nargs="*", default=[], help="Add deployable learned-hard rows that only allow replacement_hard for q >= min_q; lower q stays on base.")
    p.add_argument("--qaware-controller-json", type=Path, default=None, help="Optional E379-style q-aware deployment JSON. Adds q-aware hard replacement rows.")
    p.add_argument("--qaware-policy-modes", nargs="*", default=["q-aware", "global"], help="Policy modes to load from --qaware-controller-json. Use an empty list to load all modes.")
    p.add_argument("--fixed-gate-threshold-feature", default="", help="Feature name for threshold-selected fixed-gate HCG rows, e.g. index_entropy_mean.")
    p.add_argument("--fixed-gate-threshold-op", default=">=", choices=[">=", "<="])
    p.add_argument("--fixed-gate-threshold-values", type=float, nargs="*", default=[], help="Thresholds for --fixed-gate-threshold-feature.")
    p.add_argument("--fixed-gate-secondary-threshold-feature", default="", help="Optional second feature for AND-composed fixed-gate HCG rows.")
    p.add_argument("--fixed-gate-secondary-threshold-op", default=">=", choices=[">=", "<="])
    p.add_argument("--fixed-gate-secondary-threshold-values", type=float, nargs="*", default=[], help="Thresholds for --fixed-gate-secondary-threshold-feature.")
    p.add_argument("--fixed-gate-threshold-gates", type=float, nargs="*", default=[], help="Fixed blend gates for selected rows, e.g. 0.08 0.10 0.12.")
    p.add_argument("--fixed-gate-threshold-accounting", nargs="*", default=["replacement"], choices=["replacement", "honest"], help="Rate accounting for threshold fixed-gate rows. replacement swaps active scalar for RVQ; honest keeps base and sends active RVQ as enhancement.")
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
        context_from_scalar=bool(args.context_from_scalar or payload.get("args", {}).get("context_from_scalar", False)),
    )

    eval_paths = list_images(args.input_path, args.eval_start_index, args.eval_limit)
    if not eval_paths:
        raise SystemExit(f"no eval images in {args.input_path}")

    qaware_specs = load_qaware_specs(args.qaware_controller_json, args.qaware_policy_modes)
    dynamic_labels: list[str] = list(args.labels)
    learned_signal_base_labels = [
        label for label in args.labels
        if label in {"replacement_hard", "hard_gate", "replacement_soft", "soft_gate"}
    ]
    for label in learned_signal_base_labels:
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{label}_sig{cap_token(float(signal_bits))}b")
    if args.quantize_soft_gate_bits > 0:
        base_label = f"replacement_soft_qgate{int(args.quantize_soft_gate_bits)}b"
        dynamic_labels.append(base_label)
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{base_label}_sig{cap_token(float(signal_bits))}b")
        honest_label = f"{base_label}_honestbase"
        dynamic_labels.append(honest_label)
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{honest_label}_sig{cap_token(float(signal_bits))}b")
    for min_q in args.learned_hard_min_q:
        base_label = f"qmin{int(min_q)}_replacement_hard"
        dynamic_labels.append(base_label)
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{base_label}_sig{cap_token(float(signal_bits))}b")
    for spec_row in qaware_specs:
        base_label = f"{spec_row['tag']}_replacement_hard"
        dynamic_labels.append(base_label)
        for signal_bits in args.replacement_signal_bits:
            dynamic_labels.append(f"{base_label}_sig{cap_token(float(signal_bits))}b")
    threshold_policy_labels: list[tuple[str, float, float | None, float, str]] = []
    if args.fixed_gate_threshold_feature and args.fixed_gate_threshold_values and args.fixed_gate_threshold_gates:
        feature_tag = args.fixed_gate_threshold_feature.replace("_", "")
        op_tag = "ge" if args.fixed_gate_threshold_op == ">=" else "le"
        secondary_values = list(args.fixed_gate_secondary_threshold_values)
        secondary_feature_tag = args.fixed_gate_secondary_threshold_feature.replace("_", "")
        secondary_op_tag = "ge" if args.fixed_gate_secondary_threshold_op == ">=" else "le"
        if args.fixed_gate_secondary_threshold_feature and not secondary_values:
            raise SystemExit("--fixed-gate-secondary-threshold-feature requires --fixed-gate-secondary-threshold-values")
        secondary_thresholds: list[float | None] = [None]
        if args.fixed_gate_secondary_threshold_feature:
            secondary_thresholds = [float(v) for v in secondary_values]
        for threshold in args.fixed_gate_threshold_values:
            for secondary_threshold in secondary_thresholds:
                for gate_value in args.fixed_gate_threshold_gates:
                    for accounting in args.fixed_gate_threshold_accounting:
                        base_label = f"th_{feature_tag}_{op_tag}{cap_token(float(threshold))}"
                        if secondary_threshold is not None:
                            base_label += (
                                f"_{secondary_feature_tag}_{secondary_op_tag}"
                                f"{cap_token(float(secondary_threshold))}"
                            )
                        base_label += f"_g{cap_token(float(gate_value))}_{accounting}"
                        threshold_policy_labels.append(
                            (base_label, float(threshold), secondary_threshold, float(gate_value), str(accounting))
                        )
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
                quantized_soft_gate_mean = None
                if args.quantize_soft_gate_bits > 0:
                    gate_levels = float((1 << int(args.quantize_soft_gate_bits)) - 1)
                    gate_value = torch.round(soft_gate.mean().clamp(0, args.max_gate) * gate_levels) / gate_levels
                    quantized_soft_gate_mean = float(gate_value.item())
                    quantized_soft_mixed = base + gate_value * (branch - base)
                    qgate_label = f"replacement_soft_qgate{int(args.quantize_soft_gate_bits)}b"
                    tensors[qgate_label] = quantized_soft_mixed
                    label_bpp[qgate_label] = replacement_bpp
                    honest_qgate_label = f"{qgate_label}_honestbase"
                    tensors[honest_qgate_label] = quantized_soft_mixed
                    # Honest soft-gate accounting: reproduce both base and HCG residual path.
                    # The base GLC stream is kept, and active RVQ indices plus the gate signal are added.
                    label_bpp[honest_qgate_label] = base_bpp + float(feature_row["active_rvq_empirical_bpp"])
                for label in args.labels:
                    out_dir = args.output_root / label / f"q{q}"
                    export_one(tensors[label], out_dir, image_name)
                    bpps[(label, q)].append(float(label_bpp[label]))
                    if label in {"replacement_hard", "hard_gate", "replacement_soft", "soft_gate"}:
                        for signal_bits in args.replacement_signal_bits:
                            signal_bpp = image_signal_bpp(signal_bits, item)
                            signal_label = f"{label}_sig{cap_token(float(signal_bits))}b"
                            export_one(tensors[label], args.output_root / signal_label / f"q{q}", image_name)
                            bpps[(signal_label, q)].append(float(label_bpp[label] + signal_bpp))
                if args.quantize_soft_gate_bits > 0:
                    qgate_label = f"replacement_soft_qgate{int(args.quantize_soft_gate_bits)}b"
                    export_one(tensors[qgate_label], args.output_root / qgate_label / f"q{q}", image_name)
                    bpps[(qgate_label, q)].append(float(label_bpp[qgate_label]))
                    for signal_bits in args.replacement_signal_bits:
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_label = f"{qgate_label}_sig{cap_token(float(signal_bits))}b"
                        export_one(tensors[qgate_label], args.output_root / signal_label / f"q{q}", image_name)
                        bpps[(signal_label, q)].append(float(label_bpp[qgate_label] + signal_bpp))
                    honest_qgate_label = f"{qgate_label}_honestbase"
                    export_one(tensors[honest_qgate_label], args.output_root / honest_qgate_label / f"q{q}", image_name)
                    bpps[(honest_qgate_label, q)].append(float(label_bpp[honest_qgate_label]))
                    for signal_bits in args.replacement_signal_bits:
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_label = f"{honest_qgate_label}_sig{cap_token(float(signal_bits))}b"
                        export_one(tensors[honest_qgate_label], args.output_root / signal_label / f"q{q}", image_name)
                        bpps[(signal_label, q)].append(float(label_bpp[honest_qgate_label] + signal_bpp))
                for min_q in args.learned_hard_min_q:
                    selected = q >= int(min_q) and hard_gate_mean > 0.5
                    selected_tensor = hard_mixed if selected else base
                    selected_bpp = label_bpp["replacement_hard"] if selected else base_bpp
                    base_label = f"qmin{int(min_q)}_replacement_hard"
                    export_one(selected_tensor, args.output_root / base_label / f"q{q}", image_name)
                    bpps[(base_label, q)].append(float(selected_bpp))
                    for signal_bits in args.replacement_signal_bits:
                        signal_bpp = image_signal_bpp(signal_bits, item)
                        signal_label = f"{base_label}_sig{cap_token(float(signal_bits))}b"
                        export_one(selected_tensor, args.output_root / signal_label / f"q{q}", image_name)
                        bpps[(signal_label, q)].append(float(selected_bpp + signal_bpp))
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
                for base_label, threshold, secondary_threshold, gate_value, accounting in threshold_policy_labels:
                    feature_value = float(feature_row.get(args.fixed_gate_threshold_feature, float("nan")))
                    if args.fixed_gate_threshold_op == ">=":
                        selected = bool(torch.isfinite(torch.tensor(feature_value)) and feature_value >= threshold)
                    else:
                        selected = bool(torch.isfinite(torch.tensor(feature_value)) and feature_value <= threshold)
                    if selected and secondary_threshold is not None:
                        secondary_feature_value = float(feature_row.get(args.fixed_gate_secondary_threshold_feature, float("nan")))
                        if args.fixed_gate_secondary_threshold_op == ">=":
                            selected = bool(
                                torch.isfinite(torch.tensor(secondary_feature_value))
                                and secondary_feature_value >= float(secondary_threshold)
                            )
                        else:
                            selected = bool(
                                torch.isfinite(torch.tensor(secondary_feature_value))
                                and secondary_feature_value <= float(secondary_threshold)
                            )
                    fixed_gate = max(0.0, min(float(args.max_gate), float(gate_value)))
                    selected_tensor = base + fixed_gate * (branch - base) if selected else base
                    if selected:
                        if accounting == "honest":
                            selected_bpp = base_bpp + float(feature_row["active_rvq_empirical_bpp"])
                        else:
                            selected_bpp = replacement_bpp
                    else:
                        selected_bpp = base_bpp
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
                    "soft_honest_base_plus_rvq_bpp": base_bpp + float(feature_row["active_rvq_empirical_bpp"]),
                    "soft_gate_mean": soft_gate_mean,
                    "quantized_soft_gate_mean": quantized_soft_gate_mean,
                    "hard_gate_mean": hard_gate_mean,
                    "active_mse_ratio": float(feature_row["active_mse_ratio"]),
                    "active_scalar_bpp": float(feature_row["active_scalar_bpp"]),
                    "active_rvq_empirical_bpp": float(feature_row["active_rvq_empirical_bpp"]),
                    "index_entropy_mean": float(feature_row["index_entropy_mean"]),
                    "index_used_frac_mean": float(feature_row["index_used_frac_mean"]),
                    "index_dead_frac_mean": float(feature_row["index_dead_frac_mean"]),
                })
                print(f"[export] q={q} {idx + 1}/{len(eval_paths)} {image_name} base={base_bpp:.6f} repl={replacement_bpp:.6f} soft_gate={soft_gate_mean:.4f} hard_gate={hard_gate_mean:.4f}")
                del prepared, item, base_pad, branch_pad, base, branch, soft_mixed, soft_gate, hard_mixed, hard_gate
                torch.cuda.empty_cache()

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "export_rows.json").open("w") as f:
        json.dump({"args": vars(args), "checkpoint_step": payload.get("step"), "rows": rows}, f, indent=2, default=str)

    for label in dynamic_labels:
        for q in args.q_indexes:
            values = bpps.get((label, q), [])
            if not values:
                continue
            out_dir = args.output_root / label / f"q{q}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # A lightweight bpp stub lets us decouple expensive image export
            # from GLC's official quality evaluation. evaluate_quality later
            # overwrites this file with the full metric report.
            with (out_dir / "res.txt").open("w") as f:
                f.write(f"bpp = {sum(values) / len(values):.6f}\n")
                f.write(f"num_images = {len(values)}\n")
                f.write("metrics = pending\n")

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
