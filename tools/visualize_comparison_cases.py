from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.metrics import compute_msssim, compute_psnr
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


def load_eval_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[dict, torch.nn.Module]:
    config = load_config(config_path)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return config, model


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    array = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def normalize_map(x: torch.Tensor, vmax: float | None = None) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.abs().mean(dim=0)
    x = x - x.min()
    denom = float(vmax) if vmax is not None else float(x.max().clamp_min(1e-12))
    return (x / max(denom, 1e-12)).clamp(0, 1)


def heatmap_image(x: torch.Tensor, size: tuple[int, int] | None = None, vmax: float | None = None) -> Image.Image:
    m = normalize_map(x, vmax=vmax)
    red = m
    green = (1.0 - (m - 0.35).abs() / 0.35).clamp(0, 1) * 0.85
    blue = (1.0 - m).pow(1.7) * 0.9
    rgb = torch.stack([red, green, blue], dim=-1)
    img = Image.fromarray((rgb.numpy() * 255.0).round().astype("uint8"), mode="RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def signed_heatmap_image(x: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    m = x.detach().float().cpu()
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m.mean(dim=0)
    scale = float(m.abs().max().clamp_min(1e-12))
    pos = (m / scale).clamp(0, 1)
    neg = (-m / scale).clamp(0, 1)
    base = torch.ones_like(pos) * 0.96
    rgb = torch.stack([base - neg * 0.75, base - (pos + neg) * 0.55, base - pos * 0.75], dim=-1).clamp(0, 1)
    img = Image.fromarray((rgb.numpy() * 255.0).round().astype("uint8"), mode="RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def add_label(img: Image.Image, label: str) -> Image.Image:
    pad = 22
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return out


def make_grid(images: list[tuple[str, Image.Image]], columns: int = 4) -> Image.Image:
    labeled = [add_label(img, label) for label, img in images]
    w = max(img.width for img in labeled)
    h = max(img.height for img in labeled)
    rows = (len(labeled) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * w, rows * h), "white")
    for i, img in enumerate(labeled):
        canvas.paste(img, ((i % columns) * w, (i // columns) * h))
    return canvas


def capture_decoder_features(model: torch.nn.Module, y_hat: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    features: OrderedDict[str, torch.Tensor] = OrderedDict()
    hooks = []
    for idx, module in enumerate(model.g_s):
        hooks.append(module.register_forward_hook(lambda _module, _inputs, output, idx=idx: features.__setitem__(f"g_s_{idx}", output.detach())))
    try:
        _ = model.g_s(y_hat)
    finally:
        for hook in hooks:
            hook.remove()
    return features


def evaluate_model(
    model: torch.nn.Module,
    criterion: RateDistortionLoss,
    config: dict,
    x: torch.Tensor,
) -> tuple[dict[str, object], dict[str, float], OrderedDict[str, torch.Tensor]]:
    x_pad, hw = pad_to_multiple(x)
    output = model(x_pad)
    output["x_hat"] = crop_to_hw(output["x_hat"], hw)
    losses = criterion(output, x)
    mse = float(losses["mse"].cpu())
    bpp = float(losses["bpp_total"].cpu())
    lambda_rd = float(config.get("loss", {})["lambda_rd"])
    mse_scale = float(config.get("loss", {}).get("mse_scale", 255.0 * 255.0))
    metrics = {
        "loss": float(losses["loss"].cpu()),
        "rd_score": bpp + lambda_rd * mse_scale * mse,
        "bpp": bpp,
        "bpp_y": float(losses["bpp_y"].cpu()),
        "bpp_z": float(losses["bpp_z"].cpu()),
        "mse": mse,
        "psnr": float(compute_psnr(x, output["x_hat"]).cpu()),
        "ms_ssim": float(compute_msssim(x, output["x_hat"]).cpu()),
    }
    decoder_features = capture_decoder_features(model, output["y_hat"])
    return output, metrics, decoder_features


def conditioning_maps(output: dict[str, object], target_size: tuple[int, int]) -> list[tuple[str, Image.Image]]:
    maps: list[tuple[str, Image.Image]] = []
    conditioning = output.get("conditioning_tensors", {})
    rvq_stats = output.get("rvq_stats", {})
    if isinstance(conditioning, dict):
        if "s_q" in conditioning:
            maps.append(("HCG scale", heatmap_image(conditioning["s_q"], target_size)))
        if "mu_q" in conditioning:
            maps.append(("HCG |mu|", heatmap_image(conditioning["mu_q"], target_size)))
        if "y_norm" in conditioning and "u" in conditioning:
            maps.append(("HCG H-delta", heatmap_image(conditioning["u"] - conditioning["y_norm"], target_size)))
    if isinstance(rvq_stats, dict) and "householder_delta_rms" in rvq_stats:
        value = float(rvq_stats["householder_delta_rms"].detach().cpu())
        tiny = Image.new("RGB", target_size, "white")
        draw = ImageDraw.Draw(tiny)
        draw.text((8, 8), f"H delta RMS\n{value:.6f}", fill=(0, 0, 0))
        maps.append(("H stats", tiny))
    return maps


def select_cases(per_image_csv: Path, top_k: int) -> list[dict[str, str]]:
    with per_image_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    losses = sorted(rows, key=lambda row: float(row["delta_rd_score"]), reverse=True)[:top_k]
    gains = sorted(rows, key=lambda row: float(row["delta_rd_score"]))[:top_k]
    selected: list[dict[str, str]] = []
    for kind, case_rows in [("loss", losses), ("gain", gains)]:
        for rank, row in enumerate(case_rows, start=1):
            selected.append({"case_kind": kind, "case_rank": str(rank), **row})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Save visual/error-map panels for image-level checkpoint comparison cases.")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--a-config", required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--b-config", required=True)
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--per-image-csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config_a, model_a = load_eval_model(args.a_config, args.a_checkpoint, device)
    config_b, model_b = load_eval_model(args.b_config, args.b_checkpoint, device)
    criterion_a = RateDistortionLoss(**config_a.get("loss", {}))
    criterion_b = RateDistortionLoss(**config_b.get("loss", {}))

    dataset = ImageFolderDataset(
        [args.data_root],
        patch_size=args.patch_size,
        training=False,
        max_images=args.max_images,
        start_index=args.start_index,
    )
    cases = select_cases(Path(args.per_image_csv), args.top_k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    with torch.no_grad():
        for case in cases:
            index = int(float(case["index"]))
            x = dataset[index].unsqueeze(0).to(device)
            out_a, metrics_a, feats_a = evaluate_model(model_a, criterion_a, config_a, x)
            out_b, metrics_b, feats_b = evaluate_model(model_b, criterion_b, config_b, x)

            size = (int(x.shape[-1]), int(x.shape[-2]))
            err_a = (out_a["x_hat"].clamp(0, 1) - x).abs()
            err_b = (out_b["x_hat"].clamp(0, 1) - x).abs()
            err_vmax = max(float(err_a.max().cpu()), float(err_b.max().cpu()), 1e-6)
            yerr_a = out_a["y_hat"] - out_a["y"]
            yerr_b = out_b["y_hat"] - out_b["y"]
            yerr_vmax = max(float(yerr_a.abs().mean(dim=1).max().cpu()), float(yerr_b.abs().mean(dim=1).max().cpu()), 1e-6)

            selected_layers = ["g_s_0", "g_s_2", "g_s_4", "g_s_6"]
            decoder_maps: list[tuple[str, Image.Image]] = []
            decoder_delta_means: dict[str, float] = {}
            for layer in selected_layers:
                if layer not in feats_a or layer not in feats_b:
                    continue
                diff = feats_b[layer] - feats_a[layer]
                decoder_delta_means[layer] = float(diff.abs().mean().cpu())
                decoder_maps.append((f"dec {layer}", heatmap_image(diff, size)))

            images: list[tuple[str, Image.Image]] = [
                ("input", tensor_to_image(x)),
                (args.a_name, tensor_to_image(out_a["x_hat"])),
                (args.b_name, tensor_to_image(out_b["x_hat"])),
                ("B-A signed", signed_heatmap_image(out_b["x_hat"] - out_a["x_hat"], size)),
                (f"{args.a_name} |err|", heatmap_image(err_a, size, vmax=err_vmax)),
                (f"{args.b_name} |err|", heatmap_image(err_b, size, vmax=err_vmax)),
                ("err B-A", signed_heatmap_image(err_b - err_a, size)),
                (f"{args.a_name} yerr", heatmap_image(yerr_a, size, vmax=yerr_vmax)),
                (f"{args.b_name} yerr", heatmap_image(yerr_b, size, vmax=yerr_vmax)),
            ]
            images.extend(conditioning_maps(out_b, size))
            images.extend(decoder_maps)

            stem = f"{case['case_kind']}_{int(case['case_rank']):02d}_idx{index}_{Path(case['path']).stem}"
            panel_path = output_dir / f"{stem}.png"
            make_grid(images, columns=4).save(panel_path)

            row = {
                "case_kind": case["case_kind"],
                "case_rank": int(case["case_rank"]),
                "index": index,
                "path": case["path"],
                "panel": str(panel_path),
                "delta_rd_score_csv": float(case["delta_rd_score"]),
                "delta_bpp_csv": float(case["delta_bpp"]),
                "delta_psnr_csv": float(case["delta_psnr"]),
                "delta_ms_ssim_csv": float(case["delta_ms_ssim"]),
                "delta_rd_score": metrics_b["rd_score"] - metrics_a["rd_score"],
                "delta_bpp": metrics_b["bpp"] - metrics_a["bpp"],
                "delta_psnr": metrics_b["psnr"] - metrics_a["psnr"],
                "delta_ms_ssim": metrics_b["ms_ssim"] - metrics_a["ms_ssim"],
                **{f"decoder_delta_abs_{k}": v for k, v in decoder_delta_means.items()},
            }
            summary.append(row)
            print(f"{case['case_kind']} #{case['case_rank']}: {panel_path}")

    summary_csv = output_dir / "case_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = list(summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
