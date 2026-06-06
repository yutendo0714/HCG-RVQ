#!/usr/bin/env python3
"""Audit latent and RVQ assignment drift against an anchor checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(load_config(config_path)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    model.eval()
    return model


def load_csv_by_path(path: str | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def scalar(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu())
    return float("nan")


def summarize(values: list[float]) -> dict[str, float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"mean": float("nan")}
    n = len(finite)

    def pct(q: float) -> float:
        if n == 1:
            return finite[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return finite[lo] * (1 - frac) + finite[hi] * frac

    mu = mean(finite)
    return {
        "mean": mu,
        "std": mean((v - mu) ** 2 for v in finite) ** 0.5,
        "min": finite[0],
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": finite[-1],
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def index_match(indices: object, anchor_indices: object) -> float:
    if not isinstance(indices, list) or not isinstance(anchor_indices, list) or not indices:
        return float("nan")
    vals = []
    for idx, anchor_idx in zip(indices, anchor_indices):
        vals.append(float((idx == anchor_idx).float().mean().detach().cpu()))
    return mean(vals)


def render_md(report: dict[str, object]) -> str:
    s = report["summary"]
    c = report.get("correlations", {})
    lines = [
        "# Anchor Preservation Drift Audit",
        "",
        f"Config: `{report['config']}`",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Anchor checkpoint: `{report['anchor_checkpoint']}`",
        f"Rows: {report['rows']}",
        "",
        "| metric | mean | p50 | p90 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        "index_match",
        "index_mismatch",
        "y_hat_mse",
        "u_mse",
        "x_hat_mse",
        "rd_delta_vs_beta005",
        "rvq_latent_quant_mse",
        "rvq_dead_code_ratio",
        "rvq_householder_reliability_multiplier",
        "rvq_householder_strength",
    ]:
        if key not in s:
            continue
        d = s[key]
        lines.append(
            f"| {key} | {d.get('mean', float('nan')):.6f} | {d.get('p50', float('nan')):.6f} | {d.get('p90', float('nan')):.6f} | {d.get('max', float('nan')):.6f} |"
        )
    if c:
        lines += [
            "",
            "## Correlations With RD Delta vs Beta005",
            "",
            "| feature | correlation |",
            "|---|---:|",
        ]
        for key, value in c.items():
            lines.append(f"| {key} | {value:.6f} |")
    lines += [
        "",
        "Interpretation: a useful preservation guardrail should keep index_match high and make RD degradation small. Low y_hat MSE with large index mismatch means the continuous anchor does not protect the discrete RVQ assignment regime.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-config", default=None)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=4096)
    parser.add_argument("--start-index", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--rd-csv", default=None)
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--reference-column", default="variant500_rd")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.config, args.checkpoint, device)
    anchor_model = load_model(args.anchor_config or args.config, args.anchor_checkpoint, device)
    rd_rows = load_csv_by_path(args.rd_csv)
    ref_rows = load_csv_by_path(args.reference_csv)

    dataset = ImageFolderDataset(
        [args.data_root],
        patch_size=args.patch_size,
        training=False,
        max_images=args.max_images,
        start_index=args.start_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    rows: list[dict[str, float | str | int]] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="anchor-drift", dynamic_ncols=True)):
            x = pad_to_multiple(batch.to(device, non_blocking=True))
            out = model(x)
            anchor = anchor_model(x)
            path = str(dataset.paths[i])
            row: dict[str, float | str | int] = {
                "index": i,
                "path": path,
                "y_hat_mse": float(F.mse_loss(out["y_hat"], anchor["y_hat"]).detach().cpu()),
                "x_hat_mse": float(F.mse_loss(out["x_hat"], anchor["x_hat"]).detach().cpu()),
                "index_match": index_match(out.get("indices"), anchor.get("indices")),
            }
            row["index_mismatch"] = 1.0 - float(row["index_match"])
            cond = out.get("conditioning_tensors", {})
            acond = anchor.get("conditioning_tensors", {})
            if "u" in cond and "u" in acond:
                row["u_mse"] = float(F.mse_loss(cond["u"], acond["u"]).detach().cpu())
            for key in (
                "latent_quant_mse",
                "dead_code_ratio",
                "perplexity",
                "householder_reliability_multiplier",
                "householder_strength",
                "householder_delta_rms",
            ):
                if key in out.get("rvq_stats", {}):
                    row[f"rvq_{key}"] = scalar(out["rvq_stats"][key])
            if path in rd_rows and path in ref_rows:
                row["rd_delta_vs_beta005"] = float(rd_rows[path]["rd_score"]) - float(ref_rows[path][args.reference_column])
            rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for key in sorted({k for row in rows for k in row if k not in {"path", "index"}}):
        vals = [float(row[key]) for row in rows if key in row]
        summary[key] = summarize(vals)
    correlations = {}
    if "rd_delta_vs_beta005" in summary:
        rd = [float(row.get("rd_delta_vs_beta005", float("nan"))) for row in rows]
        for key in ("index_mismatch", "y_hat_mse", "u_mse", "x_hat_mse", "rvq_latent_quant_mse", "rvq_dead_code_ratio", "rvq_householder_strength"):
            correlations[key] = pearson([float(row.get(key, float("nan"))) for row in rows], rd)

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "anchor_config": args.anchor_config or args.config,
        "anchor_checkpoint": args.anchor_checkpoint,
        "data_root": args.data_root,
        "start_index": args.start_index,
        "max_images": args.max_images,
        "rows": len(rows),
        "summary": summary,
        "correlations": correlations,
    }
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": summary, "correlations": correlations}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
