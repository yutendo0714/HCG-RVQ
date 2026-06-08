#!/usr/bin/env python3
"""Audit whether old EF-LIC spatial teacher maps align with E317/E318 oracle.

E242 exported decoder-available spatial context tensors and map-level teacher
targets from an older local-policy pipeline. E317/E318 later produced a stronger
Kodak24 slice-powerset oracle with exact fallback. This diagnostic checks
whether E242 is still a useful supervision source before spending GPU on larger
controller training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ALL_SLICES = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/analysis/e242_eflic_spatial_teacher_contexts_kodak24/manifest_kodak24_n24.csv",
    )
    p.add_argument(
        "--slice-labels",
        type=Path,
        default=ROOT / "experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.slice_labels.csv",
    )
    p.add_argument(
        "--powerset-rows",
        type=Path,
        default=ROOT / "experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.rows.csv",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments/analysis/e323_eflic_e242_vs_e318_teacher_alignment_kodak24",
    )
    p.add_argument(
        "--thresholds",
        default="0,0.001,0.01,0.05,0.10,0.25,0.50",
        help="Comma-separated active-fraction thresholds. A slice is active if frac > threshold.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def label_for_slices(slices: set[int]) -> str:
    if not slices:
        return "none"
    if slices == set(ALL_SLICES):
        return "all"
    return ",".join(str(s) for s in sorted(slices))


def build_slice_label_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        out[(row["image"], int(row["slice"]))] = row
    return out


def build_delta_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        out[(row["image"], row["active_slices"])] = fnum(row, "delta_psnr")
    return out


def load_e242_slice_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in manifest_rows:
        tensor_path = Path(record["tensor_path"])
        if not tensor_path.is_absolute():
            tensor_path = ROOT / tensor_path
        obj = torch.load(tensor_path, map_location="cpu")
        target_map = obj["target_map"].long()
        alpha = obj["alpha_target"].float()
        if target_map.ndim != 3 or target_map.shape[0] != 4:
            raise RuntimeError(f"unexpected target_map shape {tuple(target_map.shape)} in {tensor_path}")
        if alpha.ndim != 4 or alpha.shape[0] != 4:
            raise RuntimeError(f"unexpected alpha shape {tuple(alpha.shape)} in {tensor_path}")
        for slice_id in ALL_SLICES:
            target_slice = target_map[slice_id]
            alpha_slice = alpha[slice_id]
            active = target_slice > 0
            out.append(
                {
                    "dataset": record.get("dataset", obj.get("dataset", "")),
                    "image": record["image"],
                    "slice": slice_id,
                    "tensor_path": str(tensor_path),
                    "e242_target_family": record.get("target_family", obj.get("target_family", "")),
                    "e242_teacher_policy": record.get("teacher_policy", obj.get("teacher_policy", "")),
                    "e242_image_active_frac": fnum(record, "active_frac"),
                    "e242_slice_active_frac": float(active.float().mean().item()),
                    "e242_slice_alpha_mean": float(alpha_slice.mean().item()),
                    "e242_slice_alpha_max": float(alpha_slice.max().item()),
                    "e242_slice_alpha_active_mean": float(alpha_slice[active.unsqueeze(0)].mean().item()) if bool(active.any().item()) else 0.0,
                    "latent_height": int(target_slice.shape[-2]),
                    "latent_width": int(target_slice.shape[-1]),
                    "nonzero_pixels": int(active.sum().item()),
                }
            )
    return out


def confusion(slice_rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in slice_rows:
        pred = float(row["e242_slice_active_frac"]) > threshold
        target = int(row["e318_oracle_active"]) == 1
        if pred and target:
            tp += 1
        elif pred and not target:
            fp += 1
        elif not pred and target:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "threshold": threshold,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": accuracy,
        "pred_active_frac": mean([1.0 if float(row["e242_slice_active_frac"]) > threshold else 0.0 for row in slice_rows]),
        "oracle_active_frac": mean([float(row["e318_oracle_active"]) for row in slice_rows]),
    }


def score_threshold(
    images: list[str],
    slice_rows_by_image: dict[str, list[dict[str, Any]]],
    deltas: dict[tuple[str, str], float],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    image_rows: list[dict[str, Any]] = []
    for image in images:
        active_slices = {int(row["slice"]) for row in slice_rows_by_image[image] if float(row["e242_slice_active_frac"]) > threshold}
        predicted = label_for_slices(active_slices)
        delta = deltas[(image, predicted)]
        all_delta = deltas[(image, "all")]
        none_delta = deltas[(image, "none")]
        full_oracle = max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image})
        image_rows.append(
            {
                "threshold": threshold,
                "image": image,
                "predicted_slice_set": predicted,
                "predicted_delta_psnr": delta,
                "all_delta_psnr": all_delta,
                "none_delta_psnr": none_delta,
                "full_oracle_delta_psnr": full_oracle,
                "gain_over_all": delta - all_delta,
                "gap_to_full_oracle": full_oracle - delta,
                "predicted_active_slice_count": len(active_slices),
            }
        )
    summary = {
        "threshold": threshold,
        "policy": f"e242_active_frac_gt_{threshold:g}",
        "images": float(len(images)),
        "mean_delta_psnr": mean([float(row["predicted_delta_psnr"]) for row in image_rows]),
        "worst_delta_psnr": min(float(row["predicted_delta_psnr"]) for row in image_rows),
        "mean_gain_over_all": mean([float(row["gain_over_all"]) for row in image_rows]),
        "mean_gap_to_full_oracle": mean([float(row["gap_to_full_oracle"]) for row in image_rows]),
        "mean_active_slice_count": mean([float(row["predicted_active_slice_count"]) for row in image_rows]),
    }
    return summary, image_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    manifest_rows = read_csv(args.manifest)
    e318_rows = read_csv(args.slice_labels)
    powerset_rows = read_csv(args.powerset_rows)
    e318 = build_slice_label_lookup(e318_rows)
    deltas = build_delta_lookup(powerset_rows)
    e242_rows = load_e242_slice_rows(manifest_rows)

    joined: list[dict[str, Any]] = []
    for row in e242_rows:
        key = (row["image"], int(row["slice"]))
        if key not in e318:
            raise RuntimeError(f"missing E318 row for {key}")
        label = e318[key]
        joined.append(
            row
            | {
                "e318_oracle_active": int(float(label["oracle_active"])),
                "e318_single_delta_psnr": fnum(label, "single_delta_psnr"),
                "e318_all_delta_psnr": fnum(label, "all_delta_psnr"),
                "e318_best_delta_psnr": fnum(label, "best_delta_psnr"),
                "e318_best_slice_set": label["best_slice_set"],
                "e318_contextual_positive": int(float(label["contextual_positive"])),
                "e318_single_positive": int(float(label["single_positive"])),
            }
        )

    images = sorted({row["image"] for row in joined})
    by_image: dict[str, list[dict[str, Any]]] = {image: [] for image in images}
    for row in joined:
        by_image[row["image"]].append(row)

    threshold_rows = [confusion(joined, threshold) for threshold in thresholds]
    image_policy_rows: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, float]] = [
        {
            "threshold": float("nan"),
            "policy": "all",
            "images": float(len(images)),
            "mean_delta_psnr": mean([deltas[(image, "all")] for image in images]),
            "worst_delta_psnr": min(deltas[(image, "all")] for image in images),
            "mean_gain_over_all": 0.0,
            "mean_gap_to_full_oracle": mean([max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image}) - deltas[(image, "all")] for image in images]),
            "mean_active_slice_count": 4.0,
        },
        {
            "threshold": float("nan"),
            "policy": "none",
            "images": float(len(images)),
            "mean_delta_psnr": 0.0,
            "worst_delta_psnr": 0.0,
            "mean_gain_over_all": mean([-deltas[(image, "all")] for image in images]),
            "mean_gap_to_full_oracle": mean([max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image}) for image in images]),
            "mean_active_slice_count": 0.0,
        },
        {
            "threshold": float("nan"),
            "policy": "full_subset_oracle",
            "images": float(len(images)),
            "mean_delta_psnr": mean([max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image}) for image in images]),
            "worst_delta_psnr": min(max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image}) for image in images),
            "mean_gain_over_all": mean([max(deltas[(image, label)] for label in {k[1] for k in deltas if k[0] == image}) - deltas[(image, "all")] for image in images]),
            "mean_gap_to_full_oracle": 0.0,
            "mean_active_slice_count": float("nan"),
        },
    ]
    for threshold in thresholds:
        summary, rows = score_threshold(images, by_image, deltas, threshold)
        policy_summaries.append(summary)
        image_policy_rows.extend(rows)

    active_fracs = [float(row["e242_slice_active_frac"]) for row in joined]
    oracle = [float(row["e318_oracle_active"]) for row in joined]
    single_delta = [float(row["e318_single_delta_psnr"]) for row in joined]
    summary_payload = {
        "experiment": "E323 EF-LIC E242 spatial teacher vs E318 powerset-oracle alignment",
        "purpose": "Check whether old E242 spatial-map supervision is aligned with the latest E317/E318 fallback-aware oracle before larger controller training.",
        "manifest": str(args.manifest),
        "slice_labels": str(args.slice_labels),
        "powerset_rows": str(args.powerset_rows),
        "images": len(images),
        "slices": len(joined),
        "e242_active_frac_mean": mean(active_fracs),
        "e318_oracle_active_frac": mean(oracle),
        "corr_e242_active_frac_vs_oracle_active": corr(active_fracs, oracle),
        "corr_e242_active_frac_vs_single_delta_psnr": corr(active_fracs, single_delta),
        "confusion_by_threshold": threshold_rows,
        "policy_summary": policy_summaries,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".slice_alignment.csv"), joined)
    write_csv(args.output_prefix.with_suffix(".threshold_alignment.csv"), threshold_rows)
    write_csv(args.output_prefix.with_suffix(".policy_summary.csv"), policy_summaries)
    write_csv(args.output_prefix.with_suffix(".image_policy.csv"), image_policy_rows)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    best_e242 = max(
        [row for row in policy_summaries if str(row["policy"]).startswith("e242_active_frac_gt_")],
        key=lambda row: (float(row["mean_delta_psnr"]), float(row["worst_delta_psnr"])),
    )
    best_threshold_conf = next(row for row in threshold_rows if float(row["threshold"]) == float(best_e242["threshold"]))
    with args.output_prefix.with_suffix(".md").open("w", encoding="utf-8") as fobj:
        fobj.write("# E323 EF-LIC E242 vs E318 Teacher Alignment\n\n")
        fobj.write("This audit checks whether the old E242 spatial teacher maps are aligned with the latest E317/E318 fallback-aware slice oracle.\n\n")
        fobj.write("## Summary\n\n")
        fobj.write(f"- Images: `{len(images)}`\n")
        fobj.write(f"- Slice rows: `{len(joined)}`\n")
        fobj.write(f"- E242 mean active fraction: `{summary_payload['e242_active_frac_mean']:.6f}`\n")
        fobj.write(f"- E318 oracle active fraction: `{summary_payload['e318_oracle_active_frac']:.6f}`\n")
        fobj.write(f"- Corr(E242 active frac, E318 oracle active): `{summary_payload['corr_e242_active_frac_vs_oracle_active']:.6f}`\n")
        fobj.write(f"- Corr(E242 active frac, single-slice delta PSNR): `{summary_payload['corr_e242_active_frac_vs_single_delta_psnr']:.6f}`\n\n")
        fobj.write("## Best E242 Threshold Policy\n\n")
        fobj.write(f"- Policy: `{best_e242['policy']}`\n")
        fobj.write(f"- Mean delta PSNR: `{best_e242['mean_delta_psnr']:.6f}`\n")
        fobj.write(f"- Worst delta PSNR: `{best_e242['worst_delta_psnr']:.6f}`\n")
        fobj.write(f"- Mean gain over all-on: `{best_e242['mean_gain_over_all']:.6f}`\n")
        fobj.write(f"- Precision / recall / specificity: `{best_threshold_conf['precision']:.6f}` / `{best_threshold_conf['recall']:.6f}` / `{best_threshold_conf['specificity']:.6f}`\n\n")
        fobj.write("## Interpretation\n\n")
        fobj.write("- If the E242 threshold policies underperform all-on or show weak oracle alignment, E242 should remain a smoke/trainability artifact rather than the main supervision source.\n")
        fobj.write("- The next paper-relevant controller should use decoder-available spatial/sequential context, but its labels should be regenerated from the E317/E318 fallback-aware oracle or a stronger local proxy.\n")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
