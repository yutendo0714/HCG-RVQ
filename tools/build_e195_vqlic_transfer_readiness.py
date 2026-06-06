#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "analysis" / "e195_vqlic_transfer_readiness"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def artifact_status(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for rel in paths:
        path = ROOT / rel
        rows.append({"path": rel, "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0})
    return rows


def first_row(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    for row in rows:
        if row.get("selector") == selector:
            return row
    return {}


def summary_row(summary: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    for row in summary:
        if row.get("branch") == branch:
            return row
    return {}


def load_beta_matrix() -> list[dict[str, str]]:
    path = ROOT / "experiments" / "analysis" / "beta005_paper_claim_matrix.md"
    if not path.exists():
        return []
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("| split |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 8:
                rows.append({
                    "split": parts[0],
                    "images_per_seed": parts[1],
                    "beta_minus_hcs": parts[2],
                    "beta_minus_old": parts[3],
                    "beta_minus_min090": parts[4],
                    "s_q": parts[5],
                    "delta_rms": parts[6],
                    "q_mse": parts[7],
                })
            else:
                break
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def metric_line(name: str, row: dict[str, Any]) -> str:
    if not row:
        return f"| {name} | missing | | | | | | |"
    selector = row.get("selector", row.get("branch", ""))
    branch_share = row.get("branch_share", "")
    d_dists = row.get("selected_delta_dists", row.get("delta_dists_vs_base", ""))
    d_lpips = row.get("selected_delta_lpips", row.get("delta_lpips_vs_base", ""))
    d_psnr = row.get("selected_delta_psnr", row.get("delta_psnr_vs_base", ""))
    bpp = row.get("selected_bpp", row.get("bpp", ""))
    images = row.get("images", "")
    return f"| {name} | {selector} | {fmt(branch_share)} | {fmt(d_dists)} | {fmt(d_lpips)} | {fmt(d_psnr)} | {fmt(bpp)} | {images} |"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data_dirs = [
        "experiments/data/kodak24",
        "experiments/data/kodak_first4",
        "experiments/data/eflic_selector_fit",
        "experiments/data/eflic_selector_eval",
    ]
    data_inventory = [{"path": d, "images": count_images(ROOT / d), "exists": (ROOT / d).exists()} for d in data_dirs]

    e190 = read_json(ROOT / "experiments" / "analysis" / "e190_eflic_force0_global_selector_multiobj_d1_l3.json")
    e193 = read_json(ROOT / "experiments" / "analysis" / "e193_eflic_force0_global_reliability_head_d1_l3.json")
    e194 = read_json(ROOT / "experiments" / "analysis" / "e194_eflic_reliability_head_selector_kodak24_selfcheck_d1_l3.json")
    e194_smoke = read_json(ROOT / "experiments" / "analysis" / "e194_eflic_reliability_head_selector_kodak_first4_smoke_d1_l3.json")

    e190_primary = first_row(e190.get("rows", []), "best_multiobj_threshold")
    e190_loo = first_row(e190.get("rows", []), "loocv_multiobj_threshold")
    e193_head = first_row(e193.get("rows", []), "reliability_head")
    e193_loo = first_row(e193.get("rows", []), "loocv_reliability_head")
    e194_selected = summary_row(e194.get("summary", []), "selected")
    e194_active = summary_row(e194.get("summary", []), "active")
    e194_smoke_selected = summary_row(e194_smoke.get("summary", []), "selected")

    artifacts = artifact_status([
        "third_party/EF-LIC/EF_LIC.py",
        "third_party/EF-LIC/ckpt/checkpoint.pth.tar",
        "third_party/GLC/test_image.py",
        "third_party/GLC/src/models/image_model.py",
        "tools/run_e160_eflic_projected_hcg_smoke.py",
        "tools/run_e186_eflic_global_predecision_selector_probe.py",
        "tools/analyze_e190_eflic_multiobjective_selector.py",
        "tools/analyze_e193_eflic_reliability_head.py",
        "tools/run_e194_eflic_reliability_head_selector_probe.py",
        "experiments/analysis/e168_glc_y_res_distribution_kodak24.md",
        "experiments/analysis/e172_glc_tail_vq_rvq_design_decision.md",
        "experiments/analysis/e183_glc_decoder_aware_tail_vq_split_train_q0_oi16_kodak8_distsonly.md",
        "experiments/analysis/beta005_paper_claim_matrix.md",
    ])

    payload: dict[str, Any] = {
        "experiment": "E195 VQ-LIC transfer readiness package",
        "data_inventory": data_inventory,
        "prototype_beta005_matrix": load_beta_matrix(),
        "eflic": {
            "e190_primary": e190_primary,
            "e190_loocv": e190_loo,
            "e193_same_table_head": e193_head,
            "e193_loocv_head": e193_loo,
            "e194_kodak24_selected": e194_selected,
            "e194_kodak24_active": e194_active,
            "e194_first4_selected": e194_smoke_selected,
            "current_default": "E190 scalar multi-objective selector until independent fit/eval promotes E194 learned head",
        },
        "glc": {
            "current_default": "sparse active residual subset with scalar fallback, local codebook or HCG geometry, bit-aware index prior, decoder-aware training",
            "avoid": "dense always-on VQ replacement and shared active codebook",
        },
        "blockers": [
            "Only Kodak image directories are present in experiments/data, so independent EF-LIC fit/eval cannot be run from local images yet.",
            "EF-LIC official checkout is inference oriented in this workspace, so full retraining needs official training code reconstruction or an external training protocol.",
            "GLC active branch still needs bit-aware and reliability-controlled training before paper-facing rate claims.",
        ],
        "artifacts": artifacts,
    }

    json_path = OUT.with_suffix(".json")
    csv_path = OUT.with_suffix(".artifacts.csv")
    md_path = OUT.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "exists", "size"])
        writer.writeheader()
        writer.writerows(artifacts)

    lines = [
        "# E195 VQ-LIC Transfer Readiness Package",
        "",
        "## Dataset Inventory",
        "",
        "| path | exists | images |",
        "|---|---:|---:|",
    ]
    for row in data_inventory:
        lines.append(f"| {row['path']} | {int(row['exists'])} | {row['images']} |")

    lines.extend([
        "",
        "## Prototype HCG-RVQ Evidence To Preserve",
        "",
        "| split | images/seed | beta-HCS | beta-old | beta-min090 | s_q | delta RMS | qMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["prototype_beta005_matrix"]:
        lines.append(
            f"| {row['split']} | {row['images_per_seed']} | {row['beta_minus_hcs']} | {row['beta_minus_old']} | {row['beta_minus_min090']} | {row['s_q']} | {row['delta_rms']} | {row['q_mse']} |"
        )

    lines.extend([
        "",
        "## EF-LIC Current Controller Evidence",
        "",
        "| evidence | selector/branch | branch share | dDISTS | dLPIPS | dPSNR | bpp | images |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        metric_line("E190 same-table primary", e190_primary),
        metric_line("E190 LOOCV primary", e190_loo),
        metric_line("E193 same-table learned head", e193_head),
        metric_line("E193 LOOCV learned head", e193_loo),
        metric_line("E194 direct Kodak24 selected", e194_selected),
        metric_line("E194 direct first4 selected", e194_smoke_selected),
        "",
        "Decision: keep E190 as the immediate paper-facing controlled selector. Use E194 as the direct implementation path for a learned head only after independent non-Kodak fit labels exist.",
        "",
        "## GLC Current Design Decision",
        "",
        "Use sparse active residual states with scalar fallback. Shared active codebooks and dense always-on VQ are rejected by current diagnostics. The next GLC branch must combine local HCG geometry, index-rate accounting, and decoder-aware training.",
        "",
        "## Blocking Items",
        "",
    ])
    for item in payload["blockers"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Next Commands Once An Independent Fit Split Exists",
        "",
        "```bash",
        "env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \\",
        "  --device cuda:0 \\",
        "  --kodak-dir experiments/data/eflic_selector_fit \\",
        "  --force-ind 0 --alpha 0.05 --direction-source mean \\",
        "  --output-prefix experiments/analysis/e195_eflic_fit_active_labels",
        "",
        ".venv/bin/python tools/build_e161_eflic_projected_hcg_selector_labels.py \\",
        "  --input-csv experiments/analysis/e195_eflic_fit_active_labels.csv \\",
        "  --output-prefix experiments/analysis/e195_eflic_fit_labels",
        "",
        ".venv/bin/python tools/analyze_e190_eflic_multiobjective_selector.py \\",
        "  --input-csv experiments/analysis/e195_eflic_fit_labels.csv \\",
        "  --output-prefix experiments/analysis/e195_eflic_fit_rule_multiobj_d1_l3 \\",
        "  --force 0 --feature-set global_predecision_context \\",
        "  --dists-weight 1.0 --lpips-weight 3.0 --positive-penalty 20.0",
        "",
        "env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e194_eflic_reliability_head_selector_probe.py \\",
        "  --device cuda:0 \\",
        "  --fit-csv experiments/analysis/e195_eflic_fit_labels.csv \\",
        "  --fit-manifest-csv experiments/analysis/e195_eflic_fit_labels_feature_manifest.csv \\",
        "  --eval-dir experiments/data/eflic_selector_eval \\",
        "  --output-prefix experiments/analysis/e195_eflic_reliability_head_heldout_eval_d1_l3",
        "```",
        "",
        "## Artifact Check",
        "",
        "| path | exists | size |",
        "|---|---:|---:|",
    ])
    for row in artifacts:
        lines.append(f"| {row['path']} | {int(row['exists'])} | {row['size']} |")

    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


if __name__ == "__main__":
    main()
