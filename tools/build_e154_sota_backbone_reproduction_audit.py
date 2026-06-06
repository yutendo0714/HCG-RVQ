#!/usr/bin/env python3
"""Build a reproducibility audit for SOTA/backbone plug-in experiments.

This is a provenance artifact, not a compression evaluation.  It records
which third-party repositories are executable, which pretrained checkpoints
are available, and where HCG-RVQ should attach without mixing backbone
effects with the proposed quantizer-geometry mechanism.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PREFIX = ROOT / "experiments" / "analysis" / "e154_sota_backbone_reproduction_audit"


REPOS = [
    {
        "name": "DCAE",
        "path": "third_party/DCAE",
        "paper": "Learned Image Compression with Dictionary-based Entropy Model",
        "priority": "first plug-in target",
        "boundary": "g_a/h_a -> latent_scales,latent_means -> y_hat -> g_s",
        "risk": "Google Drive checkpoints; first compare same backbone over itself",
    },
    {
        "name": "DCAE_LabShuHangGU",
        "path": "third_party/DCAE_LabShuHangGU",
        "paper": "Official clone target named in DCAE README",
        "priority": "provenance mirror",
        "boundary": "same as DCAE if commit matches",
        "risk": "Keep only as provenance check unless it diverges",
    },
    {
        "name": "MambaIC",
        "path": "third_party/MambaIC",
        "paper": "MambaIC: State Space Models for High-Performance Learned Image Compression",
        "priority": "second plug-in target",
        "boundary": "CompressAI-style y slices plus SSM/context entropy path",
        "risk": "VMamba/selective_scan dependency and reimplementation note",
    },
    {
        "name": "LIC-HPCM",
        "path": "third_party/LIC-HPCM",
        "paper": "Learned Image Compression with Hierarchical Progressive Context Modeling",
        "priority": "comparison/design reference before plug-in",
        "boundary": "progressive hpcm compress/decompress path",
        "risk": "Deep progressive codec path; use after DCAE boundary is stable",
    },
    {
        "name": "RDVQ",
        "path": "third_party/RDVQ",
        "paper": "Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression",
        "priority": "paper-facing VQ comparison",
        "boundary": "not executable yet",
        "risk": "README currently says code will come soon",
    },
]


def run_git(path: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance best effort
        return f"ERROR: {exc}"


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def extract_drive_links(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if "drive.google.com" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        links = re.findall(r"https://drive\.google\.com/file/d/([^/]+)/view[^)\s]*", line)
        numeric = [cell for cell in cells if re.fullmatch(r"\d+(?:\.\d+)?", cell)]
        metrics = [cell for cell in cells if cell in {"MSE", "MS-SSIM"}]
        for idx, file_id in enumerate(links):
            rows.append(
                {
                    "lambda_or_quality": numeric[idx] if idx < len(numeric) else "",
                    "metric": metrics[idx] if idx < len(metrics) else "",
                    "file_id": file_id,
                    "source_line": line.strip(),
                }
            )
    return rows


def repo_snapshot(repo: dict[str, str]) -> dict[str, str | int]:
    path = ROOT / repo["path"]
    readme = read(path / "README.md")
    remote = run_git(path, ["remote", "-v"])
    commit = run_git(path, ["rev-parse", "HEAD"])
    checkpoint_count = len(extract_drive_links(readme))
    has_eval = (path / "eval.py").exists() or (path / "test.py").exists()
    return {
        **repo,
        "remote": remote.replace("\n", " | "),
        "commit": commit,
        "has_eval_or_test": str(has_eval),
        "google_drive_checkpoint_links": checkpoint_count,
    }


def main() -> None:
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)

    repos = [repo_snapshot(repo) for repo in REPOS]
    dcae_match = repos[0]["commit"] == repos[1]["commit"]

    checkpoint_rows: list[dict[str, str]] = []
    for repo in REPOS:
        path = ROOT / repo["path"]
        links = extract_drive_links(read(path / "README.md"))
        for row in links:
            checkpoint_rows.append({"repo": repo["name"], **row})
    unique_checkpoint_file_ids = sorted({row["file_id"] for row in checkpoint_rows})

    next_actions = [
        {
            "rank": 1,
            "track": "SOTA/backbone plug-in",
            "action": "Reproduce DCAE official baseline with one low-rate MSE checkpoint, preferably lambda 0.0018 or 0.0035, on a small path-fixed Kodak/OpenImages subset.",
            "why": "DCAE has the cleanest y_hat -> g_s boundary and same commit across the mirrored/official clone.",
        },
        {
            "rank": 2,
            "track": "SOTA/backbone plug-in",
            "action": "Train/evaluate a DCAE-over-itself HCS/HCG branch adapter, reporting bpp, PSNR, MS-SSIM, RD, tails, and feature distributions.",
            "why": "This directly tests whether HCG branch improves a strong backbone over its own baseline.",
        },
        {
            "rank": 3,
            "track": "Controlled evidence",
            "action": "Keep E151/E152 as the main controlled branch evidence and extend only with path-matched split/rate checks.",
            "why": "It protects the paper claim from being diluted by backbone effects.",
        },
        {
            "rank": 4,
            "track": "Comparison planning",
            "action": "Use HPCM/MambaIC as strong LIC baselines and design references; defer direct plug-in until DCAE reproduction is stable.",
            "why": "They are stronger but have heavier dependencies and deeper codec paths.",
        },
        {
            "rank": 5,
            "track": "VQ comparison planning",
            "action": "Track RDVQ as a must-cite/must-compare VQ method, but do not block experiments on it until code is released.",
            "why": "The local official README still says code will come soon.",
        },
    ]

    summary = {
        "experiment": "E154 SOTA/backbone reproduction audit",
        "decision": "Proceed with two tracks: controlled E151/E152 evidence, plus DCAE-first SOTA plug-in reproduction.",
        "dcae_existing_and_official_candidate_same_commit": dcae_match,
        "repos": repos,
        "checkpoint_link_count": len(checkpoint_rows),
        "unique_checkpoint_file_id_count": len(unique_checkpoint_file_ids),
        "next_actions": next_actions,
    }

    (OUT_PREFIX.with_suffix(".json")).write_text(json.dumps(summary, indent=2, sort_keys=True))

    with (OUT_PREFIX.with_name(OUT_PREFIX.name + "_repos.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(repos[0].keys()))
        writer.writeheader()
        writer.writerows(repos)

    with (OUT_PREFIX.with_name(OUT_PREFIX.name + "_checkpoint_links.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repo", "lambda_or_quality", "metric", "file_id", "source_line"],
        )
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    with (OUT_PREFIX.with_name(OUT_PREFIX.name + "_next_actions.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "track", "action", "why"])
        writer.writeheader()
        writer.writerows(next_actions)

    lines = [
        "# E154 SOTA/Backbone Reproduction Audit",
        "",
        "This audit fixes the reproduction entry points before spending larger GPU budget.",
        "",
        f"- DCAE mirrored clone and README official clone target have same commit: `{dcae_match}`",
        f"- checkpoint links parsed from local READMEs: `{len(checkpoint_rows)}`",
        f"- unique checkpoint file IDs after mirror de-duplication: `{len(unique_checkpoint_file_ids)}`",
        "",
        "| repo | priority | commit | checkpoint links | risk |",
        "|---|---|---:|---:|---|",
    ]
    for row in repos:
        lines.append(
            f"| {row['name']} | {row['priority']} | `{str(row['commit'])[:12]}` | "
            f"{row['google_drive_checkpoint_links']} | {row['risk']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "DCAE remains the first SOTA/backbone plug-in target.  The local `CVL-UESTC/DCAE` clone and "
            "the README official clone target `LabShuHangGU/DCAE` resolve to the same commit, so E153 is "
            "not invalidated by the remote-name mismatch.  The next meaningful SOTA experiment is not a "
            "raw HCG transplant; it is a same-backbone DCAE comparison where an HCS/HCG state-preserving "
            "branch is evaluated against DCAE itself.",
            "",
            "HPCM and MambaIC should stay in the comparison/design queue for now. HPCM has strong R-D data "
            "and many checkpoints, but the progressive codec path is deeper. MambaIC has a released model "
            "entry on HuggingFace but carries VMamba/selective-scan dependency risk. RDVQ is paper-critical "
            "for VQ comparison, but its official README still marks code as forthcoming.",
            "",
            "## Next Actions",
            "",
        ]
    )
    for row in next_actions:
        lines.append(f"{row['rank']}. **{row['track']}**: {row['action']}  ")
        lines.append(f"   Reason: {row['why']}")
    lines.append("")
    (OUT_PREFIX.with_suffix(".md")).write_text("\n".join(lines))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
