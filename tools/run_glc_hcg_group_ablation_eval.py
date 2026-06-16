#!/usr/bin/env python3
"""Run group-wise HCG-RVQ/GLC export evaluations sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_group_sets(text: str) -> list[list[int]]:
    out: list[list[int]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        out.append([int(x) for x in item.split()])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, default=ROOT / "experiments/analysis/clic_test64_subset")
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--q-indexes", nargs="+", type=int, default=[3])
    parser.add_argument("--fid-patch-size", type=int, default=-1)
    parser.add_argument("--fixed-gates", nargs="+", type=float, default=[0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12])
    parser.add_argument(
        "--group-sets",
        default="1;7;10;15;1 7;1 10;1 15;7 10;7 15;10 15;1 7 10 15",
    )
    parser.add_argument("--signal-bits", type=float, default=1.0)
    parser.add_argument("--accounting", nargs="+", default=["honest"], choices=["honest", "replacement"])
    parser.add_argument("--python-bin", default=str(ROOT / ".venv/bin/python"))
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party/GLC'}:{env.get('PYTHONPATH', '')}"

    args.output_root.mkdir(parents=True, exist_ok=True)
    for groups in parse_group_sets(args.group_sets):
        tag = "g" + "-".join(str(x) for x in groups)
        group_out = args.output_root / tag
        print(f"[group-ablation] groups={groups} out={group_out}", flush=True)
        export_cmd = [
            args.python_bin,
            "tools/export_glc_hcg_qaware_branch_recon.py",
            "--branch-checkpoint",
            str(args.branch_checkpoint),
            "--input-path",
            str(args.input_path),
            "--output-root",
            str(group_out),
            "--device",
            "cuda:0",
            "--q-indexes",
            *[str(q) for q in args.q_indexes],
            "--eval-limit",
            str(args.eval_limit),
            "--labels",
            "base",
            "--context-from-scalar",
            "--active-groups",
            *[str(g) for g in groups],
            "--fixed-gate-threshold-feature",
            "index_entropy_mean",
            "--fixed-gate-threshold-op",
            ">=",
            "--fixed-gate-threshold-values",
            "-999",
            "--fixed-gate-threshold-gates",
            *[str(g) for g in args.fixed_gates],
            "--fixed-gate-threshold-accounting",
            *args.accounting,
            "--replacement-signal-bits",
            str(args.signal_bits),
            "--fid-patch-size",
            str(args.fid_patch_size),
        ]
        subprocess.run(export_cmd, cwd=ROOT, env=env, check=True)
        subprocess.run(
            [args.python_bin, "tools/summarize_glc_hcg_eval.py", str(group_out), "--write"],
            cwd=ROOT,
            env=env,
            check=True,
        )

    subprocess.run(
        [args.python_bin, "tools/summarize_glc_hcg_group_ablation.py", str(args.output_root), "--write"],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
