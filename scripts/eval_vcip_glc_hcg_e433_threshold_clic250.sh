#!/usr/bin/env bash
set -euo pipefail

# E434 / VCIP decisive evaluation.
# Official CLIC professional/test 250-image evaluation for threshold-selected
# fixed-gain HCG-RVQ policies discovered from E433 CLIC64 diagnostics.
# GPU1 is intentionally not used; pass CUDA_VISIBLE_DEVICES=0.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT="${1:-experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/e434_e433_threshold_fixedgate_clic250_$(date +%Y%m%d_%H%M%S)}"
INPUT_PATH="${INPUT_PATH:-/dpl/clic/professional/test}"
Q_INDEXES="${Q_INDEXES:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-250}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:--1}"
THRESHOLD_FEATURE="${THRESHOLD_FEATURE:-index_entropy_mean}"
THRESHOLD_VALUES="${THRESHOLD_VALUES:-1.58032 1.6146 1.62967 1.64082}"
FIXED_GATES="${FIXED_GATES:-0.06 0.08 0.10 0.12}"
ACCOUNTING="${ACCOUNTING:-replacement honest}"
SIGNAL_BITS="${SIGNAL_BITS:-1 8}"
EXTRA_LABELS="${EXTRA_LABELS:-replacement_soft soft_gate}"

mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
  --branch-checkpoint "$CHECKPOINT" \
  --input-path "$INPUT_PATH" \
  --output-root "$OUT_ROOT" \
  --device cuda:0 \
  --q-indexes $Q_INDEXES \
  --eval-limit "$EVAL_LIMIT" \
  --labels base $EXTRA_LABELS \
  --context-from-scalar \
  --fixed-gate-threshold-feature "$THRESHOLD_FEATURE" \
  --fixed-gate-threshold-op '>=' \
  --fixed-gate-threshold-values $THRESHOLD_VALUES \
  --fixed-gate-threshold-gates $FIXED_GATES \
  --fixed-gate-threshold-accounting $ACCOUNTING \
  --replacement-signal-bits $SIGNAL_BITS \
  --fid-patch-size "$FID_PATCH_SIZE"

"$PYTHON_BIN" tools/summarize_glc_hcg_eval.py "$OUT_ROOT" || true

echo "[done] threshold CLIC250 eval: $OUT_ROOT"
echo "[focus] claim candidates: th_indexentropymean_ge1p58032_g0p1_replacement_sig1b / g0p12; verify LPIPS,DISTS,MS-SSIM,FID/KID against base."
