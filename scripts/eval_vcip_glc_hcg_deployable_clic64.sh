#!/usr/bin/env bash
set -euo pipefail

# Fast CLIC64 triage for claimable HCG-RVQ + GLC checkpoints.
# Outputs learned hard rows with explicit 1/8-bit image-level selection signal,
# plus replacement_soft as a non-deployable upper-bound reference.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [OUTPUT_ROOT]" >&2
  exit 2
fi

CKPT="$1"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/eval_$(basename "${CKPT%.pt}")_clic64}"
INPUT_PATH="${INPUT_PATH:-/workspace/HCG-RVQ/experiments/analysis/clic_test64_subset}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.10}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:-256}"
LEARNED_HARD_MIN_Q="${LEARNED_HARD_MIN_Q:-}"
Q_INDEXES="${Q_INDEXES:-0 1 2 3}"
EXPORT_LABELS="${EXPORT_LABELS:-base replacement_hard replacement_soft}"
REPLACEMENT_SIGNAL_BITS="${REPLACEMENT_SIGNAL_BITS:-1 8}"
QUANTIZE_SOFT_GATE_BITS="${QUANTIZE_SOFT_GATE_BITS:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
  --branch-checkpoint "$CKPT" \
  --input-path "$INPUT_PATH" \
  --output-root "$OUT_ROOT" \
  --device cuda:0 \
  --q-indexes $Q_INDEXES \
  --labels $EXPORT_LABELS \
  --replacement-signal-bits $REPLACEMENT_SIGNAL_BITS \
  ${QUANTIZE_SOFT_GATE_BITS:+--quantize-soft-gate-bits $QUANTIZE_SOFT_GATE_BITS} \
  ${LEARNED_HARD_MIN_Q:+--learned-hard-min-q $LEARNED_HARD_MIN_Q} \
  --active-threshold "$ACTIVE_THRESHOLD" \
  --fid-patch-size "$FID_PATCH_SIZE"

echo "[done] eval output: $OUT_ROOT"
