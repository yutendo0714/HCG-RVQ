#!/usr/bin/env bash
set -euo pipefail

# Paper-safe deployable evaluation for learned hard HCG-RVQ selection.
# This exports only rows that can be decoded from a deterministic hard decision
# plus an explicit image-level selection/mode signal. Soft rows are omitted by
# default so they cannot accidentally be used as the main claim.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [OUTPUT_ROOT]" >&2
  exit 2
fi

CKPT="$1"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/eval_learnedhard_$(basename "${CKPT%.pt}")}"
INPUT_PATH="${INPUT_PATH:-/dpl/clic/professional/test}"
EVAL_LIMIT="${EVAL_LIMIT:-250}"
Q_INDEXES="${Q_INDEXES:-3}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.12}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:-256}"
REPLACEMENT_SIGNAL_BITS="${REPLACEMENT_SIGNAL_BITS:-1}"
LEARNED_HARD_MIN_Q="${LEARNED_HARD_MIN_Q:-3}"
EXPORT_LABELS="${EXPORT_LABELS:-base replacement_hard}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
  --branch-checkpoint "$CKPT" \
  --input-path "$INPUT_PATH" \
  --output-root "$OUT_ROOT" \
  --device cuda:0 \
  --q-indexes $Q_INDEXES \
  --eval-limit "$EVAL_LIMIT" \
  --labels $EXPORT_LABELS \
  --context-from-scalar \
  --replacement-signal-bits $REPLACEMENT_SIGNAL_BITS \
  --learned-hard-min-q $LEARNED_HARD_MIN_Q \
  --active-threshold "$ACTIVE_THRESHOLD" \
  --fid-patch-size "$FID_PATCH_SIZE"

"$PYTHON_BIN" tools/summarize_glc_hcg_eval.py "$OUT_ROOT" --write

echo "[done] learned-hard eval output: $OUT_ROOT"
