#!/usr/bin/env bash
set -euo pipefail

# Honest enhancement evaluation for HCG-RVQ + GLC.
# Base GLC stream is preserved; active RVQ symbols are counted as additional
# enhancement bits. This avoids replacement accounting ambiguity.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [OUTPUT_ROOT]" >&2
  exit 2
fi

CKPT="$1"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/eval_honest_enhance_$(basename "${CKPT%.pt}")}"
INPUT_PATH="${INPUT_PATH:-/dpl/clic/professional/test}"
EVAL_LIMIT="${EVAL_LIMIT:-250}"
Q_INDEXES="${Q_INDEXES:-3}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:-256}"
FIXED_GATES="${FIXED_GATES:-0.08 0.12 0.16 0.20 0.24 0.30}"
SIGNAL_BITS="${SIGNAL_BITS:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

# Use a very low threshold so every image gets the enhancement row. The bpp is
# honest: base_bpp + active_rvq_empirical_bpp + optional 1-bit mode signal.
"$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
  --branch-checkpoint "$CKPT" \
  --input-path "$INPUT_PATH" \
  --output-root "$OUT_ROOT" \
  --device cuda:0 \
  --q-indexes $Q_INDEXES \
  --eval-limit "$EVAL_LIMIT" \
  --labels base \
  --context-from-scalar \
  --fixed-gate-threshold-feature index_entropy_mean \
  --fixed-gate-threshold-op '>=' \
  --fixed-gate-threshold-values -999 \
  --fixed-gate-threshold-gates $FIXED_GATES \
  --fixed-gate-threshold-accounting honest \
  --replacement-signal-bits $SIGNAL_BITS \
  --fid-patch-size "$FID_PATCH_SIZE"

"$PYTHON_BIN" tools/summarize_glc_hcg_eval.py "$OUT_ROOT" --write

echo "[done] honest enhancement eval output: $OUT_ROOT"
