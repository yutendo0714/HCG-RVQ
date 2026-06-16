#!/usr/bin/env bash
set -euo pipefail

# VCIP E438: fine-grained reliability sweep for the current best E433 branch.
# This keeps the method simple: q=3, context-from-scalar, replacement accounting,
# 1-bit image-level selection signal, and index-entropy-based fixed reliability.
# GPU1 is broken in this environment; use GPU0 only.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT="${1:-experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/e438_e433_indexentropy_fine_sweep_clic250_$(date +%Y%m%d_%H%M%S)}"
INPUT_PATH="${INPUT_PATH:-/dpl/clic/professional/test}"
Q_INDEXES="${Q_INDEXES:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-250}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:-256}"
THRESHOLD_FEATURE="${THRESHOLD_FEATURE:-index_entropy_mean}"
THRESHOLD_VALUES="${THRESHOLD_VALUES:-1.55 1.58032 1.60 1.62}"
FIXED_GATES="${FIXED_GATES:-0.12 0.16 0.20}"
ACCOUNTING="${ACCOUNTING:-replacement}"
SIGNAL_BITS="${SIGNAL_BITS:-1}"
QUALITY_LABELS="${QUALITY_LABELS:-base th_indexentropymean_ge1p55_g0p12_replacement_sig1b th_indexentropymean_ge1p55_g0p16_replacement_sig1b th_indexentropymean_ge1p58032_g0p12_replacement_sig1b th_indexentropymean_ge1p58032_g0p16_replacement_sig1b th_indexentropymean_ge1p6_g0p16_replacement_sig1b th_indexentropymean_ge1p6_g0p2_replacement_sig1b th_indexentropymean_ge1p62_g0p2_replacement_sig1b}"

mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
  --branch-checkpoint "$CHECKPOINT" \
  --input-path "$INPUT_PATH" \
  --output-root "$OUT_ROOT" \
  --device cuda:0 \
  --q-indexes $Q_INDEXES \
  --eval-limit "$EVAL_LIMIT" \
  --labels base \
  --context-from-scalar \
  --fixed-gate-threshold-feature "$THRESHOLD_FEATURE" \
  --fixed-gate-threshold-op '>=' \
  --fixed-gate-threshold-values $THRESHOLD_VALUES \
  --fixed-gate-threshold-gates $FIXED_GATES \
  --fixed-gate-threshold-accounting $ACCOUNTING \
  --replacement-signal-bits $SIGNAL_BITS \
  --fid-patch-size "$FID_PATCH_SIZE" \
  --skip-quality

"$PYTHON_BIN" tools/eval_existing_glc_hcg_quality.py "$OUT_ROOT" \
  --input-path "$INPUT_PATH" \
  --q-indexes $Q_INDEXES \
  --patch-size "$FID_PATCH_SIZE" \
  --labels $QUALITY_LABELS

"$PYTHON_BIN" tools/summarize_glc_hcg_eval.py "$OUT_ROOT" --write

echo "[done] E438 official CLIC250 fine sweep: $OUT_ROOT"
echo "[read] $OUT_ROOT/summary.csv"
