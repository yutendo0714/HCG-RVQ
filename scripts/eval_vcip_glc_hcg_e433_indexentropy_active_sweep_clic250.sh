#!/usr/bin/env bash
set -euo pipefail

# VCIP E439: two-feature reliability sweep for E433.
# The policy is intentionally simple and paper-facing:
# activate a fixed-gain HCG-RVQ correction only when both the index entropy
# and active residual difficulty are high. GPU1 is broken; use GPU0 only.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT="${1:-experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/e439_e433_indexentropy_active_sweep_clic250_$(date +%Y%m%d_%H%M%S)}"
INPUT_PATH="${INPUT_PATH:-/dpl/clic/professional/test}"
Q_INDEXES="${Q_INDEXES:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-250}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:-256}"
INDEX_THRESHOLDS="${INDEX_THRESHOLDS:-1.55 1.58 1.60}"
ACTIVE_THRESHOLDS="${ACTIVE_THRESHOLDS:-5.8 6.0}"
FIXED_GATES="${FIXED_GATES:-0.12 0.16 0.20}"
ACCOUNTING="${ACCOUNTING:-replacement}"
SIGNAL_BITS="${SIGNAL_BITS:-1}"
QUALITY_LABELS="${QUALITY_LABELS:-base th_indexentropymean_ge1p55_activemseratio_ge5p8_g0p12_replacement_sig1b th_indexentropymean_ge1p55_activemseratio_ge5p8_g0p16_replacement_sig1b th_indexentropymean_ge1p58_activemseratio_ge5p8_g0p16_replacement_sig1b th_indexentropymean_ge1p58_activemseratio_ge6p0_g0p16_replacement_sig1b th_indexentropymean_ge1p6_activemseratio_ge5p8_g0p2_replacement_sig1b th_indexentropymean_ge1p6_activemseratio_ge6p0_g0p2_replacement_sig1b}"

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
  --fixed-gate-threshold-feature index_entropy_mean \
  --fixed-gate-threshold-op '>=' \
  --fixed-gate-threshold-values $INDEX_THRESHOLDS \
  --fixed-gate-secondary-threshold-feature active_mse_ratio \
  --fixed-gate-secondary-threshold-op '>=' \
  --fixed-gate-secondary-threshold-values $ACTIVE_THRESHOLDS \
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

echo "[done] E439 official CLIC250 two-feature sweep: $OUT_ROOT"
echo "[read] $OUT_ROOT/summary.csv"
