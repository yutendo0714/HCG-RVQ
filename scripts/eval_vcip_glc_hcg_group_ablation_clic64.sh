#!/usr/bin/env bash
set -euo pipefail

# Group-wise honest-accounting ablation for HCG-RVQ + GLC.
# This is a cheap CLIC64 selector/proofing pass: it tests whether a subset of
# latent groups keeps perceptual gains while preserving deployable accounting.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [OUTPUT_ROOT]" >&2
  exit 2
fi

CKPT="$1"
OUT_ROOT="${2:-/workspace/HCG-RVQ/experiments/analysis/eval_group_ablation_$(basename "${CKPT%.pt}")}"
INPUT_PATH="${INPUT_PATH:-/workspace/HCG-RVQ/experiments/analysis/clic_test64_subset}"
EVAL_LIMIT="${EVAL_LIMIT:-64}"
Q_INDEXES="${Q_INDEXES:-3}"
FID_PATCH_SIZE="${FID_PATCH_SIZE:--1}"
FIXED_GATES="${FIXED_GATES:-0.03 0.04 0.05 0.06 0.08 0.10 0.12}"
SIGNAL_BITS="${SIGNAL_BITS:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

# Semicolon-separated group sets. Defaults cover the four historical active
# groups plus small pairs; override when a follow-up wants a narrower sweep.
if [[ -n "${GROUP_SETS:-}" ]]; then
  mapfile -t GROUP_ARRAY < <(.venv/bin/python -c 'import os; print("\n".join(x.strip() for x in os.environ["GROUP_SETS"].split(";") if x.strip()))')
else
  GROUP_ARRAY=("1" "7" "10" "15" "1 7" "1 10" "1 15" "7 10" "7 15" "10 15" "1 7 10 15")
fi

mkdir -p "$OUT_ROOT"

for GROUPS in "${GROUP_ARRAY[@]}"; do
  [[ -n "$GROUPS" ]] || continue
  TAG="g$(echo "$GROUPS" | tr ' ' '-')"
  GROUP_OUT="$OUT_ROOT/$TAG"
  echo "[group-ablation] groups=[$GROUPS] out=$GROUP_OUT"

  "$PYTHON_BIN" tools/export_glc_hcg_qaware_branch_recon.py \
    --branch-checkpoint "$CKPT" \
    --input-path "$INPUT_PATH" \
    --output-root "$GROUP_OUT" \
    --device cuda:0 \
    --q-indexes $Q_INDEXES \
    --eval-limit "$EVAL_LIMIT" \
    --labels base \
    --context-from-scalar \
    --active-groups $GROUPS \
    --fixed-gate-threshold-feature index_entropy_mean \
    --fixed-gate-threshold-op '>=' \
    --fixed-gate-threshold-values -999 \
    --fixed-gate-threshold-gates $FIXED_GATES \
    --fixed-gate-threshold-accounting honest \
    --replacement-signal-bits $SIGNAL_BITS \
    --fid-patch-size "$FID_PATCH_SIZE"

  "$PYTHON_BIN" tools/summarize_glc_hcg_eval.py "$GROUP_OUT" --write
done

"$PYTHON_BIN" tools/summarize_glc_hcg_group_ablation.py "$OUT_ROOT" --write
echo "[done] group ablation output: $OUT_ROOT"
