#!/usr/bin/env bash
set -euo pipefail

# Next VCIP triage after e406:
# q2/q3 have real deployable bpp headroom, but the hard branch degrades
# perceptual quality. This run trains only q2/q3 and strengthens the branch
# image/semantic losses so the deployable hard output can catch up to the soft
# e393/e394 upper bound.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e407_q23_branchquality_from_e393_$(date +%Y%m%d_%H%M%S)}"

Q_INDEXES="${Q_INDEXES:-2 3}" \
SEEDS="${SEEDS:-1234}" \
STEPS="${STEPS:-2000}" \
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}" \
TRAIN_LIMIT="${TRAIN_LIMIT:-4096}" \
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-16}" \
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.05}" \
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-1.00}" \
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.50}" \
DISTS_WEIGHT="${DISTS_WEIGHT:-1.00}" \
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.10}" \
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.005}" \
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.02}" \
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.0002}" \
OUT_ROOT="$OUT_ROOT" \
bash /workspace/HCG-RVQ/scripts/run_vcip_glc_hcg_hardst_from_e393.sh

echo "[next] CLIC64 deployable qmin eval example:"
echo "CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=0.05 FID_PATCH_SIZE=-1 Q_INDEXES=\"2 3\" EXPORT_LABELS=\"base\" REPLACEMENT_SIGNAL_BITS=\"1\" LEARNED_HARD_MIN_Q=\"2 3\" bash scripts/eval_vcip_glc_hcg_deployable_clic64.sh CHECKPOINT.pt OUT_DIR"
