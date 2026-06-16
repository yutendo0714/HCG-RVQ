#!/usr/bin/env bash
set -euo pipefail

# E408: distill the strong E393 soft replacement upper bound into a deployable
# hard/ST q2-q3 HCG-RVQ branch.  This is the next VCIP-critical branch after
# E407 showed rate savings but residual perceptual degradation in hard mode.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e408_q23_teacher_from_e393_$(date +%Y%m%d_%H%M%S)}"
TEACHER_SOFT_WEIGHT="${TEACHER_SOFT_WEIGHT:-1.00}"
TEACHER_L1_WEIGHT="${TEACHER_L1_WEIGHT:-0.05}"
TEACHER_MSE_WEIGHT="${TEACHER_MSE_WEIGHT:-0.00}"
TEACHER_LPIPS_WEIGHT="${TEACHER_LPIPS_WEIGHT:-0.25}"
TEACHER_DISTS_WEIGHT="${TEACHER_DISTS_WEIGHT:-1.00}"

EXTRA_ARGS="${EXTRA_ARGS:-} --teacher-soft-weight ${TEACHER_SOFT_WEIGHT} --teacher-l1-weight ${TEACHER_L1_WEIGHT} --teacher-mse-weight ${TEACHER_MSE_WEIGHT} --teacher-lpips-weight ${TEACHER_LPIPS_WEIGHT} --teacher-dists-weight ${TEACHER_DISTS_WEIGHT}" \
Q_INDEXES="${Q_INDEXES:-2 3}" \
SEEDS="${SEEDS:-1234}" \
STEPS="${STEPS:-1500}" \
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}" \
TRAIN_LIMIT="${TRAIN_LIMIT:-4096}" \
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-16}" \
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.05}" \
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-0.30}" \
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.30}" \
DISTS_WEIGHT="${DISTS_WEIGHT:-1.00}" \
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.05}" \
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.002}" \
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.02}" \
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.0002}" \
OUT_ROOT="$OUT_ROOT" \
bash /workspace/HCG-RVQ/scripts/run_vcip_glc_hcg_hardst_from_e393.sh

echo "[next] CLIC64 deployable qmin eval example:"
echo "CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=0.05 FID_PATCH_SIZE=-1 Q_INDEXES=\"2 3\" EXPORT_LABELS=\"base\" REPLACEMENT_SIGNAL_BITS=\"1\" LEARNED_HARD_MIN_Q=\"2 3\" bash scripts/eval_vcip_glc_hcg_deployable_clic64.sh CHECKPOINT.pt OUT_DIR"
