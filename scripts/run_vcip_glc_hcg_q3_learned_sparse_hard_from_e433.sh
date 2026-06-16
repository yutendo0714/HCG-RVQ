#!/usr/bin/env bash
set -euo pipefail

# E448: VCIP claim-path experiment.
# Train a sparse learned hard selector from the strong E433 q=3 branch.
#
# Why this exists:
# - E433/E440 showed strong perceptual headroom with soft/fixed gates.
# - E442 proved symbol round-trip consistency for the same branch.
# - E443/E446 showed that all-on / broad hard replacement hurts quality.
# This run therefore keeps the deployable g=1 replacement path, but makes the
# controller conservative so the paper claim can stay hard/accounted.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

TRAIN_DIR="${TRAIN_DIR:-/dpl/openimages/open-images-v6/train/data}"
EVAL_DIR="${EVAL_DIR:-/workspace/HCG-RVQ/experiments/analysis/clic_test64_subset}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
INIT_CKPT="${INIT_CKPT:-/workspace/HCG-RVQ/experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e448_glc_hcg_q3_learned_sparse_hard_from_e433_$(date +%Y%m%d_%H%M%S)}"

SEED="${SEED:-1234}"
STEPS="${STEPS:-1800}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-300}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8192}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-32768}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-120000}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-64}"

# The threshold is intentionally above the E433 soft-gate operating point.
# Branch image loss keeps the branch trainable even when the hard selector is
# initially sparse, while gate/rate terms discourage broad replacement.
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.12}"
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-0.12}"
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.18}"
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.025}"

LR_CODEBOOK="${LR_CODEBOOK:-8e-5}"
LR_CONTROLLER="${LR_CONTROLLER:-1.5e-4}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.25}"
LPIPS_NET="${LPIPS_NET:-alex}"
DISTS_WEIGHT="${DISTS_WEIGHT:-1.15}"
L1_WEIGHT="${L1_WEIGHT:-0.08}"
MSE_WEIGHT="${MSE_WEIGHT:-0.0}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.04}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.0015}"
SOFT_INDEX_WEIGHT="${SOFT_INDEX_WEIGHT:-0.025}"
SOFT_INDEX_TARGET="${SOFT_INDEX_TARGET:-1.60}"
SOFT_INDEX_FLOOR_WEIGHT="${SOFT_INDEX_FLOOR_WEIGHT:-0.02}"
SOFT_INDEX_FLOOR="${SOFT_INDEX_FLOOR:-1.10}"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
wandb_args=()
if [[ "$WANDB_ENABLED" == "1" ]]; then
  wandb_args+=(--wandb-project "${WANDB_PROJECT:-hcg-rvq-vcip}")
  wandb_args+=(--wandb-name "e448_q3_learned_sparse_hard_seed${SEED}_steps${STEPS}")
fi

prefix="${OUT_ROOT}/glc_hcg_q3_learned_sparse_hard_seed${SEED}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$EVAL_DIR" \
  --ckpt-path "$CKPT_PATH" \
  --output-prefix "$prefix" \
  --device cuda:0 \
  --seed "$SEED" \
  --padding-size 64 \
  --train-crop-size 256 \
  --eval-crop-size 0 \
  --train-limit "$TRAIN_LIMIT" \
  --train-start-index "$TRAIN_START_INDEX" \
  --eval-start-index 0 \
  --eval-limit "$EVAL_LIMIT" \
  --q-indexes 3 \
  --active-parts 0 1 \
  --active-groups 1 7 10 15 \
  --k 4 \
  --stages 2 \
  --max-train-vectors "$MAX_TRAIN_VECTORS" \
  --train-batch-per-step "$TRAIN_BATCH_PER_STEP" \
  --steps "$STEPS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --init-branch-checkpoint "$INIT_CKPT" \
  --context-from-scalar \
  --train-hard-gate-st \
  --active-threshold "$ACTIVE_THRESHOLD" \
  --lr-codebook "$LR_CODEBOOK" \
  --lr-controller "$LR_CONTROLLER" \
  --l1-weight "$L1_WEIGHT" \
  --mse-weight "$MSE_WEIGHT" \
  --lpips-weight "$LPIPS_WEIGHT" \
  --lpips-net "$LPIPS_NET" \
  --dists-weight "$DISTS_WEIGHT" \
  --branch-image-weight "$BRANCH_IMAGE_WEIGHT" \
  --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
  --glc-code-weight "$GLC_CODE_WEIGHT" \
  --soft-index-weight "$SOFT_INDEX_WEIGHT" \
  --soft-index-target "$SOFT_INDEX_TARGET" \
  --soft-index-floor-weight "$SOFT_INDEX_FLOOR_WEIGHT" \
  --soft-index-floor-target "$SOFT_INDEX_FLOOR" \
  --gate-rate-weight "$GATE_RATE_WEIGHT" \
  --gate-l1-weight "$GATE_L1_WEIGHT" \
  --replacement-signal-bits 1 \
  --cache-images-on-cpu \
  "${wandb_args[@]}"

cat <<EOF
[done] trained: $prefix

[triage] Evaluate checkpoints on CLIC64 learned-hard accounting:
CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD Q_INDEXES="3" \
  INPUT_PATH=/workspace/HCG-RVQ/experiments/analysis/clic_test64_subset EVAL_LIMIT=64 FID_PATCH_SIZE=-1 \
  bash scripts/eval_vcip_glc_hcg_learnedhard.sh ${prefix}_stepXXXX.pt OUT_DIR

[promote] If a checkpoint is non-degrading on CLIC64, run official CLIC250:
CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD Q_INDEXES="3" EVAL_LIMIT=250 FID_PATCH_SIZE=256 \\
  bash scripts/eval_vcip_glc_hcg_learnedhard.sh ${prefix}_stepXXXX.pt OUT_DIR

[audit] For the promoted checkpoint, run symbol round-trip:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  .venv/bin/python tools/audit_glc_hcg_symbol_roundtrip.py \\
    --branch-checkpoint ${prefix}_stepXXXX.pt \\
    --input-path /dpl/clic/professional/test \\
    --output-prefix OUT_PREFIX \\
    --q-indexes 3 --eval-limit 250 --active-threshold $ACTIVE_THRESHOLD \\
    --selection-signal-bits 1
EOF
