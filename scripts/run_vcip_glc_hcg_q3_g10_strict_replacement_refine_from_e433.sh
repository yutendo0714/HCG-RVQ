#!/usr/bin/env bash
set -euo pipefail

# VCIP E455 candidate:
# Fine-tune only group 10 as a deployable strict replacement branch.
# E453/E454 showed group 10 is the cleanest perceptual group: soft/honest
# improves LPIPS/DISTS/MS-SSIM, while strict replacement saves bpp but hurts
# LPIPS/MS. This run optimizes the all-on branch itself, not a soft mix.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-/dpl/openimages/open-images-v6/train/data}"
CLIC_PRO_DIR="${CLIC_PRO_DIR:-/dpl/clic/professional/test}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
INIT_CKPT="${INIT_CKPT:-/workspace/HCG-RVQ/experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e455_glc_hcg_q3_g10_strict_replacement_refine_from_e433_$(date +%Y%m%d_%H%M%S)}"

SEED="${SEED:-1234}"
STEPS="${STEPS:-1200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-300}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-49152}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8192}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-8}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-120000}"
MAX_RATE_VECTORS="${MAX_RATE_VECTORS:-4096}"

# E453/E454: group 10 is the best single group.
ACTIVE_GROUPS="${ACTIVE_GROUPS:-10}"
ACTIVE_PARTS="${ACTIVE_PARTS:-0 1}"

# Train the deployable branch itself. The main soft-gate path remains present
# in the trainer, so branch-image loss is the primary replacement objective.
LPIPS_NET="${LPIPS_NET:-vgg}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.65}"
DISTS_WEIGHT="${DISTS_WEIGHT:-1.60}"
L1_WEIGHT="${L1_WEIGHT:-0.02}"
MSE_WEIGHT="${MSE_WEIGHT:-0.0}"
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-1.50}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.05}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.0010}"
SOFT_INDEX_WEIGHT="${SOFT_INDEX_WEIGHT:-0.02}"
SOFT_INDEX_TARGET="${SOFT_INDEX_TARGET:-1.55}"
SOFT_INDEX_FLOOR_TARGET="${SOFT_INDEX_FLOOR_TARGET:-1.00}"
SOFT_INDEX_FLOOR_WEIGHT="${SOFT_INDEX_FLOOR_WEIGHT:-0.50}"
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.00}"
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.00}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-HCG-RVQ}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT_ROOT"

prefix="${OUT_ROOT}/glc_hcg_q3_g10_strict_repl_seed${SEED}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
wandb_args=()
if [[ "$WANDB_ENABLED" != "0" ]]; then
  wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  wandb_args+=(--wandb-name "e455_q3_g10_strict_repl_seed${SEED}_steps${STEPS}")
  if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

"$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$CLIC_PRO_DIR" \
  --ckpt-path "$CKPT_PATH" \
  --init-branch-checkpoint "$INIT_CKPT" \
  --output-prefix "$prefix" \
  --device cuda:0 \
  --q-indexes 3 \
  --padding-size 64 \
  --train-crop-size 256 \
  --eval-crop-size 0 \
  --train-start-index "$TRAIN_START_INDEX" \
  --eval-start-index 0 \
  --train-limit "$TRAIN_LIMIT" \
  --train-batch-per-step "$TRAIN_BATCH_PER_STEP" \
  --eval-limit 64 \
  --group-size 16 \
  --active-groups $ACTIVE_GROUPS \
  --active-parts $ACTIVE_PARTS \
  --scope part_group \
  --k 4 \
  --stages 1 \
  --max-train-vectors "$MAX_TRAIN_VECTORS" \
  --max-rate-vectors "$MAX_RATE_VECTORS" \
  --steps "$STEPS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --lr-codebook 6e-5 \
  --lr-controller 1e-5 \
  --mse-weight "$MSE_WEIGHT" \
  --l1-weight "$L1_WEIGHT" \
  --lpips-weight "$LPIPS_WEIGHT" \
  --lpips-net "$LPIPS_NET" \
  --dists-weight "$DISTS_WEIGHT" \
  --branch-image-weight "$BRANCH_IMAGE_WEIGHT" \
  --teacher-soft-weight 0.00 \
  --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
  --glc-code-weight "$GLC_CODE_WEIGHT" \
  --soft-index-weight "$SOFT_INDEX_WEIGHT" \
  --soft-index-target "$SOFT_INDEX_TARGET" \
  --soft-index-floor-target "$SOFT_INDEX_FLOOR_TARGET" \
  --soft-index-floor-weight "$SOFT_INDEX_FLOOR_WEIGHT" \
  --soft-index-temp 0.05 \
  --gate-rate-weight "$GATE_RATE_WEIGHT" \
  --gate-l1-weight "$GATE_L1_WEIGHT" \
  --active-threshold 0.08 \
  --context-from-scalar \
  --emit-replacement-rows \
  --replacement-cap-dbpp-values 0.0000 0.0010 0.0020 \
  --replacement-signal-bits 1 \
  --cache-images-on-cpu \
  "${wandb_args[@]}" \
  $EXTRA_ARGS \
  --seed "$SEED"

cat <<EOF
[done] trained: $prefix

[eval strict replacement CLIC16/64]
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_glc_hcg_group_ablation_eval.py \\
  --branch-checkpoint ${prefix}_stepXXXX.pt \\
  --output-root OUT_DIR \\
  --input-path experiments/analysis/clic_test16_subset \\
  --eval-limit 16 --group-sets '10' --fixed-gates 1.0 --accounting replacement
EOF
