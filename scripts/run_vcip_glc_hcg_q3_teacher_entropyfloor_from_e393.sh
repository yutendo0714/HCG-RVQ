#!/usr/bin/env bash
set -euo pipefail

# E429 / VCIP focused triage.
# Distill the useful E393/E427 soft HCG correction into a deployable q3-only
# hard/ST branch.  E427 fixed codebook entropy collapse, but the hard/all-on
# branch still degraded perceptual quality; this run changes the branch target
# instead of only changing the reliability controller.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-/dpl/openimages/open-images-v6/train/data}"
CLIC_PRO_DIR="${CLIC_PRO_DIR:-/dpl/clic/professional/test}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
INIT_ROOT="${INIT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/glc_qaware_paper_branch_20260609_034248}"
E379_JSON="${E379_JSON:-/workspace/HCG-RVQ/experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e429_glc_hcg_q3_teacher_entropyfloor_from_e393_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-1234}"
STEPS="${STEPS:-1200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-300}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-8192}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8192}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-8}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-120000}"
MAX_RATE_VECTORS="${MAX_RATE_VECTORS:-4096}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.08}"
ACTIVE_GROUPS="${ACTIVE_GROUPS:-1 7 10 15}"
ACTIVE_PARTS="${ACTIVE_PARTS:-0 1}"

# Keep the objective close to the HCG story: first learn a good latent/residual
# branch, then distill the soft reliable HCG correction into a single sendable
# branch.  This is intentionally not an FID/KID-specific loss.
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-0.20}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.25}"
DISTS_WEIGHT="${DISTS_WEIGHT:-1.00}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.03}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.001}"
TEACHER_SOFT_WEIGHT="${TEACHER_SOFT_WEIGHT:-1.00}"
TEACHER_L1_WEIGHT="${TEACHER_L1_WEIGHT:-0.05}"
TEACHER_MSE_WEIGHT="${TEACHER_MSE_WEIGHT:-0.00}"
TEACHER_LPIPS_WEIGHT="${TEACHER_LPIPS_WEIGHT:-0.30}"
TEACHER_DISTS_WEIGHT="${TEACHER_DISTS_WEIGHT:-1.00}"
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.03}"
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.0003}"
SOFT_INDEX_WEIGHT="${SOFT_INDEX_WEIGHT:-0.02}"
SOFT_INDEX_TARGET="${SOFT_INDEX_TARGET:-1.60}"
SOFT_INDEX_FLOOR_TARGET="${SOFT_INDEX_FLOOR_TARGET:-0.90}"
SOFT_INDEX_FLOOR_WEIGHT="${SOFT_INDEX_FLOOR_WEIGHT:-1.00}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-HCG-RVQ}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
  init_ckpt="${INIT_ROOT}/glc_hcg_qaware_clicprof_seed${seed}_q0123_k4_p01_steps3000_batch16_step3000.pt"
  if [[ ! -f "$init_ckpt" ]]; then
    echo "[error] missing init checkpoint: $init_ckpt" >&2
    exit 2
  fi

  prefix="${OUT_ROOT}/glc_hcg_q3_teacher_entropyfloor_seed${seed}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
  wandb_args=()
  if [[ "$WANDB_ENABLED" != "0" ]]; then
    wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
    wandb_args+=(--wandb-name "e429_q3_teacher_entropyfloor_seed${seed}_steps${STEPS}")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  echo "[run] seed=${seed} init=${init_ckpt} out=${prefix}"
  "$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
    --train-dir "$TRAIN_DIR" \
    --eval-dir "$CLIC_PRO_DIR" \
    --ckpt-path "$CKPT_PATH" \
    --init-branch-checkpoint "$init_ckpt" \
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
    --lr-codebook 1e-4 \
    --lr-controller 5e-5 \
    --mse-weight 0.00 \
    --l1-weight 0.00 \
    --lpips-weight "$LPIPS_WEIGHT" \
    --dists-weight "$DISTS_WEIGHT" \
    --branch-image-weight "$BRANCH_IMAGE_WEIGHT" \
    --teacher-soft-weight "$TEACHER_SOFT_WEIGHT" \
    --teacher-l1-weight "$TEACHER_L1_WEIGHT" \
    --teacher-mse-weight "$TEACHER_MSE_WEIGHT" \
    --teacher-lpips-weight "$TEACHER_LPIPS_WEIGHT" \
    --teacher-dists-weight "$TEACHER_DISTS_WEIGHT" \
    --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
    --glc-code-weight "$GLC_CODE_WEIGHT" \
    --soft-index-weight "$SOFT_INDEX_WEIGHT" \
    --soft-index-target "$SOFT_INDEX_TARGET" \
    --soft-index-floor-target "$SOFT_INDEX_FLOOR_TARGET" \
    --soft-index-floor-weight "$SOFT_INDEX_FLOOR_WEIGHT" \
    --soft-index-temp 0.05 \
    --gate-rate-weight "$GATE_RATE_WEIGHT" \
    --gate-l1-weight "$GATE_L1_WEIGHT" \
    --active-threshold "$ACTIVE_THRESHOLD" \
    --train-hard-gate-st \
    --emit-replacement-rows \
    --replacement-cap-dbpp 0.0015 \
    --replacement-cap-dbpp-values 0.0000 0.0010 0.0020 0.0030 \
    --replacement-signal-bits 1 8 \
    --qaware-controller-json "$E379_JSON" \
    --qaware-policy-modes q-aware global \
    --cache-images-on-cpu \
    "${wandb_args[@]}" \
    $EXTRA_ARGS \
    --seed "$seed"
done

echo "[done] outputs under ${OUT_ROOT}"
echo "[checkpoint eval CLIC64]"
echo "CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=${ACTIVE_THRESHOLD} FID_PATCH_SIZE=-1 Q_INDEXES=\"3\" EXPORT_LABELS=\"base replacement_hard replacement_soft\" REPLACEMENT_SIGNAL_BITS=\"1 8\" LEARNED_HARD_MIN_Q=\"3\" bash scripts/eval_vcip_glc_hcg_deployable_clic64.sh CHECKPOINT.pt OUT_DIR"
