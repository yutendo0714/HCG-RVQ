#!/usr/bin/env bash
set -euo pipefail

# VCIP-critical experiment: convert the strong E393 replacement_soft headroom
# into a deployable hard/ST HCG-RVQ + GLC branch.
#
# Default is a short 1-seed triage. Promote to SEEDS="1234 2345 3456" only
# after CLIC64 hard rows improve perceptual metrics with correct bpp accounting.

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
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/glc_hcg_hardst_from_e393_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-1234}"
STEPS="${STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-8192}"
TRAIN_LIMIT="${TRAIN_LIMIT:-4096}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-16}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.10}"
Q_INDEXES="${Q_INDEXES:-0 1 2 3}"
Q_TAG="${Q_INDEXES// /}"
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-0.20}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.00}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.00}"
GATE_RATE_WEIGHT="${GATE_RATE_WEIGHT:-0.05}"
GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.0005}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.30}"
DISTS_WEIGHT="${DISTS_WEIGHT:-1.00}"
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

  prefix="${OUT_ROOT}/glc_hcg_hardst_from_e393_seed${seed}_q${Q_TAG}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
  wandb_args=()
  if [[ "$WANDB_ENABLED" != "0" ]]; then
    wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
    wandb_args+=(--wandb-name "vcip_hardst_from_e393_seed${seed}_steps${STEPS}")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  echo "[run] seed=${seed} init=${init_ckpt}"
  "$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
    --train-dir "$TRAIN_DIR" \
    --eval-dir "$CLIC_PRO_DIR" \
    --ckpt-path "$CKPT_PATH" \
    --init-branch-checkpoint "$init_ckpt" \
    --output-prefix "$prefix" \
    --device cuda:0 \
    --q-indexes $Q_INDEXES \
    --padding-size 64 \
    --train-crop-size 256 \
    --eval-crop-size 0 \
    --train-start-index "$TRAIN_START_INDEX" \
    --eval-start-index 0 \
    --train-limit "$TRAIN_LIMIT" \
    --train-batch-per-step "$TRAIN_BATCH_PER_STEP" \
    --eval-limit 100000 \
    --group-size 16 \
    --active-groups 1 7 10 15 \
    --active-parts 0 1 \
    --scope part_group \
    --k 4 \
    --stages 1 \
    --steps "$STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --lr-codebook 2e-4 \
    --lr-controller 1e-4 \
    --mse-weight 0.00 \
    --l1-weight 0.00 \
    --lpips-weight "$LPIPS_WEIGHT" \
    --dists-weight "$DISTS_WEIGHT" \
    --branch-image-weight "$BRANCH_IMAGE_WEIGHT" \
    --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
    --glc-code-weight "$GLC_CODE_WEIGHT" \
    --soft-index-weight 0.002 \
    --soft-index-target 2.0 \
    --soft-index-temp 0.05 \
    --gate-rate-weight "$GATE_RATE_WEIGHT" \
    --gate-l1-weight "$GATE_L1_WEIGHT" \
    --active-threshold "$ACTIVE_THRESHOLD" \
    --train-hard-gate-st \
    --emit-replacement-rows \
    --replacement-cap-dbpp 0.0035 \
    --replacement-cap-dbpp-values 0.0030 0.0040 \
    --replacement-signal-bits 1 8 \
    --qaware-controller-json "$E379_JSON" \
    --qaware-policy-modes q-aware global \
    --skip-init-eval \
    --skip-final-eval \
    "${wandb_args[@]}" \
    $EXTRA_ARGS \
    --seed "$seed"
done

echo "[done] outputs under ${OUT_ROOT}"
