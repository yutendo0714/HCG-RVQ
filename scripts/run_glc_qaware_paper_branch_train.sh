#!/usr/bin/env bash
set -euo pipefail

# Paper-facing GLC/HCG-RVQ plug-in training.
#
# This is the recommended VCIP-critical-path run: start from the official
# pretrained GLC image checkpoint, keep the baseline checkpoint fixed for fair
# comparison, and train the HCG-RVQ branch plus q-aware reliability controller
# with a perceptual Stage-III-like objective. It is not a scratch reproduction
# of GLC Stage I/II/III.
#
# Required:
#   TRAIN_DIR=/path/to/openimages/train/data
#   CLIC_PRO_DIR=/path/to/CLIC/professional/valid-or-test-images
#
# Typical tmux use:
#   cd /workspace/HCG-RVQ
#   tmux new -s glc_hcg_paper
#   CUDA_VISIBLE_DEVICES=0 TRAIN_DIR=/dpl/openimages/open-images-v6/train/data \
#     CLIC_PRO_DIR=/data/clic/professional_valid SEEDS="1234 2345 3456" \
#     bash scripts/run_glc_qaware_paper_branch_train.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

TRAIN_DIR="${TRAIN_DIR:-/dpl/openimages/open-images-v6/train/data}"
CLIC_PRO_DIR="${CLIC_PRO_DIR:-}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
E379_JSON="${E379_JSON:-/workspace/HCG-RVQ/experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/glc_qaware_paper_branch_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-1234 2345 3456}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-8192}"
TRAIN_LIMIT="${TRAIN_LIMIT:-4096}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-100000}"
STEPS="${STEPS:-3000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-60000}"
MAX_RATE_VECTORS="${MAX_RATE_VECTORS:-8192}"
KMEANS_ITERS="${KMEANS_ITERS:-12}"
LPIPS_NET="${LPIPS_NET:-alex}"
L1_WEIGHT="${L1_WEIGHT:-0.00}"
BRANCH_IMAGE_WEIGHT="${BRANCH_IMAGE_WEIGHT:-0.00}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.00}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.00}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-HCG-RVQ}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$CLIC_PRO_DIR" ]]; then
  echo "[error] CLIC_PRO_DIR is required for the paper-facing GLC run." >&2
  exit 2
fi
if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "[error] TRAIN_DIR does not exist: $TRAIN_DIR" >&2
  exit 2
fi
if [[ ! -d "$CLIC_PRO_DIR" ]]; then
  echo "[error] CLIC_PRO_DIR does not exist: $CLIC_PRO_DIR" >&2
  exit 2
fi
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[error] GLC checkpoint does not exist: $CKPT_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

if [[ ! -f "$E379_JSON" ]]; then
  echo "[setup] building E379 q-aware entropy-margin deployment spec"
  "$PYTHON_BIN" tools/build_e379_glc_qaware_entropy_margin_deployment_spec.py \
    --output-prefix experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec
fi

for seed in $SEEDS; do
  prefix="${OUT_ROOT}/glc_hcg_qaware_clicprof_seed${seed}_q0123_k4_p01_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
  wandb_args=()
  if [[ "$WANDB_ENABLED" != "0" ]]; then
    wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
    wandb_args+=(--wandb-name "glc_hcg_qaware_clicprof_seed${seed}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  echo "[run] CLIC professional seed=${seed} -> ${prefix}"
  "$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
    --train-dir "$TRAIN_DIR" \
    --eval-dir "$CLIC_PRO_DIR" \
    --ckpt-path "$CKPT_PATH" \
    --output-prefix "$prefix" \
    --device cuda:0 \
    --q-indexes 0 1 2 3 \
    --padding-size 64 \
    --train-crop-size 256 \
    --eval-crop-size 0 \
    --train-start-index "$TRAIN_START_INDEX" \
    --eval-start-index 0 \
    --train-limit "$TRAIN_LIMIT" \
    --train-batch-per-step "$TRAIN_BATCH_PER_STEP" \
    --eval-limit "$EVAL_LIMIT" \
    --group-size 16 \
    --active-groups 1 7 10 15 \
    --active-parts 0 1 \
    --scope part_group \
    --k 4 \
    --stages 1 \
    --kmeans-iters "$KMEANS_ITERS" \
    --max-train-vectors "$MAX_TRAIN_VECTORS" \
    --max-rate-vectors "$MAX_RATE_VECTORS" \
    --steps "$STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --lr-codebook 5e-4 \
    --lr-controller 2e-4 \
    --mse-weight 0.00 \
    --l1-weight "$L1_WEIGHT" \
    --lpips-weight 0.30 \
    --dists-weight 1.00 \
    --branch-image-weight "$BRANCH_IMAGE_WEIGHT" \
    --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
    --glc-code-weight "$GLC_CODE_WEIGHT" \
    --soft-index-weight 0.005 \
    --soft-index-target 2.0 \
    --soft-index-temp 0.05 \
    --gate-rate-weight 1.0 \
    --gate-l1-weight 0.01 \
    --emit-replacement-rows \
    --replacement-cap-dbpp 0.0035 \
    --replacement-cap-dbpp-values 0.0030 0.0040 \
    --replacement-signal-bits 1 8 \
    --qaware-controller-json "$E379_JSON" \
    --qaware-policy-modes q-aware global \
    "${wandb_args[@]}" \
    $EXTRA_ARGS \
    --seed "$seed" \
    --lpips-net "$LPIPS_NET"
done

echo "[done] outputs under ${OUT_ROOT}"
