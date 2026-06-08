#!/usr/bin/env bash
set -euo pipefail

# Long-run GLC/HCG-RVQ branch experiment.
#
# This script keeps the official pretrained GLC image model fixed and trains the
# HCG/RVQ branch plus reliability controller in the codec loop.  It is the
# paper-facing promotion run after the E379 q-aware entropy-margin gate, not a
# scratch reimplementation of the official GLC training recipe.
#
# Typical tmux use:
#   cd /workspace/HCG-RVQ
#   CLIC_PRO_DIR=/path/to/clic/professional/valid bash scripts/run_glc_qaware_longrun.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
KODAK_DIR="${KODAK_DIR:-/workspace/HCG-RVQ/experiments/data/kodak24}"
CLIC_PRO_DIR="${CLIC_PRO_DIR:-}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
E379_JSON="${E379_JSON:-/workspace/HCG-RVQ/experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/glc_qaware_longrun_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-1234 2345 3456}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-8192}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1024}"
EVAL_LIMIT="${EVAL_LIMIT:-100000}"
STEPS="${STEPS:-800}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-60000}"
MAX_RATE_VECTORS="${MAX_RATE_VECTORS:-8192}"
KMEANS_ITERS="${KMEANS_ITERS:-12}"
LPIPS_NET="${LPIPS_NET:-alex}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-HCG-RVQ}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$E379_JSON" ]]; then
  echo "[setup] building E379 q-aware entropy-margin deployment spec"
  "$PYTHON_BIN" tools/build_e379_glc_qaware_entropy_margin_deployment_spec.py \
    --output-prefix experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec
fi

eval_specs=("kodak24:$KODAK_DIR")
if [[ -n "$CLIC_PRO_DIR" ]]; then
  eval_specs+=("clicprof:$CLIC_PRO_DIR")
else
  echo "[warn] CLIC_PRO_DIR is empty; running Kodak24 only. Set CLIC_PRO_DIR=/path/to/CLIC/professional/valid for the paper-facing CLIC run."
fi

for seed in $SEEDS; do
  for spec in "${eval_specs[@]}"; do
    label="${spec%%:*}"
    eval_dir="${spec#*:}"
    if [[ ! -d "$eval_dir" ]]; then
      echo "[skip] missing eval dir for ${label}: ${eval_dir}"
      continue
    fi

    prefix="${OUT_ROOT}/e379_qaware_${label}_seed${seed}_q0123_k4_p01_steps${STEPS}"
    wandb_args=()
    if [[ "$WANDB_ENABLED" != "0" ]]; then
      wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
      wandb_args+=(--wandb-name "glc_qaware_${label}_seed${seed}_q0123_k4_p01_steps${STEPS}")
      if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_args+=(--wandb-entity "$WANDB_ENTITY")
      fi
    fi
    echo "[run] ${label} seed=${seed} -> ${prefix}"
    "$PYTHON_BIN" tools/run_e263_glc_fallback_gate_codec_loop_pilot.py \
      --train-dir "$TRAIN_DIR" \
      --eval-dir "$eval_dir" \
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
      --lpips-weight 0.30 \
      --dists-weight 1.00 \
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
      --seed "$seed" \
      --lpips-net "$LPIPS_NET"
  done
done

echo "[done] outputs under ${OUT_ROOT}"
