#!/usr/bin/env bash
set -euo pipefail

# E444 / VCIP strict-deployable follow-up after E443.
# E443 showed that index-entropy selection can reduce exact replacement bpp,
# but the current branch hurts LPIPS/DISTS/MS-SSIM when used as a true
# replacement (g=1).  This run trains the branch directly in the deployable
# replacement regime instead of optimizing a small soft/fixed-gain blend.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/GLC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-/dpl/openimages/open-images-v6/train/data}"
CLIC_PRO_DIR="${CLIC_PRO_DIR:-/dpl/clic/professional/test}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/GLC/checkpoints/GLC_image.pth.tar}"
INIT_CKPT="${INIT_CKPT:-/workspace/HCG-RVQ/experiments/analysis/e433_glc_hcg_q3_fixedgain_dists_from_e393_20260614_235151/glc_hcg_q3_fixedgain_dists_seed1234_steps900_batch8_step0900.pt}"
E379_JSON="${E379_JSON:-/workspace/HCG-RVQ/experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e444_glc_hcg_q3_strict_replacement_from_e433_$(date +%Y%m%d_%H%M%S)}"

SEED="${SEED:-1234}"
STEPS="${STEPS:-1200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-300}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-16384}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8192}"
TRAIN_BATCH_PER_STEP="${TRAIN_BATCH_PER_STEP:-8}"
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-120000}"
MAX_RATE_VECTORS="${MAX_RATE_VECTORS:-4096}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.08}"
ACTIVE_GROUPS="${ACTIVE_GROUPS:-1 7 10 15}"
ACTIVE_PARTS="${ACTIVE_PARTS:-0 1}"

# Direct replacement objective: mixed == branch.  This keeps the experiment
# honest for selected replacement accounting and symbol round-trip evaluation.
TRAIN_FIXED_MIX_GATE="${TRAIN_FIXED_MIX_GATE:-1.0}"
TRAIN_FIXED_MIX_LOSS_SCALE="${TRAIN_FIXED_MIX_LOSS_SCALE:-1.0}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.55}"
DISTS_WEIGHT="${DISTS_WEIGHT:-1.75}"
GLC_FEATURE_WEIGHT="${GLC_FEATURE_WEIGHT:-0.035}"
GLC_CODE_WEIGHT="${GLC_CODE_WEIGHT:-0.001}"
SOFT_INDEX_WEIGHT="${SOFT_INDEX_WEIGHT:-0.03}"
SOFT_INDEX_TARGET="${SOFT_INDEX_TARGET:-1.60}"
SOFT_INDEX_FLOOR_TARGET="${SOFT_INDEX_FLOOR_TARGET:-1.05}"
SOFT_INDEX_FLOOR_WEIGHT="${SOFT_INDEX_FLOOR_WEIGHT:-0.75}"
FIXED_GATE_VALUES="${FIXED_GATE_VALUES:-1.0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-HCG-RVQ}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "[error] missing init checkpoint: $INIT_CKPT" >&2
  exit 2
fi

prefix="${OUT_ROOT}/glc_hcg_q3_strict_replacement_seed${SEED}_steps${STEPS}_batch${TRAIN_BATCH_PER_STEP}"
wandb_args=()
if [[ "$WANDB_ENABLED" != "0" ]]; then
  wandb_args+=(--wandb-enabled --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  wandb_args+=(--wandb-name "e444_q3_strict_replacement_seed${SEED}_steps${STEPS}")
  if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

echo "[run] seed=${SEED} init=${INIT_CKPT} out=${prefix}"
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
  --lr-codebook 5e-5 \
  --lr-controller 1e-5 \
  --mse-weight 0.00 \
  --l1-weight 0.00 \
  --lpips-weight "$LPIPS_WEIGHT" \
  --dists-weight "$DISTS_WEIGHT" \
  --branch-image-weight 0.00 \
  --teacher-soft-weight 0.00 \
  --glc-feature-weight "$GLC_FEATURE_WEIGHT" \
  --glc-code-weight "$GLC_CODE_WEIGHT" \
  --soft-index-weight "$SOFT_INDEX_WEIGHT" \
  --soft-index-target "$SOFT_INDEX_TARGET" \
  --soft-index-floor-target "$SOFT_INDEX_FLOOR_TARGET" \
  --soft-index-floor-weight "$SOFT_INDEX_FLOOR_WEIGHT" \
  --soft-index-temp 0.05 \
  --gate-rate-weight 0.00 \
  --gate-l1-weight 0.00 \
  --active-threshold "$ACTIVE_THRESHOLD" \
  --context-from-scalar \
  --train-fixed-mix-gate "$TRAIN_FIXED_MIX_GATE" \
  --train-fixed-mix-loss-scale "$TRAIN_FIXED_MIX_LOSS_SCALE" \
  --emit-replacement-rows \
  --replacement-cap-dbpp 0.0015 \
  --replacement-cap-dbpp-values 0.0000 0.0010 0.0020 0.0030 \
  --replacement-signal-bits 1 8 \
  --fixed-gate-values $FIXED_GATE_VALUES \
  --qaware-controller-json "$E379_JSON" \
  --qaware-policy-modes q-aware global \
  --cache-images-on-cpu \
  "${wandb_args[@]}" \
  $EXTRA_ARGS \
  --seed "$SEED"

echo "[done] output under ${OUT_ROOT}"
echo "[next eval CLIC64]"
echo "CUDA_VISIBLE_DEVICES=0 ACTIVE_THRESHOLD=${ACTIVE_THRESHOLD} FID_PATCH_SIZE=-1 Q_INDEXES=\"3\" EXPORT_LABELS=\"base all_on replacement_hard replacement_soft\" REPLACEMENT_SIGNAL_BITS=\"1\" LEARNED_HARD_MIN_Q=\"3\" bash scripts/eval_vcip_glc_hcg_deployable_clic64.sh CHECKPOINT.pt OUT_DIR"
echo "[next official hard threshold CLIC250]"
echo "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FID_PATCH_SIZE=-1 FIXED_GATES=\"1.0\" THRESHOLD_VALUES=\"1.55 1.58032 1.60 1.62\" QUALITY_LABELS=\"base th_indexentropymean_ge1p55_g1_replacement_sig1b th_indexentropymean_ge1p58032_g1_replacement_sig1b th_indexentropymean_ge1p6_g1_replacement_sig1b th_indexentropymean_ge1p62_g1_replacement_sig1b\" bash scripts/eval_vcip_glc_hcg_e433_indexentropy_fine_sweep_clic250.sh CHECKPOINT.pt OUT_DIR"
