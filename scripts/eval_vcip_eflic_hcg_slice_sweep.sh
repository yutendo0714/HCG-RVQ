#!/usr/bin/env bash
set -euo pipefail

# Short-cycle EF-LIC/HCG-RVQ slice/action sweep.
#
# Use this before long evaluation/training. It runs the existing E313 codec-loop
# slice isolation evaluator with perceptual metrics and exact payload/decode
# contract checks. Rows are design diagnostics; promote only if the same policy
# survives the contract suite on held-out CLIC/Kodak.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/EF-LIC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
IMAGE_DIR="${IMAGE_DIR:-/dpl/clic/professional/test}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/EF-LIC/ckpt/checkpoint.pth.tar}"
CONTROLLER_STATE="${CONTROLLER_STATE:-/workspace/HCG-RVQ/experiments/analysis/e347_eflic_codec_gain_controller_train_balanced_kodak24_t16_e8_s256.pth}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e457_eflic_hcg_slice_sweep_$(date +%Y%m%d_%H%M%S)}"

FORCE_IND="${FORCE_IND:-0}"
MAX_IMAGES="${MAX_IMAGES:-64}"
START_INDEX="${START_INDEX:-0}"
MODE="${MODE:-trained_hard}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.50}"
MAX_RISK="${MAX_RISK:--0.06}"
RISK_TEMPERATURE="${RISK_TEMPERATURE:-1.0}"
MAX_ALPHA="${MAX_ALPHA:-0.02}"
DIRECTION_SOURCE="${DIRECTION_SOURCE:-fixed}"
LPIPS_NET="${LPIPS_NET:-vgg}"
SLICE_SETS="${SLICE_SETS:-all 0 1 2 3 0,1 0,2 0,3 1,2 1,3 2,3 0,1,3 0,2,3 0,1,2 1,2,3}"

mkdir -p "$OUT_ROOT"
prefix="${OUT_ROOT}/eflic_hcg_slice_sweep_force${FORCE_IND}_${LPIPS_NET}"

echo "[sweep] output: $prefix"
echo "[sweep] image_dir: $IMAGE_DIR"
echo "[sweep] lpips_net: $LPIPS_NET"

"$PYTHON_BIN" tools/run_e313_eflic_slice_isolation_sweep.py \
  --image-dir "$IMAGE_DIR" \
  --ckpt-path "$CKPT_PATH" \
  --controller-state "$CONTROLLER_STATE" \
  --output-prefix "$prefix" \
  --device cuda:0 \
  --force-ind "$FORCE_IND" \
  --direction-source "$DIRECTION_SOURCE" \
  --mode "$MODE" \
  --active-threshold "$ACTIVE_THRESHOLD" \
  --max-risk "$MAX_RISK" \
  --risk-temperature "$RISK_TEMPERATURE" \
  --max-alpha "$MAX_ALPHA" \
  --start-index "$START_INDEX" \
  --max-images "$MAX_IMAGES" \
  --compute-perceptual \
  --lpips-net "$LPIPS_NET" \
  --slice-sets $SLICE_SETS

cat <<EOF
[done] EF-LIC HCG slice sweep finished.

Read:
  ${prefix}.md
  ${prefix}.by_set.csv
  ${prefix}.by_image.csv
  ${prefix}.rows.csv

Promotion rule:
  contract_ok_frac == 1.0
  mean_perceptual_score < 0
  worst_perceptual_score <= 0 for safe policies
  then re-run scripts/eval_vcip_eflic_hcg_contract_suite.sh on CLIC250/Kodak24.
EOF
