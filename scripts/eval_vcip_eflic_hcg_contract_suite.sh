#!/usr/bin/env bash
set -euo pipefail

# Paper-facing EF-LIC/HCG-RVQ codec-contract evaluation.
#
# This wrapper deliberately evaluates through the EF-LIC codec loop:
#   compress -> fixed-length pack_inds -> unpack_inds -> decompress
# The generated rows include bpp, delta_bpp, payload equality/length checks,
# forward/decode max error, MS-SSIM, LPIPS, and DISTS.
#
# EF-LIC official inference code uses LPIPS(net="vgg") in test.py, so VGG is
# the default for EF-LIC paper comparison. Set LPIPS_NET=alex for auxiliary
# HCG-RVQ cross-model diagnostics.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/workspace/HCG-RVQ:/workspace/HCG-RVQ/third_party/EF-LIC:${PYTHONPATH:-}"

cd /workspace/HCG-RVQ

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
IMAGE_DIR="${IMAGE_DIR:-/dpl/clic/professional/test}"
CKPT_PATH="${CKPT_PATH:-/workspace/HCG-RVQ/third_party/EF-LIC/ckpt/checkpoint.pth.tar}"
CONTROLLER_STATE="${CONTROLLER_STATE:-/workspace/HCG-RVQ/experiments/analysis/e347_eflic_codec_gain_controller_train_balanced_kodak24_t16_e8_s256.pth}"
OUT_ROOT="${OUT_ROOT:-/workspace/HCG-RVQ/experiments/analysis/e456_eflic_hcg_contract_suite_$(date +%Y%m%d_%H%M%S)}"

FORCE_INDS="${FORCE_INDS:-0 1 2 3 4}"
RISKS="${RISKS:--0.06 -0.08 -0.10}"
MAX_IMAGES="${MAX_IMAGES:-250}"
START_INDEX="${START_INDEX:-0}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.50}"
RISK_TEMPERATURE="${RISK_TEMPERATURE:-1.0}"
MAX_ALPHA="${MAX_ALPHA:-0.02}"
DIRECTION_SOURCE="${DIRECTION_SOURCE:-fixed}"
ACTIVE_SLICES="${ACTIVE_SLICES:-all}"
LPIPS_NET="${LPIPS_NET:-vgg}"
MODES="${MODES:-force_zero trained_hard}"

mkdir -p "$OUT_ROOT"

risk_tag() {
  local value="$1"
  local sign="p"
  local digits
  if [[ "$value" == -* ]]; then
    sign="m"
    value="${value#-}"
  fi
  digits="$(printf '%s' "$value" | tr -d '.')"
  printf 'risk%s%s' "$sign" "$digits"
}

cat > "${OUT_ROOT}/run_config.txt" <<EOF
IMAGE_DIR=$IMAGE_DIR
CKPT_PATH=$CKPT_PATH
CONTROLLER_STATE=$CONTROLLER_STATE
FORCE_INDS=$FORCE_INDS
RISKS=$RISKS
MAX_IMAGES=$MAX_IMAGES
START_INDEX=$START_INDEX
ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD
RISK_TEMPERATURE=$RISK_TEMPERATURE
MAX_ALPHA=$MAX_ALPHA
DIRECTION_SOURCE=$DIRECTION_SOURCE
ACTIVE_SLICES=$ACTIVE_SLICES
LPIPS_NET=$LPIPS_NET
MODES=$MODES
EOF

echo "[suite] output: $OUT_ROOT"
echo "[suite] image_dir: $IMAGE_DIR"
echo "[suite] lpips_net: $LPIPS_NET"

for force_ind in $FORCE_INDS; do
  for risk in $RISKS; do
    tag="$(risk_tag "$risk")"
    prefix="${OUT_ROOT}/eflic_hcg_${tag}_force${force_ind}_${LPIPS_NET}"
    echo "[eval] force=${force_ind} risk=${risk} out=${prefix}"
    "$PYTHON_BIN" tools/run_e295_eflic_hcg_branch_controller_integration_smoke.py \
      --image-dir "$IMAGE_DIR" \
      --ckpt-path "$CKPT_PATH" \
      --controller-state "$CONTROLLER_STATE" \
      --output-prefix "$prefix" \
      --device cuda:0 \
      --force-ind "$force_ind" \
      --direction-source "$DIRECTION_SOURCE" \
      --modes $MODES \
      --start-index "$START_INDEX" \
      --max-images "$MAX_IMAGES" \
      --max-alpha "$MAX_ALPHA" \
      --active-threshold "$ACTIVE_THRESHOLD" \
      --max-risk "$risk" \
      --risk-temperature "$RISK_TEMPERATURE" \
      --active-slices "$ACTIVE_SLICES" \
      --compute-perceptual \
      --lpips-net "$LPIPS_NET"

    "$PYTHON_BIN" tools/analyze_e350_eflic_perceptual_protocol.py \
      --csv "${prefix}.csv" \
      --output-prefix "${prefix}_perceptual" \
      --lpips-weight 3.0
  done
done

cat <<EOF
[done] EF-LIC HCG contract suite finished.

Primary files:
  ${OUT_ROOT}/run_config.txt
  ${OUT_ROOT}/*.md
  ${OUT_ROOT}/*.csv

Paper-safe contract checks to require before using a row:
  max_decode_diff == 0
  nonfinite_rows == 0
  payload_len_equal_frac == 1
  max_abs_delta_bpp == 0 for fixed-payload HCG geometry claims

EF-LIC paper comparison should use LPIPS_NET=vgg.
Auxiliary cross-model diagnostics can re-run with LPIPS_NET=alex.
EOF
