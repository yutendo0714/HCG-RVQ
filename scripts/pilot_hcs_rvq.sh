#!/usr/bin/env bash
set -euo pipefail

INIT_ARGS=()
if [[ $# -gt 0 ]]; then
  INIT_ARGS=(--init-model "$1")
fi

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/pilot_hcs_rvq.yaml --device cuda "${INIT_ARGS[@]}"
