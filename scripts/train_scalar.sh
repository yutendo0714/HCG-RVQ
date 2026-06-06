#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/scalar_baseline.yaml --device cuda
