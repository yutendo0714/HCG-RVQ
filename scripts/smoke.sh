#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python train.py --config configs/tiny_smoke.yaml --device cpu
