#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python train.py --config configs/hcg_rvq_h.yaml

