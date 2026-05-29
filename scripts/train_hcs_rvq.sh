#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python train.py --config configs/hcs_rvq.yaml

