#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python train.py --config configs/scalar_baseline.yaml

