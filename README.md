# HCG-RVQ

Hyperprior-Conditioned Geometric Residual Vector Quantization for learned image compression.

This repository is organized as an independent research project that uses CompressAI as the LIC foundation while keeping HCG-RVQ modules, configs, logs, and experiment notes in a clean local structure.

## Layout

```text
hcg_rvq/          Python package: models, quantizers, entropy models, data, metrics
configs/          YAML experiment configs
scripts/          Small launch helpers
docs/             Reading notes, progress logs, implementation notes
experiments/      Checkpoints and local run artifacts
results/          Evaluation outputs, plots, tables
third_party/      Optional cloned reference repositories
```

## First Goals

1. MeanScaleHyperprior scalar baseline.
2. MeanScaleHyperprior + Global RVQ.
3. HCS-RVQ: hyperprior-conditioned shift/scale RVQ.
4. HCS-RVQ + hyper-categorical index prior.
5. HCG-RVQ-H: HCS-RVQ with Householder geometry.

The first core ablation is:

```text
Global RVQ vs HCS-RVQ vs HCG-RVQ-H
```

under the same backbone, entropy model, data, and training schedule.

## Environment

Create and use the project virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Training logs use Weights & Biases when enabled in the config.

