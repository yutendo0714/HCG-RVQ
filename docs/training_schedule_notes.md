# Training Schedule Notes

Date: 2026-05-29

## What the Literature/Repos Suggest

- CompressAI model zoo describes training LIC models on 256x256 patches, batch size 16 or 32, initial learning rate `1e-4`, for roughly `1-2M` optimization steps, with `ReduceLROnPlateau` after validation plateaus. Some zoo models are also described as using 4-5M total steps for full SOTA reproduction.
- CompressAI example training commonly exposes `--epochs 300`, `--batch-size 16`, and `--learning-rate 1e-4`; this is an example pipeline, not a hard rule for every dataset size.
- DCAE README uses `CUDA_VISIBLE_DEVICES=0`, `--epochs 50`, `--lr_epoch 46`, `--batch-size 8`, `lr=1e-4` when training/fine-tuning from a checkpoint.
- DCAE paper reports full models trained for 80 epochs at `1e-4` plus 20 epochs at `1e-5` for about 1.875M iterations, and ablations trained for 20 epochs plus 5 low-LR epochs with batch size 8.
- Recent strong LIC systems usually report settings in steps or use large datasets where an epoch is expensive. Therefore, epochs cannot be compared without dataset size.

## Decision for HCG-RVQ

Use two regimes:

1. Pilot/ablation sanity runs:
   - `max_steps`: 500-5000
   - dataset subset: 4096-50000 images
   - purpose: verify rate accounting, codebook usage, wandb logging, and relative trends.

2. Full comparison runs:
   - initial default: 50 epochs, LR drop at epoch 46, batch size 8, patch size 256, LR `1e-4 -> 1e-5`.
   - reason: matches the public DCAE training command and is less wasteful than 100 epochs for the first HCG-RVQ validation.

If the local OpenImages subset is much smaller than 300k images, step count should be monitored. For paper-grade models, target optimization budget should be closer to the CompressAI/DCAE step counts after the method is validated.

## Current Config Policy

- Full configs use `/dpl/openimages/open-images-v6/train/data`.
- Scripts force `CUDA_VISIBLE_DEVICES=0`.
- Pilot configs are separate and use `max_steps` to avoid accidental multi-day runs.
