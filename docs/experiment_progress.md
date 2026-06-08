# Experiment Progress

Date: 2026-05-29

## E000: Project Bootstrap

Status: Done

Notes:

- Created a clean top-level research project structure instead of keeping all work under `proposed/`.
- Preserved the pre-implementation literature and implementation reading notes in `docs/research_reading_notes.md`.
- Initial implementation target is a CompressAI-style MeanScaleHyperprior backbone with RVQ/HCS/HCG variants.

Next:

- Install dependencies into `.venv`.
- Run a CPU smoke test on a random tensor.
- Start scalar baseline and tiny RVQ pilot configs.


## E001: Environment and Smoke Test

Status: Done

Results:

- Created `.venv` and installed project dependencies.
- Corrected PyTorch to `2.7.1+cu128` so CUDA works with the local CUDA 12.8 driver.
- Verified imports: torch, CompressAI, wandb, numpy.
- Verified CUDA visibility on NVIDIA GeForce RTX 3090.
- Ran random forward/loss checks for scalar, global RVQ, HCS-RVQ, and HCG-RVQ-H variants.
- Ran CPU tiny smoke training on Kodak for 1 epoch and saved `experiments/tiny_smoke/checkpoint_latest.pth.tar`.
- Ran CPU eval smoke on 2 Kodak images: bpp 0.178448, PSNR 9.322829, MS-SSIM 0.294151.

Next:

- Run the first real scalar baseline with wandb enabled.
- Then run `global_rvq`, `hcs_rvq`, and `hcg_rvq_h` under the same tiny/pilot schedule.


## E002: Training Schedule and CUDA Policy

Status: Done

Results:

- Verified W&B login via `wandb login --verify`; account is `mayuyuto0714-waseda-university`.
- Confirmed DCAE public README uses `CUDA_VISIBLE_DEVICES=0`, `--epochs 50`, `--lr_epoch 46`, `--batch-size 8`, `lr=1e-4`.
- Added `max_steps`, LR milestones, checkpoint intervals, and LR logging to `train.py`.
- Updated full configs to 50 epochs with LR milestone at epoch 46.
- Added pilot configs with `max_steps=500` for scalar/global RVQ/HCS/HCG-H.
- Updated training scripts to force `CUDA_VISIBLE_DEVICES=0`.

Next:

- Run a short GPU+W&B pilot scalar run.
- Then run pilot Global RVQ, HCS-RVQ, and HCG-RVQ-H under the same budget.


## E003: GPU 0 and W&B Smoke Test

Status: Done

Results:

- Ran `configs/gpu_wandb_smoke.yaml` with `CUDA_VISIBLE_DEVICES=0` on GPU.
- W&B online sync succeeded.
- W&B run: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/injdd906
- Training command completed 5 steps and saved checkpoints under `experiments/gpu_wandb_smoke`.

Next:

- Run `scripts/pilot_scalar.sh` for the first 500-step scalar pilot.
- Evaluate pilot checkpoint on Kodak.
- Continue with pilot Global RVQ, HCS-RVQ, and HCG-RVQ-H using the same budget.


## E004: Pilot Scalar Baseline 500 Steps

Status: Done

Command:

- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/pilot_scalar_baseline.yaml --device cuda`

Results:

- W&B run: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/5zqijx4x
- Trained 500 steps on a 4096-image OpenImages subset with batch size 8.
- Checkpoint: `experiments/pilot_scalar_baseline_lambda0035/checkpoint_latest.pth.tar`
- Kodak eval over 24 images:
  - bpp: 0.283923
  - bpp_y: 0.058857
  - bpp_z: 0.225066
  - PSNR: 19.152838
  - MS-SSIM: 0.625586

Notes:

- This is only a pilot sanity point, not a converged baseline.
- The W&B summary logs every 20 steps, so the displayed final global step can show the last logged step rather than the final checkpoint step.

Next:

- Run `pilot_global_rvq`, evaluate Kodak, and compare rate accounting/codebook statistics against scalar.

## E005: RVQ Rate and Stability Diagnosis

Status: Done

Problem found:

- The first Global RVQ pilot used `group_size=32`, `num_stages=2`, `codebook_size=256` without an index prior.
- That setting has fixed-length index cost `10 groups * 2 stages * 8 bits / 16^2 = 0.625 bpp`, already far above the scalar pilot total rate of `0.283923 bpp`.
- A normalized Global RVQ debug run still showed commit-loss explosion, so the issue was not only rate accounting.

Decision:

- Move the first RVQ comparison to `group_size=64`, `num_stages=1`, `codebook_size=128`.
- This gives Global RVQ fixed-length `R_y = 5 groups * 1 stage * 7 bits / 16^2 = 0.136719 bpp`.
- Use scalar-checkpoint latent codebook initialization and freeze warmup before full end-to-end unfreezing.

Notes:

- The failed end-to-end Global RVQ run with low-rate codebooks still degraded from the initialized checkpoint.
- W&B run: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/ghnz4pum
- After 500 steps, commit loss rose to about `12443`, and Kodak dropped to `0.356681 bpp / 16.706942 dB / MS-SSIM 0.604188`.

## E006: Scalar-Checkpoint RVQ Codebook Initialization

Status: Done

Changes:

- Added `tools/init_rvq_codebook.py`.
- Added partial model-weight initialization through `--init-model` in `train.py`.
- Added conditioning-head identity initialization so HCS/HCG start from approximately `mu=0`, `scale=1`, and identity Householder behavior.
- Added matching-shape checkpoint loading helper in `hcg_rvq/utils/checkpoint.py`.

Initialized checkpoints:

- `experiments/init/pilot_global_rvq_g64_l1_k128_from_scalar.pth.tar`
- `experiments/init/pilot_hcs_rvq_g64_l1_k128_from_scalar.pth.tar`
- `experiments/init/pilot_hcg_rvq_h_g64_l1_k128_from_scalar.pth.tar`

Initialization diagnostics:

- Global RVQ uses sampled scalar-latent channel mean/std as fixed global normalization.
- Global normalized sampled latent quantization MSE: `0.493109`.
- HCS/HCG sampled latent quantization MSE with identity conditioning: `0.016803`.

Initial Global RVQ checkpoint Kodak eval:

- bpp: `0.361784`
- bpp_y: `0.136719`
- bpp_z: `0.225066`
- PSNR: `19.802803`
- MS-SSIM: `0.675989`

## E007: Global RVQ Freeze Warmup Pilot

Status: Done

Config:

- `configs/pilot_global_rvq_frozen.yaml`
- Init checkpoint: `experiments/init/pilot_global_rvq_g64_l1_k128_from_scalar.pth.tar`
- Frozen prefixes: `g_a`, `h_a`, `h_s`, `entropy_bottleneck`, `global_mu`, `global_log_s`
- Trained decoder/RVQ only for 500 steps on a 4096-image OpenImages subset.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/qgxgkrt3

W&B summary:

- bpp_total: `0.36177`
- bpp_y: `0.13672`
- bpp_z: `0.22505`
- commit_loss: `0.55183`
- loss: `2.33374`

Kodak eval:

- bpp: `0.361784`
- bpp_y: `0.136719`
- bpp_z: `0.225066`
- PSNR: `21.047032`
- MS-SSIM: `0.723126`

Interpretation:

- Freezing the encoder/hyper path prevents the Global RVQ baseline from collapsing.
- Compared with the scalar 500-step pilot, this is higher-rate but substantially better quality: scalar was `0.283923 bpp / 19.152838 dB / MS-SSIM 0.625586`.

## E008: HCS-RVQ Freeze Warmup Pilot

Status: Done

Config:

- `configs/pilot_hcs_rvq_frozen.yaml`
- Init checkpoint: `experiments/init/pilot_hcs_rvq_g64_l1_k128_from_scalar.pth.tar`
- Frozen prefixes: `g_a`, `h_a`, `h_s`, `entropy_bottleneck`
- Trained decoder, RVQ, shift/scale heads, and index prior for 500 steps.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/0fmwt93u

W&B summary:

- bpp_total: `0.30410`
- bpp_y: `0.07894`
- bpp_z: `0.22516`
- commit_loss: `0.13935`
- loss: `2.45321`

Kodak eval:

- bpp: `0.316533`
- bpp_y: `0.091467`
- bpp_z: `0.225066`
- PSNR: `21.152914`
- MS-SSIM: `0.750836`

Interpretation:

- HCS-RVQ improves over Global RVQ freeze warmup at lower total rate.
- The main immediate gain is from hyper-conditioned index entropy and shift/scale adaptation.
- This is the first positive evidence for the required `Global RVQ vs HCS-RVQ` comparison.

## E009: HCG-RVQ-H Freeze Warmup Pilot

Status: Done

Config:

- `configs/pilot_hcg_rvq_h_frozen.yaml`
- Init checkpoint: `experiments/init/pilot_hcg_rvq_h_g64_l1_k128_from_scalar.pth.tar`
- Frozen prefixes: `g_a`, `h_a`, `h_s`, `entropy_bottleneck`
- Trained decoder, RVQ, shift/scale heads, Householder head, and index prior for 500 steps.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/ukfotdyi

W&B summary:

- bpp_total: `0.30480`
- bpp_y: `0.07964`
- bpp_z: `0.22516`
- commit_loss: `0.14980`
- loss: `2.34187`

Kodak eval:

- bpp: `0.317002`
- bpp_y: `0.091937`
- bpp_z: `0.225066`
- PSNR: `21.252954`
- MS-SSIM: `0.752188`

Interpretation:

- HCG-RVQ-H is slightly better than HCS-RVQ in this pilot: about `+0.10 dB` PSNR and `+0.00135` MS-SSIM at nearly the same rate.
- This is promising but still weak evidence; Householder should be tested with more steps, multiple seeds, and a no-geometry matched control after warmup.
- The next important action is a controlled freeze-warmup-to-unfreeze schedule rather than direct end-to-end training from step 0.

## E010: Staged `h_s` Unfreeze Diagnosis

Status: Done

Changes:

- Added config-driven `train.freeze_schedule` support in `train.py`.
- Added optimizer prefix LR multipliers via `train.param_lr_multipliers`.
- Added RVQ conditioning diagnostics to W&B: `mu_q_abs_mean`, `mu_q_std`, `s_q_std`, `y_norm_abs_mean`, and raw Householder vector magnitude.
- Added staged configs:
  - `configs/pilot_hcs_rvq_staged_hs.yaml`
  - `configs/pilot_hcg_rvq_h_staged_hs.yaml`
  - `configs/pilot_hcs_rvq_staged_hs_slow.yaml`
  - `configs/pilot_hcg_rvq_h_staged_hs_slow.yaml`

Schedule tested:

- Steps `[0, 500)`: freeze `g_a`, `h_a`, `h_s`, `entropy_bottleneck`.
- Steps `[500, 1000)`: freeze `g_a`, `h_a`, `entropy_bottleneck`; unfreeze `h_s`.

Results:

- Normal-LR HCG-H staged unfreeze:
  - Config: `configs/pilot_hcg_rvq_h_staged_hs.yaml`
  - W&B: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/93ob15nq
  - Kodak latest: `0.332727 bpp / 16.559856 dB / MS-SSIM 0.442196`
  - Commit loss rose to `55.87057` in the W&B summary.
- Low-LR HCG-H staged unfreeze with `h_s` multiplier `0.02`:
  - Config: `configs/pilot_hcg_rvq_h_staged_hs_slow.yaml`
  - W&B: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/kyfxs55i
  - Kodak latest: `0.330909 bpp / 18.108051 dB / MS-SSIM 0.654894`
  - Commit loss improved versus normal-LR unfreeze but remained too high at `4.68124`.

Interpretation:

- Directly unfreezing `h_s` changes the conditional quantizer frame too aggressively and destroys latent/codebook alignment.
- Even a very small `h_s` LR is not sufficient by itself.
- Before any `h_s` adaptation becomes part of the main path, add an explicit conditioning-drift regularizer or anchor loss against the initialized `mu_q`, `s_q`, and Householder frame.
- For the next paper-oriented evidence, keep `h_s` frozen and use validation-selected checkpoints.

## E011: Frozen 1000-Step Extension and Checkpoint Selection

Status: Done

Configs:

- `configs/pilot_hcg_rvq_h_frozen_1000.yaml`
- `configs/pilot_hcs_rvq_frozen_1000.yaml`

HCG-RVQ-H frozen 1000:

- W&B: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/b0fblk7p
- `checkpoint_step_500` Kodak: `0.317023 bpp / 21.236728 dB / MS-SSIM 0.751724`
- `checkpoint_latest` Kodak: `0.322877 bpp / 20.833721 dB / MS-SSIM 0.735942`

HCS-RVQ frozen 1000:

- W&B: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/kos0jc0u
- `checkpoint_step_500` Kodak: `0.316531 bpp / 21.150005 dB / MS-SSIM 0.750917`
- `checkpoint_latest` Kodak: `0.321632 bpp / 21.041694 dB / MS-SSIM 0.738647`

Interpretation:

- Both HCS and HCG-H peak around 500 steps under the current pilot data/budget and then lose Kodak generalization.
- At matched 500-step checkpoints, HCG-H remains slightly ahead of HCS: about `+0.087 dB` PSNR and `+0.000807` MS-SSIM at `+0.000492 bpp`.
- At 1000 steps, HCS is better than HCG-H, so the current Householder geometry is promising but not yet robust enough for a final claim.

Next:

- Completed the matched no-geometry control in E012.
- Completed checkpoint-sweep evaluation tooling in E013.
- Run at least three seeds for the frozen 500-step HCS/no-transform/HCG-H comparison before treating the HCG-H gain as evidence.
- Add a Householder-strength regularizer and keep `h_s` frozen until conditioning-drift regularization is implemented.

## E012: No-Geometry Matched Control

Status: Done

Changes:

- Added `hcg_rvq_h_no_transform` as a matched control.
- The control instantiates the Householder head and logs the same Householder-vector diagnostics, but does not apply the orthogonal transform before/after RVQ.
- Added `configs/pilot_hcg_rvq_h_no_transform_frozen.yaml`.

Config:

- Init checkpoint: `experiments/init/pilot_hcg_rvq_h_g64_l1_k128_from_scalar.pth.tar`
- Frozen prefixes: `g_a`, `h_a`, `h_s`, `entropy_bottleneck`
- Trained decoder, RVQ, shift/scale heads, and index prior for 500 steps; the Householder head is instantiated for parameter/control matching but is not on the transform loss path.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/augdz4ei

W&B summary:

- bpp_total: `0.30409`
- bpp_y: `0.07893`
- bpp_z: `0.22516`
- commit_loss: `0.13932`
- loss: `2.45224`

Kodak eval:

- bpp: `0.316533`
- bpp_y: `0.091467`
- bpp_z: `0.225066`
- PSNR: `21.153417`
- MS-SSIM: `0.750906`

Interpretation:

- The no-transform control is numerically almost identical to HCS-RVQ at the same training budget.
- This confirms that the large gain from Global RVQ to HCS/HCG comes from hyper-conditioned shift/scale plus index entropy, not from merely adding the Householder head.
- The small HCG-H 500-step advantage over HCS/no-transform remains possible evidence for geometry, but it is not yet strong enough for a paper claim without multi-seed validation and a geometry-strength/stability ablation.

## E013: Checkpoint-Sweep Evaluation Tool

Status: Done

Changes:

- Added `tools/evaluate_checkpoints.py` to evaluate every saved checkpoint in an experiment directory.
- The tool reports `loss`, validation-style `rd_score`, `bpp`, `bpp_y`, `bpp_z`, `mse`, `PSNR`, and `MS-SSIM`.
- It writes CSV/JSON summaries and prints the best checkpoint by `rd_score`, `loss`, `bpp`, `PSNR`, or `MS-SSIM`.
- Added Householder transform-strength diagnostics: `householder_delta_abs_mean` and `householder_delta_rms`.

Commands run:

- `tools/evaluate_checkpoints.py` on `experiments/pilot_hcg_rvq_h_frozen_1000_g64_l1_k128_lambda0035`
- `tools/evaluate_checkpoints.py` on `experiments/pilot_hcs_rvq_frozen_1000_g64_l1_k128_lambda0035`
- `tools/evaluate_checkpoints.py` on `experiments/pilot_hcg_rvq_h_no_transform_frozen_g64_l1_k128_lambda0035`

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_frozen_1000_kodak.csv`
- `experiments/analysis/pilot_hcs_rvq_frozen_1000_kodak.csv`
- `experiments/analysis/pilot_hcg_rvq_h_no_transform_frozen_kodak.csv`

Results:

- HCG-RVQ-H frozen 1000 best by RD score: `checkpoint_step_500.pth.tar`
  - `rd_score=2.170041`, `0.317023 bpp`, `21.236730 dB`, `MS-SSIM 0.751724`
- HCS-RVQ frozen 1000 best by RD score: `checkpoint_step_500.pth.tar`
  - `rd_score=2.198715`, `0.316531 bpp`, `21.150005 dB`, `MS-SSIM 0.750917`
- HCG-RVQ-H no-transform frozen 500 best by RD score: `checkpoint_step_500.pth.tar`
  - `rd_score=2.197728`, `0.316533 bpp`, `21.153418 dB`, `MS-SSIM 0.750906`

Interpretation:

- The earlier conclusion is now reproducible by script: `checkpoint_latest` is not the right model-selection policy for the current pilot schedule.
- For paper runs, select checkpoints using a validation split or validation set, then reserve Kodak for test reporting.

## E014: Retrospective Checkpoint Sweep

Status: Done

Purpose:

- Re-evaluate past experiments with the same checkpoint-sweep protocol instead of relying on `checkpoint_latest` or one-off eval notes.
- Cover scalar, failed Global RVQ, frozen Global RVQ, HCS, HCG-H, no-transform, and staged `h_s` unfreeze runs.

Outputs:

- `experiments/analysis/checkpoint_summary.csv`
- `experiments/analysis/checkpoint_best_summary.md`
- Per-run CSVs under `experiments/analysis/*_kodak.csv`

Key findings:

- Scalar, Global frozen, HCS frozen, HCG-H frozen, and no-transform all select the 500-step checkpoint or equivalent latest checkpoint.
- Direct Global RVQ remains bad even at its best checkpoint: `0.356681 bpp / 16.706942 dB / MS-SSIM 0.604188`.
- High-rate `g32_l2_k256` Global RVQ remains a failed setup: best Kodak is `0.872583 bpp / 17.240326 dB / MS-SSIM 0.547151`.
- Normal staged `h_s` unfreeze is healthy at step 500 but collapses by step 750: `21.255815 dB -> 16.322948 dB`.
- Slow staged `h_s` unfreeze delays but does not solve the drift: `21.159279 dB` at step 500, `20.491187 dB` at step 750, and `18.108026 dB` at step 1000.

Interpretation:

- The current paper path should use checkpoint selection systematically; `checkpoint_latest` is actively misleading for several runs.
- The `h_s` unfreeze failure is not a late-overfit artifact; it begins as soon as the conditional quantizer frame is allowed to move.


## E015: Intermediate Feature Distribution Analysis

Status: Done

Changes:

- Added `tools/feature_distribution_analysis.py`.
- The tool collects RD metrics, latent statistics, hyper-feature statistics, quantization error, RVQ usage, empirical index entropy, conditioning-head statistics, and Householder transform magnitude.

Outputs:

- `experiments/analysis/feature_summary.csv`
- `experiments/analysis/feature_summary.md`
- Per-checkpoint feature summaries under `experiments/analysis/feature_*.json` and `feature_*.csv`

Key findings:

- Direct Global RVQ failure is visible in latent scale: `y_std=14.439488`, `y_error_rms=14.360433`, and `rvq_latent_quant_mse=4588.053716`.
- Frozen Global RVQ restores latent scale (`y_std=0.170745`) and stable quantization (`rvq_latent_quant_mse=0.378932`) but still has high latent-index rate.
- HCS/no-transform/HCG-H sharply reduce latent quantization MSE versus Global frozen:
  - HCS: `0.068028`
  - no-transform: `0.068025`
  - HCG-H: `0.072757`
- HCS and no-transform have nearly identical feature distributions and Kodak quality, confirming the matched-control result.
- HCG-H improves image PSNR despite slightly higher latent quantization MSE, suggesting that the geometry changes the error direction rather than just reducing latent MSE.
- Normal `h_s` unfreeze causes frame explosion by step 750: `hyper_features_std=3.395880`, `householder_delta_rms=2.268818`, `rvq_latent_quant_mse=43.273949`, and `dead_code_ratio=0.403320`.
- Slow `h_s` unfreeze delays the same phenomenon: `householder_delta_rms` grows from `0.071743` to `0.243113` to `0.799914` over steps 500/750/1000.

Interpretation:

- The main stable contribution is HCS plus conditioned index entropy.
- Householder is not merely a parameter-count effect, but the current transform can become too strong and destabilize the codebook/decoder alignment.
- The next method improvement should be a conditioning/frame drift regularizer, not another naive `h_s` LR reduction.

## E016: Drift-Regularized Staged Config

Status: Done

Changes:

- Added optional conditioning/frame-strength penalties to `RateDistortionLoss`:
  - `rho_mu_q_abs` for `mu_q_abs_mean`
  - `rho_s_q_std` for `s_q_std`
  - `rho_householder_delta` for `householder_delta_rms`
- Added `configs/pilot_hcg_rvq_h_staged_hs_slow_reg.yaml` as the next staged `h_s` unfreeze candidate.

Config intent:

- Keep the previous slow `h_s` unfreeze schedule.
- Penalize conditioning drift and excessive Householder transform magnitude after `h_s` unfreezes.
- Use this as a targeted follow-up to the feature-distribution finding that collapse is associated with `hyper_features_std`, `householder_delta_rms`, and latent quantization MSE growth.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/g6hi0685

Checkpoint sweep on Kodak:

- `checkpoint_step_250`: `0.314544 bpp / 20.496148 dB / MS-SSIM 0.736678`
- `checkpoint_step_500`: `0.306870 bpp / 20.577316 dB / MS-SSIM 0.728218`
- `checkpoint_step_750`: `0.304812 bpp / 20.084959 dB / MS-SSIM 0.708893`
- `checkpoint_step_1000/latest`: `0.319786 bpp / 19.583592 dB / MS-SSIM 0.653792`

Feature diagnostics:

- `checkpoint_step_500`: `rvq_latent_quant_mse=0.100918`, `householder_delta_rms=0.076098`
- `checkpoint_step_750`: `rvq_latent_quant_mse=0.939869`, `householder_delta_rms=0.219852`
- `checkpoint_step_1000`: `rvq_latent_quant_mse=4.704360`, `householder_delta_rms=0.669120`

Interpretation:

- The regularizer controls the normal-unfreeze explosion, but it hurts the healthy 500-step frozen-phase solution.
- This variant improves over the catastrophic normal unfreeze, but it does not beat the non-regularized frozen HCG-H/HCS evidence and should not be used as a main paper result.
- The failure mode is now more precise: simple shrinkage of `mu_q`, `s_q`, and Householder magnitude is not enough. The next version should anchor the conditioning frame to the pre-unfreeze checkpoint instead of shrinking it toward a generic small-magnitude target.

Verification:

- `py_compile` passed for `hcg_rvq/losses.py` and the analysis tools.
- A random forward/loss smoke test with the regularized config returned a finite `conditioning_loss`.


## E017: Loss-Scheduled Drift Regularization After 500

Status: Done

Changes:

- Added `train.loss_schedule` support so loss coefficients can change by step without changing the optimizer or freeze schedule.
- Added `configs/pilot_hcg_rvq_h_staged_hs_slow_reg_after500.yaml`.
- The config keeps drift penalties at zero for steps `[0, 500)` and applies the same regularizer only after `h_s` is unfrozen.

W&B:

- https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/ywzyzh00

Checkpoint sweep on Kodak:

- `checkpoint_step_250`: `0.318537 bpp / 21.004732 dB / MS-SSIM 0.752418`
- `checkpoint_step_500`: `0.316976 bpp / 21.285610 dB / MS-SSIM 0.751540`
- `checkpoint_step_750`: `0.321154 bpp / 20.281686 dB / MS-SSIM 0.720381`
- `checkpoint_step_1000/latest`: `0.329011 bpp / 19.789570 dB / MS-SSIM 0.648762`

Feature diagnostics:

- `checkpoint_step_500`: `rvq_latent_quant_mse=0.073027`, `householder_delta_rms=0.072146`
- `checkpoint_step_750`: `rvq_latent_quant_mse=0.900671`, `householder_delta_rms=0.239734`
- `checkpoint_step_1000`: `rvq_latent_quant_mse=4.289382`, `householder_delta_rms=0.709898`

Interpretation:

- The 500-step checkpoint is the best pilot RD point so far, but this is before the regularizer is active and should be treated as another frozen/staged checkpoint sample, not proof that the regularizer works.
- After `h_s` unfreezes, quality still degrades and latent quantization MSE grows by more than an order of magnitude.
- Compared with the always-on regularizer, delaying the penalty preserves the good warmup solution, but neither regularization schedule solves the moving-coordinate problem.

Next:

- Use validation-selected `checkpoint_step_500` for current paper-oriented pilot comparisons.
- Implement a frame-anchor loss that compares current `mu_q`, `s_q`, and Householder-applied latents to the saved step-500 conditioning frame.
- Run multi-seed frozen 500 comparisons before spending full compute on `h_s` adaptation.


## E018: Step-500 Frame Anchor for `h_s` Adaptation

Status: Done

Changes:

- Added differentiable conditioning tensors to model outputs: `mu_q`, `log_s_q`, `s_q`, `y_norm`, and Householder-transformed normalized latent `u`.
- Added optional anchor losses to `RateDistortionLoss`:
  - `rho_anchor_mu` compares current `mu_q` with an anchor model.
  - `rho_anchor_log_s` compares current `log_s_q` with an anchor model.
  - `rho_anchor_u` compares the current pre-RVQ transformed latent frame `u` with an anchor model.
- Added `train.anchor_model` support. When anchor loss is active, training runs a frozen anchor model in `torch.no_grad()` and attaches `anchor_conditioning` to the current output.
- Added `configs/pilot_hcg_rvq_h_staged_hs_anchor_after500.yaml`.

Run setup:

- Resume checkpoint: `experiments/pilot_hcg_rvq_h_staged_hs_slow_reg_after500_g64_l1_k128_lambda0035/checkpoint_step_500.pth.tar`
- Anchor model: the same step-500 checkpoint.
- Active loss after step 500: `rho_anchor_mu=0.1`, `rho_anchor_log_s=0.1`, `rho_anchor_u=0.5`.
- W&B: https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/t7k7g606

Checkpoint sweep on Kodak:

- `checkpoint_step_750`: `0.325913 bpp / 20.428828 dB / MS-SSIM 0.725705`, `RD=2.558084`
- `checkpoint_step_1000/latest`: `0.331253 bpp / 18.781318 dB / MS-SSIM 0.650649`, `RD=3.547392`

Feature diagnostics:

- `checkpoint_step_750`: `rvq_latent_quant_mse=0.732994`, `householder_delta_rms=0.210435`, `commit_loss=0.916243`
- `checkpoint_step_1000`: `rvq_latent_quant_mse=3.682138`, `householder_delta_rms=0.774070`, `commit_loss=4.602672`

Interpretation:

- The anchor improves the step750 RD score relative to the after500 shrinkage regularizer (`2.558084` vs `2.611545`) and reduces step750 latent quantization MSE (`0.732994` vs `0.900671`).
- The long-horizon step1000 checkpoint still collapses. Anchor loss rises in W&B, and the model pays high commit loss while failing to keep image quality.
- This confirms that warmup and stable checkpoint initialization remain necessary, but `h_s` adaptation needs a more constrained design. A simple frame anchor is useful diagnosis, not yet a paper-result method.

Decision:

- Keep the main submission path on validation-selected frozen/staged step500 comparisons.
- Continue using warmup/stable checkpoint initialization. The new analyses make that choice stronger: direct RVQ and late checkpoints are visibly unstable in both RD and feature space.
- Defer more `h_s` adaptation until after multi-seed HCS/no-transform/HCG-H evidence is collected, unless testing a tightly isolated `h_s`-only or decoder/RVQ-frozen variant.


## E019: Prompt Re-Alignment and Validation Holdout

Status: Done

Prompt target re-read:

- Re-read `docs/prompt.txt` and re-centered the project goal: the key HCG-RVQ claim is that the hyperprior should predict not only entropy parameters but also local quantizer geometry.
- The main ablation ladder remains `Global RVQ -> HCS-RVQ -> HCG-RVQ-H no-transform -> HCG-RVQ-H`, with matched entropy model, backbone, and training budget.
- The paper-critical proof points are validation-selected checkpoint results, VQ/codebook statistics, index entropy behavior, and feature/geometry distributions, not just final-step Kodak numbers.

Implementation changes:

- Added validation-offset support to `hcg_rvq/data/image_folder.py` via `start_index`.
- Added `eval.start_index`, `eval.patch_size`, and `eval.max_images` support in `eval.py`.
- Added `--start-index` and `--patch-size` to `tools/evaluate_checkpoints.py`.
- Added `--start-index` and `--patch-size` to `tools/feature_distribution_analysis.py`.

Validation protocol:

- Used OpenImages train data after the 4096-image pilot training subset: `start_index=4096`, `max_images=256`, `patch_size=256`.
- Kept Kodak as a test-style evaluation split.
- Checkpoints are selected by validation RD score rather than by `checkpoint_latest`.

Seed-1234 OpenImages holdout results:

| Method | Best step | RD | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|
| HCS-RVQ | 500 | 2.999436 | 0.310322 | 20.209100 | 0.750540 |
| NoTransform | 500 | 2.998997 | 0.310326 | 20.211408 | 0.750652 |
| HCG-RVQ-H | 500 | 2.997846 | 0.310909 | 20.244426 | 0.750286 |

Feature read on seed 1234:

- HCG-RVQ-H has better image RD/PSNR despite worse latent quantization MSE than HCS/no-transform.
- This supports the geometry hypothesis as an error-direction effect, not simply lower latent error magnitude.

## E020: Multi-Seed HCS/NoTransform/HCG-H Geometry Check

Status: Done

Runs:

- Added and trained seed `2345` and seed `3456` configs for HCS-RVQ, HCG-RVQ-H no-transform, and HCG-RVQ-H.
- Cleaned the generated seed configs so future `run_name` values use one seed suffix. The completed experiment directories keep the originally generated double-suffix names and are referenced by the analysis CSVs.
- Evaluated all saved checkpoints on OpenImages holdout and Kodak.
- Ran intermediate feature diagnostics on validation-best checkpoints.

Consolidated outputs:

- `experiments/analysis/multiseed_hcg_geometry_checkpoint_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_checkpoint_means.csv`
- `experiments/analysis/multiseed_hcg_geometry_feature_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_summary.md`

OpenImages holdout, validation-best checkpoint mean over 3 seeds:

| Method | RD mean+-std | bpp mean+-std | PSNR mean+-std | MS-SSIM mean+-std | Steps |
|---|---:|---:|---:|---:|---|
| HCS-RVQ | 3.040305+-0.045972 | 0.311772+-0.001287 | 20.117441+-0.103738 | 0.749681+-0.001022 | 500/250/250 |
| NoTransform | 3.040674+-0.044819 | 0.311781+-0.001293 | 20.116747+-0.102200 | 0.749798+-0.000972 | 500/250/250 |
| HCG-RVQ-H | 3.015330+-0.034915 | 0.311474+-0.001343 | 20.200736+-0.123357 | 0.749007+-0.001223 | 500/500/250 |

Kodak, checkpoint selected per split:

| Method | RD mean+-std | bpp mean+-std | PSNR mean+-std | MS-SSIM mean+-std | Steps |
|---|---:|---:|---:|---:|---|
| HCS-RVQ | 2.206217+-0.030258 | 0.318314+-0.001588 | 21.141664+-0.092295 | 0.749433+-0.001312 | 500/250/250 |
| NoTransform | 2.189912+-0.013770 | 0.317552+-0.001785 | 21.210728+-0.050294 | 0.750700+-0.001793 | 500/500/250 |
| HCG-RVQ-H | 2.162222+-0.018052 | 0.317176+-0.000296 | 21.264001+-0.071777 | 0.749293+-0.005310 | 500/500/500 |

Interpretation:

- HCG-RVQ-H is best on average for RD/PSNR on both OpenImages holdout and Kodak, but the gain is seed-sensitive.
- On OpenImages seed 3456, HCG-RVQ-H loses to HCS/no-transform at the validation-best checkpoint.
- HCS and no-transform are almost indistinguishable, which strengthens the attribution that the large gain over Global RVQ is from hyper-conditioned shift/scale plus index entropy.
- Householder geometry remains plausible and promising, but the current evidence is not yet robust enough for a strong final conference claim.

Next:

- Keep validation-selected checkpointing as mandatory for all paper results.
- Add more seeds or a larger validation subset before claiming robust HCG-H superiority.
- Run multi-rate lambda points only after the seed-sensitive geometry story is tightened.


## E021: Larger OpenImages Holdout and Seed3456 Feature Diagnosis

Status: Done

Motivation:

- Re-read the HCG-RVQ goal from `docs/prompt.txt`: the paper needs evidence that hyperprior-conditioned local quantizer geometry improves LIC, not just final-step pilot numbers.
- The previous 256-image holdout suggested HCG-H was average-positive but seed-sensitive. I expanded the checkpoint analysis to a 1024-image OpenImages holdout and then inspected the seed3456 failure case with feature-distribution diagnostics.

Validation setup:

- Dataset: OpenImages train holdout after the 4096-image pilot training subset.
- `start_index=4096`, `max_images=1024`, `patch_size=256`.
- Compared HCS-RVQ, HCG-RVQ-H no-transform, and HCG-RVQ-H for seeds `1234`, `2345`, and `3456`.
- Checkpoints were selected by validation RD over `checkpoint_step_250` and `checkpoint_step_500`; `checkpoint_latest` duplicates step500 in these runs.

Consolidated outputs:

- `experiments/analysis/multiseed_hcg_geometry_val1024_checkpoint_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_val1024_checkpoint_means.csv`
- `experiments/analysis/multiseed_hcg_geometry_val1024_summary.md`
- `experiments/analysis/feature_seed3456_val1024_summary.csv`
- `experiments/analysis/feature_seed3456_val1024_summary.md`
- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh500_val1024.csv`
- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh500_val1024.json`
- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh500_val1024_summary.md`

OpenImages val1024 best-checkpoint results:

| Method | Seed | Best step | RD | bpp | PSNR | MS-SSIM | Delta vs HCS |
|---|---:|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 1234 | 500 | 2.881269 | 0.310258 | 20.3841 | 0.751109 | +0.000000 |
| NoTransform | 1234 | 500 | 2.880802 | 0.310262 | 20.3867 | 0.751209 | -0.000467 |
| HCG-RVQ-H | 1234 | 500 | 2.875107 | 0.310846 | 20.4272 | 0.751290 | -0.006163 |
| HCS-RVQ | 2345 | 250 | 2.945318 | 0.312419 | 20.2253 | 0.749747 | +0.000000 |
| NoTransform | 2345 | 250 | 2.943384 | 0.312419 | 20.2288 | 0.749797 | -0.001935 |
| HCG-RVQ-H | 2345 | 500 | 2.870701 | 0.310437 | 20.4802 | 0.749885 | -0.074617 |
| HCS-RVQ | 3456 | 250 | 2.907870 | 0.312991 | 20.3188 | 0.748333 | +0.000000 |
| NoTransform | 3456 | 250 | 2.911528 | 0.313015 | 20.3089 | 0.748484 | +0.003658 |
| HCG-RVQ-H | 3456 | 500 | 2.931155 | 0.310774 | 20.2559 | 0.742348 | +0.023284 |

Mean over seeds:

| Policy | Method | RD mean | RD std | bpp mean | PSNR mean | MS-SSIM mean |
|---|---|---:|---:|---:|---:|---:|
| best | HCS-RVQ | 2.911486 | 0.026273 | 0.311889 | 20.3094 | 0.749730 |
| best | NoTransform | 2.911905 | 0.025550 | 0.311898 | 20.3081 | 0.749830 |
| best | HCG-RVQ-H | 2.892321 | 0.027518 | 0.310686 | 20.3878 | 0.747841 |
| step250 | HCS-RVQ | 2.924819 | 0.015493 | 0.312456 | 20.2659 | 0.750166 |
| step250 | NoTransform | 2.924934 | 0.013485 | 0.312465 | 20.2648 | 0.750228 |
| step250 | HCG-RVQ-H | 2.930410 | 0.031750 | 0.313172 | 20.2695 | 0.750253 |
| step500 | HCS-RVQ | 3.008294 | 0.123439 | 0.309929 | 20.1745 | 0.747331 |
| step500 | NoTransform | 2.955691 | 0.054883 | 0.309889 | 20.2645 | 0.748750 |
| step500 | HCG-RVQ-H | 2.892321 | 0.027518 | 0.310686 | 20.3878 | 0.747841 |

Seed3456 feature diagnostics:

| Method | Step | RD | bpp_y | PSNR | MS-SSIM | y_error_rms | index bpp | index ppl | RVQ ppl | dead ratio | s_q mean | H delta RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 250 | 2.907870 | 0.087921 | 20.3188 | 0.748333 | 0.136270 | 0.132065 | 108.51 | 60.37 | 0.197235 | 0.568808 | - |
| HCS-RVQ | 500 | 2.968102 | 0.084763 | 20.1741 | 0.744118 | 0.132898 | 0.133109 | 112.61 | 63.73 | 0.182259 | 0.452208 | - |
| NoTransform | 250 | 2.911528 | 0.087945 | 20.3089 | 0.748484 | 0.136283 | 0.132066 | 108.52 | 60.37 | 0.197281 | 0.568895 | - |
| NoTransform | 500 | 2.975473 | 0.084785 | 20.1571 | 0.745662 | 0.132987 | 0.133099 | 112.57 | 63.73 | 0.182449 | 0.451913 | - |
| HCG-RVQ-H | 250 | 2.943222 | 0.088145 | 20.2033 | 0.747635 | 0.139121 | 0.132376 | 109.72 | 60.97 | 0.194420 | 0.563719 | 0.091907 |
| HCG-RVQ-H | 500 | 2.931155 | 0.085704 | 20.2559 | 0.742348 | 0.134382 | 0.133068 | 112.44 | 63.84 | 0.180534 | 0.450012 | 0.101297 |

Image-level HCS step250 vs HCG-H step500 comparison on seed3456:

| Metric | Value |
|---|---:|
| HCG-H better by RD | 428 / 1024 images |
| HCG-H worse by RD | 596 / 1024 images |
| Mean delta RD | +0.023284 |
| Mean delta bpp | -0.002217 |
| Mean delta PSNR | -0.062853 dB |
| Mean delta MS-SSIM | -0.005984 |

Interpretation:

- The larger holdout keeps HCG-RVQ-H best on average RD, so the geometry direction remains promising.
- The seed3456 counterexample remains real: HCG-H step500 beats its own step250 but still loses to HCS/no-transform step250, especially in MS-SSIM.
- HCS/no-transform degrade from step250 to step500 even though `y_error_rms` decreases, so raw latent quantization error is not sufficient to explain RD/quality collapse.
- HCG-H uses nontrivial geometry by step500 (`s_q_mean≈0.45`, Householder delta RMS `0.101297`) and lowers `bpp_y`, but the perceptual quality drop points to reconstruction/prior/frame mismatch rather than simple dead-code failure.

Decision:

- Continue validation-selected checkpointing as a mandatory protocol.
- Do not claim robust Householder superiority yet; claim average-positive evidence plus a clearly analyzed seed sensitivity.
- Image-level RD deltas are now available; next diagnostic should generate decoder-side error/feature maps for the largest HCG-H losses and gains.
- Keep warmup and stable scalar-compatible initialization as the main training recipe. Full scratch training should be tested as an ablation later, but current VQ-collapse evidence makes it too risky as the main path.


## E022: Visual Error and Decoder-Feature Case Diagnostics

Status: Done

Motivation:

- E021 showed that HCG-H seed3456 is not globally collapsed: it improves RD on `428/1024` validation images and worsens `596/1024`.
- The next paper-critical question was whether the failure is visible in spatial error maps and decoder-side activations, rather than only aggregate feature/codebook statistics.

Implementation:

- Added `tools/visualize_comparison_cases.py`.
- The tool loads two checkpoints, selects top HCG-H loss/gain images from a per-image CSV, and writes panel PNGs containing:
  - input image,
  - both reconstructions,
  - signed reconstruction difference,
  - pixel error maps,
  - latent error maps,
  - HCG-H conditioning maps (`s_q`, `mu_q`, Householder delta),
  - decoder `g_s` activation-difference maps.

Run:

- A: HCS-RVQ seed3456 `checkpoint_step_250`.
- B: HCG-RVQ-H seed3456 `checkpoint_step_500`.
- Dataset: OpenImages val1024 slice, `start_index=4096`, `max_images=1024`, `patch_size=256`.
- Selected top 5 HCG-H losses and top 5 HCG-H gains by per-image RD delta.

Outputs:

- `experiments/analysis/visual_seed3456_hcs250_vs_hcgh500_val1024/visual_case_summary.md`
- `experiments/analysis/visual_seed3456_hcs250_vs_hcgh500_val1024/case_summary.csv`
- `experiments/analysis/visual_seed3456_hcs250_vs_hcgh500_val1024/summary.json`
- Panel PNGs under `experiments/analysis/visual_seed3456_hcs250_vs_hcgh500_val1024/`.

Selected-case summary:

| Group | n | Mean delta RD | Mean delta bpp | Mean delta PSNR | Mean delta MS-SSIM | Mean decoder delta g_s_0 | Mean decoder delta g_s_6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| largest HCG-H losses | 5 | +1.665789 | +0.000758 | -0.9148 | -0.041555 | 0.025198 | 0.053116 |
| largest HCG-H gains | 5 | -1.383967 | +0.004994 | +1.8094 | -0.030552 | 0.024364 | 0.058575 |

Interpretation:

- The largest HCG-H losses are not simply rate-saving artifacts; PSNR and MS-SSIM both drop strongly.
- The largest HCG-H gains improve RD and PSNR, but often still lose MS-SSIM. HCG-H is currently biased toward MSE/RD improvement at the cost of structural similarity on some images.
- Decoder activation-difference magnitudes are similar between the selected loss and gain groups, so failure is unlikely to be separable by a single global decoder-drift magnitude threshold.
- The next method action should be a geometry-strength gate or regularizer: preserve Householder where it improves RD, but reduce it where it damages structure.

Decision:

- Keep HCG-H as promising but not robust enough for final submission claims.
- Add a controlled geometry-strength ablation before expensive multi-rate sweeps: for example full Householder vs reduced-strength Householder vs learned/per-location geometry gate, all under validation-selected checkpointing.
- Continue reporting MS-SSIM alongside PSNR/RD, because current HCG-H gains can hide structural degradation.

## E023: Reduced-Strength Householder Geometry Diagnostic

Status: Done

Motivation:

- E021/E022 showed that full HCG-RVQ-H is conditionally useful but seed-sensitive. On seed3456 it improves RD on some images but loses to HCS overall and can damage MS-SSIM.
- The immediate question was whether the Householder rotation is simply too strong, or whether the method needs image/location-specific reliability control.

Implementation:

- Added `quantizer.householder_strength` to the HCG-H model path.
- `householder_strength=1.0` preserves the original orthogonal Householder behavior.
- `householder_strength=0.25` applies an invertible partial reflection as a diagnostic ablation; it is not treated as the final method.
- Feature diagnostics now report `householder_strength` and recompute conditioning statistics with the partial transform.

Run:

- Config: `configs/pilot_hcg_rvq_h_strength025_frozen_seed3456.yaml`.
- Training: frozen 500-step pilot from the scalar-compatible HCG-H init.
- Dataset for analysis: OpenImages val1024 slice, `start_index=4096`, `max_images=1024`, `patch_size=256`.
- Outputs:
  - `experiments/analysis/pilot_hcg_rvq_h_strength025_seed3456_openimages_val1024.csv`
  - `experiments/analysis/feature_hcg_h_strength025_seed3456_step250_openimages_val1024.json`
  - `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh025_250_val1024.json`
  - `experiments/analysis/per_image_seed3456_hcgh500_vs_hcgh025_250_val1024.json`
  - `experiments/analysis/visual_seed3456_hcs250_vs_hcgh025_250_val1024/visual_case_summary.md`
  - `experiments/analysis/strength025_seed3456_val1024_summary.md`

Checkpoint sweep:

| Method | Best step | RD | bpp | bpp_y | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 250 | 2.907870 | 0.312991 | 0.087921 | 20.3188 | 0.748333 |
| HCG-RVQ-H full | 500 | 2.931155 | 0.310774 | 0.085704 | 20.2559 | 0.742348 |
| HCG-RVQ-H strength0.25 | 250 | 2.934233 | 0.313435 | 0.088366 | 20.2480 | 0.747560 |

Per-image results:

| Comparison | B better by RD | B worse by RD | Mean delta RD | Mean delta PSNR | Mean delta MS-SSIM |
|---|---:|---:|---:|---:|---:|
| HCS250 -> full HCG-H500 | 428/1024 | 596/1024 | 0.023284 | -0.062853 | -0.005984 |
| HCS250 -> strength0.25 step250 | 382/1024 | 642/1024 | 0.026363 | -0.070841 | -0.000772 |
| full HCG-H500 -> strength0.25 step250 | 531/1024 | 493/1024 | 0.003072 | -0.007970 | 0.005212 |

Interpretation:

- Strength0.25 reduces the learned geometry magnitude (`H_delta_rms=0.054287`) compared with full HCG-H step500 (`0.101297`) and restores most of the MS-SSIM loss.
- It still loses to HCS step250 in mean RD (`2.934233` vs `2.907870`) and improves only `382/1024` images against HCS.
- It wins `531/1024` images against full HCG-H step500 and improves MS-SSIM by `+0.005212`, but its mean RD is still `+0.003072` worse than full HCG-H because fixed damping also removes useful geometry.
- The visual top-case analysis confirms the same pattern: selected-case losses are much smaller than full HCG-H, but selected-case gains are also much smaller.

Decision:

- A fixed global Householder strength is a useful diagnostic but not the final answer.
- Next method action: implement a learned hyper-conditioned geometry gate or regularizer, then evaluate it with the same checkpoint sweep, feature distribution analysis, per-image comparison, and visual diagnostics.
- Continue using stable scalar-compatible initialization, warmup, and validation-selected checkpointing as the main protocol.

## E024: Learned Householder Geometry Gate Diagnostic

Status: Done

Motivation:

- E023 showed that fixed `householder_strength=0.25` reduced structural damage but also reduced the upside, and still failed to beat HCS on seed3456.
- The next method hypothesis was that geometry needs reliability control from hyper features, not a single global damping value.

Implementation:

- Added a learned Householder strength gate: `quantizer.householder_gate_enabled`, `householder_gate_max`, and `householder_gate_init`.
- The gate is predicted per spatial position and RVQ group from hyper features, then used as an invertible partial Householder strength capped below `0.5`.
- Added gate/strength statistics to `rvq_stats`, `conditioning_tensors`, and `tools/feature_distribution_analysis.py`.
- Config: `configs/pilot_hcg_rvq_h_gate025_frozen_seed3456.yaml`.

Run:

- Training: frozen 500-step pilot from `experiments/init/pilot_hcg_rvq_h_g64_l1_k128_from_scalar.pth.tar`.
- W&B run: `https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/lwri15nr`.
- Dataset for analysis: OpenImages val1024 slice, `start_index=4096`, `max_images=1024`, `patch_size=256`.
- Main outputs:
  - `experiments/analysis/pilot_hcg_rvq_h_gate025_seed3456_openimages_val1024.csv`
  - `experiments/analysis/feature_hcg_h_gate025_seed3456_step250_openimages_val1024.json`
  - `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh_gate025_250_val1024.json`
  - `experiments/analysis/per_image_seed3456_hcgh025_250_vs_hcgh_gate025_250_val1024.json`
  - `experiments/analysis/per_image_seed3456_hcgh500_vs_hcgh_gate025_250_val1024.json`
  - `experiments/analysis/visual_seed3456_hcs250_vs_hcgh_gate025_250_val1024/visual_case_summary.md`
  - `experiments/analysis/gate025_seed3456_val1024_summary.md`

Checkpoint sweep:

| Method | Best step | RD | bpp | bpp_y | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 250 | 2.907870 | 0.312991 | 0.087921 | 20.3188 | 0.748333 |
| HCG-RVQ-H full | 500 | 2.931155 | 0.310774 | 0.085704 | 20.2559 | 0.742348 |
| HCG-RVQ-H strength0.25 | 250 | 2.934233 | 0.313435 | 0.088366 | 20.2480 | 0.747560 |
| HCG-RVQ-H gate0.25 | 250 | 2.903028 | 0.312918 | 0.087848 | 20.2335 | 0.751104 |

Interpretation:

- Gate0.25 is the first seed3456 geometry variant to beat HCS step250 by mean RD (`2.903028` vs `2.907870`) and MS-SSIM (`0.751104` vs `0.748333`).
- Mean gate strength is `0.254163` with small variation (`std=0.007656`, min/max about `0.243/0.279`), so the gain is not yet a strong binary routing effect.
- Against HCS, gate0.25 improves mean RD by `-0.004843` but wins only `425/1024` images, so tail failures remain.
- Against fixed strength0.25 and full HCG-H, gate0.25 improves mean RD by `-0.031206` and `-0.028127`, respectively, and improves MS-SSIM in both comparisons.

Decision:

- Promote learned geometry gating to the main HCG-H stability direction.
- Do not jump to multi-rate paper curves yet. First repeat gate0.25 on the other seeds and inspect whether the gain survives validation-selected checkpointing.
- Keep fixed strength0.25 as a diagnostic ablation, not as the final method.

## E025: Learned Householder Gate Multi-seed Validation

Status: Done

Motivation:

- E024 showed that learned gate0.25 fixed the seed3456 counterexample in mean RD, but only on one seed.
- The submission-relevant question was whether the geometry gate is a real stability improvement over HCS/no-transform/full HCG-H, or just a single-seed rescue.

Implementation and runs:

- Added configs for the remaining seeds:
  - `configs/pilot_hcg_rvq_h_gate025_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_frozen_seed2345.yaml`
- Trained both from the scalar-compatible HCG-H initialization with the same frozen 500-step protocol.
- W&B runs:
  - seed1234: `https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/acri302l`
  - seed2345: `https://wandb.ai/mayuyuto0714-waseda-university/HCG-RVQ/runs/6p7grn8v`
- Main outputs:
  - `experiments/analysis/pilot_hcg_rvq_h_gate025_seed1234_openimages_val1024.csv`
  - `experiments/analysis/pilot_hcg_rvq_h_gate025_seed2345_openimages_val1024.csv`
  - `experiments/analysis/feature_hcg_h_gate025_seed1234_step250_openimages_val1024.json`
  - `experiments/analysis/feature_hcg_h_gate025_seed2345_step250_openimages_val1024.json`
  - `experiments/analysis/per_image_seed1234_hcsbest_vs_hcgh_gate025_best_val1024.json`
  - `experiments/analysis/per_image_seed2345_hcsbest_vs_hcgh_gate025_best_val1024.json`
  - `experiments/analysis/gate025_multiseed_val1024_summary.md`
  - `experiments/analysis/gate025_tail_analysis_val1024.md`
  - `experiments/analysis/visual_seed1234_hcsbest_vs_hcgh_gate025_best_val1024/visual_case_summary.md`
  - `experiments/analysis/visual_seed2345_hcsbest_vs_hcgh_gate025_best_val1024/visual_case_summary.md`

Best-checkpoint mean over seeds:

| Method | RD mean | RD std | bpp mean | bpp_y mean | PSNR mean | MS-SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 2.911486 | 0.026273 | 0.311889 | 0.086819 | 20.3094 | 0.749730 |
| NoTransform | 2.911905 | 0.025550 | 0.311898 | 0.086829 | 20.3081 | 0.749830 |
| HCG-RVQ-H full | 2.892321 | 0.027518 | 0.310686 | 0.085616 | 20.3878 | 0.747841 |
| HCG-RVQ-H gate0.25 | 2.859852 | 0.031195 | 0.313430 | 0.088360 | 20.3455 | 0.750814 |

Gate0.25 vs HCS:

| Seed | HCS step | Gate step | Delta RD | Delta bpp | Delta PSNR | Delta MS-SSIM | Gate better | Gate worse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 500 | 250 | -0.050850 | +0.003788 | +0.0457 | -0.000845 | 614/1024 | 410/1024 |
| 2345 | 250 | 250 | -0.099210 | +0.000906 | +0.1479 | +0.001326 | 670/1024 | 354/1024 |
| 3456 | 250 | 250 | -0.004843 | -0.000073 | -0.0853 | +0.002771 | 425/1024 | 599/1024 |

Feature diagnostics:

| Seed | H delta RMS | Strength mean | Strength std | Index bpp | Index ppl | RVQ ppl | Dead ratio | y error RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.060919 | 0.258210 | 0.008257 | 0.132690 | 110.95 | 61.05 | 0.198074 | 0.136919 |
| 2345 | 0.056958 | 0.256529 | 0.009009 | 0.132557 | 110.42 | 60.88 | 0.197510 | 0.137068 |
| 3456 | 0.054329 | 0.254163 | 0.007656 | 0.132623 | 110.68 | 61.03 | 0.196991 | 0.135984 |

Interpretation:

- Gate0.25 improves mean RD by `-0.051634` vs HCS, `-0.052053` vs NoTransform, and `-0.032469` vs full HCG-H.
- The gain is not merely from no-transform or checkpoint noise: all three seeds select gate step250, and the 3-seed mean beats every current geometry/control baseline.
- The feature distribution is stable: index entropy stays near `0.1326` empirical bpp, RVQ perplexity stays near `61/128`, and dead-code ratio stays near `0.197`.
- The learned gate still stays close to its initialization (`strength_mean≈0.256`, `strength_std≈0.008`), so the current method is better described as a learned reliability-controlled partial geometry transform, not yet a strongly discrete spatial router.
- Tail failures remain: seed3456 improves mean RD and MS-SSIM but wins only `425/1024` images against HCS.
- Tail analysis suggests gate0.25 helps high-HCS-RD images more than easy images: corr(HCS RD, delta RD) is `-0.507`, `-0.287`, and `-0.275` for seeds 1234/2345/3456, while seed3456 Q1 worsens by `+0.137167` mean RD and wins only `52/256`.

Decision:

- Promote HCG-RVQ-H gate0.25 to the current main HCG geometry variant for the paper track.
- Keep HCS-RVQ as the strongest simple internal baseline and no-transform as the geometry attribution control.
- Next method actions: add a gate-strength/geometry-damage regularizer or tail-aware selection diagnostic, then run multi-rate curves for HCS, no-transform, full HCG-H, and gate0.25 only after confirming the same behavior on a larger validation slice.

## E026: Gate0.25 Larger Validation and Tail Re-check

Status: Done

Motivation:

- E025 made learned gate0.25 the current main geometry variant on OpenImages val1024, but the seed3456 tail remained fragile.
- The next submission-critical check was whether the gate survives a larger validation slice and whether the seed3456 failure reappears under more images.
- The target from `docs/prompt.txt` remains the same: prove that hyperprior-conditioned quantizer geometry adds value beyond entropy/index modeling, not just that a single checkpoint looks good.

Validation setup:

- Dataset: OpenImages train holdout after the 4096-image pilot training subset.
- `start_index=4096`, `max_images=4096`, `patch_size=256`.
- Compared HCS-RVQ vs HCG-RVQ-H gate0.25 for seeds `1234`, `2345`, and `3456`.
- Checkpoints were selected by validation RD over saved checkpoints; all gate0.25 seeds selected step250.

Consolidated outputs:

- `experiments/analysis/gate025_multiseed_val4096_summary.csv`
- `experiments/analysis/gate025_multiseed_val4096_summary.md`
- `experiments/analysis/gate025_tail_analysis_val4096.md`
- `experiments/analysis/gate025_tail_correlation_val4096.csv`
- `experiments/analysis/per_image_seed1234_hcsbest_vs_hcgh_gate025_best_val4096.{csv,json}`
- `experiments/analysis/per_image_seed2345_hcsbest_vs_hcgh_gate025_best_val4096.{csv,json}`
- `experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_best_val4096.{csv,json}`
- `experiments/analysis/feature_hcg_h_gate025_seed1234_step250_openimages_val4096.{csv,json}`
- `experiments/analysis/feature_hcg_h_gate025_seed2345_step250_openimages_val4096.{csv,json}`
- `experiments/analysis/feature_hcg_h_gate025_seed3456_step250_openimages_val4096.{csv,json}`

Gate0.25 vs HCS on OpenImages val4096:

| Seed | HCS best RD | Gate best RD | Delta RD | Gate better | Delta bpp | Delta PSNR | Delta MS-SSIM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.889061 | 2.843577 | -0.045484 | 2404/4096 | +0.003574 | +0.025629 | -0.001155 |
| 2345 | 2.941778 | 2.859697 | -0.082081 | 2593/4096 | +0.000959 | +0.118587 | +0.000893 |
| 3456 | 2.906864 | 2.923148 | +0.016284 | 1625/4096 | +0.000023 | -0.119677 | +0.002295 |

Mean over seeds:

- HCS RD: `2.912568`; Gate RD: `2.875474`; delta RD: `-0.037094` (`-1.27%`).
- HCS bpp: `0.311918`; Gate bpp: `0.313437`; delta bpp: `+0.001518`.
- HCS PSNR: `20.330887`; Gate PSNR: `20.339066`; delta PSNR: `+0.008180`.
- HCS MS-SSIM: `0.751029`; Gate MS-SSIM: `0.751707`; delta MS-SSIM: `+0.000678`.
- Mean Gate per-image win rate: `53.89%`.

Feature diagnostics:

| Seed | H delta RMS | Strength mean | Strength std | Index bpp | y error RMS |
|---:|---:|---:|---:|---:|---:|
| 1234 | 0.061265 | 0.258263 | 0.008256 | 0.132700 | 0.137006 |
| 2345 | 0.057304 | 0.256579 | 0.009024 | 0.132555 | 0.137158 |
| 3456 | 0.054590 | 0.254198 | 0.007681 | 0.132627 | 0.136072 |

Tail analysis:

| Seed | corr(HCS RD, delta RD) | Q1 low-RD delta | Q2 | Q3 | Q4 high-RD delta |
|---:|---:|---:|---:|---:|---:|
| 1234 | -0.501500 | +0.045084 | +0.005263 | -0.032637 | -0.199645 |
| 2345 | -0.199171 | +0.035773 | -0.058252 | -0.123226 | -0.182619 |
| 3456 | -0.159458 | +0.130779 | +0.036509 | -0.036182 | -0.065971 |

Interpretation:

- The larger validation slice still supports gate0.25 on average, but less strongly than val1024: mean RD improvement shrinks from `-0.051634` to `-0.037094`.
- The seed3456 counterexample reappears at val4096. Gate0.25 is better on the hardest quartile but worse on Q1-Q2 and loses the mean RD by `+0.016284`.
- Index empirical bpp and dead-code behavior stay close across seeds, so this is not codebook collapse. The likely issue is reliability of applying local geometry on easy/low-distortion images.
- Gate0.25 should remain the current main geometry variant, but the paper claim must be phrased as promising average-positive internal evidence until the seed/tail fragility is fixed or fully explained.

Decision:

- Keep gate0.25 as the main HCG geometry branch for the next method iteration.
- Add a tail-aware geometry reliability regularizer or a conservative gate schedule that suppresses geometry on low-risk images.
- Do not yet claim superiority to latest/SOTA LIC or VQ-GIC models. The current evidence is an internal ablation win on a small MeanScaleHyperprior/RVQ pilot, not a SOTA comparison.

## E027: Risk-Aware Householder Gate on Seed3456

Status: Done

Motivation:

- E026 showed that learned gate0.25 is average-positive across three seeds on OpenImages val4096, but seed3456 regresses against HCS by `+0.016284` RD.
- The tail analysis suggested a concrete reliability failure: old gate0.25 improves high-HCS-RD images but damages low-HCS-RD images.
- A decoder-valid reliability signal cannot depend on the original `y` or measured per-image RD, so the next conservative test used hyperprior-predicted local uncertainty `s_q`.

Implementation:

- Added optional risk-aware Householder gating to `hcg_rvq/models/hyperprior_rvq.py`.
- The raw learned Householder gate is multiplied by a decoder-known risk multiplier derived from `s_q`.
- Existing configs are unchanged unless `householder_gate_risk_enabled: true`.
- Added builder/config support and feature diagnostics for:
  - `householder_gate_raw`
  - `householder_risk_multiplier`
  - min/max/std of the risk multiplier
- Added configs:
  - `configs/pilot_hcg_rvq_h_gate025_risk_s056_min05_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_s056_min05_frozen_seed2345.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_s056_min05_frozen_seed3456.yaml`

Validation setup:

- First tested the fragile seed3456 only.
- Dataset: OpenImages train holdout after the pilot subset.
- `start_index=4096`, `max_images=4096`, `patch_size=256`.
- Compared HCS best checkpoint, old gate0.25 best checkpoint, and risk-aware gate0.25.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_s056_min05_seed3456_openimages_val4096.csv`
- `experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_risk_s056_min05_best_val4096.{csv,json}`
- `experiments/analysis/feature_hcg_h_gate025_risk_s056_min05_seed3456_step500_openimages_val4096.{csv,json}`
- `experiments/analysis/gate025_risk_s056_min05_seed3456_tail_val4096.{csv,md}`

Checkpoint sweep:

| checkpoint | RD | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| step250 | 2.919329 | 0.312612 | 20.231631 | 0.752973 |
| step500/latest | 2.914991 | 0.310266 | 20.452365 | 0.748765 |

Seed3456 comparison against HCS:

| Method | RD | Delta RD | Better images | Delta bpp | Delta PSNR | Delta MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 2.923148 | +0.016284 | 1625/4096 | +0.000023 | -0.119677 | +0.002295 |
| risk-aware gate0.25 | 2.914991 | +0.008127 | 2052/4096 | -0.002662 | +0.108822 | -0.001012 |

Tail analysis:

| Bucket | old gate delta RD | risk-aware delta RD | risk-aware minus old |
|---|---:|---:|---:|
| Q1 low HCS RD | +0.130779 | -0.043602 | -0.174382 |
| Q2 | +0.036509 | -0.059493 | -0.096002 |
| Q3 | -0.036182 | -0.036639 | -0.000457 |
| Q4 high HCS RD | -0.065971 | +0.172243 | +0.238214 |

Feature diagnostics:

| Feature | old gate0.25 | risk-aware gate0.25 |
|---|---:|---:|
| index empirical bpp | 0.132627 | 0.133255 |
| index perplexity | 110.699539 | 113.195953 |
| rvq dead-code ratio | 0.195873 | 0.185013 |
| effective gate mean | 0.254198 | 0.184702 |
| Householder delta RMS | 0.054590 | 0.046407 |
| latent error RMS | 0.136072 | 0.133291 |
| risk multiplier mean | - | 0.610657 |

Interpretation:

- Risk-aware gating partially fixes seed3456: mean RD regression is reduced from `+0.016284` to `+0.008127`, and the image-level win count improves from `1625/4096` to `2052/4096`.
- The mechanism is not a codebook-collapse fix; index entropy and dead-code statistics remain stable.
- The `s_q` schedule protects low/mid-HCS-RD images but removes too much useful geometry on high-HCS-RD images.
- Feature diagnostics are consistent with risk-signal co-adaptation: `s_q_mean` drops from `0.556998` to `0.439628`, while raw gate mean rises to `0.304322` and the effective gate falls to `0.184702`. E028 checks this more carefully in per-image feature space.
- This is useful evidence for the diagnosis, but it is not yet a final paper variant.

Decision:

- Do not repeat this exact risk setting on all seeds yet.
- Next method iteration should tune or learn the risk schedule so Q1-Q3 gains are preserved while Q4 is not over-suppressed.
- Candidate directions: move the center lower than `0.56`, reduce `risk_min`, anchor or regularize the risk signal, add a learned residual risk head, or combine risk gating with a small hard-image floor for Householder strength.

## E028: Per-Image Feature Tail Analysis and Inverse-Risk Control

Status: Done

Motivation:

- E027 showed that `s_q` risk-aware gating improves the seed3456 mean regression but flips the tail: easy images improve, while Q4 hard images become worse.
- The next question was whether the diagnosis is real at the intermediate-feature level, or just an artifact of aggregate RD metrics.

Implementation:

- Added per-image feature extraction in `tools/per_image_feature_diagnostics.py`.
- Added joined tail/feature analysis in `tools/analyze_gate_risk_tail_features.py`.
- Added inverse/detached risk-gate controls:
  - `householder_gate_risk_invert`
  - `householder_gate_risk_detach`
- Added the diagnostic config:
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s056_min05_frozen_seed3456.yaml`

Outputs:

- `experiments/analysis/per_image_features_hcg_h_gate025_seed3456_step250_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_s056_min05_seed3456_step500_val4096.{csv,json}`
- `experiments/analysis/gate025_risk_s056_min05_seed3456_feature_tail_val4096.{csv,md,json}`
- `experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_invrisk_s056_min05_posthoc_val4096.{csv,json}`
- `experiments/analysis/gate025_invrisk_s056_min05_posthoc_seed3456_tail_val4096.{csv,md,json}`

Per-image feature tail analysis:

| Bucket | old delta RD | risk delta RD | old `s_q` mean | risk `s_q` mean | risk multiplier | old latent quant MSE | risk latent quant MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 low HCS RD | +0.130779 | -0.043602 | 0.578890 | 0.477291 | 0.647315 | 0.030177 | 0.049618 |
| Q2 | +0.036509 | -0.059493 | 0.566466 | 0.451376 | 0.629035 | 0.058655 | 0.098488 |
| Q3 | -0.036182 | -0.036639 | 0.547497 | 0.426939 | 0.606211 | 0.079553 | 0.180353 |
| Q4 high HCS RD | -0.065971 | +0.172243 | 0.535140 | 0.402699 | 0.578067 | 0.138815 | 0.333560 |

Correlation diagnostics:

| Statistic | Value |
|---|---:|
| corr(HCS RD, risk `s_q` mean) | -0.603673 |
| corr(HCS RD, risk multiplier mean) | -0.549000 |
| corr(HCS RD, risk effective strength mean) | -0.518352 |
| corr(HCS RD, risk latent quant MSE) | +0.784892 |
| corr(risk delta RD, risk latent quant MSE) | +0.354808 |

Q4 extremes under risk-aware gate:

| Subset | Delta RD | risk `s_q` mean | risk multiplier | effective gate | latent quant MSE | commit loss | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 worst20 | +3.051308 | 0.358585 | 0.548470 | 0.177530 | 0.693206 | 0.866507 | 13.4576 | 0.5800 |
| Q4 best20 | -1.385704 | 0.367893 | 0.553918 | 0.178915 | 0.240555 | 0.300694 | 18.6202 | 0.7407 |

Post-hoc inverse-risk control:

| Bucket | Delta RD | Delta bpp | Delta PSNR | Delta MS-SSIM | RD wins |
|---|---:|---:|---:|---:|---:|
| Q1 low HCS RD | +0.447378 | -0.000610 | -1.649312 | +0.000774 | 80/1024 |
| Q2 | +0.294486 | -0.000400 | -0.627623 | +0.002210 | 144/1024 |
| Q3 | +0.177387 | -0.000127 | -0.235276 | +0.003886 | 246/1024 |
| Q4 high HCS RD | +0.105954 | +0.000023 | -0.069960 | +0.004766 | 325/1024 |

Interpretation:

- The E027 tail flip is real. It is visible both in RD and in intermediate features.
- Raw `s_q` is not a reliable hard-image risk signal in this run. It is anti-aligned with HCS-RD hardness: harder images have lower `s_q`, so the current multiplier suppresses geometry more on Q4 than Q1.
- The co-adaptation statement should be treated as a supported hypothesis, not a proven mechanism. The observed facts are: risk training changes `s_q_mean` from `0.556998` to `0.439628`, raises raw gate from `0.254198` to `0.304322`, and lowers effective gate to `0.184702`; this is consistent with co-adaptation of the signal and gate.
- Q4 failure is not only a gate-threshold issue. Risk-aware Q4 latent quantization MSE jumps from `0.138815` to `0.333560`, so suppressing geometry also changes the local coordinate/codebook fit.
- Post-hoc inverse risk is not a fair trained variant and is much worse overall. It confirms that geometry-strength rules cannot be swapped after training without breaking decoder/RVQ alignment.

Decision:

- Do not train the exact inverse-risk config blindly as the next main experiment.
- The next useful method test should be a trained reliability-controlled gate with a decoder-known but anchored signal: detach/anchor `s_q`, ramp the risk multiplier after warmup, and calibrate the center/min so hard images keep a geometry floor.
- Keep gate0.25 as the main HCG geometry branch until a reliability-controlled variant beats it on seed3456 and then survives the 3-seed val4096 check.

## E029: Calibrated Inverse/Detached Risk Gate on Seed3456

Status: Done

Motivation:

- E028 showed that raw `s_q` is anti-aligned with HCS-RD hardness, so the first non-inverted `s_q` risk gate suppressed geometry too much on Q4 hard images.
- A direct post-hoc inverse-risk swap was much worse, so the next test had to be trained, detached, and conservative rather than applied after the fact.

Implementation:

- Added calibrated seed3456 config:
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_frozen_seed3456.yaml`
- Differences from the direct inverse config:
  - `householder_gate_risk_center: 0.44`
  - `householder_gate_risk_min: 0.75`
  - `householder_gate_risk_invert: true`
  - `householder_gate_risk_detach: true`
- Initial random-input smoke keeps gate partially active instead of collapsing it:
  - risk multiplier: `0.750301`
  - effective strength: `0.187575`

Training and validation:

- Trained seed3456 for 500 steps with the same frozen protocol.
- Output directory:
  - `experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_frozen_g64_l1_k128_lambda0035_seed3456`
- OpenImages val4096 checkpoint sweep:

| checkpoint | RD | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| step250 | 2.891653 | 0.313031 | 20.297079 | 0.753777 |
| step500/latest | 2.936486 | 0.310122 | 20.389764 | 0.748504 |

Comparison against HCS seed3456:

| Method | RD | Delta RD | Better images | Delta bpp | Delta PSNR | Delta MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 2.923148 | +0.016284 | 1625/4096 | +0.000023 | -0.119677 | +0.002295 |
| non-inverted `s_q` risk | 2.914991 | +0.008127 | 2052/4096 | -0.002662 | +0.108822 | -0.001012 |
| calibrated inverse/detached risk | 2.891653 | -0.015211 | 1884/4096 | +0.000103 | -0.046463 | +0.004001 |

Tail analysis:

| Bucket | old gate delta RD | non-inverted risk delta RD | calibrated delta RD | calibrated risk multiplier | calibrated strength | calibrated latent quant MSE |
|---|---:|---:|---:|---:|---:|---:|
| Q1 low HCS RD | +0.130779 | -0.043602 | +0.096617 | 0.791067 | 0.201645 | 0.029915 |
| Q2 | +0.036509 | -0.059493 | +0.004225 | 0.797832 | 0.203946 | 0.054919 |
| Q3 | -0.036182 | -0.036639 | -0.066554 | 0.803551 | 0.205856 | 0.082066 |
| Q4 high HCS RD | -0.065971 | +0.172243 | -0.095130 | 0.811599 | 0.208478 | 0.137397 |

Feature diagnostics:

| Feature | calibrated inverse/detached risk |
|---|---:|
| `s_q_mean` | 0.559189 |
| raw gate mean | 0.255876 |
| risk multiplier mean | 0.801012 |
| effective strength mean | 0.204981 |
| Householder delta RMS | 0.044060 |
| latent quant MSE | 0.076074 |
| commit loss | 0.095093 |
| index empirical bpp | 0.112959 |
| dead-code ratio | 0.195763 |

Correlations:

| Statistic | Value |
|---|---:|
| corr(HCS RD, calibrated `s_q` mean) | -0.615712 |
| corr(HCS RD, calibrated risk multiplier) | +0.637829 |
| corr(HCS RD, calibrated strength) | +0.587055 |
| corr(HCS RD, calibrated latent quant MSE) | +0.855205 |
| corr(calibrated delta RD, calibrated latent quant MSE) | -0.187811 |

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_seed3456_openimages_val4096.csv`
- `experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min075_best_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min075_seed3456_step250_val4096.{csv,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min075_seed3456_tail_val4096.{csv,md,json}`

Interpretation:

- This is the first reliability-controlled HCG-H variant that fixes the seed3456 val4096 mean regression against HCS.
- The calibrated inverse multiplier now increases on harder images instead of decreasing, while keeping the variation small enough not to break easy images catastrophically.
- Q1 is still positive (`+0.096617` RD), so this is not final, but Q2 is nearly neutral and Q3/Q4 are improved.
- The result supports the diagnosis from E028: the issue was not “risk control is bad,” but that the first `s_q` risk direction was wrong for hard images and too aggressive.

Decision:

- Promote calibrated inverse/detached risk to the next candidate reliability-controlled geometry branch.
- Repeat this exact config on seeds 1234 and 2345 before any multi-rate curves.
- If the 3-seed mean improves over old gate0.25 without a new tail failure, use it as the paper-track HCG geometry variant; otherwise tune the center/min around this configuration.




## E030: Calibrated Inverse/Detached Risk Gate 3-Seed Follow-Up

Status: Done

Motivation:

- E029 fixed the fragile seed3456 with calibrated inverse/detached risk, but the prompt-level target is a publication-ready method, so the reliability control had to survive the same 3-seed val4096 protocol as gate0.25.
- I repeated the exact `center=0.44`, `risk_min=0.75`, `invert=true`, `detach=true` setting on seeds 1234 and 2345, then added per-image and intermediate-feature diagnostics for both old gate0.25 and calibrated risk.

New runs and diagnostics:

- Added configs:
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_frozen_seed2345.yaml`
- Trained both for 500 steps with the same frozen protocol and scalar-compatible initialization.
- Evaluated checkpoint sweeps on OpenImages val4096 (`train/data`, `start_index=4096`, `max_images=4096`, `patch_size=256`).
- Added per-image comparisons and per-image feature diagnostics for seeds 1234/2345.
- Fixed `tools/analyze_gate_risk_tail_features.py` reading text so inverse/detached runs no longer inherit the non-inverted-risk interpretation.

Checkpoint results:

| seed | HCS RD | old gate0.25 RD | old gate delta | calibrated risk RD | risk delta | risk - old gate | risk RD wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.889062 | 2.843577 | -0.045484 | 2.906425 | +0.017364 | +0.062848 | 1914/4096 |
| 2345 | 2.941778 | 2.859697 | -0.082081 | 2.890208 | -0.051570 | +0.030511 | 2362/4096 |
| 3456 | 2.906864 | 2.923148 | +0.016284 | 2.891653 | -0.015211 | -0.031495 | 1884/4096 |
| mean/sum | 2.912568 | 2.875474 | -0.037094 | 2.896096 | -0.016473 | +0.020621 | 6160/12288 |

Tail and feature reading:

| bucket | old gate delta RD | calibrated risk delta RD | risk - old | risk multiplier | old strength | risk strength |
|---|---:|---:|---:|---:|---:|---:|
| Q1 low HCS RD | +0.070545 | +0.083721 | +0.013176 | 0.790467 | 0.256589 | 0.203693 |
| Q2 | -0.005493 | +0.009347 | +0.014840 | 0.796710 | 0.257395 | 0.206882 |
| Q3 | -0.064015 | -0.042258 | +0.021757 | 0.801941 | 0.258045 | 0.208151 |
| Q4 high HCS RD | -0.149412 | -0.116698 | +0.032714 | 0.809733 | 0.258330 | 0.211742 |

Interpretation:

- Calibrated inverse/detached risk is positive against HCS on the 3-seed mean, but it is not better than old gate0.25.
- It fixes the seed3456 reliability failure and keeps seed2345 positive, but it breaks seed1234. This means it is a useful diagnostic and reliability-control proof, not the current paper-main method.
- The risk signal direction is now correct for hard images: the multiplier rises from Q1 to Q4 instead of falling. The remaining issue is global shrinkage: effective Householder strength drops from about `0.257-0.260` to about `0.204-0.213`, so the method loses too much of old gate0.25 Q4 upside while still hurting Q1/easy images.
- Detached risk prevents direct gate-gradient pressure on `s_q`, so the earlier co-adaptation claim should stay cautious. The correct reading is: the first non-inverted risk run showed signal/gate co-adaptation-like behavior; the detached inverse run shows that even without direct gate gradients, the RD objective can still reshape the coordinate frame and the effective geometry schedule.

Decision:

- Keep old gate0.25 as the current main HCG-RVQ-H pilot variant.
- Keep calibrated inverse/detached risk as evidence that the seed3456 failure is controllable, but do not promote it unless a selective/milder version beats old gate0.25 on the 3-seed val4096 check.
- Next method experiment should avoid global shrinkage. Candidate directions are: higher risk floor, anneal risk only after gate warmup, add a residual reliability term on top of old gate0.25, or validation-select/fallback between old gate and risk gate before multi-rate curves.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_seed1234_openimages_val4096.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min075_seed2345_openimages_val4096.csv`
- `experiments/analysis/per_image_seed1234_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min075_best_val4096.{csv,json}`
- `experiments/analysis/per_image_seed2345_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min075_best_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed1234_step250_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed2345_step250_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min075_seed1234_step250_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min075_seed2345_step250_val4096.{csv,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min075_seed1234_tail_val4096.{csv,md,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min075_seed2345_tail_val4096.{csv,md,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min075_seed3456_tail_val4096.{csv,md,json}`


## E031: Milder Calibrated Inverse/Detached Risk Floor min090

Status: Done

Motivation:

- E030 showed that calibrated inverse/detached risk (`risk_min=0.75`) can control the seed3456 failure, but it globally shrinks Householder geometry too much and loses to old gate0.25 on the 3-seed val4096 mean.
- I tested a milder floor, `householder_gate_risk_min=0.90`, keeping `center=0.44`, `sharpness=12.0`, `invert=true`, and `detach=true`. The goal was to preserve more old gate0.25 geometry while still reducing fragile seed3456 regressions.

New configs:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed1234.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed2345.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed3456.yaml`

Protocol:

- Same frozen 500-step protocol as gate0.25 and min075.
- Same OpenImages holdout: `start_index=4096`, `max_images=4096`, `patch_size=256`.
- Checkpoint-selected by val4096 RD.
- Added per-image HCS comparisons, per-image feature diagnostics, and quartile/tail summaries against old gate0.25.

Checkpoint results:

| seed | HCS RD | old gate0.25 RD | old gate delta | min075 RD | min090 RD | min090 delta | min090 - old | min090 - min075 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.889061 | 2.843577 | -0.045484 | 2.906425 | 2.891932 | +0.002871 | +0.048355 | -0.014493 |
| 2345 | 2.941778 | 2.859697 | -0.082081 | 2.890208 | 2.873066 | -0.068713 | +0.013368 | -0.017143 |
| 3456 | 2.906864 | 2.923148 | +0.016284 | 2.891653 | 2.911665 | +0.004801 | -0.011483 | +0.020012 |
| mean | 2.912568 | 2.875474 | -0.037094 | 2.896096 | 2.892221 | -0.020347 | +0.016747 | -0.003875 |

Per-image HCS comparison:

| seed | mean delta RD | mean delta bpp | mean delta PSNR | RD wins |
|---:|---:|---:|---:|---:|
| 1234 | +0.002871 | -0.000478 | -0.021844 | 2056/4096 |
| 2345 | -0.068713 | +0.000983 | +0.088905 | 2475/4096 |
| 3456 | +0.004801 | +0.000043 | -0.095526 | 1704/4096 |

Feature diagnostics:

| seed | checkpoint | s_q mean | raw gate | risk multiplier | effective strength | Householder delta RMS | latent quant MSE | index empirical bpp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | step500 | 0.447509 | 0.298652 | 0.947971 | 0.283407 | 0.097668 | 0.153426 | 0.114517 |
| 2345 | step250 | 0.561131 | 0.258088 | 0.920022 | 0.237463 | 0.053975 | 0.075442 | 0.112884 |
| 3456 | step250 | 0.557362 | 0.255509 | 0.920774 | 0.235275 | 0.051488 | 0.076627 | 0.112981 |

Tail reading against old gate0.25:

| seed | bucket | old gate delta RD | min090 delta RD | min090 - old | min090 win rate |
|---:|---|---:|---:|---:|---:|
| 1234 | Q1 low HCS RD | +0.045084 | +0.022309 | -0.022775 | 0.458984 |
| 1234 | Q4 high HCS RD | -0.199645 | -0.018416 | +0.181229 | 0.498047 |
| 2345 | Q1 low HCS RD | +0.035773 | +0.048568 | +0.012795 | 0.416016 |
| 2345 | Q4 high HCS RD | -0.182619 | -0.166221 | +0.016398 | 0.724609 |
| 3456 | Q1 low HCS RD | +0.130779 | +0.119577 | -0.011202 | 0.226562 |
| 3456 | Q4 high HCS RD | -0.065971 | -0.077523 | -0.011553 | 0.548828 |

Interpretation:

- min090 is a genuine improvement over min075 on the 3-seed mean (`2.892221` vs `2.896096`) and lowers the old gate0.25 seed3456 regression from `+0.016284` to `+0.004801`.
- It is still not the paper-main variant because old gate0.25 remains better by `0.016747` RD on the 3-seed mean.
- The seed behavior is asymmetric. Seed2345 remains clearly positive, seed3456 is only partially rescued, and seed1234 no longer has the min075-sized failure but still loses the large old gate0.25 gain.
- The feature diagnostics explain why: min090 no longer applies the strong global shrinkage of min075, but seed1234 moves into a different regime (`s_q=0.447509`, raw gate `0.298652`, effective strength `0.283407`, latent quant MSE `0.153426`). This is not proof of harmful co-adaptation, but it is strong evidence that the risk signal and quantizer frame can move together under the RD objective.
- On seed3456, min090 improves Q1/Q2/Q3/Q4 relative to old gate0.25 by about `0.011` RD, so the milder risk floor does what it was designed to do. The remaining issue is that seed1234 loses most of old gate0.25 Q4 upside.

Decision:

- Keep old gate0.25 as the current paper-main pilot.
- Keep min090 as a useful reliability-control diagnostic: it confirms that the min075 global-shrinkage failure can be relaxed and that seed3456 is partially controllable.
- Do not spend multi-rate paper compute on min090 yet.
- The next method action should be selective reliability control rather than a global multiplier. Candidate implementations:
  - apply risk shrink only where raw gate exceeds a learned/validation-selected threshold;
  - add a residual risk correction around old gate0.25 instead of multiplying the entire gate;
  - use validation-selected fallback between old gate0.25 and risk-gated checkpoints at image or checkpoint level;
  - regularize risk/gate co-adaptation by anchoring `s_q` or gating the reliability branch after a warmup.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed2345_openimages_val4096.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed3456_openimages_val4096.csv`
- `experiments/analysis/per_image_seed1234_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min090_best_val4096.{csv,json}`
- `experiments/analysis/per_image_seed2345_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min090_best_val4096.{csv,json}`
- `experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_risk_inv_detach_s044_min090_best_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step250_val4096.{csv,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min090_seed1234_tail_val4096.{csv,md,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min090_seed2345_tail_val4096.{csv,md,json}`
- `experiments/analysis/gate025_risk_inv_detach_s044_min090_seed3456_tail_val4096.{csv,md,json}`

## E032: Current Re-Eval and rng-safe Learnable Reliability Gate

Status: Done for seed3456; superseded by E033 for the completed current 3-seed re-eval.

Motivation:

- The previous min090 selector/oracle analysis showed real image-level headroom: an oracle old/min090 mixture reached mean delta RD -0.059695 against HCS, while the best simple threshold only gained -0.002516 over old gate0.25. This suggested that a learned local reliability signal is more appropriate than another hand-tuned global multiplier.
- I implemented a learnable reliability multiplier on top of the Householder gate, initialized near identity. The first reliability run was invalid as evidence because adding the Conv2d head advanced the RNG and changed later initialization/data order. I then made the head construction RNG-safe by restoring the CPU RNG state immediately after constructing the reliability head.

Implementation:

- Added quantizer config keys: householder_gate_reliability_enabled, householder_gate_reliability_min, and householder_gate_reliability_init.
- Added a 1x1 hyper-feature reliability head whose sigmoid output maps to [reliability_min, 1].
- The effective Householder strength is now raw gate * reliability multiplier * optional risk multiplier.
- Feature diagnostics now record householder_reliability_multiplier statistics.
- Added rng-safe configs for seeds 1234/2345/3456; only seed3456 was trained/evaluated in this action.

Current-code seed3456 val4096 checkpoint results:

| method | checkpoint | RD | bpp | bpp_y | bpp_z | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | step250 | 2.268639 | 0.313956 | 0.083882 | 0.230074 | 21.354937 | 0.776282 |
| old gate0.25 | step250 | 2.285962 | 0.313979 | 0.083905 | 0.230074 | 21.272076 | 0.780281 |
| min090 inverse/detached risk | step250 | 2.273224 | 0.314004 | 0.083930 | 0.230074 | 21.306910 | 0.780624 |
| rng-safe learnable reliability | step500 | 2.279754 | 0.311333 | 0.081259 | 0.230074 | 21.433170 | 0.782085 |

Per-image HCS comparison for rng-safe reliability:

| comparison | mean delta RD | mean delta bpp | mean delta PSNR | mean delta MS-SSIM | RD wins |
|---|---:|---:|---:|---:|---:|
| reliability - HCS | +0.011115 | -0.002623 | +0.078233 | +0.005802 | 2051/4096 |

Feature diagnostics, old gate0.25 vs rng-safe reliability:

| metric | old gate0.25 | rng-safe reliability |
|---|---:|---:|
| s_q mean | 0.564647 | 0.449888 |
| raw gate mean | 0.253815 | 0.292822 |
| reliability multiplier mean | n/a | 0.993058 |
| effective strength mean | 0.253815 | 0.290807 |
| Householder delta RMS | 0.055915 | 0.099215 |
| latent quant MSE | 0.058575 | 0.122340 |
| index empirical bpp | 0.118051 | 0.120066 |
| dead-code ratio | 0.033667 | 0.026306 |
| perplexity | 65.782034 | 69.801497 |

Interpretation:

- The current-code re-eval supersedes older seed3456 val4096 CSVs that reported RD around 2.90. Those files are stale relative to the current checkpoints/code and should not be mixed with the new reliability result.
- rng-safe reliability improves seed3456 over current old gate0.25 by -0.006207 RD and improves rate/average PSNR/MS-SSIM, but it is still worse than HCS by +0.011115 RD and worse than current min090 by +0.006531 RD.
- The reliability multiplier barely moves from identity: mean 0.993058 with std 0.000880. This means the branch is not yet learning the selective reliability control we wanted. The run instead shifts the whole coordinate/geometry regime: s_q drops, raw gate rises, Householder delta RMS almost doubles, and latent quant MSE more than doubles.
- Therefore this is a useful diagnostic and implementation foundation, but not the paper-main method.

Decision:

- Do not promote learnable reliability yet.
- Treat the first non-rngsafe reliability run as invalid/confounded and do not cite its strong RD as evidence.
- Re-evaluate seeds 1234 and 2345 under the current code/checkpoints before making a new 3-seed mean claim. The older 3-seed tables remain useful as historical diagnostics, but the paper-facing table needs a single current snapshot.
- Next implementation should make reliability selective/residual and regularized, for example by penalizing deviation from identity, warming it up after old gate stabilizes, or constraining it to only shrink high-risk local regions.

Outputs:

- experiments/analysis/gate025_min090_selector_val4096_summary.md
- experiments/analysis/gate025_min090_selector_val4096_thresholds.csv
- experiments/analysis/gate025_min090_selector_val4096_summary.json
- experiments/analysis/pilot_hcs_rvq_frozen_seed3456_openimages_val4096_reeval_current.csv
- experiments/analysis/pilot_hcg_rvq_h_gate025_seed3456_openimages_val4096_reeval_current.csv
- experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed3456_openimages_val4096_reeval_current.csv
- experiments/analysis/pilot_hcg_rvq_h_gate025_reliability_min05_init099_rngsafe_seed3456_openimages_val4096.csv
- experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_reliability_min05_init099_rngsafe_best_val4096.csv
- experiments/analysis/per_image_seed3456_hcsbest_vs_hcgh_gate025_reliability_min05_init099_rngsafe_best_val4096.json
- experiments/analysis/per_image_features_hcg_h_gate025_reliability_min05_init099_rngsafe_seed3456_step500_val4096.csv
- experiments/analysis/per_image_features_hcg_h_gate025_reliability_min05_init099_rngsafe_seed3456_step500_val4096.json
- experiments/analysis/per_image_features_hcg_h_gate025_seed3456_step250_val4096_reeval_current.csv
- experiments/analysis/per_image_features_hcg_h_gate025_seed3456_step250_val4096_reeval_current.json

## E033: Current 3-Seed Re-Eval for HCS, old gate0.25, and min090

Status: Superseded by E034 for paper-facing holdout claims.

Correction: this run did not pass `--start-index`, so it evaluates `start_index=0`. Keep it as a current-code sanity check, not as the holdout4096 paper-facing result.

Motivation:

- E032 found that older val4096 CSVs were stale relative to the current code/checkpoints.
- Before making any paper-facing 3-seed claim, I re-evaluated seeds 1234/2345/3456 under one current snapshot, with checkpoint selection by val4096 RD.

Protocol:

- OpenImages holdout val4096: `data-root=/dpl/openimages/open-images-v6/train/data`, `max_images=4096`, CUDA evaluation.
- Methods: HCS-RVQ, HCG-RVQ-H old gate0.25, and min090 inverse/detached risk.
- Checkpoints: `checkpoint_step_*.pth.tar`, best selected by RD for each seed/method.

Best checkpoint results:

| seed | method | checkpoint | RD | delta vs HCS | bpp | PSNR | MS-SSIM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1234 | HCS-RVQ | step500 | 2.214147 | +0.000000 | 0.311250 | 21.517303 | 0.780520 |
| 1234 | old gate0.25 | step250 | 2.198001 | -0.016146 | 0.315139 | 21.506023 | 0.778609 |
| 1234 | min090 inverse/detached risk | step500 | 2.233269 | +0.019122 | 0.311096 | 21.445840 | 0.778001 |
| 2345 | HCS-RVQ | step250 | 2.319984 | +0.000000 | 0.313341 | 21.224199 | 0.777615 |
| 2345 | old gate0.25 | step250 | 2.221798 | -0.098186 | 0.314580 | 21.440841 | 0.779854 |
| 2345 | min090 inverse/detached risk | step250 | 2.233758 | -0.086226 | 0.314671 | 21.407967 | 0.779290 |
| 3456 | HCS-RVQ | step250 | 2.268639 | +0.000000 | 0.313956 | 21.354937 | 0.776282 |
| 3456 | old gate0.25 | step250 | 2.285962 | +0.017323 | 0.313979 | 21.272076 | 0.780281 |
| 3456 | min090 inverse/detached risk | step250 | 2.273224 | +0.004585 | 0.314004 | 21.306910 | 0.780624 |

Three-seed aggregate:

| method | mean RD | mean delta vs HCS | wins vs HCS | mean bpp | mean PSNR | mean MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 2.267590 | +0.000000 | 0/3 | 0.312849 | 21.365479 | 0.778139 |
| old gate0.25 | 2.235253 | -0.032337 | 2/3 | 0.314566 | 21.406313 | 0.779581 |
| min090 inverse/detached risk | 2.246750 | -0.020840 | 1/3 | 0.313257 | 21.386906 | 0.779305 |

Interpretation:

- The current re-eval reverses the E032 caution in a good way: old gate0.25 is again the strongest tested pilot variant on the 3-seed mean.
- The old 2.9-RD 3-seed table should be treated as stale and not used for paper-facing claims.
- min090 is still valuable as a seed3456 control diagnostic, but it is not the main method because it is +0.011497 RD worse than old gate0.25 on the current 3-seed mean and breaks seed1234.
- The remaining target is selective reliability around old gate0.25, not a global gate multiplier.

Outputs:

- `experiments/analysis/pilot_hcs_rvq_frozen_seed1234_openimages_val4096_reeval_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_reeval_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_reeval_current.csv`
- `experiments/analysis/pilot_hcs_rvq_frozen_seed2345_openimages_val4096_reeval_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_seed2345_openimages_val4096_reeval_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed2345_openimages_val4096_reeval_current.csv`
- `experiments/analysis/gate025_min090_multiseed_val4096_reeval_current_summary.{csv,json,md}`

## E034: True Holdout4096 Current 3-Seed Audit for HCS, old gate0.25, and min090

Status: Done.

Motivation:

- E033 used current code but defaulted to `start_index=0`, while earlier paper-facing analyses used an OpenImages holdout slice after the pilot subset.
- To avoid mixing evaluation slices, I reran the 3-seed checkpoint sweeps with `--start-index 4096` and generated a new holdout-specific summary.

Protocol:

- Data: `/dpl/openimages/open-images-v6/train/data`.
- Slice: `start_index=4096`, `max_images=4096`, `patch_size=256`.
- Device: CUDA.
- Methods: HCS-RVQ, HCG-RVQ-H old gate0.25, and min090 inverse/detached risk.
- Checkpoints: `checkpoint_step_*.pth.tar`; best selected by minimum RD for each seed/method.

Best checkpoint results:

| seed | method | checkpoint | RD | delta vs HCS | bpp | bpp_y | bpp_z | PSNR | MS-SSIM |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | HCS-RVQ | step500 | 2.211475 | +0.000000 | 0.311385 | 0.081253 | 0.230132 | 21.506715 | 0.781253 |
| 1234 | old gate0.25 | step250 | 2.193095 | -0.018380 | 0.315083 | 0.084951 | 0.230132 | 21.514720 | 0.779345 |
| 1234 | min090 inverse/detached risk | step500 | 2.228412 | +0.016937 | 0.311179 | 0.081047 | 0.230132 | 21.439251 | 0.778710 |
| 2345 | HCS-RVQ | step250 | 2.316296 | +0.000000 | 0.313219 | 0.083088 | 0.230132 | 21.239011 | 0.778564 |
| 2345 | old gate0.25 | step250 | 2.221826 | -0.094470 | 0.314491 | 0.084359 | 0.230132 | 21.432059 | 0.780538 |
| 2345 | min090 inverse/detached risk | step250 | 2.234240 | -0.082057 | 0.314570 | 0.084438 | 0.230132 | 21.397117 | 0.779975 |
| 3456 | HCS-RVQ | step250 | 2.263345 | +0.000000 | 0.313856 | 0.083724 | 0.230132 | 21.377084 | 0.777200 |
| 3456 | old gate0.25 | step500 | 2.285704 | +0.022358 | 0.311460 | 0.081329 | 0.230132 | 21.398422 | 0.782738 |
| 3456 | min090 inverse/detached risk | step500 | 2.227673 | -0.035673 | 0.311303 | 0.081172 | 0.230132 | 21.564738 | 0.778873 |

Three-seed aggregate:

| method | mean RD | mean delta vs HCS | wins vs HCS | mean bpp | mean bpp_y | mean bpp_z | mean PSNR | mean MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 2.263705 | +0.000000 | 0/3 | 0.312820 | 0.082688 | 0.230132 | 21.374270 | 0.779006 |
| old gate0.25 | 2.233542 | -0.030164 | 2/3 | 0.313678 | 0.083546 | 0.230132 | 21.448400 | 0.780873 |
| min090 inverse/detached risk | 2.230108 | -0.033597 | 2/3 | 0.312351 | 0.082219 | 0.230132 | 21.467035 | 0.779186 |

Interpretation:

- The previous concern that the strong result was purely an evaluation artifact is false: the true holdout4096 audit still shows a positive HCG-RVQ-H signal over HCS.
- The exact main variant is not settled. min090 is best by 3-seed mean, but only by `-0.003434` RD against old gate0.25.
- The seed profiles are complementary: old gate0.25 wins seeds 1234/2345, while min090 rescues seed3456 and loses seed1234.
- Therefore the next research action is not to pick a global gate permanently, but to build/select a reliability controller from true-holdout per-image and intermediate-feature diagnostics.

Outputs:

- `experiments/analysis/pilot_hcs_rvq_frozen_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcs_rvq_frozen_seed2345_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_seed2345_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed2345_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcs_rvq_frozen_seed3456_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_seed3456_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed3456_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/gate025_min090_multiseed_val4096_holdout4096_current_summary.{csv,json,md}`

## E035: True Holdout4096 Seed3456 Per-Image and Intermediate-Feature Audit

Status: Done.

Motivation:

- E034 showed that old gate0.25 and min090 both beat HCS on the 3-seed true holdout mean, but with different seed behavior.
- Seed3456 is the key failure/rescue case: old gate0.25 regresses, while min090 improves strongly.
- I refreshed per-image and intermediate-feature diagnostics on the same true holdout slice (`start_index=4096`, `max_images=4096`) to avoid reusing stale/start0 conclusions.

Protocol:

- Data: `/dpl/openimages/open-images-v6/train/data`.
- Slice: `start_index=4096`, `max_images=4096`, `patch_size=256`.
- HCS checkpoint: seed3456 step250.
- old gate0.25 checkpoint: seed3456 step500.
- min090 inverse/detached risk checkpoint: seed3456 step500.
- Also fixed `tools/analyze_gate_risk_tail_features.py` so it reads prefixed per-image columns such as `HCS_rd_score`; otherwise HCS-RD correlations become `nan`.

Per-image RD:

| comparison | mean delta RD | mean delta bpp | mean delta PSNR | mean delta MS-SSIM | wins | losses |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 - HCS | +0.022359 | -0.002395 | +0.021337 | +0.005538 | 1883 | 2213 |
| min090 - HCS | -0.035673 | -0.002553 | +0.187652 | +0.001673 | 2356 | 1740 |

Feature summary, old gate0.25 -> min090:

| feature | old | min090 | reading |
|---|---:|---:|---|
| `s_q_mean` | 0.448851 | 0.449871 | true-holdout best checkpoints do not show the earlier large `s_q` drop |
| raw gate mean | 0.292674 | 0.296461 | raw gate rises slightly |
| risk multiplier mean | n/a | 0.947354 | min090 globally suppresses effective geometry |
| effective gate mean | 0.292674 | 0.281185 | effective gate drops despite raw gate rising |
| Householder delta RMS | 0.099553 | 0.099314 | nearly unchanged/slightly smaller |
| latent quant MSE | 0.120477 | 0.119785 | slightly improved |
| commit loss | 0.150596 | 0.149732 | slightly improved |
| index empirical bpp | 0.119917 | 0.119938 | essentially unchanged |

Quartiles by HCS per-image RD:

| bucket | old delta RD | min090 delta RD | min090 - old | min090 win rate vs HCS | min090 beats old rate |
|---|---:|---:|---:|---:|---:|
| all | +0.022359 | -0.035673 | -0.058031 | 0.575195 | 0.879883 |
| Q1 low HCS RD | +0.006029 | -0.045590 | -0.051619 | 0.612305 | 0.921875 |
| Q2 | -0.018831 | -0.071590 | -0.052760 | 0.641602 | 0.893555 |
| Q3 | +0.000948 | -0.056510 | -0.057458 | 0.579102 | 0.870117 |
| Q4 high HCS RD | +0.101288 | +0.030999 | -0.070289 | 0.467773 | 0.833984 |

Interpretation:

- The earlier broad statement that calibrated/detached risk flips the tail is not valid for the current true holdout best-checkpoint audit. Here min090 improves the old-gate result in every HCS-RD quartile, including Q4, although Q4 still remains worse than HCS on average.
- The mechanism is not simply `s_q` collapsing. On this audit `s_q_mean` is almost unchanged/slightly higher, while raw gate rises slightly and the risk multiplier reduces effective gate.
- The co-adaptation concern should stay as a hypothesis, not a proven fact: detached risk blocks direct gate-gradient pressure through `s_q`, but the trained system can still co-adapt through the RD objective.
- The next experiment should use the true-holdout per-image tables to estimate an oracle old/min090 selector and then test whether a small reliability head can approximate it without moving the hyperprior frame.

Outputs:

- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh_gate025_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh_gate025_risk_inv_detach_s044_min090_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed3456_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/gate025_min090_seed3456_tail_features_val4096_holdout4096_current.{csv,json,md}`

## E036: True Holdout4096 Old/Min090 Oracle and Feature Reading

Status: Done.

Motivation:

- E034/E035 established that both old gate0.25 and min090 improve the 3-seed true-holdout mean over HCS, but neither fixed rule is reliable enough to be the final paper-main method.
- I completed the missing seed1234/2345 per-image comparisons and intermediate-feature diagnostics, then combined all three seeds into an old/min090 oracle analysis.

Protocol:

- Data: `/dpl/openimages/open-images-v6/train/data`.
- Slice: `start_index=4096`, `max_images=4096`, `patch_size=256`.
- Device: CUDA physical device 0 only via `CUDA_VISIBLE_DEVICES=0`; device 1 was not used.
- Compared HCS, old gate0.25, and min090 inverse/detached risk at the validation-selected checkpoints already used in the true-holdout audit.

Oracle summary:

| scope | HCS RD | old delta | min090 delta | oracle delta | oracle gain vs best fixed | old selected | min090 selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | -0.018379 | +0.016937 | -0.052511 | -0.034131 | 61.84% | 38.16% |
| 2345 | 2.316296 | -0.094470 | -0.082056 | -0.095886 | -0.001415 | 86.77% | 13.23% |
| 3456 | 2.263345 | +0.022359 | -0.035673 | -0.040231 | -0.004558 | 12.01% | 87.99% |
| ALL | 2.263705 | -0.030164 | -0.033597 | -0.062876 | -0.029278 | 53.54% | 46.46% |

Feature reading:

| seed | old s_q | min090 s_q | old strength | min090 strength | old H delta RMS | min090 H delta RMS | old latent MSE | min090 latent MSE | Q4 old delta | Q4 min090 delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.569798 | 0.459787 | 0.257688 | 0.279810 | 0.062684 | 0.100800 | 0.056989 | 0.112679 | -0.103524 | -0.001850 |
| 2345 | 0.567712 | 0.568090 | 0.256065 | 0.236612 | 0.058327 | 0.055089 | 0.057454 | 0.057436 | -0.199964 | -0.186023 |
| 3456 | 0.448851 | 0.449871 | 0.292674 | 0.281185 | 0.099553 | 0.099314 | 0.120477 | 0.119785 | +0.101288 | +0.030999 |

Interpretation:

- The best fixed global rule is not the real opportunity. min090 is only `-0.003434` RD better than old gate0.25 on the aggregate, but the per-image oracle is `-0.029278` RD better than min090 and `-0.062876` RD better than HCS.
- The oracle selects both modes often enough that the target is a selective reliability controller: old gate0.25 on 53.54% of images and min090 on 46.46%.
- Seed1234 shows the failure mode for fixed min090: lower `s_q`, higher effective strength, larger Householder displacement, and nearly doubled latent quantization MSE.
- Seed3456 shows why fixed old gate0.25 is unsafe: min090 beats old on 87.99% of images and rescues the seed mean.
- Seed2345 is a stable-control case where old remains best and min090 mainly suppresses geometry without materially changing latent MSE.
- The co-adaptation concern should remain framed as evidence-based risk, not proof: detached risk blocks direct gate-gradient pressure on `s_q`, but the whole trained system can still move hyperprior, gate, and RVQ fit jointly through the RD objective.

Next action:

- Implement a detached/frozen-evidence selective reliability controller that approximates the old/min090 oracle, evaluate it against this oracle target on true holdout4096, then rerun the 3-seed checkpoint sweep before moving to paper-scale curves.

Outputs:

- `experiments/analysis/per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_seed1234_hcs500_vs_hcgh_gate025_risk_inv_detach_s044_min090_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_seed2345_hcs250_vs_hcgh_gate025_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_seed2345_hcs250_vs_hcgh_gate025_risk_inv_detach_s044_min090_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed2345_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed2345_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/gate025_min090_oracle_val4096_holdout4096_current_summary.{csv,json,md}`
- `experiments/analysis/gate025_min090_oracle_feature_reading_val4096_holdout4096_current.{json,md}`
- `experiments/analysis/gate025_min090_seed1234_tail_features_val4096_holdout4096_current.{csv,json,md}`
- `experiments/analysis/gate025_min090_seed2345_tail_features_val4096_holdout4096_current.{csv,json,md}`
- `experiments/analysis/gate025_min090_seed3456_tail_features_val4096_holdout4096_current.{csv,json,md}`

## E037: Current-Holdout Selector Analysis and Detached Reliability Configs

Status: Implemented and smoke-tested; full training pending.

Motivation:

- E036 showed a large old/min090 oracle gap, but fixed old gate0.25 and fixed min090 each fail on different seeds.
- Before adding another heuristic, I tested whether current-holdout per-image features can predict when min090 should replace old gate0.25.

Selector analysis:

- Updated `tools/analyze_gate_selector.py` with `--protocol current_holdout` so it reads the paper-facing `start_index=4096`, `max_images=4096` artifacts instead of stale `hcsbest` filenames.
- Ran `tools/analyze_gate_selector.py --protocol current_holdout --output-prefix gate025_min090_selector_val4096_holdout4096_current`.

Key results:

| policy | mean delta RD | vs old gate0.25 | min090 fraction |
|---|---:|---:|---:|
| old gate0.25 | -0.030164 | +0.000000 | 0.000000 |
| min090 risk | -0.033597 | -0.003434 | 1.000000 |
| oracle old/min090 | -0.062876 | -0.032712 | 0.464600 |
| best single feature: old_raw_gate_mean >= 0.260788 | -0.049500 | -0.019336 | 0.356445 |

Interpretation:

- A simple deployable feature, old raw gate mean, recovers more than half of the oracle gap relative to old gate0.25.
- It is still meaningfully weaker than the oracle (`-0.049500` vs `-0.062876`), so a learned local reliability branch is justified.
- Because seed1234 showed that trained risk variants can move quantizer statistics, the reliability branch should use detached/frozen evidence where possible.

Implementation:

- Added `householder_gate_reliability_detach` to `HCGMeanScaleHyperprior` and `build_model`.
- When enabled, the reliability head receives `hyper_features.detach()`, so reliability-head gradients do not flow back through the evidence feature path.
- Added 3 seed configs:
  - `configs/pilot_hcg_rvq_h_gate025_reliability_detach_min05_init099_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_reliability_detach_min05_init099_frozen_seed2345.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_reliability_detach_min05_init099_frozen_seed3456.yaml`

Validation:

- Built the seed3456 detached config and confirmed `householder_gate_reliability_enabled=True`, `householder_gate_reliability_detach=True`, min `0.5`, init `0.99`.
- Ran a 64x64 CPU forward smoke test; output shape was `[1, 3, 64, 64]` and initial reliability multiplier was `0.990000`.
- Checked current-holdout analysis CSV/JSON files for non-finite values: 47 files checked, 0 NaN/Inf values.
- `py_compile` passed for `hcg_rvq/models/hyperprior_rvq.py`, `hcg_rvq/models/builder.py`, and `tools/analyze_gate_selector.py`.
- `git diff --check` passed.

Next action:

- Train the detached reliability controller for seeds 1234/2345/3456 on CUDA device 0 only, then evaluate true-holdout checkpoints and compare against old, min090, threshold selector, and oracle.

Outputs:

- `experiments/analysis/gate025_min090_selector_val4096_holdout4096_current_summary.{md,json}`
- `experiments/analysis/gate025_min090_selector_val4096_holdout4096_current_thresholds.csv`

## E038: Detached Reliability True-Holdout Evaluation and Feature Diagnosis

Status: Done; the naive learned detached reliability branch is rejected as paper-main.

Motivation:

- E037 created detached reliability configs to test whether a learnable controller could approximate the old/min090 oracle while preventing direct reliability-head gradients from moving the evidence path.
- The paper-facing question was not whether reliability control is useful in principle, but whether this free RD-trained detached branch can replace the fixed old/min090 rules.

Protocol:

- Trained/evaluated seeds 1234/2345/3456 with CUDA device 0 only.
- Evaluated `checkpoint_step_250.pth.tar`, `checkpoint_step_500.pth.tar`, and `checkpoint_latest.pth.tar` on OpenImages true holdout4096 with `start_index=4096`, `max_images=4096`.
- Ran per-image intermediate-feature diagnostics for the best detached checkpoint on each seed.

Checkpoint-selected result:

| seed | HCS RD | old gate0.25 RD | min090 RD | detached reliability RD | delta vs HCS |
|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | 2.193095 | 2.228412 | 2.844109 | +0.632634 |
| 2345 | 2.316296 | 2.221826 | 2.234240 | 2.863034 | +0.546738 |
| 3456 | 2.263345 | 2.285704 | 2.227673 | 2.920301 | +0.656956 |
| mean | 2.263705 | 2.233542 | 2.230108 | 2.875814 | +0.612109 |

Feature diagnosis at the selected detached checkpoints:

| seed | s_q_mean | raw gate | reliability mult | effective strength | Householder delta RMS | latent quant MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.563015 | 0.258458 | 0.990675 | 0.256054 | 0.060873 | 0.075152 |
| 2345 | 0.561289 | 0.256581 | 0.990519 | 0.254156 | 0.056665 | 0.075125 |
| 3456 | 0.557039 | 0.254383 | 0.990340 | 0.251931 | 0.054160 | 0.076778 |

Interpretation:

- The failure is decisive across seeds: detached reliability loses to HCS, old gate0.25, and min090 on every seed.
- The reliability multiplier stayed near initialization (`~0.990`), so the failure is not caused by over-suppressing geometry.
- The branch behaved almost like old gate0.25 in gate magnitude but still entered a much worse RD basin. This makes the free RD-trained reliability head unsafe as the current main method.
- Selective reliability control remains promising because E036/E037 showed a strong old/min090 oracle and a useful raw-gate threshold. The next action should constrain the selector rather than let it freely reshape training.
- A newly generated seed3456 detached-vs-HCS per-image comparison was excluded: its detached side matched checkpoint evaluation, but its HCS side was inconsistent with the trusted HCS checkpoint evaluation and prior valid per-image comparisons.

Next action:

- Implement a constrained old/min090 selector: first promote the raw-gate threshold into a reproducible inference-time policy, then test a frozen-evidence lightweight classifier only if the threshold leaves too much oracle gap.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_reliability_detach_min05_init099_seed{1234,2345,3456}_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/per_image_features_hcg_h_gate025_reliability_detach_min05_init099_seed{1234,2345,3456}_step250_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/reliability_detach_min05_init099_val4096_holdout4096_current_summary.{md,json}`

## E039: Fold-Calibrated Old/Min090 Selector Check

Status: Done; selector direction is strengthened.

Motivation:

- E037 found a strong full-holdout single-feature selector, `old_raw_gate_mean >= 0.260788`, but that threshold was selected and evaluated on the same holdout images.
- E038 showed that the free detached reliability head is unsafe, so the next paper-track direction should only survive if a constrained selector generalizes under a stricter split.

Implementation:

- Added `tools/analyze_gate_selector_cv.py`.
- The tool loads the existing current-holdout old/min090 per-image results and intermediate features, calibrates thresholds on train image-index folds, and evaluates on held-out folds.
- No GPU was used for this analysis.

4-fold current-holdout result:

| policy | mean delta RD | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 | -0.030164 | +0.000000 | 0.000000 | n/a | n/a | n/a |
| min090 risk | -0.033597 | -0.003434 | 1.000000 | n/a | n/a | n/a |
| oracle old/min090 | -0.062876 | -0.032712 | 0.464600 | n/a | n/a | n/a |
| CV selected deployable threshold | -0.049023 | -0.018859 | 0.342529 | -0.017248 | -0.094277 | -0.035545 |
| fixed old_raw_gate_mean >= 0.260788 | -0.049500 | -0.019336 | 0.356445 | -0.018744 | -0.094083 | -0.035673 |

Fold details:

| fold | selected feature | direction | threshold | train delta | test delta |
|---:|---|---|---:|---:|---:|
| 0 | old_raw_gate_mean | ge | 0.260803 | -0.050918 | -0.045063 |
| 1 | old_raw_gate_mean | ge | 0.276800 | -0.049620 | -0.048764 |
| 2 | old_raw_gate_mean | ge | 0.260792 | -0.049083 | -0.050590 |
| 3 | old_raw_gate_mean | ge | 0.276797 | -0.048649 | -0.051676 |

Interpretation:

- The constrained selector result is not just a same-holdout threshold artifact. Fold-calibrated thresholds stay close to the full-data threshold and retain most of the gain.
- The selector improves old gate0.25 by `-0.018859` RD under cross-validation and remains much closer to the oracle than either fixed rule.
- This is now the strongest practical direction: keep HCG geometry, avoid the free reliability head, and use frozen/constrained evidence to decide when to switch from old-like geometry to min090-like suppression.

Next action:

- Validate the raw-gate selector on another image slice or dataset, then decide whether a frozen-evidence lightweight classifier is needed to close the remaining `+0.013853` RD gap to the old/min090 oracle.

Outputs:

- `tools/analyze_gate_selector_cv.py`
- `experiments/analysis/gate025_min090_selector_cv_val4096_holdout4096_current.{md,json}`

## E040: Start0 Current Recheck for Old/Min090 Selector Transfer

Status: Done; selector transfer is useful but checkpoint-sensitive.

Motivation:

- E039 showed that the paper-facing holdout selector was not a same-slice artifact.
- The next risk was whether that selector survives on a different OpenImages slice (`start_index=0`) and whether older `reeval_current` artifacts were still trustworthy.
- The user also flagged CUDA device 1 as unreliable, so every GPU run here used only `CUDA_VISIBLE_DEVICES=0`.

Data integrity finding:

- Existing start0 `reeval_current` checkpoint-summary CSVs were inconsistent with current reruns. Example: seed1234 old gate step250 was rechecked with `tools/evaluate_checkpoints.py` and current code gives `RD=2.862647`, not the stale `~2.198` line in the old summary CSV.
- Therefore the start0 claims below use regenerated per-image feature/RD tables, not the stale checkpoint-summary CSVs.

Regenerated start0 current artifacts:

- HCS per-image diagnostics for seeds 1234/2345/3456.
- old gate0.25 per-image diagnostics for the transfer checkpoint plan: seed1234 step250, seed2345 step250, seed3456 step500.
- min090 inverse/detached risk diagnostics for the transfer checkpoint plan: seed1234 step500, seed2345 step250, seed3456 step500.
- Additional slice-best diagnostic for seed3456 step250 old/risk, because start0 conclusions change materially with checkpoint step.
- Added `tools/analyze_gate_selector_start0_recheck.py` to combine HCS/old/risk feature tables into selector, oracle, CV, and correlation summaries.

Transfer checkpoint plan result:

| policy | mean RD | mean delta RD vs HCS | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.931295 | +0.000000 | n/a | 0.000000 | n/a | n/a | n/a |
| old gate0.25 | 2.910176 | -0.021119 | +0.000000 | 0.000000 | -0.043587 | -0.088218 | +0.068447 |
| min090 risk | 2.915664 | -0.015631 | +0.005488 | 1.000000 | +0.005551 | -0.075452 | +0.023008 |
| oracle old/min090 | 2.873311 | -0.057984 | -0.036865 | 0.470866 | n/a | n/a | n/a |
| fixed old_raw_gate_mean >= 0.260788 | 2.898147 | -0.033148 | -0.012028 | 0.379150 | -0.035018 | -0.087433 | +0.023008 |
| CV selected deployable threshold | 2.895861 | -0.035434 | -0.014315 | 0.332357 | -0.041186 | -0.088194 | +0.023077 |

Slice-best diagnostic result:

| policy | mean RD | mean delta RD vs HCS | vs old gate0.25 | min090 fraction | seed1234 | seed2345 | seed3456 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.931295 | +0.000000 | n/a | 0.000000 | n/a | n/a | n/a |
| old gate0.25 | 2.891076 | -0.040219 | +0.000000 | 0.000000 | -0.043587 | -0.088218 | +0.011148 |
| min090 risk | 2.908039 | -0.023256 | +0.016963 | 1.000000 | +0.005551 | -0.075452 | +0.000133 |
| oracle old/min090 | 2.868445 | -0.062850 | -0.022631 | 0.449137 | n/a | n/a | n/a |
| fixed old_raw_gate_mean >= 0.260788 | 2.894194 | -0.037101 | +0.003118 | 0.045817 | -0.035018 | -0.087433 | +0.011148 |
| CV selected deployable threshold | 2.887695 | -0.043600 | -0.003381 | 0.626546 | -0.054770 | -0.080970 | +0.004941 |

Interpretation:

- The fixed holdout threshold transfers in the important sense that it improves transfer-checkpoint old gate0.25 by `-0.012028` RD and CV selection improves it by `-0.014315` RD.
- However, the transfer plan carries seed3456 old/risk step500 to start0, where both are worse than HCS. The selector can rescue old-vs-risk choice but cannot fix a bad checkpoint choice.
- When seed3456 uses the start0 slice-best step250, old gate0.25 is already stronger. The holdout fixed threshold no longer helps old gate, but start0 CV can still find a small gain (`-0.003381` vs old).
- This means the paper-safe claim is not just "raw gate threshold works everywhere." The stronger and more accurate claim is: frozen image-level reliability evidence is real, but it must be paired with explicit checkpoint selection and validation protocol.
- Feature correlations support the mechanism: in transfer mode, `old_raw_gate_mean` is the best deployable threshold feature and has negative correlation with min090-old delta (`-0.219930`), while latent quantization MSE remains the strongest nonselector difficulty signal.

Next action:

- Keep the old/min090 selector as the main constrained reliability direction, but implement it behind a validation-selected checkpoint protocol.
- Do not promote a learned reliability head yet.
- Add a small artifact validator to mark stale start0 checkpoint-summary CSVs as excluded, then move to a deployable inference-time selector implementation and evaluate on a reporting split.

Outputs:

- `tools/analyze_gate_selector_start0_recheck.py`
- `experiments/analysis/gate025_min090_selector_start0_current_recheck_transfer.{md,json}`
- `experiments/analysis/gate025_min090_selector_start0_current_recheck_transfer_thresholds.csv`
- `experiments/analysis/gate025_min090_selector_start0_current_recheck_slice_best.{md,json}`
- `experiments/analysis/gate025_min090_selector_start0_current_recheck_slice_best_thresholds.csv`
- `experiments/analysis/per_image_features_hcs_seed{1234,2345,3456}_step*_val4096_start0_current_recheck.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_seed3456_step500_val4096_start0_current_recheck.{csv,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed3456_step500_val4096_start0_current_recheck.{csv,json}`

## E041: Start0 Artifact Consistency Audit

Status: Done; stale start0 checkpoint-summary CSVs are now explicitly excluded.

Motivation:

- E040 found that older `*_reeval_current.csv` checkpoint-summary files can be inconsistent with current-code start0 reruns.
- Before adding another reliability-selector implementation, the result provenance needed a machine-checkable guardrail so paper tables do not accidentally mix stale checkpoint-summary rows with regenerated per-image diagnostics.

Implementation:

- Added `tools/audit_start0_artifacts.py`.
- The audit compares legacy checkpoint-summary CSV rows against current per-image/debug start0 reference artifacts for HCS, old gate0.25, and inverse/detached min090 risk.
- It also checks trusted JSON references for non-finite values. This run is CPU-only; the underlying regenerated GPU artifacts from E040 were produced with `CUDA_VISIBLE_DEVICES=0`.

Audit result:

| family | checked | excluded legacy | legacy RD mean | reference RD mean | legacy-reference |
|---|---:|---:|---:|---:|---:|
| hcs | 3 | 3 | 2.267590 | 2.931295 | -0.663705 |
| old_gate025 | 3 | 3 | 2.235253 | 2.891076 | -0.655822 |
| risk_inv_detach_min090 | 3 | 3 | 2.246750 | 2.908039 | -0.661289 |

Reference consistency:

- Independent debug CSVs and per-image JSON references match for the available HCS rows and old gate seed1234 row within `1e-4`.
- All checked trusted JSON references contain finite numeric values; no NaN/Infinity was found in this audit.

Decision:

- Do not use the legacy start0 checkpoint-summary files named like `pilot_*_openimages_val4096_reeval_current.csv` for start0 paper claims.
- Use regenerated per-image/debug artifacts and the selector summaries from E040/E041 as the trusted start0 record.
- This does not change the current scientific conclusion: old/min090 selection remains promising, but must be reported under a validation-selected checkpoint protocol.

Next action:

- Move from retrospective old/min090 selection to a deployable selector evaluation on a reporting split, while keeping artifact consistency checks in the loop.

Outputs:

- `tools/audit_start0_artifacts.py`
- `experiments/analysis/start0_artifact_consistency_audit.{md,json}`

## E042: Validation-Calibrated Reporting Selector Protocol

Status: Done; deployable selector evidence is now separated into calibration and reporting splits.

Motivation:

- E039/E040 showed that old gate0.25 and inverse/detached min090 risk have complementary per-image wins.
- The remaining question was whether the selector is just same-split threshold fitting or whether a policy calibrated on one split can be applied unchanged to a reporting split.
- This also tests the paper-facing protocol implied by E041: use trusted regenerated per-image artifacts only, and keep checkpoint selection explicit.

Implementation:

- Added `tools/analyze_selector_reporting_protocol.py`.
- Calibration split: OpenImages `start_index=4096` current holdout.
- Reporting split: OpenImages `start_index=0` current recheck.
- The script reads existing threshold CSVs instead of recomputing the full threshold search, then applies the calibrated deployable policy unchanged to the reporting rows.

Calibrated policy:

- `old_raw_gate_mean >= 0.260788`
- calibration mean delta: `-0.049500` vs HCS
- calibration improvement over old gate0.25: `-0.019336` RD
- calibration oracle gap closed: `0.591113`

Reporting result:

| split | HCS RD | old gate RD | min090 RD | oracle RD | calibrated policy RD | calibrated vs old | min090 fraction | oracle gap closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation holdout4096 | 2.263705 | 2.233542 | 2.230108 | 2.200830 | 2.214205 | -0.019336 | 0.356445 | 0.591113 |
| reporting start0 transfer | 2.931295 | 2.910176 | 2.915664 | 2.873311 | 2.898147 | -0.012028 | 0.379150 | 0.326284 |
| reporting start0 slice_best | 2.931295 | 2.891076 | 2.908039 | 2.868445 | 2.894194 | +0.003118 | 0.045817 | -0.137773 |

Same-split best deployable policies:

| split | best deployable policy | mean delta | min090 fraction |
|---|---|---:|---:|
| validation holdout4096 | `old_raw_gate_mean >= 0.260788` | -0.049500 | 0.356445 |
| reporting start0 transfer | `old_raw_gate_mean >= 0.275260` | -0.036222 | 0.326742 |
| reporting start0 slice_best | `risk_y_error_rms <= 0.145200` | -0.043627 | 0.623779 |

Interpretation:

- The calibrated raw-gate policy transfers to the reporting transfer protocol and improves old gate0.25 by `-0.012028` RD without using reporting labels to choose the threshold.
- It closes about one third of the reporting transfer old/min090 oracle gap (`0.326284`), so the reliability signal is real but not saturated.
- The same policy does not improve the slice-best checkpoint protocol, where old gate0.25 is already stronger and the best deployable feature changes to `risk_y_error_rms`.
- This strengthens the current paper stance: reliability-controlled geometry is promising, but the current old/min090 switch is still a multi-checkpoint diagnostic. It must be converted into a single-checkpoint controller before being claimed as the final codec. Do not claim a universal raw-gate threshold.

Next action:

- Turn this retrospective selector into an actual inference-time/evaluation config path, or evaluate the same protocol on another reporting split/rate before promoting it as the main paper result.

Outputs:

- `tools/analyze_selector_reporting_protocol.py`
- `experiments/analysis/gate025_min090_selector_reporting_protocol.{md,json}`

## E043: Selector Claim-Readiness / Single-Codec Audit

Status: Done; selector headroom is strong, but current old/min090 switching is not yet a single-codec result.

Motivation:

- E042 separated calibration and reporting splits, but the wording still needed one more audit: does the selected policy correspond to a single HCG-RVQ checkpoint, or does it switch between separately trained old and min090 checkpoints?
- This matters for an international-paper claim. A model-switching diagnostic is useful evidence, but a unified single-checkpoint controller is a stronger and cleaner method contribution.

Implementation:

- Added `tools/audit_selector_claim_readiness.py`.
- The audit reads `gate025_min090_selector_reporting_protocol.json` and the old/min090 seed configs.
- It checks whether old and min090 share architecture family, gate policy, run/checkpoint identity, and initialization.

Findings:

| property | result | implication |
|---|---|---|
| same architecture family | `True` | old/min090 are comparable HCG-RVQ-H variants. |
| same gate policy | `False` | min090 is a different deterministic gate rule, not just a reporting-time threshold. |
| same run/checkpoint | `False` | current per-image selector is a multi-checkpoint/codec-selection diagnostic. |
| same initialization | `True` | shared initialization reduces confounding but does not make the checkpoints identical. |

Claim-tier decision:

| candidate | status | paper use |
|---|---|---|
| old gate0.25 | single-checkpoint variant | Safe as the current main single-model HCG geometry result. |
| min090 risk | single-checkpoint diagnostic | Useful ablation, but not current main because transfer/slice-best behavior is weaker. |
| calibrated old/min090 selector | multi-checkpoint diagnostic | Strong evidence that reliability control has headroom; do not present as final single-codec method yet. |

Additional note:

- A one-bit image-level model flag costs only about `0.00001526` bpp for a 256x256 crop, so signaling overhead is not the main problem.
- The real issue is protocol strength: a model-switching ensemble is a weaker paper claim than a unified HCG-RVQ checkpoint.

Next action:

- Promote selector evidence into a single-checkpoint reliability controller, then compare that unified model against HCS, old gate0.25, min090 risk, and the multi-checkpoint selector headroom.
- Until that is done, use the selector result as mechanism/headroom evidence, not as the final method row.

Outputs:

- `tools/audit_selector_claim_readiness.py`
- `experiments/analysis/selector_claim_readiness.{md,json}`

## E044: Related-Work and Official-Code Re-reading

Status: Done; the paper/code reading was refreshed after the selector claim-readiness audit.

Motivation:

- The project had reached an important decision point: whether to promote the old/min090 selector as a main result, pivot toward existing VQ/RD methods, or keep pushing HCG geometry into a single-checkpoint controller.
- The re-reading focused on papers and repositories that could invalidate, sharpen, or strengthen the HCG-RVQ claim.

Sources re-checked:

| area | source | project implication |
|---|---|---|
| RD-aware VQ | RDVQ paper and repository | strongest VQ-RD competitor; useful future fallback for differentiable index/rate training, but not a reason to abandon HCG geometry. |
| hyperprior VQ entropy | HVQ-CGIC paper | closest entropy-only neighbor; makes HCS/index-prior and geometry ablations mandatory. |
| adaptive vector quantization | Adaptive LVQ / MLVIC | supports quantizer adaptation as a publishable direction; HCG differs by local hyperprior-generated shared-RVQ geometry. |
| strong entropy model | DCAE repository and model code | important future strong-baseline integration; also reinforces the need for a single codec path. |
| VQ stability | SimVQ, FSQ/vector-quantize-pytorch, Rotation Trick | useful stability toolbox, but current bottleneck is geometry reliability more than dead-code collapse. |
| strong LIC backbones | HPCM, MLIC/ELIC, MambaIC | future plug-in comparisons after the simple-family HCG mechanism is stable. |

Decision:

- Do not switch the project main line to RDVQ/FSQ/DCAE yet.
- Keep the central claim from `prompt.txt`: the hyperprior should generate local quantizer geometry, not only entropy parameters.
- Use the selector results as mechanism/headroom evidence, but convert them into a unified single-checkpoint reliability controller before using them as the main method row.

Progress assessment:

- The research is moving well relative to the goal: HCS/index-prior is stable, HCG-H geometry has positive pilot evidence, and reliability analysis has identified when geometry should be weakened.
- The paper is not claim-complete yet. The missing bridge is a deployable single-checkpoint controller, followed by multi-rate and stronger-baseline comparisons.

Outputs:

- `docs/research_reading_notes.md`


## E045: Posthoc Single-Checkpoint Risk Controller Audit

Status: Done; the posthoc old-weights/min090 controller is rejected as the main path.

Motivation:

- E043/E044 identified the strongest old/min090 reliability result as a multi-checkpoint diagnostic, so the next obvious test was a stricter single-checkpoint question: can the min090 inverse/detached risk gate be applied at inference/evaluation time to the already trained old gate0.25 weights?
- This would have been attractive because it keeps one checkpoint and changes only the deterministic Householder gate rule.

Important audit correction:

- The first posthoc per-image comparison CSVs looked superficially positive for seeds 1234/2345, but their HCS side was not the trusted paper-facing current-holdout HCS artifact.
- The HCS mean inside those posthoc compare CSVs was `+0.625` to `+0.678` RD worse than the trusted HCS rows.
- Therefore those delta values are excluded from paper-facing interpretation. The posthoc model itself is instead evaluated by absolute per-image feature RD and matched by image path to the trusted current-holdout HCS/old/min090 artifacts.

Trusted matched result:

| method | mean RD | delta vs HCS |
|---|---:|---:|
| HCS | 2.263705 | +0.000000 |
| old gate0.25 | 2.233542 | -0.030164 |
| trained min090 risk | 2.230108 | -0.033597 |
| posthoc min090 on old weights | 2.915251 | +0.651546 |

Per seed:

| seed | HCS | old gate0.25 | trained min090 | posthoc oldw/min090 | posthoc-HCS | posthoc-old |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | 2.193095 | 2.228412 | 2.831683 | +0.620209 | +0.638588 |
| 2345 | 2.316296 | 2.221826 | 2.234240 | 2.908287 | +0.591991 | +0.686461 |
| 3456 | 2.263345 | 2.285704 | 2.227673 | 3.005783 | +0.742438 | +0.720079 |

Intermediate-feature readout:

| seed | s_q | raw gate | risk mult | effective strength | delta RMS | latent qMSE | index bpp | dead code |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.562929 | 0.258263 | 0.919659 | 0.237529 | 0.056725 | 0.075191 | 0.113006 | 0.196959 |
| 2345 | 0.560737 | 0.256579 | 0.920100 | 0.236092 | 0.053089 | 0.075460 | 0.112873 | 0.196751 |
| 3456 | 0.436136 | 0.295341 | 0.950925 | 0.281135 | 0.093935 | 0.164911 | 0.114679 | 0.185289 |

Interpretation:

- Posthoc gate weakening is not equivalent to training a min090 checkpoint. Even an apparently modest deterministic multiplier change creates a decoder/quantizer mismatch and sharply worsens RD.
- The failure increases on harder HCS-difficulty quartiles; for seed3456 Q4, posthoc-HCS is `+1.092873` RD.
- This result strengthens, rather than weakens, the protocol lesson: reliability control must be part of the trained codec or be constrained with validation-selected safeguards. It should not be retrofitted onto a checkpoint whose decoder/codebook adapted to the old geometry policy.
- No NaN/CUDA-device issue was observed in this run; feature diagnostics were run on `cuda:0`.

Next action:

- Do not promote posthoc min090-on-old-weights.
- Keep old gate0.25 as the current safest single-checkpoint HCG-H row and trained min090 as a deterministic ablation.
- Design the next unified controller so the reliability signal is present during training, but avoid the previously failed free reliability head: start from conservative low-capacity/regularized control, validation-selected checkpoints, and explicit feature-distribution monitoring.

Outputs:

- `tools/analyze_posthoc_single_checkpoint_controller.py`
- `experiments/analysis/posthoc_single_checkpoint_controller_val4096_holdout4096_current.{md,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_posthoc_min090_oldw_seed{1234,2345,3456}_step{250/500}_val4096_holdout4096_current.{csv,json}`


## E046: Old-Checkpoint Risk Fine-Tune Probe on Seed3456

Status: Done; 250-step fine-tuning from the old gate0.25 checkpoint made the risk-gated model worse.

Motivation:

- E045 showed that applying min090 risk to old gate0.25 weights posthoc is invalid and severely worsens RD.
- The next check was whether the mismatch is simply a decoder/codebook adaptation issue: initialize from the old gate0.25 seed3456 checkpoint, enable the trained min090 inverse/detached risk rule, and fine-tune for 250 steps on the same frozen-base protocol.

Setup:

- Config: `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_ftold500_seed3456.yaml`
- Init: `experiments/pilot_hcg_rvq_h_gate025_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar`
- Output checkpoint: `experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_ftold500_g64_l1_k128_lambda0035_seed3456/checkpoint_step_250.pth.tar`
- Device: `cuda:0`

RD result on trusted OpenImages holdout4096 seed3456:

| method | mean RD | delta vs HCS |
|---|---:|---:|
| HCS | 2.263345 | +0.000000 |
| old gate0.25 | 2.285704 | +0.022359 |
| trained min090 | 2.227673 | -0.035673 |
| posthoc min090 on old weights | 3.005783 | +0.742438 |
| ftold500 min090 250-step | 3.063368 | +0.800023 |

Feature shift:

| model | s_q | strength | latent qMSE |
|---|---:|---:|---:|
| posthoc oldw/min090 | 0.436136 | 0.281135 | 0.164911 |
| ftold500 min090 | 0.347017 | 0.311987 | 0.370800 |

Interpretation:

- The old-checkpoint fine-tune did not recover the posthoc failure; it amplified it.
- The model moved into a worse geometry/scale regime: lower `s_q`, stronger Householder transform, and much larger latent quantization MSE.
- This argues against a naive old-checkpoint warm-start for the next single-checkpoint controller.
- The next controller should be more conservative: lower LR, explicit `rho_householder_delta` or strength regularization, or a staged schedule that keeps `s_q` and geometry near validated regimes before allowing decoder/codebook adaptation.
- No NaN/CUDA issue was observed; this run used `cuda:0`.

Outputs:

- `experiments/analysis/ftold500_risk_min090_seed3456_val4096_holdout4096_current.{md,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_ftold500_seed3456_step250_val4096_holdout4096_current.{csv,json}`

## E047: Conservative Risk Floor min095 Audit

Status: Done; `householder_gate_risk_min=0.95` is rejected as the next main single-checkpoint controller.

Motivation:

- E045/E046 showed that posthoc risk control and naive old-checkpoint fine-tuning are unsafe.
- The next conservative test was to keep the inverse/detached reliability idea but raise the lower floor from `0.90` to `0.95`, so the controller stays closer to old gate0.25 while still encoding a weak reliability signal during training.

Setup:

- Configs:
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_seed2345.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_seed3456.yaml`
- Checkpoints:
  - `experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_500.pth.tar`
  - `experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_g64_l1_k128_lambda0035_seed2345/checkpoint_step_500.pth.tar`
  - `experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min095_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar`
- Feature diagnostics: OpenImages holdout4096, `start_index=4096`, `cuda:0`.

Trusted matched result:

| method | mean RD | delta vs HCS |
|---|---:|---:|
| HCS | 2.263705 | +0.000000 |
| old gate0.25 | 2.233542 | -0.030164 |
| trained min090 risk | 2.230108 | -0.033597 |
| trained min095 risk | 2.972039 | +0.708334 |

Per seed:

| seed | HCS | old gate0.25 | min090 | min095 | min095-HCS | min095-old |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | 2.193095 | 2.228412 | 2.914509 | +0.703035 | +0.721414 |
| 2345 | 2.316296 | 2.221826 | 2.234240 | 3.033045 | +0.716749 | +0.811219 |
| 3456 | 2.263345 | 2.285704 | 2.227673 | 2.968563 | +0.705218 | +0.682859 |

Intermediate-feature readout:

| seed | s_q | raw gate | risk mult | effective strength | delta RMS | latent qMSE | index bpp | dead code |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0.447285 | 0.296701 | 0.974014 | 0.289132 | 0.098164 | 0.153295 | 0.114517 | 0.187649 |
| 2345 | 0.447380 | 0.296743 | 0.974004 | 0.289175 | 0.099153 | 0.147357 | 0.114426 | 0.187990 |
| 3456 | 0.437947 | 0.296912 | 0.975229 | 0.289705 | 0.097146 | 0.162526 | 0.114660 | 0.186167 |

Interpretation:

- Raising the risk floor from `0.90` to `0.95` does not rescue the unified controller. It fails on all three seeds and all HCS-difficulty quartiles.
- The multiplier itself is conservative (`risk mult` about `0.974`), yet raw gate and effective geometry strength grow to about `0.297` and `0.289`.
- Compared with the useful old/min090 checkpoints, the min095 run has much larger latent quantization MSE and large Q4 degradation (`+1.02` to `+1.11` RD vs HCS).
- This isolates the failure mode: the problem is not simply too much gate suppression. The training schedule permits scale/geometry co-adaptation into a bad regime.

Next action:

- Do not continue with plain floor tuning as the main path.
- Keep old gate0.25 as the safest current single-checkpoint HCG-H row and trained min090 as a deterministic ablation.
- Move to a directly constrained geometry controller: lower LR and/or explicit `householder_delta`/strength regularization, with feature-distribution checks after checkpoint evaluation.
- Continue using `cuda:0`; no NaN or CUDA device issue was observed in this run.

Outputs:

- `tools/analyze_risk_floor_min095.py`
- `experiments/analysis/risk_floor_min095_val4096_holdout4096_current.{md,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min095_seed{1234,2345,3456}_step500_val4096_holdout4096_current.{csv,json}`

## E048: Direct Householder Delta Regularization Probe

Status: Done; direct `rho_householder_delta=0.10` is rejected as the next controller.

Motivation:

- E047 showed that conservative risk-floor tuning is not enough; the model can still co-adapt scale/geometry into a bad regime.
- The next probe tested whether directly shrinking the Householder displacement itself stabilizes the fragile seed3456 case.

Setup:

- Config: `configs/pilot_hcg_rvq_h_gate025_delta_reg010_frozen_seed3456.yaml`
- Checkpoint: `experiments/pilot_hcg_rvq_h_gate025_delta_reg010_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar`
- Feature diagnostics: OpenImages holdout4096, `start_index=4096`, `cuda:0`.
- Analysis: `tools/analyze_delta_reg010_seed3456.py`

Trusted seed3456 path-matched result:

| method | mean RD | delta vs HCS | delta vs old | delta vs min090 |
|---|---:|---:|---:|---:|
| HCS | 2.263345 | +0.000000 | n/a | n/a |
| old gate0.25 | 2.285704 | +0.022359 | +0.000000 | +0.058031 |
| trained min090 risk | 2.227673 | -0.035673 | -0.058031 | +0.000000 |
| trained min095 risk | 2.968563 | +0.705218 | +0.682859 | +0.740890 |
| delta_reg010 | 3.079296 | +0.815950 | +0.793592 | +0.851623 |

Intermediate-feature readout:

| method | s_q | raw gate | effective strength | delta RMS | latent qMSE | index bpp | dead code |
|---|---:|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.448851 | 0.292674 | 0.292674 | 0.099553 | 0.120477 | 0.119917 | 0.026749 |
| trained min090 risk | 0.449871 | 0.296461 | 0.281185 | 0.099314 | 0.119785 | 0.119938 | 0.026701 |
| trained min095 risk | 0.437947 | 0.296912 | 0.289705 | 0.097146 | 0.162526 | 0.114660 | 0.186167 |
| delta_reg010 | 0.465360 | 0.256979 | 0.256979 | 0.038196 | 0.206089 | 0.112264 | 0.216028 |

Interpretation:

- The regularizer worked mechanically: Householder delta RMS dropped from the useful old/min090 regime of about `0.099` to `0.038`.
- The RD result became worse than HCS, old gate0.25, trained min090, and even the failed min095 risk floor.
- The failure is not "geometry should be removed"; it is "geometry should not be collapsed toward zero." Shrinking displacement too strongly increases latent quantization MSE and damages hard images, especially Q4 (`+1.117536` RD vs HCS).

Next action:

- Do not run a 3-seed `delta_reg010` sweep.
- Replace zero-shrink regularization with a target/anchor controller that keeps geometry near the validated old/min090 regime, or use a much weaker/staged version.
- Candidate implementation: add a target loss for `householder_delta_rms` or `householder_strength`, then evaluate seed3456 before any multi-seed sweep.

Outputs:

- `tools/analyze_delta_reg010_seed3456.py`
- `experiments/analysis/delta_reg010_seed3456_val4096_holdout4096_current.{md,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_delta_reg010_seed3456_step500_val4096_holdout4096_current.{csv,json}`

## E049: Householder Delta Target Probe

Status: Done; matching the old/min090 delta RMS target alone is also rejected.

Motivation:

- E048 showed that shrinking Householder displacement toward zero is harmful.
- The next probe added a target loss so the model should keep `householder_delta_rms` near the useful old/min090 regime instead of collapsing it.

Implementation:

- Added loss fields in `hcg_rvq/losses.py`:
  - `rho_householder_delta_target`
  - `householder_delta_target`
- Config: `configs/pilot_hcg_rvq_h_gate025_delta_target095_rho5_frozen_seed3456.yaml`
- Checkpoint: `experiments/pilot_hcg_rvq_h_gate025_delta_target095_rho5_frozen_g64_l1_k128_lambda0035_seed3456/checkpoint_step_500.pth.tar`
- Device: `cuda:0`; no NaN observed.

Trusted seed3456 path-matched result:

| method | mean RD | delta vs HCS | delta vs old | delta vs min090 |
|---|---:|---:|---:|---:|
| HCS | 2.263345 | +0.000000 | n/a | n/a |
| old gate0.25 | 2.285704 | +0.022359 | +0.000000 | +0.058031 |
| trained min090 risk | 2.227673 | -0.035673 | -0.058031 | +0.000000 |
| delta_reg010 | 3.079296 | +0.815950 | +0.793592 | +0.851623 |
| delta_target095_rho5 | 3.332857 | +1.069512 | +1.047153 | +1.105184 |

Intermediate-feature readout:

| method | s_q | strength | delta RMS | latent qMSE | index bpp | dead code |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.448851 | 0.292674 | 0.099553 | 0.120477 | 0.119917 | 0.026749 |
| trained min090 risk | 0.449871 | 0.281185 | 0.099314 | 0.119785 | 0.119938 | 0.026701 |
| delta_reg010 | 0.465360 | 0.256979 | 0.038196 | 0.206089 | 0.112264 | 0.216028 |
| delta_target095_rho5 | 0.488443 | 0.245312 | 0.092294 | 0.293593 | 0.108350 | 0.261705 |

Interpretation:

- The target loss succeeded at matching the intended displacement scale: delta RMS reached `0.092294`.
- However RD became even worse than delta_reg010. The model can satisfy the delta target while moving `s_q`, codebook usage, and latent quantization MSE into a bad regime.
- Therefore, delta magnitude alone is not the missing reliability control variable.

Next action:

- Do not sweep this target controller as a main method.
- Move from scalar geometry-magnitude control to full conditioning/quantization anchoring: anchor `log_s_q`/`u` or latent quantization error regime against a validated checkpoint, then reevaluate seed3456.
- Keep old gate0.25 as the safest single-checkpoint HCG-H row and min090/selector as reliability-headroom diagnostics.

Outputs:

- `tools/analyze_delta_target095_rho5_seed3456.py`
- `experiments/analysis/delta_target095_rho5_seed3456_val4096_holdout4096_current.{md,json}`
- `experiments/analysis/per_image_features_hcg_h_gate025_delta_target095_rho5_seed3456_step500_val4096_holdout4096_current.{csv,json}`

## E050: Old-Conditioning Anchor Probe

Status: Done; the full old-frame anchor is rejected as a main controller.

Motivation:

- E048/E049 showed that scalar Householder delta shrinkage or target matching is not enough.
- The next hypothesis was that min090 risk control might become stable if the model is anchored to the validated old gate0.25 conditioning state.
- This directly tests whether preserving old `log_s_q`/`u` behavior while applying conservative inverse/detached risk can keep the codec in a decoder-compatible quantization regime.

Implementation:

- Added `train.anchor_config` support in `train.py`, so the anchor checkpoint can be built from the old gate0.25 config while the current model uses the risk-gated config.
- Anchor loading is non-strict to tolerate unused newly added heads; the only missing old-checkpoint keys were `householder_reliability_head.weight` and `.bias`.
- Config: `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_u05_logs01_frozen_seed1234.yaml`
- Eval config with anchor losses disabled: `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_u05_logs01_frozen_seed1234_eval.yaml`
- Anchor checkpoint: `experiments/pilot_hcg_rvq_h_gate025_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_250.pth.tar`
- Loss weights: `rho_anchor_mu=0.05`, `rho_anchor_log_s=0.10`, `rho_anchor_u=0.50`
- Device: `cuda:0`; no NaN observed.

Trusted seed1234 holdout4096 checkpoint result:

| method | checkpoint | mean RD | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|
| HCS | step500 | 2.211475 | - | - | - |
| old gate0.25 | step250 | 2.193095 | - | - | 0.779345 |
| trained min090 risk | step500 | 2.228412 | - | - | 0.778710 |
| old-anchor min090 risk | step250 | 2.375139 | 0.313636 | 21.120743 | 0.749647 |
| old-anchor min090 risk | step500 | 2.413253 | 0.307912 | 21.014136 | 0.775481 |

Best anchored result deltas:

| comparison | delta RD |
|---|---:|
| anchored - HCS | +0.163664 |
| anchored - old gate0.25 | +0.182044 |
| anchored - trained min090 | +0.146727 |

Intermediate-feature readout for the best anchored checkpoint:

| method | s_q | raw gate | risk multiplier | effective strength | delta RMS | latent qMSE | index bpp | dead code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.569798 | 0.257688 | n/a | 0.257688 | 0.062684 | 0.056989 | 0.117960 | 0.034916 |
| trained min090 risk | 0.459787 | 0.295823 | 0.944806 | 0.279810 | 0.100800 | 0.112679 | 0.119754 | 0.027657 |
| old-anchor min090 risk | 0.566673 | 0.252133 | 0.919107 | 0.231738 | 0.054844 | 0.059710 | 0.117619 | 0.035179 |

Interpretation:

- The anchor worked mechanically: `s_q`, latent qMSE, index bpp, and dead-code ratio are close to the old gate0.25 operating point and much safer than the trained min090 risk checkpoint.
- But the RD and especially MS-SSIM are worse. The effective geometry became under-powered: strength dropped to `0.231738` and delta RMS to `0.054844`, below the old gate0.25 values.
- Therefore preserving quantization error/codebook statistics is not sufficient. The controller must also preserve decoder-compatible geometry strength and perceptual structure.
- A full `u` anchor with risk gating appears to over-constrain the model and suppress useful HCG geometry rather than selectively stabilizing it.

Next action:

- Do not sweep this full old-conditioning anchor as a main method.
- Test a weaker/staged anchor that avoids pinning the full Householder direction `u`.
- Most promising next probe: anchor `log_s_q`/scale only and add a light strength or delta floor/target so effective HCG geometry does not collapse below the validated old gate0.25 regime.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_u05_logs01_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_anchor_old_u05_logs01_seed1234_step250_val4096_holdout4096_current.{csv,json}`

## E051: Scale-Only Old-Anchor Probe

Status: Done; scale-only anchoring is also rejected as a main controller, but it gives a clearer next constraint.

Motivation:

- E050 showed that full old-conditioning anchoring (`mu_q`, `log_s_q`, `u`) preserves qMSE/codebook statistics but suppresses useful geometry and hurts RD/MS-SSIM.
- The next question was whether anchoring only scale (`log_s_q`) can keep the quantizer numerically stable while leaving the Householder direction free.

Implementation:

- Configs:
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly01_frozen_seed1234.yaml`
  - `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly10_frozen_seed1234.yaml`
- Eval configs disable anchor losses for validation.
- Anchor checkpoint/config are the same old gate0.25 seed1234 step250 pair used in E050.
- Device: `cuda:0`; no NaN observed.

Trusted seed1234 holdout4096 checkpoint result:

| method | checkpoint | mean RD | delta vs HCS | delta vs old | delta vs min090 | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| HCS | step500 | 2.211475 | +0.000000 | n/a | n/a | n/a |
| old gate0.25 | step250 | 2.193095 | -0.018380 | +0.000000 | n/a | 0.779345 |
| trained min090 risk | step500 | 2.228412 | +0.016937 | +0.035317 | +0.000000 | 0.778710 |
| full old-anchor min090 risk | step250 | 2.375139 | +0.163664 | +0.182044 | +0.146727 | 0.749647 |
| log_s-only 0.10 | step500 | 2.365288 | +0.153813 | +0.172193 | +0.136876 | 0.779420 |
| log_s-only 1.00 | step500 | 2.356073 | +0.144598 | +0.162978 | +0.127661 | 0.779629 |

Intermediate-feature readout:

| method | s_q | strength | delta RMS | latent qMSE | index bpp | dead code |
|---|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.569798 | 0.257688 | 0.062684 | 0.056989 | 0.117960 | 0.034916 |
| trained min090 risk | 0.459787 | 0.279810 | 0.100800 | 0.112679 | 0.119754 | 0.027657 |
| full old-anchor | 0.566673 | 0.231738 | 0.054844 | 0.059710 | 0.117619 | 0.035179 |
| log_s-only 0.10 | 0.454945 | 0.279498 | 0.100042 | 0.117664 | 0.119738 | 0.027483 |
| log_s-only 1.00 | 0.465048 | 0.279093 | 0.099012 | 0.113027 | 0.119714 | 0.028025 |

Per-image / difficulty analysis for log_s-only 1.00:

| HCS quartile | old-HCS | min090-HCS | full-anchor-HCS | log_s 0.10-HCS | log_s 1.00-HCS | log_s 1.00-min090 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 easy | +0.035546 | +0.033191 | +0.136500 | +0.097290 | +0.097270 | +0.064079 |
| Q2 | +0.009905 | +0.023863 | +0.147064 | +0.125660 | +0.121375 | +0.097512 |
| Q3 | -0.015446 | +0.012545 | +0.167408 | +0.166153 | +0.155681 | +0.143136 |
| Q4 hard | -0.103524 | -0.001850 | +0.203685 | +0.226150 | +0.204068 | +0.205917 |

Feature correlation with log_s-only 1.00 RD degradation vs HCS:

| feature | Pearson r |
|---|---:|
| HCS RD difficulty | +0.203128 |
| s_q | -0.522559 |
| Householder strength | +0.543714 |
| Householder delta RMS | +0.707003 |
| latent qMSE | +0.343302 |
| risk multiplier | +0.533519 |

Interpretation:

- log_s-only anchoring restores MS-SSIM compared with full anchoring and slightly improves RD over full anchoring, but it remains far worse than HCS, old gate0.25, and trained min090.
- Increasing `rho_anchor_log_s` from `0.10` to `1.00` only moves `s_q` from `0.454945` to `0.465048`, still far from old gate0.25 (`0.569798`). The operating point remains essentially min090-like.
- The hard-image tail is the decisive failure. old gate0.25 improves Q4 by `-0.103524` RD vs HCS, while log_s-only 1.00 worsens Q4 by `+0.204068` and is `+0.205917` worse than trained min090 in Q4.
- Degradation correlates most strongly with Householder delta RMS (`r=+0.707003`) and strength (`r=+0.543714`), while higher `s_q` is protective (`r=-0.522559`). This suggests the next controller must jointly bound scale and geometry amplitude in hard/high-risk regions, not only anchor scale globally.

Next action:

- Do not sweep scale-only anchors as a main method.
- The next probe should be selective, not global: reduce geometry only where local risk/large delta predicts damage, while preserving old gate0.25-like hard-tail behavior.
- Candidate: risk-conditioned delta/strength cap or penalty using detached risk, plus a mild scale floor/anchor. Avoid full `u` anchoring.

Outputs:

- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly01_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly01_seed1234_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly10_seed1234_openimages_val4096_holdout4096_current.csv`
- `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_anchor_old_logsonly10_seed1234_step500_val4096_holdout4096_current.{csv,json}`
- `experiments/analysis/old_anchor_scale_only_seed1234_val4096_holdout4096_current.{md,json}`


## E052: Local Delta Control and Current-Code Consistency Audit

Status: Done; local cap is provisionally positive only under the current-code protocol, but paper-facing claims are blocked until the baseline/code mismatch is resolved.

Motivation:

- E050/E051 suggested that reliability control should be local and selective rather than a global scale or full-conditioning anchor.
- I added local Householder delta diagnostics and trained two seed1234 local-control probes on the min090 inverse/detached risk gate:
  - local cap080/rho1: local delta excess above 0.080, risk-weighted and detached.
  - local band063+cap080: global delta target 0.063 with rho 5.0 plus local cap080/rho1.

Implementation and device:

- Code now records local Householder delta mean/max/std in rvq stats and exposes the local delta map to the loss.
- Training/evaluation used only physical GPU 0 via CUDA_VISIBLE_DEVICES=0 and --device cuda:0.
- No NaN was observed.

Current-code checkpoint results on seed1234 OpenImages holdout4096:

| method | checkpoint | RD | delta vs current HCS | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| HCS current recheck | step500 | 2.889062 | +0.000000 | 0.310475 | 20.397598 | 0.752244 |
| old gate0.25 current recheck | step250 | 2.843577 | -0.045484 | 0.314049 | 20.423228 | 0.751089 |
| min090 current recheck | step500 | 2.891932 | +0.002871 | 0.309998 | 20.375755 | 0.749153 |
| local cap080/rho1 | step250 | 2.828428 | -0.060633 | 0.314640 | 20.438995 | 0.751071 |
| local band063+cap080 | step250 | 3.031029 | +0.141968 | 0.308669 | 20.029071 | 0.740271 |

Intermediate-feature readout under current code:

| method | s_q | strength | delta RMS | local mean | local max | latent qMSE | index bpp | dead code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.562929 | 0.258263 | 0.061265 | 0.045810 | 0.226732 | 0.075195 | 0.113024 | 0.197060 |
| local cap080/rho1 | 0.565083 | 0.233936 | 0.030558 | 0.024222 | 0.108607 | 0.077881 | 0.112893 | 0.197746 |
| local band063+cap080 | 0.584372 | 0.219255 | 0.062100 | 0.054575 | 0.170352 | 0.152749 | 0.109988 | 0.234999 |

Important consistency finding:

- The May29 trusted CSVs are not reproducible under the current code state.
- On the same checkpoint paths, HCS shifts from 2.211475 to 2.889062 RD, old gate0.25 from 2.193095 to 2.843577, and min090 from 2.228412 to 2.891932.
- Therefore, mixed comparisons between May29 trusted baselines and May30 local-control checkpoints are not paper-safe.

Interpretation:

- Within current-code rechecks only, local cap080/rho1 is the best seed1234 row and improves current HCS by -0.060633 RD and current old gate0.25 by -0.015149 RD.
- This is encouraging, but it is not yet a paper-facing claim because the baseline reproducibility mismatch is larger than the local-cap gain.
- The band probe is useful as a negative result: matching global delta RMS near old is not sufficient. It restores delta magnitude but lowers effective strength, doubles latent qMSE, and increases dead-code ratio.
- Mechanistically, the next controller should preserve a joint operating regime: scale, effective geometry strength, local-delta tail, latent qMSE, and codebook usage.

Next action:

- First pin or restore the evaluation/model code that produced the May29 trusted CSVs, or explicitly declare the current code as a new protocol and rerun all baselines from scratch.
- Then run local cap080/rho1 across seeds 1234/2345/3456 under one fixed protocol.
- Do not use mixed-protocol numbers in the manuscript.

Outputs:

- experiments/analysis/local_delta_controls_current_code_consistency_seed1234_holdout4096.md
- experiments/analysis/local_delta_controls_current_code_consistency_seed1234_holdout4096.json
- experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_seed1234_openimages_val4096_holdout4096_current.csv
- experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_seed1234_openimages_val4096_holdout4096_current.csv


## E053: Holdout Protocol Correction and 3-Seed Local-Cap Audit

Status: Done for OpenImages holdout4096; promising paper-main candidate, pending secondary-split checks.

This entry supersedes the fragile part of E052. The earlier statement that the May29 trusted holdout CSVs were not reproducible was too strong. A direct path-aligned probe shows that HCS, old gate0.25, and min090 trusted holdout rows are reproducible within numerical noise when the same paths, checkpoints, and exact Householder inverse convention are used. The quarantined artifacts are instead `*_current_recheck_after_localstats.csv` and old gate0.25 `*_current_localstats.csv`, which are inconsistent with direct probes and should not be used in paper-facing tables.

Key reproducibility artifacts:

- `experiments/analysis/holdout4096_artifact_consistency_audit.{md,json}`
- `tools/analyze_holdout_artifact_consistency.py`
- `tools/probe_householder_inverse_modes.py`

Householder inverse audit:

- old gate0.25 and min090 match historical rows only with the mathematically exact partial-reflection inverse.
- `same_partial`, full Householder, and identity inverse modes are much worse on the same images.
- This means the historical HCG artifacts are valid under the exact-inverse convention; the bad recheck/localstats files are protocol artifacts.

Local cap080/rho1 was then evaluated under the trusted protocol with direct current-code probes on physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No nonfinite rows were observed.

3-seed OpenImages holdout4096 results:

| method | mean RD | delta vs HCS | delta vs old gate0.25 | delta vs min090 |
|---|---:|---:|---:|---:|
| HCS | 2.263705 | +0.000000 | +0.030163 | +0.033598 |
| old gate0.25 | 2.233542 | -0.030163 | +0.000000 | +0.003435 |
| min090 | 2.230108 | -0.033598 | -0.003435 | +0.000000 |
| local cap080/rho1 | 2.221143 | -0.042562 | -0.012398 | -0.008965 |

Per-seed best local checkpoints:

| seed | best local checkpoint | local RD | local-HCS | local-old | local-min090 |
|---|---:|---:|---:|---:|---:|
| 1234 | step250 | 2.189942 | -0.021532 | -0.003153 | -0.038469 |
| 2345 | step250 | 2.228262 | -0.088034 | +0.006436 | -0.005978 |
| 3456 | step250 | 2.245225 | -0.018120 | -0.040478 | +0.017553 |

Checkpoint selection matters. For all three seeds, step250 is best. Step500 consistently moves to lower `s_q`, higher latent quantization MSE, and worse RD:

- seed1234 step500: RD 2.203286, `s_q=0.467472`, qMSE 0.111983
- seed2345 step500: RD 2.325470, `s_q=0.465532`, qMSE 0.109701
- seed3456 step500: RD 2.313249, `s_q=0.456958`, qMSE 0.119436

Intermediate-feature summary for best checkpoints:

| method | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | risk mult |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old gate0.25 | 0.528787 | 0.268809 | 0.073521 | n/a | 0.078307 | 66.810569 | 0.032061 | n/a |
| min090 | 0.492583 | 0.265869 | 0.085068 | n/a | 0.096634 | 68.021232 | 0.029606 | 0.936998 |
| local cap080/rho1 | 0.569045 | 0.232035 | 0.030007 | 0.022989 | 0.059622 | 65.234520 | 0.034445 | 0.918647 |

Difficulty-quartile analysis shows the main benefit is in the hard tail:

| quartile | old-HCS | min090-HCS | local-HCS | local-old | local-min090 |
|---|---:|---:|---:|---:|---:|
| Q1 easy | +0.023151 | +0.009024 | +0.050885 | +0.027733 | +0.041861 |
| Q2 | -0.021963 | -0.030319 | -0.013253 | +0.008711 | +0.017066 |
| Q3 | -0.053677 | -0.059434 | -0.071727 | -0.018050 | -0.012293 |
| Q4 hard | -0.068166 | -0.053660 | -0.136153 | -0.067987 | -0.082493 |

Interpretation:

- local cap080/rho1 is now the strongest unified-controller candidate under this trusted holdout protocol.
- It does not simply remove HCG geometry. It keeps scale/codebook behavior in a safe range while sharply reducing local Householder delta and improving the hard-image tail.
- The cost is easy-image Q1 degradation, so the next improvement target is selective control that preserves local cap's hard-tail benefit while reducing easy-image over-suppression.

Next actions:

1. Evaluate the 3-seed local-cap step250 checkpoints on Kodak and another validation split if available.
2. Add a selective/easy-safe variant, such as a weaker cap or risk-conditioned cap that is inactive on easy/low-risk images.
3. Convert the protocol audit into a manuscript-ready reproducibility paragraph: exact inverse convention, quarantined artifact list, checkpoint-selection rule, and GPU/device note.
4. Keep old gate0.25 and min090 as ablations, and report the old/min090 selector/oracle only as headroom diagnostics.


## E054: Kodak Secondary-Split Audit for Local Cap080/rho1

Status: Done for Kodak selected-checkpoint audit and local checkpoint sweep; useful caution for the paper-main claim.

The local cap080/rho1 candidate was evaluated on Kodak with the same checkpoint selections used in the trusted OpenImages holdout4096 protocol. All runs used exact-inverse direct probes on physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`). No nonfinite rows were observed.

Selected-checkpoint Kodak 3-seed results:

| method | mean RD | delta vs HCS | delta vs old gate0.25 | delta vs min090 | nonfinite |
|---|---:|---:|---:|---:|---:|
| HCS | 2.206217 | +0.000000 | +0.020033 | +0.011168 | 0 |
| old gate0.25 | 2.186184 | -0.020033 | +0.000000 | -0.008866 | 0 |
| min090 | 2.195049 | -0.011168 | +0.008866 | +0.000000 | 0 |
| local cap080/rho1 step250-selected | 2.214630 | +0.008413 | +0.028446 | +0.019581 | 0 |

Per-seed selected-checkpoint local deltas:

| seed | local-HCS | local-old | local-min090 |
|---|---:|---:|---:|
| 1234 | -0.024608 | +0.006351 | -0.025499 |
| 2345 | -0.031932 | +0.009314 | -0.012851 |
| 3456 | +0.081778 | +0.069673 | +0.097092 |

The split-level conclusion is therefore not the same as OpenImages. Local cap080/rho1 remains promising, but selected step250 does not cleanly transfer to Kodak. The failure is concentrated in seed3456, while seed1234/2345 still improve HCS.

A checkpoint sweep for local cap080/rho1 on Kodak shows the checkpoint dependency:

| seed | step250 RD | step500 RD | step500-step250 | step250 s_q | step500 s_q | step250 qMSE | step500 qMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.173300 | 2.129414 | -0.043886 | 0.583361 | 0.486637 | 0.041288 | 0.069392 |
| 2345 | 2.207830 | 2.179888 | -0.027942 | 0.581112 | 0.484823 | 0.041139 | 0.068535 |
| 3456 | 2.262769 | 2.284793 | +0.022024 | 0.577871 | 0.476784 | 0.041047 | 0.073244 |

The step500 Kodak mean is `2.198032`, improving the local step250 mean by `-0.016601` and beating HCS by about `-0.008185`, but it is still worse than old gate0.25 by about `+0.011848` and min090 by about `+0.002983`. Since OpenImages selected step250 and Kodak prefers step500 for two seeds, Kodak cannot be used for checkpoint selection in a paper-safe protocol; it should be reported as a secondary split showing remaining selection sensitivity.

Feature interpretation:

- Step250 local cap is very conservative on Kodak (`s_q=0.580781`, strength `0.231294`, local delta mean `0.017240`, qMSE `0.041158`), but this over-suppresses easy images and hurts seed3456.
- Step500 moves toward stronger geometry and lower scale (`s_q` about `0.48`, strength about `0.256`, local delta mean about `0.0226`) and improves seeds 1234/2345 on Kodak, but worsens seed3456 and raises qMSE.
- The OpenImages hard-tail benefit is real, but the controller is not yet universally easy-safe or split-stable.

Next actions:

1. Keep local cap080/rho1 as the leading research direction, but do not claim it is final paper-main until secondary-split stability is improved.
2. Implement an easy-safe/selective local controller that preserves the OpenImages Q4 hard-tail benefit while reducing Q1/Kodak easy-image degradation.
3. Keep the checkpoint-selection rule fixed on OpenImages validation; use Kodak only for reporting/diagnosis, not for picking step250 vs step500.
4. Add a compact paper table with OpenImages holdout, Kodak selected-checkpoint transfer, nonfinite count, and local checkpoint-sweep diagnostic.

Artifacts:

- `experiments/analysis/local_cap080_rho1_multiseed_kodak_trusted_protocol.{md,json,csv}`
- `experiments/analysis/local_cap080_rho1_kodak_checkpoint_sweep.{md,json,csv}`
- `tools/analyze_local_cap_kodak.py`
- `tools/analyze_local_cap_kodak_checkpoint_sweep.py`



## E055: Excess-Risk Local Cap080/rho1 Checkpoint Sweep

Status: Done for 3-seed training and exact-inverse direct evaluation on OpenImages holdout4096 and Kodak.

This probe keeps the local cap080/rho1 direction, but makes the cap depend on detached excess risk above the min090 risk floor. The intent is to avoid treating low-risk/easy regions as if they needed the same geometry suppression as genuinely fragile regions.

All training and evaluation runs used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed. The new reusable sweep artifact is `tools/analyze_excessrisk_local_cap_checkpoint_sweep.py`.

OpenImages holdout4096 exact-inverse results:

| method/checkpoint rule | mean RD | delta vs HCS | delta vs old gate0.25 | delta vs min090 | delta vs previous local best |
|---|---:|---:|---:|---:|---:|
| excess-risk local cap step250 fixed | 2.223519 | -0.040186 | -0.010023 | -0.006589 | +0.002376 |
| excess-risk local cap step500 fixed | 2.218792 | -0.044913 | -0.014750 | -0.011316 | -0.002351 |
| excess-risk local cap per-seed best | 2.216082 | -0.047623 | -0.017459 | -0.014026 | -0.005061 |

Per-seed OpenImages sweep:

| seed | step250 RD | step500 RD | best | step500-step250 |
|---|---:|---:|---:|---:|
| 1234 | 2.197417 | 2.176471 | 500 | -0.020945 |
| 2345 | 2.225868 | 2.233997 | 250 | +0.008128 |
| 3456 | 2.247272 | 2.245907 | 500 | -0.001364 |

Kodak exact-inverse results are unexpectedly strong for the same fixed step500 rule:

| method/checkpoint rule | mean RD | delta vs HCS | delta vs old gate0.25 | delta vs min090 | delta vs previous local step500 |
|---|---:|---:|---:|---:|---:|
| excess-risk local cap step250 fixed | 2.199571 | -0.006646 | +0.013387 | +0.004522 | +0.001539 |
| excess-risk local cap step500 fixed | 2.132528 | -0.073689 | -0.053656 | -0.062521 | -0.065504 |

Per-seed Kodak sweep:

| seed | step250 RD | step500 RD | best | step500-step250 |
|---|---:|---:|---:|---:|
| 1234 | 2.141605 | 2.132566 | 500 | -0.009039 |
| 2345 | 2.192414 | 2.132939 | 500 | -0.059475 |
| 3456 | 2.264695 | 2.132080 | 500 | -0.132615 |

Intermediate-feature summary:

| split/step | s_q | strength | delta RMS | local delta mean | qMSE | perplexity | dead code | risk mult |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenImages step250 | 0.568803 | 0.233568 | 0.033074 | 0.025209 | 0.059102 | 65.282681 | 0.034435 | 0.918694 |
| OpenImages step500 | 0.465916 | 0.261636 | 0.041632 | 0.029470 | 0.110920 | 67.098902 | 0.033853 | 0.943204 |
| Kodak step250 | 0.580560 | 0.232686 | 0.025492 | 0.018848 | 0.040780 | 66.168278 | 0.043077 | 0.916331 |
| Kodak step500 | 0.485199 | 0.256944 | 0.032255 | 0.023213 | 0.068965 | 64.471053 | 0.061740 | 0.937721 |

Interpretation:

- The paper-safer result is the fixed step500 rule: it improves the trusted OpenImages 3-seed mean over HCS, old gate0.25, min090, and previous local cap080/rho1, and it transfers very strongly to Kodak.
- The per-seed-best OpenImages number is useful as a ceiling, but should not be the headline unless a separate validation split is used for checkpoint selection.
- Unlike the previous local cap080/rho1 checkpoint sweep, step500 no longer collapses. It lowers `s_q` and raises qMSE, but this stronger-geometry regime now improves RD on average, especially on Kodak.
- The remaining OpenImages weakness is seed2345 step500, where step250 is still better. The next decision is whether to keep fixed step500 as the main protocol or add a validation-only checkpoint-selection rule.

Next actions:

1. Add difficulty-quartile and per-image selector analysis for excess-risk local cap step500 vs HCS/old/min090/previous local cap.
2. Evaluate fixed step500 on another secondary split if available, because Kodak is now very positive but only 24 images.
3. Prepare a manuscript-ready ablation table separating HCS, old gate0.25, min090, local cap080/rho1, and excess-risk local cap080/rho1 fixed step500.
4. Keep all GPU/device and nonfinite-row checks in the paper-facing reproducibility notes.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_kodak_checkpoint_sweep.{md,json,csv}`
- `tools/analyze_excessrisk_local_cap_checkpoint_sweep.py`



## E056: Excess-Risk Tail and Per-Image Analysis

Status: Done for OpenImages holdout4096 per-image tail analysis; no GPU execution was needed.

The fixed-step excess-risk result is still strong, but the mechanism is different from the previous local cap080/rho1 story. The new artifact joins HCS, old gate0.25, min090, previous local cap080/rho1 step250, and excess-risk step250/step500 by exact image path over all 3 seeds.

Overall OpenImages holdout4096 per-image summary:

| method | RD | vs HCS | vs old | vs min090 | vs previous local | win vs HCS | win vs previous local |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous local cap080/rho1 step250 | 2.221143 | -0.042562 | -0.012398 | -0.008965 | +0.000000 | 0.555094 | 0.000000 |
| excess-risk step250 | 2.223519 | -0.040186 | -0.010023 | -0.006589 | +0.002376 | 0.555257 | 0.452311 |
| excess-risk step500 fixed | 2.218792 | -0.044913 | -0.014750 | -0.011316 | -0.002351 | 0.658854 | 0.554199 |

HCS-difficulty quartiles for fixed step500:

| quartile | excess500-HCS | excess500-old | excess500-prev local | win vs HCS |
|---|---:|---:|---:|---:|
| Q1 easy | -0.054984 | -0.078135 | -0.105868 | 0.766602 |
| Q2 | -0.059420 | -0.037456 | -0.046167 | 0.701497 |
| Q3 | -0.056804 | -0.003127 | +0.014923 | 0.643555 |
| Q4 hard | -0.008446 | +0.059720 | +0.127707 | 0.523763 |

Interpretation:

- Fixed step500 is the cleanest average-RD and Kodak-transfer candidate, but it is not simply a stronger version of the previous hard-tail local cap.
- Previous local step250 remains the clearest evidence for hard-tail reliability control: Q4 was `-0.136153` vs HCS, while excess-risk fixed step500 is only `-0.008446` vs HCS in Q4.
- Excess-risk fixed step500 wins average RD because it repairs Q1/Q2 strongly and keeps enough Q3 benefit, not because it dominates the hard tail.
- Paper positioning should separate these two claims: local cap step250 demonstrates the hard-tail control mechanism, while excess-risk fixed step500 is the stronger fixed-checkpoint average and secondary-split candidate.

Next actions:

1. Use the tail split to design a hybrid or schedule that keeps excess-risk step500 easy/Q2 transfer while preserving local step250 Q4 hard-tail benefit.
2. Add a validation-only checkpoint-selection rule or separate held-out split before using per-seed best numbers.
3. Keep both rows in the manuscript draft table: local step250 for mechanism, excess-risk fixed step500 for paper-safe average transfer.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_tail_holdout4096.{md,json,csv}`
- `tools/analyze_excessrisk_tail_holdout.py`


## E057: Previous-Local vs Excess-Risk Selector Headroom

Status: Done for OpenImages holdout4096 per-image headroom analysis; no GPU execution was needed.

This diagnostic switches per image between the previous local cap080/rho1 step250 result and the excess-risk local cap080/rho1 fixed step500 result. It is intentionally a headroom analysis only, because it combines two separately trained checkpoints and therefore is not a deployable single-codec result.

Base OpenImages holdout4096 3-seed means:

| method | RD | vs HCS |
|---|---:|---:|
| HCS | 2.263705 | +0.000000 |
| old gate0.25 | 2.233542 | -0.030164 |
| min090 | 2.230108 | -0.033597 |
| previous local step250 | 2.221143 | -0.042562 |
| excess-risk fixed step500 | 2.218792 | -0.044913 |

Headroom result:

| policy | RD | vs HCS | vs previous local | vs excess500 |
|---|---:|---:|---:|---:|
| oracle min(previous local, excess500) | 2.147093 | -0.116613 | -0.074051 | -0.071699 |
| best single-feature threshold | 2.170055 | -0.093650 | -0.051088 | -0.048737 |

The best simple threshold uses `rvq_householder_delta_rms <= 0.045256` to select excess-risk fixed step500 on about `64.998%` of images and previous local step250 otherwise.

Interpretation:

- The oracle and threshold numbers are too strong to ignore, but they are not paper-main results because they switch across two independently trained checkpoints.
- The useful scientific signal is complementarity: excess-risk fixed step500 is very strong in easier/lower-delta regimes, while previous local step250 protects the hard tail.
- The next single-controller experiment should keep the step500 easy/Q2 behavior, but back off or cap the stronger-geometry state when local Householder delta RMS becomes high.
- A promising implementation path is a detached local-delta controller that modulates Householder amplitude or blends toward the previous-local cap when `rvq_householder_delta_rms` exceeds a validation-chosen threshold.

Artifacts:

- `experiments/analysis/excessrisk090_prevlocal_selector_headroom_holdout4096.{md,json}`


## E058: Cap080-to-Cap060 Schedule Probe

Status: Done for 3-seed OpenImages holdout4096, Kodak, and tail analysis. This is a useful negative/control result, not the new paper-main method.

This probe kept the excess-risk local cap setup but tightened `householder_delta_local_cap` from `0.080` to `0.060` after step250. The intent was to preserve the step250 hard-tail behavior while preventing step500 high-delta failure.

All training and exact-inverse evaluations used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No nonfinite rows were observed.

OpenImages holdout4096:

| checkpoint rule | mean RD | vs previous excess-risk cap080 | note |
|---|---:|---:|---|
| cap080to060 step250 fixed | 2.219987 | -0.003531 vs old excess step250 | Slightly better early checkpoint |
| cap080to060 step500 fixed | 2.242705 | +0.023913 vs old excess step500 | Not usable as fixed checkpoint |
| cap080to060 per-seed best | 2.213494 | -0.002589 vs old excess per-seed best | Diagnostic only |

Per-seed OpenImages sweep:

| seed | step250 RD | step500 RD | best | step500-step250 |
|---|---:|---:|---:|---:|
| 1234 | 2.196663 | 2.203795 | 250 | +0.007132 |
| 2345 | 2.215687 | 2.296190 | 250 | +0.080503 |
| 3456 | 2.247612 | 2.228131 | 500 | -0.019481 |

Kodak:

| checkpoint rule | mean RD | note |
|---|---:|---|
| cap080to060 step250 fixed | 2.197569 | Slightly better than old excess step250 |
| cap080to060 step500 fixed | 2.155794 | Worse than old excess step500 2.132528 |

Tail and feature diagnosis:

| HCS quartile | variant250-HCS | variant500-HCS | variant500-variant250 |
|---|---:|---:|---:|
| Q1 easy | +0.044808 | -0.021157 | -0.066147 |
| Q2 | -0.016586 | -0.030684 | -0.014098 |
| Q3 | -0.071506 | -0.032866 | +0.038640 |
| Q4 hard | -0.131770 | +0.000706 | +0.132476 |

Feature means show that step500 still moves into the low-scale, high-qMSE regime: `s_q=0.466299`, `strength=0.259659`, `delta RMS=0.039586`, and `qMSE=0.111822`.

Interpretation:

- The cap080to060 schedule rescued seed3456 step500 on OpenImages, but it severely damaged seed2345 and did not transfer to Kodak.
- The failure mode is the same qualitative split as before: step500 helps easy images but gives back the hard-tail benefit.
- Therefore the next controller should not be a simple lower cap after step250. It should control the step500 qMSE/codebook regime while keeping useful geometry active.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080to060_rho1_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080to060_rho1_kodak_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080to060_rho1_tail_holdout4096.{md,json,csv}`


## E059: Beta-Commit Guard After Step250

Status: Done for 3-seed training, OpenImages holdout4096, Kodak, and per-image tail analysis. This supersedes E055 as the current strongest fixed-checkpoint candidate.

This probe keeps excess-risk local cap080/rho1 unchanged, but raises `beta_commit` from `0.01` to `0.05` after step250. The hypothesis was that step500 degradation was not solved by shrinking Householder cap directly; instead, the model needed a guard against the latent qMSE/codebook drift that appears after step250.

All runs used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed in training or evaluation.

OpenImages holdout4096 exact-inverse results:

| seed | step250 RD | step500 RD | step500-step250 | best | step250 qMSE | step500 qMSE |
|---|---:|---:|---:|---:|---:|---:|
| 1234 | 2.195928 | 2.158153 | -0.037775 | 500 | 0.058964 | 0.106899 |
| 2345 | 2.217019 | 2.206494 | -0.010525 | 500 | 0.059117 | 0.101180 |
| 3456 | 2.247406 | 2.156416 | -0.090990 | 500 | 0.059256 | 0.112377 |

Aggregate OpenImages:

| method/checkpoint rule | mean RD | vs HCS | vs old gate0.25 | vs min090 | vs previous local |
|---|---:|---:|---:|---:|---:|
| beta-commit guard step250 | 2.220118 | -0.043588 | -0.013424 | -0.009990 | -0.001026 |
| beta-commit guard step500 fixed | 2.173688 | -0.090018 | -0.059854 | -0.056420 | -0.047456 |

Kodak exact-inverse results:

| seed | step250 RD | step500 RD | step500-step250 | best |
|---|---:|---:|---:|---:|
| 1234 | 2.141469 | 2.084875 | -0.056594 | 500 |
| 2345 | 2.182894 | 2.110460 | -0.072434 | 500 |
| 3456 | 2.265055 | 2.106310 | -0.158744 | 500 |

Aggregate Kodak step500 is `2.100549`, improving the previous excess-risk fixed step500 `2.132528` by `-0.031980` RD.

Tail analysis on OpenImages:

| HCS quartile | beta step500-HCS | beta step500-prev local | win vs HCS |
|---|---:|---:|---:|
| Q1 easy | -0.075655 | -0.126539 | 0.795573 |
| Q2 | -0.094492 | -0.081239 | 0.784180 |
| Q3 | -0.105463 | -0.033736 | 0.751628 |
| Q4 hard | -0.084461 | +0.051692 | 0.648112 |

Feature means for beta step500: `s_q=0.463063`, `strength=0.263283`, `delta RMS=0.040208`, `local delta mean=0.028513`, `qMSE=0.106819`, `perplexity=68.286368`, `dead code=0.031892`, and `risk mult=0.943947`.

Interpretation:

- This is the first fixed-step candidate that is strong on OpenImages average, improves Kodak transfer, and still improves the hard Q4 tail vs HCS.
- It does not fully preserve the previous local step250 Q4 gain (`-0.084461` vs HCS instead of `-0.136153`), so the hard-tail mechanism remains a separate ablation point.
- For the paper, beta-commit guard step500 should become the current main quantitative candidate, while previous local step250 remains the cleanest hard-tail reliability-control evidence.
- The next scientific check is whether the gain survives an additional held-out split or a validation-selected protocol, and whether `beta_commit=0.03/0.07` gives a smoother tradeoff.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_kodak_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.{md,json,csv}`


## E060: Beta-Commit Strength Boundary Check on Fragile Seed3456

Status: Done for seed3456 training and direct exact-inverse evaluation on OpenImages holdout4096 and Kodak. This is a boundary/tuning diagnostic, not a replacement for E059.

This probe checks whether the E059 `beta_commit=0.05` improvement is simply caused by "more commitment after step250" or whether the value is close to a useful operating point. Two adjacent strengths were trained from the same frozen initialization and schedule:

- `beta_commit=0.03` after step250
- `beta_commit=0.07` after step250

All runs used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed.

OpenImages holdout4096, seed3456:

| after-step250 beta_commit | step250 RD | step500 RD | step500-step250 | step500 qMSE | nonfinite rows |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 2.248322 | 2.206961 | -0.041361 | 0.111946 | 0 |
| 0.05 | 2.247406 | 2.156416 | -0.090990 | 0.112377 | 0 |
| 0.07 | 2.247064 | 2.197884 | -0.049180 | 0.111984 | 0 |

Kodak, seed3456:

| after-step250 beta_commit | step250 RD | step500 RD | step500-step250 | nonfinite rows |
|---:|---:|---:|---:|---:|
| 0.03 | 2.265961 | 2.112923 | -0.153039 | 0 |
| 0.05 | 2.265055 | 2.106310 | -0.158744 | 0 |
| 0.07 | 2.264628 | 2.093526 | -0.171102 | 0 |

Interpretation:

- On the main OpenImages holdout protocol, `beta_commit=0.05` is clearly best for the most fragile seed: it beats `0.03` by `-0.050545` RD and `0.07` by `-0.041468` RD at step500.
- Kodak shows the opposite small trend for this seed, with `0.07` beating `0.05` by `-0.012783` RD.
- Therefore the current paper-main choice should remain E059 `beta_commit=0.05`, because it is the only setting with completed 3-seed evidence and the strongest main-protocol fragile-seed result.
- `beta_commit=0.07` is still useful as a future transfer/tuning candidate, but should not be promoted without full 3-seed OpenImages and tail checks.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit003_after250_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit003_after250_kodak_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit007_after250_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit007_after250_kodak_checkpoint_sweep.{md,json,csv}`


## E061: Beta005 Guard Transfer-Split Audit

Status: Done for a 3-seed OpenImages transfer pilot with 512 images per seed. This is a protocol audit for paper safety, not yet the final full-split result.

The purpose was to test whether the E059 beta-commit guard only wins on the trusted holdout4096 slice used for checkpoint selection, or whether it transfers to another OpenImages region when the checkpoint rule is fixed in advance. The split was `/dpl/openimages/open-images-v6/train/data`, `start_index=8192`, `max_images_per_seed=512`. Checkpoint selection was fixed from the trusted holdout4096 protocol; no checkpoint was selected on this transfer split.

All evaluations used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed.

Transfer-split method means:

| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | strength | delta RMS | qMSE | dead code | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.223952 | +0.000000 | 0.313380 | 21.470124 | 0.783649 | 0.538824 | n/a | n/a | 0.078135 | 0.032399 | 0 |
| old gate0.25 | 2.198195 | -0.025756 | 0.314266 | 21.559556 | 0.785425 | 0.528764 | 0.268830 | 0.073651 | 0.081472 | 0.032832 | 0 |
| min090 | 2.195762 | -0.028189 | 0.312970 | 21.580884 | 0.783560 | 0.492590 | 0.265908 | 0.085345 | 0.101225 | 0.030828 | 0 |
| beta005 guard | 2.130398 | -0.093554 | 0.316844 | 21.812266 | 0.784528 | 0.463094 | 0.263316 | 0.040389 | 0.112163 | 0.034058 | 0 |

Per-seed RD:

| method | seed1234 | seed2345 | seed3456 |
|---|---:|---:|---:|
| HCS | 2.165975 | 2.282382 | 2.223499 |
| old gate0.25 | 2.154737 | 2.192721 | 2.247129 |
| min090 | 2.188528 | 2.206692 | 2.192067 |
| beta005 guard | 2.114123 | 2.162921 | 2.114151 |

Same-seed deltas show that beta005 wins every seed against HCS: `-0.051852`, `-0.119462`, and `-0.109347` RD for seeds 1234, 2345, and 3456. This is important because earlier candidates often rescued one seed while damaging another.

HCS-difficulty quartiles for beta005 guard:

| quartile | images | method-HCS | win vs HCS |
|---|---:|---:|---:|
| Q1 easy | 384 | -0.079099 | 0.783854 |
| Q2 | 384 | -0.098651 | 0.812500 |
| Q3 | 384 | -0.105755 | 0.747396 |
| Q4 hard | 384 | -0.090709 | 0.588542 |

Interpretation:

- The beta005 guard transfers beyond the checkpoint-selection slice in this n512/seed pilot. Its RD gain is larger than old gate0.25 and min090, and it wins all HCS-difficulty quartiles.
- The Q4 win rate remains lower than the easy/mid regions, so the previous local step250 row is still the cleaner hard-tail specialist. But beta005 is not just an easy-image trick: Q4 still improves by `-0.090709` RD vs HCS on this transfer split.
- The feature distribution is consistent with the E059 story: beta005 uses a low-scale, higher-latent-qMSE regime (`s_q=0.463094`, `qMSE=0.112163`) while keeping delta RMS low (`0.040389`). This supports the view that the beta guard stabilizes usable geometry rather than simply turning geometry off.

Next action:

- Run the same transfer audit at `max_images=4096` before promoting this to a paper-table claim.
- If the full split holds, use it as the main external-within-dataset robustness evidence alongside Kodak.

Artifacts:

- `experiments/analysis/beta005_transfer_openimages_start8192_n512.{md,json,csv}`


## E062: Beta005 Guard Full Transfer-Split Audit

Status: Done for a full 3-seed OpenImages transfer audit with 4096 images per seed. This upgrades E061 from a pilot check to a much stronger paper-safety result.

The protocol matches E061 but uses `max_images_per_seed=4096` on the unselected OpenImages slice starting at `start_index=8192`. Checkpoint selection remains fixed from the trusted holdout4096 protocol; no checkpoint is selected on this transfer split.

All evaluations used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed.

Transfer-split method means:

| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | strength | delta RMS | qMSE | dead code | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.222382 | +0.000000 | 0.312861 | 21.469770 | 0.780936 | 0.539389 | n/a | n/a | 0.075241 | 0.031943 | 0 |
| old gate0.25 | 2.197166 | -0.025216 | 0.313728 | 21.539798 | 0.782825 | 0.529357 | 0.268735 | 0.073179 | 0.078306 | 0.032262 | 0 |
| min090 | 2.193251 | -0.029132 | 0.312379 | 21.565045 | 0.781166 | 0.493277 | 0.265705 | 0.084711 | 0.096849 | 0.029777 | 0 |
| beta005 guard | 2.135355 | -0.087027 | 0.316280 | 21.810733 | 0.782684 | 0.463857 | 0.263139 | 0.040056 | 0.107123 | 0.031993 | 0 |

Beta005 deltas on this full transfer split are `-0.087027` vs HCS, `-0.061811` vs old gate0.25, and `-0.057895` vs min090. This is close to the n512 pilot delta (`-0.093554` vs HCS), so the pilot was not a small-sample accident.

Per-seed RD:

| method | seed1234 | seed2345 | seed3456 |
|---|---:|---:|---:|
| HCS | 2.170370 | 2.273434 | 2.223343 |
| old gate0.25 | 2.156276 | 2.187699 | 2.247524 |
| min090 | 2.188101 | 2.200954 | 2.190696 |
| beta005 guard | 2.120464 | 2.167476 | 2.118126 |

Same-seed deltas show that beta005 again wins every seed against HCS: seed1234 `-0.049906`, seed2345 `-0.105958`, and seed3456 `-0.105216` RD. Old gate0.25 still damages seed3456 (`+0.024182`), and min090 still damages seed1234 (`+0.017731`).

HCS-difficulty quartiles for beta005 guard:

| quartile | images | method-HCS | win vs HCS |
|---|---:|---:|---:|
| Q1 easy | 3072 | -0.079769 | 0.809896 |
| Q2 | 3072 | -0.100822 | 0.801432 |
| Q3 | 3072 | -0.103274 | 0.751628 |
| Q4 hard | 3072 | -0.064243 | 0.615234 |

Interpretation:

- This is now strong within-dataset transfer evidence: beta005 improves an unselected OpenImages slice, all three seeds, and all HCS-difficulty quartiles.
- The hard-tail gain is smaller than Q1-Q3 but remains clearly positive on 3072 Q4 rows. This supports beta005 as the paper-main average/transfer candidate, while previous local step250 remains the sharper hard-tail ablation.
- Feature means match the intended mechanism: beta005 keeps Householder delta RMS low (`0.040056`) while using a stronger low-scale quantized-latent regime (`s_q=0.463857`, `qMSE=0.107123`). Old/min090 have larger delta RMS (`0.073179`/`0.084711`) with weaker RD gains.

Paper implication:

- The main evidence stack is now much safer: trusted holdout4096, Kodak transfer, and full unselected OpenImages transfer all support beta005 guard.
- The next missing piece is not another same-split confirmation, but manuscript-oriented ablations and possibly a final external dataset or stronger-backbone plug-in check.

Artifacts:

- `experiments/analysis/beta005_transfer_openimages_start8192_n4096.{md,json,csv}`


## E063: CLIC External-Style Audit for Beta005 Guard

Status: Done for CLIC mobile valid and CLIC professional valid using the same fixed checkpoint rules as E062. This adds an external-style robustness check beyond OpenImages and Kodak.

The transfer audit tool was generalized to accept `--data-root`, then run on:

- `/dpl/clic/mobile/valid` with 61 images per seed
- `/dpl/clic/professional/valid` with 41 images per seed

Checkpoint selection remained fixed from the trusted OpenImages holdout4096 protocol. No checkpoint was selected on either CLIC split.

All evaluations used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed.

CLIC mobile valid:

| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | delta RMS | qMSE | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS | 1.821108 | +0.000000 | 0.316740 | 22.453990 | 0.794056 | 0.552111 | n/a | 0.048618 | 0 |
| old gate0.25 | 1.809649 | -0.011460 | 0.317661 | 22.528598 | 0.795737 | 0.542817 | 0.065083 | 0.050068 | 0 |
| min090 | 1.796757 | -0.024351 | 0.315988 | 22.575382 | 0.794711 | 0.509352 | 0.074184 | 0.059888 | 0 |
| beta005 guard | 1.736549 | -0.084559 | 0.318707 | 22.849406 | 0.795335 | 0.482086 | 0.032413 | 0.065153 | 0 |

CLIC professional valid:

| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | delta RMS | qMSE | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.002328 | +0.000000 | 0.315852 | 21.961874 | 0.785059 | 0.546310 | n/a | 0.058803 | 0 |
| old gate0.25 | 1.970373 | -0.031955 | 0.316745 | 22.017022 | 0.786559 | 0.536684 | 0.069964 | 0.060955 | 0 |
| min090 | 1.955854 | -0.046474 | 0.315248 | 22.079361 | 0.785357 | 0.502072 | 0.080232 | 0.074658 | 0 |
| beta005 guard | 1.872124 | -0.130204 | 0.319796 | 22.508226 | 0.787454 | 0.473880 | 0.034772 | 0.082303 | 0 |

Paper evidence summary:

| split | images/seed | beta-HCS | beta-old | beta-min090 |
|---|---:|---:|---:|---:|
| OpenImages trusted holdout4096 | 4096 | -0.090017 | -0.059854 | -0.056420 |
| OpenImages transfer start8192 | 4096 | -0.087027 | -0.061811 | -0.057895 |
| Kodak | 24 | -0.105668 | -0.085633 | -0.094501 |
| CLIC mobile valid | 61 | -0.084559 | -0.073099 | -0.060208 |
| CLIC professional valid | 41 | -0.130204 | -0.098249 | -0.083730 |

Interpretation:

- Beta005 guard improves over HCS, old gate0.25, and min090 on every audited split.
- CLIC mobile/professional both show all-seed and all-quartile improvements vs HCS, so the gain is not limited to OpenImages-like validation slices.
- The same feature pattern persists: beta005 uses a lower-scale, higher-qMSE quantized-latent regime, but keeps Householder delta RMS lower than old/min090. This supports the stability-control interpretation rather than a simple geometry shutoff story.
- CLIC is still small, so these rows should be framed as external-style robustness evidence, not as a replacement for full standard benchmark comparison.

Artifacts:

- `experiments/analysis/beta005_external_clic_mobile_valid.{md,json,csv}`
- `experiments/analysis/beta005_external_clic_professional_valid.{md,json,csv}`
- `experiments/analysis/beta005_paper_evidence_summary.{md,json}`

## E064: Beta005 Paper Claim Matrix

Status: Done. I converted the current beta005 evidence into a manuscript-facing claim matrix that separates what can be used as the paper-main prototype claim from mechanism-only, diagnostic, and negative-control evidence.

The generated matrix uses only already audited artifacts. It does not rerun GPU evaluation, and it preserves the fixed-checkpoint protocol: checkpoint choices are fixed by trusted OpenImages holdout4096, while OpenImages transfer, Kodak, and CLIC rows are reporting-only splits.

Main evidence table:

| split | images/seed | beta-HCS | beta-old | beta-min090 | beta s_q | beta delta RMS | beta qMSE | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenImages trusted holdout4096 | 4096 | -0.090018 | -0.059854 | -0.056420 | 0.463063 | 0.040208 | 0.106819 | 0 |
| OpenImages transfer start8192 | 4096 | -0.087027 | -0.061811 | -0.057895 | 0.463857 | 0.040056 | 0.107123 | 0 |
| Kodak | 24 | -0.105668 | -0.085633 | -0.094501 | 0.482487 | 0.030612 | 0.065989 | 0 |
| CLIC mobile valid | 61 | -0.084559 | -0.073099 | -0.060208 | 0.482086 | 0.032413 | 0.065153 | 0 |
| CLIC professional valid | 41 | -0.130204 | -0.098249 | -0.083730 | 0.473880 | 0.034772 | 0.082303 | 0 |

Claim tiers:

| tier | paper use | status |
|---|---|---|
| paper-main prototype codec | Fixed-checkpoint HCG-RVQ with reliability-aware local geometry control and beta-commit stabilization improves HCS, old gate0.25, and min090 across trusted, transfer, and external-style splits. | Ready for the current MeanScaleHyperprior/RVQ prototype claim. |
| mechanism ablation | Hyperprior-conditioned geometry is useful when reliability is controlled; beta005 keeps Householder displacement lower than old/min090 while preserving gains. | Ready as explanation/ablation. |
| hard-tail mechanism | Previous local cap080/rho1 step250 remains the cleaner hard-tail specialist, while beta005 is the stronger average/transfer codec. | Ready as a secondary mechanism result. |
| diagnostic/headroom | Old/min090 selector shows reliability-control headroom. | Use for motivation only, because it is multi-checkpoint. |
| negative controls | Posthoc min090, min095, direct delta regularization, and cap080-to-cap060 prevent over-claiming a trivial threshold or shrinkage rule. | Useful for appendix/ablation narrative. |

Interpretation:

- The main result is now a fixed-checkpoint prototype-codec claim, not a selector/oracle claim.
- The mechanism story is sharper: beta005 does not merely turn geometry off; it keeps geometry active while limiting unstable Householder displacement.
- The result is promising for an international-conference paper, but should still be framed as a controlled MeanScaleHyperprior/RVQ prototype result rather than SOTA dominance over modern LIC/GIC systems.
- The next best paper action is to convert this matrix into the main ablation table and an appendix guardrail table before spending GPU time on stronger-backbone plug-in checks.

Artifacts:

- `tools/build_beta005_claim_matrix.py`
- `experiments/analysis/beta005_paper_claim_matrix.{md,json}`

## E065: Manuscript Table Candidates for Beta005

Status: Done. I generated manuscript-oriented table candidates from the E064 claim matrix and the existing guardrail analyses.

Main ablation table candidate:

| row | role | RD | vs HCS | Q4 vs HCS | delta RMS | qMSE | paper use |
|---|---|---:|---:|---:|---:|---:|---|
| HCS-RVQ | shift/scale RVQ baseline | 2.263705 | +0.000000 | +0.000000 | n/a | n/a | baseline |
| HCG old gate0.25 | adds Householder geometry | 2.233542 | -0.030164 | -0.068166 | 0.073521 | 0.078307 | geometry ablation |
| HCG min090 risk | risk-suppressed geometry | 2.230108 | -0.033597 | -0.053660 | 0.085068 | 0.096634 | reliability ablation |
| local cap080/rho1 step250 | hard-tail specialist | 2.221143 | -0.042562 | -0.136153 | 0.030007 | 0.059622 | mechanism, not main codec |
| beta005 guard step500 | fixed-checkpoint stabilized HCG | 2.173688 | -0.090018 | -0.084461 | 0.040208 | 0.106819 | paper-main prototype row |

Appendix guardrail table candidate:

| row | scope | vs HCS | why rejected as paper-main |
|---|---|---:|---|
| posthoc min090 on old weights | 3-seed holdout4096 | +0.651546 | pure inference-time risk switching collapses RD and dead-code use |
| trained min095 risk floor | 3-seed holdout4096 | +0.708334 | too-high risk floor drives high qMSE and broad degradation |
| delta_reg010 | seed3456 holdout4096 | +0.815950 | directly shrinking delta RMS breaks codebook/latent usage |
| cap080-to-cap060 step500 | 3-seed holdout4096 | -0.021000 | smaller local cap loses the beta005 average and hard-tail balance |

Interpretation:

- The main paper table can now show a clean progression: HCS shift/scale -> Householder geometry -> risk control -> local hard-tail control -> beta-commit-stabilized fixed checkpoint.
- The guardrail table is important because it shows the gain is not explained by a trivial inference-time threshold, a stronger risk floor, or simply shrinking Householder displacement.
- The cap080-to-cap060 row is still better than HCS, but is rejected as paper-main because it is much weaker than beta005 and loses the intended balance.
- No GPU rerun was needed for this step; this was an analysis/table-freezing action over already verified artifacts.

Artifacts:

- `tools/build_beta005_paper_tables.py`
- `experiments/analysis/beta005_paper_tables.{md,json}`

## E066: Kodak Fixed-Protocol Audit for Beta005

Status: Done. I ran Kodak under the same fixed-checkpoint HCS/old gate0.25/min090/beta005 audit protocol used for OpenImages transfer and CLIC. This converts the earlier beta-only Kodak checkpoint-sweep evidence into an aligned paper-table row.

The evaluation used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN or nonfinite rows were observed.

Kodak fixed-protocol means:

| method | mean RD | vs HCS | bpp | PSNR | MS-SSIM | s_q | delta RMS | qMSE | dead code | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS | 2.206217 | +0.000000 | 0.318314 | 21.141664 | 0.749433 | 0.552640 | n/a | 0.049036 | 0.038411 | 0 |
| old gate0.25 | 2.186182 | -0.020035 | 0.318871 | 21.181249 | 0.750119 | 0.543347 | 0.055672 | 0.050499 | 0.039822 | 0 |
| min090 | 2.195049 | -0.011168 | 0.317437 | 21.157727 | 0.749893 | 0.509731 | 0.062015 | 0.060149 | 0.037869 | 0 |
| beta005 guard | 2.100549 | -0.105668 | 0.318701 | 21.445817 | 0.750414 | 0.482487 | 0.030612 | 0.065989 | 0.060330 | 0 |

Per-seed beta005 deltas vs same-seed HCS are seed1234 `-0.113032`, seed2345 `-0.129302`, and seed3456 `-0.074671`. Kodak HCS-difficulty quartiles are also all positive for beta005: Q1 `-0.075307`, Q2 `-0.135958`, Q3 `-0.148649`, and Q4 `-0.062759` RD vs HCS.

Interpretation:

- Kodak is now a clean standard-test-set row for the fixed beta005 claim, not just a beta-only transfer checkpoint sweep.
- The same mechanism signature appears again: beta005 keeps Householder delta RMS lower than old/min090 while operating in a stronger low-scale quantized-latent regime.
- The updated paper evidence stack is now: OpenImages trusted holdout4096, OpenImages transfer start8192, Kodak, CLIC mobile, and CLIC professional. Beta005 improves over HCS, old gate0.25, and min090 on every one of these aligned rows.
- SOTA plug-in remains important, but this result makes the prototype claim table stronger enough that the next GPU priority can be chosen deliberately rather than urgently.

Artifacts:

- `experiments/analysis/beta005_external_kodak_fixed_protocol.{md,json,csv}`
- `experiments/analysis/beta005_paper_evidence_summary.{md,json}`
- `experiments/analysis/beta005_paper_claim_matrix.{md,json}`
- `experiments/analysis/beta005_paper_tables.{md,json}`

## E067: Parallel Paper-Claim and Method-Improvement Plan

Status: Done. I re-read `docs/prompt.txt` and converted the current evidence into an explicit two-track research plan. This is a no-GPU analysis step over the already verified artifacts.

Prompt anchor:

> Hyperprior should not only predict entropy parameters; it should also generate the local geometry of the quantizer.

The current paper-claim track is now strong enough for the controlled prototype stage. Beta005 has five aligned fixed-checkpoint rows and improves HCS, old gate0.25, and min090 on all of them:

| split | beta-HCS | beta-old | beta-min090 | nonfinite |
|---|---:|---:|---:|---:|
| OpenImages trusted holdout4096 | -0.090018 | -0.059854 | -0.056420 | 0 |
| OpenImages transfer start8192 | -0.087027 | -0.061811 | -0.057895 | 0 |
| Kodak | -0.105668 | -0.085633 | -0.094501 | 0 |
| CLIC mobile valid | -0.084559 | -0.073099 | -0.060208 | 0 |
| CLIC professional valid | -0.130204 | -0.098249 | -0.083730 | 0 |

The method-improvement track is also active, but it should now be narrow rather than a broad sweep. The selector-headroom audit shows why: switching per image between previous-local step250 and excess-risk step500 reaches RD `2.147093` (`-0.116613` vs HCS), and a simple Householder delta-RMS threshold reaches RD `2.170055` (`-0.093650` vs HCS). This is not a publishable method because it switches between checkpoints, but it is strong evidence for the next single-checkpoint controller.

Current priority order:

| rank | action | track | reason |
|---:|---|---|---|
| 1 | Freeze beta005 paper tables | paper claim | five aligned fixed-checkpoint rows already support the controlled MeanScaleHyperprior/RVQ claim |
| 2 | Single-checkpoint reliability controller | method improvement | directly targets measured selector headroom without leaving the HCG geometry thesis |
| 3 | Strong-backbone/SOTA plug-in feasibility audit | SOTA bridge | needed eventually, but should start as interface/protocol work rather than a broad clone-and-train run |
| 4 | Validation-selected beta-commit boundary check | method improvement | `0.07` has a small seed3456 Kodak signal, but `0.05` remains safer on the fragile OpenImages seed |
| 5 | Stage context/gating and index-prior expansion | prompt extension | aligned with the full spec, but less urgent than stabilizing geometry |

Interpretation:

- The paper and method tracks are being run in parallel, but not with equal GPU allocation at every moment.
- The paper track keeps beta005 as the current fixed-checkpoint prototype row and records guardrails so the claim does not become an oracle/selector story.
- The method track should next turn the selector headroom into a deployable single-checkpoint reliability controller. It should be evaluated first on the fragile seed, then promoted to 3-seed only if it beats beta005 on both average RD and hard-tail behavior.
- Strong-backbone plug-in work is now justified by the stable prototype table, but it should begin with official-baseline/repository feasibility and a single selected backbone. It is not yet the next GPU-heavy action.

Artifacts:

- `tools/build_hcg_rvq_parallel_next_plan.py`
- `experiments/analysis/hcg_rvq_parallel_next_plan.{md,json}`

## E068: Beta005 rel075 Constrained Reliability Probe

Status: Done. I tested the narrow single-checkpoint reliability-controller probe proposed in E067. The config adds a mild detached reliability multiplier (`min=0.75`, init near identity) on top of the beta005 local-delta/risk/beta-commit guard, and evaluates the fragile seed3456 first before any 3-seed promotion.

The run used physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). Training, checkpoint sweep, and feature analysis completed with no NaN/nonfinite outputs.

Checkpoint sweep on OpenImages holdout4096:

| row | RD | delta vs beta005 seed3456 step500 | bpp | PSNR | MS-SSIM | nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| rel075 step250 | 2.243897 | +0.087481 | 0.314449 | 21.373629 | 0.781701 | 0 |
| rel075 step500 | 2.210781 | +0.054365 | 0.316386 | 21.606825 | 0.772467 | 0 |
| beta005 seed3456 step500 | 2.156416 | +0.000000 | n/a | n/a | n/a | 0 |

Feature comparison against beta005 seed3456 step500:

| feature | beta005 | rel075 | delta |
|---|---:|---:|---:|
| s_q_mean | 0.456233 | 0.456180 | -0.000053 |
| latent qMSE | 0.112377 | 0.112445 | +0.000068 |
| Householder delta RMS | 0.040308 | 0.039998 | -0.000310 |
| Householder strength | 0.264690 | 0.263329 | -0.001361 |
| risk multiplier | 0.945710 | 0.945724 | +0.000014 |
| dead code ratio | 0.030539 | 0.030619 | +0.000080 |
| perplexity | 68.779175 | 68.751643 | -0.027532 |

Interpretation:

- `rel075` is rejected for 3-seed promotion. It is worse than beta005 by `+0.054365` RD on the fragile seed3456 checkpoint protocol.
- The added reliability multiplier stayed almost inactive: mean `0.992290`, min `0.990848`, max `0.994993`, std `0.000657`. It did not learn the selector-like reliability behavior suggested by the diagnostic headroom.
- Intermediate features are nearly identical to beta005, so this is not a collapse or excessive-geometry failure. The RD loss comes mainly from worse image-domain distortion while preserving the same broad feature regime.
- The paper-main beta005 claim remains unchanged. The method-improvement track should not spend GPU on 3-seed `rel075`; the next controller should use an explicit measured reliability signal or supervised/teacher-style target derived from selector headroom.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rel075_after250_frozen_seed3456.yaml`
- `tools/analyze_rel075_probe.py`
- `experiments/analysis/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rel075_seed3456_holdout4096_checkpoint_sweep.csv`
- `experiments/analysis/feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rel075_seed3456_step500_val4096_holdout4096_current.{json,csv}`
- `experiments/analysis/betacommit005_rel075_seed3456_probe.{md,json}`

## E069: Decoder-Safe Selector Audit for the Next Controller

Status: Done. After rejecting `rel075`, I audited whether the beta005/previous-local complementarity can be approximated by a single feature available from the hyperprior-generated conditioning. This is still a diagnostic selector, not a paper-facing method, because it switches between two checkpoints. The point is to choose the next single-checkpoint controller target.

OpenImages holdout4096 base RD:

| method | RD | vs HCS |
|---|---:|---:|
| HCS | 2.263705 | +0.000000 |
| old gate0.25 | 2.233542 | -0.030164 |
| min090 | 2.230108 | -0.033597 |
| previous local step250 | 2.221143 | -0.042562 |
| beta005 step500 | 2.173688 | -0.090018 |

Headroom:

| policy | RD | vs HCS | vs previous local | vs beta005 |
|---|---:|---:|---:|---:|
| oracle min(previous local, beta005) | 2.121089 | -0.142617 | -0.100055 | -0.052599 |

Best decoder-safe threshold:

| feature | dir | threshold | RD | vs HCS | vs previous local | vs beta005 | beta fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| rvq_householder_gate_raw | le | 0.284059 | 2.158794 | -0.104912 | -0.062350 | -0.014894 | 0.869954 |

Best diagnostic threshold:

| feature | dir | threshold | RD | vs HCS | vs previous local | vs beta005 | beta fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| rvq_householder_delta_rms | le | 0.047937 | 2.145928 | -0.117777 | -0.075215 | -0.027760 | 0.760010 |

Interpretation:

- This is stronger than expected: a decoder-safe raw-gate mean threshold already beats beta005 by `-0.014894` RD in the diagnostic switch.
- The diagnostic delta-RMS threshold is stronger (`2.145928`) but is not directly deployable because it depends on the pre-quantization latent/quantization outcome. It remains useful as a teacher signal.
- The next single-checkpoint method should not be another free reliability multiplier. A better target is raw-gate-informed selective geometry backoff, trained or regularized so high raw-gate images move toward the previous-local behavior while low raw-gate images keep beta005.
- Paper-main evidence stays beta005; this selector audit is method-improvement guidance and should be reported as headroom/appendix unless converted into one checkpoint.

Artifacts:

- `tools/analyze_beta005_decoder_safe_selector.py`
- `experiments/analysis/beta005_previous_local_decoder_safe_selector.{md,json,csv}`

## E070: Raw-Gate Backoff Probe Rejects the Simple Continuous Multiplier

Status: Done. I implemented a decoder-safe raw-gate backoff controller, trained it on the fragile seed3456 protocol, and also evaluated the same backoff posthoc on the existing beta005 checkpoint. All GPU runs were pinned to physical GPU 0 (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaN/nonfinite rows were observed.

The result is a useful negative control rather than a promotion candidate:

| row | RD | delta vs beta005 seed3456 step500 | bpp | PSNR | MS-SSIM | nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| beta005 seed3456 step500 | 2.156416 | +0.000000 | n/a | n/a | n/a | 0 |
| trained rawbackoff step250 | 2.242941 | +0.086525 | 0.314427 | 21.376119 | 0.781800 | 0 |
| trained rawbackoff step500 | 2.228286 | +0.071870 | 0.316409 | 21.550365 | 0.784102 | 0 |
| posthoc rawbackoff on beta005 step500 | 2.180182 | +0.023766 | 0.316422 | 21.698707 | 0.779308 | 0 |

Feature analysis explains the failure. The posthoc multiplier is already broad (`0.854361`) and the trained multiplier becomes even stronger (`0.810320`). Householder strength drops from beta005 `0.264690` to `0.225906` posthoc and `0.219493` after training. The trained model also worsens latent qMSE from `0.112377` to `0.115936`. This means the controller suppresses useful beta005 geometry too broadly, and training introduces additional co-adaptation rather than recovering the discrete selector headroom.

Decision: reject `rawbackoff065_t0284` for 3-seed promotion. The raw-gate selector audit remains valuable as headroom, but a simple continuous gate multiplier is the wrong implementation. The next method-improvement controller should be teacher/supervised or distribution-aware: preserve beta005 geometry for low-risk images, and only push high-risk cases toward previous-local-like safer geometry.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rawbackoff065_t0284_after250_frozen_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_posthoc_rawbackoff065_t0284_eval_seed3456.yaml`
- `tools/analyze_rawbackoff_probe.py`
- `experiments/analysis/betacommit005_rawbackoff065_t0284_seed3456_probe.{md,json}`
- `experiments/analysis/feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rawbackoff065_t0284_seed3456_step500_val4096_holdout4096_current.{json,csv}`
- `experiments/analysis/feature_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_posthoc_rawbackoff065_t0284_seed3456_step500_val4096_holdout4096_current.{json,csv}`

## E071: Raw-Gate Tail Regularizer Also Stays Below Beta005

Status: Done. I tested the narrower alternative suggested by E069/E070: instead of multiplying the whole Householder geometry by a raw-gate backoff factor, train a tail-only raw-gate regularizer that penalizes only image-mean raw gate above the decoder-safe selector threshold `0.284059`. The implementation exposes `householder_gate_raw` from the model conditioning and adds `rho_householder_gate_raw_tail` to the rate-distortion loss. The run used seed3456, beta005 stabilization, physical GPU 0 only, and holdout4096 checkpoint evaluation at steps 250 and 500.

The run is numerically clean but not a promotion candidate:

| method | RD | delta vs beta005 seed3456 |
|---|---:|---:|
| beta005 step500 | 2.156416 | +0.000000 |
| rawbackoff065 step500 | 2.228286 | +0.071870 |
| rawtail t0284 rho100 step250 | 2.247834 | +0.091418 |
| rawtail t0284 rho100 step500 | 2.193294 | +0.036878 |

Per-image comparison confirms the decision. Rawtail step500 beats beta005 on only `32.8369%` of the 4096 images, with mean `+0.036878` RD and median `+0.027002` RD. It improves from its own step250 checkpoint (`-0.054540` RD), so the training is not broken, but the final row still remains clearly behind beta005.

The feature distribution is the useful part. Step250 suppresses raw gate strongly (`raw_gate_mean=0.252301`, tail fraction `0.000000`) and performs poorly. Step500 moves back toward beta005-like geometry (`strength=0.264145` vs beta005 `0.264690`, delta RMS `0.040086` vs `0.040308`, risk multiplier `0.945881` vs `0.945710`) while keeping worse RD. Raw-gate quartiles show the failure is concentrated exactly where the regularizer was supposed to help: in beta005 high raw-gate Q4, rawtail is `+0.101462` RD worse and wins only `15.4297%` of images.

Decision: reject `rawtail_t0284_rho100_betacommit005` for 3-seed promotion. This is a stronger negative control than rawbackoff: even a tail-only penalty on the endogenous raw-gate signal can be co-adapted around. The next method-improvement target should use a detached/train-split teacher target from diagnostic delta RMS or the per-image beta005-vs-previous-local labels, not raw-gate penalties alone.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rawtail_t0284_rho100_after250_frozen_seed3456.yaml`
- `tools/analyze_beta005_teacher_targets.py`
- `tools/analyze_rawtail_probe.py`
- `experiments/analysis/beta005_teacher_target_audit.{md,json,csv}`
- `experiments/analysis/rawtail_t0284_rho100_betacommit005_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/betacommit005_rawtail_t0284_rho100_seed3456_probe.{md,json}`

## E072: Leave-One-Seed-Out Teacher Target Audit

Status: Done. Before launching another single-checkpoint controller, I added a split-discipline audit for the beta005/previous-local teacher target. The policy is selected on two seeds and evaluated on the held-out seed, so this no longer relies only on same-row validation-set threshold fitting.

The held-out result supports the next direction:

| feature group | mean held-out delta vs beta005 | mean held-out F1 |
|---|---:|---:|
| decoder-safe conditioning features | -0.006046 | 0.415974 |
| diagnostic delta-style features | -0.024650 | 0.574891 |

The decoder-safe features retain a small average gain, but they are seed-dependent: seed1234 and seed2345 improve, while held-out seed3456 worsens by `+0.017255` RD. Diagnostic delta-RMS is more consistent, improving all held-out seeds (`-0.024138`, `-0.039529`, `-0.010283` vs beta005). This supports using diagnostic delta-style labels as teacher targets, while keeping the deployable model single-checkpoint and decoder-safe at inference.

Decision: do not continue raw-gate-only penalties. The next training experiment should generate teacher labels or detached targets under split discipline, then train a reliability controller to approximate that teacher inside one checkpoint. This keeps the proposal aligned with hyperprior-generated quantizer geometry while avoiding validation-oracle leakage.

Artifacts:

- `experiments/analysis/beta005_teacher_target_loso.{md,json}`

## E073: Delta-RMS Tail Regularizer Rejects Direct Diagnostic Penalization

Status: Done. I implemented the next narrow controller suggested by the LOSO audit: a direct image-level Householder delta-RMS tail penalty at the diagnostic threshold `0.047937`, with `rho_householder_delta_image_tail=100.0`. The run used the same seed3456 fragile protocol, beta005 stabilization, and physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). The checkpoint sweep on OpenImages holdout4096 completed for steps 250 and 500 with zero nonfinite rows.

The result is a stable but clear rejection:

| method | RD | delta vs beta005 seed3456 |
|---|---:|---:|
| beta005 step500 | 2.156416 | +0.000000 |
| rawtail t0284 rho100 step500 | 2.193294 | +0.036878 |
| deltatail t0479 rho100 step250 | 2.241103 | +0.084687 |
| deltatail t0479 rho100 step500 | 2.225577 | +0.069162 |

Per-image comparison confirms that this is not a checkpoint-selection fluke. Deltatail step500 beats beta005 on only `17.3828%` of the 4096 images, with mean `+0.069162` RD and median `+0.047333` RD. It does improve over its own step250 checkpoint (`-0.015526` RD, win fraction `60.0586%`), so training is moving, but it moves to a worse operating point than beta005.

The feature distribution explains why direct diagnostic penalization is not the right single-checkpoint controller. Deltatail step500 reduces delta RMS from beta005 `0.040308` to `0.038047` and reduces the delta tail fraction from `0.242432` to `0.184082`, but latent qMSE rises from `0.112377` to `0.126267` and RD worsens. The worst region is exactly the region the diagnostic teacher wanted to help: in beta005 high delta-RMS Q4, deltatail is `+0.136271` RD worse and wins only `9.9609%` of images.

Decision: reject `deltatail_t0479_rho100_betacommit005` for 3-seed promotion. Diagnostic delta RMS is useful as a teacher/selector signal under split discipline, but a direct end-to-end penalty on the measured delta tail is too blunt and allows co-adaptation with the quantized latent. The next method-improvement experiment should train a deployable reliability controller from explicit split-generated labels, while beta005 remains the paper-main fixed-checkpoint prototype.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_deltatail_t0479_rho100_after250_frozen_seed3456.yaml`
- `experiments/analysis/deltatail_t0479_rho100_betacommit005_holdout4096_checkpoint_sweep.{md,json,csv}`
- `experiments/analysis/betacommit005_deltatail_t0479_rho100_seed3456_probe.{md,json}`

## E074: Teacher-Label Reliability Controller Infrastructure

Status: Implemented and smoke-tested. After E070-E073 rejected direct raw-gate/delta-RMS suppression, I implemented the next safer branch: train a single-checkpoint reliability controller from explicit per-image teacher labels. This is infrastructure and a two-step smoke test, not a paper result yet.

The exported teacher labels compare beta005 step500 against previous-local step250 per image. `householder_reliability_keep=1` means beta005 has lower RD and the model should keep beta005-like Householder geometry; `0` means previous-local wins and the reliability controller should learn to suppress or fall back from that geometry. The aggregate label audit is:

| split | rows | previous-local win frac | diagnostic delta fallback frac | oracle RD | beta005 RD | previous-local RD |
|---|---:|---:|---:|---:|---:|---:|
| aggregate | 12288 | 0.347087 | 0.240072 | 2.121089 | 2.173688 | 2.221143 |
| seed1234 | 4096 | 0.364014 | 0.233398 | 2.109521 | 2.158153 | 2.189942 |
| seed2345 | 4096 | 0.424561 | 0.244385 | 2.139281 | 2.206494 | 2.228262 |
| seed3456 | 4096 | 0.252686 | 0.242432 | 2.114464 | 2.156416 | 2.245225 |

Implementation details:

- `ImageFolderDataset` can now return image paths and can start at a split offset. This lets training align per-image teacher labels with the same OpenImages slice used for audits.
- `train.py` can load CSV/JSON teacher labels and attach `output["teacher_targets"]` for each mini-batch.
- `RateDistortionLoss` now supports a binary teacher loss on `householder_reliability_multiplier`, with the reliability multiplier converted into a keep probability using the configured minimum reliability floor.
- `HyperpriorRVQ` now exposes `householder_reliability_multiplier` in `conditioning_tensors` so the teacher loss can supervise it.

Validation:

- A synthetic gradient check confirms the teacher loss pushes reliability up for keep-label images and down for fallback-label images.
- `configs/smoke_teacher_reliability.yaml` completes a two-step CUDA smoke run on physical GPU 0 only (`--device cuda:0`), loads 4096 labels, writes checkpoints, and reports no NaN/nonfinite issue.

Decision: this branch is now ready for a real split-disciplined pilot, but the smoke itself must not be used as paper evidence because its labels are exported from the audited holdout artifacts. The next serious experiment should generate teacher labels from a training/teacher split, train one deployable checkpoint, and then evaluate it on the fixed trusted holdout/Kodak/CLIC protocol. Beta005 remains the paper-main fixed-checkpoint row until that single-checkpoint teacher controller beats it under the same protocol.

Artifacts:

- `tools/export_beta005_teacher_labels.py`
- `experiments/analysis/beta005_previous_local_teacher_labels.{csv,json,md}`
- `configs/smoke_teacher_reliability.yaml`


## E075: Transfer-Split Teacher Labels and From-Scalar Controller Rejection

Status: Done. I moved the teacher-label controller from holdout-smoke to a split-disciplined setup. The labels are now exported on OpenImages start_index=8192, separate from the fixed holdout4096 evaluation split. They compare the paper-main beta005 checkpoint against previous-local cap080/rho1 per image.

The transfer split confirms that the selector target has real headroom:

| split | rows | previous-local win frac | oracle RD | beta005 RD | previous-local RD |
|---|---:|---:|---:|---:|---:|
| aggregate | 12288 | 0.337484 | 2.079763 | 2.135355 | 2.186255 |
| seed1234 | 4096 | 0.359131 | 2.066997 | 2.120464 | 2.151898 |
| seed2345 | 4096 | 0.407471 | 2.097713 | 2.167476 | 2.193800 |
| seed3456 | 4096 | 0.245850 | 2.074580 | 2.118126 | 2.213065 |

I then trained the first real transfer-label controller from the scalar initialization, using the same seed3456 fragile protocol and physical GPU 0 only. It is numerically clean but not a promotion candidate:

| method | holdout4096 RD | delta vs beta005 seed3456 | nonfinite |
|---|---:|---:|---:|
| beta005 step500 | 2.156416 | 0.000000 | 0 |
| previous-local step250 | 2.245225 | +0.088809 | 0 |
| teacher transfer rel075 from-scalar step250 | 2.366959 | +0.210543 | 0 |
| teacher transfer rel075 from-scalar step500 | 2.244678 | +0.088262 | 0 |

Feature analysis shows why this is rejected. Step500 has s_q_mean 0.467465, latent qMSE 0.102987, reliability mean 0.984273, delta RMS 0.039341, and strength 0.258911. It lands near the previous-local operating point and beats beta005 on only 14.5020% of images. The teacher labels are useful, but training the whole HCG branch from the scalar initialization does not preserve the beta005 fixed-checkpoint behavior.

Decision: do not 3-seed the from-scalar transfer controller. The right substrate for this branch is beta005 initialization with the existing codec path frozen, so the reliability controller can add selectivity without relearning the entire HCG geometry.

Artifacts:

- tools/export_split_teacher_labels.py
- experiments/analysis/beta005_previous_local_teacher_labels_transfer8192.{csv,json,md}
- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_after250_frozen_seed3456.yaml
- experiments/analysis/teacher_transfer8192_rel075_betacommit005_holdout4096_checkpoint_sweep.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_betacommit005_seed3456_probe.{json,md}

## E076: Beta005-Initialized Head-Only Reliability Probe

Status: Done. To isolate whether E075 failed because the controller design is bad or because the scalar initialization cannot reproduce beta005, I initialized from beta005 seed3456 step500, froze every existing codec component, and trained only householder_reliability_head.weight/bias on transfer8192 labels. The run used physical GPU 0 only and produced no NaN/nonfinite outputs.

This is the cleanest result of the controller branch so far:

| method | holdout4096 RD | delta vs HCS | delta vs beta005 | win vs beta005 | nonfinite |
|---|---:|---:|---:|---:|---:|
| beta005 step500 | 2.156416 | -0.106929 | 0.000000 | n/a | 0 |
| head-only from beta005 step250 | 2.156985 | -0.106361 | +0.000569 | 0.340576 | 0 |
| head-only from beta005 step500 | 2.157168 | -0.106178 | +0.000752 | 0.338379 | 0 |

The feature distribution says this is a preservation success but not yet an improvement. s_q_mean stays exactly at beta005 scale 0.456233, qMSE stays 0.11238, and reliability remains close to identity: 0.991905 at step250 and 0.987704 at step500. HCS-difficulty Q4 is the only region with a tiny beta005-relative improvement: -0.000194 at step250 and -0.000497 at step500, but the average does not improve.

Decision: beta005 initialization plus head-only training is the correct safety mechanism, because it avoids the E075 collapse and preserves the paper-main checkpoint almost exactly. However, rho_householder_reliability_teacher=0.05 is too weak or too identity-biased to recover the selector/oracle headroom. The next controller experiment should keep the beta005-initialized head-only protocol, but use a stronger or better-shaped teacher objective: higher teacher weight, soft margin targets, or pairwise/ranking supervision that directly rewards choosing beta005 vs previous-local behavior only where the teacher split supports it.

Artifacts:

- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_headonly_from_beta005_seed3456.yaml
- tools/analyze_headonly_teacher_probe.py
- experiments/analysis/teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_headonly_from_beta005_seed3456_probe.{json,md}


## E077: Stronger Head-Only Teacher Weight Does Not Recover Selector Headroom

Status: Done. I kept the split-disciplined teacher-controller protocol from E076, initialized from beta005 seed3456 step500, froze every existing codec tensor, and trained only the 965-parameter householder reliability head. The only change was increasing rho_householder_reliability_teacher from 0.05 to 0.50. Training and all holdout4096 evaluations were pinned to physical GPU 0 only with CUDA_VISIBLE_DEVICES=0 and --device cuda:0. No NaN/nonfinite rows were observed.

The checkpoint result is stable but not a promotion candidate:

| method | holdout4096 RD | delta vs HCS | delta vs beta005 | win vs beta005 | nonfinite |
|---|---:|---:|---:|---:|---:|
| beta005 step500 | 2.156416 | -0.106929 | 0.000000 | n/a | 0 |
| rho0.05 head-only step250 | 2.156985 | -0.106361 | +0.000569 | 0.340576 | 0 |
| rho0.05 head-only step500 | 2.157168 | -0.106178 | +0.000752 | 0.338379 | 0 |
| rho0.50 head-only step250 | 2.157003 | -0.106342 | +0.000587 | 0.342041 | 0 |
| rho0.50 head-only step500 | 2.157284 | -0.106061 | +0.000868 | 0.336182 | 0 |

The feature distribution shows that a stronger teacher weight moves the head slightly more but still keeps it close to identity. At step500, reliability mean changes from 0.987704 at rho0.05 to 0.985157 at rho0.50, delta RMS changes from 0.039666 to 0.039456, and strength changes from 0.261384 to 0.260677. The fallback-signal audit is directionally correct but too weak: on holdout images where previous-local beats beta005, rho0.50 step500 lowers reliability from 0.985740 on keep-like images to 0.983433 on fallback-like images, with AUC 0.717021 for low reliability identifying fallback-needed images. That separation is real, but the absolute gap is only -0.002307 and the RD still worsens by +0.000868 against beta005.

Decision: do not promote rho0.50 head-only teacher control to 3-seed or external datasets. E076 and E077 together show that beta005-initialized head-only training is safe, but binary teacher weight alone is not enough to capture the selector/oracle headroom. The next method-improvement branch should keep the beta005-initialized frozen-codec safety pattern, but change the supervision shape: soft margin, ranking, or excess-risk-weighted reliability targets that make the head distinguish harmful geometry more strongly without broad Householder shrinkage. Beta005 remains the paper-main fixed-checkpoint row.

Artifacts:

- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456.yaml
- experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_probe.{json,md}
- experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.{json,md}


## E078: Margin-Balanced Teacher Weighting Is Rejected

Status: Done. After E077 showed that plain teacher-weight scaling was too weak, I implemented weighted reliability-teacher support and tested a margin-balanced transfer-label variant. The new CSV balances keep/fallback classes per seed and upweights examples with large beta005-vs-previous-local RD margins. The run still used the safe beta005-initialized head-only shell: only householder_reliability_head.weight/bias were trainable, 965 parameters total, and all GPU work was pinned to physical GPU 0 with CUDA_VISIBLE_DEVICES=0 and --device cuda:0. No NaN/nonfinite rows appeared.

The result is a clean rejection:

| method | holdout4096 RD | delta vs beta005 | win vs beta005 | reliability mean | qMSE | dead-code ratio | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| beta005 step500 | 2.156416 | 0.000000 | n/a | n/a | 0.112377 | 0.030539 | 0 |
| rho0.50 unweighted step500 | 2.157284 | +0.000868 | 0.336182 | 0.985157 | 0.112389 | 0.030436 | 0 |
| margin-weighted rho0.50 step250 | 2.850224 | +0.693808 | 0.242188 | 0.990860 | 0.153393 | 0.183611 | 0 |
| margin-weighted rho0.50 step500 | 2.850449 | +0.694033 | 0.241699 | 0.982952 | 0.153409 | 0.183558 | 0 |

A checkpoint drift audit confirmed that only the reliability head changed relative to beta005: 2 changed tensors and 0 non-reliability tensors. Therefore this is not a freeze leak. The likely failure mode is discrete VQ fragility: the margin-weighted objective changes the spatial/group reliability pattern enough to push assignments into a much worse codebook-usage regime, even though the image-mean reliability scalar still looks near identity. That shows why average reliability alone is an insufficient safety metric.

Decision: reject margin-balanced binary teacher weighting and do not promote it. The positive takeaway is narrow but important: the implementation can now carry per-image teacher weights, and the audit proves that aggressive weighted BCE can damage codebook usage without any nonfinite failure. The next method-improvement experiment should add an explicit preservation constraint or distillation term for beta005 assignments/code usage if it uses weighted teacher targets. Safer alternatives are low-amplitude residual reliability around beta005, KL/entropy preservation on assignment usage, or a two-head design where the reliability head is trained with a codebook-usage guardrail.

Artifacts:

- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456.yaml
- experiments/analysis/beta005_previous_local_teacher_labels_transfer8192_margin_weighted.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_probe.{json,md}
- experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.{json,md}


## E079: Low-Amplitude Margin-Weighted Reliability Still Breaks RVQ Usage

Status: Done. I tested whether E078 was simply caused by too much reliability-head freedom. The new run keeps the same beta005-initialized frozen-codec shell and the same margin-balanced transfer labels, but raises `householder_gate_reliability_min` and the teacher probability floor from `0.75` to `0.95`, while lowering the reliability-head LR with `param_lr_multipliers.householder_reliability_head=0.25`. Only the 965-parameter reliability head was trainable. Training and both holdout4096 evaluations were pinned to physical GPU 0 with `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`; no NaN/nonfinite rows appeared.

The result is another clean rejection:

| method | holdout4096 RD | delta vs beta005 | win vs beta005 | reliability mean | qMSE | dead-code ratio | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| beta005 step500 | 2.156416 | 0.000000 | n/a | n/a | 0.112377 | 0.030539 | 0 |
| margin-weighted rho0.50 step500 | 2.850449 | +0.694033 | 0.241699 | 0.982952 | 0.153409 | 0.183558 | 0 |
| rel095 margin-weighted low-LR step250 | 2.849892 | +0.693476 | 0.242676 | 0.998840 | 0.153383 | 0.183668 | 0 |
| rel095 margin-weighted low-LR step500 | 2.849903 | +0.693487 | 0.242676 | 0.998645 | 0.153383 | 0.183668 | 0 |

The feature distribution is decisive. Even with reliability almost pinned to identity (`0.998645` at step500), latent qMSE and dead-code ratio remain in the same bad regime as E078. The fallback-signal audit also shows that the rel095 variant barely separates fallback-needed images (`gap=-0.000054`, AUC `0.669372`) while still causing roughly `+0.6935` RD against beta005. A direct checkpoint drift audit found only 2 changed tensors, both under `householder_reliability_head`, with 0 non-reliability tensors changed.

Decision: reject `rel095_marginw_rho050_lrm025` and do not promote any margin-weighted BCE variant without an explicit preservation guardrail. E078 was not mainly an overlarge reliability floor or learning-rate problem. The next method-improvement experiment should add a direct beta005-assignment/codebook-usage preservation term, or switch to a safer ranking/distillation objective that penalizes qMSE/dead-code movement while learning selectivity. Beta005 remains the paper-main fixed-checkpoint method.

Artifacts:

- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456.yaml
- tools/analyze_headonly_teacher_variant_probe.py
- tools/analyze_headonly_reliability_signal_audit.py
- experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step250_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step500_val4096_holdout4096_current.{csv,json,md}
- experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_probe.{json,md}
- experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_checkpoint_drift.{json,md}
- experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.{json,md}

## E080: Y-Hat Anchor Probe and Current-Code Re-Anchor Correction

Status: Done. I added a differentiable `y_hat` anchor preservation loss so a reliability-controller probe can be checked against the beta005 quantized-latent path directly. The new loss term is optional (`rho_anchor_y_hat`) and is only active when an anchor model is configured. The probe keeps the same beta005-initialized head-only safety shell as E076-E079: only `householder_reliability_head.weight` and `householder_reliability_head.bias` are trainable. All training and evaluation were pinned to physical GPU 0 with `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`; no NaN/nonfinite rows were observed.

The important result is a correction to the E078/E079 interpretation. The previous `+0.694 RD vs beta005` reading used a historical `variant500_rd` reference CSV. A direct current-code beta005 evaluation under the same config/protocol as E078-E080 gives RD `2.850149`, and the same checkpoint evaluated with its original beta005 config gives RD `2.849824`, while the historical beta005 reference is RD `2.156416`. The original-config difference is `+0.693408`, almost exactly the apparent E078-E080 failure. Therefore E078-E080 are not evidence that the margin-weighted head-only controller catastrophically broke RVQ assignments. They are mainly evidence that historical and current-code evaluation artifacts were mixed.

Current-code re-anchor table:

| variant | RD | vs current beta005 | win vs current beta005 | vs historical beta005 |
|---|---:|---:|---:|---:|
| current beta005 direct | 2.850149 | +0.000000 | n/a | +0.693733 |
| current beta005 original config | 2.849824 | -0.000324 | 0.817383 | +0.693408 |
| E078 marginw step250 | 2.850224 | +0.000075 | 0.361572 | +0.693808 |
| E078 marginw step500 | 2.850449 | +0.000301 | 0.348633 | +0.694033 |
| E079 rel095 low-LR step250 | 2.849892 | -0.000257 | 0.639160 | +0.693476 |
| E079 rel095 low-LR step500 | 2.849903 | -0.000246 | 0.641602 | +0.693487 |
| E080 yhat-anchor step250 | 2.850223 | +0.000075 | 0.361816 | +0.693807 |
| E080 yhat-anchor step500 | 2.850430 | +0.000281 | 0.349121 | +0.694014 |

The original-config direct probe rules out the simpler explanation that the `+0.694` came only from reading beta005 with the E080 reliability-controller config. The anchor-drift audit supports the correction. E080 step250 preserves the anchor RVQ indices with mean `index_match=0.999681` and mean `y_hat_mse=2.201963e-06`; step500 still preserves them with mean `index_match=0.999085` and mean `y_hat_mse=7.281669e-06`. This is incompatible with the earlier claim that the large `+0.694` delta was caused by RVQ assignment collapse. The large delta is instead a historical/current-code reference mismatch.

Decision: revise the E078/E079 conclusion. Margin-weighted head-only control is not promoted, because it does not yet beat a correctly re-anchored beta005 by a meaningful margin. But it should no longer be described as a proven catastrophic RVQ-usage failure. E079 is slightly better than the current beta005 direct row (`-0.00025 RD`), but the effect is tiny and blocked by the protocol mismatch. The next serious action is to pin or restore one code/evaluation state, regenerate beta005/HCS/old/min090 and the controller rows under that state, then only compare variants inside that frozen protocol.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `tools/analyze_anchor_preservation_drift.py`
- `experiments/analysis/beta005_after250_seed3456_current_config_direct_step500_val4096_holdout4096_current.{csv,json,md}`
- `experiments/analysis/beta005_after250_seed3456_original_config_direct_step500_val4096_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_current_beta005_reanchor_audit.{json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_anchor_drift.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_anchor_drift.{csv,json,md}`


## E081: Full-Image Protocol Re-Anchor for E078-E080

Status: Done. I rechecked the apparent E078-E080 beta005 mismatch under the paper-facing full-image protocol. The key correction is that `tools/probe_householder_inverse_modes.py` now records `patch_size` in JSON and Markdown outputs, because the old current-code direct probes used `patch_size=256` center crops while the historical beta005 paper rows used full images (`patch_size=None`).

The full-image direct beta005 reproduction is exact for practical purposes. The seed3456 beta005 step500 checkpoint gives RD `2.156416023` on OpenImages holdout4096, with mean RD difference `9.75e-08` against the historical per-image `variant500_rd` reference and `nonfinite_rows=0`. The same checkpoint under the center-crop diagnostic gives RD around `2.849824`, so the apparent `+0.694` jump was a protocol mismatch, not current-code collapse and not a reliability-head RVQ failure.

I then re-evaluated the strongest candidate from that branch, E079 `rel095_marginw_rho050_lrm025`, under the same full-image protocol. Step250 gives RD `2.156499396` (`+0.000083` vs beta005) and step500 gives RD `2.156507720` (`+0.000092` vs beta005), both with zero nonfinite rows. Intermediate features stay essentially beta005-preserving: qMSE remains `0.112378`, `s_q_mean` remains `0.456233`, and dead-code ratio remains `0.030533`.

Decision: E079 is safe but not a promotion candidate. The correct interpretation is now: margin-weighted head-only reliability did not cause catastrophic RVQ assignment collapse, but it also did not recover selector headroom under the paper-facing full-image protocol. Beta005 remains the manuscript-safe fixed-checkpoint prototype row.

Artifacts:

- `experiments/analysis/beta005_after250_seed3456_original_config_direct_fullimage_step500_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_fullimage_protocol_reanchor_audit.{json,md}`


## E082: Full-Image Controller Closure and Local-Cap Priority Check

Status: Done. I closed the E080 y-hat-anchor controller branch under the corrected full-image protocol. Both new evaluations used `patch_size=None`, `start_index=4096`, `max_images=4096`, physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), and produced zero nonfinite rows. The y-hat-anchor controller preserves the beta005 operating regime but does not improve RD: step250 is RD `2.157014` (`+0.000598` vs beta005 full-image), and step500 is RD `2.157328` (`+0.000912`). Its qMSE remains around `0.11238`, `s_q_mean` remains `0.456233`, and dead-code ratio remains around `0.0304`, so this is a safe negative/control result rather than a collapse.

I also rechecked whether `local cap080/rho1` should become the new paper-main direction. The answer is no for now. It is strong on trusted OpenImages holdout as a hard-tail specialist (`2.221143`, `-0.042562` vs HCS), but beta005 is still much better on the same split (`2.173688`). On transfer8192, local cap is `2.186255` while beta005 is `2.135355`; on Kodak, local cap is `2.214630` while beta005 is `2.100549`. Therefore local cap is valuable mechanistic evidence and a good source for selective-controller labels, but not a replacement for beta005.

Decision: beta005 remains the manuscript-safe fixed-checkpoint prototype row. E079/E080 remain safe controller guardrails. The next method-improvement branch should be beta005-preserving selective control: learn where to borrow local-cap-like weak geometry, but keep y_hat/assignment/codebook usage close to beta005 unless split-generated evidence says geometry is risky.

Artifacts:

- `experiments/analysis/e082_fullimage_controller_closure.{json,md}`
- `experiments/analysis/e082_local_cap_vs_beta005_crosssplit_audit.{json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`


## E083: Full-Image Controller Family Closure and Teacher-Headroom Gap

Status: Done. I closed the beta005-preserving reliability-controller branch under the corrected paper-facing full-image protocol. The new audit includes E076 binary teacher, E077 stronger rho, E078 margin-weighted teacher, E079 low-amplitude margin-weighted teacher, and E080 y-hat anchor rows, all compared against the same seed3456 beta005 full-image reference. The last missing E078 margin-weighted step500 full-image evaluation was run on physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and produced RD `2.157350`, beta005 delta `+0.000934`, and `nonfinite_rows=0`.

The closure result is clear: every controller variant is numerically stable, but none beats beta005 on mean RD. The best controller is E079 rel095 low-LR step250 at RD `2.156499`, only `+0.000083` worse than beta005. Intermediate features stay in the beta005 regime: qMSE deltas are at most about `+0.000013`, `s_q_mean` remains `0.456233`, and dead-code deltas stay small and negative. This makes the branch a safe negative/control result rather than a collapse.

The useful positive signal is in the per-image/tail analysis. On holdout4096 seed3456, previous-local/local-cap alone is worse than beta005 on average (`+0.088809` RD), but a beta005/previous-local oracle would improve beta005 by `-0.041952` RD and previous-local wins on `25.2686%` of images. On transfer8192 across 3 seeds, the analogous oracle headroom is `-0.055592` RD with a `33.7484%` previous-local win rate. The trained controllers already improve the Q4 hard quartile slightly, but they degrade Q1-Q3 enough to lose on average.

Decision: stop plain BCE/margin-weight scaling for this branch. The next method-improvement experiment should preserve the Q4 hard-image gain while removing easy-image damage: use beta005-preserving selective control with a stronger locality-aware target, such as map/region-level reliability, ranking or distillation against beta005, and explicit qMSE/dead-code/RVQ-assignment preservation. Beta005 remains the manuscript-safe fixed-checkpoint prototype row.

Artifacts:

- `tools/analyze_e083_fullimage_controller_family.py`
- `experiments/analysis/e083_fullimage_controller_family_closure.{json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`


## E084: Selective Controller Threshold Headroom

Status: Done. I analyzed whether the E076-E080 controller rows are uniformly bad or only too broadly applied. This was a diagnostic posthoc audit, not a paper-valid validation protocol, because the feature thresholds are selected on the same holdout4096 rows used for scoring. It is still useful for deciding the next implementation direction.

The strongest deployable-looking signal uses beta005-side Householder delta RMS. E078 margin-weighted step500 is mean-worse when used everywhere (`+0.000934` RD vs beta005), but if it is applied only to high beta005 `rvq_householder_delta_rms` images around threshold `0.052714`, the mixed beta005/controller row improves by about `-0.000610` RD. The selected subset is about 15% of holdout4096 and has selected mean delta around `-0.004062`. The quartile profile is also aligned with the desired failure mode: Q1 is essentially unchanged, Q2 is near zero, Q3 improves slightly, and Q4 hard images improve by about `-0.001965` RD.

Decision: this is not enough to claim a result, but it changes the next experiment. The problem is no longer "the controller is useless"; it is "the controller should not be used globally." The next trainable branch should preserve beta005 on low-risk images/locations and spend reliability suppression only where beta005-side local geometry features indicate risk.

Artifacts:

- `tools/analyze_e084_selective_controller_thresholds.py`
- `experiments/analysis/e084_selective_controller_threshold_headroom.{json,md}`


## E085: Local-Delta Weighted Teacher Negative Control

Status: Done. I implemented a locality-aware teacher weighting path in `hcg_rvq/losses.py` and trained the E085 seed3456 head-only controller from beta005 with `y_hat` anchoring. The teacher label remains image-level, but the loss is weighted on `householder_delta_rms_map` using threshold `0.052714`, sharpness `80.0`, and low-risk weight floor `0.05`. Training and both full-image evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), and no NaN/nonfinite rows appeared.

The result is stable but not a promotion. Step250 reaches RD `2.157009` (`+0.000594` vs beta005), and step500 reaches RD `2.157297` (`+0.000881` vs beta005), with `nonfinite_rows=0` for both. Intermediate features remain safe: qMSE is about `0.112383-0.112390`, `s_q_mean` stays `0.456233`, and dead-code ratio stays around `0.03043-0.03047`.

The mechanism analysis rejects the local-weight-only hypothesis. E085 still improves Q4 hard images slightly but hurts Q1-Q3, so the same global-use failure remains. For step500, quartile deltas are Q1 `+0.002115`, Q2 `+0.001547`, Q3 `+0.000795`, Q4 `-0.000935`. On the high beta005 delta-RMS selector, E085 step500 gives mixed apply-selected-only delta `-0.000571`, slightly weaker than E080 step500 (`-0.000593`) and E078 step500 (`-0.000609`). The reliability high-low separation is also weaker than E080 step500 (`-0.005178` vs `-0.005560`).

Decision: do not promote E085. It is an informative negative showing that local weighting of a scalar teacher is too indirect. The next branch should make selectivity explicit: a selector-style objective that keeps beta005 on low-risk images/locations and supervises suppression only on independently defined high-risk regions, with threshold/teacher chosen on transfer8192 rather than holdout4096.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `tools/analyze_e085_localdelta_weighted_teacher.py`
- `experiments/analysis/e085_localdelta_weighted_teacher_audit.{json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_localdelta_t0527_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`

## E086: Transfer-Derived Selector Keep-Target Closure

Status: Done. I implemented an explicit local keep-target path for the reliability teacher in `hcg_rvq/losses.py`. Unlike E085, which only reweighted a scalar image teacher, E086 changes the local target itself: low-risk locations keep beta005-like reliability, while high `householder_delta_rms_map` locations receive the transfer8192 teacher target. The selector threshold was chosen from the independent transfer8192 teacher-label audit, not from holdout4096: beta005 `rvq_householder_delta_rms >= 0.045151`, selected fraction `0.300049`, previous-local win precision `0.684839`, and recall `0.608874`.

The seed3456 head-only controller was trained from beta005 with the same conservative frozen-codec shell and `y_hat` anchor as E085. Both full-image holdout4096 evaluations were run on physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), and both produced zero nonfinite rows. Step250 gives RD `2.156993` (`+0.000577` vs beta005), and step500 gives RD `2.157220` (`+0.000804`). Intermediate features remain safe: qMSE stays around `0.112383-0.112388`, `s_q_mean` stays `0.456233`, and dead-code ratio stays around `0.03043-0.03048`.

The decision is a careful negative, not a collapse. E086 slightly improves the mean over E085/E080, but it still does not beat beta005 and it weakens the high-risk selector/tail gain. With the transfer-derived selector, E086 step500 has mixed apply-selected-only delta `-0.000387`, weaker than E085 step500 (`-0.000444`) and E080 step500 (`-0.000459`). With the stricter E084 holdout diagnostic selector, E086 step500 gives `-0.000500`, weaker than E078/E080/E085. Quartiles show the same pattern: easy-image damage is reduced, but the Q4 hard-image gain also shrinks (`-0.000811` at step500).

Decision: close the BCE-style reliability-teacher branch for now. It is stable and mechanistically useful, but not paper-main. The next improvement branch should use a direct deployable selector or mixture objective with RD/ranking supervision and explicit beta005 reconstruction/index preservation, rather than another reliability-target BCE variant.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `tools/analyze_e086_transfer_selector_thresholds.py`
- `tools/analyze_e086_selector_keep_target.py`
- `experiments/analysis/e086_transfer_selector_thresholds.{json,md}`
- `experiments/analysis/e086_selector_keep_target_audit.{json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_selector_t0451_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`

## E087: RD-Only Y-Hat Anchored Head Probe

Status: Done. I tested the next controller hypothesis after E086: keep the beta005-initialized frozen-codec shell and `y_hat` anchor, but remove the BCE-style reliability teacher entirely. Only the reliability head is trainable, so the experiment asks whether the actual RD objective can move the head more safely than target BCE.

Training used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). The run loaded the beta005 step500 checkpoint, froze 74 tensors / 12,996,845 parameters, and trained only 2 reliability-head tensors / 965 parameters for 500 steps on the transfer split. No NaN or nonfinite loss was observed.

Full-image holdout4096 evaluations used `patch_size=None`, `start_index=4096`, `max_images=4096`, exact inverse mode, physical GPU0 only, and the beta005 full-image `variant500_rd` reference. Both checkpoints produced zero nonfinite rows:

- Step250: RD `2.156698`, beta005 delta `+0.000282`, qMSE `0.112379`, `s_q_mean=0.456233`, dead-code `0.030525`.
- Step500: RD `2.156626`, beta005 delta `+0.000210`, qMSE `0.112378`, `s_q_mean=0.456233`, dead-code `0.030529`.

The multi-view audit shows E087 step500 is the best row among the E080/E085/E086/E087 controller branch: it improves over E086 step500 by `-0.000594` RD and over E080 step500 by `-0.000703` RD. However, it still does not beat beta005, and it becomes very conservative on the hard-tail subset. On the transfer-derived high delta-RMS selector, E087 step500 has selected delta `+0.000041` and mixed apply-selected-only delta `+0.000013`, while E086/E085/E080 had negative selected deltas. HCS quartiles are all small positive deltas for E087 step500: Q1 `+0.000299`, Q2 `+0.000262`, Q3 `+0.000199`, Q4 `+0.000080`.

Decision: E087 is not a paper-main promotion, but it is the new safest controller baseline. It confirms that outcome-level RD training plus y-hat anchoring is better than BCE teacher targets for preserving beta005, while also showing that RD-only head training is too conservative to recover the beta005/local-cap selective headroom.

Next: move from reliability-target BCE to explicit selection/ranking. The next experiment should keep the E087 RD/y-hat anchored shell, but add a deployable selective or pairwise ranking objective against beta005/local-cap outcomes, with explicit preservation of beta005 reconstruction, RVQ assignment/code usage, qMSE, `s_q`, and dead-code statistics on low-risk regions.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_rdonly_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `tools/analyze_e087_rdonly_head_probe.py`
- `experiments/analysis/e087_rdonly_head_probe_audit.{json,md}`
- `experiments/analysis/rdonly_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/rdonly_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`

## E088: Transfer-Learned Selector Audit

Status: Done. I tested the post-E087 hypothesis before launching another GPU training branch: if the real issue is selectivity, can a selector learned only from the independent transfer8192 split decide when to use beta005 versus previous-local/local-cap behavior on the paper-facing holdout4096 split?

This is a diagnostic multi-checkpoint switch, not a single-checkpoint HCG-RVQ result. It is still important because the selector is trained and threshold-selected on transfer8192, then applied unchanged to holdout4096. The deployable feature group uses only decoder-safe beta005 hyperprior-side quantities such as `s_q`, `mu_q`, raw gate, risk multiplier, Householder strength, and `v_abs_mean`. Diagnostic features include latent/codebook outcome quantities and are treated as an upper bound.

The split headroom is consistent across train and test. On transfer8192, beta005 is RD `2.135355`, previous-local is `2.186255` (`+0.050899`), and the oracle min(beta005, previous-local) is `2.079763` (`-0.055592`), with previous-local winning on `33.7484%` of images. On holdout4096, beta005 is `2.173688`, previous-local is `2.221143` (`+0.047456`), and the oracle is `2.121089` (`-0.052599`), with previous-local winning on `34.7087%` of images.

The main result is strong for a diagnostic. The decoder-safe logistic selector trained on transfer improves holdout beta005 by `-0.027586` RD: holdout RD `2.146101`, HCS delta `-0.117604`, selected fraction `0.242920`, precision `0.728308`, recall `0.509730`, and F1 `0.599724`. The gain is not seed-specific: seed1234 improves `-0.024880`, seed2345 improves `-0.039129`, and seed3456 improves `-0.018749` versus beta005. It also has the desired difficulty profile: HCS Q1 has only a small `+0.000768` RD cost while Q4 hard images improve by `-0.089238` RD, with selected fraction rising from `0.022461` in Q1 to `0.596029` in Q4.

The diagnostic upper bound is only slightly better: diagnostic logistic gives holdout RD `2.145604` (`-0.028083` versus beta005), selected fraction `0.278320`, precision `0.708480`, recall `0.568113`, and F1 `0.630579`. The small gap between decoder-safe and diagnostic policies is encouraging: the information needed for selection is mostly present in decoder-side hyperprior-generated conditioning, not only in unavailable codebook outcome features.

Decision: E088 does not replace beta005 as the manuscript-safe fixed-checkpoint row because it switches between two checkpoints. However, it is the strongest evidence so far that the next single-model improvement should be an explicit selector/ranking controller, not another scalar reliability BCE target. The next branch should distill this decoder-safe transfer selector into one beta005-initialized checkpoint while preserving beta005 `y_hat`, RVQ assignment/code usage, qMSE, `s_q`, and dead-code statistics on low-risk images/locations.

Artifacts:

- `tools/analyze_e088_transfer_learned_selector.py`
- `experiments/analysis/e088_transfer_learned_selector.{csv,json,md}`



## E089/E090: E088 Selector Distillation and RD-Only Polish

Status: Done. I trained the first single-checkpoint distillation of the E088 decoder-safe transfer selector. E089 starts from beta005 seed3456 step500, freezes the codec, trains only the reliability head with `rho_householder_reliability_teacher=0.25`, `householder_reliability_teacher_min=0.75`, `rho_anchor_y_hat=100.0`, and transfer8192 E088 decoder-safe keep labels. Training and all full-image evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), and no NaN/nonfinite rows appeared.

The E089 full-image holdout4096 result is stable but not a promotion. Step250 is RD `2.156997`, beta005 delta `+0.000581`, qMSE `0.112383`, `s_q_mean=0.456233`, dead-code `0.030481`, `nonfinite_rows=0`. Step500 is RD `2.157204`, beta005 delta `+0.000788`, qMSE `0.112387`, `s_q_mean=0.456233`, dead-code `0.030441`, `nonfinite_rows=0`. Intermediate features stay beta005-like, so this is not collapse; it is insufficient selectivity.

I then ran E090 as a small follow-up: initialize from E089 step250 and polish the same reliability head for 250 RD-only steps with the beta005 `y_hat` anchor. E090 improves E089 step250 to RD `2.156843`, beta005 delta `+0.000427`, qMSE `0.112380`, `s_q_mean=0.456233`, dead-code `0.030516`, `nonfinite_rows=0`, but it still does not beat E087 RD-only step500 (`+0.000210`) or beta005.

The audit result is decisive. E088's multi-checkpoint decoder-safe switch improves seed3456 beta005 by `-0.018749` RD, but E089/E090 recover only tiny subset gains. On the E088-selected subset, E089 step250 gives `-0.000638`, E089 step500 gives `-0.001469`, and E090 polish falls back to `-0.000176`. HCS quartiles show why the mean loses: E089 step250 has Q1/Q2/Q3/Q4 deltas `+0.001153/+0.000889/+0.000527/-0.000247`, while E090 reduces easy damage but also removes the hard-tail gain (`+0.000720/+0.000584/+0.000402/+0.000002`).

Decision: do not promote E089/E090. They are useful stable negatives showing that image-mean selector-label BCE, even followed by RD-only polishing, is too indirect to distill E088. The next branch should be an explicit selector/ranking or mixture objective with beta005 preservation, not another scalar reliability target.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_rho025_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_step250_rdpolish_yhatanchor100_headonly_seed3456.yaml`
- `tools/analyze_e089_e088_selector_distill.py`
- `experiments/analysis/e089_e088_selector_distill_audit.{json,csv,md}`
- `experiments/analysis/e088sel_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e088sel_rho025_yhatanchor100_headonly_from_beta005_seed3456_step500_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e090_e088sel_step250_rdpolish_yhatanchor100_headonly_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`

## E091/E092: Selector Gain Decomposition and Distortion-Margin Probe

Status: Done. I decomposed the E088 beta005/previous-local selector headroom into rate and distortion terms before adding another GPU branch. The result is clear: the useful E088 selected-image gain is mostly distortion-side, not rate-side. On transfer8192, the E088 mixed policy improves beta005 by `-0.030460` RD, with `-0.001020` from bpp and `-0.029440` from the distortion term. On holdout4096, it improves beta005 by `-0.027586` RD, with `-0.001027` from bpp and `-0.026559` from distortion. The E088-selected holdout subset has delta `-0.113561` RD, made of `-0.004229` bpp and `-0.109332` distortion. This justified trying a selected-image distortion/ranking loss before exposing a more complex per-image rate objective.

I then implemented E092, a conservative single-checkpoint probe that starts from beta005 seed3456, freezes the codec, trains only the reliability head, keeps a beta005 `y_hat` anchor, and adds a selected-image distortion margin against the beta005 anchor using transfer8192 E088 labels. Training and full-image evaluation used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). The run completed with no observed NaNs, and the holdout4096 full-image exact-inverse evaluation produced `nonfinite_rows=0`.

E092 is stable but not a promotion. Step250 reaches RD `2.156726`, beta005 delta `+0.000310`, qMSE `0.112379`, `s_q_mean=0.456233`, dead-code `0.030527`, delta RMS `0.040178`, and strength `0.263715`. This improves over E089 step250 (`+0.000581`) and E090 (`+0.000427`), but remains worse than E087 RD-only step500 (`+0.000210`) and beta005. More importantly, it does not recover the E088-selected subset or hard-tail gain: selected-subset delta is only `-0.000029`, mixed selected-only delta is `-0.000005`, and Q4 hard images are `+0.000071` vs beta005.

Decision: E091 says the E088 signal is real and distortion-dominated, but E092 says an image-level reliability head plus selected-image distortion margin is still too weak. The next method-improvement branch should use a more explicit local selector/mixture path or another trainable local action with enough capacity to change selected high-risk regions, while preserving beta005 `y_hat`, RVQ assignment/code usage, qMSE, `s_q`, and dead-code statistics on low-risk regions. Beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ row.

Artifacts:

- `tools/analyze_e091_selector_gain_decomposition.py`
- `experiments/analysis/e091_selector_gain_decomposition.{json,csv,md}`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_distmargin_yhatanchor25_headonly_from_beta005_seed3456.yaml`
- `experiments/analysis/e092_e088sel_distmargin_yhatanchor25_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e089_e088_selector_distill_audit.{json,csv,md}`

## E093/E094: Capacity and Reliability-Range Closure for E088 Distillation

Status: Done. I closed two follow-up hypotheses after E092 while staying on the full-image holdout4096 protocol, exact inverse mode, and physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). Both evaluations produced `nonfinite_rows=0`, so these are method results rather than CUDA/device or NaN failures.

E093 tested whether E092 failed only because the reliability head had too little capacity. Starting from beta005 seed3456 step500, it kept the backbone, RVQ, entropy path, and decoder frozen, but unfroze the hyperprior-conditioned geometry heads (`mu_head`, `log_s_head`, Householder head, gate head, and reliability head; 10 tensors / 187,210 trainable parameters). This was a clean negative. Full-image holdout4096 RD is `2.272053`, or `+0.115637` vs beta005. The intermediate distribution moves badly: qMSE rises to `0.154673`, `s_q_mean` drops to `0.418462`, delta RMS rises to `0.054885`, and the E088-selected subset worsens by `+0.301152`. HCS quartiles all degrade, especially Q4 (`+0.231466`). This rules out "just add head capacity" as the next safe path.

E094 then tested the opposite narrow hypothesis: keep only the reliability head trainable, but widen the bounded multiplier range from `0.75..1.0` to `0.50..1.0` while keeping the beta005 `y_hat` anchor. This is stable but still not a promotion. RD is `2.157671`, or `+0.001255` vs beta005. qMSE (`0.112388`), `s_q_mean` (`0.456233`), and dead-code (`0.030437`) remain beta005-like. The desired direction appears only weakly: E088-selected images improve by `-0.000909` and Q4 improves by `-0.000223`, but Q1/Q2/Q3 damage (`+0.002253`, `+0.001809`, `+0.001182`) dominates the mean.

Decision: E093/E094 make the implementation direction clearer. The failure is not CUDA instability, not lack of trainable parameters, and not fixed by widening one scalar reliability multiplier. The next single-checkpoint method branch should expose an explicit bounded selector/mixture path that defaults exactly to beta005 and spends controlled local action only on transfer-selected high-risk regions, with `y_hat`, RVQ assignment/code usage, qMSE, `s_q`, and dead-code guardrails.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_distmargin_yhatanchor10_condheads_from_beta005_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_relmin05_rho025_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `experiments/analysis/e093_e088sel_distmargin_yhatanchor10_condheads_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e094_e088sel_relmin05_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e089_e088_selector_distill_audit.{json,csv,md}`

## E095: Local-Target E088 Reliability Probe

Status: Done. After E094, I tested whether the same safe head-only controller becomes more selective if the E088 teacher is applied locally rather than through an image-mean reliability target. E095 keeps the beta005-initialized frozen-codec shell, keeps only the reliability head trainable (2 tensors / 965 parameters), keeps `rho_anchor_y_hat=100.0`, and uses the transfer-derived high local delta threshold `0.045151` to apply the E088 keep/suppress target only in high-risk local regions. Training and full-image holdout4096 evaluation used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). No NaNs or nonfinite rows appeared.

The result is stable but still not a promotion. Full-image holdout4096 RD is `2.157624`, or `+0.001208` vs beta005. This is a small improvement over E094 (`+0.001255`), but it remains worse than E092 (`+0.000310`), E087 (`+0.000210`), and beta005. Intermediate features stay safe: qMSE `0.112387`, `s_q_mean=0.456233`, dead-code `0.030436`, delta RMS `0.039594`, and strength `0.260640`, with `nonfinite_rows=0`.

The subset analysis shows why it does not promote. E095 improves E088-selected images by only `-0.000836` and Q4 by only `-0.000189`, while Q1/Q2/Q3 still lose `+0.002156`, `+0.001730`, and `+0.001137`. Compared with E094, the local target slightly reduces easy-image damage but also weakens the selected/Q4 benefit. This confirms that reusing the current scalar reliability head with local BCE targets is still too blunt.

Decision: close this scalar-head distillation family for now. The next implementation should add an explicit bounded selector/mixture module rather than another reliability-head BCE variant. The required default behavior is exact beta005; the controlled branch should be activated only on transfer-selected high-risk regions and must preserve `y_hat`, RVQ assignment/code usage, qMSE, `s_q`, and dead-code statistics.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_relmin05_localtarget_t0451_rho025_yhatanchor100_headonly_from_beta005_seed3456.yaml`
- `experiments/analysis/e095_e088sel_relmin05_localtarget_t0451_rho025_yhatanchor100_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e089_e088_selector_distill_audit.{json,csv,md}`


## E096/E097: Exact-Default Residual Selector Branch

Status: Done. I implemented and tested the first explicit residual-selector branch after the E088/E095 scalar-head family closed. The implementation adds a trainable `householder_residual_selector_head` that defaults exactly to beta005: at zero-initialized head weights, selector probability is `0` and the Householder gate multiplier is `1`. A CPU identity check confirmed `x_hat_max_abs=0.0`, `y_hat_max_abs=0.0`, `selector_prob=0.0`, and `selector_mult=1.0` when loading the beta005 checkpoint into the selector-enabled config. A loss smoke test also confirmed nonzero selector-head gradients.

E096 trains only the residual selector head from beta005 seed3456 step500 using transfer8192 E088 labels, local delta weighting at threshold `0.045151`, `rho_householder_residual_selector_teacher=0.25`, `selector_max=0.50`, and `rho_anchor_y_hat=100.0`. Training used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), loaded 4096 teacher labels, froze 76 tensors / 12,997,810 parameters, and trained only 2 tensors / 965 parameters. Full-image holdout4096 evaluation also used physical GPU0 only and produced `nonfinite_rows=0`.

E096 is stable and directionally useful, but not a paper-main promotion. Path-matched full-image RD is `2.156687`, beta005 delta `+0.000271`, HCS delta `-0.106658`, mean abs delta `0.001614`, and max abs delta `0.051807`. Intermediate features remain beta005-like: qMSE `0.112382` vs beta005 `0.112377`, `s_q_mean=0.456233`, dead-code `0.030499`, perplexity `68.788603`, delta RMS `0.039959`, and Householder strength `0.263105`. The selector is strongly conservative: mean selector prob `0.011674`, mean max prob `0.034769`, and mean multiplier `0.994163`.

Subset analysis explains the mean. E096 improves the targeted regions: E088-selected images `-0.000874`, previous-local-win images `-0.000519`, beta delta-RMS >= `0.052714` tail `-0.001709`, and HCS Q4 `-0.000486`. But Q1/Q2/Q3 lose `+0.000792`, `+0.000548`, and `+0.000229`, leaving the average worse. The selector is aligned with the intended signals: selector prob correlation is `+0.622207` with the E088 decoder-safe score and `+0.812003` with beta delta RMS; it is also negatively correlated with RD delta (`-0.320898`), meaning higher selector action tends to coincide with better per-image delta. The problem is magnitude/precision, not collapse.

E097 tested whether stronger, more tail-focused action fixes E096. It uses the same exact-default selector but changes `selector_max=0.75`, `rho_householder_residual_selector_teacher=1.0`, local delta threshold `0.052714`, and low-risk local teacher weight `0.0`. Training again used physical GPU0 only, kept the same 965 trainable parameters, and full-image evaluation had `nonfinite_rows=0`. E097 improves targeted subsets more than E096: E088-selected `-0.001322`, previous-local-wins `-0.000765`, beta delta-RMS >= `0.052714` `-0.002606`, and Q4 `-0.000711`. However, easy damage also increases: Q1/Q2/Q3 become `+0.001293`, `+0.000893`, and `+0.000410`, and mean RD worsens to `2.156887` (`+0.000471` vs beta005).

Decision: the exact-default residual selector is a useful implementation primitive and should be kept. It validates a safe single-checkpoint way to expose selective local control without moving the broader hyperprior geometry heads. But E096/E097 are not paper-main rows yet. The next method-improvement step should improve precision rather than simply increasing suppression strength: use ranking/outcome supervision, add a controlled alternative geometry branch, or regularize easy-image no-op behavior more directly. Beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ result.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_yhatanchor100_from_beta005_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0527_rho100_max075_lmin000_yhatanchor100_from_beta005_seed3456.yaml`
- `experiments/analysis/e096_e088sel_residualselector_localdelta_t0451_rho025_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e097_e088sel_residualselector_localdelta_t0527_rho100_max075_lmin000_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e096_residual_selector_audit.{csv,json,md}`
- `experiments/analysis/e089_e088_selector_distill_audit.{csv,json,md}`

## E098: Low-Delta No-Op Regularized Residual Selector

Status: Done. I tested the precision fix proposed after E097: keep the exact-default residual selector, keep the stronger E097 tail action (selector_max=0.75, rho_householder_residual_selector_teacher=1.0, high-delta threshold 0.052714), and add a local no-op BCE loss on low householder_delta_rms_map regions below threshold 0.045151. The goal was to keep the Q4/tail benefit while reducing Q1-Q3 easy-image damage. Training and full-image holdout4096 evaluation used physical GPU0 only (CUDA_VISIBLE_DEVICES=0, --device cuda:0), trained only the residual selector head (2 tensors / 965 parameters), and produced nonfinite_rows=0.

The result is directionally correct but too small to promote. Full-image holdout4096 RD is 2.156862, beta005 delta +0.000446, HCS delta -0.106483, mean abs delta versus beta005 0.002410, and max abs delta 0.075172. Intermediate features remain safe: qMSE 0.112384, s_q_mean=0.456233, dead-code 0.030464, perplexity 68.793567, delta RMS 0.039784, and Householder strength 0.262350.

The quartile/subset behavior confirms the interpretation. Compared with E097, E098 slightly reduces easy damage: Q1/Q2/Q3 go from +0.001293/+0.000893/+0.000410 to +0.001230/+0.000847/+0.000387. But it also slightly weakens the desired tail benefit: Q4 goes from -0.000711 to -0.000679, and E088-selected images go from -0.001322 to -0.001256. So the no-op loss is aligned with the failure mode, but the recovery is not large enough; it remains worse than E096 mean (+0.000271) and beta005.

Decision: keep the no-op-loss machinery as a diagnostic/regularization option, but do not make E098 a paper-main row. The next branch should be more outcome-aware: pairwise/ranking supervision against beta005 versus previous-local outcomes, or a controlled alternative geometry branch gated by the exact-default selector. Further increasing suppression or adding broad regularizers is unlikely to solve the precision bottleneck by itself.

Artifacts:

- configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0527_rho100_max075_nooplow_t0451_rho100_yhatanchor100_from_beta005_seed3456.yaml
- experiments/analysis/e098_e088sel_residualselector_localdelta_t0527_rho100_max075_nooplow_t0451_rho100_yhatanchor100_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}
- experiments/analysis/e089_e088_selector_distill_audit.{csv,json,md}


## E099-E105: Dead-Zone Residual Selector Calibration

Status: Done for the seed3456 step250 diagnostic branch. I evaluated a deploy-time dead-zone on the same E099 exact-default residual-selector checkpoint. The dead-zone leaves low-confidence residual-selector probabilities at exact beta005 no-op, so this tests whether the precision failure in E096-E099 can be fixed without changing the trained checkpoint. All full-image evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and produced `nonfinite_rows=0`.

The holdout4096 sweep is now monotonic through the tested range. Relative to beta005, deadzone010 is `+0.000126`, deadzone012 is `-0.000062`, deadzone014 is `-0.000253`, deadzone016 is `-0.000373`, and deadzone018 is `-0.000423`. E104/deadzone018 is the current best single-checkpoint diagnostic: RD `2.155993`, HCS delta `-0.107352`, mean abs delta vs beta `0.000802`, qMSE `0.112380`, `s_q_mean=0.456233`, dead-code `0.030527`, delta RMS `0.040125`, strength `0.264286`, and `nonfinite_rows=0`.

The subset and quartile behavior is the important part. E104 improves the E088-selected subset by `-0.001359` and Q4 hard images by `-0.000993`; Q1/Q2/Q3 are `-0.000018`, `-0.000200`, and `-0.000481`, so the previous easy-image damage has essentially disappeared. This is the first single-checkpoint residual-selector row in this branch that beats beta005 on the mean while also preserving the hard-tail gain.

I also checked the start8192 transfer/calibration slice instead of continuing to tune on holdout. There, deadzone014 gives `-0.000258`, deadzone016 gives `-0.000380`, deadzone018 gives `-0.000425`, and deadzone020 falls back to `-0.000380` vs the seed3456 beta005 reference. This brackets the useful threshold around 0.018 and gives a reason to stop before extra holdout tuning.

Decision: E104/deadzone018 is promising but not paper-main yet. It is a calibrated-threshold candidate that must be locked by a split/rule and then rerun path-matched over all seeds and checkpoint choices. Beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ row until the 3-seed confirmation is done, but E104 is now the best method-improvement branch.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_deadzone018_from_beta005_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_deadzone020_from_beta005_seed3456.yaml`
- `experiments/analysis/e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e104_e099_deadzone018_from_beta005_seed3456_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e105_e099_deadzone020_from_beta005_seed3456_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e089_e088_selector_distill_audit.{csv,json,md}`

## E106: E104 Dead-Zone Multi-Seed Confirmation

Status: Done for fixed checkpoint step250 direct-probe confirmation. I trained the E099 residual-selector head for seed1234 and seed2345 from the beta005 checkpoints, then evaluated E104/deadzone018 against newly regenerated beta005 direct references on holdout4096. All training and evaluation commands used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0` or `--device cuda`) and produced zero nonfinite rows.

The new seed results are positive. Seed1234 beta005 direct RD is `2.158153`; E104/deadzone018 reaches `2.155694`, delta `-0.002459`, win rate `0.833740`, qMSE delta `+0.000003`, delta-RMS delta `-0.000307`, strength delta `-0.000837`, and `nonfinite_rows=0`. Seed2345 beta005 direct RD is `2.206494`; E104 reaches `2.201672`, delta `-0.004822`, win rate `0.930908`, qMSE delta `+0.000001`, delta-RMS delta `-0.000281`, strength delta `-0.000723`, and `nonfinite_rows=0`.

Combining seed1234/2345 with the existing seed3456 E104 result gives the current strongest single-checkpoint HCG-RVQ branch: beta005 RD `2.173688`, E104 RD `2.171120`, mean delta `-0.002568`, median per-image delta `-0.000895`, win rate `0.763021`, q05/q95 delta `-0.011170`/`+0.000591`, and `nonfinite_rows=0` over 12,288 evaluated images. The aggregate beta-RD quartiles all improve: Q1 `-0.001294`, Q2 `-0.002117`, Q3 `-0.002881`, and Q4 `-0.003980`.

Intermediate-feature distributions support a controlled-policy interpretation rather than collapse. Aggregate qMSE changes only `+0.000002`, `s_q_mean` is unchanged to numerical precision, dead-code ratio improves by `-0.000025`, delta RMS decreases by `-0.000257`, and Householder strength decreases by `-0.000654`. This means the dead-zone is not moving the quantizer into a new unstable regime; it is removing low-confidence residual-selector action while keeping useful local geometry control.

Decision: E104/deadzone018 graduates from a seed3456 rescue diagnostic to the next manuscript-candidate HCG-RVQ branch under the fixed-checkpoint direct-probe protocol. It is still not final paper-main until the threshold selection rule is locked independently and checkpoint step250/500 selection is audited, but the multi-seed evidence is now strong enough to prioritize checkpoint sweep and protocol-locking over more one-off threshold tuning.

Artifacts:

- `tools/analyze_e104_multiseed_deadzone.py`
- `experiments/analysis/beta005_after250_seed1234_direct_exact_step500_holdout4096_current.{csv,json,md}`
- `experiments/analysis/beta005_after250_seed2345_direct_exact_step500_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e104_deadzone018_from_beta005_seed1234_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e104_deadzone018_from_beta005_seed2345_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e104_multiseed_deadzone018_audit.{json,md,per_seed.csv,quartiles.csv,per_image.csv}`

## E107: E104 Dead-Zone Transfer-vs-Holdout Confirmation

Status: Done. I checked whether the E104/deadzone018 gain is a holdout-only artifact by evaluating the same fixed checkpoints on the independent OpenImages start8192 transfer split. For seed1234 and seed2345 I first exported seed-specific beta005 guard reference CSVs from the existing transfer artifact, avoiding path-only reference mixing across seeds. Then I evaluated the E104/deadzone018 checkpoints with physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). All runs produced `nonfinite_rows=0`.

The result is unusually aligned with holdout. On start8192, the 3-seed beta005 RD is `2.135355` and E104/deadzone018 is `2.132781`, giving delta `-0.002574` and win rate `0.760498`. The holdout delta from E106 is `-0.002568`, so transfer-minus-holdout is only `-0.000006`. Per seed, seed1234 is `-0.002485` on transfer versus `-0.002459` on holdout, seed2345 is `-0.004813` versus `-0.004822`, and seed3456 is `-0.000425` versus `-0.000423`.

The intermediate feature deltas on transfer are also controlled: qMSE changes only `+0.000001` to `+0.000003`, `s_q_mean` is unchanged to numerical precision, dead-code decreases, delta RMS decreases, and Householder strength decreases. This matches the holdout interpretation that dead-zone calibration suppresses low-confidence residual-selector action without moving the quantizer into a new distribution.

Decision: E104/deadzone018 is no longer just a holdout-positive row. It now has a near-identical 3-seed gain on an independent transfer split, which makes it the current manuscript-candidate HCG-RVQ branch. It still needs a pre-declared threshold-selection rule and checkpoint sweep before final paper-main promotion, but the "holdout overfit" risk is now substantially reduced.

Artifacts:

- `experiments/analysis/beta005_seed1234_transfer_start8192_reference.csv`
- `experiments/analysis/beta005_seed2345_transfer_start8192_reference.csv`
- `experiments/analysis/e104_deadzone018_from_beta005_seed1234_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e104_deadzone018_from_beta005_seed2345_step250_fullimage_start8192_current.{csv,json,md}`
- `tools/analyze_e106_transfer_vs_holdout_deadzone.py`
- `experiments/analysis/e106_deadzone018_transfer_vs_holdout_audit.{json,md,per_seed.csv}`

## E108: Independent Dead-Zone Threshold Selection

Status: Done. I used the independent start8192 transfer split to select the residual-selector dead-zone threshold instead of continuing to tune on holdout4096. For seed1234 and seed2345 I added the missing deadzone014/016/020 configs under the same frozen-codec, beta005-initialized, exact-default residual-selector branch. I then evaluated deadzone014/016/018/020 for all three seeds on start8192 using physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). All runs produced `nonfinite_rows=0`.

The transfer split selects deadzone014 by mean RD. Aggregate start8192 deltas are: dz014 `-0.002827`, dz016 `-0.002762`, dz018 `-0.002574`, and dz020 `-0.002253`. The previous manuscript candidate dz018 remains close, only `0.000252` RD worse than dz014, and it has the safer tail behavior: dz018 win rate is `0.760498` and q95 damage is `+0.000641`, while dz014 win rate is `0.729085` and q95 damage is `+0.001516`.

The per-seed transfer deltas are all positive evidence for dz014: seed1234 `-0.002633`, seed2345 `-0.005590`, and seed3456 `-0.000258`. All beta-RD quartiles also improve for dz014: Q1 `-0.001394`, Q2 `-0.002476`, Q3 `-0.003191`, and Q4 `-0.004246`. Intermediate features remain in the controlled regime: qMSE `0.107127`, `s_q_mean=0.463857`, dead-code `0.031939`, delta RMS `0.039725`, strength `0.262118`, and no nonfinite rows.

Decision: the pre-declared transfer-split rule chooses dz014, but dz018 remains the conservative holdout-safe threshold. Therefore dz014 needed a separate holdout confirmation on seed1234/2345 before replacing dz018 as the main candidate.

Artifacts:

- `tools/run_e108_deadzone_transfer_sweep.py`
- `tools/analyze_e108_deadzone_threshold_selection.py`
- `experiments/analysis/e108_deadzone014_from_beta005_seed1234_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone016_from_beta005_seed1234_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone020_from_beta005_seed1234_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone014_from_beta005_seed2345_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone016_from_beta005_seed2345_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone020_from_beta005_seed2345_step250_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e108_deadzone_transfer_threshold_selection_audit.{json,md,thresholds.csv,per_seed.csv,quartiles.csv,seed3456_holdout.csv}`

## E109: Deadzone014 Holdout Confirmation

Status: Done. I confirmed the transfer-selected deadzone014 threshold on holdout4096 for seed1234 and seed2345, then combined those results with the existing seed3456 dz014 holdout anchor. The evaluations used the same fixed-checkpoint direct-probe protocol and physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). All runs produced `nonfinite_rows=0`.

Deadzone014 is holdout-confirmed as the stronger mean-RD setting. The 3-seed holdout4096 aggregate is beta005 RD `2.173688`, dz014 RD `2.170857`, delta `-0.002830`, win rate `0.729329`, q05/q95 delta `-0.012103`/`+0.001343`, and `nonfinite_rows=0`. Deadzone018 remains close at RD `2.171120`, delta `-0.002568`, win rate `0.763021`, q05/q95 `-0.011170`/`+0.000591`, and `nonfinite_rows=0`.

The same-image dz014-minus-dz018 comparison is the cleanest checkpoint-free view: aggregate mean is `-0.000262`, median `-0.000147`, and dz014 is better on `62.7197%` of images. Per seed, dz014 improves over dz018 for seed1234 by `-0.000144` and seed2345 by `-0.000813`, but is worse for seed3456 by `+0.000170`. This explains the trade-off: dz014 is the mean-RD choice selected by transfer and confirmed on holdout, while dz018 is still the safer tail/win-rate setting.

Intermediate features remain controlled, so the gain is not a collapse or co-adaptation artifact. For dz014, qMSE is `0.106822`, `s_q_mean=0.463063`, dead-code `0.031848`, delta RMS `0.039878`, and strength `0.262256`; dz018 is essentially the same regime with qMSE `0.106821`, `s_q_mean=0.463063`, dead-code `0.031867`, delta RMS `0.039951`, and strength `0.262628`.

Decision: promote dz014 as the mean-RD manuscript-candidate threshold under the independently selected dead-zone protocol, while retaining dz018 as a conservative ablation/safety setting. The next required action is checkpoint/protocol auditing: because the residual-selector configs currently train for 250 steps, the paper should explicitly state this pre-declared checkpoint or run a separate long-step audit before using checkpoint selection as a claim.

Artifacts:

- `tools/run_e109_deadzone014_holdout_confirmation.py`
- `tools/analyze_e109_deadzone014_holdout_confirmation.py`
- `experiments/analysis/e109_deadzone014_from_beta005_seed1234_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e109_deadzone014_from_beta005_seed2345_step250_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e109_deadzone014_holdout_confirmation_audit.{json,md,thresholds.csv,per_seed.csv,quartiles.csv,pairwise.csv}`

## E110: Residual-Selector Max500 Checkpoint Audit

Status: Done. I added separate max500 residual-selector configs for seed1234/2345/3456, trained the same exact-default controller budget to checkpoint_step_500, and evaluated both deadzone014 and deadzone018 on holdout4096. All training/evaluation used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and every max500 evaluation produced `nonfinite_rows=0`.

The checkpoint audit gives a strong but not yet paper-clean result. The step250 baseline from E109 was dz014 RD `2.170857`, delta `-0.002830`, and dz018 RD `2.171120`, delta `-0.002568`. The max500 runs improve the 3-seed mean substantially: dz014 reaches RD `2.165810`, delta `-0.007877`, and dz018 reaches RD `2.165786`, delta `-0.007901`. The gain is mainly from seed1234 and seed2345: max500 dz014 deltas are seed1234 `-0.006894`, seed2345 `-0.017357`, and seed3456 `+0.000619`; max500 dz018 deltas are seed1234 `-0.006878`, seed2345 `-0.017349`, and seed3456 `+0.000522`.

The same-image checkpoint comparison confirms the trade-off. Max500 beats step250 on average by `-0.005047` RD for dz014 and `-0.005333` RD for dz018, but it wins on only about `66.6%` of images and reopens a large positive tail: q95 is `+0.006271` for dz014 and `+0.006098` for dz018, compared with step250 q95 `+0.001343` and `+0.000591`. Intermediate features stay controlled rather than collapsed: max500 qMSE remains about `0.106832`, `s_q_mean` stays `0.463063`, dead-code is `0.031743`, and the mean selector probability increases from the very sparse step250 regime to about `0.030`-`0.032`.

Decision: max500 is a promising high-mean checkpoint-budget candidate, but not yet the clean manuscript-main rule. It improves average RD much more than step250, yet seed3456 becomes worse than its beta reference and the q95 damage increases. For a conference-facing claim, the next step should either predeclare an independent checkpoint-selection rule or stabilize the max500 branch so seed3456 no longer regresses. Until then, step250 dz014 remains the cleaner independently selected threshold result, and max500 should be reported as a checkpoint-audit/high-mean candidate.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_max500_seed1234.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_max500_seed2345.yaml`
- `configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_excessrisk090_betacommit005_e088sel_residualselector_localdelta_t0451_rho025_distmargin_m020_rho250_keep005_yhatanchor50_from_beta005_max500_seed3456.yaml`
- `tools/run_e110_residual_selector_max500_audit.py`
- `tools/analyze_e110_residual_selector_max500_checkpoint_audit.py`
- `experiments/analysis/e110_deadzone014_from_beta005_max500_seed{1234,2345,3456}_step500_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e110_deadzone018_from_beta005_max500_seed{1234,2345,3456}_step500_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e110_residual_selector_max500_checkpoint_audit.{json,md,budget_thresholds.csv,per_seed.csv,quartiles.csv,pairwise.csv}`

## E111: Max500 Transfer Checkpoint Audit

Status: Done. I evaluated the E110 max500 checkpoints on the independent start8192 transfer split for both deadzone014 and deadzone018, using the same seed-specific beta005 transfer references as E107/E108. All evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and every run produced `nonfinite_rows=0`.

The independent transfer split confirms the main E110 pattern. Step250 dz014 had delta `-0.002827`, and max500 dz014 reaches `-0.007890`, a further `-0.005063` RD gain. Step250 dz018 had delta `-0.002574`, and max500 dz018 reaches `-0.007913`, a further `-0.005339` RD gain. The best transfer mean is max500 dz018: RD `2.127442` versus beta005 RD `2.135355`, win rate `0.684489`, q05/q95 `-0.036662`/`+0.006287`, and `nonfinite_rows=0`.

The caveat also transfers. Max500 is very strong for seed1234 and seed2345 but still slightly worse than beta005 on seed3456: dz014 has seed deltas `-0.007007`, `-0.017256`, and `+0.000592`; dz018 has `-0.006989`, `-0.017247`, and `+0.000496`. Same-image max500-minus-step250 comparisons are negative on average (`-0.005063` for dz014 and `-0.005339` for dz018), but seed3456 is positive (`+0.000850`/`+0.000921`) and q95 damage remains around `+0.0051` to `+0.0058` in the pairwise view.

Intermediate features remain controlled on transfer. The max500 qMSE values stay close to the step250 regime (`0.107239`, `0.101411`, `0.112764` by seed), `s_q_mean` is unchanged by checkpoint budget at the displayed precision, and dead-code remains around `0.030`-`0.033`. This confirms that the issue is checkpoint-policy/tail risk, not NaN, VQ collapse, or `s_q` co-adaptation.

Decision: max500 is now supported by both holdout and independent transfer as a high-mean strengthening direction, but it is still not the cleanest paper-main rule because the fragile seed3456 regression and larger q95 tail reproduce across splits. The safest current manuscript posture is step250 dz014 as the independently selected threshold result, dz018 as conservative ablation, and max500 as a checkpoint-budget candidate that needs either a predeclared checkpoint-selection rule or a stabilization experiment before promotion.

Artifacts:

- `tools/run_e111_residual_selector_max500_transfer_audit.py`
- `tools/analyze_e111_residual_selector_max500_transfer_checkpoint_audit.py`
- `experiments/analysis/e111_deadzone014_from_beta005_max500_seed{1234,2345,3456}_step500_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e111_deadzone018_from_beta005_max500_seed{1234,2345,3456}_step500_fullimage_start8192_current.{csv,json,md}`
- `experiments/analysis/e111_residual_selector_max500_transfer_checkpoint_audit.{json,md,budget_thresholds.csv,per_seed.csv,quartiles.csv,pairwise.csv}`

## E112: Max500 Selector-Cap Seed3456 Probe

Status: Done. I tested whether the max500 seed3456 regression is caused mainly by overly strong residual-selector action. Without retraining, I lowered `householder_gate_residual_selector_max` for the max500 deadzone018 seed3456 checkpoint and evaluated caps `0.25`, `0.35`, and `0.45` on the independent start8192 transfer split. The cap `0.50` baseline is the E111 result. The selected cap was then checked on holdout4096. All evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and produced `nonfinite_rows=0`.

The transfer split selected cap `0.25`: cap0.50 delta `+0.000496`, cap0.45 `+0.000413`, cap0.35 `+0.000278`, and cap0.25 `+0.000167` versus beta005. The same selected cap also reduced the holdout seed3456 damage from the E110/E111 max500 regime to `+0.000178` versus beta005. This is a meaningful reduction of the fragile-seed regression, but it does not cross into a positive seed3456 win.

Intermediate features stayed in the same controlled regime. At the selected cap0.25, transfer qMSE is `0.112758`, `s_q_mean=0.457048`, dead-code `0.030458`; holdout qMSE is `0.112383`, `s_q_mean=0.456233`, dead-code `0.030476`. This supports the E110/E111 diagnosis that max500's problem is policy strength/tail risk rather than NaN, VQ collapse, or `s_q` co-adaptation.

Decision: selector capping is useful as a stabilization diagnostic and a future control knob for the max500 branch, but it is not enough to promote max500 as paper-main. The current paper-safe ordering remains: step250 dz014 as the independently selected mean-RD branch, dz018 as conservative safety ablation, and max500 as a high-mean strengthening branch that still needs either a predeclared checkpoint/cap rule or a training-time stabilization mechanism.

Artifacts:

- `tools/run_e112_max500_selector_cap_seed3456_probe.py`
- `experiments/analysis/e112_deadzone018_cap025_max500_seed3456_step500_fullimage_transfer_current.{csv,json,md}`
- `experiments/analysis/e112_deadzone018_cap035_max500_seed3456_step500_fullimage_transfer_current.{csv,json,md}`
- `experiments/analysis/e112_deadzone018_cap045_max500_seed3456_step500_fullimage_transfer_current.{csv,json,md}`
- `experiments/analysis/e112_deadzone018_cap025_max500_seed3456_step500_fullimage_holdout_current.{csv,json,md}`
- `experiments/analysis/e112_max500_selector_cap_seed3456_probe.{json,csv,md}`

## E113: Max500 Selector-Cap Multi-Seed Audit

Status: Done. I extended the E112 selector-cap probe from seed3456-only to a 3-seed protocol for max500 deadzone018. The cap was selected on the independent start8192 transfer split before looking at holdout4096. I evaluated deploy-time caps `0.25`, `0.35`, and `0.45` for seed1234 and seed2345, reused the E112 seed3456 cap probes, and reused the E111/E110 cap0.50 baseline rows. All evaluations used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and all rows had `nonfinite_rows=0`.

The 3-seed transfer split selects the original cap0.50, not a reduced cap. Aggregate transfer deltas versus beta005 are cap0.25 `-0.004244`, cap0.35 `-0.005774`, cap0.45 `-0.007220`, and cap0.50 `-0.007913`. The selected cap0.50 holdout result is the E110 max500 deadzone018 row: RD `2.165786`, delta `-0.007901`, qMSE `0.106832`, `s_q_mean=0.463063`, dead-code `0.031743`, and `nonfinite_rows=0`.

The per-seed pattern explains the decision. Reducing the cap helps seed3456: E112 showed transfer damage improves from cap0.50 `+0.000496` to cap0.25 `+0.000167`, and holdout damage improves to `+0.000178`. But the same cap reduction removes too much useful action from the strong seeds. On transfer, seed1234 moves from cap0.50 `-0.006989` to cap0.25 `-0.003809`, and seed2345 moves from `-0.017247` to `-0.009089`. Thus a single global deploy-time cap is not a clean paper-main fix.

Decision: E113 should be treated as a negative control and stabilization diagnosis. It confirms that max500 fragile-seed regression is related to action strength, but also shows that global selector-cap calibration alone cannot make max500 paper-clean without losing the mean gain. The safe manuscript ordering remains step250 dz014 as the independently selected mean-RD branch, dz018 as conservative safety ablation, and max500 as a high-mean strengthening candidate. The next serious max500 action should be training-time stabilization or a predeclared learned/per-image cap policy, not another global cap sweep.

Artifacts:

- `tools/run_e113_max500_selector_cap_multiseed_audit.py`
- `experiments/analysis/e113_deadzone018_cap{025,035,045}_max500_seed{1234,2345}_step500_fullimage_transfer_current.{csv,json,md}`
- `experiments/analysis/e113_max500_selector_cap_multiseed_audit.{json,md,per_seed.csv,aggregates.csv}`

## E114: Max500 Per-Image Selector-Cap Headroom

Status: Done. I analyzed the E113/E112 per-image transfer rows to estimate whether conditional selector-cap control has enough headroom to justify a learned/per-image controller. This used only existing start8192 transfer artifacts for max500 deadzone018 caps `0.25`, `0.35`, `0.45`, and `0.50`; no new GPU evaluation was required.

The per-image oracle is positive but modest. The cap0.50 transfer baseline is delta `-0.007913`; choosing the best cap per image among the four caps reaches `-0.008545`, an extra `-0.000632` RD. The oracle also fixes the fragile seed3456 transfer sign, moving it from cap0.50 `+0.000496` to `-0.000606`. Oracle cap usage is mostly cap0.50 (`7725` images) with cap0.25 used on `3641` images, cap0.35 on `468`, and cap0.45 on `454`.

A simple single-feature threshold is not reliable enough. The best in-sample threshold policy uses `rvq_householder_residual_selector_multiplier_min >= 0.953656` to switch to cap0.25 on about `20.0%` of images, improving the transfer mean to `-0.008064` (`-0.000150` vs cap0.50). But leave-one-seed CV gives delta `-0.007894`, which is `+0.000020` worse than cap0.50. The feature correlations are still informative: lower-cap gain is most related to selector probability/multiplier features (`|r|` around `0.42`) and then local delta statistics.

Decision: conditional cap control has real headroom, including a route to rescue seed3456, but a hand-written single-threshold rule is not paper-clean. The next strengthening direction should be a learned or multi-feature reliability/cap controller trained on an independent teacher split, with an explicit no-overfit protocol. Until that exists, E114 supports max500 as future headroom and keeps step250 dz014/dz018 as the safer manuscript-main branch.

Artifacts:

- `tools/analyze_e114_max500_per_image_cap_headroom.py`
- `experiments/analysis/e114_max500_per_image_cap_headroom.{json,md,policies.csv,cv.csv,correlations.csv}`

## E115: Max500 Learned Cap-Selector Cross-Validation

Status: Done. I tested whether E114 oracle headroom can be recovered by a simple learned multi-feature selector instead of a hand-written threshold. The audit is transfer-only and uses nested leave-one-seed validation: for each held seed, feature set and ridge regularization are selected by inner seed CV on the remaining two seeds, then evaluated on the held seed. No GPU evaluation was required.

The learned selector improves over cap0.50, but only modestly. The cap0.50 transfer baseline is delta `-0.007913`; the per-image oracle is `-0.008545`; the learned leave-one-seed CV selector reaches `-0.008009`, a gain of `-0.000096` over cap0.50. It is better than the E114 single-threshold CV result, but it recovers only about 15% of the oracle headroom.

The per-seed behavior is useful for diagnosis. Held seed1234 is essentially unchanged (`-0.000003` vs cap0.50), seed2345 is slightly worse (`+0.000044`), and seed3456 improves by `-0.000329`, reducing its transfer regression from `+0.000496` to `+0.000167`. The learned policy mostly chooses cap0.25 on seed3456, while keeping cap0.50 for most seed1234/2345 images.

Decision: multi-feature learned selection is a real improvement over a single threshold and supports the existence of a learnable reliability signal, but the gain is too small to justify promoting max500 as paper-main. It is also not yet worth adding a complicated deployable controller unless the next version can recover more oracle headroom without sacrificing seed2345. The current manuscript ordering remains step250 dz014 as the main branch, dz018 as conservative safety ablation, and max500 as validated headroom with a clear future path.

Artifacts:

- `tools/analyze_e115_max500_learned_cap_selector_cv.py`
- `experiments/analysis/e115_max500_learned_cap_selector_cv.{json,md,outer_cv.csv,in_sample.csv}`

## E116: HCG-RVQ Submission Readiness Package

Status: Done. I consolidated the current paper-facing evidence into a submission-readiness package so that the claim boundary is explicit before adding more experiments. This was an analysis-only step and did not require GPU evaluation.

The readiness statement is positive but not final: the research is progressing well enough for a serious international-conference submission, because the beta005 prototype already has broad fixed-checkpoint evidence and the newer dead-zone branch is protocol-clean on OpenImages. The remaining blockers are external confirmation for the stronger dead-zone branch and later SOTA/backbone comparisons.

The current claim stack is now explicit. `beta005 guard` remains the broad prototype baseline because it has OpenImages holdout/transfer, Kodak, and CLIC mobile/professional evidence. Step250 `deadzone014` is the current mean-RD manuscript candidate because it was selected on start8192 and confirmed on holdout4096. Step250 `deadzone018` is the conservative safety ablation because it has lower positive-tail damage. Max500 remains high-mean headroom rather than paper-main because it reopens seed3456/tail risk.

Decision: continue with the two-layer manuscript structure: beta005 as the broad, externally checked prototype row; deadzone014/deadzone018 as the stronger HCG-RVQ controller branch; max500 as headroom/future stabilization. The immediate next action is to evaluate dz014/dz018 on Kodak and CLIC against seed-matched beta005 references.

Artifacts:

- `tools/build_e116_submission_readiness_package.py`
- `experiments/analysis/e116_hcg_rvq_submission_readiness_package.{json,md,claim_rows.csv,selector_rows.csv}`

## E117: Dead-Zone External Fixed-Protocol Audit

Status: Done. I evaluated step250 `deadzone014` and `deadzone018` on Kodak, CLIC mobile valid, and CLIC professional valid against seed-matched `beta005 guard` references. The run used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) and produced `nonfinite_rows=0` for all rows.

The result is stronger than the previous boundary. Both dead-zone thresholds improve over beta005 on all three external splits in aggregate. Weighted over all external images, dz014 reaches mean delta `-0.001650` versus beta005, while dz018 reaches `-0.001473`. The external split deltas are: Kodak dz014 `-0.001115`, dz018 `-0.001033`; CLIC mobile dz014 `-0.001493`, dz018 `-0.001295`; CLIC professional dz014 `-0.002196`, dz018 `-0.001995`.

The safety trade-off is also clear. Dz014 has the better mean RD, but dz018 has higher win rate and lower q95 damage: all-external win rate is `0.671958` for dz014 versus `0.724868` for dz018, and all-external q95 damage is `+0.000930` for dz014 versus `+0.000300` for dz018. Per-seed inspection shows the remaining weak spot: dz014 has tiny positive seed3456 deltas on Kodak (`+0.000197`), CLIC mobile (`+0.000085`), and CLIC professional (`+0.000039`), while dz018 makes seed3456 negative on CLIC but remains very slightly positive on Kodak (`+0.000029`).

Intermediate features are stable and do not indicate collapse. Across external splits, qMSE stays around `0.065` on Kodak/CLIC mobile and `0.082` on CLIC professional, `s_q_mean` stays around `0.474`-`0.482`, dead-code remains modest, and every run has zero nonfinite rows. This supports the interpretation that dz014/dz018 generalize as a controlled policy difference rather than a numerical artifact.

Decision: E117 promotes the dead-zone branch beyond OpenImages. The current paper table order should be dz014 as the stronger external-confirmed mean-RD branch, dz018 as the conservative safety ablation, beta005 as the broad historical prototype baseline, and max500 as high-mean headroom. The next safe research action is to freeze this prototype table and then start stronger-backbone/SOTA plug-in preparation without blurring the attribution of the HCG geometry gain.

Artifacts:

- `tools/run_e117_deadzone_external_fixed_protocol.py`
- `experiments/analysis/e117_deadzone_external_fixed_protocol_audit.{json,md,threshold.csv,split_threshold.csv,per_seed.csv,pairwise.csv,quartiles.csv}`
- `experiments/analysis/e117_deadzone{014,018}_from_beta005_seed{1234,2345,3456}_step250_fullimage_{kodak,clic_mobile_valid,clic_professional_valid}_current.{csv,json,md}`

## E118: Prototype Main Table Package

Status: Done. I froze the current prototype manuscript table after E117 external confirmation. This is an analysis-only packaging step that combines beta005 broad evidence, E108 transfer threshold selection, E109 holdout confirmation, and E117 external fixed-protocol confirmation.

The table now supports a cleaner paper order. Across five reporting splits, both dz014 and dz018 improve over beta005 in every split. Dz014 has the stronger mean result: mean split delta versus beta005 is `-0.002092`, worst split delta is still negative at `-0.001115`, and `nonfinite_rows=0`. Dz018 is slightly weaker on mean (`-0.001893`) but safer in tail behavior: mean win rate `0.726444` and worst q95 `+0.000641`, compared with dz014 mean win rate `0.674802` and worst q95 `+0.001539`.

The prototype table should therefore be fixed as follows: dz014 is the main mean-RD row for the current MeanScaleHyperprior/RVQ prototype, dz018 is the conservative reliability/tail ablation, beta005 is the earlier broad guard baseline and initialization context, and max500 remains out of the main row until checkpoint policy or conditional reliability control is fixed.

Decision: the HCG-RVQ prototype claim is now much healthier for a conference paper than before E117/E118. It has protocol-clean threshold selection, holdout confirmation, external split confirmation, stable intermediate features, and a compact table story. The next priority can move from proving this prototype branch is real to preparing stronger-backbone/SOTA plug-in comparisons while preserving attribution to HCG geometry.

Artifacts:

- `tools/build_e118_prototype_main_table_package.py`
- `experiments/analysis/e118_hcg_rvq_prototype_main_table_package.{json,md,table.csv,threshold_summary.csv}`

## E119: SOTA/Backbone Plug-In Readiness Audit

Status: Done. I built a readiness audit that separates the frozen prototype claim from the next strong-backbone/SOTA integration work. This was an analysis-only step and did not require GPU evaluation.

The audit keeps the conference status precise: HCG-RVQ now has a promising prototype claim, but it is not yet a final SOTA claim. The main result remains E118: dz014 and dz018 improve over beta005 on all five reporting splits, with dz014 mean split delta `-0.002092` and dz018 `-0.001893` versus beta005.

The next method action is to extract or audit an HCG quantizer adapter boundary before plugging into stronger backbones. The current portable contract is `y`, `hyper_features`, and `image_hw` as inputs, producing `y_hat`, RVQ indices, commitment loss, RVQ/intermediate stats, and conditioning tensors. This boundary is currently embedded in `HCGMeanScaleHyperprior._conditioned_rvq`, so a direct SOTA plug-in now would risk tangling the novelty with backbone-specific implementation details.

The backbone priority is also fixed. Start with local CompressAI-compatible stronger backbones (`JointAutoregressiveHierarchicalPriors` / `mbt2018_mean`, then `Cheng2020Attention`) because CompressAI is already installed and shares the hyperprior latent contract. Official external repos such as DCAE, MambaIC, and HPCM should come after adapter proof-of-portability, not before.

Decision: do not clone or integrate a large external SOTA repo yet. First do E120 adapter extraction/smoke, and in parallel build E121 explicit component ablation table. This preserves attribution: if the later strong-backbone row improves, the paper can still say the gain comes from HCG quantizer geometry rather than an uncontrolled architecture swap.

Artifacts:

- `tools/build_e119_sota_plugin_readiness_audit.py`
- `experiments/analysis/e119_sota_plugin_readiness_audit.{json,md,readiness.csv,adapter.csv,backbones.csv,next_experiments.csv}`

## E121: Component Ablation Table

Status: Done. I converted existing fixed-protocol evidence into a paper-readable component ablation table. This was analysis-only and did not require GPU evaluation.

The component table supports the current method story. Across the five reporting splits, HCS-RVQ is the shift/scale + index-entropy baseline. Raw Householder geometry (`old gate0.25`) improves HCS by mean split delta `-0.023766`; conservative risk geometry (`min090`) improves by `-0.028944`; beta005 guard improves by `-0.099495`; and the final dead-zone controller improves further. Dz014 is best on mean with delta `-0.101587` versus HCS, while dz018 is almost tied at `-0.101388` and remains the safety ablation.

The table also exposes missing rows honestly. A pure entropy-only/HVQ-like final fixed-protocol row is still missing because HCS includes both shift/scale and index entropy. Multi-rate repetition is also missing because the final dead-zone evidence is still at lambda `0.0035`. Strong-backbone plug-in is also missing, as already noted in E119.

Decision: E121 is usable for the prototype manuscript ablation section, as long as the missing entropy-only and multi-rate limitations are stated. It strengthens the core claim that the final gain is not just adding raw Householder geometry; the reliability-controlled residual-selector geometry is what upgrades beta005 into dz014/dz018.

Artifacts:

- `tools/build_e121_component_ablation_table.py`
- `experiments/analysis/e121_component_ablation_table.{json,md,summary.csv,splits.csv,missing.csv}`

## E120: HCG Adapter Contract Smoke

Status: Done. I added a non-invasive contract smoke for the eventual HCG quantizer adapter boundary. It compares the full forward path with a manual path through `g_a`, `h_a`, entropy bottleneck, `h_s`, `_conditioned_rvq`, index entropy, and `g_s` on one Kodak image using the dz014 seed1234 step250 checkpoint. The run fixed the machine to physical GPU0 with `CUDA_VISIBLE_DEVICES=0` and used `cuda:0`.

The tolerance-based contract passed. The exact-zero check was false because repeated GPU convolutions produce tiny numerical differences, but the differences are far below the `1e-4` contract tolerance: `max_abs_x_hat_diff=4.87e-05`, `max_abs_y_hat_diff=3.20e-05`, `bpp_y_index_diff=1.49e-08`, `max_loss_diff=2.98e-08`, `max_rvq_stat_diff=3.73e-09`, and `nonfinite=0`.

Decision: the current adapter boundary is ready for non-invasive extraction. The next implementation step should move the `_conditioned_rvq` logic into an adapter module and require this E120 smoke to stay within tolerance before any strong-backbone plug-in.

Artifacts:

- `tools/run_e120_hcg_adapter_contract_smoke.py`
- `experiments/analysis/e120_hcg_adapter_contract_smoke.{json,md}`

## E122: HCG Adapter Extraction Smoke

Status: Done. I extracted the HCG quantizer boundary into `hcg_rvq/quantizers/hcg_adapter.py` and changed `HCGMeanScaleHyperprior._conditioned_rvq` into a thin delegation call. This is intentionally parameter-free so existing checkpoint state-dict keys do not change.

The post-extraction E120 smoke passed on physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`). The tolerance-based contract remains valid: `max_abs_x_hat_diff=5.17e-05`, `max_abs_y_hat_diff=2.49e-05`, `bpp_y_index_diff=0.0`, `max_loss_diff=4.77e-07`, `max_rvq_stat_diff=2.98e-08`, and `nonfinite=0`.

Checkpoint compatibility is preserved. Loading the dz014 seed1234 step250 checkpoint into the extracted-code model produced `missing=[]`, `unexpected=[]`, no adapter state keys, and `114/114` model/checkpoint keys.

Decision: the HCG-RVQ quantizer strategy is now in a portable adapter boundary without disturbing the current prototype evidence. The next method-strengthening action is a local strong-backbone smoke, starting with a CompressAI-compatible hyperprior/JointAutoregressive candidate, while keeping E118/E121 as the paper-facing prototype table and ablation base.

Artifacts:

- `hcg_rvq/quantizers/hcg_adapter.py`
- `experiments/analysis/e122_hcg_adapter_extraction_smoke.{json,md}`

## E123: Local CompressAI Backbone Contract Audit

Status: Done. I audited the locally installed CompressAI backbones as the next low-risk strong-backbone path after E122. This was CPU-only and did not use pretrained downloads or GPU evaluation.

Both candidate backbones expose the required transform boundary. mbt2018_mean produces y=[1,192,4,4], h_s(z_hat)=[1,384,4,4], and g_s(y)=[1,3,64,64]. cheng2020_attn produces y=[1,128,4,4], h_s(z_hat)=[1,256,4,4], and g_s(y)=[1,3,64,64], with context and entropy-parameter modules present.

The key integration detail is that CompressAI h_s returns Gaussian-parameter channels, 2*M, rather than the current prototype fixed N-channel hyper feature contract. Therefore the next adapter implementation should accept explicit latent_channels and hyper_channels, instead of assuming hyper_channels=N.

Decision: the next strong-backbone action should be a small HCG adapter module with configurable hyper_channels, then a no-training smoke on mbt2018_mean first and cheng2020_attn second. This keeps the SOTA plug-in path aligned with the prompt while avoiding a large external repo integration too early.

Artifacts:

- tools/build_e123_local_compressai_backbone_contract_audit.py
- experiments/analysis/e123_local_compressai_backbone_contract_audit.{json,md,csv}

## E124: Local CompressAI HCG Adapter Smoke

Status: Done. I added a standalone `HCGQuantizerAdapter` with explicit `latent_channels` and `hyper_channels`, then smoked it inside local CompressAI backbone contracts. This is the first executable plug-in path after E122/E123. The run used physical GPU0 only with `CUDA_VISIBLE_DEVICES=0` and `cuda:0`.

Both local backbones passed the shape and numerics smoke. `mbt2018_mean` used `y=[1,192,4,4]`, `h_s=[1,384,4,4]`, `y_hat=[1,192,4,4]`, `x_hat=[1,3,64,64]`, one RVQ index tensor `[1,3,4,4]`, and `nonfinite=0`. `cheng2020_attn` used `y=[1,128,4,4]`, `h_s=[1,256,4,4]`, `y_hat=[1,128,4,4]`, `x_hat=[1,3,64,64]`, one RVQ index tensor `[1,2,4,4]`, and `nonfinite=0`.

The intermediate features are in the expected identity-initialized regime. `mu_q_abs_mean=0`, `s_q_mean≈1.000001`, and Householder delta is exactly zero at initialization for both backbones. RVQ stats are finite, with high dead-code ratios as expected from a random one-image, untrained adapter smoke.

Decision: the SOTA/backbone path is now past the pure planning stage. The next research action should be a tiny trainable local-backbone pilot, starting with `mbt2018_mean` and a frozen backbone or head-only adapter warmup, while keeping the E118/E121 prototype table as the current paper claim. No quality claim should be made from E124 alone.

Artifacts:

- `hcg_rvq/quantizers/hcg_adapter.py`
- `tools/run_e124_local_compressai_hcg_adapter_smoke.py`
- `experiments/analysis/e124_local_compressai_hcg_adapter_smoke.{json,md}`

## E125: mbt2018 Adapter Trainability and Safe Geometry Activation

Status: Done. I turned the local CompressAI adapter pilot into a variant/tagged trainability harness, added gradient-nonfinite detection, and ran the key branches on physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`). This is a trainability/checkpoint/intermediate-feature audit on a random local `mbt2018_mean` backbone, not a paper-quality RD result.

The old direct `hcg_rvq_h` failure was diagnosed more precisely. The forward pass was finite at step 1, but the first training row already had `grad_norm=NaN`, then step 2 onward became nonfinite. The likely cause was loss plumbing rather than GPU/device failure: `RateDistortionLoss` was adding zero-weight conditioning terms to the graph, including `householder_delta_rms=sqrt(0)`, which can create NaN gradients even when the coefficient is zero. I fixed this by only adding `rho_mu_q_abs`, `rho_s_q_std`, and `rho_householder_delta` terms when their coefficients are nonzero.

After that loss guard, direct `hcg_rvq_h` completes all 30 steps with `nonfinite=0`, `grad_nonfinite=0`, and no skipped optimizer steps. Its final eval row is finite: RD `38.172665`, qMSE `0.000507`, dead-code ratio `0.517578`, and `s_q_mean=1.003757`. However, Householder remains inactive in this branch: `householder_delta_rms=0` and `householder_v_abs_mean=0` through step 30. So the NaN is fixed, but zero-initialized geometry does not wake itself up.

The HCS warmup branch also completes all 30 steps cleanly. It improves qMSE from `0.002228` to `0.000488`, dead-code ratio from `0.698242` to `0.505859`, and keeps `nonfinite=0` throughout. RD stays essentially flat (`38.171261` to `38.172087`), which is expected for this random-backbone smoke.

Finally, I added an optional safe geometry activation path for the standalone adapter: nonzero Householder bias initialization plus a small Householder gate. With `householder_bias_init_scale=0.01`, `householder_gate_init=0.01`, and `householder_gate_max=0.1`, the gated HCG branch completes all 30 steps with zero nonfinite values and active geometry. `householder_delta_rms` moves from `7.69e-05` to `2.60e-04`, and `householder_v_abs_mean` moves from `0.007418` to `0.030915`. RD is worse (`40.838304`) in this random-backbone smoke, so this is not a quality claim; it is evidence that geometry can be activated safely when it is explicitly gated.

Decision: the strong-backbone lane should not treat the old NaN as a method failure. The correct next path is staged: HCS/no-transform warmup for stable adapter learning, then small-gate/nonzero-direction geometry activation, then real RD evaluation after the adapter/backbone are no longer random. This keeps the prompt's core claim alive while avoiding an unstable direct full-geometry start.

Artifacts:

- `hcg_rvq/losses.py`
- `hcg_rvq/quantizers/hcg_adapter.py`
- `tools/run_e125_mbt2018_hcg_adapter_trainability_pilot.py`
- `tools/build_e125_trainability_summary.py`
- `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_summary.{json,md,csv}`
- `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_pilot_{direct_hcg_lossguard,hcs_warmup,gated_hcg_initbias001_gate001}.{json,md}`

## E126: Staged HCS-to-Gated-HCG Adapter Pilot

Status: Done. I extended the E125 trainability harness so it can load an adapter checkpoint and then reset only the Householder direction/gate after loading. This lets us test the actual staged path proposed by E125: train a stable HCS adapter first, then turn on geometry gently from that checkpoint. The run again used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`).

The main staged run loads `e125_mbt2018_hcg_adapter_trainability_pilot_hcs_warmup/checkpoint_step_30.pth.tar`, switches to `hcg_rvq_h`, resets the Householder head bias with std `0.01`, and resets the Householder gate to `0.01` under max `0.1`. Loading was clean: `missing_keys=[]`, `unexpected_keys=[]`, checkpoint step `30`. It completed all 30 additional steps with `nonfinite=0`, `grad_nonfinite=0`, and no skipped optimizer steps.

This staged run gives the best signal so far on the local random-backbone plug-in lane. Eval RD moves from `38.172090` at loaded step0 to `38.171590` at step30 (`delta RD=-0.000500`), and qMSE moves from `0.000488` to `0.000437`. Geometry is active: `householder_delta_rms` reaches `0.000244`, and `householder_v_abs_mean` reaches `0.027682`. This is still not a paper-quality RD result because the backbone is random/frozen, but it does show that staged geometry can connect to the HCS warmup checkpoint and improve the local smoke objective slightly.

I also ran a half-amplitude staged variant with bias std `0.005`, gate init `0.005`, and gate max `0.05`. It is also fully finite, but it does not solve the usage trade-off. RD is essentially flat/slightly worse (`38.172089` to `38.172128`), qMSE still improves (`0.000488` to `0.000428`), and dead-code ratio still rises (`0.506836` to `0.564453`). The larger-gate staged run also increases dead-code (`0.506836` to `0.579102`).

Decision: the staged path is now experimentally supported, but the next bottleneck is codebook usage, not NaN stability. Simply shrinking geometry amplitude is insufficient. The next method action should add a usage-aware guard or selection rule for geometry activation, likely by monitoring per-batch/per-image perplexity or dead-code changes and suppressing geometry where it narrows the index set too much.

Artifacts:

- `tools/run_e125_mbt2018_hcg_adapter_trainability_pilot.py`
- `tools/build_e125_trainability_summary.py`
- `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias001_gate001.{json,md}`
- `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_pilot_staged_hcs30_gated_hcg_initbias0005_gate0005.{json,md}`
- `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_summary.{json,md,csv}`

## E127: Staged Geometry Per-Image Checkpoint Audit

Status: Done. I added a per-image checkpoint audit for the E126 staged-geometry branch. The first draft of this audit exposed an important protocol pitfall: because the local `mbt2018_mean` backbone is random and only the adapter checkpoint is saved, each compared model must be instantiated with the same seed. I fixed the audit to reset `torch.manual_seed(1234)` before constructing every case model, so the comparison isolates adapter/checkpoint differences rather than backbone randomness.

The corrected audit compares three checkpoints on the same 8 Kodak images: HCS warmup step30, staged gated-HCG step30 with gate `0.01`, and staged half-gate step30 with gate `0.005`. It ran on physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`) and every row had `nonfinite=0`.

The full staged gate `0.01` checkpoint is consistently better than the HCS warmup checkpoint on this local smoke split: mean RD `38.171587` versus HCS `38.172083`, mean delta RD `-0.000496`, win rate `1.0`, and q95 damage `0.0`. It also improves latent qMSE by `-0.000051` on average and keeps active geometry (`mean householder_delta_rms=0.000244`). This is a stronger signal than the aggregate-only E126 view because every image wins under the corrected path-matched comparison.

The usage caveat is equally clear. Full staged geometry raises dead-code ratio by `+0.073242` and lowers perplexity by `-6.282425`. The half-gate checkpoint improves qMSE more (`-0.000060`) but loses the RD mean (`+0.000042`) and still worsens dead-code by `+0.058594`. This confirms the E126 diagnosis: shrinking geometry amplitude alone does not solve the codebook-usage trade-off.

Selector headroom is nonzero. An oracle over the two staged checkpoints improves mean RD by `-0.000497` but selects HCG on all images and keeps the dead-code penalty. A simple usage-safe policy requiring RD win and dead-code delta `<=0.05` selects 3/8 images and still improves mean RD by `-0.000083` with mean dead delta only `+0.014648`. With a `<=0.075` cap, it selects 4/8 images and improves `-0.000175`. This suggests a paper-relevant next step: learn or design a reliability/usage controller that keeps only geometry applications with acceptable index-usage damage.

Decision: staged HCG geometry is now a real positive signal on the local strong-backbone lane, not just a stability smoke, but the result is still not a SOTA claim. The next implementation target should be a usage-aware geometry gate, using per-image/per-batch features such as predicted Householder delta, perplexity/drop proxies, or codebook-usage regularization to avoid narrowing the index set.

Artifacts:

- `tools/run_e127_staged_geometry_per_image_audit.py`
- `experiments/analysis/e127_staged_geometry_per_image_audit.{json,md}`
- `experiments/analysis/e127_staged_geometry_per_image_audit_{per_image,summary,selectors}.csv`

## E128: Usage-Aware Gate Feature Audit

Status: Done. I added an analysis-only feature audit on top of the corrected E127 per-image results. The goal was to separate three questions that should not be mixed in the paper: the posthoc upper bound, features visible after running the candidate HCG path, and features visible from the baseline path alone. This did not require GPU evaluation.

The main result is that the `staged_gate001_step30` branch remains the more promising usage-aware target. Under a strict per-selected dead-code cap of `0.05`, a candidate-forward rule using `hcg_latent_quant_mse <= 0.000349` selects 1/8 image and keeps a small gain (`delta RD=-0.000073`) with selected max dead-code increase `0.039063`. The posthoc diagnostic cap rule using `delta_dead_code_ratio <= 0.046875` selects 3/8 images and gives `delta RD=-0.000083`, which matches the E127 usage-safe selector headroom. Under cap `0.075`, baseline-only or candidate-forward latent-qMSE rules select 2/8 images and improve `-0.000165` with selected max dead-code increase `0.054688`; the posthoc upper bound selects 4/8 and improves `-0.000175`.

The half-gate branch is less attractive once strict usage safety is enforced. `staged_gate0005_step30` can improve under mean dead-code budgets, but under strict cap `0.025` or `0.05` the best selected image actually worsens RD (`+0.000049` or `+0.000055` depending on scope). It only becomes positive around cap `0.075`, where latent-qMSE rules select 2/8 images and improve `-0.000096`. This supports the earlier conclusion that simply shrinking geometry amplitude is not the right main lever.

The feature correlations give a useful mechanism hint. Candidate HCG perplexity/stage entropy correlate strongly with dead-code damage for half-gate (`Spearman about -0.88`), and candidate HCG latent qMSE is one of the best practical selectors for the full gate. Baseline-only features have some signal but are weaker and unstable on this 8-image smoke split. Therefore the next implementation should not be a manually tuned single threshold in the paper-main method. It should be a small usage-aware reliability head or deterministic candidate-forward usage guard, selected on an independent split and then confirmed on holdout.

Decision: E128 promotes `staged_gate001` plus usage-aware control as the active strong-backbone development direction. The immediate next experiment should make the guard trainable or protocol-clean rather than hand-picking thresholds from the same 8-image audit. The paper claim remains unchanged: E118/E121 are still the prototype evidence, while E127/E128 justify the next plug-in mechanism.

Artifacts:

- `tools/analyze_e128_usage_aware_gate_features.py`
- `experiments/analysis/e128_usage_aware_gate_feature_audit.{json,md}`
- `experiments/analysis/e128_usage_aware_gate_feature_audit_{features,policies,best,strict_cap,loo,correlations}.csv`

## E129: Full-Kodak Staged Geometry and Usage-Aware Audit

Status: Done. I extended the corrected E127 path-matched checkpoint audit from 8 Kodak images to the full 24-image Kodak split. The run used physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `cuda:0`) and wrote to a separate E129 prefix so the original E127 smoke artifacts remain intact.

The full-gate staged geometry branch remains positive, but the larger split makes the tail risk visible. `staged_gate001_step30` improves mean RD from HCS `43.976532` to `43.976207` (`delta RD=-0.000325`), with win rate `0.791667`, `nonfinite_sum=0`, mean latent qMSE delta `-0.0000817`, and active geometry (`householder_delta_rms=0.000296`). However, q95 RD damage is no longer zero (`0.000569`), max damage is `0.000626`, dead-code rises by `+0.068034`, and perplexity drops by `-4.143783`. This turns the E127 conclusion into a stronger and more realistic one: active geometry helps on average, but needs usage/tail control before becoming a paper-main strong-backbone result.

The half-gate branch is demoted by the full split. `staged_gate0005_step30` improves qMSE (`-0.000091`) but worsens mean RD by `+0.000252`, has win rate only `0.458333`, and q95 damage `0.001149`. This confirms that reducing geometry amplitude is not sufficient and can actually hurt RD.

Selector headroom persists at 24 images. The oracle over staged checkpoints selects HCG on 19/24 images and improves `-0.000395` RD. A usage-safe oracle requiring RD win and dead-code delta `<=0.05` selects 6/24 images and still improves `-0.000098` with mean dead-code increase only `+0.010417`; cap `<=0.075` selects 13/24 and improves `-0.000247`; cap `<=0.10` selects 16/24 and improves `-0.000298`.

I also reran the E128 feature audit on the 24-image per-image table. For `staged_gate001`, mean-budget candidate-forward rules remain strong: under mean dead budget `0.05`, `hcg_latent_quant_mse <= 0.000430` selects 16/24 images and improves `-0.000342` with q95 damage `0.0`, though selected max dead-code increase reaches `0.132812`. Leave-one-image checking gives similar direction: candidate-forward under budget `0.05` selects 16/24 and improves `-0.000302`, while posthoc diagnostic improves `-0.000344`. Under strict per-selected dead-code caps, simple features are much more conservative: at strict cap `0.10`, candidate-forward latent-qMSE selects 8/24 and improves `-0.000178` with selected max dead `0.09375`; at strict cap `0.05`, candidate-forward selects only 1/24 with negligible gain.

Decision: E129 confirms the active strong-backbone development direction should be `staged_gate001` plus a usage-aware controller, not weaker geometry. It also clarifies the controller requirement: mean-budget reliability is already promising, but a strict per-image safety claim needs either a learned controller, a better candidate-side usage proxy, or explicit signaling/protocol design. The paper-main evidence remains E118/E121; E129 is a strong roadmap and risk-control result for the plug-in lane.

Artifacts:

- `tools/run_e127_staged_geometry_per_image_audit.py` (`--out-prefix` added)
- `experiments/analysis/e129_staged_geometry_kodak24_audit.{json,md}`
- `experiments/analysis/e129_staged_geometry_kodak24_audit_{per_image,summary,selectors}.csv`
- `experiments/analysis/e129_usage_aware_gate_feature_audit.{json,md}`
- `experiments/analysis/e129_usage_aware_gate_feature_audit_{features,policies,best,strict_cap,loo,correlations}.csv`

## E130: Usage Controller Split-Protocol Audit

Status: Done. I added a split-protocol audit for the E129 Kodak24 per-image results. Instead of choosing thresholds on the same images being reported, the script selects a single-feature policy on one split and evaluates it on a disjoint split: first half to second half, second half to first half, even to odd, and odd to even. This is still a small diagnostic, but it is much closer to the protocol discipline needed for a paper-safe usage controller.

The split result strongly favors the full `staged_gate001` branch over the half-gate branch. For `staged_gate001`, candidate-forward mean-budget controllers win on all four split protocols. At mean dead budget `0.05`, the candidate-forward policy selects `8.25/12` held-out images on average and improves RD by `-0.000318`, with mean dead-code delta `+0.049967` and mean q95 damage `0.000113`. At budgets `0.075` and `0.10`, candidate-forward improves `-0.000370`, also winning `4/4` protocols. Baseline-only features are also surprisingly useful (`-0.000336` at budget `0.05`), but candidate-forward is the better method direction because it aligns with the quantizer's actual usage signal.

The strict per-selected dead-code cap view is harder, as expected, but still positive for the full gate. For `staged_gate001`, candidate-forward strict cap `0.075` wins all four protocols, selects `2.5/12` images on average, and improves `-0.000096` with mean dead-code delta `+0.017253`. At strict cap `0.10`, it again wins all four protocols, selects `4.25/12`, and improves `-0.000116` with mean dead-code delta `+0.025553`. Caps `0.025` and `0.05` are too conservative for a useful simple controller.

The half-gate branch is effectively ruled out as the main route. `staged_gate0005` candidate-forward mean-budget policies lose on all four protocols at every budget and have positive mean RD deltas. Strict-cap policies are either no-op or tiny/unstable. This confirms E129's interpretation: smaller geometry amplitude is not a substitute for usage-aware full-gate control.

Decision: E130 upgrades the next action from “maybe usage-aware control” to a concrete method target: implement or train a candidate-forward usage controller for `staged_gate001`, then select it on an independent split and confirm on holdout. The paper-main prototype evidence still remains E118/E121, but the strong-backbone lane now has a protocol-backed controller direction rather than only an oracle headroom story.

Artifacts:

- `tools/analyze_e130_usage_controller_split_protocol.py`
- `experiments/analysis/e130_usage_controller_split_protocol.{json,md}`
- `experiments/analysis/e130_usage_controller_split_protocol_{train,test,summary}.csv`

## E131: Usage Controller Decision Package

Status: Done. I consolidated the E129 checkpoint/oracle results and the E130 split-protocol controller results into a decision package. This is analysis-only; it does not add new GPU evaluation. The goal was to turn the recent per-image, checkpoint, and feature-distribution evidence into a concrete next implementation target rather than keeping several plausible controller variants alive.

The recommendation is now precise. The next implementation target should be the full `staged_gate001` branch with a candidate-forward expected-usage controller at mean dead-code budget `0.05`. Across the four E130 split protocols, this policy wins `4/4`, selects `8.25/12` held-out images on average, improves RD by `-0.000318`, keeps mean dead-code delta at `+0.049967`, and has mean q95 RD damage `0.000113`. This is the best balance between mean RD gain and usage cost.

Two ablations should be carried with it. A looser mean-budget `0.075` variant improves more (`-0.000370`, `4/4` wins, `9.75/12` selected) and is useful as a mean-RD ablation. A strict selected-dead cap `0.075` variant is the conservative safety ablation: it also wins `4/4`, selects `2.5/12`, and improves `-0.000096` with mean dead-code delta `+0.017253`.

The feature-stability table says not to overclaim a single magic scalar. Split-selected thresholds alternate among `hcg_dead_code_ratio`, `hcg_householder_delta_rms`, and `hcg_latent_quant_mse`. Therefore the next code target should be either a small candidate-forward reliability head or a small deterministic multi-feature guard, selected on an independent split and confirmed on holdout.

Artifacts:

- `tools/build_e131_usage_controller_decision_package.py`
- `experiments/analysis/e131_usage_controller_decision_package.{json,md}`
- `experiments/analysis/e131_usage_controller_decision_package_{checkpoint_context,controller_options,feature_stability,recommendations}.csv`

## E132: Usage Controller Teacher Labels

Status: Done. I converted the E129 full-Kodak staged-geometry audit into teacher labels for a future usage-aware controller. This is analysis-only and uses the already measured `staged_gate001_step30` per-image table.

The label distribution is healthy enough for a small controller probe. Broad `rd_win` is positive on `19/24` images, with positive mean delta RD `-0.000499`. Safety-labeled positives are smaller but usable: `safe_win_dead_le_0.050` has `6/24` positives, `safe_win_dead_le_0.075` has `12/24`, and `safe_win_dead_le_0.100` has `15/24`. The `0.075` label is the best near-term compromise because it has enough positives while still separating lower dead-code damage (`+0.052734` for positives vs `+0.083333` for negatives).

Feature separation matches the E130 direction. For broad `rd_win`, the strongest candidate-side signal is `hcg_householder_delta_rms` (effect `1.072326`), followed by `hcg_latent_quant_mse` (effect `0.802289`). For safety labels, `hcg_latent_quant_mse`, `hcg_stage_entropy`, `hcg_perplexity`, and `hcg_dead_code_ratio` are useful. Baseline-only features have signal too, but candidate-forward features are the right controller inputs because they observe the actual HCG geometry path.

Artifacts:

- `tools/build_e132_usage_controller_teacher_labels.py`
- `experiments/analysis/e132_usage_controller_teacher_labels.{json,md}`
- `experiments/analysis/e132_usage_controller_teacher_labels_{labels,summary,feature_separation}.csv`

## E133: Supervised Usage Controller Probe

Status: Done. I added a tiny split-protocol logistic probe on top of the E132 labels. Thresholds are selected only on the train split using the same RD/dead-code objectives, then evaluated on the held-out split. This is a feasibility check for a future learned reliability head, not a final paper result.

The learned signal is real but not yet safer than the E131 deterministic guard. The strongest mean-budget probe uses the compact candidate feature set (`hcg_latent_quant_mse`, `hcg_householder_delta_rms`, `hcg_dead_code_ratio`) with the `rd_win` label. At budget `0.075`, it wins `4/4`, selects `9.0/12`, improves RD by `-0.000342`, and has mean dead-code delta `+0.053711`. At budget `0.05`, it wins `4/4` and improves `-0.000324`, but its held-out mean dead-code delta is `+0.051432`, slightly above the nominal target. A more conservative combined-feature `safe_win_dead_le_0.075` probe at budget `0.05` stays under budget (`+0.047363`) while improving `-0.000311`.

The strict-cap probes show why this should remain a probe for now. They improve RD more than the very conservative E131 strict-cap one-feature guard, but train-selected strict caps do not guarantee held-out max selected dead-code damage: for example, the combined `safe_win_dead_le_0.075` strict-cap `0.075` probe improves `-0.000183` but has held-out mean max selected dead `0.085938`. This is useful evidence that a learned head can increase coverage, but not yet a paper-safe strict guarantee.

Decision: E133 validates the direction of a trainable reliability controller, but the next implementation default remains the E131 candidate-forward deterministic guard at mean budget `0.05`. The learned head should be developed in parallel as the higher-upside route, using E132 labels and an independent teacher split before any strong paper claim.

Artifacts:

- `tools/analyze_e133_usage_controller_supervised_probe.py`
- `experiments/analysis/e133_usage_controller_supervised_probe.{json,md}`
- `experiments/analysis/e133_usage_controller_supervised_probe_{detail,summary,weights}.csv`

## E134: Usage Guard Cross-Fit Decision Package

Status: Done. I materialized the E130 split-selected candidate-forward policies into per-image decisions. This is analysis-only and does not add new GPU evaluation. The goal was to move from a summary-table statement to a reusable artifact that says exactly which held-out images are selected by the recommended usage guard, what features triggered the decision, and how RD/usage/tail behave.

The main E131 guard is exactly reproduced. The `next_implementation_target` candidate-forward mean dead-code budget `0.05` policy wins all four split protocols, selects `8.25/12` held-out images on average, improves RD by `-0.000318`, keeps mean dead-code delta at `+0.049967`, and has mean q95 RD damage `0.000113`. This confirms the E130/E131 result was not a reporting artifact.

The new per-image materialization clarifies the safety boundary. The main guard's selected-image win rate is `0.846528`, but selected max dead-code delta averages `0.128906`, and only `0.542361` of selected decisions satisfy the `safe_win_dead_le_0.075` teacher label. Therefore this default guard is an expected-usage controller, not a hard per-image safety controller. The strict-cap probes remain positive but also should be reported as held-out probes rather than guaranteed hard caps: strict cap `0.075` improves `-0.000096`, but its held-out mean selected max dead is `0.115234` because the cap is selected on the train split.

Vote stability is useful but imperfect. For the main guard, 13/24 images are selected by both held-out policies, 7/24 by one policy, and 4/24 by neither. This supports the current decision to avoid claiming a universal one-feature threshold. The guard is reproducible and useful, but a learned or multi-feature reliability controller remains the higher-upside path.

The intermediate-feature contrast is now explicit. Main-guard selected decisions have lower candidate latent qMSE (`0.000366` vs `0.000574`) and lower Householder delta RMS (`0.000265` vs `0.000365`) than rejected decisions, but they also have higher HCG dead-code ratio (`0.607955` vs `0.547917`) and lower HCG perplexity (`14.048196` vs `15.755943`). This is the core trade-off: the guard finds stable geometry applications, but codebook-usage control still needs improvement before a strong safety claim.

Decision: keep E131/E134 deterministic guard as the immediate implementation baseline, and keep E133 learned reliability as the parallel stronger route. The next implementation step should freeze a decoder-reproducible guard or train a small reliability head on independent labels, then evaluate it with the same checkpoint, RD, qMSE, dead-code, perplexity, vote stability, and nonfinite audits.

Artifacts:

- `tools/build_e134_usage_guard_crossfit_package.py`
- `experiments/analysis/e134_usage_guard_crossfit_package.{json,md}`
- `experiments/analysis/e134_usage_guard_crossfit_package_{detail,summary,votes,feature_groups}.csv`


## E135: Decoder-Reproducible Guard Feature-Tier Audit

Status: Done. I split the E130/E134 candidate-forward usage guard into deployability tiers and re-ran the same four split protocols on the E129 full Kodak24 per-image table. This is analysis-only and does not use GPU. The goal was to prevent a paper-design mistake: E130's strongest guard uses candidate-forward statistics, but not all of those statistics are necessarily known to the decoder before selecting the geometry path.

The no-side-bit candidate tier is useful but smaller. The `hyper_preindex` tier (`hcg_s_q_mean`, `hcg_s_q_std`, `hcg_mu_q_abs_mean`, `hcg_householder_v_abs_mean`) wins `4/4` protocols under mean dead budget `0.05`, selects `3.00/12` held-out images, improves RD by `-0.000116`, and keeps mean dead-code delta at `+0.016276` with q95 damage `0.000041`. This is the safest decoder-reproducible deterministic guard evidence, but it recovers only about one third of the E131 main guard gain.

The full E130-strength signal mostly comes from non-preindex candidate information. The E130-like `all_candidate_forward` tier at budget `0.05` reproduces the main guard: `4/4` wins, `8.25/12` selected, delta RD `-0.000318`, mean dead `+0.049967`, q95 damage `0.000113`. The `encoder_candidate_error` tier is nearly as strong (`-0.000304`, mean dead `+0.046224`, q95 `0.000042`) but requires encoder-side error features or a proxy. The `candidate_index_usage` tier is also strong (`-0.000322`) but its held-out mean dead is `+0.055176`, above the nominal `0.05` target, and it requires candidate indices or an explicit signal.

The strict-cap view confirms the same boundary. `hyper_preindex` is no-op for strict caps up to `0.075` and only becomes useful at cap `0.10` (`-0.000116`). In contrast, `all_candidate_forward` and `encoder_candidate_error` are positive at strict cap `0.075` (about `-0.000096` to `-0.000097`) but depend on non-preindex features. So the paper-safe implementation fork is now clear: either report a small no-side-bit hyper-preindex guard, or add explicit signaling/proxy distillation for the stronger controller.

Decision: keep E131/E134 as the expected-usage controller target, but do not call it decoder-preindex without an implementation mechanism. The next implementation step should be either (1) a deterministic `hyper_preindex` guard as the conservative no-side-bit baseline, or (2) a learned reliability head/proxy trained from E132/E135 teacher features and evaluated with the same split protocol.

Artifacts:

- `tools/analyze_e135_decoder_reproducible_guard_audit.py`
- `experiments/analysis/e135_decoder_reproducible_guard_audit.{json,md}`
- `experiments/analysis/e135_decoder_reproducible_guard_audit_{feature_tiers,summary,test,train}.csv`


## E136: Decoder Proxy Supervised Probe

Status: Done. I ran a split-protocol supervised proxy audit on the E132 teacher labels to test whether a tiny learned controller using only `hyper_preindex` features can beat the E135 deterministic hyper-preindex guard. This is analysis-only, uses no GPU, and keeps the same first/second-half and even/odd split discipline.

The main deployable result is conservative. Under mean dead budget `0.05`, the best `hyper_preindex` logistic probe is the `safe_win_dead_le_0.100` label: it wins `4/4` protocols, selects `3.00/12` images, improves RD by `-0.000116`, keeps mean dead delta at `+0.016276`, and has q95 damage `0.000041`. This exactly matches the useful E135 hyper-preindex threshold level rather than improving on it. Other hyper labels are weaker at budget `0.05`: `rd_win` and `safe_win_dead_le_0.075` win only `2/4` protocols with `-0.000045`, and `safe_win_dead_le_0.050` wins `3/4` with `-0.000101`.

The apparent `hyper_preindex` gain at budget `0.075` is not a reliability controller. All labels select all 12 held-out images on every split, giving the same global staged-gate result: delta RD `-0.000325`, mean dead `+0.068034`, and q95 damage `0.000360`. This is useful as a global-use reference, but it does not solve the guard problem.

Reference probes confirm that richer non-preindex signals are still where the higher upside is. `baseline_diagnostic` and `hyper_plus_baseline_diagnostic` can reach roughly `-0.00031` to `-0.00034` at budget `0.075`, and candidate-reference probes remain around `-0.00030` to `-0.00033`, but these are not pure decoder-preindex claims. They should be treated as teacher/proxy-design evidence rather than the final deployable controller.

Decision: do not promote the current `hyper_preindex` learned proxy over the deterministic hyper-preindex guard. The next implementation path should either add explicit signaling/two-pass/proxy distillation for the stronger candidate-forward signal, or design richer decoder-known local summaries beyond the current scalar hyper summaries. The safe no-side-bit baseline remains E135/E136 `hyper_preindex` at budget `0.05` with `-0.000116` RD.

Artifacts:

- `tools/analyze_e136_decoder_proxy_supervised_probe.py`
- `experiments/analysis/e136_decoder_proxy_supervised_probe.{json,md}`
- `experiments/analysis/e136_decoder_proxy_supervised_probe_{detail,summary,weights}.csv`

## E137: Prompt-Aligned Next Action Package

Status: Done. I added a prompt-aligned synthesis package after the literature/code refresh and the E135/E136 deployability analyses. This is analysis-only and uses no GPU. The goal was to reconnect the current evidence to `docs/prompt.txt`: hyperprior-conditioned local quantizer geometry, checkpoint-selected evaluation, intermediate-feature analysis, deployable reliability control, and the later SOTA/backbone bridge.

The decision is clear: the project is `on_track_but_not_submission_complete`. The main claim should remain hyperprior-conditioned local quantizer geometry, not merely index entropy or a SOTA-backbone story. The current best prototype row is `deadzone014`, with `deadzone018` as the lower-tail safety row. On the five reporting splits, `deadzone014` has mean delta vs HCS `-0.101587`, worst split delta vs HCS `-0.086052`, and nonfinite rows `0`; `deadzone018` is almost tied on mean (`-0.101388`) and is the safer q95 ablation.

The newest controller conclusion is also sharper. Decoder-known reliability control is plausible, but current `hyper_preindex` signals are weaker than candidate-forward diagnostic guards. At budget `0.05`, the deployable hyper-preindex guard/proxy gives delta RD `-0.000116` with mean dead-code delta `+0.016276` and q95 damage `+0.000041`. The candidate-forward diagnostic controller gives `-0.000318` at the same nominal budget, but it is not directly decoder-preindex without explicit signaling, two-pass use, or proxy distillation.

Decision: prioritize paper-claim gaps before broad SOTA claims. The next actions are (1) add an explicit entropy-only / HVQ-like ablation row, (2) repeat dz014/dz018 at two additional lambda/rate points, (3) if controller work continues, evaluate a no-side-bit hyper-preindex proxy as a checkpointed pilot with qMSE/dead-code/nonfinite checks, and (4) only then move to local CompressAI strong-backbone adapter smoke and external SOTA plug-in work.

Artifacts:

- `tools/build_e137_prompt_aligned_next_action_package.py`
- `experiments/analysis/e137_prompt_aligned_next_action_package.{json,md}`
- `experiments/analysis/e137_prompt_aligned_next_action_package.{prompt_status,method_evidence,controllers,next_actions}.csv`

## E138: Entropy-Only / HVQ-like Ablation

Status: Done. I added the explicit entropy-only / HVQ-like ablation requested by `docs/prompt.txt`. This keeps the Global RVQ quantizer geometry fixed (`global_rvq`, global normalization, no local shift/scale/Householder geometry) and enables only the hyperprior-conditioned index entropy path. I trained three seeds with `CUDA_VISIBLE_DEVICES=0` and evaluated Kodak24 checkpoint sweeps plus intermediate feature distributions.

The result is useful and paper-important, but it should not become the main HCG claim. With checkpoint selection by minimum RD per method and seed, entropy-only reaches mean RD `2.195836` versus HCS `2.206217`, so the mean delta is `-0.010381` and it wins `2/3` seeds. It strongly improves the single-seed Global RVQ reference on seed1234 (`-0.070766` RD, `bpp_y` `0.136719 -> 0.079620`). This validates that hyperprior-conditioned VQ index modeling is a real control, not a dummy ablation.

The caveat is equally important. HCS still has higher Kodak MS-SSIM on all three seeds (`0.749433` mean vs entropy-only `0.726527`), and seed1234 entropy-only is worse than HCS by `+0.016291` RD. Therefore this row should be reported as a strong entropy-only control for the HCG geometry claim, not as the final method. The next paper-facing step is to run the same entropy-only row on the OpenImages holdout/start8192 protocols used by dz014/dz018, then compare it against HCG geometry under the same split.

Intermediate-feature readout is stable across seeds: empirical index bpp is about `0.108477`, RVQ dead-code ratio is `0.172201`, latent qMSE is `0.378896`, and `y_error_rms` is `0.105458`. No NaN/nonfinite issue was observed in the GPU0 training/evaluation logs.

Artifacts:

- `configs/pilot_entropy_only_index_global_rvq_frozen_seed{1234,2345,3456}.yaml`
- `tools/build_e138_entropy_only_ablation_package.py`
- `experiments/analysis/e138_entropy_only_seed{1234,2345,3456}_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e138_entropy_only_seed{1234,2345,3456}_kodak_step500_feature_distribution.{json,csv}`
- `experiments/analysis/e138_entropy_only_ablation_package.{json,md,summary.csv,per_seed.csv,features.csv,checkpoint_choices.csv}`

## E139: Entropy-Only Holdout4096 Ablation

Status: Done. I extended the E138 entropy-only / HVQ-like ablation to the OpenImages holdout4096 protocol used by the current HCG geometry rows. All evaluations were path-aligned, used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`, and produced zero nonfinite rows. This makes the ablation paper-usable because it tests whether hyperprior-conditioned index entropy alone can explain the HCG-RVQ gains on the same split.

The holdout result is stronger for the paper claim than Kodak alone. Entropy-only is not sufficient on holdout4096: mean RD is `2.200237` versus HCS `2.173688`, i.e. `+0.026550` worse. In contrast, `deadzone014` reaches `2.170857` (`-0.002830` vs HCS, `-0.029380` vs entropy-only) and `deadzone018` reaches `2.171120` (`-0.002568` vs HCS, `-0.029118` vs entropy-only). Therefore the safe holdout claim is not merely "index entropy helps"; it is that index-entropy-only conditioning does not explain the result, and local HCG geometry adds beyond it under the same split.

The quartile analysis is important for interpretation. Entropy-only helps the hardest HCS quartile (`Q4`: `-0.179808` vs HCS) but badly hurts easier images (`Q1/Q2`: `+0.129828` and `+0.115475`). HCG geometry is much flatter across difficulty (`dz014`: `-0.001540`, `-0.002434`, `-0.003159`, `-0.004188` from Q1 to Q4). This suggests entropy-only is an unstable difficulty-dependent control, while local geometry is a more consistent holdout improvement.

Intermediate features reinforce the same story. Entropy-only has much higher latent qMSE (`0.459843`) and dead-code ratio (`0.111233`) with lower perplexity (`43.671937`), while HCG geometry keeps latent qMSE around `0.106822`, dead-code around `0.03185`, and perplexity around `68.29`. This is a useful component-ablation result for the manuscript: the geometry path is not just changing rates; it also preserves quantizer usage and latent reconstruction much better.

Artifacts:

- `tools/build_e139_entropy_only_holdout_package.py`
- `experiments/analysis/e139_entropy_only_seed{1234,2345,3456}_step500_fullimage_holdout4096_current.{csv,json,md}`
- `experiments/analysis/e139_entropy_only_holdout4096_package.{json,md}`
- `experiments/analysis/e139_entropy_only_holdout4096_package.{summary,quartiles,per_image,json_checks,headline}.csv`

## E140: Multi-Rate Lambda0018 Seed1234 Scaffold

Status: Done for the first low-rate seed. I started the prompt-required multi-rate lane by training a rate-specific scalar baseline at `lambda_rd=0.0018`, then initializing and training matched HCS/HCG checkpoints from that scalar model. All GPU runs used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; numeric finite checks in the generated package pass.

The low-rate scaffold exposed an important implementation issue. The first HCG gate0.25 run used the normal scalar-loaded initialization, but its Householder geometry stayed inactive: `householder_delta_rms=0.0` and `householder_v_abs_mean=0.0`. Its RD was therefore essentially HCS-like (`1.196093` vs HCS `1.196004`, delta `+0.000088`). This is a useful audit result because it prevents us from accidentally reporting an HCG rate point that is really just HCS.

I added a safe nonzero Householder-bias initialization path to `HCGMeanScaleHyperprior` and `tools/init_rvq_codebook.py`, then reran a `bias010` low-rate HCG pilot. This activates geometry and gives a promising single-seed result: scalar RD `1.712800`, HCS RD `1.196004`, and HCG bias010 step250 RD `1.174884`. That is `-0.021121` vs HCS and `-0.537916` vs scalar. The best checkpoint is step250; step500 drifts back by `+0.006291` RD, so checkpoint selection remains essential.

Intermediate features show why this is promising but not yet paper-final. The active bias010 checkpoint has lower qMSE (`0.008983` vs HCS `0.014354`) and nonzero geometry (`householder_delta_rms=0.028753`, `householder_v_abs_mean=0.017468`), but dead-code rises (`0.109049` vs HCS `0.072591`) and perplexity drops (`47.020414` vs HCS `47.736692`). This strengthens the method-development direction, while also confirming that usage control/checkpoint selection must remain part of the analysis.

Decision: continue the multi-rate lane, but do not claim a rate curve from this one seed. The immediate next steps are to repeat the `bias010` low-rate check on seeds 2345/3456, add holdout4096 evaluation for the best low-rate checkpoint, and then mirror the same protocol at `lambda_rd=0.0067`.

Artifacts:

- `configs/pilot_scalar_baseline_lambda0018_seed1234.yaml`
- `configs/pilot_hcs_rvq_frozen_lambda0018_seed1234.yaml`
- `configs/pilot_hcg_rvq_h_gate025_frozen_lambda0018_seed1234.yaml`
- `configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed1234.yaml`
- `tools/build_e140_multirate_lambda0018_seed1234_package.py`
- `experiments/analysis/e140_multirate_lambda0018_seed1234_package.{json,md,summary.csv,checks.csv}`

## E140 Update: Multi-Rate Lambda0018 Two-Seed Active-Geometry Check

Status: Done for seeds `1234` and `2345`. I continued the low-rate lane by training and evaluating the matched seed2345 scalar, HCS, and active-HCG `bias010` checkpoints. The full run used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; no NaN or nonfinite rows appeared in the training/evaluation/feature-distribution checks.

The seed2345 result repeats the active-geometry signal. Scalar best RD is `1.734813`; HCS best is step250 with RD `1.204447`; HCG `bias010` best is also step250 with RD `1.193005`. This gives HCG `-0.011443` RD vs HCS on seed2345. Combined with seed1234 (`-0.021121` vs HCS), the two-seed low-rate mean is now: scalar `1.723806`, HCS `1.200226`, HCG `bias010` `1.183944`. HCG wins `2/2` seeds and improves mean RD by `-0.016282` vs HCS.

The checkpoint audit remains important. On seed2345, HCS step500 drifts by `+0.083574` relative to step250, and HCG step500 drifts by `+0.071768` relative to step250. This means the low-rate geometry result is not just a final-checkpoint story; checkpoint selection must remain part of the paper protocol.

Intermediate features show a promising but not yet finished method-strengthening direction. Householder geometry is active in both seeds (`householder_delta_rms`: `0.028753` on seed1234 and `0.048612` on seed2345). However, usage gets narrower: mean dead-code delta vs HCS is `+0.036133` and mean perplexity delta is `-2.240702`. Mean qMSE delta is `-0.002610`, but seed2345 is nearly neutral on qMSE (`+0.000152`) while still improving RD. Therefore the result supports active geometry at low rate, but also reinforces the need for usage control and holdout confirmation before making a final rate-curve claim.

Decision: keep this as the method-strengthening lane, not the paper-main claim yet. The next steps are to complete seed3456 under the same protocol, evaluate the best low-rate checkpoints on holdout4096, and then mirror the protocol at `lambda_rd=0.0067`.

Artifacts:

- `configs/pilot_scalar_baseline_lambda0018_seed2345.yaml`
- `configs/pilot_hcs_rvq_frozen_lambda0018_seed2345.yaml`
- `configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed2345.yaml`
- `tools/build_e140_multirate_lambda0018_two_seed_package.py`
- `experiments/analysis/e140_scalar_lambda0018_seed2345_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_hcs_lambda0018_seed2345_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_hcg_gate025_bias010_lambda0018_seed2345_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_*seed2345*kodak*feature_distribution.{json,csv}`
- `experiments/analysis/e140_multirate_lambda0018_two_seed_package.{json,md,seed_summary.csv,method_summary.csv,checks.csv}`

## E140 Update: Multi-Rate Lambda0018 Three-Seed Active-Geometry Check

Status: Done for seeds `1234`, `2345`, and `3456`. I completed the seed3456 low-rate scalar, HCS, and active-HCG `bias010` lane, then generated a three-seed package. All training, evaluation, and feature extraction used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; numeric finite checks are `True` for all sweeps and feature tables.

The seed3456 result changes the interpretation. Scalar best RD is `1.675580`; HCS best is step500 with RD `1.178273`; HCG `bias010` best is also step500 with RD `1.207323`. So active-HCG worsens seed3456 by `+0.029050` RD vs HCS. The three-seed mean is still slightly better for HCG (`1.191737`) than HCS (`1.192908`), but only by `-0.001171`, with HCG winning `2/3` seeds.

The intermediate-feature readout points to reliability/usage control rather than more raw geometry. HCG remains active on seed3456 (`householder_delta_rms=0.040809`), but y-error worsens by `+0.004033`, qMSE is slightly worse by `+0.000218`, index bpp rises by `+0.001209`, and RD/MS-SSIM both degrade. Across three seeds, mean dead-code delta is `+0.023220`, mean qMSE delta is `-0.001667`, and mean y-error delta is `+0.003646`.

Decision: do not promote low-rate `bias010` as a paper-main rate claim yet. It is useful method-development evidence because active geometry can help two seeds and nearly matches/improves the mean, but the fragile seed means the next priority is a reliability/usage-controlled low-rate variant or holdout-confirmed selector, not simply stronger geometry.

Artifacts:

- `configs/pilot_scalar_baseline_lambda0018_seed3456.yaml`
- `configs/pilot_hcs_rvq_frozen_lambda0018_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_bias010_frozen_lambda0018_seed3456.yaml`
- `tools/build_e140_multirate_lambda0018_three_seed_package.py`
- `experiments/analysis/e140_scalar_lambda0018_seed3456_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_hcs_lambda0018_seed3456_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_hcg_gate025_bias010_lambda0018_seed3456_kodak_checkpoint_sweep.csv`
- `experiments/analysis/e140_*seed3456*kodak*feature_distribution.{json,csv}`
- `experiments/analysis/e140_multirate_lambda0018_three_seed_package.{json,md,seed_summary.csv,method_summary.csv,checks.csv}`

## E141/E142: Low-Rate Bias010 Reliability Headroom

Status: Done for Kodak24 diagnostics. I evaluated the E140 `lambda_rd=0.0018` HCS and active-HCG `bias010` checkpoints per image across seeds `1234`, `2345`, and `3456`, then audited reliability/selector headroom. All GPU evaluation was pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and every generated table reports zero nonfinite rows.

The per-image result explains the fragile three-seed mean. Fixed HCG `bias010` is only slightly better than HCS on average: HCS RD `1.192908`, HCG RD `1.191737`, delta `-0.001171`. But the per-image HCS/HCG oracle reaches RD `1.167911`, which is `-0.024997` vs HCS and `-0.023826` vs fixed HCG. HCG wins `33/72` images, so the low-rate geometry is not useless; the problem is reliability selection.

A leave-one-seed-out image-level selector using high `hcg_rvq_s_q_mean` is the best simple signal: mixed RD `1.185735`, delta `-0.007173` vs HCS, wins `2/3` held-out seeds, and reduces the seed3456 damage from fixed HCG `+0.029050` to `+0.001559`. This is promising because `s_q` is hyperprior-side and decoder-known, but it is still posthoc image-level selection, not a fixed codec.

I also tested a more direct model-side idea: applying a local `s_q` risk multiplier posthoc to the already-trained HCG checkpoints. That did not work. The best row is center `0.65`, min multiplier `0.75`, mean delta `+0.000231` vs HCS, with seed3456 worsening to `+0.035453`. Therefore the next method target should not be a continuous local `s_q` multiplier. The better next target is image-level or learned reliability control that preserves useful low-rate geometry while avoiding fragile-image damage.

Artifacts:

- `tools/analyze_e141_lowrate_bias010_selector_headroom.py`
- `tools/build_e141_lowrate_selector_summary.py`
- `tools/analyze_e142_lowrate_sq_risk_posthoc.py`
- `experiments/analysis/e141_lowrate_bias010_selector_headroom.{md,json,pairs.csv,selectors.csv}`
- `experiments/analysis/e141_lowrate_bias010_selector_summary.{md,json,csv}`
- `experiments/analysis/e142_lowrate_sq_risk_posthoc.{md,json,aggregate.csv,summary.csv}`

## E143: Low-Rate Bias010 Holdout4096 Selector Audit

Status: Done. I reran the low-rate `lambda_rd=0.0018` HCS vs active-HCG `bias010` comparison on the OpenImages holdout4096 split, using path-aligned direct evaluation for seeds `1234`, `2345`, and `3456`. All evaluation used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and the generated package reports `0` nonfinite rows across `12288` paired image-seed rows.

The holdout result is much stronger than the Kodak-only E141 readout. Fixed HCG `bias010` reaches mean RD `1.182335` versus HCS `1.200992`, a `-0.018656` RD improvement. Per-seed, HCG improves seed1234 by `-0.081201` and seed2345 by `-0.010787`, but worsens seed3456 by `+0.036019`. Therefore the method-strengthening result is real on an external split, but the fragile-seed reliability problem remains.

The per-image oracle shows large remaining headroom: oracle RD is `1.150651`, which is `-0.050341` vs HCS and `-0.031685` vs fixed HCG. HCG wins `6535/12288` paired images. By HCS difficulty quartile, fixed HCG worsens the easiest quartile (`Q1`: `+0.020572`) but improves harder quartiles increasingly (`Q2`: `-0.001161`, `Q3`: `-0.019836`, `Q4`: `-0.074200`). This is a useful paper-facing mechanism result: low-rate geometry mainly helps hard images and can hurt easy images without reliability control.

The best leave-one-seed-out selector family is low `hcg_rvq_householder_strength`. It gives mean held-out delta `-0.030576` vs HCS, wins `2/3` held-out seeds, preserves the full seed1234 gain (`-0.081201`), keeps seed2345 positive (`-0.010551`), and almost removes the seed3456 damage (`+0.000023` instead of `+0.036019`). This is stronger than the E141 Kodak selector because it transfers to holdout4096 and uses an HCG intermediate geometry statistic rather than only image difficulty.

Decision: promote low-rate `bias010` from "fragile Kodak signal" to "promising method-strengthening lane with holdout evidence", but do not report the LOSO selector as a final codec yet. The next implementation target is a fixed reliability controller based on decoder-known geometry summaries, starting from `householder_strength` and cross-checking with `s_q_mean`, then re-evaluating without validation-time switching. In parallel, keep the controlled component-ablation lane because it is still the safest manuscript claim.

Artifacts:

- `tools/analyze_e143_lowrate_bias010_holdout_selector.py`
- `experiments/analysis/e143_lowrate_bias010_holdout4096_selector.{md,json}`
- `experiments/analysis/e143_lowrate_bias010_holdout4096_selector.{all_rows,pairs,per_seed,selectors,selector_summary,quartiles,correlations}.csv`

## E144: Low-Rate Bias010 Transfer-To-Holdout Controller Audit

Status: Done. I converted the E143 selector diagnosis into a cleaner split protocol. The threshold is trained on an independent OpenImages transfer split (`start_index=8192`, `4096` images) and then applied unchanged to the E143 holdout4096 pairs. All transfer evaluations used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and the transfer package reports `0` nonfinite rows across `12288` paired image-seed rows.

The transfer split repeats the E143 signal. Fixed HCG `bias010` reaches transfer RD `1.160089` versus HCS `1.178765`, a `-0.018676` RD gain. Per-seed, transfer HCG improves seed1234 by `-0.081840` and seed2345 by `-0.008050`, but worsens seed3456 by `+0.033860`. This mirrors holdout4096 closely, so the low-rate geometry behavior is not a one-split artifact.

The best transfer-trained controller is exactly the prespecified E143 family: low `hcg_rvq_householder_strength`. The transfer-trained threshold is `0.271352783`. Applying it unchanged to holdout4096 gives mixed RD `1.170368`, which is `-0.030624` vs HCS and `-0.011968` vs fixed HCG. It selects `8204/12288` rows (`0.667643`), keeps seed1234 and seed2345 at the full fixed-HCG gains (`-0.081201` and `-0.010787`), and reduces the fragile seed3456 damage from `+0.036019` to `+0.000115`.

The feature distribution is also mechanism-consistent. Selected holdout rows are harder under HCS (`hcs_rd=1.219957` vs rejected `1.162894`) and have lower predicted Householder strength (`0.259484` vs `0.283398`). HCG helps selected rows strongly (`hcg_minus_hcs=-0.045869`, win fraction `0.682228`) but hurts rejected rows (`+0.036009`, win fraction `0.229677`). This supports a clean reliability-control story: geometry should be used where the hyperprior predicts a low-strength transform on harder images, and suppressed where a stronger transform would overfit easy/fragile images.

Decision: this is strong controlled evidence, but still not a final codec row because it posthoc switches between separately evaluated HCS and HCG checkpoints. Promote it as a protocol-clean diagnostic and next implementation target. The next method action is to implement one decoder-reproducible reliability gate inside HCG that uses the same `householder_strength` signal to suppress geometry, then train/evaluate it on the same transfer/holdout protocol. In parallel, keep the SOTA/backbone plug-in lane queued, using this controller as the reliability design to port.

Artifacts:

- `tools/analyze_e144_lowrate_bias010_transfer_controller.py`
- `experiments/analysis/e144_lowrate_bias010_transfer_to_holdout_controller.{md,json}`
- `experiments/analysis/e144_lowrate_bias010_transfer_to_holdout_controller.{controllers,best_feature_stats,preset_feature_stats}.csv`
- `experiments/analysis/e144_lowrate_bias010_transfer_to_holdout_controller.transfer_start8192_{all_rows,pairs,per_seed}.csv`

## E145/E146: Low-Rate Reliability Controller Implementation Audit

Status: Done for the first single-checkpoint controller attempts. I evaluated two ways to turn the E144 independent-split HCS/HCG diagnostic switch into one deployable HCG checkpoint: a deterministic householder-strength backoff (E145), and a learned teacher-head reliability controller trained on the transfer split (E146). All training and evaluations were pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and all generated tables report `0` nonfinite rows.

E145 is a negative single-checkpoint result. The E144 threshold (`householder_strength < 0.271352783`) is useful when switching between separately evaluated HCS and HCG checkpoints, but applying it as a strength backoff inside the same HCG checkpoint with backoff min `0.650` worsens RD: HCS `1.200992`, fixed HCG `1.182335` (`-0.018656` vs HCS), and strength-backoff `1.215586` (`+0.014594` vs HCS, `+0.033251` vs fixed HCG). It also worsens the fragile seed3456 by `+0.107861` vs HCS. This says that simply shrinking the Householder gate after training changes the operating point rather than reproducing the HCS/HCG oracle switch.

E146 is also negative, but more diagnostic. The teacher labels are independent of holdout and are well-defined: for seed3456 on the transfer split the keep fraction is only `0.244873`, so a useful head should strongly lower reliability on suppress-label images. Instead, the trained head remains saturated even on the same transfer split: step250 reliability `0.983447`, step500 `0.972388`, both with predicted keep fraction `1.000000` at `rel >= 0.5`. Label fit is below random (`AUC=0.467288` at step250 and `0.468973` at step500), and keep/suppress reliability means are nearly identical (`0.983337/0.983483` and `0.972033/0.972502`). On holdout4096 the same head also loses slightly to fixed HCG and remains worse than HCS for seed3456.

Decision: keep E144 as strong controlled evidence, not as a final deployed codec row. Do not promote E145/E146 as paper-main. For the method-strengthening lane, the next useful experiment is not another small posthoc multiplier; it should either (1) test whether a stronger teacher-head calibration can actually fit the transfer labels, or (2) move to an explicit fallback/selector design that preserves the HCG operating point instead of multiplying the existing geometry gate. In parallel, the paper-safe lane remains E139/E143/E144: entropy-only ablation plus independent-split reliability evidence.

Artifacts:

- `tools/analyze_e145_lowrate_strength_backoff_single_checkpoint.py`
- `tools/export_e146_lowrate_bias010_teacher_labels.py`
- `tools/analyze_e146_lowrate_bias010_teacher_headonly.py`
- `tools/analyze_e146_teacher_head_transfer_fit.py`
- `configs/pilot_hcg_rvq_h_gate025_bias010_teacher_transfer8192_relmin000_rho050_headonly_lambda0018_seed3456.yaml`
- `experiments/analysis/e145_lowrate_strength_backoff_single_checkpoint.md`
- `experiments/analysis/e146_lowrate_bias010_transfer8192_reliability_teacher_labels.{md,json,csv}`
- `experiments/analysis/e146_lowrate_bias010_teacher_headonly_holdout4096.{md,json,aligned.csv,by_step.csv,quartiles.csv}`
- `experiments/analysis/e146_lowrate_bias010_teacher_headonly_transfer8192_fit.{md,json,aligned.csv,by_step.csv}`

## E147: Strong Teacher-Head Calibration Audit

Status: Done for the transfer-fit audit. I trained a stronger seed3456 teacher-head controller from the same fixed low-rate HCG `bias010` checkpoint, using the independent transfer split labels but increasing the reliability loss weight and head-only learning rate (`rho=20.0`, `lr=5.0e-3`). Training and evaluation were pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and the full transfer audit reports `0` nonfinite rows for both checkpoints.

The stronger calibration confirms that E146's saturated head was not the only possible behavior. On the transfer split, E147 step500 improves label separation relative to E146: reliability keep/suppress means become `0.532626/0.456699`, AUC reaches `0.640196`, and label correlation reaches `+0.205475`. This is not a perfect classifier, but it is a real change from E146's near-constant reliability (`AUC < 0.47`).

However, the RD behavior is decisively negative. The same step500 head gives teacher-head RD `1.532191` on the transfer split, which is `+0.392222` worse than HCS and `+0.358362` worse than fixed HCG. Step250 is also collapsed (`+0.363879` vs HCS). The q95 damage is very large (`0.937980` at step250 and `1.010057` at step500), with only `197/4096` and `189/4096` wins vs HCS.

Decision: this closes the small "maybe the head was just too weak" loophole. Stronger head training can fit the teacher labels better, but multiplying down the Householder/geometry path inside one fixed HCG checkpoint still destroys the trained operating point. The method-strengthening lane should now move away from posthoc gate multiplication and toward an explicit fallback/selector design that preserves two operating states, or a jointly trained HCS/HCG mixture that learns the fallback from the start. The controlled-evidence lane remains E139/E143/E144.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_bias010_teacher_transfer8192_relmin000_rho20_headonly_lr005_lambda0018_seed3456.yaml`
- `tools/analyze_e147_teacher_head_transfer_fit.py`
- `tools/analyze_e147_lowrate_bias010_teacher_headonly.py`
- `experiments/analysis/e147_lowrate_bias010_teacher_headonly_rho20_lr005_transfer8192_fit.{md,json,aligned.csv,by_step.csv}`

## E148/E149: Low-Rate Residual-Selector Transfer-Fit Stress Test

Status: Done for the seed3456 transfer-fit branch. I tested whether the E144 independent-split HCS/HCG reliability signal can be turned into one low-rate `bias010` checkpoint using the exact-default residual selector. E148 used a conservative setting (`rho=1.0`, `lr=1.0e-4`, `rho_anchor_y_hat=50.0`, deadzone014). E149 kept the same exact-default selector but made the teacher objective strong (`rho=20.0`, `lr=5.0e-3`, no y_hat anchor) and evaluated both step250 and step500. All training/evaluation was pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and every generated table reports `nonfinite_rows=0`.

E148 is stable but underfit. The selector stays near default: mean selector probability is only `0.006506`, predicted suppress at `p>=0.5` is `0.000000`, AUC is `0.462855`, and BCE is `4.063077`. RD is essentially the fixed-HCG operating point (`selector-HCS=+0.033355`, `selector-fixed=-0.000505`). This means the exact-default branch can preserve the checkpoint, but the conservative objective does not learn the transfer suppress label.

E149 answers the stronger question. With `rho=20.0` and `lr=5.0e-3`, the selector does move: step500 mean probability reaches `0.499493`, suppress/keep probabilities separate as `0.511250/0.463237`, AUC improves to `0.580217`, and BCE drops to `0.719839`. But RD collapses: step500 is `+0.166399` worse than HCS and `+0.132538` worse than fixed HCG, with q95 damage `0.536466` and only `499/4096` wins vs HCS. Step250 is also negative (`+0.127946` vs HCS).

Decision: this closes the main residual-selector loophole for the low-rate `bias010` single-checkpoint deployment. E148 shows that a conservative exact-default selector preserves RD but does not learn; E149 shows that forcing it to learn creates the wrong operating point. Therefore the controlled-evidence lane remains E139/E143/E144, while the method-strengthening/SOTA-plug-in lane should move to an explicit branch/fallback or jointly trained mixture that preserves HCS-like and HCG-like states, instead of continuously shrinking a trained HCG geometry gate.

Artifacts:

- `configs/pilot_hcg_rvq_h_gate025_bias010_residualselector_transfer8192_suppress_rho100_yhatanchor50_deadzone014_lambda0018_seed3456.yaml`
- `configs/pilot_hcg_rvq_h_gate025_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_lambda0018_seed3456.yaml`
- `tools/analyze_e148_lowrate_bias010_residualselector_transfer_fit.py`
- `tools/analyze_e149_lowrate_bias010_residualselector_strong_transfer_fit.py`
- `experiments/analysis/e148_lowrate_bias010_residualselector_transfer8192_suppress_rho100_yhatanchor50_deadzone014_fit.{md,json,aligned.csv,by_step.csv}`
- `experiments/analysis/e149_lowrate_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_fit.{md,json,aligned.csv,by_step.csv}`

## E150: Branch-vs-Continuous Controller Audit

Status: Done. I consolidated the low-rate controller evidence into one reusable audit table, explicitly separating state-preserving branch/fallback behavior from continuous geometry suppression. This is a CPU-only analysis over existing artifacts; no GPU was used. The script also reads actual image sizes to estimate the cost of signaling a one-bit image-level branch choice.

The central result is strong. HCS has holdout4096 RD `1.200992`, fixed HCG `bias010` has `1.182335` (`-0.018656` vs HCS), and the E144 transfer-trained strength branch has `1.170368` (`-0.030624` vs HCS, `-0.011968` vs fixed HCG). The per-image oracle is still better at `1.150651`, so the branch signal has remaining headroom, but the transfer-trained branch already keeps about `1.64x` the fixed-HCG gain relative to HCS because it removes much of the fragile tail.

The one-bit signaling check is important for a deployable path. The actual one-bit-per-image side cost on these OpenImages rows is only `0.000001362` bpp/RD units, and even the conservative `256x256` patch assumption is `0.000015259`. With either penalty, the branch remains around `-0.03061` vs HCS and `-0.01195` vs fixed HCG. This means a signaled image-level HCS/HCG branch is not ruled out by rate cost; the bottleneck is implementing compatible states cleanly.

The continuous suppression contrast is decisive. E145 strength backoff is `+0.014594` worse than HCS and `+0.033251` worse than fixed HCG on holdout4096. E147 strong reliability learning and E149 strong residual selector can move their heads, but they damage seed3456 transfer RD badly (`+0.392222` and `+0.166399` vs HCS at step500). E148 is stable but nearly a no-op. This confirms that the next method-strengthening track should preserve explicit HCS-like and HCG-like states instead of multiplying down a trained HCG geometry path.

Decision: keep E139/E143/E144/E150 as the controlled-evidence lane for the paper claim. For the HCG-strengthening and SOTA/backbone plug-in lane, implement a state-preserving branch/fallback design next: either a signaled image-level branch as a conservative codec row, or a jointly trained two-state mixture with an identity/HCS-like branch and an HCG branch. Do not spend more cycles on small variants of continuous gate shrinkage unless they explicitly preserve both states.

Artifacts:

- `tools/build_e150_branch_vs_continuous_controller_audit.py`
- `experiments/analysis/e150_branch_vs_continuous_controller_audit.{md,json,methods.csv}`

## E151: Signaled Branch Direct Evaluation

Status: Done. I reran the matched low-rate HCS and active-HCG `bias010` checkpoints directly on holdout4096, then applied the same transfer-trained branch rule from E144/E150: use HCG when `hcg_rvq_householder_strength <= 0.271352783`, otherwise use the HCS state. This was a full GPU evaluation, not only arithmetic over the E144/E150 audit CSVs. It was pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. The package reports `0` nonfinite rows across `12288` paired image-seed rows.

The direct results match the branch audit cleanly. HCS has RD `1.200992`, fixed HCG `bias010` has RD `1.182335` (`-0.018656` vs HCS), and the state-preserving branch has RD `1.170368` (`-0.030624` vs HCS, `-0.011968` vs fixed HCG). Adding the actual one-bit image-level signal cost gives RD `1.170369`, still `-0.030623` vs HCS. The measured side cost is only `0.000001362` bpp/RD units.

The per-seed behavior is exactly the behavior needed for the reliability claim. Seed1234 keeps the full HCG gain (`-0.081201` vs HCS), seed2345 keeps the full HCG gain (`-0.010787`), and seed3456 is almost fully protected: fixed HCG is `+0.036019` worse than HCS, while the branch is only `+0.000115` worse. The branch selects HCG on all rows for seeds1234/2345 and on only `0.002930` of rows for seed3456.

Decision: this upgrades E150 from a consolidated audit to a directly reproduced protocol result. It is still not yet a single integrated codec row, because it switches between matched HCS and HCG states, but it is now a strong paper-facing controlled evidence result for the two-state reliability mechanism. For the controlled-evidence lane, E139/E143/E144/E150/E151 form the current safest story. For the method-strengthening and SOTA/backbone plug-in lane, the next implementation should preserve explicit states first, then port that branch/fallback design into a stronger backbone.

Artifacts:

- `tools/run_e151_signaled_branch_direct_eval.py`
- `experiments/analysis/e151_signaled_branch_direct_eval.{md,json,raw_rows.csv,pairs.csv,per_seed.csv}`

## E152: Branch Manifest and SOTA/Backbone Scout

Status: Done for the manifest smoke and static SOTA/backbone scouting. I turned the E151 state-preserving branch into a reusable manifest/evaluator pair instead of leaving it hardcoded in the E151 script. The manifest fixes the low-rate `lambda_rd=0.0018` HCS/HCG `bias010` state pair, the transfer-trained rule `hcg_bias010_rvq_householder_strength <= 0.27135278284549713`, and the one-bit image-level signal accounting. A 2-image smoke over the same holdout protocol was run with `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; it reports `0` nonfinite rows and reproduces the intended behavior: seeds1234/2345 select HCG, while the fragile seed3456 falls back to HCS.

The full paper-facing branch evidence remains E151: HCS RD `1.200992`, fixed HCG RD `1.182335`, branch RD `1.170368`, and signaled branch RD `1.170369`, with `0` nonfinite rows across `12288` paired image-seed rows. E152 makes that protocol portable to new splits and future rate points.

I also cloned and inspected official SOTA/competitor repositories under `third_party/`: DCAE, MambaIC, LIC-HPCM, and RDVQ. The scouting conclusion is that the first real SOTA/backbone probe should use the state-preserving branch interface, not raw fixed HCG or continuous geometry shrinkage. DCAE is the cleanest first plug-in target because its `forward` path exposes `y = g_a(x)`, hyper-derived means/scales, sliced `y_hat` construction, and `g_s(y_hat)`. MambaIC is a strong second target but has heavier VMamba/selective-scan dependency risk. HPCM is very relevant to future RVQ stage-context design, but its progressive `forward_hpcm/compress_hpcm/decompress_hpcm` path is too deep for the first plug-in smoke. RDVQ remains a primary comparison paper, but its official repository currently says code will come soon, so it should not block experiments.

Decision: continue the two-track plan. The controlled-evidence lane uses E152 manifests for new split/rate evaluations. The method-strengthening/SOTA lane starts with a DCAE forward-only adapter smoke and keeps MambaIC/HPCM as later staged targets.

Artifacts:

- `configs/e152_lowrate_hcs_hcg_signaled_branch_manifest.yaml`
- `tools/evaluate_signaled_branch_manifest.py`
- `tools/build_e152_branch_sota_package.py`
- `experiments/analysis/e152_lowrate_hcs_hcg_signaled_branch_smoke.{md,json,raw_rows.csv,pairs.csv,per_seed.csv}`
- `experiments/analysis/e152_branch_manifest_sota_package.{md,json,sota_repos.csv,next_actions.csv}`



## E153: DCAE HCG Adapter Forward-Only Smoke

Status: Done for the first official-backbone plug-in smoke. I connected the local `HCGQuantizerAdapter` to the official DCAE repository under `third_party/DCAE`, using the DCAE analysis/hyperprior boundary: `y = g_a(x)`, `z = h_a(y)`, `latent_scales = h_z_s1(z_hat)`, `latent_means = h_z_s2(z_hat)`, and `x_hat = g_s(y_hat)`. The HCG adapter receives `torch.cat([latent_scales, latent_means], dim=1)` as hyper features and produces an adapter-side `y_hat` that is decoded by DCAE `g_s`.

The first 128x128 probe failed in the DCAE window-attention reshape because an internal 5x5 feature map is not divisible by the window partition size. I reran the smoke at 256x256, pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. The 256x256 run passed with `0` nonfinite rows. DCAE baseline forward has finite `x_hat`, `y`, and likelihoods. The HCS-style adapter identity init is finite with qMSE `0.188207`, dead-code ratio `0.703125`, and perplexity `14.617975`. The active HCG geometry adapter is also finite with qMSE `0.186152`, Householder strength `0.250000`, local delta RMS `0.026319`, dead-code ratio `0.703125`, and perplexity `14.136592`.

This smoke also exposed and fixed a real adapter-contract bug: `HCGQuantizerAdapter` lacked `_householder_gate_strength_backoff_multiplier`, even though the model-side `HCGMeanScaleHyperprior` had the helper and the shared quantizer runner expected it. I added the same helper to the adapter class and verified `py_compile` before rerunning the DCAE smoke.

Decision: E153 is not an RD, pretrained-DCAE, or SOTA quality result. Its value is architectural: an official strong-backbone codebase exposes a compatible boundary for HCG local geometry, and the local adapter can run there without NaNs. This starts the method-strengthening/SOTA plug-in lane while the controlled-evidence lane remains E151/E152.

Artifacts:

- `tools/run_e153_dcae_hcg_adapter_smoke.py`
- `experiments/analysis/e153_dcae_hcg_adapter_smoke.{md,json}`
- `hcg_rvq/quantizers/hcg_adapter.py`


## E154: SOTA/Backbone Reproduction Audit

Status: Done. I converted the SOTA/backbone discussion into a reproducible audit artifact before spending larger GPU budget. The audit covers DCAE, a second DCAE clone from the README official URL, MambaIC, LIC-HPCM, and RDVQ. It records remote URLs, commits, checkpoint availability, evaluation entry points, plug-in priority, and risk.

The key provenance result is that the existing `third_party/DCAE` clone from `CVL-UESTC/DCAE` and the README official clone target `LabShuHangGU/DCAE` resolve to the same commit `e2525a00467cbc326045674c7e5e0f1d9964604b`. Therefore E153 is not invalidated by the remote-name mismatch. The DCAE README exposes 12 Google Drive checkpoint links, and HPCM exposes 26; after de-duplicating the DCAE mirror, there are 38 unique checkpoint file IDs across the audited READMEs.

The SOTA/backbone priority remains DCAE first. It has the cleanest `g_a/h_a -> latent_scales,latent_means -> y_hat -> g_s` boundary and official low-rate checkpoints. HPCM has stronger R-D data and many checkpoints, but its progressive codec path is deeper. MambaIC is a strong second plug-in target, but its VMamba/selective-scan dependency and reimplementation note make it riskier for the first integration. RDVQ is paper-critical as a VQ comparison, but the local official README still says code will come soon.

Decision: continue the two-track plan. Controlled evidence stays anchored on E151/E152. The SOTA/backbone lane now moves to DCAE baseline reproduction with a low-rate MSE checkpoint, then a DCAE-over-itself HCS/HCG state-preserving branch comparison. This addresses the concern that small smoke tests may not scale, without turning the first large experiment into an ambiguous raw transplant.

Artifacts:

- `tools/build_e154_sota_backbone_reproduction_audit.py`
- `third_party/DCAE_LabShuHangGU`
- `experiments/analysis/e154_sota_backbone_reproduction_audit.{md,json}`
- `experiments/analysis/e154_sota_backbone_reproduction_audit_repos.csv`
- `experiments/analysis/e154_sota_backbone_reproduction_audit_checkpoint_links.csv`
- `experiments/analysis/e154_sota_backbone_reproduction_audit_next_actions.csv`


## E155: DCAE Pretrained Baseline Reproduction

Status: Done for the first pretrained SOTA/backbone baseline reproduction. I installed `gdown`, downloaded the official DCAE MSE `lambda=0.0018` checkpoint from the README Google Drive link, and verified that it loads correctly: the checkpoint is 1.4GB, `epoch=91`, and contains `1142` state tensors including `module.dt` with shape `(128, 640)`. All runs were pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used.

I first ran the official DCAE `eval.py` on a path-fixed Kodak first4 subset. It completed with average PSNR `29.75`, MS-SSIM dB `11.2839`, and bitrate `0.073` bpp. I then added a local E155 evaluator that mirrors the forward-eval path but saves per-image metrics plus intermediate feature distributions for later same-backbone HCG comparison.

The E155 first4 artifact reports bpp `0.073300`, PSNR `29.756827`, MS-SSIM dB `11.281572`, and `0` nonfinite rows. The full Kodak 24 artifact reports bpp `0.109397`, PSNR `29.245565`, MS-SSIM dB `11.753408`, and `0` nonfinite rows. The CSV includes per-image bpp, PSNR, MS-SSIM dB, and finite/distribution stats for DCAE `y`, predicted means, predicted scales, and likelihoods.

Decision: this is now a real pretrained-backbone reproduction baseline, not only a random-weight architecture smoke. It is still not a full multi-rate SOTA benchmark, but it is the correct foundation for the next DCAE-over-itself HCS/HCG branch comparison.

Artifacts:

- `tools/run_e155_dcae_pretrained_baseline_smoke.py`
- `third_party/checkpoints/dcae/dcae_lambda0018_mse.pth.tar`
- `experiments/data/kodak_first4`
- `experiments/analysis/e155_dcae_lambda0018_kodak_first4_baseline_smoke.{md,json,csv}`
- `experiments/analysis/e155_dcae_lambda0018_kodak24_baseline.{md,json,csv}`

## E156: EF-LIC Pretrained Baseline Reproduction and Feature Audit

Status: Done for the first official VQ/RVQ generative-codec baseline reproduction. I downloaded the official EF-LIC checkpoint from the repository link into `third_party/EF-LIC/ckpt/checkpoint.pth.tar` and evaluated it on Kodak24 using GPU0 only. The command path used `env CUDA_VISIBLE_DEVICES=0 ... --device cuda:0`; device 1 was not used.

I first ran the official `third_party/EF-LIC/test.py` on `experiments/data/kodak_first4`, then expanded to all `experiments/data/kodak24`. After that I added `tools/run_e156_eflic_pretrained_baseline.py` so the result is reproducible and saves CSV/JSON/Markdown artifacts with intermediate RVQ-boundary statistics.

Kodak24 summary:

| force_ind | bpp | PSNR | LPIPS | DISTS | nonfinite | y index entropy | y used frac | y norm std | scale mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.035645 | 21.6693 | 0.27228 | 0.09227 | 0 | 7.1489 | 0.5464 | 0.8386 | 4.7722 |
| 1 | 0.071289 | 23.0063 | 0.21043 | 0.06437 | 0 | 7.8022 | 0.7685 | 0.7251 | 5.3178 |
| 2 | 0.106934 | 23.8530 | 0.18049 | 0.05270 | 0 | 8.0302 | 0.8637 | 0.6881 | 5.5234 |
| 3 | 0.142578 | 24.4878 | 0.16255 | 0.04687 | 0 | 8.1524 | 0.9122 | 0.6744 | 5.5992 |
| 4 | 0.178223 | 24.9058 | 0.15168 | 0.04327 | 0 | 8.2219 | 0.9433 | 0.6694 | 5.6217 |

Interpretation:

- EF-LIC is now a concrete SOTA/VQ-GIC baseline in the workspace, not only a paper idea.
- The RVQ boundary is exactly the kind of place HCG-RVQ should be tested: EF-LIC normalizes each `y_slice` by context-predicted mean/scale, runs RVQ, then de-normalizes before the next slice.
- The measured code usage rises with `force_ind`, as expected from the larger RVQ budget. This gives a useful diagnostic axis for HCG adapter experiments: an HCG variant should improve perceptual/RD behavior without destroying the near-high-entropy index usage that EF-LIC relies on.
- Because EF-LIC's public repository is inference-only, any retraining-based paper claim will require either official training code or a carefully labeled local reconstruction. The immediate next experiment should be a forward-only HCG adapter smoke at the normalized `y_slice` RVQ boundary.

Artifacts:

- `tools/run_e156_eflic_pretrained_baseline.py`
- `third_party/EF-LIC/ckpt/checkpoint.pth.tar`
- `experiments/data/kodak24`
- `experiments/analysis/e156_eflic_pretrained_kodak24.{md,json,csv}`

## E157: EF-LIC HCG Insertion Design Audit

Status: Done for design/protocol audit. After E156, I inspected the EF-LIC RVQ implementation in detail and found the important insertion constraint: EF-LIC does not quantize each 256-channel `y_slice` directly. Each `VectorQuantizerProjInfer` first projects the slice from `in_dim` to `CODEBOOK_DIM=8`, performs nearest-neighbor VQ in the 8-D projected space, then decodes the selected codeword back to the original slice dimension through `out_proj`. RVQ is built by stacking these projected-space VQ modules.

This means the current generic groupwise `HCGQuantizerAdapter` should not be naively dropped over 256 channels if the claim is "HCG improves EF-LIC's RVQ". The cleaner plug-in is projected-space HCG: keep EF-LIC's `in_proj`, `out_proj`, codebook sizes, stage counts, and bit packing, and add hyper/context-conditioned shift/scale/Householder geometry around the 8-D residual/codebook space.

Decision:

- EF-LIC remains the best first VQ-SOTA target in principle.
- The next implementation should first build an identity projected-space wrapper that exactly reproduces EF-LIC's original RVQ path.
- Only after exact reproduction should active HCG geometry be introduced and evaluated.
- GLC remains a useful second target/benchmark, but EF-LIC is cleaner for a direct RVQ-improvement claim.

Artifact:

- `experiments/analysis/e157_eflic_hcg_insertion_design.md`

## E158: EF-LIC Projected-Space Identity Wrapper

Status: Done. I implemented a projected-space wrapper around EF-LIC's RVQ path that exposes the exact HCG insertion point while keeping the current geometry as identity: no local shift, unit scale, and no Householder transform. This wrapper preserves EF-LIC's `in_proj`, 8-D codebook search, `out_proj`, stage count, `force_ind`, and packed-index protocol.

The Kodak24 direct check is exact across all 5 `force_ind` points and all 24 images. Payload equality is `24/24` for every point, z-index mismatch is `0`, y-index mismatch is `0`, max `x_hat` difference is `0`, mean `x_hat` difference is `0`, and nonfinite rows are `0`. This means the wrapper itself introduces no codec drift; future active projected-HCG changes can be attributed to the geometry/conditioning logic rather than an evaluation or bitstream-path artifact.

Kodak24 summary:

| force_ind | bpp | PSNR | y index H | y used frac | payload equal | z/y mismatch | max xhat diff | nonfinite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.035645 | 21.6693 | 7.1489 | 0.5464 | 24/24 | 0/0 | 0.000e+00 | 0 |
| 1 | 0.071289 | 23.0063 | 7.3151 | 0.5803 | 24/24 | 0/0 | 0.000e+00 | 0 |
| 2 | 0.106934 | 23.8530 | 7.3415 | 0.5860 | 24/24 | 0/0 | 0.000e+00 | 0 |
| 3 | 0.142578 | 24.4878 | 7.3533 | 0.5878 | 24/24 | 0/0 | 0.000e+00 | 0 |
| 4 | 0.178223 | 24.9058 | 7.3606 | 0.5856 | 24/24 | 0/0 | 0.000e+00 | 0 |

Decision: EF-LIC is now ready for active projected-HCG smoke tests. The first active variant should be conservative and diagnostic-first: geometry bounded in 8-D projected residual space, GPU0 only, per-image metric deltas, index entropy/usage, qMSE, residual norm, and nonfinite checks. This should run before any retraining claim.

Artifacts:

- `tools/run_e158_eflic_projected_identity_wrapper.py`
- `experiments/analysis/e158_eflic_projected_identity_first4.{md,json,csv}`
- `experiments/analysis/e158_eflic_projected_identity_kodak24.{md,json,csv}`

## E159: GLC Reproduction and HCG Insertion Audit

Status: Done for the GLC risk-hedge audit. I reread the local GLC paper PDF and official clone to decide whether it should run in parallel with EF-LIC for the VQ-LIC plug-in story. GLC is a strong low-bitrate generative image-compression benchmark, but its main compression-side `y` path is not already an RVQ module. In code, `GLC_Image.test()` follows `vqgan.encoder -> enc -> hyper_enc -> z_vq -> hyper_dec/y_prior -> forward_four_part_prior -> dec -> vqgan.generator`. The explicit VQ is `z_vq` for hyper side information, while the main `y` path is masked scalar rounding around predicted mean/scale and quant step.

Paper/protocol facts were fixed from the PDF: Stage I trains on ImageNet, Stage II/III on OpenImages test patches, natural-image crops are `256x256`, optimizer is AdamW, batch size is `8`, natural-image evaluation is CLIC2020 original resolution, with Kodak/DIV2K/MS-COCO in supplementary results. Primary perceptual reporting is DISTS/FID/KID, with LPIPS/PSNR/MS-SSIM for completeness; FID/KID use `256x256` patches and are omitted on Kodak.

Decision: keep EF-LIC as the first direct HCG-RVQ plug-in target because it already has an RVQ bottleneck and E158 locked the projected 8-D insertion boundary. Keep GLC as a risk hedge and low-bitrate generative benchmark: first reproduce pretrained GLC inference once weights are downloaded from the official release, then consider either `z_vq` geometry or a larger `y`-residual HCG-RVQ replacement. GLC should not displace EF-LIC for the first direct RVQ-improvement claim.

Artifact:

- `experiments/analysis/e159_glc_reproduction_and_hcg_insertion_audit.md`

## E160: EF-LIC Active Projected-HCG Diagnostic Smoke

Status: Done for diagnostic active geometry. Building on the exact E158 identity wrapper, I added an EF-LIC projected-HCG smoke that applies a decoder-reproducible rank-1 Householder-style geometry in the 8-D projected RVQ space. The direction is generated from EF-LIC's decoder-side context (`mean` by default), so both encoder and decoder can reconstruct the same transformed codebook geometry from the same bitstream indices. The z RVQ remains unchanged; the active geometry targets the four y-slice RVQ paths.

The first4 sweep over `alpha={0.02,0.05,0.10}` and all five `force_ind` points passed with `0` nonfinite rows and `max_decode_diff=0`. Then I expanded the most stable candidate (`direction_source=mean`, `alpha=0.05`) to Kodak24 on GPU0. It also passed with `0` nonfinite rows and `max_decode_diff=0` for all 120 rows. Device 1 was not used.

Kodak24 alpha0.05 summary:

| force_ind | bpp | dLPIPS | dDISTS | dPSNR | y mismatch frac | geom RMS | nonfinite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.035645 | +0.000120 | -0.000875 | +0.012821 | 0.322076 | 0.026904 | 0 |
| 1 | 0.071289 | +0.000020 | +0.001605 | -0.012520 | 0.390544 | 0.017840 | 0 |
| 2 | 0.106934 | +0.000527 | +0.000618 | -0.001865 | 0.431270 | 0.014312 | 0 |
| 3 | 0.142578 | +0.000337 | +0.000432 | -0.010137 | 0.458021 | 0.012379 | 0 |
| 4 | 0.178223 | +0.000018 | +0.000182 | +0.007735 | 0.490017 | 0.010873 | 0 |

Interpretation: this is not a paper-quality improvement yet. Always-on untrained geometry is only mildly positive at the ultra-low-rate force0 point under DISTS and is neutral-to-worse at higher rates. The important positive result is architectural: active HCG geometry can be inserted into EF-LIC's projected RVQ and decoded exactly at unchanged bpp.

The per-image tails show selector headroom. A DISTS-oracle that keeps the original EF-LIC row unless active geometry improves the image gives average dDISTS of `-0.001695`, `-0.000579`, `-0.001017`, `-0.001043`, and `-0.000974` for force0 through force4, with active shares between `0.25` and `0.50`. One image-level side bit on Kodak is only about `0.000002543` bpp. Therefore the next EF-LIC step should be a trained projected-HCG head or teacher-label reliability controller, not an always-on fixed transform.

Artifacts:

- `tools/run_e160_eflic_projected_hcg_smoke.py`
- `experiments/analysis/e160_eflic_projected_hcg_first4.{md,json,csv}`
- `experiments/analysis/e160_eflic_projected_hcg_kodak24_alpha005.{md,json,csv}`
- `experiments/analysis/e160_eflic_projected_hcg_kodak24_alpha005_deep_dive.md`

## E161: EF-LIC Projected-HCG Selector Label Package

Status: Done for teacher-label packaging. I converted the E160 Kodak24 active-vs-baseline deltas into selector labels for future reliability-control experiments. The package records DISTS, LPIPS, strict both-metric, and PSNR labels per image/rate, plus a feature manifest that separates decoder-safe context features from encoder/active diagnostic features.

Summary for `mean/alpha=0.05`:

| force_ind | DISTS active labels | LPIPS active labels | both labels | DISTS oracle dDISTS | mean dDISTS |
|---:|---:|---:|---:|---:|---:|
| 0 | 12/24 | 11/24 | 6/24 | -0.001695 | -0.000875 |
| 1 | 6/24 | 10/24 | 2/24 | -0.000579 | +0.001605 |
| 2 | 10/24 | 10/24 | 4/24 | -0.001017 | +0.000618 |
| 3 | 12/24 | 10/24 | 5/24 | -0.001043 | +0.000432 |
| 4 | 7/24 | 12/24 | 4/24 | -0.000974 | +0.000182 |

Interpretation: the active branch is useful for a subset of images at every rate, but the subset size and metric agreement are rate-dependent. This argues for a rate-conditioned reliability controller or a signaled image-level selector, not a single global always-on transform. A no-side-bit controller must restrict itself to decoder-safe context features; a signaled selector can use richer encoder/active diagnostics with explicit side-bit accounting.

Artifacts:

- `tools/build_e161_eflic_projected_hcg_selector_labels.py`
- `experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005.{md,json,csv}`
- `experiments/analysis/e161_eflic_projected_hcg_selector_labels_alpha005_feature_manifest.csv`

## E162: GLC Pretrained Baseline Reproduction and Feature Audit

Status: Done for the first GLC pretrained baseline reproduction. I downloaded the official `GLC_image.pth.tar` checkpoint from the GitHub release into `third_party/GLC/checkpoints/GLC_image.pth.tar`, verified strict load with `0` missing and `0` unexpected keys, and ran a custom evaluator on Kodak first4 and Kodak24. All runs were pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used.

The evaluator follows the official GLC image path and saves BPP, PSNR, MS-SSIM, LPIPS (`alex`, matching the official GLC metric script), DISTS, runtime, nonfinite flags, and intermediate distributions for `y_ori`, `y`, `z`, `z_hat`, `params`, `prior_q_enc/q_dec`, `prior_scales/means`, `y_res`, `y_q`, `y_hat_prior`, `y_hat_dec`, `scales_hat`, and `z_vq` index usage.

Kodak24 summary:

| q | bpp | bpp_y | bpp_z | PSNR | MS-SSIM | LPIPS(alex) | DISTS | nonfinite | z H | z used | y std | y_q std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.024313 | 0.020895 | 0.003418 | 20.4376 | 0.74874 | 0.19608 | 0.11306 | 0 | 5.5549 | 0.0034 | 0.0856 | 0.0817 |
| 1 | 0.028210 | 0.024792 | 0.003418 | 20.8085 | 0.76828 | 0.18019 | 0.10447 | 0 | 5.8004 | 0.0040 | 0.1003 | 0.0979 |
| 2 | 0.032786 | 0.029368 | 0.003418 | 21.1121 | 0.78218 | 0.16742 | 0.09822 | 0 | 5.9963 | 0.0044 | 0.1208 | 0.1213 |
| 3 | 0.036989 | 0.033571 | 0.003418 | 21.3030 | 0.79139 | 0.16081 | 0.09536 | 0 | 5.9697 | 0.0043 | 0.1383 | 0.1476 |

Interpretation: GLC is now an active parallel main track, not only a future hedge. The result confirms the official pretrained model runs in the expected ultra-low-bitrate range on this machine. Because GLC uses LPIPS-alex in its official metric script while EF-LIC uses LPIPS-vgg in its paper/evaluator path, cross-method LPIPS comparisons must be standardized before making a paper table. Within-GLC HCG-vs-original comparisons should use the official GLC metric first.

Artifacts:

- `tools/run_e162_glc_pretrained_baseline.py`
- `third_party/GLC/checkpoints/GLC_image.pth.tar`
- `experiments/analysis/e162_glc_pretrained_kodak_first4.{md,json,csv}`
- `experiments/analysis/e162_glc_pretrained_kodak24.{md,json,csv}`
- `experiments/analysis/e162_glc_pretrained_kodak_first4_recon/`
- `experiments/analysis/e162_glc_pretrained_kodak24_recon/`

## E163: Dual-Main EF-LIC/GLC Training Strategy

Status: Done for planning and protocol positioning. I wrote a strategy note that treats EF-LIC and GLC as parallel main tracks with different claim types. EF-LIC supports the direct RVQ plug-in claim because its bottleneck is already projected-space RVQ. GLC supports the low-bitrate generative compression extension claim, but its main `y` path is scalar-rounded transform coding, so the HCG integration is a larger codec variant.

Decision on pretrained vs scratch: do not choose only one. Use official pretrained checkpoints first to reproduce protocol, metrics, padding, bit accounting, and intermediate features. Use frozen/pretrained plug-in smoke to verify bitstream validity. For conference-grade claims, run matched retraining or matched fine-tuning: EF-LIC original vs EF-LIC+HCG under the same reconstructed EF-LIC recipe, and GLC Stage II/III baseline vs GLC+HCG under the same OpenImages/VQGAN-latent recipe. Full GLC Stage I VQ-VAE scratch training is valuable but should not block the first HCG plug-in evidence.

Artifact:

- `experiments/analysis/e163_dual_main_training_strategy.md`



## E164: GLC Official-Test vs Instrumented-Path Identity Check

Status: Done. I added and ran a GLC identity/provenance check that compares the official `GLC_Image.test()` output against the E162 instrumented feature-audit path on the same checkpoint, images, q-indexes, padding, and GPU. This is a guardrail before HCG insertion: if this check failed, later feature-distribution and metric deltas could be evaluator artifacts rather than codec effects.

The first4 smoke and full Kodak24 runs both passed on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`; device 1 was not used. Across Kodak24, all 96 rows (`24` images x `q=0..3`) had `max_abs_xhat_diff=0`, `mean_abs_xhat_diff=0`, `bit_y_diff=0`, `bit_z_diff=0`, `bit_total_diff=0`, and `nonfinite=0`. This means the E162 GLC baseline metrics and intermediate distributions are exactly aligned with the official test path.

Decision: GLC is now ready for HCG insertion experiments on a trusted evaluation path. The next GLC branch should start with the least invasive explicit-VQ test around `z_vq`, while designing the more paper-relevant `y`-path HCG-RVQ replacement for `forward_four_part_prior()` in parallel.

Artifacts:

- `tools/run_e164_glc_instrumented_identity_check.py`
- `experiments/analysis/e164_glc_instrumented_identity_first4.{md,json,csv}`
- `experiments/analysis/e164_glc_instrumented_identity_kodak24.{md,json,csv}`


## E165: GLC HCG-RVQ Insertion Blueprint

Status: Done for GLC insertion planning. I wrote a concrete GLC HCG-RVQ blueprint after E162/E164 locked the official evaluation path. The main conclusion is that GLC should remain a parallel main track, but the insertion points support different claims. `z_vq` is a useful low-risk explicit-VQ smoke branch, while the paper-relevant HCG-RVQ integration is the main `y` path inside `forward_four_part_prior()`, where `z_hat`, `common_params`, `means/scales`, `y_hat_so_far`, q-index, and masks are decoder-safe context for generating local quantizer geometry.

The blueprint also fixes the training policy: official pretrained reproduction first, identity modularization and frozen smoke second, matched Stage II/III fine-tuning or retraining third. Full GLC Stage I VQ-VAE scratch training is a valuable robustness step if compute permits, but it should not block the first conference-grade HCG plug-in evidence because HCG modifies the compression-side latent coding path.

Artifact:

- `experiments/analysis/e165_glc_hcg_insertion_blueprint.md`


## E166: GLC y-Prior HCG-Ready Identity Wrapper

Status: Done. I implemented a HCG-ready identity wrapper for GLC `forward_four_part_prior()`, the main paper-relevant `y` residual coding path identified in E165. The wrapper preserves the original scalar quantization behavior exactly, but exposes per-part insertion hooks and records residual, quantized residual, scale, mean, and combined `y_hat` distributions.

The first4 smoke and full Kodak24 runs both passed on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`; device 1 was not used. Across Kodak24 and `q=0..3`, the wrapper has `max_abs_xhat_diff=0`, `max_abs_ref_latent_diff=0`, `bit_y_diff=0`, `bit_z_diff=0`, `bit_total_diff=0`, and `nonfinite=0`. This means GLC main `y` path can now be modified through a trusted identity boundary, not only through the easier `z_vq` branch.

The recorded residual distributions are useful for HCG-RVQ design. Combined residual std increases with q: `0.0765`, `0.0930`, `0.1171`, and `0.1440` for q0-q3. Per-part residual std is uneven and decreases from part0 to part3, so RVQ group/stage settings should be checked by part rather than assumed uniform.

Artifacts:

- `tools/run_e166_glc_y_prior_identity_wrapper.py`
- `experiments/analysis/e166_glc_y_prior_identity_first4.{md,json,csv}`
- `experiments/analysis/e166_glc_y_prior_identity_kodak24.{md,json,csv}`

## E167: HCG Prior-Results Transfer Audit

Status: Done for prior-result consolidation. I audited the short-cycle HCG-RVQ evidence and mapped it into EF-LIC/GLC transfer rules. The central conclusion is that we should port the state-preserving reliability mechanism, not raw always-on geometry or continuous posthoc gate shrinkage. E150/E151 show the strongest prior deployment lesson: the HCS/HCG state-preserving branch reaches RD `1.170368` vs HCS `1.200992` and fixed HCG `1.182335`, while continuous suppression variants such as E145/E147/E149 damage RD.

For EF-LIC, this means preserving the original projected-RVQ state and adding a projected-HCG active state with rate-conditioned selector/side-bit accounting. For GLC, this means using the E166 `y`-path wrapper and introducing HCS-like fallback plus HCG-RVQ active state together, rather than testing only a raw active branch. `z_vq` remains a useful smoke branch but not the main novelty claim.

Artifact:

- `experiments/analysis/e167_hcg_prior_results_transfer_audit.md`


## E168: GLC y-Residual Distribution Audit

Status: Done for the first GLC `y`-path distribution audit. I added a read-only analyzer at the E166 identity-preserving `forward_four_part_prior()` boundary and ran it on Kodak24 for q0-q3 with `CUDA_VISIBLE_DEVICES=0` / `cuda:0`. Device 1 was not used. The run produced `384` per-image/part rows and `6144` per-image/part/group rows, with no nonfinite symptoms in the live run.

The main finding is that the active residual distribution is sparse and heavy-tailed. Per-part active residual std grows with q and is larger in earlier masked parts, but the p95 residual is small while p99 grows strongly. For example part0 goes from `res_abs_p95=0.05010`, `res_abs_p99=0.32966` at q0 to `0.04545`, `0.57547` at q3. The scalar residual symbols are mostly zero: q=0 fraction is about `0.989` to `0.996`, with symbol entropy only `0.044` to `0.109` bits/symbol in the part averages.

The group audit is the most actionable result for HCG-RVQ transfer. A few channel groups dominate the tails, especially groups `1`, `7`, `10`, and `15` in early parts; at q3 part0 group1 has `res_abs_p95=0.33593` and `res_abs_p99=1.87256`, while many groups are essentially all-zero after scalar rounding. This means the first GLC HCG-RVQ branch should be part/group aware and state-preserving: preserve scalar/identity fallback for the easy zero-heavy regions, and focus HCG-RVQ capacity on the heavy-tail groups where scalar rounding is actually doing work.

Artifacts:

- `tools/analyze_e168_glc_y_res_distribution.py`
- `experiments/analysis/e168_glc_y_res_distribution_first4.{md,json,csv}`
- `experiments/analysis/e168_glc_y_res_distribution_kodak24.md`
- `experiments/analysis/e168_glc_y_res_distribution_kodak24_*`


## E169: GLC y-Path Tail Branch Identity Scaffold

Status: Done for a decoder-known state-preserving branch scaffold on the GLC `y` path. I used the E168 heavy-tail groups as a static active subset: active parts `[0, 1, 2]`, active groups `[1, 7, 10, 15]`, group size `16`. Both active and fallback states still use the original scalar rounding, so this is an identity/provenance and coverage experiment rather than a quality-improvement row.

The first4 and full Kodak24 runs both passed on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`; device 1 was not used. Across q0-q3, max `x_hat`, ref-latent, and bit differences were all `0`, and nonfinite rows were `0`.

The coverage result is strong. The selected subset covers only `0.1875` of valid residual positions and `0.0469` of all latent elements, but it captures most of the residual energy and nonzero scalar symbols:

| q | residual energy covered | qerr energy covered | nonzero symbols covered | active nonzero rate | inactive nonzero rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7949 | 0.6507 | 0.8195 | 0.0249 | 0.0013 |
| 1 | 0.8074 | 0.6533 | 0.8052 | 0.0299 | 0.0017 |
| 2 | 0.8151 | 0.6612 | 0.7920 | 0.0353 | 0.0022 |
| 3 | 0.8231 | 0.6628 | 0.7856 | 0.0396 | 0.0025 |

Interpretation: E168 was not just descriptive. It found a compact, decoder-known branch subset that captures the majority of difficult residual mass. The next GLC implementation should keep this state-preserving scaffold and replace only the active state with an HCG-RVQ or HCS-RVQ branch first, while preserving scalar fallback exactly.

Artifacts:

- `tools/run_e169_glc_y_tail_branch_identity.py`
- `experiments/analysis/e169_glc_y_tail_branch_identity_first4.{md,json,csv}`
- `experiments/analysis/e169_glc_y_tail_branch_identity_kodak24.{md,json,csv}`

## E170/E171: GLC Tail VQ/RVQ Residual Probes

Status: Done for diagnostic codebook/headroom analysis. I ran leave-one-image-out residual VQ/RVQ probes on the E169 active subset with `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. These probes do not modify image reconstruction yet and are not paper-quality codec rows. They answer the design question: should the GLC active state use a shared codebook, part/group codebooks, one-stage VQ, or multi-stage RVQ?

E170 shows that a shared codebook is the wrong direction for GLC tail residuals. Even shared K=8 is worse than scalar rounding at every q: MSE ratios are `1.6410`, `2.4966`, `4.2712`, and `6.4520` for q0-q3. Part/group conditioning is necessary. Part/group K=4 improves q0 and nearly breaks even at q1, while part/group K=8 improves residual MSE for all q (`0.2458`, `0.3500`, `0.5402`, `0.8122` ratios). The cost is rate: empirical VQ bpp is higher than scalar active bpp, so index-prior/entropy accounting is required.

E171 checks RVQ stages at matched fixed rates. K=2/L2 has the same fixed bpp as K=4/L1, and K=2/L3 matches K=8/L1. K=2 stages help q0 but are not enough for q2/q3: K=2/L3 ratios are `0.3857`, `0.5801`, `0.9347`, and `1.4425` for q0-q3. This means RVQ stages matter, but high-q tail control still needs larger/local capacity or HCG geometry plus an index prior.

Decision: the next GLC active branch should be state-preserving, sparse, part/group conditioned, and index-prior aware. The fallback remains original scalar rounding. The active state should first target the E169 subset, then add HCS/HCG conditioning and rate-aware codebook/stage selection.

Artifacts:

- `tools/run_e170_glc_tail_vq_rate_distortion_probe.py`
- `experiments/analysis/e170_glc_tail_vq_probe_kodak24.{md,json}`
- `experiments/analysis/e170_glc_tail_vq_probe_kodak24_*`
- `experiments/analysis/e171_glc_tail_rvq_stage_probe_k2_kodak24.{md,json}`
- `experiments/analysis/e171_glc_tail_rvq_stage_probe_k2_kodak24_*`
- `experiments/analysis/e172_glc_tail_vq_rvq_design_decision.md`

## E173/E174: GLC Integrated Tail VQ Diagnostic Branch

Status: Done for the first non-identity GLC `y`-path active branch diagnostic. I added `tools/run_e173_glc_tail_vq_integrated_probe.py`, which keeps the E166/E169 trusted GLC boundary, preserves original scalar fallback, and replaces only the sparse active subset with leave-one-image-out part/group VQ codebooks. Runs were fixed to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. All K=8 and K=4 Kodak24 rows had `nonfinite=0`.

K=8 shows real distortion-side headroom. It improves PSNR on average at all q (`+0.5371`, `+0.3692`, `+0.1936`, `+0.1095`) and improves active residual MSE at all q (ratios `0.2573`, `0.3720`, `0.5789`, `0.8741`). It also improves MS-SSIM and mostly LPIPS at q0/q1. However, it increases empirical hybrid bpp by `+0.015745`, `+0.012943`, `+0.009558`, and `+0.006343`, and DISTS worsens at nearly every image/rate.

K=4 is lower-rate but underpowered. It helps q0 PSNR slightly, then becomes marginal or harmful at higher q. Active MSE ratios are `0.6566`, `0.9795`, `1.5905`, and `2.4489`; q2/q3 are not valid active quantization settings despite lower estimated rate.

Decision: the GLC branch should not become a fixed residual-MSE VQ. The result supports the HCG-RVQ thesis that the active subset matters, but the final branch must be trainable, decoder/perceptual-aware, index-prior-aware, and reliability-controlled. The next GLC step is a trainable HCS/HCG active state with the original scalar fallback preserved.

Artifacts:

- `tools/run_e173_glc_tail_vq_integrated_probe.py`
- `experiments/analysis/e173_glc_tail_vq_integrated_k8_kodak24.{md,json,csv}`
- `experiments/analysis/e173_glc_tail_vq_integrated_k4_kodak24.{md,json,csv}`
- `experiments/analysis/e173_glc_tail_vq_integrated_k8_first4_recon.md`
- `experiments/analysis/e173_glc_tail_vq_integrated_k8_first4_recon_images/`
- `experiments/analysis/e174_glc_integrated_tail_vq_diagnostic_analysis.md`

## E175/E176: GLC Decoder-Aware Trainable Tail VQ Diagnostic

Status: Done for the first trainable active-branch diagnostic on the GLC `y` path. I added `tools/run_e175_glc_decoder_aware_tail_vq_train.py`, which freezes pretrained GLC and optimizes only the sparse active part/group VQ codebooks through the downstream GLC decoder/generator. This is an upper-bound diagnostic on a tiny Kodak subset, not a paper-quality training protocol.

The implementation required two training-specific safeguards: instantiate GLC with `inplace=False` for backward through the frozen decoder path, and clear the GLC mask cache after `torch.inference_mode()` residual collection. All E175 runs were fixed to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used, and all reported rows had `nonfinite=0`.

Main diagnostic results:

| run | q | images | branch | DISTS | LPIPS | PSNR | empirical dbpp |
|---|---:|---:|---|---:|---:|---:|---:|
| one-image DISTS40 | 0 | 1 | baseline | 0.12375 | 0.23785 | 18.9641 | 0.000000 |
| one-image DISTS40 | 0 | 1 | init VQ | 0.16740 | 0.22776 | 19.5777 | +0.014520 |
| one-image DISTS40 | 0 | 1 | trained VQ | 0.13440 | 0.21694 | 19.3860 | +0.014425 |
| two-image DISTS20 | 0 | 2 | baseline | 0.12772 | 0.21627 | 21.0055 | 0.000000 |
| two-image DISTS20 | 0 | 2 | init VQ | 0.15863 | 0.20833 | 21.7005 | +0.016076 |
| two-image DISTS20 | 0 | 2 | trained VQ | 0.13823 | 0.20548 | 21.5271 | +0.015736 |
| two-image DISTS20 | 3 | 2 | baseline | 0.10424 | 0.17463 | 21.9076 | 0.000000 |
| two-image DISTS20 | 3 | 2 | init VQ | 0.11542 | 0.17890 | 22.0244 | +0.005804 |
| two-image DISTS20 | 3 | 2 | trained VQ | 0.10963 | 0.17653 | 22.0057 | +0.005682 |

Interpretation: decoder-aware training can substantially recover the DISTS damage caused by fixed residual-MSE VQ, and even produces a per-image DISTS win on `kodim02` at q0. However, average DISTS still trails the original baseline and the rate increase remains large. The strongest mechanism lesson is that active residual MSE and perceptual quality diverge: q0 one-image active MSE ratio worsens from `0.2939` to `0.6129` while DISTS improves from `0.16740` to `0.13440`. The final GLC branch must therefore be trainable, perceptual/decoder-aware, bit-aware, and reliability-controlled.

Artifacts:

- `tools/run_e175_glc_decoder_aware_tail_vq_train.py`
- `experiments/analysis/e175_glc_decoder_aware_tail_vq_train_q0_oneimg_smoke.{md,json,csv}`
- `experiments/analysis/e175_glc_decoder_aware_tail_vq_train_q0_oneimg_dists_smoke.{md,json,csv}`
- `experiments/analysis/e175_glc_decoder_aware_tail_vq_train_q0_oneimg_dists40.{md,json,csv}`
- `experiments/analysis/e175_glc_decoder_aware_tail_vq_train_q0_twoimg_dists20.{md,json,csv}`
- `experiments/analysis/e175_glc_decoder_aware_tail_vq_train_q3_twoimg_dists20.{md,json,csv}`
- `experiments/analysis/e176_glc_decoder_aware_tail_vq_analysis.md`

## E177/E178: GLC Decoder-Aware Tail VQ Split-Train Diagnostic

Status: Done for a small train/eval split diagnostic. I added `tools/run_e177_glc_decoder_aware_tail_vq_split_train.py`, which trains only the sparse active part/group VQ codebooks on OpenImages 256 crops and evaluates on separate Kodak images. This moves beyond the E175 Kodak-overfit upper bound, but it is still not full matched GLC fine-tuning.

All runs used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. Reported rows had `nonfinite=0`.

Main results:

| run | q | train/eval | branch | empirical dbpp | PSNR | MS-SSIM | LPIPS | DISTS | active MSE ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| OI2 -> Kodak2 | 0 | 2 crops / 2 images | baseline | 0.000000 | 21.0055 | 0.71950 | 0.21627 | 0.12772 | - |
| OI2 -> Kodak2 | 0 | 2 crops / 2 images | init eval | +0.014276 | 21.5988 | 0.74667 | 0.20790 | 0.15895 | 0.2793 |
| OI2 -> Kodak2 | 0 | 2 crops / 2 images | trained eval | +0.014286 | 21.6176 | 0.74810 | 0.20591 | 0.15102 | 0.2948 |
| OI2 -> Kodak2 | 3 | 2 crops / 2 images | baseline | 0.000000 | 21.9076 | 0.76692 | 0.17463 | 0.10424 | - |
| OI2 -> Kodak2 | 3 | 2 crops / 2 images | init eval | +0.005362 | 22.0632 | 0.76657 | 0.17697 | 0.11373 | 0.9312 |
| OI2 -> Kodak2 | 3 | 2 crops / 2 images | trained eval | +0.005325 | 22.0742 | 0.76671 | 0.17554 | 0.11099 | 0.9564 |
| OI8 -> Kodak4 | 0 | 8 crops / 4 images | baseline | 0.000000 | 22.3664 | 0.77170 | 0.19076 | 0.11448 | - |
| OI8 -> Kodak4 | 0 | 8 crops / 4 images | init eval | +0.014260 | 22.9522 | 0.79583 | 0.18049 | 0.12619 | 0.2564 |
| OI8 -> Kodak4 | 0 | 8 crops / 4 images | trained eval | +0.014176 | 22.8546 | 0.79531 | 0.17860 | 0.12227 | 0.2931 |

The q0 OI8 -> Kodak4 run is the most informative. PSNR, MS-SSIM, and LPIPS beat the original GLC baseline on all four evaluated Kodak images. DISTS wins increase from `1/4` with the fixed init codebook to `2/4` after decoder-aware training. Average DISTS still trails the baseline (`0.12227` vs `0.11448`), and bpp is still too high (`+0.014176`).

Decision: the GLC track remains worth pursuing, but always-on K=8/L1 codebooks should not be scaled up as the final method. The next implementation needs scalar fallback, reliability selection, q-dependent stage/capacity, index-prior or bit-aware training, and HCG-style local shift/scale/geometry conditioning.

Artifacts:

- `tools/run_e177_glc_decoder_aware_tail_vq_split_train.py`
- `experiments/analysis/e177_glc_decoder_aware_tail_vq_split_train_q0_oi2_kodak2_smoke.{md,json,csv}`
- `experiments/analysis/e177_glc_decoder_aware_tail_vq_split_train_q3_oi2_kodak2_smoke.{md,json,csv}`
- `experiments/analysis/e177_glc_decoder_aware_tail_vq_split_train_q0_oi8_kodak4.{md,json,csv}`
- `experiments/analysis/e178_glc_split_train_tail_vq_analysis.md`

## E179/E181/E183: GLC Selector, Larger Split, and Loss Check

Status: Done for the next GLC reliability-control diagnostic. I added `tools/analyze_e179_glc_branch_selector.py` and used it on the E177 split-train rows, then ran a larger q0 OpenImages16 -> Kodak8 split-train diagnostic. The first attempt OOMed because the training loop retained every decoder graph before backward; I fixed `tools/run_e177_glc_decoder_aware_tail_vq_split_train.py` to backpropagate per train crop. The rerun completed on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`; device 1 was not used, and all rows had `nonfinite=0`.

E181 trained_eval shows a clear metric split: PSNR, MS-SSIM, and LPIPS improve on all 8 Kodak images, but DISTS improves on only `1/8` and worsens on average (`0.11899 -> 0.13134`) with dbpp `+0.014548`. The DISTS oracle selector uses the branch on only `12.5%` of images and gives only a tiny gain (`0.11876`, dDISTS `-0.000231`, dbpp `+0.001903`), while LOOCV thresholds still worsen DISTS.

E183 repeats the same split with DISTS-only loss (`mse_weight=0`). It slightly improves average branch DISTS (`0.13134 -> 0.13106`) but still wins only `1/8` images. This means the DISTS failure is not just caused by the auxiliary MSE term. The current fixed active codebooks are useful for LPIPS/PSNR but are not DISTS-safe; the next GLC method needs scalar fallback, deployable reliability selection, bit-aware/index-prior training, and HCG-conditioned local geometry rather than always-on K8/L1 codebooks.

Artifact: `experiments/analysis/e182_e183_glc_selector_loss_metric_protocol_analysis.md`.

## E184: EF-LIC Projected-HCG Selector CV Audit

Status: Done for the first reliability-control audit on the EF-LIC projected-HCG smoke labels. This is a controlled diagnostic, not a paper-quality codec row. It reuses E160/E161 Kodak24 per-image active-vs-baseline rows and separates two feature regimes:

- `decoder_safe_context`: decoder-reproducible mean/scale/z context features, no side bit.
- `encoder_active_diagnostic`: encoder/active HCG diagnostics, counted with one signaled image-level bit and treated as an upper-bound/failure-analysis feature set.

The main DISTS-target result is selective. Always-active projected HCG only helps force0 on average (`dDISTS=-0.000875`) and hurts force1-4. Metric oracle rows show headroom for all forces (`dDISTS` from `-0.000579` to `-0.001695`), but this is non-deployable. The deployable decoder-safe LOOCV threshold remains positive only for force0 (`dDISTS=-0.000592`, branch share `0.667`, 8/24 DISTS wins). Decoder-safe LOOCV fails for force1-4 (`dDISTS` from `+0.000183` to `+0.000762`). Encoder/active diagnostic LOOCV helps force0 (`-0.001029`) and barely force1/force4, but those features are not a no-side-bit paper claim without a decoder-side proxy.

The LPIPS-target audit does not produce a stable decoder-safe LPIPS controller: decoder-safe LOOCV worsens LPIPS for all forces. This reinforces the earlier GLC result that metric choice and reliability control matter; active HCG should not be scaled as an always-on branch.

Decision: for EF-LIC, the next implementation target should be a conservative force0-style weak geometry branch with decoder-safe reliability control and scalar/RVQ fallback. Force1-4 remain ablations or diagnostic upper bounds. This aligns with the HCG-RVQ thesis, but the result is still short-cycle evidence; final claims require matched EF-LIC training/evaluation or an official-training-compatible reproduction.

Artifacts:

- `tools/analyze_e184_eflic_selector_cv.py`
- `experiments/analysis/e184_eflic_projected_hcg_selector_cv_dists.{md,json,csv}`
- `experiments/analysis/e184_eflic_projected_hcg_selector_cv_lpips.{md,json,csv}`

## E185/E186: EF-LIC Selector Provenance And Direct Predecision Probe

Status: Done for the deployability audit after E184. E185 separates selector features by when the decoder can actually know them:

- `global_predecision_context`: z context plus slice0 mean/scale, available before choosing a whole-image active/fallback branch.
- `sequential_context`: later slice mean/scale features, decoder-reproducible only inside a sequential per-slice policy.
- `legacy_decoder_safe_context`: the broader E184 class, kept to expose optimistic later-slice context.
- `encoder_active_diagnostic`: active/encoder-only diagnostics, not deployable without signaling or a decoder-side proxy.

This corrected an important nuance: E184's best legacy rule for force0 used `slice2_mean_max`, which is decoder-reproducible during a specific sequential path but is too strong for a no-side-bit whole-image predecision. Under the stricter `global_predecision_context`, force0 still keeps useful signal. DISTS-target LOOCV gives `dDISTS=-0.000641`, branch share `0.833`, and 10/24 DISTS wins. LPIPS-target LOOCV on the same global feature class gives `dDISTS=-0.000761` and `dLPIPS=-0.000281`.

E186 then implements the fixed force0 global predecision rule directly in the EF-LIC evaluation path: `slice0_mean_abs_mean <= 0.455596`, `alpha=0.05`, `direction_source=mean`. On Kodak24, encoder/decoder selector decisions match exactly (`24/24`), max selector value difference is `0`, max active decode diff is `0`, and `nonfinite=0`. The selected branch uses active HCG on 20/24 images with the same bpp as the base force0 row (`0.035645`). It improves DISTS and LPIPS over baseline:

| branch | bpp | PSNR | LPIPS | DISTS | dPSNR | dLPIPS | dDISTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.035645 | 21.6693 | 0.27228 | 0.09227 | +0.0000 | +0.000000 | +0.000000 |
| always active | 0.035645 | 21.6662 | 0.27240 | 0.09139 | +0.0128 | +0.000120 | -0.000875 |
| selected | 0.035645 | 21.6667 | 0.27217 | 0.09103 | +0.0057 | -0.000112 | -0.001238 |

Decision: EF-LIC now has direct implementation evidence for a no-side-bit, decoder-reproducible global reliability selector. This remains a Kodak-fitted short-cycle probe, not a final paper claim. The next paper-facing step is to fit the threshold/controller on an independent split, or train a tiny decoder-side reliability head under an EF-LIC-compatible protocol, then evaluate on full Kodak/CLIC using the original paper metrics.

Artifacts:

- `tools/analyze_e185_eflic_selector_provenance.py`
- `tools/run_e186_eflic_global_predecision_selector_probe.py`
- `experiments/analysis/e185_eflic_selector_provenance_dists.{md,json,csv}`
- `experiments/analysis/e185_eflic_selector_provenance_lpips.{md,json,csv}`
- `experiments/analysis/e186_eflic_force0_global_predecision_selector_first4.{md,json,csv}`
- `experiments/analysis/e186_eflic_force0_global_predecision_selector_kodak24.{md,json,csv}`

## E187/E188: EF-LIC Split-Fit Selector And LPIPS-Balanced Direct Probe

Status: Done for the next EF-LIC reliability-control audit. E187 adds `tools/analyze_e187_eflic_selector_splitfit.py`, which fits a scalar selector threshold on one Kodak split and evaluates it on the held-out split using only `global_predecision_context` features. This remains a small Kodak audit, not final paper evidence, but it checks whether the E186 selector is only a same-table artifact.

For DISTS-target split fitting, the held-out selector improves DISTS in `3/4` splits. The held-out average is `dDISTS=-0.000402`, `dLPIPS=+0.000206`, `dPSNR=+0.007820`, with average branch share `0.6875`. This confirms a real DISTS signal, but the learned feature/rule changes across splits and does not consistently beat always-active DISTS (`-0.000875` on the same held-out split aggregation).

For LPIPS-target split fitting, the held-out selector is more balanced: `dDISTS=-0.000677`, `dLPIPS=-0.000128`, `dPSNR=+0.006175`, with average branch share `0.4792`. It improves both DISTS and LPIPS in `3/4` held-out splits. This suggests the next controller should not optimize DISTS alone; a balanced perceptual objective or multi-metric reliability label is safer.

E188 then runs the LPIPS-balanced full-Kodak rule from E185 directly in the EF-LIC forward path: `slice0_mean_min >= -10.7448`, `alpha=0.05`, force0. It uses `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; device 1 was not used. Encoder/decoder selector decisions match exactly (`24/24`), max selector abs diff is `0`, max active decode diff is `0`, and nonfinite rows are `0`. The selected branch keeps bpp unchanged and improves both DISTS and LPIPS:

| branch | bpp | PSNR | LPIPS | DISTS | dPSNR | dLPIPS | dDISTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.035645 | 21.6693 | 0.27228 | 0.09227 | +0.0000 | +0.000000 | +0.000000 |
| active | 0.035645 | 21.6662 | 0.27240 | 0.09139 | +0.0128 | +0.000120 | -0.000875 |
| selected | 0.035645 | 21.6604 | 0.27181 | 0.09140 | -0.0043 | -0.000468 | -0.000870 |

Decision: EF-LIC now has two direct no-side-bit selector probes. E186 is stronger for DISTS (`dDISTS=-0.001238`) and E188 is safer for LPIPS (`dLPIPS=-0.000468`) while still matching always-active DISTS. The paper-facing next step is to replace Kodak-fit scalar rules with an independent/OpenImages-fitted or learned decoder-side controller, then evaluate under the EF-LIC paper protocol. The current result is a controlled implementation and mechanism result, not yet a final RD claim.

Artifacts:

- `tools/analyze_e187_eflic_selector_splitfit.py`
- `experiments/analysis/e187_eflic_force0_global_selector_splitfit_dists.{md,json,csv}`
- `experiments/analysis/e187_eflic_force0_global_selector_splitfit_lpips.{md,json,csv}`
- `experiments/analysis/e188_eflic_force0_global_predecision_selector_lpipsrule_kodak24.{md,json,csv}`

## E189: EF-LIC Independent-Selector Protocol Tooling

Status: Done for protocol tooling and Kodak diagnostic smoke. I added `tools/fit_e189_eflic_global_selector_rule.py`, which fits a scalar selector rule from an E161-style EF-LIC active-branch label table using a specified feature provenance class. The main purpose is to fit on an independent validation image directory and then run direct held-out evaluation through `tools/run_e186_eflic_global_predecision_selector_probe.py`.

Kodak diagnostic smoke reproduces the two existing global-predecision rules:

| target | fitted rule | branch share | dDISTS | dLPIPS | dPSNR |
|---|---|---:|---:|---:|---:|
| DISTS | `slice0_mean_abs_mean <= 0.455595642328` | 0.833 | -0.001238 | -0.000112 | +0.005692 |
| LPIPS | `slice0_mean_min >= -10.7447786331` | 0.625 | -0.000870 | -0.000468 | -0.004288 |

The LPIPS-target rule remains the safer default for the next independent validation experiment because it improves both LPIPS and DISTS on the Kodak diagnostic table and was more balanced in E187 split-fit. The DISTS-target rule remains useful as a DISTS-oriented ablation.

I also added `docs/vq_lic_full_eval_protocol.md`, which records exact EF-LIC commands for: active-label generation on a fit split, E161 label/manifest creation, E189 rule fitting, and direct held-out evaluation. It also records the current GLC protocol decision: sparse active state with scalar fallback, index prior, and decoder/perceptual-aware training rather than dense always-on VQ.

Artifacts:

- `tools/fit_e189_eflic_global_selector_rule.py`
- `experiments/analysis/e189_eflic_force0_global_selector_rulefit_lpips_kodakdiag.{md,json,csv}`
- `experiments/analysis/e189_eflic_force0_global_selector_rulefit_dists_kodakdiag.{md,json,csv}`
- `docs/vq_lic_full_eval_protocol.md`

## E190: EF-LIC Multi-Objective Selector Strengthening

Status: Done for the next selector-strengthening audit. I added `tools/analyze_e190_eflic_multiobjective_selector.py`, which searches decoder-side scalar global-predecision rules under a multi-objective score instead of a single DISTS or LPIPS target. The search uses the same force0 projected-HCG label table and only `global_predecision_context` features, so the selected rule remains no-side-bit and decoder-reproducible in the same sense as E186/E188.

The key result is that the safer E188 rule is rediscovered by multi-objective search. With weights `DISTS=1.0`, `LPIPS=3.0`, `PSNR=0.0`, and positive-metric penalty `20.0`, the best full-Kodak rule is again `slice0_mean_min >= -10.7447786`. It keeps bpp unchanged and gives the same direct-probe diagnostic behavior as E188: `dDISTS=-0.000870`, `dLPIPS=-0.000468`, `dPSNR=-0.004288`, branch share `0.625`.

The anti-overfit checks are the main reason to prefer this multi-objective setting over DISTS-only control. In LOOCV it gives `dDISTS=-0.000598` and `dLPIPS=-0.000393`. Across the four split-fit/eval audits, the held-out averages are `dDISTS=-0.000282`, `dLPIPS=-0.000171`, `dPSNR=+0.005528`, with both DISTS and LPIPS improving in `4/4` splits. By contrast, the DISTS-heavy setting (`DISTS=2`, `LPIPS=1`) reaches a stronger same-table DISTS row (`dDISTS=-0.001238`) but has positive LPIPS in LOOCV/split evaluations.

Decision: the EF-LIC main controller should move from single-metric threshold fitting to a multi-objective reliability objective. The paper-facing default is now the E190 `DISTS=1, LPIPS=3` setting because it strengthens HCG-RVQ performance without relying on a DISTS-only metric tradeoff. E186 remains a DISTS-oriented ablation; E188/E190 are the safer mainline. This is still Kodak diagnostic evidence, so the next required experiment is to fit the same multi-objective rule on an independent fit split and evaluate it directly on held-out Kodak/Tecnick/DIV2K/CLIC.

Artifacts:

- `tools/analyze_e190_eflic_multiobjective_selector.py`
- `experiments/analysis/e190_eflic_force0_global_selector_multiobj_d1_l3.{md,json,csv}`
- `experiments/analysis/e190_eflic_force0_global_selector_multiobj_d05_l2.{md,json,csv}`
- `experiments/analysis/e190_eflic_force0_global_selector_multiobj_d1_l2.{md,json,csv}`
- `experiments/analysis/e190_eflic_force0_global_selector_multiobj_d2_l1.{md,json,csv}`

## E191/E192: EF-LIC Selector Failure Modes And Two-Stage Headroom

Status: Done for per-image failure analysis and a two-stage selector headroom audit. I added `tools/analyze_e191_eflic_selector_failure_modes.py` and `tools/analyze_e192_eflic_two_stage_selector.py` to explain the E190 controller beyond average metric rows.

E191 shows why the E190 multi-objective rule is the current mainline. With `slice0_mean_min >= -10.7447786`, `DISTS=1`, `LPIPS=3`, and force0 projected-HCG, the selected policy activates on `15/24` Kodak images and averages `dDISTS=-0.000870`, `dLPIPS=-0.000468`. Under the weighted objective, it has `10` selected-good images, `5` selected-bad images, and `0` missed-good images. The DISTS-oriented E186 rule activates on `20/24` images and reaches stronger DISTS (`dDISTS=-0.001238`), but it has `10` selected-bad images under the same multi-objective criterion. This explains the E186/E190 tradeoff: E186 is more aggressive; E190 is better reliability control.

The feature separation also supports the mechanism. The E190 selected branch differs most strongly on `slice0_mean_min` (selected mean `-8.425329` vs fallback `-14.328183`, standardized gap `+1.611`, correlation with active score `-0.442`). This means the controller is not using an arbitrary metric artifact; it is using an early hyperprior/slice0 geometry statistic that is available before the whole-image branch decision.

E192 tests whether a second scalar condition can strengthen E190. Same-table search finds a better-looking conjunctive rule: `slice0_mean_min >= -10.7447786 AND slice0_mean_abs_mean <= 0.455595642`, improving the Kodak diagnostic to `dDISTS=-0.000987`, `dLPIPS=-0.000577`. But anti-overfit checks do not support making it the main rule. LOOCV drops to `dDISTS=-0.000607`, `dLPIPS=-0.000259`, and split-fit/eval rules move among `z_index_entropy`, `slice0_scale_abs_mean`, and `z_hat_abs_mean`. Split-eval averages are weaker than the E190 primary rule.

Decision: keep E190 primary as the paper-facing default for independent fit/eval. Use E191 selected-bad rows and E192 secondary features as targets for a learned decoder-side reliability head, not as a hand-tuned two-stage paper rule. This strengthens the research story: HCG-RVQ needs reliability control, and the current simple controller has explainable headroom without yet over-claiming a fragile second threshold.

Artifacts:

- `tools/analyze_e191_eflic_selector_failure_modes.py`
- `tools/analyze_e192_eflic_two_stage_selector.py`
- `experiments/analysis/e191_eflic_force0_e190_d1_l3_rule_failure_modes.{md,json}`
- `experiments/analysis/e191_eflic_force0_e190_d1_l3_rule_failure_modes_{per_image,groups,feature_separation}.csv`
- `experiments/analysis/e191_eflic_force0_e186_dists_rule_failure_modes_under_d1_l3.{md,json}`
- `experiments/analysis/e192_eflic_force0_e190_primary_plus_second_global_d1_l3.{md,json,csv}`

## E193: EF-LIC Learned Reliability Head Overfit Audit

Status: Done for the first decoder-side learned controller diagnostic. I added tools/analyze_e193_eflic_reliability_head.py, which trains a tiny L2-regularized logistic reliability head using only global_predecision_context features. This keeps the same no-side-bit premise as E190: z statistics plus slice0 mean/scale are available before choosing the whole-image active/fallback branch.

The same-table result shows real headroom. Under the E190 multi-objective weights (DISTS=1, LPIPS=3, positive penalty=20), the E190 primary rule has branch share 0.625, dDISTS=-0.000870, dLPIPS=-0.000468, and 5 selected-bad images. The learned head lowers the branch share to 0.500, keeps all 10 beneficial active cases, reduces selected-bad to 2, and improves the score to -0.002506 with dDISTS=-0.000881 and dLPIPS=-0.000542. The top coefficients are still interpretable: slice0_mean_min is the largest weight, followed by z_hat_abs_mean, z_hat_std, and z_hat_rms.

The anti-overfit result is the important decision point. LOOCV drops to dDISTS=-0.000445 and dLPIPS=-0.000006, with only 5/10 beneficial cases selected. Split-fit/eval is unstable and often worse than E190 primary; increasing L2 from 0.25 to 5.0 does not fix this. Therefore, the learned head is not the current paper-facing default. It is evidence that a learned reliability controller can filter false positives, but it needs an independent fit split or more training labels before being integrated into EF-LIC as the main controller.

Decision: keep E190 primary as the default for the next held-out/full evaluation. Use E193 to justify the next implementation stage: train a decoder-side reliability head on non-Kodak fit data, then run direct EF-LIC evaluation on held-out Kodak/Tecnick/CLIC with exact encoder/decoder matching. This avoids over-claiming a Kodak24 learned controller while preserving the stronger HCG-RVQ story: projected geometry helps, and reliability control is the mechanism that makes it usable.

Artifacts:

- tools/analyze_e193_eflic_reliability_head.py
- experiments/analysis/e193_eflic_force0_global_reliability_head_d1_l3.{md,json,csv}
- experiments/analysis/e193_eflic_force0_global_reliability_head_d1_l3_coefficients.csv
- experiments/analysis/e193_eflic_force0_global_reliability_head_d1_l3_l2_{050,100,200,500}.{md,json,csv}

## E194: EF-LIC Reliability Head Direct-Path Probe

Status: Done for direct-path deployment validation of the E193 learned reliability head. I added `tools/run_e194_eflic_reliability_head_selector_probe.py`, which fits the E193 logistic head from an E161-style label table and evaluates the selected active/fallback policy inside the actual EF-LIC forward/decode path. The probe checks encoder/decoder probability equality, branch decision equality, active decode equality, bpp, and nonfinite rows.

The first implementation attempt exposed an important missing feature issue: the direct predecision stats did not include `z_index_*` features even though the E193 head can use them. I fixed this by adding z-index entropy/perplexity/usage statistics from `z_inds` on both encoder and decoder sides. This is decoder-reproducible and keeps the no-side-bit premise intact.

On the `kodak_first4` smoke run, the learned selector uses the active branch on `3/4` images and improves the diagnostic metrics at unchanged bpp: `dDISTS=-0.001836`, `dLPIPS=-0.000771`, with encoder/decoder decisions matching `4/4`, max selector probability diff `0`, max active decode diff `0`, and `0` nonfinite rows.

On the full Kodak24 self-check, the direct-path result reproduces the E193 same-table behavior: selected branch share `0.500`, unchanged bpp `0.035645`, `dDISTS=-0.000881`, `dLPIPS=-0.000542`, encoder/decoder decisions matching `24/24`, max selector probability diff `0`, max active decode diff `0`, and `0` nonfinite rows. This proves the learned head can be implemented as a codec-side deterministic reliability controller. It is not a final paper result because the current fit table and eval split are both Kodak-derived.

Decision: E194 is an implementation bridge, not the main paper row. The paper-facing default remains E190 until the controller is fit on independent non-Kodak labels and evaluated on held-out data. E194 makes that next step direct: fit the scalar rule and the learned head on an independent fit split, then run the same direct EF-LIC probe on Kodak/Tecnick/CLIC with exact encoder/decoder matching.

Artifacts:

- `tools/run_e194_eflic_reliability_head_selector_probe.py`
- `experiments/analysis/e194_eflic_reliability_head_selector_kodak_first4_smoke_d1_l3.{md,json,csv}`
- `experiments/analysis/e194_eflic_reliability_head_selector_kodak24_selfcheck_d1_l3.{md,json,csv}`

## E195: VQ-LIC Transfer Readiness Package

Status: Done for cross-track planning and artifact readiness. I added `tools/build_e195_vqlic_transfer_readiness.py`, which builds a compact transfer-readiness package from the current HCG-RVQ prototype evidence, EF-LIC controller artifacts, GLC diagnostic artifacts, and local dataset inventory.

The package confirms three important points. First, the prototype HCG-RVQ beta005 evidence should be preserved as the current broad MeanScaleHyperprior/RVQ claim: it improves HCS, old geometry gating, and min090 across OpenImages trusted holdout4096, OpenImages transfer start8192, Kodak, CLIC mobile valid, and CLIC professional valid. Second, the EF-LIC line now has a clear controller hierarchy: E190 remains the immediate controlled paper-facing selector, while E194 is the direct implementation path for a learned reliability head once independent non-Kodak fit labels exist. Third, GLC should proceed with sparse active residual states plus scalar fallback, not dense always-on VQ.

The local dataset inventory is the blocking result: `experiments/data` currently contains only `kodak24` and `kodak_first4`. The planned independent EF-LIC fit/eval directories do not exist yet, so the next full held-out EF-LIC controller test needs external fit/eval images before it can be run honestly. This is a useful guardrail, not a failure: it prevents a same-Kodak controller from being promoted to a paper claim.

Artifacts:

- `tools/build_e195_vqlic_transfer_readiness.py`
- `experiments/analysis/e195_vqlic_transfer_readiness.md`
- `experiments/analysis/e195_vqlic_transfer_readiness.json`
- `experiments/analysis/e195_vqlic_transfer_readiness.artifacts.csv`


## E196-E201: EF-LIC Independent OpenImages-To-Kodak Reliability Transfer

Status: Done for the first independent-fit EF-LIC controller transfer cycle. I added split controls to `tools/run_e160_eflic_projected_hcg_smoke.py` and `tools/run_e194_eflic_reliability_head_selector_probe.py` (`--start-index`, `--max-images`) so large external directories can be sampled reproducibly without accidentally evaluating every OpenImages file. I also made `tools/analyze_e190_eflic_multiobjective_selector.py` robust to tiny smoke splits and added `tools/analyze_e201_eflic_head_threshold_transfer.py` for threshold-transfer diagnostics.

Correction to E195: the earlier dataset blocker was too narrow because it only inspected `experiments/data`. The machine does have non-Kodak data under `/dpl`, including `/dpl/openimages/open-images-v6/train/data` with `300000` images. The new EF-LIC independent-fit smoke uses OpenImages from `start-index=8192` and evaluates on `/dpl/kodak`, with GPU fixed to `CUDA_VISIBLE_DEVICES=0` / `cuda:0`.

E196/E197/E199 generate independent OpenImages active-vs-baseline labels for EF-LIC force0 projected HCG (`alpha=0.05`, `direction_source=mean`). Always-active is not good enough: OpenImages16 averages `dDISTS=+0.001302`, `dLPIPS=+0.000134`; OpenImages64 averages `dDISTS=+0.000582`, `dLPIPS=+0.000153`. But per-image headroom exists. OpenImages64 has DISTS-positive labels `29/64`, LPIPS-positive labels `31/64`, and both-positive labels `16/64`, with oracle deltas `dDISTS=-0.001025`, `dLPIPS=-0.000518`. All runs had `0` nonfinite rows and active decode max diff `0`.

E198 is the first useful independent-fit direct transfer result. A logistic reliability head fit on OpenImages16 and evaluated directly on Kodak24 selects the active branch on `17/24` images (`branch_share=0.708`) and improves both held-out metrics at unchanged bpp: `dDISTS=-0.000350`, `dLPIPS=-0.000036`. Encoder/decoder selector decisions match `24/24`, max probability diff is `0`, max active decode diff is `0`, and nonfinite rows are `0`. This is still small-fit evidence, but it proves the OpenImages-to-Kodak direct path can work.

E200 is the important caution. A head fit on OpenImages64 transfers with a threshold that is too conservative: only `1/24` Kodak images use active, giving `dDISTS=+0.000117`, `dLPIPS=+0.000029`. E201 shows this is largely a calibration issue rather than complete score failure. Sweeping the threshold on the E200 evaluation rows gives a diagnostic best-score point at threshold `0.292586`, branch share `0.583`, `dDISTS=-0.000254`, `dLPIPS=-0.000296`. This sweep is not paper-facing because it uses the eval rows to choose the threshold, but it tells us the next method needs an independent calibration split or temperature/margin calibration.

Decision: do not promote always-active projected HCG or the raw OpenImages64 threshold. The EF-LIC mainline is now: projected HCG geometry plus decoder-reproducible reliability head, with fit/calibration/eval separated. The next paper-facing experiment should fit the head on OpenImages, choose threshold/temperature on a disjoint calibration split, and evaluate on Kodak/Tecnick/CLIC without touching eval labels. If that preserves the E201 best-score behavior, HCG-RVQ has a credible VQ-LIC plug-in claim.

Artifacts:

- `tools/run_e160_eflic_projected_hcg_smoke.py`
- `tools/run_e194_eflic_reliability_head_selector_probe.py`
- `tools/analyze_e190_eflic_multiobjective_selector.py`
- `tools/analyze_e201_eflic_head_threshold_transfer.py`
- `experiments/analysis/e196_eflic_openimages8192_fit4_active_labels.{md,json,csv}`
- `experiments/analysis/e197_eflic_openimages8192_fit16_active_labels.{md,json,csv}`
- `experiments/analysis/e197_eflic_openimages8192_fit16_selector_labels.{md,json,csv}`
- `experiments/analysis/e198_eflic_openimages16_head_to_kodak24_direct_probe.{md,json,csv}`
- `experiments/analysis/e199_eflic_openimages8192_fit64_active_labels.{md,json,csv}`
- `experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels.{md,json,csv}`
- `experiments/analysis/e199_eflic_openimages8192_fit64_multiobj_selector.{md,json,csv}`
- `experiments/analysis/e200_eflic_openimages64_head_to_kodak24_direct_probe.{md,json,csv}`
- `experiments/analysis/e201_e198_openimages16_head_kodak24_threshold_audit.{md,json,csv}`
- `experiments/analysis/e201_e200_openimages64_head_kodak24_threshold_audit.{md,json,csv}`


## E202-E207: EF-LIC Three-Way Fit/Calibration/Eval Probe

Status: Done for the first explicit fit/calibration/eval controller probe. I added `--override-threshold` to `tools/run_e194_eflic_reliability_head_selector_probe.py`, allowing a threshold chosen on an independent calibration split to be fixed before held-out evaluation.

E202 fits the reliability head on OpenImages64 (`start=8192`) and evaluates it on a disjoint OpenImages64 calibration split (`start=16384`). With the raw fit threshold `0.533285`, calibration branch share is `0.125`, with `dDISTS=-0.000154` and `dLPIPS=-0.000008`; encoder/decoder decisions match `64/64`, active decode diff is `0`, and nonfinite rows are `0`. E203 threshold sweep on this calibration split selects best-score threshold `0.324296`, branch share `0.750`, `dDISTS=-0.000596`, `dLPIPS=-0.000014`.

E204 applies that independently calibrated threshold to Kodak24. It does not transfer: branch share `0.417`, `dDISTS=+0.000034`, `dLPIPS=+0.000071`, with exact encoder/decoder agreement and no nonfinite rows. This is a useful negative result: OpenImages calibration improves OpenImages calibration metrics but still selects too many harmful Kodak active branches.

E205 repeats the calibration probe for the smaller OpenImages16 head, because E198 was the best OpenImages-to-Kodak result. On calibration64, the raw threshold selects `0.469` active with `dDISTS=-0.000620` but `dLPIPS=+0.000105`. E206 chooses a conservative best-score threshold `0.775014`, `dDISTS=-0.000217`, `dLPIPS=+0.000005`. E207 applies that threshold to Kodak24 and also fails to transfer: branch share `0.083`, `dDISTS=+0.000221`, `dLPIPS=+0.000022`.

Decision: independent threshold calibration is necessary but not sufficient. E198 remains the strongest positive transfer smoke, but it should not be promoted because the more principled three-way calibration protocol fails on Kodak. The next controller should diagnose distribution shift between OpenImages calibration and Kodak using the E194 probability/features, then test either feature normalization, domain-mixed calibration (OpenImages+CLIC/Tecnick without Kodak tuning), or a stricter objective that penalizes LPIPS/DISTS false positives more directly. The paper line remains viable, but the honest claim is now: HCG projected geometry has codec-valid headroom, and the hard part is robust reliability transfer.

Artifacts:

- `experiments/analysis/e202_eflic_openimages64_head_to_openimages_calib64_direct_probe.{md,json,csv}`
- `experiments/analysis/e203_eflic_openimages64_head_openimages64_calibration_threshold_audit.{md,json,csv}`
- `experiments/analysis/e204_eflic_openimages64_head_calib64_threshold_to_kodak24_direct_probe.{md,json,csv}`
- `experiments/analysis/e205_eflic_openimages16_head_to_openimages_calib64_direct_probe.{md,json,csv}`
- `experiments/analysis/e206_eflic_openimages16_head_openimages64_calibration_threshold_audit.{md,json,csv}`
- `experiments/analysis/e207_eflic_openimages16_head_calib64_threshold_to_kodak24_direct_probe.{md,json,csv}`


## E208-E210: EF-LIC Reliability Shift Audit And Active-State Retuning

Status: Done for the next EF-LIC diagnostic cycle. I updated `tools/run_e194_eflic_reliability_head_selector_probe.py` so direct reliability-head eval CSVs now retain the exact decoder-safe feature values used by the head. I also added `tools/analyze_e208_eflic_reliability_shift_audit.py` to compare fit/calibration/eval probability, active-good labels, and feature distributions.

E208 reruns the OpenImages64-trained `alpha=0.05`, `direction_source=mean` head with feature logging on OpenImages calibration64 and Kodak24, then audits the split shift. The result explains the E202-E207 failure mode. OpenImages calibration is close to the fit distribution, but the head ranking is weak (`AUC=0.517`) and selects only `8/64` active rows. Kodak shifts lower in probability (`prob_mean=0.310` vs fit `0.422`), has essentially random ranking (`AUC=0.500`, `corr(prob, score)=+0.020`), and selects only one image, which is harmful (`selected dDISTS=+0.000117`, `dLPIPS=+0.000029`). The largest Kodak-vs-fit shifts occur in the high-coefficient decoder-safe features: `slice0_mean_abs_mean`, `slice0_mean_rms/std`, `z_hat_abs_mean/std/rms`, `slice0_mean_max`, and `z_index_used_frac`. This is a domain/calibration failure, not a codec-validity or NaN issue: every direct check has exact encoder/decoder match, active decode diff `0`, and nonfinite rows `0`.

E209 retunes the active projected-HCG state itself before adding more controller capacity. On OpenImages calibration32, `logscale/alpha=0.02` is best under the joint score (`dDISTS=-0.000909`, `dLPIPS=-0.000410`), but it transfers poorly to Kodak LPIPS (`dLPIPS=+0.000413`). The current robust candidate is `mean/alpha=0.02`: it improves both OpenImages calibration32 (`dDISTS=-0.000665`, `dLPIPS=-0.000034`) and Kodak24 (`dDISTS=-0.000835`, `dLPIPS=-0.000070`) with unchanged bpp, exact active decode reproduction, and `0` nonfinite rows. `mean/alpha=0.01` and `mean/alpha=0.03` are close, but each has one split where one metric slightly worsens.

E210 tests that new active-state candidate in the learned-head protocol. OpenImages64 fit labels for `mean/alpha=0.02` show headroom (`DISTS labels=30/64`, `LPIPS labels=31/64`, `both=19/64`, DISTS oracle `-0.000739`, LPIPS oracle `-0.000587`), but always-active on that fit split has mixed average (`dDISTS=+0.000173`, `dLPIPS=-0.000017`). When a learned head fit on those labels is applied directly to Kodak24, it selects `5/24` active rows and gives `dDISTS=-0.000162`, `dLPIPS=+0.000115`, worse than simply using the active branch on all Kodak images (`dDISTS=-0.000835`, `dLPIPS=-0.000070`). The E210 threshold audit confirms this: the eval best-score threshold is effectively all-active.

Decision: update the EF-LIC mainline from `mean/alpha=0.05 + reliability head` to `mean/alpha=0.02` as the current active-state candidate, but do not promote the current learned head. The next paper-facing step should evaluate `mean/alpha=0.02` always-active and simple safety rules on additional held-out datasets, then rebuild reliability control around distribution-robust calibration or domain-mixed labels. The useful claim is becoming sharper: HCG projected geometry can improve a VQ-LIC RVQ bottleneck at unchanged bpp, but the reliability controller must be robust to dataset feature shifts.

Artifacts:

- `tools/analyze_e208_eflic_reliability_shift_audit.py`
- `tools/run_e194_eflic_reliability_head_selector_probe.py`
- `experiments/analysis/e208_eflic_head64_openimages_to_kodak_shift_audit.{md,json,csv}`
- `experiments/analysis/e208_eflic_head64_openimages_to_kodak_shift_audit.feature_shift.csv`
- `experiments/analysis/e208_eflic_head64_openimages_to_kodak_shift_audit.top_feature_shift.csv`
- `experiments/analysis/e209_eflic_projected_hcg_openimages_calib32_alpha_direction_sweep.{md,json,csv}`
- `experiments/analysis/e209_eflic_projected_hcg_kodak24_alpha_direction_sweep.{md,json,csv}`
- `experiments/analysis/e209_eflic_alpha_direction_transfer_comparison.{md,csv}`
- `experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_active_labels.{md,json,csv}`
- `experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_selector_labels.{md,json,csv}`
- `experiments/analysis/e210_eflic_openimages64_mean_alpha002_head_to_kodak24_direct_probe.{md,json,csv}`
- `experiments/analysis/e210_eflic_openimages64_mean_alpha002_head_kodak24_threshold_audit.{md,json,csv}`


## E211-E213: EF-LIC Cross-Dataset Active-State Audit And Strength-Controller Headroom

Status: Done for the first cross-dataset validation of the retuned EF-LIC projected-HCG active state. I evaluated `mean/alpha=0.02`, `force_ind=0`, always-active with `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0` on additional held-out splits: OpenImages `start=24576`/64, CLIC mobile 24, CLIC professional 24, DIV2K valid 24, and Tecnick B01R01 24. I also added `tools/analyze_e211_eflic_cross_dataset_active_state.py` to aggregate average deltas, win counts, mismatch/geometry statistics, and feature correlations.

E211 shows the active branch is codec-valid but not universally paper-safe as an always-on rule. Every split has `0` nonfinite rows and max decode diff `0`, so the implementation remains deterministic and decoder-consistent on GPU0. The average metric signs are mixed: OpenImages24576 has `dDISTS=-0.000034` but `dLPIPS=+0.000157`; CLIC mobile improves both (`dDISTS=-0.000141`, `dLPIPS=-0.000030`); CLIC professional worsens both (`dDISTS=+0.000462`, `dLPIPS=+0.000090`); DIV2K improves DISTS but worsens LPIPS (`dDISTS=-0.000152`, `dLPIPS=+0.000123`); Tecnick worsens both at `alpha=0.02` (`dDISTS=+0.000269`, `dLPIPS=+0.000103`). This confirms that the active geometry itself is real and safe to run, but `mean/alpha=0.02` cannot be the final universal always-active paper method.

E212 tests whether the harmful splits can be rescued by weakening the geometry strength. On CLIC professional, alpha weakening alone is not enough: all tested alphas (`0.005/0.01/0.015/0.02/0.03`) remain average-harmful under the joint score. On Tecnick, however, `alpha=0.01` flips the result from harmful to beneficial (`dDISTS=-0.000120`, `dLPIPS=-0.000098`). This is a useful design signal: the right controller should not only choose active/fallback; it should also control geometry strength.

E213 computes a per-image alpha oracle over the failure splits, including `alpha=0` fallback. This gives clear headroom for a learned reliability/strength controller. On CLIC professional, the score oracle reaches `dDISTS=-0.000976`, `dLPIPS=-0.000333` with active fraction `0.708`. On Tecnick, the score oracle reaches `dDISTS=-0.001016`, `dLPIPS=-0.000474` with active fraction `0.708`. Since the oracle includes fallback, this is direct evidence that harmful average results are not because HCG geometry is useless on these domains; they are because the controller must choose when and how strongly to apply it.

Decision: keep `mean/alpha=0.02` as the current EF-LIC active-state candidate for Kodak/CLIC-mobile/DIV2K-style positive evidence, but update the main HCG-RVQ strengthening direction to a conservative strength controller with `alpha=0` as an explicit safe option. The next implementation should generate mixed-domain labels over a small alpha set, train/calibrate a decoder-reproducible controller on disjoint data, and evaluate on held-out Kodak/Tecnick/CLIC/OpenImages without touching final labels. This is more publishable than either a Kodak-tuned scalar threshold or a universal always-active branch.

Artifacts:

- `tools/analyze_e211_eflic_cross_dataset_active_state.py`
- `experiments/analysis/e211_eflic_openimages24576_eval64_mean_alpha002_active.{md,json,csv}`
- `experiments/analysis/e211_eflic_clic_mobile24_mean_alpha002_active.{md,json,csv}`
- `experiments/analysis/e211_eflic_clic_professional24_mean_alpha002_active.{md,json,csv}`
- `experiments/analysis/e211_eflic_div2k_valid24_mean_alpha002_active.{md,json,csv}`
- `experiments/analysis/e211_eflic_tecnick_b01r01_24_mean_alpha002_active.{md,json,csv}`
- `experiments/analysis/e211_eflic_cross_dataset_mean_alpha002_active.{md,summary.csv,per_image.csv}`
- `experiments/analysis/e212_eflic_clic_professional24_mean_alpha_sweep_active.{md,json,csv}`
- `experiments/analysis/e212_eflic_tecnick_b01r01_24_mean_alpha_sweep_active.{md,json,csv}`
- `experiments/analysis/e212_eflic_failure_split_alpha_sweep_summary.{md,csv}`
- `experiments/analysis/e213_eflic_failure_split_alpha_oracle_summary.{md,csv}`


## E214-E216: EF-LIC Mixed-Domain Strength-Controller Probe

Status: Done for the first mixed-domain strength-controller diagnostic. I added `tools/analyze_e214_eflic_strength_controller_probe.py`, which consumes E160-style multi-alpha CSVs, adds an explicit `alpha=0` fallback, and trains/evaluates only decoder-reproducible global predecision features: `z_hat_*`, `z_index_*`, `slice0_mean_*`, and `slice0_scale_*`. It reports fixed-alpha, centroid, decision-stump, ridge-score, and oracle selection under the joint score `DISTS + 3*LPIPS`.

E214 first used only the two failure splits from E212. This was useful as a negative control: CLIC professional has strong oracle headroom, but fixed alpha chooses fallback and learned global controllers remain harmful or near-harmful; Tecnick can be partly rescued by fixed `alpha=0.01` or ridge-score. This showed that failure-split-only labels are not enough to train a robust policy.

E215 generated additional mixed-domain multi-alpha labels with GPU0 fixed and no nonfinite rows: CLIC mobile 24, OpenImages `start=24576` eval32, and DIV2K valid24, each using `force_ind=0`, `direction_source=mean`, and `alpha in {0.005, 0.01, 0.015, 0.02, 0.03}`. These are controller-design labels, not final paper rows.

E216 aggregates E212+E215 into a 128-image mixed-domain controller probe. The main result is that oracle headroom is large, but the current whole-image global-predecision feature/controller family is not robust enough for the paper method. Pooled LOOCV oracle reaches `dDISTS=-0.001220`, `dLPIPS=-0.000471`, score `-0.002631`; the best non-oracle pooled policy is only the decision stump with `dDISTS=-0.000311`, `dLPIPS=-0.000029`, score `-0.000399`. Leave-one-domain-out is weaker: CLIC professional remains harmful for all learned policies, Tecnick is best served by fixed `alpha=0.01`, OpenImages prefers ridge-score, and DIV2K remains close to neutral.

Decision: do not promote the current 22-feature whole-image strength controller as the HCG-RVQ method. The positive evidence is still important: HCG geometry creates useful per-image alternatives at unchanged bpp, and the oracle gap is too large to ignore. But the next method-strengthening step should move from a single whole-image predecision to a decoder-reproducible local/sequential controller, where each slice can use the local hyperprior/slice context available immediately before its own quantizer decision. This is better aligned with the core prompt claim that the hyperprior generates local quantizer geometry, not merely a global image-level switch.

Artifacts:

- `tools/analyze_e214_eflic_strength_controller_probe.py`
- `experiments/analysis/e214_eflic_strength_controller_probe.{md,json,csv,features.json}`
- `experiments/analysis/e215_eflic_clic_mobile24_mean_alpha_sweep_active.{md,json,csv}`
- `experiments/analysis/e215_eflic_openimages24576_eval32_mean_alpha_sweep_active.{md,json,csv}`
- `experiments/analysis/e215_eflic_div2k_valid24_mean_alpha_sweep_active.{md,json,csv}`
- `experiments/analysis/e216_eflic_mixed_domain_strength_controller_probe.{md,json,csv,features.json}`

## E217 EF-LIC Slice-Wise HCG Strength Probe

E217 implements fixed decoder-reproducible slice alpha schedules in `tools/run_e160_eflic_projected_hcg_smoke.py` via `--slice-alpha-schedule name:a0,a1,a2,a3`. This preserves the old scalar `--alpha` path, but lets EF-LIC use different HCG projected-geometry strengths for the four y-slices without side bits. Encoder/decode reproducibility remains exact because the same schedule is known to both sides.

The probe was run on GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) for CLIC professional valid first 12 images and Tecnick C00R01 first 12 images. All rows had `nonfinite=0` and `max_decode_diff=0`.

Key results are in `experiments/analysis/e217_eflic_slice_schedule_probe_summary.md`. CLIC professional remains difficult for a fixed schedule: `all020` improves DISTS (`dDISTS=-0.000591`) but worsens LPIPS (`dLPIPS=+0.000251`), while the safest fixed schedule by the DISTS+3*LPIPS score is `slice3only020` with only a tiny gain/near-neutral row (`dDISTS=-0.000060`, `dLPIPS=+0.000043`). The per-image schedule oracle is much stronger (`dDISTS=-0.001426`, `dLPIPS=-0.000194`).

Tecnick shows the opposite useful pattern: late/local schedules are genuinely beneficial. `slice2only020` gives `dDISTS=-0.000218`, `dLPIPS=-0.000213`, and `late020` gives `dDISTS=-0.000200`, `dLPIPS=-0.000214`. The Tecnick oracle is stronger again (`dDISTS=-0.000605`, `dLPIPS=-0.000718`). Pooled across the two splits, the best fixed-score schedule is `late020` (`dDISTS=+0.000009`, `dLPIPS=-0.000073`, score `-0.000210`), while the oracle reaches `dDISTS=-0.001016`, `dLPIPS=-0.000456`, score `-0.002382`.

Interpretation: fixed slice scheduling is a useful diagnostic, not yet the method. The domain difference between CLIC professional and Tecnick, plus the large per-image oracle gap, supports the E216 conclusion that the paper-facing EF-LIC plug-in needs a local/context-conditioned decoder-safe controller, not one global image-level alpha or one fixed schedule. The next implementation target is a per-slice strength selector driven by the local pre-quantization context (`mean`, `scale`, support buffer, z/index stats) with exact encoder/decoder matching.

## E218-E220: EF-LIC Expanded Local Slice Controller Diagnostics

Status: Done for the first 5-domain local/slice controller audit. I re-read `docs/prompt.txt` and kept the target aligned with the core HCG-RVQ claim: hyperprior/context should control local quantizer geometry, not merely add a post-hoc image-level selector. The experiments here use EF-LIC projected HCG geometry as a diagnostic plug-in and only decoder-safe predecision features: z/z-index statistics, slice id, and the current slice mean/scale statistics.

E218 adds `tools/analyze_e218_eflic_local_slice_predictability.py`. The first 2-domain run on E217 CLIC professional + Tecnick showed local signal but weak transfer. I then expanded the label set in E219 by running single-slice `alpha=0.02` schedules on CLIC mobile 12, DIV2K valid 12, and OpenImages train `start=24576` 12. All E219 evaluations used GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), with `nonfinite=0` and `max_decode_diff=0` for every schedule.

The 5-domain E219 stump audit is informative but not yet paper-main. Pooled all-on single-slice is slightly harmful under `score=dDISTS+3*dLPIPS` (`+0.000035`), while the marginal oracle is strong (`score=-0.000735`, `dDISTS=-0.000323`, `dLPIPS=-0.000137`). The best same-table stump improves (`score=-0.000215`) using `z_index_entropy >= 5.801702499`, and sample-LOOCV is still positive evidence but smaller (`score=-0.000155`). Leave-dataset-out is mixed: CLIC mobile, DIV2K, and Tecnick improve, but CLIC professional and OpenImages do not. Top pooled correlations are weak (`z_index_used_frac` Spearman `-0.156`, `z_index_entropy` `-0.151`), which argues against hard-coding a single threshold from these small labels.

E220 adds `tools/analyze_e220_eflic_local_slice_linear_controller.py` to test whether a ridge score predictor can harvest the same local features better than a stump. Same-table rows are strong, e.g. pooled ridge reaches `score=-0.000328`, and each per-domain same-table ridge moves much closer to its oracle. However, sample-LOOCV drops to only `score=-0.000083`, and leave-dataset-out mostly fails: CLIC professional `+0.000389`, DIV2K `+0.000347`, OpenImages `+0.000098`, Tecnick `+0.000061`, with only CLIC mobile improving (`-0.000178`). This says the feature family contains signal, but the current small marginal-label controller overfits domain-specific feature/metric relationships.

Decision: do not promote stump or ridge local controllers as the HCG-RVQ method. Use E218-E220 as design evidence: local/slice HCG alternatives are valid and have large oracle headroom at unchanged bpp, but a publishable EF-LIC plug-in needs a stronger learned local HCG geometry/strength head trained on independent mixed-domain labels, or a full fine-tuning path where the controller is learned with the codec objective. The next EF-LIC step should move from hand-built marginal selectors to a trained local controller with explicit fallback and strength selection, evaluated with leave-domain-out and final held-out full evaluation.

Artifacts:

- `tools/analyze_e218_eflic_local_slice_predictability.py`
- `tools/analyze_e220_eflic_local_slice_linear_controller.py`
- `experiments/analysis/e219_eflic_clic_mobile12_single_slice_probe.{md,json,csv}`
- `experiments/analysis/e219_eflic_div2k_valid12_single_slice_probe.{md,json,csv}`
- `experiments/analysis/e219_eflic_openimages24576_12_single_slice_probe.{md,json,csv}`
- `experiments/analysis/e219_eflic_expanded_local_slice_predictability_probe.{md,json,summary.csv,samples.csv,correlations.csv}`
- `experiments/analysis/e220_eflic_local_slice_linear_controller_probe.{md,json,summary.csv}`

## E221-E222: EF-LIC Spatial Local Quant-MSE Controller Diagnostics

Status: Done for the first spatial-position diagnostic of EF-LIC projected-HCG local control. I added `tools/analyze_e221_eflic_spatial_quant_mse_probe.py`, which compares fixed EF-LIC RVQ against projected-HCG RVQ at sampled spatial positions inside each normalized y-slice. The labels use raw active-vs-baseline quantization MSE only for diagnostic supervision; the candidate predictor features are restricted to decoder-reproducible predecision maps and statistics: z/index features, slice id, current slice mean/scale, support buffer summaries, hyper support, and previous decoded support. The probe used GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) on 4-image subsets from CLIC professional, CLIC mobile, Tecnick C00R01, and OpenImages start24576. It produced 24,576 spatial samples with no nonfinite slice rows.

E221 gives strong local headroom but also shows why an always-on spatial perturbation is not the method. Pooled all-on projected HCG increases normalized quantization MSE by `+0.000063`, but the local oracle over positions reaches `-0.000458`, with `40.3%` helpful positions. The oracle gain is stable across domains: CLIC mobile `-0.000427`, CLIC professional `-0.000476`, OpenImages `-0.000462`, and Tecnick `-0.000466`. Slice 0 has the largest oracle opportunity, while later slices are more fragile. Simple univariate signals are weak; the largest pooled Spearman correlations with local delta MSE are only around `0.17-0.18`, mostly slice/support/mean magnitude signals.

E222 adds `tools/analyze_e222_eflic_spatial_linear_controller.py`, which fits ridge score predictors on the E221 spatial samples. Same-table rows confirm that decoder-safe features contain some signal: pooled ridge with a trained threshold improves dMSE to `-0.000042` versus all-on `+0.000063`, and per-domain same-table rows improve all four domains. However, transfer is not ready for a paper method. Leave-dataset-out ridge remains harmful on every held-out domain (`+0.000013` to `+0.000114` dMSE), and leave-image-out pooled ridge is still positive (`+0.000026`). This means the current feature set and post-hoc linear policy can expose mechanism/headroom, but it is not a deployable HCG-RVQ controller.

Decision: keep E221-E222 as controlled evidence for local hyperprior/context-dependent quantizer geometry. Do not promote the spatial stump/ridge controller as the proposed method. The next EF-LIC strengthening step should be a trained local HCG geometry/strength head with explicit fallback, optimized with codec/quantization objectives and evaluated on held-out domains, rather than another hand-coded threshold on diagnostic labels.

Artifacts:

- `tools/analyze_e221_eflic_spatial_quant_mse_probe.py`
- `tools/analyze_e222_eflic_spatial_linear_controller.py`
- `experiments/analysis/e221_eflic_spatial_quant_mse_probe.{md,json,summary.csv,slice_summary.csv,samples.csv,correlations.csv}`
- `experiments/analysis/e222_eflic_spatial_linear_controller_probe.{md,json,summary.csv}`

## E223: EF-LIC Spatial Normalized Controller Probe

Status: Done for the normalization/threshold follow-up to E221-E222. I added `tools/analyze_e223_eflic_spatial_normalized_controller.py`, which reuses the E221 spatial quant-MSE labels and tests whether simple decoder-safe fixes can rescue transfer: image/slice-relative features, raw-plus-relative features, and dataset-balanced threshold selection. This is still a diagnostic controller, not a codec method.

The result closes an important branch. Same-table rows can improve slightly beyond E222, e.g. `raw_plus_image_slice_rel` reaches `-0.000054` dMSE versus E222 raw ridge `-0.000042`, but held-out transfer remains weak. Leave-dataset-out is still positive/harmful for almost all domains and settings. The only negative LODO rows are tiny (`tecnick -0.000003` for raw-plus-image-slice relative; `clicpro -0.000006` for raw-plus-image relative), while OpenImages remains strongly harmful (`+0.000115` to `+0.000136` for the raw-plus-relative variants). Leave-image-out also stays positive for every tested feature mode (`+0.000016` to `+0.000030`). Dataset-balanced thresholding gives essentially the same decisions as pooled thresholding on this balanced sample.

Decision: feature normalization and balanced thresholding are not enough to make a hand-built spatial controller paper-safe. E223 strengthens the conclusion from E218-E222: HCG-RVQ has real local oracle headroom, but the publishable EF-LIC plug-in should learn the local geometry/strength/fallback behavior with a codec-aware objective, rather than post-hoc selecting positions from fixed diagnostic labels.

Artifacts:

- `tools/analyze_e223_eflic_spatial_normalized_controller.py`
- `experiments/analysis/e223_eflic_spatial_normalized_controller_probe.{md,json,summary.csv}`

## E224: EF-LIC Spatial MLP Teacher-Head Probe

Status: Done for the first small learned-head diagnostic on the E221 spatial quant-MSE labels. I added `tools/analyze_e224_eflic_spatial_mlp_head_probe.py`, which trains a lightweight MLP on sampled spatial positions using decoder-safe features and the active-vs-baseline quantization-MSE teacher signal. This is still a diagnostic bridge toward an in-codec local HCG head, not a final paper metric row.

The capacity check is positive but bounded. Same-table training can turn the harmful all-on row (`+0.000063` dMSE) into an improving controller: `raw_plus_image_rel` reaches `-0.000027` dMSE, and `raw_plus_image_slice_rel` reaches `-0.000050` dMSE with active fraction `0.337`, precision `0.504`, and recall `0.422`. This confirms that the local decoder-safe feature family contains learnable information beyond the linear/ridge probes.

The transfer check is the important result. Leave-dataset-out remains mostly harmful. With `raw_plus_image_rel`, every held-out domain is positive/harmful (`+0.000016` to `+0.000061` dMSE). With `raw_plus_image_slice_rel`, only CLIC professional improves (`-0.000036`), while CLIC mobile, OpenImages, and Tecnick remain harmful. Therefore, a post-hoc teacher-label MLP is not robust enough to become the proposed EF-LIC plug-in.

Decision: keep E224 as evidence that a learned local head has capacity, but do not promote teacher-label MLP selection as the HCG-RVQ method. The next strengthening step should place a local HCG geometry/strength/fallback head inside the codec path after the EF-LIC `_mean_scale(support_buf, i)` point, train it with codec-aware quantization/perceptual losses and explicit false-positive control, and then evaluate on mixed-domain held-out splits. This is the more faithful route to the prompt goal: hyperprior/context should generate and control local quantizer geometry, not merely fit a fragile offline selector.

Artifacts:

- `tools/analyze_e224_eflic_spatial_mlp_head_probe.py`
- `experiments/analysis/e224_eflic_spatial_mlp_head_probe.{md,json,summary.csv}`

## E225: EF-LIC Spatial Alpha-Map Codec-Path Smoke

Status: Done for the first decoder-reproducible spatial strength-control scaffold. I added `tools/run_e225_eflic_spatial_alpha_map_smoke.py`, which moves the next HCG-RVQ strengthening step into the EF-LIC codec path after `_mean_scale(support_buf, i)`. Instead of one scalar projected-HCG strength for an entire image, the script builds a spatial `alpha_map` from decoder-available predecision context (`mean`, `scale`, `support_buf`, and previous decoded support). The same deterministic alpha map is recomputed during decompression, so this remains a no-sidebit plug-in.

The implementation includes a `zero` mode as a baseline identity check, a `constant` mode matching the old scalar projected-HCG reference through the same alpha-map path, and local diagnostic modes (`mean_abs_top`, `scale_rms_top`, `support_rms_top`, `prev_rms_top`). I fixed a tie-related top-fraction issue by selecting exact top-k positions instead of thresholding by a quantile, and `prev_rms_top` now correctly returns an all-zero map for slice 0 because no previous decoded y-slices are available yet.

The full CLIC professional validation split (`/dpl/clic/professional/valid`, 41 images) was run on GPU0 only with `force_ind=0`, `alpha=0.02`, and `active_frac=0.25`. All rows have `delta_bpp=0`, `max_decode_diff=0`, and `nonfinite_rows=0`. `zero` exactly reproduces EF-LIC. The local `prev_rms_top` controller is the best CLIC professional row among the tested fixed policies: `dPSNR=+0.003486`, `dDISTS=-0.000103`, `dLPIPS=+0.000015`, with only `0.027835` y-index mismatch fraction and `0.1875` active alpha fraction. This is small, but it is important because it improves the difficult CLIC professional split without side bits and without post-hoc teacher labels.

Kodak24 gives a different but equally useful signal. `constant` is still the strongest tested row there (`dPSNR=+0.000791`, `dDISTS=-0.000835`, `dLPIPS=-0.000070`), while `prev_rms_top` remains perceptually helpful but weaker (`dDISTS=-0.000303`, `dLPIPS=-0.000008`, `dPSNR=-0.000751`). Therefore a fixed local-sparse rule is not the final method. The evidence instead says the next publishable module should learn when to use all-on, local-sparse, weaker strength, or fallback from decoder-safe local context.

Decision: promote spatial alpha-map control from a diagnostic idea to the next implementation target, but not as a fixed-rule paper result. E225 gives a codec-valid positive route after the E224 teacher-head transfer failure: use previous decoded support and local hyperprior/context maps to produce a trainable local HCG strength/fallback head inside the codec loop. For dataset protocol, CLIC professional should be the main CLIC validation split in the paper-facing table; CLIC mobile is better kept as a robustness or appendix split unless a specific related work requires it.

Artifacts:

- `tools/run_e225_eflic_spatial_alpha_map_smoke.py`
- `experiments/analysis/e225_eflic_clicpro8_spatial_alpha_map_smoke.{md,json,csv}`
- `experiments/analysis/e225_eflic_clicpro41_spatial_alpha_map_keymodes.{md,json,csv}`
- `experiments/analysis/e225_eflic_kodak24_spatial_alpha_map_keymodes.{md,json,csv}`

## E226: EF-LIC Spatial Alpha Strength Follow-Up

Status: Done for the first strength follow-up to E225. I kept the exact same codec-path alpha-map scaffold and reran the key modes at `alpha=0.01` on the full CLIC professional validation split and Kodak24. All runs used GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). Every summarized row has `delta_bpp=0`, `max_decode_diff=0`, and `nonfinite_rows=0`, so the comparison is a geometry/strength effect rather than a bitstream or CUDA artifact.

The CLIC professional result improves the fixed-rule picture. At `alpha=0.01`, `prev_rms_top` reaches `dDISTS=-0.000193`, `dLPIPS=-0.000009`, score `-0.000220`, and score-win fraction `0.634`. This is better than the earlier `alpha=0.02` `prev_rms_top` score `-0.000058`, and much safer than all-on constant rows whose LPIPS remains harmful. Weak local control is therefore a plausible reliability/fallback direction for hard CLIC professional images.

Kodak24 shows the complementary behavior. Stronger all-on `constant/alpha=0.02` remains best under the diagnostic score (`dDISTS=-0.000835`, `dLPIPS=-0.000070`, score `-0.001047`), while `prev_rms_top/alpha=0.01` improves LPIPS and DISTS but is weaker (`score=-0.000259`). This means the final method should not hard-code a local-sparse rule. It should learn a decoder-reproducible local strength policy that can choose all-on for easy/Kodak-like cases and weak local/fallback for harder CLIC-like cases.

Decision: use E226 as the concrete next-design target for a trained codec-path head. The head should output at least a small discrete strength/fallback set, e.g. `alpha in {0, 0.01, 0.02}` and possibly a local active map, rather than a binary active/off switch. For paper-facing evidence, CLIC professional is now the most important stress split because it exposes why conservative local control is needed; Kodak remains useful for showing that HCG geometry can produce larger perceptual gains when all-on is safe.

Artifacts:

- `tools/analyze_e226_eflic_spatial_alpha_strength_summary.py`
- `experiments/analysis/e226_eflic_clicpro41_spatial_alpha_map_alpha001_keymodes.{md,json,csv}`
- `experiments/analysis/e226_eflic_kodak24_spatial_alpha_map_alpha001_keymodes.{md,json,csv}`
- `experiments/analysis/e226_eflic_spatial_alpha_strength_summary.{md,csv}`

## E227-E228: EF-LIC Spatial Alpha Candidate Oracle and Weak-Strength Expansion

Status: Done for the next selector-headroom audit. I added `tools/analyze_e227_eflic_spatial_alpha_candidate_selector.py`, which combines all codec-valid E225/E226/E228 candidates and measures fixed-candidate rows, per-image candidate oracle, leave-dataset-out fixed transfer, and a tiny decoder-safe feature stump. The candidate set now includes `zero` fallback, `constant`, `support_rms_top`, and `prev_rms_top` at `alpha in {0.005, 0.01, 0.02}`. All candidate generation was run on GPU0 only with unchanged bpp, exact decode reproduction, and zero nonfinite rows.

E228 adds the missing weak-strength point `alpha=0.005` on CLIC professional 41 and Kodak24. This updates the fixed-rule ranking. On CLIC professional, the best fixed row is now `prev_rms_top/alpha=0.005` with `dDISTS=-0.000148`, `dLPIPS=-0.000033`, score `-0.000248`, and y-mismatch fraction only `0.009326`. On Kodak24, `alpha=0.005` is too weak or harmful on average; the best fixed row remains `constant/alpha=0.02` with score `-0.001047`.

The candidate oracle is the most important result. Pooled over CLIC professional + Kodak, the per-image oracle over codec-valid candidates reaches `dDISTS=-0.001391`, `dLPIPS=-0.000634`, score `-0.003292`, with score-win fraction `0.969`. CLIC professional oracle reaches score `-0.002416`; Kodak oracle reaches score `-0.004789`. The oracle chooses a mixture of `constant`, `support_rms_top`, `prev_rms_top`, and occasional `zero`, so the best policy is not a single branch or a single alpha.

The transfer diagnostics prevent over-claiming. Same-table stump improves to score `-0.001039`, but leave-dataset-out is still not robust: training on Kodak and applying to CLIC professional gives `+0.000504`, and training on CLIC professional then applying to Kodak is only `-0.000100`. Leave-dataset-out best-fixed has the same issue: Kodak-trained `constant@0.02` hurts CLIC, while CLIC-trained weak local control is too weak for Kodak. This means the candidate set is strong, but handcrafted or post-hoc dataset-transfer selectors are not paper-safe.

Decision: keep E227/E228 as the strongest motivation so far for a codec-trained local HCG strength head. The next implementation should not be another fixed alpha sweep. It should train a decoder-reproducible local module inside the EF-LIC loop that can choose fallback, weak local geometry, or stronger geometry using local context. For full-evaluation planning, the ablation ladder should include EF-LIC baseline, best fixed all-on, best fixed local, candidate oracle, and the trained local strength head.

Artifacts:

- `tools/analyze_e227_eflic_spatial_alpha_candidate_selector.py`
- `experiments/analysis/e228_eflic_clicpro41_spatial_alpha_map_alpha0005_keymodes.{md,json,csv}`
- `experiments/analysis/e228_eflic_kodak24_spatial_alpha_map_alpha0005_keymodes.{md,json,csv}`
- `experiments/analysis/e227_eflic_spatial_alpha_candidate_selector.{md,json,summary.csv,oracle_choices.csv}`
- `experiments/analysis/e226_eflic_spatial_alpha_strength_summary.{md,csv}`

## E229: EF-LIC Continuous Spatial Alpha-Map Candidates

Status: Done for the first continuous decoder-reproducible alpha-map branch. I extended `tools/run_e225_eflic_spatial_alpha_map_smoke.py` with smooth alpha maps derived from decoder-side local scores. The new modes are `support_rms_top_soft`, `prev_rms_top_soft`, `support_over_scale_top_soft`, and `prev_over_scale_top_soft`. They keep the same codec scaffold: alpha is recomputed during decompression from `mean`, `scale`, `support_buf`, and previous decoded support, so no side bits are added.

A 2-image Kodak smoke passed first, then full Kodak24 and CLIC professional 41 were evaluated at `alpha=0.02` with GPU0 fixed (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). All E229 rows have `delta_bpp=0`, `max_decode_diff=0`, and `nonfinite_rows=0`. GPU1 was not used.

Fixed-rule results are mixed. On CLIC professional, the continuous soft maps are not better than the previous weak sparse rule: the best fixed CLIC row remains `prev_rms_top@0.005` with score `-0.000248`, while soft maps improve DISTS in some cases but hurt LPIPS on average. On Kodak24, `support_over_scale_top_soft@0.02` gives a strong DISTS-only row (`dDISTS=-0.000888`) but its score (`-0.000828`) is still weaker than fixed `constant@0.02` (`-0.001047`).

The important positive result is candidate-set headroom. Adding the soft candidates improves the per-image oracle from the previous E227 pooled score `-0.003292` to `-0.003613`. CLIC professional oracle improves from `-0.002416` to `-0.002664`; Kodak oracle improves from `-0.004789` to `-0.005233`. The oracle chooses soft candidates on 14/65 images, so smooth local strength is useful as a branch even though it is not a standalone fixed policy.

Decision: keep continuous alpha maps as candidate states for the next learned controller, especially `support_over_scale_top_soft`, but do not promote any soft fixed rule as the method. The next implementation target remains a codec-trained decoder-safe head that chooses among fallback, weak sparse, strong all-on, and soft support/scale-conditioned geometry.

Artifacts:

- `experiments/analysis/e229_eflic_clicpro41_spatial_alpha_soft_alpha002.{md,json,csv}`
- `experiments/analysis/e229_eflic_kodak24_spatial_alpha_soft_alpha002.{md,json,csv}`
- `experiments/analysis/e229_eflic_spatial_alpha_candidate_selector_with_soft.{md,json,summary.csv,oracle_choices.csv}`
- Updated `tools/run_e225_eflic_spatial_alpha_map_smoke.py`
- Updated `tools/analyze_e226_eflic_spatial_alpha_strength_summary.py`
- Updated `tools/analyze_e227_eflic_spatial_alpha_candidate_selector.py`

## E230 EF-LIC Weak Continuous Alpha-Map Strength Sweep

Status: Done for the weak continuous alpha-map strength sweep. I ran the same decoder-reproducible soft local geometry branches from E229 at `alpha=0.01` on Kodak24 and CLIC professional 41 images, always with `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`. The generated artifacts are:

- `experiments/analysis/e230_eflic_kodak24_spatial_alpha_soft_alpha001.{csv,json,md}`
- `experiments/analysis/e230_eflic_clicpro41_spatial_alpha_soft_alpha001.{csv,json,md}`
- `experiments/analysis/e230_eflic_spatial_alpha_candidate_selector_with_soft_strengths.{summary.csv,oracle_choices.csv,json,md}`

All E230 codec rows preserved unchanged bpp, exact decode reproduction (`max_decode_diff=0`), and zero nonfinite rows. The evaluation therefore isolates quantizer-geometry and strength effects rather than bitstream or device artifacts.

The fixed-policy result is mixed but informative. On CLIC professional, weak soft previous-support geometry becomes a credible fixed rule: `prev_over_scale_top_soft@0.01` reaches `dPSNR=+0.003849`, `dDISTS=-0.000075`, `dLPIPS=-0.000049`, and score `-0.000222`, close to the best fixed CLIC row `prev_rms_top@0.005` at score `-0.000248`. This is an improvement over E229 `alpha=0.02`, where soft maps were generally too strong and hurt LPIPS on average.

On Kodak24, weakening the soft maps makes them too conservative or poorly aligned as fixed policies. `support_rms_top_soft@0.01` is nearly neutral in the joint score (`+0.000039`), while the strongest fixed Kodak row remains `constant@0.02` with score `-0.001047`. This preserves the earlier conclusion that Kodak-like images can tolerate stronger all-on geometry, whereas CLIC professional needs conservative local control.

Adding the weak soft branches still increases oracle headroom. With E230 included, the pooled per-image oracle improves from the E229 value around `-0.003613` to `-0.003649`; CLIC professional improves to `-0.002696`, and Kodak improves to `-0.005277`. The oracle now chooses soft-alpha branches on 21 of 65 images, including both weak and strong soft modes. This supports the next method design: a codec-trained no-sidebit local controller should expose fallback, weak sparse previous-support, strong all-on, support-conditioned, and smooth support/scale-conditioned geometry states, rather than committing to one handcrafted branch.

## E231 EF-LIC Lower Soft Alpha Sweep

Status: Done for the lower continuous alpha-map sweep. I evaluated the E229 soft local geometry branches at `alpha=0.005` on CLIC professional 41 images and Kodak24, again with `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`. The artifacts are:

- `experiments/analysis/e231_eflic_clicpro41_spatial_alpha_soft_alpha0005.{csv,json,md}`
- `experiments/analysis/e231_eflic_kodak24_spatial_alpha_soft_alpha0005.{csv,json,md}`
- `experiments/analysis/e231_eflic_spatial_alpha_candidate_selector_with_soft_strengths.{summary.csv,oracle_choices.csv,json,md}`

All E231 codec rows preserve unchanged bpp, exact decoder reproduction (`max_decode_diff=0`), and zero nonfinite rows.

The lower strength does not beat the E230 fixed soft rule. On CLIC professional, `prev_over_scale_top_soft@0.005` gives `dPSNR=+0.002443`, `dDISTS=-0.000017`, `dLPIPS=-0.000051`, and score `-0.000170`. This is safe for LPIPS but weaker than `prev_over_scale_top_soft@0.01` with score `-0.000222`, and still weaker than sparse `prev_rms_top@0.005` with score `-0.000248`. Thus the fixed soft branch has a useful strength band around `0.01`; dropping to `0.005` becomes too conservative.

Kodak24 confirms that very weak soft geometry is not the right fixed policy for easier/aggressive-geometry-friendly images. All `alpha=0.005` soft rows have positive joint score, while `constant@0.02` remains the best fixed Kodak row.

The candidate oracle does improve again when the `0.005` soft states are added: pooled score becomes `-0.003725`, CLIC professional becomes `-0.002762`, and Kodak becomes `-0.005370`. The oracle selects weak soft states on a few images, so they should stay in the branch library for a learned controller, but they should not be promoted to a universal handcrafted rule.


## E232 EF-LIC Branch Library Audit

Status: Done for the branch-family audit that should guide the next learned-controller implementation. I added `tools/analyze_e232_eflic_branch_library_audit.py`, aggregating E225-E231 codec-valid candidate CSVs into fixed-candidate summaries, per-image oracle choices, leave-one-family-out ablations, greedy family-set construction, and risk correlations. This is a CPU-only post-processing step over existing artifacts; no GPU was used.

The top-level result confirms that the branch library is useful but cannot be reduced to one fixed rule. Over CLIC professional 41 + Kodak24, the best pooled fixed candidate is `prev_rms_top@a0.010` with score `-0.000234`, while the per-image branch oracle reaches score `-0.003725`. Dataset behavior differs: CLIC best fixed candidate is conservative `prev_rms_top@a0.005` (`-0.000248`), whereas Kodak best fixed candidate is aggressive `constant@a0.020` (`-0.001047`).

The leave-one-family-out audit gives the safest implementation guidance. Pooled removal losses are: `constant +0.000483`, `soft_support +0.000257`, `sparse_support +0.000184`, `soft_prev +0.000153`, and `sparse_prev +0.000069`. Thus all active families are non-redundant, but `constant` and `soft_support` carry the largest immediate value. Greedy family selection also supports this: pooled oracle improves from zero to `constant+zero` (`-0.002557`), then to `constant+soft_support+zero` (`-0.003102`), then `soft_prev` (`-0.003436`), `sparse_support` (`-0.003657`), and finally `sparse_prev` (`-0.003725`).

Decision: stop expanding hand-coded fixed alpha branches for now. The next EF-LIC implementation should expose a compact decoder-safe branch vocabulary: zero fallback, aggressive constant geometry, soft support/scale local geometry, soft previous-context geometry, sparse support, and sparse previous-context. The trained controller should prioritize false-positive control because CLIC and Kodak have opposite risk correlations: stronger y-index/geometry perturbation correlates with worse score on CLIC but better score on Kodak.

Artifacts:

- `tools/analyze_e232_eflic_branch_library_audit.py`
- `experiments/analysis/e232_eflic_branch_library_audit.{md,json,summary.csv,fixed_summary.csv,oracle_choices.csv,leave_one_family_out.csv,greedy_family_set.csv,risk_correlations.csv}`


## E233 EF-LIC Decoder-Safe Branch Feature and Label-Readiness Audit

Status: Done. After E232 showed that the useful EF-LIC HCG branch families are non-redundant but fixed rules are not enough, I added a decoder-safe feature dump and a branch-readiness audit. The feature dump is `tools/run_e233_eflic_decoder_safe_branch_features.py`; it records only predecision context available to both encoder and decoder before each y-slice quantization decision: z-hat statistics, z-index statistics, and per-slice summaries of mean, scale, support buffer, and previous decoded support. It explicitly excludes current y-index/residual outcomes as controller inputs.

The full feature dumps were run on GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`). Kodak24 and CLIC professional 41 both completed with zero nonfinite rows: `experiments/analysis/e233_eflic_kodak24_decoder_safe_branch_features.{csv,json,md}` has 24 images, and `experiments/analysis/e233_eflic_clicpro41_decoder_safe_branch_features.{csv,json,md}` has 41 images. GPU1 was not used.

I then added `tools/analyze_e233_eflic_decoder_safe_branch_label_readiness.py`, joining the E233 safe features with the E232 branch-candidate oracle. The audit uses 65 images, 22 codec-valid candidates, and 327 safe feature columns after excluding IDs and image-shape columns to avoid dataset-proxy leakage. Oracle family labels are diverse: `constant=19`, `soft_support=16`, `soft_prev=10`, `sparse_support=10`, `sparse_prev=8`, and `zero=2`.

The key result is mixed in exactly the useful way. Same-table ridge score prediction can almost harvest the candidate library (`pooled_resub ridge_score_topall_l210` score `-0.003251` vs oracle `-0.003725`, much better than best fixed `-0.000234`), showing that decoder-safe context contains real branch-selection signal. But leave-one-image-out drops to `-0.000122`, and leave-dataset-out is weak or harmful (`train CLIC -> Kodak` about `+0.000087`; `train Kodak -> CLIC` about `-0.000161`). Nearest-centroid family selection is weaker.

Decision: do not promote a post-hoc handcrafted/linear selector as paper-main. E233 supports the next implementation target more strongly: a codec-path learned branch/strength controller after EF-LIC `_mean_scale(support_buf, i)`, trained with actual reconstruction/perceptual loss and explicit false-positive control. The safe feature families with the strongest oracle-family separation are support/scale/mean statistics such as `slice0_support_rms_std` (`R2=0.192`), support tail statistics, and later-slice support-over-scale/prev-over-scale statistics, which are good candidates for the learned local head input.

Artifacts:

- `tools/run_e233_eflic_decoder_safe_branch_features.py`
- `tools/analyze_e233_eflic_decoder_safe_branch_label_readiness.py`
- `experiments/analysis/e233_eflic_kodak2_decoder_safe_branch_features_smoke.{csv,json,md}`
- `experiments/analysis/e233_eflic_kodak24_decoder_safe_branch_features.{csv,json,md}`
- `experiments/analysis/e233_eflic_clicpro41_decoder_safe_branch_features.{csv,json,md}`
- `experiments/analysis/e233_eflic_decoder_safe_branch_label_readiness.{summary.csv,predictions.csv,labels.csv,feature_separation.csv,json,md}`

## E234 EF-LIC Branch-Controller Scaffold

Status: Done for the codec-path branch-controller scaffold. I added
`tools/run_e234_eflic_branch_controller_scaffold_smoke.py`, which wraps the
E232 branch vocabulary behind a single controller preset interface: zero
fallback, aggressive constant geometry, sparse previous-support geometry,
sparse support geometry, smooth previous-context geometry, and smooth
support/scale geometry. This is not a trained controller yet; it is the
implementation contract needed before replacing the fixed preset selector with a
learned local head.

The full Kodak24 and CLIC professional 41 runs used GPU0 only
(`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`) with perceptual metrics enabled.
No run used GPU1. Every preset on both splits preserved unchanged payload
length/bpp, exact decoder reproduction (`max_decode_diff=0`), and zero
nonfinite rows. The zero preset exactly reproduced the EF-LIC baseline, so the
scaffold isolates HCG branch geometry rather than bitstream or evaluation
artifacts.

Kodak24 confirms the known aggressive-geometry behavior in the unified
controller interface. `constant020` is the best fixed preset in this subset with
`dDISTS=-0.000835`, `dLPIPS=-0.000070`, and score `-0.001047`. The smooth
support branch is close but weaker (`soft_support020` score `-0.000828`), while
weak previous-support branches are much smaller. This matches E232: Kodak-like
images can often tolerate stronger all-position geometry.

CLIC professional 41 confirms the opposite risk profile. `constant020` is not a
safe fixed rule on CLIC (`score=+0.000435`) despite unchanged bpp and exact
decode. Conservative previous-context control is best: `sparse_prev005` reaches
`dDISTS=-0.000148`, `dLPIPS=-0.000033`, score `-0.000248`; `soft_prev010` is
nearby at score `-0.000222`, and `sparse_prev010` is also useful at score
`-0.000220`. This cleanly preserves the earlier conclusion that CLIC needs
false-positive control rather than all-on geometry.

I also added `tools/analyze_e234_eflic_branch_controller_scaffold.py` to separate
fixed-preset ablations from the per-image oracle over the same E234 vocabulary.
All 455 rows are codec-valid. The pooled best fixed preset is `sparse_prev010`
(score `-0.000234`), while the oracle over only these seven executable presets
reaches score `-0.002815`, with diverse choices across `constant`, `soft_prev`,
`soft_support`, `sparse_prev`, `sparse_support`, and `zero`. This is smaller than
the full E232 oracle because the branch set is intentionally compact, but it is
still large enough to justify training a controller on this exact interface.

Decision: E234 upgrades the E232/E233 design conclusion into an executable
codec-path interface. The next implementation target should be E235: a learned
decoder-reproducible local branch/strength controller after EF-LIC
`_mean_scale(support_buf, i)`, trained with actual codec/perceptual loss and an
explicit fallback/false-positive cost. E234 should appear in the paper package
as implementation-safety evidence and as the ablation interface for fixed
branches, not as the final method.

Artifacts:

- `tools/run_e234_eflic_branch_controller_scaffold_smoke.py`
- `tools/analyze_e234_eflic_branch_controller_scaffold.py`
- `experiments/analysis/e234_eflic_branch_controller_scaffold_summary.{md,json}`
- `experiments/analysis/e234_eflic_branch_controller_scaffold_summary_{fixed,oracle}.csv`
- `experiments/analysis/e234_eflic_kodak2_branch_controller_scaffold_smoke.{csv,json,md}`
- `experiments/analysis/e234_eflic_clicpro2_branch_controller_scaffold_smoke.{csv,json,md}`
- `experiments/analysis/e234_eflic_kodak24_branch_controller_scaffold.{csv,json,md}`
- `experiments/analysis/e234_eflic_clicpro41_branch_controller_scaffold.{csv,json,md}`

## E235 EF-LIC Compact Controller Readiness

Status: Done for the compact controller-readiness audit. I added
`tools/analyze_e235_eflic_compact_controller_readiness.py`, which joins the E233
decoder-safe feature tables with the E234 executable no-sidebit preset rows.
The goal is to test whether the compact E234 controller vocabulary can be
selected from decoder-reproducible context without relying on a paper-unsafe
post-hoc oracle.

The audit covers 65 images: Kodak24 plus CLIC professional 41. It uses 327
safe feature columns and the seven E234 presets: `zero`, `constant020`,
`sparse_prev005`, `sparse_prev010`, `sparse_support010`, `soft_prev010`, and
`soft_support020`. The per-image oracle over this compact vocabulary remains
large: pooled score `-0.002815`, Kodak24 `-0.003915`, and CLIC professional
`-0.002171`. Oracle choices are diverse (`constant=18`, `soft_prev=13`,
`soft_support=14`, `sparse_prev=11`, `sparse_support=6`, `zero=3` pooled),
which preserves the need for a controller.

The positive result is that the signal is readable in resubstitution. A ridge
score predictor over all decoder-safe features reaches pooled score
`-0.002548`, close to the compact oracle and far better than the best fixed
pooled preset `sparse_prev010` at `-0.000234`. This confirms that the E234
branch choices are not random; support/previous-support RMS, mean RMS, and
z-index usage features separate useful families.

The negative result is more important for the paper path. In pooled
leave-one-image-out, the same compact-selector family collapses to about
`-0.000116` at best, and train-domain/test-domain transfer is unstable:
training on CLIC and testing Kodak gives only `-0.001355` with all features,
while training on Kodak and testing CLIC is weak or harmful (`+0.000080` to
`+0.000102` for all-feature ridge/fallback). A simple fallback margin does not
solve this. Therefore, E235 reinforces the E233 conclusion: do not promote a
post-hoc image-level selector as paper-main.

Decision: the next implementation should move inside the codec path. E236
should implement a decoder-reproducible local controller head after EF-LIC
`_mean_scale(support_buf, i)` that predicts branch/strength maps from local
support/scale/mean context, with explicit false-positive and zero-fallback
regularization. E235 is useful as a controller-design audit and feature
diagnostic, not as a final performance row.

Artifacts:

- `tools/analyze_e235_eflic_compact_controller_readiness.py`
- `experiments/analysis/e235_eflic_compact_controller_readiness.{md,json}`
- `experiments/analysis/e235_eflic_compact_controller_readiness.{summary,predictions,labels,feature_separation}.csv`

## E236 EF-LIC Local Controller-Map Smoke and Full Audit

Status: Done for the next codec-path local-controller diagnostic. I added
`tools/run_e236_eflic_local_controller_map_smoke.py`, which composes local HCG
alpha maps inside the EF-LIC sequential encode/decode path immediately after
`_mean_scale(support_buf, i)`. The policies are still hand-coded diagnostics,
not the final learned controller, but they test the exact decoder-safe interface
needed for a trained local head: zero fallback, all-position `constant020`,
soft previous/support blends, sparse previous/support union, and guarded
support/constant variants.

The smoke and full runs used GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `--device
cuda:0`) on Kodak24 and CLIC professional 41 with perceptual metrics enabled.
All 520 full rows are codec-valid: unchanged bpp, exact forward/decode match
(`max_decode_diff=0`), unchanged payload length, and zero nonfinite rows. This
keeps the E234 safety contract while allowing richer local alpha-map
composition.

The fixed-policy result is intentionally mixed. On CLIC professional,
`constant020` remains harmful under `DISTS + 3*LPIPS` (`+0.000435`), while the
guarded constant policy improves the split (`guarded_constant020_support25`
score `-0.000263`) by reducing y-index mismatch from about `0.197` to `0.073`.
On Kodak24, the opposite side of the tradeoff remains: `constant020` is still
best among fixed E236 policies (`-0.001047`), while guarded/local policies give
smaller gains or become neutral. The best Kodak local guard is
`guarded_support020_top50` at `-0.000485`, which is useful but only about half
of the all-on score gain.

The per-image oracle over E236 policies is the main design signal. It reaches
`-0.002913` pooled over 65 images, slightly above the compact E234 oracle
(`-0.002815`), with diverse choices: `constant020=20`,
`guarded_constant020_support25=13`, `guarded_support020_top50=9`,
`soft_prev010_support020_mean=7`, `sparse_prev_support010_union=6`,
`soft_prev_support010_max=4`, `hybrid_prev005_support010=1`, and `zero=5`.
This says the new local compositions add real headroom, but no fixed
hand-coded controller harvests it robustly.

Risk correlations further match the existing diagnosis. On CLIC, higher
y-index mismatch and stronger index usage correlate with worse score
(`corr(y_mismatch_frac, score)=+0.122`), while on Kodak the sign flips
(`-0.120`). Therefore the next paper-main implementation should not be another
fixed local policy. It should train a decoder-reproducible local branch/strength
controller with explicit false-positive/fallback regularization, using E236
policies as basis states or ablation references.

Artifacts:

- `tools/run_e236_eflic_local_controller_map_smoke.py`
- `tools/analyze_e236_eflic_local_controller_map.py`
- `experiments/analysis/e236_eflic_kodak2_local_controller_map_smoke.{csv,json,md}`
- `experiments/analysis/e236_eflic_clicpro2_local_controller_map_smoke.{csv,json,md}`
- `experiments/analysis/e236_eflic_kodak24_local_controller_map.{csv,json,md}`
- `experiments/analysis/e236_eflic_clicpro41_local_controller_map.{csv,json,md}`
- `experiments/analysis/e236_eflic_local_controller_map_summary.{md,json}`
- `experiments/analysis/e236_eflic_local_controller_map_summary_{fixed,oracle,oracle_choices,risk_correlations}.csv`

## E237 EF-LIC Local Policy Controller Split Audit

Status: Done for the E236 policy-controller readiness gate. I added
`tools/analyze_e237_eflic_local_policy_controller_split.py`, which joins the
E236 executable local-policy rows with the E233 decoder-safe feature table. The
selector inputs are intentionally restricted: global decoder-safe support/mean/
scale/z features plus E236 alpha-map design statistics. Outcome fields such as
DISTS/LPIPS deltas, y-index mismatch, residual errors, and index outcomes are
used only as targets or diagnostics, not as inference features.

The result is a useful negative result for shallow image-level control. The
pooled E236 oracle remains strong (`score=-0.002913`, win fraction `0.923`), but
the split-safe long-ridge score predictor does not harvest it. In pooled
leave-one-image-out, the best fallback variant among the reported key rows is
`long_ridge_fallback_top64_l225` with `score=+0.000505`, still worse than zero
and far from the oracle. Cross-dataset transfer also fails to recover the
headroom: training on CLIC and testing Kodak chooses the conservative guarded
constant policy and gives `+0.000040`, while training on Kodak and testing CLIC
keeps too much aggressive constant behavior unless fallback suppresses it, with
the best reported fallback still only `+0.000083`.

The important positive control is `true_family_train_best_policy`. If the oracle
family is known and only the within-family policy is selected from the training
side, the score is close to the E236 oracle: pooled LOIO `-0.002675` vs oracle
`-0.002913`; train CLIC -> Kodak `-0.003587` vs `-0.004095`; train Kodak ->
CLIC `-0.002114` vs `-0.002220`. This says the missing component is reliable
family/activation selection, not fine policy selection inside a family.

Top family-separating safe features are mechanistically plausible: `z_index_used_frac`,
`slice0_scale_rms_max`, `slice1_mean_abs_p95/std`, `slice0_mean_abs_std`, and
`z_index_perplexity`. This supports the HCG-RVQ thesis that hyperprior/context
contains geometry-control signal, but the signal must be learned locally and
regularized against false positives rather than fitted as a small post-hoc
image-level table.

Decision: E237 should not be a performance row. It is a design audit proving
that E236's oracle headroom is mostly a family-selection problem. The next
paper-facing EF-LIC implementation should train an in-codec local family/strength
controller after `_mean_scale(support_buf, i)`, with zero fallback and
false-positive regularization, rather than adding another hand-written fixed
policy or shallow image-level selector.

Artifacts:

- `tools/analyze_e237_eflic_local_policy_controller_split.py`
- `experiments/analysis/e237_eflic_local_policy_controller_split.{md,json}`
- `experiments/analysis/e237_eflic_local_policy_controller_split.{summary,predictions,labels,feature_separation}.csv`


E237 addendum: I also added a nearest-centroid oracle-family selector to test
whether the problem was merely score regression. It is still not enough. In
pooled leave-one-image-out, `nearest_family_top16` reaches only `+0.000222`,
while top64/all are also positive (`+0.000315`, `+0.000324`). Cross-dataset
family prediction remains weak: CLIC->Kodak is at best `-0.000047` with the
all-feature nearest-family row, far from the `-0.004095` oracle, and
Kodak->CLIC stays positive. This strengthens the conclusion: the useful HCG
state is mostly family-selectable, but global image-level safe features do not
select the family robustly. The next controller needs local map-level inputs and
a trained in-codec head.

## E238 EF-LIC Teacher Label Margin Audit

Status: Done for the supervision-design audit. I added
`tools/analyze_e238_eflic_teacher_label_margins.py`, which converts the E236
codec-valid local-policy oracle into teacher labels for the next learned EF-LIC
local HCG controller. It measures oracle improvement, coarse family margin,
curriculum headroom under zero fallback, and wrong-family costs. This is a
training-target audit, not a deployable selector.

The main result is encouraging for a learned controller. The E236 oracle
headroom is not made only of tiny ambiguous wins. Pooled over Kodak24 and CLIC
professional, the oracle score is `-0.002913`, the active-label fraction is
`0.923`, and `0.831` of images have at least `5e-4` score improvement over the
zero fallback. Mean family margin is `0.001077`, but `0.369` of images have
family margin below `5e-4`, so a hard all-image family classifier would be the
wrong objective.

The curriculum/fallback view is the important implementation guidance. If the
controller activates only labels with improvement at least `5e-4` and family
margin at least `5e-5`, it activates on `0.738` of pooled images and still
retains `0.912` of the oracle headroom (`score=-0.002657` vs oracle
`-0.002913`). On CLIC, a similarly conservative threshold retains about `0.930`
of oracle headroom while activating `0.732` of images. On Kodak, a high-gain
threshold of `0.002` retains `0.941` of headroom while activating `0.708` of
images. Therefore the first learned head should optimize high-confidence
nonzero activation plus explicit zero fallback, not force a nonzero family on
every image or map.

Wrong-family costs are asymmetric and large. For true zero images, activating
hybrid/soft-blend/constant families gives positive scores around `+0.0045` to
`+0.0051`. For true soft-blend images, choosing constant loses about `+0.0050`
relative to oracle, and for true sparse-union images, choosing constant loses
about `+0.0047`. Missing a strong family and choosing zero is also costly, but
false-positive nonzero activation on true-zero images is the most dangerous
failure mode. The loss should therefore weight false-positive family activation
more heavily than missed small gains.

Decision: E238 upgrades the next EF-LIC implementation target. The learned head
should output a zero/fallback logit and coarse family/strength logits at local
slice or spatial resolution. Training should start from high-confidence labels
defined by gain and family-margin thresholds, then broaden the labels only if
held-out false-positive rates remain controlled.

Artifacts:

- `tools/analyze_e238_eflic_teacher_label_margins.py`
- `experiments/analysis/e238_eflic_teacher_label_margins.{md,json}`
- `experiments/analysis/e238_eflic_teacher_label_margins.{summary,labels,curriculum,family_costs}.csv`

## E239 EF-LIC Local HCG Head Training Plan

Status: Done for the first trainable-controller bridge. I added
`hcg_rvq/eflic_local_controller.py` and
`tools/build_e239_eflic_local_head_training_plan.py`. This moves the E238
teacher-label audit into an actual EF-LIC-compatible head design: a small
zero-biased local family head that can be evaluated immediately after
`_mean_scale(support_buf, i)` using only decoder-safe maps.

The local context builder currently produces 11 channels from current `mean`,
current `scale`, support-prefix RMS statistics, previous decoded-y RMS
statistics, and slice-id bits. The head outputs the E236 coarse family
vocabulary: zero, constant, guarded constant, guarded support, soft blend,
sparse union, and hybrid. The loss includes ordinary family cross entropy plus
false-positive and missed-active penalties, and can consume the E238-derived
family cost matrix. This is deliberately conservative: the initial bias favors
zero/fallback, because E238 showed that false-positive nonzero geometry on
true-zero images is the most dangerous failure mode.

The generated E239 manifest uses the E238 high-confidence curriculum
(`gain >= 5e-4`, `family_margin >= 5e-5`). Pooled over Kodak24 and CLIC
professional, it keeps `73.8%` confident nonzero labels and retains `91.2%` of
the oracle headroom (`target_score=-0.002657` vs oracle `-0.002913`). CLIC
retains `93.0%` and Kodak retains `89.6%` of oracle headroom. The target-family
counts are zero 17, constant 19, guarded constant 9, guarded support 7,
soft blend 9, sparse union 4, and hybrid 0. Hybrid therefore remains a candidate
output class for compatibility with the E236 vocabulary, but it should not be
promoted in the first supervised pilot.

Decision: E239 is not a full performance experiment yet. It is the training
contract for E240: train only the frozen EF-LIC local family head from this
manifest, broadcast image labels spatially for the first smoke if necessary,
then refine to slice/spatial labels once encoder/decode equality and held-out
false-positive rates are stable.

Artifacts:

- `hcg_rvq/eflic_local_controller.py`
- `tools/build_e239_eflic_local_head_training_plan.py`
- `experiments/analysis/e239_eflic_local_head_training_plan.{md,json}`
- `experiments/analysis/e239_eflic_local_head_training_plan.{manifest,summary,class_weights,cost_matrix}.csv`

## E240 EF-LIC Local Head Context Export Smoke

Status: Done for the first real EF-LIC context-export smoke. I added
`tools/export_e240_eflic_local_head_contexts.py`, which runs the original
EF-LIC forward path, captures decoder-safe local context maps immediately after
`_mean_scale(support_buf, i)`, and attaches the E239 image-level target label.
The support prefix is teacher-forced with the original EF-LIC decoded prefix,
which keeps the first frozen-head training stage separated from later codec-loop
feedback effects.

Smoke command used GPU0 only:

- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_e240_eflic_local_head_contexts.py --image-dir experiments/data/kodak24 --dataset kodak24 --max-images 1 --device cuda:0 --output-dir experiments/analysis/e240_eflic_local_head_contexts_smoke`

Result: `kodim01.png` exported successfully with context shape `4x11x16x24`,
`float16` storage, finite context fraction `1.000000`, and target family
`constant` / target index `1`. No NaN/non-finite context was observed, and
`nvidia-smi` showed no remaining compute process after completion.

Decision: E240 validates the data path for E241 frozen-head training. The next
step is to export the full Kodak24 and CLIC professional context sets with GPU0
fixed, train only `LocalHCGFamilyHead` from those tensors, and evaluate held-out
false-positive control before integrating the head into EF-LIC compress/decompress.

Artifacts:

- `tools/export_e240_eflic_local_head_contexts.py`
- `experiments/analysis/e240_eflic_local_head_contexts_smoke/kodak24__kodim01.pt`
- `experiments/analysis/e240_eflic_local_head_contexts_smoke/manifest_kodak24_n1.{md,json,csv}`

## E241 EF-LIC Frozen Local Head Training Audit

Status: Done for the first frozen-head training audit on real EF-LIC context
tensors. I exported all Kodak24 images with the E240 path on GPU0 only:
`experiments/analysis/e240_eflic_local_head_contexts_kodak24` contains 24
finite `[4, 11, H, W]` context tensors and a manifest with target counts
constant 8, guarded constant 3, guarded support 2, soft blend 5, and zero 6.
No non-finite contexts were observed.

I added `tools/train_e241_eflic_local_head_from_contexts.py` to train only
`LocalHCGFamilyHead` from those frozen tensors. This is deliberately not a
paper performance experiment; it checks whether the E239 image-level teacher
labels are good enough to supervise the first local head.

The answer is negative but useful. Pixelwise/broadcast supervision collapses to
a frequent nonzero family even with strong false-positive and zero-class
weighting. Validation false-positive nonzero rate stays at `1.0`, and the head
predicts nonzero on every held-out sample. Switching to image-level pooled loss
lets the head fit the train split better, but four-fold validation still has
mean accuracy `0.125`, mean family cost `0.006830`, predicted nonzero fraction
`0.750`, false-positive nonzero rate `1.0`, and missed-active rate `0.3125`.

Decision: E241 validates the training plumbing but rejects image-level labels
as paper-main supervision for a local controller. The next implementation step
should generate slice/spatial teacher labels or a binary activation teacher
from the E236 local-policy outputs, then train a two-stage controller:
zero-vs-active first, conditional family/strength second. This directly follows
the E238 risk analysis, where false-positive nonzero geometry on true-zero
images was the most dangerous failure mode.

Artifacts:

- `tools/train_e241_eflic_local_head_from_contexts.py`
- `experiments/analysis/e240_eflic_local_head_contexts_kodak24/manifest_kodak24_n24.{md,json,csv}`
- `experiments/analysis/e241_eflic_local_head_kodak24_training_audit.{md,json,summary.csv}`
- `experiments/analysis/e241_eflic_local_head_kodak24_split*.{md,json,summary.csv,predictions.csv}`

## E242 EF-LIC Spatial Teacher and Map-Level Head Audit

Status: Done for the first map-level supervision bridge. I added
`tools/export_e242_eflic_spatial_teacher_contexts.py`, which re-runs EF-LIC at
the same decoder-safe insertion point as E240 and stores both local context maps
and exact E236 policy alpha maps. The nonzero family is still confidence-gated
by E239; inactive alpha positions become explicit zero/fallback labels in
`target_map`.

Full Kodak24 export used GPU0 only and produced 24 finite tensors. Mean active
target fraction is `0.614583`, mean alpha is `0.009137`, target-family counts
are constant 8, guarded constant 3, guarded support 2, soft blend 5, and zero 6.
Teacher-policy counts are constant020 8, guarded constant 3, guarded support 2,
soft-prev/support variants 5, and zero 6. This fixes the main E241 label issue:
the training target is now spatially sparse where the policy is sparse or
fallback-heavy.

I also added `tools/train_e242_eflic_local_head_from_spatial_teacher.py` to train
only `LocalHCGFamilyHead` from the E242 `target_map`. The result is informative
but not yet paper-main. Conservative weighting collapses toward zero on the
held-out split (`pred_active_frac=0.000000`, `missed_active_frac=1.000000`),
while stronger active/missed-active weights flip toward all-active behavior
(`pred_active_frac` near `1.0`, false-positive nonzero near `1.0`). Therefore
map-level labels are necessary but not sufficient for a single multiclass head.

Decision: E242 upgrades the next implementation target again. The controller
should be split into a calibrated binary activation head and a conditional
family/strength head. The binary head should be tuned against false-positive
nonzero rate first; family prediction should be trained only on active teacher
regions. This matches the E238 cost asymmetry and avoids both E241's all-nonzero
collapse and E242's zero/all-active tradeoff.

Artifacts:

- `tools/export_e242_eflic_spatial_teacher_contexts.py`
- `tools/train_e242_eflic_local_head_from_spatial_teacher.py`
- `experiments/analysis/e242_eflic_spatial_teacher_contexts_kodak24/manifest_kodak24_n24.{md,json,csv}`
- `experiments/analysis/e242_eflic_local_head_kodak24_map_split_rem0*.{md,json,summary.csv,predictions.csv}`
- `experiments/analysis/e242_eflic_spatial_teacher_training_audit.{md,json,summary.csv}`

## E243 EF-LIC Binary Activation Head Calibration Audit

Status: Done for the first two-stage-controller activation gate audit. I added
`LocalHCGActivationHead` and `binary_activation_loss` to
`hcg_rvq/eflic_local_controller.py`, plus
`tools/train_e243_eflic_activation_head.py`. This isolates the zero-vs-active
HCG reliability decision from family prediction, using the E242 spatial
`target_map` labels. It still does not change EF-LIC R-D training; it is a
frozen-head calibration gate before codec-loop insertion.

The first rem0 local-only run is weak but informative: train AUROC/AUPRC is
`0.661/0.713`, val AUROC/AUPRC is `0.572/0.726`. Thresholds chosen to keep train
FPR below `0.10` transfer to val with only `0.144` recall, while F1-oriented
thresholds reach high recall but have very high FPR. This says a tiny local
head can read some signal, but not enough for safe deployment.

I then added decoder-safe global-summary augmentation inside the E243 training
script. On rem0, this improves the FPR-constrained behavior somewhat: the
train-FPR<=`0.10` threshold gives val recall `0.261` at FPR `0.131`, compared
with local-only recall `0.144` at FPR `0.185`. However, the four-fold result is
not stable. Global-summary four-fold val AUROC mean/min/max is `0.4595` /
`0.3507` / `0.5955`, and train-selected conservative thresholds still miss most
active regions. F1-oriented thresholds retain high recall but keep high false
positive rates.

Decision: E243 validates the two-stage decomposition as the right diagnostic
unit, but the current activation head is not yet strong enough for EF-LIC
compress/decompress insertion. The next EF-LIC step should enrich the activation
features or architecture before codec-loop RD: e.g. a true global pooling branch,
z-prior/q-index summaries, slice-sequential state, or independent non-Kodak fit
labels. The family/strength head should remain blocked until activation
calibration improves.

Artifacts:

- `hcg_rvq/eflic_local_controller.py`
- `tools/train_e243_eflic_activation_head.py`
- `experiments/analysis/e243_eflic_activation_head_kodak24_rem*_fp4_miss2_globalsum.{md,json,thresholds.csv,images.csv}`
- `experiments/analysis/e243_eflic_activation_head_kodak24_globalsum_4fold.{md,json,threshold_summary.csv,auc.csv}`

## E244 EF-LIC Cross-Slice Activation Head Audit

Status: Done for the first stronger decoder-safe activation architecture audit.
I added `tools/train_e244_eflic_crossslice_activation_head.py`, which stacks all
four EF-LIC slice context maps into one `[44, H, W]` input and predicts four
activation maps jointly. The goal was to test whether missing cross-slice/global
state explained E243's weak activation calibration before touching EF-LIC's
R-D/perceptual training loop.

The result is a useful negative. Four-fold Kodak24 validation shows strong
overfitting: train AUROC/AUPRC mean is `0.958515` / `0.974339`, while validation
AUROC/AUPRC mean is only `0.269167` / `0.501585`. Thresholds selected on train
do not transfer reliably. For example, the train-FPR<=`0.10` row has validation
recall `0.251272` at FPR `0.733803`, and the fixed `0.25` threshold mostly
activates too much area with validation FPR `0.892857`.

Decision: cross-slice/global context by itself is not enough, and a larger
learned activation head on Kodak24 teacher labels should not be promoted into
codec-loop training. The next EF-LIC strengthening step needs independent fit
labels, richer decoder-safe signals such as z-prior/q-index summaries, or an
R-D-dominant end-to-end objective where activation is a weak auxiliary rather
than a hard teacher.

Artifacts:

- `tools/train_e244_eflic_crossslice_activation_head.py`
- `experiments/analysis/e244_eflic_crossslice_activation_head_kodak24_rem*_fp4_miss2.{md,json,thresholds.csv,images.csv}`
- `experiments/analysis/e244_eflic_crossslice_activation_head_kodak24_4fold.{md,json,auc.csv,threshold_summary.csv}`

## E245 EF-LIC Activation Feature CV Audit

Status: Done for the follow-up signal audit. I added
`tools/analyze_e245_eflic_activation_feature_cv.py` to test whether any single
decoder-safe EF-LIC context channel can separate the E242 active-vs-zero
regions under image-held-out cross-validation. I also corrected the AUC
calculation to handle constant/tied slice-bit features as `0.5` AUROC, avoiding
spurious strength from constant cues.

The result explains why E243/E244 are fragile. Simple feature thresholds can
either recover active regions by activating almost everything, or keep false
positives low by missing almost all active regions. The validation `best_f1`
selector reaches F1 `0.739571` and recall `0.971469`, but only with FPR
`0.978462`. The validation min-weighted-risk selector controls FPR to
`0.061502`, but recall falls to `0.027936` and F1 to `0.051218`. The
FPR<=`0.10` selector gives validation recall `0.131747` at FPR `0.161185`.

Decision: the bottleneck is not just neural head capacity. The present
decoder-safe mean/scale/support context is insufficient for a reliable
paper-main activation controller when supervised only by Kodak24 teacher maps.
For EF-LIC, the next useful implementation should add richer decoder-known
state or generate independent training labels from a non-Kodak split before
full codec-loop insertion. For GLC, this reinforces the same lesson from the
sparse active-state diagnostics: a reliability selector must be bit-aware and
trained on enough independent data, not simply made larger on the evaluation
set.

Artifacts:

- `tools/analyze_e245_eflic_activation_feature_cv.py`
- `experiments/analysis/e245_eflic_activation_feature_cv_kodak24.{md,json,folds.csv,summary.csv,top_train.csv,auc.csv}`

## E242/E244/E245 CLIC Professional Spatial Teacher Extension

Status: Done for extending the EF-LIC spatial-teacher activation audits beyond
Kodak. I found the CLIC professional validation images at
`/dpl/clic/professional/valid` and exported the same E242 decoder-safe context
and spatial teacher tensors for all 41 images using GPU0 only:
`experiments/analysis/e242_eflic_spatial_teacher_contexts_clicpro41`.
All exported tensors are finite. Mean active target fraction is `0.487835`, mean
alpha is `0.007623`, and target family counts are constant 11, guarded constant
6, guarded support 5, soft blend 4, sparse union 4, and zero 11.

I then reran the E245 single-feature CV on CLIC. The conclusion matches Kodak:
F1-oriented thresholds recover active regions by activating almost everything
(validation `best_f1` F1 `0.628781`, recall `0.985761`, FPR `0.968395`), while
conservative thresholds miss almost all active regions (min-weighted-risk recall
`0.045241` at FPR `0.034851`). This confirms that the current decoder-safe
mean/scale/support channels alone are not a robust activation signal.

Finally, I ran the E244 cross-slice activation head for four CLIC folds. It is
less overfit than Kodak but still not usable for codec-loop insertion: train
AUROC/AUPRC mean is `0.787321` / `0.778980`, while validation AUROC/AUPRC mean
is only `0.430026` / `0.480928`. Train-selected FPR<=`0.10` transfers to
validation recall `0.266397` at FPR `0.390393`; fixed/F1-oriented rows still
activate too much area.

Decision: CLIC confirms the diagnosis rather than rescuing the current head.
The next EF-LIC implementation should use the new CLIC tensors as independent
fit data only after adding richer decoder-known signals or changing to an
R-D-dominant objective. Promoting the present E244 head to full training would
likely test overfitting behavior, not HCG-RVQ's core quantizer-geometry claim.

Artifacts:

- `experiments/analysis/e242_eflic_spatial_teacher_contexts_clicpro41/manifest_clicpro41_n41.{md,json,csv}`
- `experiments/analysis/e245_eflic_activation_feature_cv_clicpro41.{md,json,folds.csv,summary.csv,top_train.csv,auc.csv}`
- `experiments/analysis/e244_eflic_crossslice_activation_head_clicpro41_rem*_fp4_miss2.{md,json,thresholds.csv,images.csv}`
- `experiments/analysis/e244_eflic_crossslice_activation_head_clicpro41_4fold.{md,json,auc.csv,threshold_summary.csv}`

## E246 EF-LIC Decoder-Safe Feature-Group Audit

Status: Done for the richer-signal promotion gate before full EF-LIC training.
I added `tools/analyze_e246_eflic_decoder_safe_feature_groups.py` to join the
E242 Kodak/CLIC spatial-teacher manifests with E233 decoder-safe feature
summaries. The goal was to test whether z-prior/index summaries, mean/scale
statistics, and support/state summaries can make the activation/family
controller generalize before inserting HCG-RVQ into the codec loop. Teacher
alpha/family outputs are used only as labels, not as controller inputs.

The result is a strong negative for the current frozen-controller route. Rich
feature groups have high in-table capacity: `e233_z_plus_local` and
`manifest_plus_e233_all` reach pooled resubstitution activation AUROC/AUPRC
`1.000000` / `1.000000`. However, held-out behavior collapses. In pooled
leave-one-image-out, `manifest_context_shape` gives the best min-risk row
(F1 `0.761`, recall `0.729`, FPR `0.529`), while `e233_z_plus_local` falls to
F1 `0.548`, recall `0.479`, FPR `0.765`. Conservative FPR-selected thresholds
do not transfer cleanly either; for example, `e233_z_prior` gives recall
`0.167` at FPR `0.235`, and richer local groups often return to high FPR.

Family selection is weaker still. Pooled leave-one-image-out family accuracy is
only `0.200` for `manifest_context_shape`, `0.138` for `e233_mean_scale`, and
`0.092` to `0.122` for the richer E233 groups. Kodak/CLIC cross-dataset rows
show the same issue. This means z/index-rich decoder-safe summaries alone do
not rescue the current teacher-label controller.

Decision: do not promote "E244/E245 plus z/index summaries" as the next
paper-main EF-LIC controller. E246 supports a stricter path: either create a
larger independent fit/calibration label split, or move into codec-loop
training with the original EF-LIC R-D/perceptual objective dominant and HCG
activation/family supervision used only as weak warmup/regularization. The
full-training candidate should preserve zero/scalar fallback, compare all-on,
guarded, oracle, and learned policies, and report whether any learned local
geometry gain survives held-out RD/perceptual evaluation.

Artifacts:

- `tools/analyze_e246_eflic_decoder_safe_feature_groups.py`
- `experiments/analysis/e246_eflic_decoder_safe_feature_groups.{md,json}`
- `experiments/analysis/e246_eflic_decoder_safe_feature_groups.active_summary.csv`
- `experiments/analysis/e246_eflic_decoder_safe_feature_groups.active_predictions.csv`
- `experiments/analysis/e246_eflic_decoder_safe_feature_groups.family_summary.csv`
- `experiments/analysis/e246_eflic_decoder_safe_feature_groups.family_predictions.csv`

## E247 Loss Objective Audit

Status: Done for the loss-function safety audit before EF-LIC/GLC full-training
promotion. I added `tools/analyze_e247_loss_objective_audit.py`, which scans
`configs/*.yaml` and separates core codec objective terms from teacher,
selector, anchor, and geometry/gate regularization terms.

The audit scanned `151` configs. `76` are RD/commit-only under the audit
definition (`lambda_rd`, `beta_commit`, and `mse_scale`). `46` configs include
teacher/selector or anchor losses, and `72` include geometry/gate regularizers.
This confirms that the previous short-cycle search used many diagnostic losses,
which is fine for mechanism discovery but must not be mixed into paper-main
full-training claims without explicit ablation.

Decision: EF-LIC/GLC full-training candidates should keep the original
R-D/perceptual objective dominant. Teacher, selector, anchor, and strong
geometry penalties can be used for initialization, warmup, or failure isolation,
but the main claim should be supported by runs where the HCG-RVQ branch improves
the codec under a clean objective and matched evaluation protocol.

Artifacts:

- `tools/analyze_e247_loss_objective_audit.py`
- `experiments/analysis/e247_loss_objective_audit.{md,json,csv}`

## E248 Full-Training Candidate Gate

Status: Done for converting the EF-LIC/GLC short-cycle evidence into a
paper-safe promotion matrix. I added
`tools/build_e248_full_training_candidate_gate.py`, which joins the recent
EF-LIC branch/oracle audits (E234/E236), compact/local split-controller audits
(E235/E237), feature-generalization failure gate (E246), loss audit (E247), and
GLC tail-VQ/RVQ diagnostics (E170/E171/E181).

The result keeps both EF-LIC and GLC alive as main VQ-LIC integration tracks,
but narrows what should be promoted. EF-LIC should promote a decoder-safe
compact/local HCG branch with conservative zero/scalar fallback into a
mid-scale codec-loop gate, not the frozen teacher/selector controller from
E244/E246. The evidence is that E236 has a strong pooled local-policy oracle
(`score_dists_3lpips = -0.002913`) while the best fixed row is much smaller
(`guarded_constant020_support25`, `-0.000151`), but E246 shows that current
feature classifiers do not generalize enough to carry a paper-main claim.

For GLC, E248 promotes a bit-aware q0 tail VQ/HCG branch over active residual
states. E170 shows strong q0 residual headroom with part-group K=8
(`mse_ratio = 0.245810`) but also a large empirical bpp increase
(`+0.015746`). E181 confirms decoder-aware training can improve PSNR,
MS-SSIM, and LPIPS on Kodak8 after OI16 training, but DISTS and bpp remain
fragile. Therefore dense or residual-MSE-only all-on RVQ is explicitly kept as
a mechanism probe rather than a paper-main path.

Decision: the next implementation target is not another larger offline
classifier. It is an R-D/perceptual-dominant codec-loop branch with clean
fallbacks and matched controls. Future full-training rows must include
baseline, zero/scalar fallback, all-on, fixed guarded/scalar-VQ ablations,
oracle/teacher upper bound, and learned HCG under the same split and checkpoint
policy. GPU jobs should remain fixed to GPU0.

Artifacts:

- `tools/build_e248_full_training_candidate_gate.py`
- `experiments/analysis/e248_full_training_candidate_gate.{md,json,csv}`

## E249 GLC Bit-Aware Score Gate

Status: Done for the GLC rate/perceptual promotion gate. I added
`tools/analyze_e249_glc_bitaware_score_gate.py`, which re-scores the E181
OI16-to-Kodak8 decoder-aware q0 tail VQ rows with an explicit empirical bpp
penalty:

`score = delta_DISTS + 3 * delta_LPIPS + w * delta_bpp`.

The trained E181 branch is meaningfully better than the initialized branch
under the perceptual-only combined score. `trained_eval` has
`delta_DISTS + 3 * delta_LPIPS = -0.016977`, compared with `+0.000081` for
`init_eval`. It also improves PSNR by `+0.558983`, MS-SSIM by `+0.023729`, and
LPIPS by `-0.009774` with no nonfinite rows.

The important constraint is rate. `trained_eval` still has empirical bpp delta
`+0.014548` and DISTS delta `+0.012346`. Its break-even bpp weight is only
`1.166984`; at `w=1.0` it remains slightly negative but close to zero
(`score=-0.002429`), while at `w=1.5` it becomes harmful (`+0.004845`). Therefore the current GLC branch is
promising as a mechanism, but not ready to scale as-is.

Decision: GLC should proceed with a bit-aware/index-aware q0 tail HCG/RVQ
branch. The next implementation must either reduce empirical index rate, gate
activation more selectively, or train with a codec objective that directly
penalizes rate/perceptual tradeoff. Residual-MSE-only or dense all-on variants
remain diagnostic ablations.

Artifacts:

- `tools/analyze_e249_glc_bitaware_score_gate.py`
- `experiments/analysis/e249_glc_bitaware_score_gate.{md,json,csv}`

## E250 GLC Soft-Index Bit-Aware Tail VQ Gate

Status: Done for the first differentiable rate-aware GLC q0 branch diagnostic.
I added `tools/run_e250_glc_bitaware_tail_vq_split_train.py`, extending the
E177 decoder-aware tail VQ split-train probe with a soft codebook-usage entropy
excess penalty. I also added `tools/analyze_e250_glc_bitaware_variants.py` to
aggregate the E250 variants against the E249/E181 promotion reference.

The two-image smoke initially looked only diagnostic, but the mid-scale
OpenImages8 -> Kodak8 gate is substantially stronger. The current best E250
variant is the K=8 part-group branch with LPIPS in the image loss and a stronger
soft-index penalty (`soft_index_weight=1.0`). On Kodak8 it improves PSNR by
`+0.588917`, MS-SSIM by `+0.024542`, and LPIPS by `-0.010767`, while empirical
bpp rises by `+0.013677`. Its combined perceptual score before bpp is
`DISTS+3LPIPS = -0.020319`, and after bpp it remains improved at `-0.006642`.
This beats the E181 OI16 -> Kodak8 trained reference from E249
(`score1 = -0.002429`). The matched OI16 -> Kodak8 rerun also beats E181:
`dPSNR=+0.583724`, `dMS-SSIM=+0.024469`, `dLPIPS=-0.010576`,
`dbpp=+0.014432`, `score0=-0.018201`, and `score1=-0.003769`, with
`nonfinite_rows=0`. DISTS remains the main weak point (`+0.013527`, slightly
worse than E181's `+0.012346`).

Two negative controls are important. K=4 reduces rate (`dbpp=+0.003902`) but
loses quality (`dPSNR=-0.035293`, `dLPIPS=+0.025500`, `dDISTS=+0.052740`).
A shared K=8 codebook nearly removes the rate cost (`dbpp=+0.000563`) but
destroys quality (`dPSNR=-1.225562`, `dLPIPS=+0.115932`). This means simple
rate shrinking is not enough; GLC needs selective activation or an explicit
index prior, not just smaller/shared dictionaries.

Decision: promote E250 from smoke-only diagnostic to the next GLC candidate
gate. I fixed the runner's memory behavior by freeing eval tensors before
training and by backpropagating per image instead of keeping all train graphs.
The matched OI16 -> Kodak8 rerun now completes and beats the E181 bpp-charged
score, so the next step is full Kodak and CLIC Professional scaling. All
completed E250 GPU runs were pinned to GPU0 and had `nonfinite_rows=0`; the
intermediate failed OI16 attempts were CUDA OOM, not NaN.

Artifacts:

- `tools/run_e250_glc_bitaware_tail_vq_split_train.py`
- `tools/analyze_e250_glc_bitaware_variants.py`
- `experiments/analysis/e250_glc_bitaware_tail_vq_split_train_*.{md,json,csv}`
- `experiments/analysis/e250_glc_bitaware_variant_summary.{md,json,csv}`

## E251 GLC E250 Activation Gate Analysis

Status: Done for the full-Kodak activation/headroom audit. I added
`tools/analyze_e251_glc_e250_activation_gate.py`, which reads the E250
OpenImages16 -> Kodak24 K=8 part-group LPIPS+soft-index run and scores each
image with:

`delta_DISTS + 3 * delta_LPIPS + delta_empirical_bpp`.

This confirms that the full-Kodak all-on branch is not yet a paper-main result:
it selects all `24/24` images and has mean score `+0.002014`, despite improving
PSNR by `+0.562163`, MS-SSIM by `+0.022049`, and LPIPS by `-0.008434`. The
problem is the combination of DISTS regression (`+0.012494`) and index rate
(`+0.014818`).

The important new result is that the branch is not uniformly bad. A per-image
oracle with a one-bit/image side overhead selects `11/24` images and reaches
mean score `-0.005773`. Even a single-feature threshold with leave-one-out
selection keeps a negative score (`-0.001565`) while selecting `6/24` images.
The strongest in-sample explanatory feature is base PSNR, but codec-safe
candidates such as empirical/index bpp, active residual error, and index
entropy also show nonzero selection signal. Restricting to codec-safe features,
the best in-sample rule is empirical bpp delta <= 0.015252 with score
-0.001625, but its leave-one-out score becomes +0.006953. This means a simple
threshold is not yet robust enough for a paper claim.

Decision: do not scale the E250 GLC branch as all-on. Promote it as a selective
activation/index-prior branch. The next implementation should learn or predict
where the q0 K=8 local residual codebook pays for its bits, then evaluate on
full Kodak and CLIC Professional under the same score. All analysis here is
CPU-only and inherited the E250 GPU0/nonfinite-safe artifacts.

Artifacts:

- `tools/analyze_e251_glc_e250_activation_gate.py`
- `experiments/analysis/e251_glc_e250_activation_gate.{md,json}`
- `experiments/analysis/e251_glc_e250_activation_gate.{policies,per_image}.csv`

## E252/E253 GLC CLIC Professional External Gate

Status: Done for the first GLC E250 cross-domain check. I ran the same q0 K=8
part-group branch with OpenImages16 training and CLIC Professional first-8
evaluation, pinned to `CUDA_VISIBLE_DEVICES=0` and `cuda:0`. E252 used the E250
LPIPS+DISTS+soft-index recipe; E253 increased DISTS pressure (`lpips_weight=0`,
`dists_weight=2`) to test whether the CLIC failure was simply LPIPS overfitting.
Both runs had `nonfinite_rows=0` and left GPU0 clean afterward.

E252 improves PSNR by `+0.475843`, MS-SSIM by `+0.015449`, and LPIPS by
`-0.003742`, but DISTS worsens by `+0.019549` and empirical bpp rises by
`+0.016190`. The bpp-charged score is `+0.024513`, and the per-image oracle
selects `0/8` images. E253 is not a rescue: DISTS weight 2 gives score
`+0.025632`, still with oracle `0/8`.

Decision: E250 remains promising on Kodak, but it should not be promoted to CLIC
Professional by simple scaling. The GLC next step must include domain-mixed or
CLIC-calibrated activation/index-prior training, and probably a stronger
DISTS-aware reliability signal. This is a useful negative result because it
prevents a weak all-on full-training run and clarifies that CLIC Professional is
the hard external gate for the current q0 tail branch.

Artifacts:

- `experiments/analysis/e252_glc_e250_oi16_clicpro8_lpips1_w100.{md,json,csv}`
- `experiments/analysis/e252_glc_e250_oi16_clicpro8_activation_gate.{md,json}`
- `experiments/analysis/e253_glc_e250_oi16_clicpro8_dists2_w100.{md,json,csv}`
- `experiments/analysis/e253_glc_e250_oi16_clicpro8_activation_gate.{md,json}`

## E254 GLC Domain-Mixed Gate Readiness

Status: Done for the first Kodak/CLIC mixed gate-readiness audit. I added
`tools/analyze_e254_glc_domain_mixed_gate_readiness.py`, which combines the
E250 Kodak24 result with the E252 CLIC Professional first-8 result under the
same paper-facing score:

`delta_DISTS + 3 * delta_LPIPS + delta_empirical_bpp`.

The pooled primary all-on branch is clearly harmful (`+0.007638`) even though it
improves PSNR by `+0.540583`, MS-SSIM by `+0.020399`, and LPIPS by `-0.007261`.
The pooled oracle still has headroom: it selects `11/32` images and reaches
`-0.004329`, with all positives coming from Kodak. This preserves the E251
evidence that the local RVQ mode is useful on some images.

The important negative result is cross-domain generalization. The best pooled
branch-internal threshold is `active_rvq_mse >= 0.00188907`, selecting `4/32`
and scoring `-0.001061` in sample, but leave-domain-out selection becomes
harmful (`+0.001671`). When the Kodak-trained threshold is applied to CLIC, it
selects `2/8` images and scores `+0.006684`; when CLIC is the training side, the
held-out Kodak fold selects `0/24`, losing the Kodak headroom. This means the
current simple threshold controller is not ready for a paper-main method.

Decision: keep E250 as a useful GLC local-geometry branch, but do not full-train
or scale it as all-on. The next GLC implementation should train a small
hyperprior/index-prior reliability controller on a domain-mixed split, with
DISTS as a validation guard and with no-branch fallback explicitly allowed.

Artifacts:

- `tools/analyze_e254_glc_domain_mixed_gate_readiness.py`
- `experiments/analysis/e254_glc_domain_mixed_gate_readiness.{md,json}`
- `experiments/analysis/e254_glc_domain_mixed_gate_readiness.{policies,folds,groups,rows,feature_separation}.csv`

## E255 GLC Linear Controller Proxy

Status: Done for the first learned-controller proxy on the E254 mixed
Kodak/CLIC rows. I added `tools/analyze_e255_glc_linear_controller_proxy.py`,
which trains tiny logistic and linear score-regression controllers from
branch-internal and rate-proxy features, then evaluates resubstitution,
leave-one-out, leave-domain-out, and leave-variant-out policies.

The result is useful but not yet a promotion gate. In resubstitution, small
models can recover a fair amount of the oracle headroom: branch-internal logistic
selects `8/32` images and scores `-0.002896`, while the analysis-only upper
bound reaches `-0.003081`. The best held-out-image row is
`loocv_branch_plus_rate_score_regressor`, selecting `8/32` and scoring
`-0.000865`. This is weakly positive evidence that a compact controller can
learn some Kodak-side selection signal.

The cross-domain result remains the blocker. Leave-domain-out branch-internal
and branch-plus-rate controllers select `0/32` images, matching no-branch rather
than recovering the Kodak positives. Analysis-only leave-domain rows select a
few images but are harmful (`+0.001983` to `+0.003088`). This means the current
E254 rows do not contain enough domain-stable signal for a paper-main GLC
controller.

Decision: keep the linear/score-regression controller as a warmup and reporting
baseline, but do not promote it to the GLC codec loop yet. The next GLC
implementation should generate domain-mixed calibration labels by running the
E250-style branch on CLIC-like training/calibration images, then train the
controller with a DISTS/bpp guard inside or immediately adjacent to the codec
objective.

Artifacts:

- `tools/analyze_e255_glc_linear_controller_proxy.py`
- `experiments/analysis/e255_glc_linear_controller_proxy.{md,json}`
- `experiments/analysis/e255_glc_linear_controller_proxy.{summary,model_info}.csv`

## E256-E258 GLC CLIC Calibration Slice and Controller Proxy

Status: Done for the first non-overlapping CLIC Professional calibration slice.
E256 reran the E250 q0 K=8 part-group branch with OpenImages16 training and
CLIC Professional valid images `8:16`, pinned to `CUDA_VISIBLE_DEVICES=0` and
`cuda:0`. The run completed with `nonfinite_rows=0`.

The CLIC calibration slice mostly confirms the E252 external-domain failure. The
all-on trained branch improves PSNR by `+0.458254`, MS-SSIM by `+0.015674`, and
LPIPS by `-0.002097`, but DISTS worsens by `+0.030436` and empirical bpp by
`+0.016525`, giving a score of `+0.040670`. Unlike E252, the oracle is not
exactly zero: it selects `1/8` images and reaches `-0.000830`. That means CLIC
does contain a very small local-RVQ headroom signal, but it is much weaker and
more fragile than Kodak.

E257 pooled Kodak24, CLIC first-8, and the new CLIC calibration-8 rows. The
40-row all-on policy is harmful (`+0.014245`), while the oracle selects `12/40`
and scores `-0.003630`. The best branch-internal threshold selects `4/40` and
scores `-0.000849` in sample, but leave-domain-out remains harmful
(`+0.001337`). The positive rows are `11/24` Kodak, `0/8` CLIC first-8, and
`1/8` CLIC calibration.

E258 repeated the tiny learned-controller proxy on the E257 rows. Resubstitution
again shows capacity: branch-internal score regression selects `11/40` and
scores `-0.002728`. Held-out behavior is still not paper-ready. The best
deployable leave-one-image row is only weakly negative for rate-proxy score
regression (`-0.000171`), while branch-internal leave-one-image rows are harmful
(`+0.000251` to `+0.000728`) and leave-domain-out either becomes silent or
positive.

Decision: E256-E258 strengthen the case that GLC needs a trained
reliability/index-prior controller, but they do not yet justify promoting the
current E250 branch into the codec loop or full training. The next GLC step
should generate more CLIC-like calibration labels or train the controller inside
the codec objective with an explicit DISTS+bpp guard and a no-branch fallback.

Artifacts:

- `experiments/analysis/e256_glc_e250_oi16_clicpro8_calib8_16_lpips1_w100.{md,json,csv}`
- `experiments/analysis/e256_glc_e250_oi16_clicpro8_calib8_16_activation_gate.*`
- `experiments/analysis/e257_glc_domain_mixed_with_cliccalib_gate_readiness.*`
- `experiments/analysis/e258_glc_linear_controller_proxy_with_cliccalib.*`

## E259 Full-Training Readiness Gate After GLC Calibration

Status: Done. E259 consolidates EF-LIC E246, loss-objective E247, and GLC
E257/E258 into a single promotion audit for paper-safe full-training decisions.
It is explicitly a readiness gate, not a final performance claim.

The combined conclusion is conservative but useful. EF-LIC has enough local
signal to justify a learned HCG branch, but the frozen decoder-safe controller
is blocked for paper-main use: pooled resub min-risk reaches `0.000000`, while
LOIO aggregate min-risk is still `0.953846`, CLIC->Kodak min-risk is
`0.916667`, and Kodak->CLIC min-risk is `1.219512`. This says the signal exists,
but the selector must be learned inside or adjacent to the codec objective.

For GLC, E259 records the current blocker clearly: E257 all-on is harmful
(`+0.014245`) while oracle selection is helpful (`-0.003630`). E258's best
deployable LOOCV proxy is only weakly helpful (`-0.000171`) and leave-domain
remains harmful (`+0.000684`). Therefore the E250 all-on tail branch is not
ready for full-training claims.

Decision: do not launch dense/all-on EF-LIC or GLC HCG-RVQ full training yet.
The next full-training candidate should be a compact local HCG-RVQ branch with
learned reliability/index control and a no-branch fallback, trained with the
original codec objective dominant. E247 remains the loss guardrail: among 151
audited configs, 76 are RD/commit-only and 46 use teacher/selector/anchor terms,
so auxiliary-heavy losses stay diagnostic unless explicitly ablated.

Artifacts:

- `tools/build_e259_full_training_readiness_after_glc_calib.py`
- `experiments/analysis/e259_full_training_readiness_after_glc_calib.{md,json,csv}`

## E260 Reliability/Index Controller Module Probe

Status: Done. E260 adds the shared compact reliability/index controller contract
needed by the E259 promotion rule and probes it on the current GLC E257
domain-mixed rows.

Implementation:

- `hcg_rvq/reliability_index_controller.py` defines a per-image MLP controller
  and a spatial conv controller. Both output an active logit plus a signed
  branch risk score, and both are initialized toward zero/fallback.
- `reliability_index_loss` keeps the objective simple: asymmetric binary
  activation loss plus optional signed score regression. False positives are
  weighted higher because E246-E259 show all-on activation is the main failure
  mode.
- `select_with_fallback` applies the explicit no-branch fallback contract.

Smoke passed. Spatial init active probability mean is `0.124703` with selected
fraction `0.000000`; MLP init active probability mean is `0.220631` with
selected fraction `0.000000`. The loss and gradients are finite.

On the GLC E257 rows, the MLP exactly recovers the oracle in resubstitution
(`12/40`, `-0.003630`), but fails held-out protocols: LOOCV selects `15/40` and
scores `+0.003572`, while leave-domain selects `12/40` and scores `+0.003267`.
This is an important negative result: the controller class is usable, but
offline fitting on the current rows is not enough for full-training promotion.

Decision: keep the E260 controller module as the shared implementation contract
for EF-LIC/GLC, but do not use this offline MLP as the paper-main selector. The
next implementation should place the controller in or adjacent to the codec loop
with domain-mixed calibration, simple loss terms, and conservative fallback.

Artifacts:

- `hcg_rvq/reliability_index_controller.py`
- `tools/analyze_e260_reliability_index_controller_probe.py`
- `experiments/analysis/e260_reliability_index_controller_probe.{md,json,summary.csv}`

## E261 Domain-Robust Controller Calibration Audit

Status: Done. E261 tests whether a simple, interpretable threshold controller
can safely select the GLC local branch on the E257 domain-mixed rows before we
promote any branch to codec-loop/full training. Score and reconstruction-delta
columns are used only as labels; predictors are branch diagnostics such as
active residual MSE, index entropy/usage, and bpp deltas.

The audit gives a clean blocker for offline threshold control. Baselines are:
no-branch `0.000000`, all-on `+0.014245`, and oracle `-0.003630` with `12/40`
selected. The best in-sample threshold still finds a small useful subset:
`active_rvq_mse >= 0.0018890685` selects `4/40` and scores `-0.000849`.
However, this does not survive held-out checks. LOOCV with the same free/domain
guards selects `4/40` but is slightly harmful (`+0.000015`), and leave-domain
and leave-variant both select `2/40` with `+0.001337` and `0.000000` win rate.
Margin guards make the result more conservative but not useful.

Decision: do not spend full-training budget on another offline-threshold or
all-on GLC branch. E261 confirms the E259/E260 direction: the paper-main method
needs a compact reliability/index controller trained in or adjacent to the codec
loop, with no-branch fallback and the original codec objective dominant. E261 is
also a useful negative control: it shows that the headroom is real, but not
recoverable by a hand threshold over the current row diagnostics.

Artifacts:

- `tools/analyze_e261_domain_robust_controller_calibration.py`
- `experiments/analysis/e261_domain_robust_controller_calibration.{md,json,summary.csv,selected.csv}`

## E262 Controller Fallback-Mix Smoke

Status: Done. E262 adds the shared codec-loop insertion primitive for the E259-E261
promotion path. `mix_with_fallback` blends a branch output into the original
codec output through the E260 reliability/index gate; zero gate exactly recovers
the base codec, while soft gates remain differentiable for training.

Smoke results are healthy. The spatial EF-LIC-style path has soft gate mean
`0.057882`, max `0.085962`, hard gate sum `0.000000`, and exactly reconstructs
the base path under hard fallback. The per-image GLC-style path has soft gate
mean `0.077224` and max `0.105869`. Both spatial and MLP losses/gradients are
finite, and the explicit hard fallback mask selects `0` positions.

Decision: use this shared fallback-mix primitive for the next mid-scale EF-LIC
and GLC codec-loop pilots. This preserves the simple model-design rule: the
original codec objective remains dominant, and HCG-RVQ only enters through a
small, conservative, rate-aware gate.

Artifacts:

- `hcg_rvq/reliability_index_controller.py`
- `tools/run_e262_controller_fallback_mix_smoke.py`
- `experiments/analysis/e262_controller_fallback_mix_smoke.{md,json}`

## E263 GLC Fallback-Gate Codec-Loop Pilot

Status: Done. E263 is the first GLC-side codec-loop pilot after the E261 offline
controller blocker and E262 fallback primitive. It trains an E250-style local
RVQ branch and the compact reliability/index controller together, while keeping
the loss intentionally simple: image reconstruction loss remains dominant, with
only gate-weighted empirical bpp delta, weak gate sparsity, and the existing
soft codebook usage term.

The result is the strongest short-cycle evidence so far for the updated design.
Across 1-, 2-, and 4-image Kodak pilots on GPU0, dense/all-on activation pays
about `+0.014` to `+0.015` bpp and is score-harmful, while soft fallback uses
only about `+0.0013` to `+0.0016` effective bpp and improves the guarded score
on every evaluated image. The 4-image trained run gives:

- `trained_all_on`: score `+0.012332`, bpp `0.038544`, dbpp `+0.014682`,
  win rate `0/4`
- `trained_soft_gate`: score `-0.019209`, bpp `0.025438`, dbpp `+0.001576`,
  dPSNR `+0.388592`, dMS-SSIM `+0.011077`, dLPIPS `-0.006471`,
  dDISTS `-0.001372`, gate `0.107217`, win rate `4/4`
- `trained_hard_gate`: score `0.000000`, exact no-branch fallback, selected
  `0/4`

No NaNs or nonfinite rows were observed, and all runs were fixed to
`CUDA_VISIBLE_DEVICES=0` / `cuda:0`. The hard gate is still deliberately
conservative, so E263 is not yet a paper-main result. Its value is narrower but
important: it shows that the main HCG-RVQ integration path should be
codec-loop-trained fallback gating, not dense RVQ activation or offline
threshold selection.

The same pattern also holds on the CLIC Professional first-8 slice, which is the
more important promotion signal because earlier E250 all-on CLIC runs showed
clear DISTS/bpp fragility. With `train-limit=8`, `eval-limit=8`, and 8 steps,
`trained_all_on` scores `+0.034813` with dbpp `+0.015739` and win rate `0/8`,
while `trained_soft_gate` scores `-0.010313` with dbpp `+0.001313`, dPSNR
`+0.262517`, dMS-SSIM `+0.006720`, dLPIPS `-0.003879`, and win rate `8/8`.
This moves E263 from Kodak-only promise to a domain-shift smoke pass.

Artifacts:

- `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_smoke_t1_e1_s2.{md,json,csv}`
- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak2_t2_e2_s4.{md,json,csv}`
- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4_t4_e4_s8.{md,json,csv}`
- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8_t8_e8_s8.{md,json,csv}`

## E263 Held-Out Slice Extension

Status: Done. I reran the E263 GLC fallback-gate pilot on held-out image slices
instead of only the first Kodak/CLIC slices, keeping the same compact
codec-loop controller and GPU0-only execution. This checks whether the positive
short-cycle result is a first-slice artifact.

The pattern holds. On Kodak held-out images 4-7, dense all-on remains harmful
(`trained_all_on` score `+0.034114`, dbpp `+0.013975`, win rate `0/4`), while
the soft fallback gate improves every image (`trained_soft_gate` score
`-0.012564`, dbpp `+0.001233`, dPSNR `+0.316897`, dMS-SSIM `+0.010250`,
dLPIPS `-0.004279`, dDISTS `-0.000959`, gate `0.087469`, win rate `4/4`).
Hard fallback again exactly recovers the base path.

On CLIC Professional held-out images 8-15, all-on is even more clearly invalid
(`+0.055320`, dbpp `+0.016103`, win rate `0/8`), while the soft fallback gate
keeps the result non-harmful on all images (`-0.007876`, dbpp `+0.001262`,
dPSNR `+0.254267`, dMS-SSIM `+0.006816`, dLPIPS `-0.002928`, dDISTS
`-0.000354`, gate `0.079208`, win rate `8/8`). No nonfinite rows appeared.

Decision: E263 is now a stable design signal across first and held-out
Kodak/CLIC slices. The next blocker is no longer "does fallback-gated local RVQ
help at all?" but "can the final method pay the correct index/rate cost while
keeping the same sparse benefit?"

Artifacts:

- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_kodak4held_t4_e4_s8.{md,json,csv}`
- `experiments/analysis/e263_glc_fallback_gate_codec_loop_pilot_clicpro8held_t8_e8_s8.{md,json,csv}`

## E264 E263 Fallback-Gate Rate Audit

Status: Done. E264 converts the E263 soft-gate results into a readiness audit
for final bit accounting. The normal E263 score uses a diagnostic gate-scaled
bpp proxy; E264 also charges every nonzero soft-gate row the full all-on branch
bpp as a conservative upper bound.

Across all 54 soft-gate rows, the diagnostic accounting is very strong: mean
score `-0.011948`, diagnostic win rate `1.000`, and dbpp only `+0.001302`.
However, full-branch-bpp accounting changes the conclusion: mean score becomes
`+0.002064`, full-bpp win rate drops to `0.389`, and full dbpp rises to
`+0.015314`. The trained rows show the same pattern (`-0.012035` diagnostic
score vs `+0.001893` full-branch-bpp score).

Dataset detail matters. Kodak first/held-out remains close to robust even under
conservative accounting, but CLIC Professional does not: CLIC first trained is
`+0.004114` and CLIC held trained is `+0.006966` under full-branch-bpp charging.
This means the soft-gate result is real as a geometry/reconstruction signal, but
it is not yet a final paper benchmark unless the transmitted index/rate actually
tracks sparse activation.

Decision: do not promote diagnostic soft-gate bpp as final evidence. The
paper-main path should either learn hard sparse activation with final bit
accounting, or implement an entropy-coded/progressive branch where paid index
rate follows gate strength. E264 strengthens the method direction and prevents
an overclaim.

Artifacts:

- `tools/analyze_e264_e263_fallback_gate_rate_audit.py`
- `experiments/analysis/e264_e263_fallback_gate_rate_audit.{md,json,csv}`

## E265 EF-LIC Fallback-Gate Context Smoke

Status: Done. E265 adds an EF-LIC-side wiring smoke for the post-E264 promotion
path. It uses the existing E242 decoder-safe context tensors and teacher alpha
maps, instantiates the shared E260/E262 spatial reliability/index controller,
and verifies that the controller can train with finite gradients while hard
fallback exactly recovers the base/no-branch path.

This is explicitly not EF-LIC codec RD evidence. The branch target is the E242
teacher alpha map, so the experiment checks artifact provenance and controller
compatibility, not final compression performance.

Results are healthy. With 4 Kodak and 4 CLIC Professional training artifacts,
and 4+4 held-out artifacts, all checks pass. Hard fallback selects `0.000000`
and recovers the base exactly in every split. No nonfinite records appear. The
soft gate remains conservative: held-out gate mean moves from `0.187135` at
initialization to `0.141146` after the small supervised smoke. Family loss also
falls on train (`3.715882` to `2.180607`) while held-out stays finite.

Decision: EF-LIC now has the same safe controller insertion contract as GLC.
The next EF-LIC implementation should connect this controller to the real local
HCG/RVQ branch and report selected-index/rate accounting under the E264 rule.
Do not claim EF-LIC RD improvement from E265 alone.

Artifacts:

- `tools/run_e265_eflic_fallback_gate_context_smoke.py`
- `experiments/analysis/e265_eflic_fallback_gate_context_smoke.{md,json}`

## E266 E263 Hard-Policy Rate-Accounting Audit

Status: Done. E266 audits whether the original E263 soft-gated GLC
reconstructions still look useful when selected rows are charged the full
branch bpp instead of the diagnostic gate-scaled bpp proxy. This is deliberately
stricter than E263, but it still evaluates the soft-gated reconstruction, not
the all-on branch output.

The result is mixed and prevents an overclaim. Across all E263 soft-gate rows,
the diagnostic score is `-0.011948`, but charging full branch bpp moves the
selected-soft/full-bpp score to `+0.002064` with selected win rate `0.389`.
Trained rows show the same pattern: diagnostic `-0.012035` vs
selected-soft/full-bpp `+0.001893`. Kodak remains somewhat usable
(`-0.003412` trained Kodak), but CLIC Professional fails under the conservative
charge (`+0.005540` trained CLIC).

Threshold selection also does not transfer robustly. Resubstitution can find
negative scores, but first-to-held and Kodak-to-CLIC protocols become harmful
or select no useful cases. This means the original dense branch cannot be
promoted to paper-main by a simple post-hoc threshold. The next design must
lower the branch cost or make the branch progressive/entropy-coded.

Artifacts:

- `tools/analyze_e266_e263_hard_policy_rate_accounting.py`
- `experiments/analysis/e266_e263_hard_policy_rate_accounting.{md,json,csv}`

## E267 GLC Low-Rate Fallback-Gate Pilot

Status: Done. E267 tests the main post-E266 hypothesis: if the local RVQ branch
is made cheaper, the same fallback-gated reconstruction benefit may survive
conservative rate accounting. The pilot reuses the E263 codec-loop setup but
reduces the branch to `K=4` and active parts `[0, 1]`.

This is the strongest GLC direction so far. On CLIC Professional held-out
images 8-15, the trained soft gate scores `-0.009876` under diagnostic
accounting with dbpp only `+0.000227`, dPSNR `+0.295971`, dMS-SSIM
`+0.007641`, dLPIPS `-0.003089`, dDISTS `-0.000837`, and gate mean
`0.095167`. When the same rows are charged full branch bpp, the
selected-soft/full-bpp score remains negative at `-0.007632` with selected win
rate `0.875`.

On Kodak held-out images 4-7, the trained soft gate scores `-0.017813` with
dbpp `+0.000187`, dPSNR `+0.406039`, dMS-SSIM `+0.012382`, dLPIPS
`-0.005487`, dDISTS `-0.001538`, and gate mean `0.113461`. Full-branch-bpp
accounting still leaves the selected-soft/full-bpp score at `-0.016357` with
selected win rate `1.000`.

The all-on output remains a negative control, not the method. It is harmful in
both low-rate pilots (`+0.038137` on CLIC held-out and `+0.055960` on Kodak
held-out), even though its bpp penalty is now much smaller. The useful signal is
therefore specifically the low-rate branch plus soft fallback control, not
dense activation.

Decision: promote low-rate fallback-gated GLC (`K=4`, active parts `[0, 1]`) to
the next larger multi-split and selected-index/progressive-rate experiment.
Mirror this lower-rate branch design when moving EF-LIC from context smoke to
real codec-loop insertion.

Artifacts:

- `experiments/analysis/e267_glc_lowrate_fallback_gate_clicpro8held_k4_parts01_t8_e8_s8.{md,json,csv}`
- `experiments/analysis/e267_glc_lowrate_fallback_gate_kodak4held_k4_parts01_t4_e4_s8.{md,json,csv}`
- `experiments/analysis/e267_glc_lowrate_fallback_gate_clicpro8held_k4_parts01_rate_audit.{md,json,csv}`
- `experiments/analysis/e267_glc_lowrate_fallback_gate_kodak4held_k4_parts01_rate_audit.{md,json,csv}`

## E268 GLC Low-Rate First/Held-Out Split Audit

Status: Done. E268 extends the E267 low-rate GLC branch from held-out-only
checks to both first and held-out Kodak/CLIC slices. The setting remains
`K=4`, active parts `[0, 1]`, 8 codec-loop steps, GPU0-only execution, and the
same simple loss as E263/E267.

The first-slice runs match the held-out pattern. On CLIC Professional first-8,
`trained_soft_gate` scores `-0.012153` with dbpp `+0.000231`, dPSNR
`+0.309788`, dMS-SSIM `+0.007676`, dLPIPS `-0.004217`, and gate mean
`0.102628`. The all-on output remains harmful at `+0.041137`. On Kodak first-4,
`trained_soft_gate` scores `-0.019613` with dbpp `+0.000226`, dPSNR
`+0.394753`, dMS-SSIM `+0.011117`, dLPIPS `-0.006370`, and gate mean
`0.112888`; all-on is again harmful at `+0.025440`. No nonfinite rows appeared.

The combined first+held-out rate audit is the key result. Across 24 trained
soft-gate rows, selected-soft/full-bpp score is `-0.011600`, selected win rate
is `0.958`, and diagnostic soft score is `-0.013581`. The effect is not only a
Kodak artifact: trained Kodak is `-0.017096`, while trained CLIC remains
negative at `-0.008852`. Simple threshold transfer is also no longer obviously
fragile: first-to-held trained evaluation gives `-0.010269`, Kodak-to-CLIC gives
`-0.008336`, and CLIC-to-Kodak gives `-0.017096` under this selected-soft/full
bpp audit.

Decision: the low-rate fallback-gated GLC branch is now the leading short-cycle
candidate. It still is not final paper evidence because it is a soft tensor
blend and the rate audit is a conservative proxy, but it has crossed the main
E264/E266 blocker: useful reconstruction remains after charging full branch bpp
on both Kodak and CLIC first/held-out slices.

Artifacts:

- `experiments/analysis/e268_glc_lowrate_fallback_gate_clicpro8first_k4_parts01_t8_e8_s8.{md,json,csv}`
- `experiments/analysis/e268_glc_lowrate_fallback_gate_kodak4first_k4_parts01_t4_e4_s8.{md,json,csv}`
- `experiments/analysis/e268_glc_lowrate_fallback_gate_k4_parts01_firstheld_rate_audit.{md,json,csv}`

## E269 GLC Progressive-Rate Margin Audit

Status: Done. E269 turns the E268 rate-accounting result into a margin audit:
for each existing soft-gated reconstruction, it asks how much branch bpp can be
paid before the guarded perceptual score becomes non-negative. This is still an
accounting audit over soft reconstructions, not a final entropy-coded codec, but
it tells us whether selected-index or progressive coding has a realistic budget.

The low-rate branch is in the right regime. Across 24 trained low-rate rows
(`K=4`, active parts `[0, 1]`), the no-bpp score is `-0.013802`, diagnostic
score is `-0.013581`, and full-branch-bpp score remains `-0.011600` with full
win rate `0.958`. The observed full branch cost is only `+0.002202` bpp on
average, while the mean affordable-rate fraction is `7.092` and even the p10 is
`1.603`. In other words, most rows can pay the entire observed branch bpp and
still remain useful.

The original E263 high-rate branch fails the same test. Its trained rows have a
similar no-bpp reconstruction benefit (`-0.013372`), but the full branch cost is
`+0.015265` bpp, so the full-branch-bpp score becomes `+0.001893` with full win
rate only `0.370`. The CLIC split explains the failure most clearly: low-rate
trained CLIC is `-0.008852` under full bpp, while original-branch trained CLIC
is `+0.005540`. This is a rate-regime problem, not evidence that local residual
geometry is useless.

There is one low-rate CLIC failure row under full-rate accounting:
`casey-fyfe-999.png` in the held-out trained slice has full score `+0.000406`,
max rate fraction `0.877`, and therefore needs selected/progressive bit savings
or fallback. The nearby first-slice `alejandro-escamilla-6.png` row is barely
negative at `-0.000056`. These rows are now the concrete targets for the next
hard/annealed activation or progressive-rate implementation.

Decision: keep low-rate fallback-gated GLC as the primary GLC path. Do not spend
full-training budget on dense/all-on or original high-rate branch variants
unless they first pass this margin audit. The next implementation should make
the selected/progressive rate real, then repeat the same audit with actual
selected-index bpp, codebook usage, residual-stage contribution, and failure-case
reporting.

Artifacts:

- `tools/analyze_e269_progressive_rate_margin.py`
- `experiments/analysis/e269_glc_lowrate_progressive_rate_margin.{md,json,csv}`
- `experiments/analysis/e269_glc_lowrate_progressive_rate_margin_curve.csv`
- `experiments/analysis/e269_e263_original_branch_progressive_rate_margin.{md,json,csv}`
- `experiments/analysis/e269_e263_original_branch_progressive_rate_margin_curve.csv`

## E270 GLC Selected-Rate Policy Transfer Audit

Status: Done. E270 tests whether a simple hard fallback policy can remove the
remaining E269 tail before implementing a full selected-index/progressive
bitstream. A selected row pays the E269 full-branch-bpp score; an unselected row
falls back to the base and contributes zero score. Deployable policies use only
branch diagnostics such as bpp, gate, entropy, and residual MSE; oracle policies
use quality-derived margins only as upper bounds.

The result is surprisingly clean. The best deployable policy in every protocol
is a simple upper bound on `full_branch_dbpp`:

- resubstitution on trained rows: `full_branch_dbpp <= 0.00317804`, score
  `-0.011617`, selected fraction `0.958`, selected win rate `1.000`.
- first-to-held transfer: `full_branch_dbpp <= 0.00303411`, held score
  `-0.010269`, selected fraction `0.833`, selected win rate `1.000`.
- Kodak-to-CLIC transfer: `full_branch_dbpp <= 0.00242854`, CLIC score
  `-0.007675`, selected fraction `0.688`, selected win rate `1.000`.
- CLIC-to-Kodak transfer: `full_branch_dbpp <= 0.00317804`, Kodak score
  `-0.017096`, selected fraction `1.000`, selected win rate `1.000`.

This is not final codec evidence, because `full_branch_dbpp` is currently an
audit-side branch-rate diagnostic. But it is a strong implementation signal:
the remaining failure cases are mostly high-rate tail cases, and a simple
encoder-side branch-rate guard is enough to remove them in the current splits.
The next selected/progressive implementation should therefore start with a
simple rate cap and exact fallback before adding any more elaborate reliability
loss.

Artifacts:

- `tools/analyze_e270_selected_rate_policy_transfer.py`
- `experiments/analysis/e270_glc_lowrate_selected_rate_policy_transfer.{md,json,csv}`

## E271 GLC Rate-Cap Soft/All-On Codec-Loop Pilot

Status: Done. E271 turns the E270 rate-cap policy from an audit rule into a
direct codec-loop pilot output. The pilot keeps the E267/E268 low-rate branch
(`K=4`, active parts `[0, 1]`) and adds `rate_cap_soft` and
`rate_cap_all_on` rows. A selected row pays the full branch bpp; an unselected
row falls back exactly to the base. The starting cap is the E270 Kodak-to-CLIC
transfer value, `0.00242854` dbpp. All runs used `CUDA_VISIBLE_DEVICES=0` and
`cuda:0`.

On CLIC Professional held-out 8, `trained_rate_cap_soft` remains useful even
after paying selected full branch bpp: score `-0.007255`, dbpp `+0.001655`,
dPSNR `+0.252805`, dMS-SSIM `+0.006609`, dLPIPS `-0.002738`, dDISTS
`-0.000696`, selected fraction `0.750000`, and nonfinite rows `0`. The matched
negative control, `trained_rate_cap_all_on`, is strongly harmful at
`+0.035364` despite selecting the same `0.750000` fraction.

On Kodak held-out 4, the same pattern is cleaner. `trained_rate_cap_soft`
scores `-0.016177` with dbpp `+0.001619`, dPSNR `+0.405502`, dMS-SSIM
`+0.012428`, dLPIPS `-0.005394`, dDISTS `-0.001614`, selected fraction
`1.000000`, and nonfinite rows `0`. `trained_rate_cap_all_on` is harmful at
`+0.053973`.

Decision: rate capping alone is not the method. The useful branch needs the
low-rate local residual geometry and the soft/progressive amplitude control;
all-on remains a negative-control ablation. The next implementation target is
therefore an actual selected/progressive bitstream that measures selected-index
bpp and preserves exact fallback, rather than a dense all-on RVQ branch.

Artifacts:

- `experiments/analysis/e271_glc_lowrate_ratecap_clicpro8held_k4_parts01_cap00242854_t8_e8_s8.{md,json,csv}`
- `experiments/analysis/e271_glc_lowrate_ratecap_kodak4held_k4_parts01_cap00242854_t4_e4_s8.{md,json,csv}`

## E272 GLC Gate-Signal Overhead Audit

Status: Done. E272 audits how much side information can be charged for the
E271 `rate_cap_soft` output before its gain disappears. This is an accounting
audit over existing E271 rows, not a final entropy-coded codec, but it directly
informs whether the next selected/progressive implementation should signal a
scalar/coarse gate or avoid dense gate maps.

The pooled 12-row result remains negative for all tested overhead profiles, but
fine spatial maps visibly reduce the safety margin. With no gate overhead, the
pooled adjusted score is `-0.010229` and win fraction is `0.833` because exact
fallback rows score zero. Scalar gate signaling is effectively free: `scalar8`
changes the score only to `-0.010220`. Coarse maps are still safe on average:
`tile64_1bit` gives `-0.010020`, `tile32_1bit` gives `-0.009408`, and
`tile32_2bit` gives `-0.008588`. Dense maps are more risky but still negative
in this small audit: `tile16_1bit` gives `-0.006961` and `tile16_2bit` gives
`-0.003693`.

The CLIC split is the useful warning. CLIC starts at `-0.007255`; `tile32_1bit`
keeps the mean negative at `-0.006512` but drops win fraction from `0.750` to
`0.625`, while `tile16_2bit` leaves only `-0.001357` with win fraction `0.500`.
Kodak has much more margin: even `tile16_2bit` stays `-0.008365` with win
fraction `1.000`.

Decision: the next GLC implementation should not transmit a dense local gate map
unless a larger audit proves it is necessary. Prefer decoder-predicted or
coarsely signaled selection/progressive strength, with exact fallback and
measured selected-index bpp. EF-LIC should inherit the same rule.

Artifacts:

- `tools/analyze_e272_gate_signal_overhead.py`
- `experiments/analysis/e272_glc_lowrate_gate_signal_overhead.{md,json,csv}`

## E273 GLC Progressive-Extra Bitstream Accounting Pilot

Status: Done. E273 extends the E271 low-rate GLC branch (`K=4`, active parts
`[0, 1]`) with a stricter base-plus-enhancement accounting row. The previous
`rate_cap_soft` result charges a selected branch bpp proxy; E273 additionally
asks what happens if the base scalar bitstream is kept and the active RVQ indices
are sent as an extra progressive enhancement. This is intentionally conservative
and closer to an actual bitstream than a free soft tensor blend. All runs used
`CUDA_VISIBLE_DEVICES=0` and `cuda:0`; nonfinite rows were `0`.

On CLIC Professional held-out 8, the soft reconstruction signal is still strong:
`trained_soft_gate` scores `-0.009935` with dbpp `+0.000224`, dPSNR
`+0.294381`, dMS-SSIM `+0.007602`, dLPIPS `-0.003086`, and dDISTS
`-0.000902`. But paying the full active-RVQ enhancement cost flips
`trained_progressive_extra_soft` to `+0.003321` at dbpp `+0.013480`. The
matching all-on enhancement is much worse: `+0.048346`. The old E271-style
`trained_rate_cap_soft` remains negative at `-0.007259`, selected fraction
`0.750000`.

On Kodak held-out 4, the same conservative enhancement is less damaging:
`trained_soft_gate` scores `-0.017740` with dbpp `+0.000184`, and
`trained_progressive_extra_soft` remains slightly negative at `-0.003543` even
after dbpp `+0.014381`. All-on again fails badly at `+0.066243`.

Decision: the quality gain from the low-rate local branch is real, but the naive
base-plus-full-active-RVQ enhancement is too expensive on CLIC. The next GLC
implementation should reduce the active index cost rather than scaling branch
capacity upward: selected active groups, entropy-coded active indices,
stage-wise/fractional progressive coding, or a replacement-style branch where
active scalar residual bits are not also transmitted. Dense all-on remains a
negative-control ablation.

Artifacts:

- `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
- `experiments/analysis/e273_glc_lowrate_progressive_extra_clicpro8held_k4_parts01_cap00242854_t8_e8_s8.{md,json,csv}`
- `experiments/analysis/e273_glc_lowrate_progressive_extra_kodak4held_k4_parts01_cap00242854_t4_e4_s8.{md,json,csv}`

## E274 GLC Progressive-Extra Fraction Margin Audit

Status: Done. E274 audits the E273 `trained_progressive_extra_soft` rows and
asks how much of the active RVQ extra bpp can be afforded before the soft branch
quality gain is spent away. This is an accounting audit, not a final entropy
coder, but it gives a concrete target for the next bitstream implementation.

| source | images | no-extra score | full-extra score | extra bpp | active scalar bpp | replacement dbpp | afford frac mean | p10 | min | full win | half score | half win | 0.75 score | 0.75 win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIC | 8 | -0.010160 | +0.003321 | 0.013480 | 0.010965 | +0.002516 | 0.734 | 0.275 | 0.241 | 0.250 | -0.003420 | 0.625 | -0.000050 | 0.375 |
| Kodak | 4 | -0.017924 | -0.003543 | 0.014381 | 0.012978 | +0.001403 | 1.236 | 0.910 | 0.881 | 0.500 | -0.010733 | 1.000 | -0.007138 | 1.000 |
| All | 12 | -0.012748 | +0.001033 | 0.013781 | 0.011636 | +0.002145 | 0.902 | 0.309 | 0.241 | 0.333 | -0.005858 | 0.750 | -0.002412 | 0.583 |

The practical reading is sharp. If the branch is treated as pure enhancement,
CLIC cannot afford the full active RVQ index stream; on average it can afford
about `0.734` of that extra cost, but the tail can afford only `0.241`. Paying
half of the active RVQ extra cost is still negative on average (`-0.003420`) but
wins only `62.5%` of CLIC rows. Kodak is more forgiving: even a `0.75` fraction
is still safely negative.

The replacement-style margin is also informative. The active RVQ stream exceeds
the active scalar stream by only `+0.002145` bpp on average (`+0.002516` CLIC,
`+0.001403` Kodak). That means the paper-main path should not be forced into a
base-plus-enhancement framing. A selected replacement branch that avoids sending
both scalar and RVQ residual information may be much closer to the E269/E270
negative-score regime.

Next action: implement or simulate a true selected/replacement active branch and
a fractional/stage-wise progressive branch, then re-run the same CLIC/Kodak
held-out audit with measured selected-index bpp, codebook usage, residual-stage
contribution, and failure-row reporting.

Artifacts:

- `tools/analyze_e274_progressive_extra_fraction_margin.py`
- `experiments/analysis/e274_glc_lowrate_progressive_extra_fraction_margin.{md,json,csv}`

## E275 GLC Replacement-Rate Margin Audit

Status: Done. E275 reuses the same E273 `trained_progressive_extra_soft` rows,
but changes the accounting question. Instead of treating active RVQ as an extra
enhancement on top of the scalar base stream, it asks whether active RVQ could
replace active scalar residual bits. This is still an audit, not a final entropy
coder, but it is the closest current numerical proxy to the intended selected
active branch.

| source | images | no-rate score | full-extra score | replacement score | full extra dbpp | active scalar bpp | replacement dbpp | scalar saved frac | full win | replacement win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIC | 8 | -0.010160 | +0.003321 | -0.007644 | 0.013480 | 0.010965 | +0.002516 | 0.806 | 0.250 | 0.875 |
| Kodak | 4 | -0.017924 | -0.003543 | -0.016521 | 0.014381 | 0.012978 | +0.001403 | 0.900 | 0.500 | 1.000 |
| All | 12 | -0.012748 | +0.001033 | -0.010603 | 0.013781 | 0.011636 | +0.002145 | 0.838 | 0.333 | 0.917 |

The cap sweep is especially useful for implementation. With replacement dbpp
cap `0.0025`, CLIC selects `0.750` of rows, selected win rate is `1.000`, and
score is `-0.007312`; Kodak selects all rows with score `-0.016521`. Pooled
score is `-0.010382` with selected fraction `0.833` and selected win rate
`1.000`.

Decision: the next paper-main GLC implementation should prioritize selected
replacement accounting over base-plus-full-enhancement accounting. The branch
should either replace active scalar residual coding, or approximate that with a
bit-exact selected-index path whose replacement dbpp is measured. The hard CLIC
row remains `casey-fyfe-999.png`, where replacement score is still slightly
positive at `+0.000500`; this row should be the first failure-case target for a
branch-rate cap or reliability fallback.

Artifacts:

- `tools/analyze_e275_replacement_rate_margin.py`
- `experiments/analysis/e275_glc_lowrate_replacement_rate_margin.{md,json,csv}`
- `experiments/analysis/e275_glc_lowrate_replacement_rate_margin_caps.csv`

## E276 GLC Direct Replacement-Row Codec-Loop Pilot

Status: Done. E276 upgrades the E275 post-hoc replacement audit into direct
codec-loop output rows. `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
now supports `--emit-replacement-rows` and `--replacement-cap-dbpp`, emitting
`replacement_soft`, `replacement_all_on`, `rate_cap_replacement_soft`, and
`rate_cap_replacement_all_on` rows alongside the previous base, all-on,
soft-gate, rate-cap, and progressive-extra rows. All runs used
`CUDA_VISIBLE_DEVICES=0` and `cuda:0`; nonfinite rows were `0`.

On CLIC Professional held-out 8, direct replacement accounting reproduces the
E275 conclusion. `trained_soft_gate` scores `-0.009876` at dbpp `+0.000227`.
Naive full progressive extra is still harmful, `trained_progressive_extra_soft`
`+0.003378` at dbpp `+0.013480`. Direct replacement keeps the method useful:
`trained_replacement_soft` scores `-0.007584` at dbpp `+0.002518`. The
conservative cap `replacement_dbpp <= 0.0025` selects `0.750000` of images and
scores `-0.007291`. All-on replacement remains harmful at `+0.039746`.

On Kodak held-out 4, `trained_replacement_soft` scores `-0.015999` at dbpp
`+0.001407`, close to `trained_soft_gate` (`-0.017221`) and much better than
`trained_progressive_extra_soft` (`-0.003023`) or all-on replacement
(`+0.052192`).

Decision: selected replacement is now the direct pilot path, not only a
spreadsheet interpretation. The next implementation should make replacement
bit accounting more exact at the active-part/group level, then scale the same
rows to a larger CLIC/Kodak slice and later EF-LIC. The branch should still be
presented as short-cycle evidence until longer/full training confirms the trend.

Artifacts:

- `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
- `experiments/analysis/e276_glc_lowrate_replacement_rows_clicpro8held_k4_parts01_cap0025_t8_e8_s8.{md,json,csv}`
- `experiments/analysis/e276_glc_lowrate_replacement_rows_kodak4held_k4_parts01_cap0025_t4_e4_s8.{md,json,csv}`

## E277 GLC Direct Replacement Scaling Pilot

Status: Done. E277 scales the E276 direct replacement-row codec-loop pilot from
small CLIC8/Kodak4 slices to CLIC Professional held-out 16 and Kodak held-out
16. The model and accounting are unchanged: low-rate GLC local branch, `K=4`,
active parts `[0, 1]`, direct replacement rows, and conservative replacement
cap `0.0025`. All runs used `CUDA_VISIBLE_DEVICES=0` and `cuda:0`; nonfinite
rows were `0`.

On CLIC Professional held-out 16, `trained_soft_gate` remains strongly negative
at `-0.010452` with dbpp `+0.000215`, dPSNR `+0.265566`, dMS-SSIM `+0.006212`,
dLPIPS `-0.003221`, and dDISTS `-0.001003`. Naive additive progressive
accounting is still not usable: `trained_progressive_extra_soft` is `+0.003068`
at dbpp `+0.013736`. Direct replacement accounting keeps the result negative:
`trained_replacement_soft` is `-0.007795` at dbpp `+0.002872`, with win fraction
`0.875`. The old cap `0.0025` is safe but too conservative on this larger CLIC
slice: `trained_rate_cap_replacement_soft` is `-0.004878` and selects only
`0.375` of images.

On Kodak held-out 16, the same direction is stronger. `trained_soft_gate` is
`-0.014366`, `trained_replacement_soft` is `-0.012450`, and capped replacement
is `-0.010976` while selecting `0.8125` of images. Additive progressive is only
weakly negative (`-0.000668`), and all-on replacement is strongly harmful
(`+0.050286`).

Decision: the replacement-mode interpretation survives a larger held-out slice.
The next GLC action should not be to increase dense branch strength. It should
calibrate the replacement cap and make the replacement-rate accounting exact in
the bitstream. The CLIC tail suggests that the cap `0.0025` is safe but may
throw away too many useful images; a cap sweep over the direct replacement rows
is needed before selecting the next full-training candidate.

Artifacts:

- `experiments/analysis/e277_glc_lowrate_replacement_rows_clicpro16held_k4_parts01_cap0025_t16_e16_s12.{md,json,csv}`
- `experiments/analysis/e277_glc_lowrate_replacement_rows_kodak16held_k4_parts01_cap0025_t16_e16_s12.{md,json,csv}`

## E278 GLC Replacement Scaling Diagnostics

Status: Done. E278 aggregates E276 and E277 into one diagnostic table spanning
CLIC24 and Kodak20, then reports domain/source summaries, replacement-cap sweep,
codebook statistics, and worst cases. This is still short-cycle evidence, but it
is now a materially stronger protocol check than the earlier tiny splits.

Pooled over 44 images, `trained_replacement_soft` scores `-0.010195` with win
fraction `0.931818` and dbpp `+0.002406`. CLIC remains the harder split but is
still negative: CLIC24 replacement score `-0.007725`, win fraction `0.875000`,
dbpp `+0.002754`. Kodak20 is stronger: `-0.013159`, win fraction `1.000000`,
dbpp `+0.001987`. Additive progressive remains the wrong framing: pooled
`trained_progressive_extra_soft` is `+0.001212`, and CLIC is `+0.003171`.
All-on rows remain a clean negative control: pooled `trained_replacement_all_on`
is `+0.044610`.

The cap sweep gives a better next threshold than the old conservative value. At
replacement delta-bpp cap `0.0025`, pooled score is `-0.008545` with selected
fraction `0.659091` and selected win `1.000000`. Raising the cap to `0.0035`
keeps selected win `1.000000`, increases selected fraction to `0.818182`, and
improves the score to `-0.009980`. Cap `0.0040` gives the best pooled mean
score (`-0.010265`) and selects `0.954545`, but selected win drops to
`0.976190`, so `0.0035` is the safer next paper-facing cap candidate.

The codebook usage diagnostics are clean for this small active codebook: mean
index entropy is `1.670696` on CLIC and `1.792948` on Kodak, used fraction is
`1.000000`, and dead-code fraction is `0.000000`. This should not be overclaimed
as full-codebook robustness, but it means the current failure is not collapse;
it is rate/fallback calibration. The worst CLIC replacement rows are
`michael-durana-82941.png` (`+0.002223`, replacement dbpp `+0.004138`),
`casey-fyfe-999.png` (`+0.000990`, `+0.003733`), and
`jason-briscoe-149782.png` (`+0.000869`, `+0.004360`). These are exactly the
rows a `0.0035` cap should suppress.

Decision: the main GLC branch should be promoted as selected local replacement,
not additive enhancement. The next implementation target is bit-exact active
replacement: do not transmit scalar residual information for selected active
regions, entropy-code or coarsely signal the selected RVQ indices, keep fallback
exact, and report the same diagnostics at checkpoint/full-training scale.

Artifacts:

- `tools/analyze_e278_glc_replacement_scaling.py`
- `experiments/analysis/e278_glc_replacement_scaling_diagnostics.{md,json}`
- `experiments/analysis/e278_glc_replacement_scaling_diagnostics.domain_summary.csv`
- `experiments/analysis/e278_glc_replacement_scaling_diagnostics.source_summary.csv`
- `experiments/analysis/e278_glc_replacement_scaling_diagnostics.cap_sweep.csv`
- `experiments/analysis/e278_glc_replacement_scaling_diagnostics.worst_cases.csv`


## E279/E280 GLC Multi-Cap Replacement Scaling

Status: Done. E279 extends `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
with `--replacement-cap-dbpp-values`, so one codec-loop run can emit the legacy
cap row plus additional suffixed cap rows such as
`trained_rate_cap_replacement_soft_cap0p0035` and
`trained_rate_cap_replacement_soft_cap0p004`. A GPU0 smoke test passed with
nonfinite rows `0`. The CLIC Professional tail run used
`CUDA_VISIBLE_DEVICES=0` and `cuda:0`, `K=4`, active parts `[0, 1]`,
train-limit `16`, eval-start-index `32`, eval-limit `16`; the directory tail
contained 9 images, and nonfinite rows were `0`.

On this CLIC tail-9 slice, the pattern remains consistent. `trained_soft_gate`
scores `-0.009957` at dbpp `+0.000221`, while additive progressive accounting is
not valid as the main codec claim (`trained_progressive_extra_soft` `+0.003786`
at dbpp `+0.013964`). Direct replacement remains useful:
`trained_replacement_soft` scores `-0.007279` at dbpp `+0.002899`, dPSNR
`+0.260965`, dMS-SSIM `+0.006288`, dLPIPS `-0.003259`, and dDISTS `-0.000400`.
All-on remains a clean negative control: `trained_replacement_all_on` is
`+0.038589`.

The new multi-cap rows show the safety/aggressiveness tradeoff. On the tail-9
slice, cap `0.0025` selects `0.333333` and scores `-0.004619`; cap `0.0035`
selects `0.555556` and scores `-0.005123`; cap `0.0040` selects all images and
scores `-0.007279`, but has win fraction `0.888889`. Thus `0.0040` is a stronger
mean candidate on this slice, while `0.0035` remains the safer cap.

E280 aggregates E276/E277 plus the E279 CLIC tail into CLIC33 + Kodak20 = 53
images. Pooled `trained_soft_gate` is `-0.011973` with win fraction `1.000000`
and dbpp `+0.000216`. Pooled `trained_replacement_soft` is `-0.009700`, win
fraction `0.924528`, and dbpp `+0.002489`. Additive progressive is still not the
paper-main framing (`+0.001649` pooled), and all-on replacement remains harmful
(`+0.043588`).

The 53-image cap sweep clarifies the next controller candidates. Cap `0.0035`
selects `0.773585`, scores `-0.009155`, and has selected win `1.000000`. Cap
`0.0040` selects `0.962264` and improves the mean to `-0.009758`, but selected
win falls to `0.960784`. For a paper-facing controlled claim, `0.0035` is the
safe cap; for an aggressive performance candidate, `0.0040` should be evaluated
in parallel and explained through worst-case analysis.

Artifacts:

- `tools/run_e263_glc_fallback_gate_codec_loop_pilot.py`
- `tools/analyze_e278_glc_replacement_scaling.py`
- `experiments/analysis/e279_glc_multi_replacement_cap_smoke_t1_e1_s1.{md,json,csv}`
- `experiments/analysis/e279_glc_multi_cap_replacement_rows_clicpro16held32_k4_parts01_t16_e16_s12.{md,json,csv}`
- `experiments/analysis/e280_glc_replacement_scaling_plus_clic_tail_diagnostics.{md,json}`
- `experiments/analysis/e280_glc_replacement_scaling_plus_clic_tail_diagnostics.domain_summary.csv`
- `experiments/analysis/e280_glc_replacement_scaling_plus_clic_tail_diagnostics.source_summary.csv`
- `experiments/analysis/e280_glc_replacement_scaling_plus_clic_tail_diagnostics.cap_sweep.csv`
- `experiments/analysis/e280_glc_replacement_scaling_plus_clic_tail_diagnostics.worst_cases.csv`


## E281 GLC Replacement Accounting Audit

Status: Done. E281 audits the E276/E277/E279 replacement rows from the stricter bit-accounting side. It separates the empirical-index replacement score from a conservative fixed-index score, because the eventual GLC/EF-LIC paper claim cannot rely on a vague soft-gate or hidden entropy benefit. Inputs are the same CLIC33 + Kodak20 = 53 images used in E280, and no GPU was needed.

The empirical replacement result remains the strongest current GLC evidence: pooled score `-0.009700`, win fraction `0.924528`, and replacement delta-bpp `+0.002489`. CLIC is harder but still negative (`-0.007603`, win fraction `0.878788`), while Kodak is clean (`-0.013159`, win fraction `1.000000`). This confirms the earlier conclusion that the branch should be framed as selected local replacement, not additive enhancement.

The new fixed-index bound is the important caution. If active RVQ indices are charged at the fixed code length instead of empirical index entropy, the pooled score is still negative (`-0.007449`) but weaker, and the pooled win fraction drops to `0.773585`. CLIC drops from `-0.007603` to `-0.004969`, with fixed-win fraction only `0.636364`; Kodak stays robust (`-0.011542`, win fraction `1.000000`). The mean fixed-index penalty is `+0.002251` bpp pooled and `+0.002634` bpp on CLIC.

The cap audit sharpens the controller choice. Under empirical accounting, cap `0.0035` remains safe: pooled score `-0.009155`, selected fraction `0.773585`, selected empirical win `1.000000`. Under fixed-index accounting with the same selection, it is still negative (`-0.007736`) but selected fixed-win falls to `0.926829`; CLIC selected fixed-win is only `0.863636`. Cap `0.0040` gives slightly stronger empirical mean (`-0.009758`) but fixed-index selected win falls to `0.803922` pooled and `0.677419` on CLIC. Therefore, `0.0035` is the paper-facing safe cap, while `0.0040` remains an aggressive performance candidate that needs better rate/index coding or a stricter reliability head.

Failure correlations also point to rate/reliability rather than codebook collapse. Pooled empirical score correlates strongly with replacement delta-bpp (`+0.706521`) and scalar coverage (`-0.669530`), while LPIPS gain dominates the quality side (`+0.962696` correlation with score because lower LPIPS is better). The dead-code fraction remains `0.000000`. The 0.0035-to-0.0040 decision band contains two positive CLIC rows (`stefan-kunze-26931.png`, `casey-fyfe-999.png`) and several rows that are empirical-negative but become fixed-positive. That is exactly the band where a full codec needs either real selected-index coding, coarse entropy/index coding, or a better decoder-safe rate guard.

Decision: keep GLC replacement as the strongest current HCG-RVQ performance track, but do not overclaim empirical-index accounting as final bitstream evidence. The next implementation target is selected replacement with explicit index signaling/accounting. EF-LIC should inherit the same lesson even more strictly: because EF-LIC core story is entropy-coding-free fixed-index compression, HCG-RVQ must either win under fixed-index accounting or report any index statistics/coarse signaling as an explicit design choice.

Artifacts:

- `tools/analyze_e281_glc_replacement_accounting.py`
- `experiments/analysis/e281_glc_replacement_accounting_audit.{md,json}`
- `experiments/analysis/e281_glc_replacement_accounting_audit.domain_summary.csv`
- `experiments/analysis/e281_glc_replacement_accounting_audit.source_summary.csv`
- `experiments/analysis/e281_glc_replacement_accounting_audit.cap_accounting.csv`
- `experiments/analysis/e281_glc_replacement_accounting_audit.correlations.csv`
- `experiments/analysis/e281_glc_replacement_accounting_audit.worst_cases.csv`
- `experiments/analysis/e281_glc_replacement_accounting_audit.cap0035_to_0040_band.csv`

## E282 GLC Replacement Controller Transfer Audit

Status: Done. E282 checks whether the replacement-rate cap found in E280/E281
is a same-split artifact. It selects simple cap policies on one domain/source
and evaluates them on another, using the same CLIC33 + Kodak20 replacement rows
and both empirical-index and fixed-index accounting. This is CPU-only analysis;
no GPU was used.

The main finding is that Kodak-only tuning is too forgiving. Policies selected
on Kodak choose cap `0.0040`, because all Kodak selected rows remain safe. When
that cap is transferred to CLIC, the empirical score is still negative
(`-0.007697`), but selected empirical win drops to `0.935484` and selected
fixed-index win drops to `0.677419`. This is useful for aggressive performance
search, but it is not safe enough as the paper-main fixed-index controller.

CLIC-aware or all-domain fixed policies are more conservative. Training on CLIC
with `fixed_best` or `safe_empirical_best` chooses cap `0.0035`; evaluated on
all 53 images, it gives empirical score `-0.009155`, fixed score `-0.007736`,
selected empirical win `1.000000`, and selected fixed win `0.926829`. A stricter
`fixed_win095_best` policy trained on all rows chooses cap `0.0030`; it is less
negative (`-0.008627` empirical, `-0.007471` fixed) but gives selected fixed win
`0.972222`. This is the safest no-entropy/fixed-index controller candidate.

Leave-one-source confirms the tail tradeoff. On held-out CLIC tail-9, cap
`0.0040` selects all images and scores `-0.007279`, but selected empirical/fixed
wins are only `0.888889`/`0.666667`. Cap `0.0035` selects `0.555556`, scores
`-0.005123`, and keeps selected empirical win `1.000000` but fixed win only
`0.800000`. The strict `0.0030` policy selects `0.333333` and keeps both
selected wins at `1.000000`, but leaves more performance on the table.

Decision: carry two controller tracks. Paper-facing GLC should use cap `0.0035`
when empirical/coarse index accounting is allowed and report fixed-index
ablation; if the claim is fully fixed-index/no-entropy, start from cap `0.0030`
or add a better decoder-safe reliability/rate guard. EF-LIC should inherit the
stricter rule, because its core claim avoids entropy coding.

Artifacts:

- `tools/analyze_e282_glc_replacement_controller_transfer.py`
- `experiments/analysis/e282_glc_replacement_controller_transfer_audit.{md,json}`
- `experiments/analysis/e282_glc_replacement_controller_transfer_audit.domain_focus.csv`
- `experiments/analysis/e282_glc_replacement_controller_transfer_audit.leave_one_source.csv`
- `experiments/analysis/e282_glc_replacement_controller_transfer_audit.transfer.csv`

## E283 GLC Replacement Signal Overhead Audit

Status: Done. E283 adds explicit decoder-visible selection-signal overhead to
the E276/E277/E279 selected-replacement rows. This closes an important gap
between the spreadsheet replacement estimate and a paper-facing codec claim:
the decoder must know whether the HCG-RVQ replacement mode is used.

The main result is favorable. Coarse image-level signaling is effectively free
at the current image sizes. For the pooled CLIC33 + Kodak20 audit, cap `0.0035`
without a signal gives empirical/fixed scores `-0.009155`/`-0.007736`; charging
one image-level bit changes them only to `-0.009154`/`-0.007734`, and charging
eight image-level bits changes them to `-0.009144`/`-0.007724`. Selected
empirical/fixed win fractions remain `1.000000`/`0.926829`.

Dense local signaling is not free. At cap `0.0035`, a selected `64x64` tile map
still keeps pooled empirical/fixed scores negative (`-0.008962`/`-0.007542`),
but a selected `32x32` tile map weakens them to `-0.008393`/`-0.006973` and
drops selected fixed win to `0.902439`. CLIC shows the same pattern: cap
`0.0035` with image-level signaling stays at `-0.006846`/`-0.005468`, while
`64x64` tile signaling weakens it to `-0.006676`/`-0.005299`.

Decision: the next GLC implementation target should be a coarse decoder-safe
selected-replacement mode, not a dense spatial gate map. For paper-facing GLC,
cap `0.0035` remains the balanced controller when coarse signaling/index
accounting is allowed. For a stricter no-entropy/fixed-index framing, cap
`0.0030` remains safer. EF-LIC should inherit this rule because its core claim
avoids entropy coding: start with a fixed/coarsely signaled mode and only add
dense local maps if their signaling is explicitly compressed or predictable from
decoder-available state.

Artifacts:

- `tools/analyze_e283_glc_replacement_signal_overhead.py`
- `experiments/analysis/e283_glc_replacement_signal_overhead_audit.{md,json}`
- `experiments/analysis/e283_glc_replacement_signal_overhead_audit.summary.csv`

## E284 GLC Signal-Accounted Replacement Rows in Codec-Loop Pilot

Status: Done. E284 promotes the E283 signal-overhead audit from an offline spreadsheet calculation into the GLC codec-loop pilot output itself. The goal is small but important for a paper-facing claim: every selected-replacement row can now carry an explicit decoder-visible selection/mode signal cost through CSV, JSON summary, and Markdown tables.

Implementation changes in tools/run_e263_glc_fallback_gate_codec_loop_pilot.py:

- Added --replacement-signal-bits, e.g. 1 8, for image-level mode/selection signaling costs.
- Added selection_signal_bpp = signal_bits / (H * W) per image.
- Added _sig1b, _sig8b, etc. capped replacement rows while preserving the original no-signal rows.
- Added selection_signal_bpp to per-image CSV rows and summary tables.

GPU0 smoke command completed with CUDA_VISIBLE_DEVICES=0 and --device cuda:0 on one train image, one Kodak image, one q index, and one step. No nonfinite rows were observed. For trained_rate_cap_replacement_soft_cap0p003 on kodim01.png, the no-signal row has score -0.016858 at dbpp +0.000736. Adding a 1-bit image signal charges 0.00000254 bpp and gives score -0.016856; adding an 8-bit signal charges 0.00002035 bpp and gives score -0.016838. This is only a smoke result, but it verifies that signal accounting is wired into the same protocol that will be used for larger GLC runs.

Decision: use signal-accounted rows by default in the next GLC replacement pilot/full-eval audits. The paper-facing row family should include no-signal, 1-bit image signal, and 8-bit image signal variants, with cap 0.0035 as the balanced controller and cap 0.0030 as the stricter no-entropy/fixed-index controller. EF-LIC should inherit this discipline: selected HCG-RVQ modes must pay any non-reconstructible signal explicitly, especially because EF-LIC claims entropy-coding-free compression.

Artifacts:

- tools/run_e263_glc_fallback_gate_codec_loop_pilot.py
- experiments/analysis/e284_glc_replacement_signal_row_smoke_t1_e1_s1.{md,json,csv}
- experiments/analysis/e284_glc_replacement_signal_row_smoke_t1_e1_s1.trace.csv

## E285 GLC CLIC-Tail Signal-Accounted Replacement Run

Status: Done. E285 reruns the E279 CLIC professional tail condition with the new E284 signal-accounted replacement rows. This is larger than the one-image smoke but still a short-cycle design probe, not a final full-training claim. The run used CUDA_VISIBLE_DEVICES=0 and --device cuda:0 only, with train-limit 16 from OpenImages start8192, eval-start-index 32 on CLIC professional valid, q=0, K=4, active parts 0/1, and 12 training steps. The tail contained 9 images and all rows were finite.

Key trained rows on the CLIC tail-9 slice:

- trained_soft_gate: score -0.009805, dbpp +0.000216, selected 1.000000.
- trained_replacement_soft: score -0.007122, dbpp +0.002899, selected 1.000000.
- trained_replacement_all_on: score +0.039950, confirming dense all-on replacement remains the negative control.
- cap 0.0025 soft replacement: score -0.004586, dbpp +0.000572, selected 0.333333.
- cap 0.0035 soft replacement: score -0.005066, dbpp +0.001269, selected 0.555556, selected win 1.000000.
- cap 0.0035 plus 8-bit image signal: score -0.005059, signal bpp 0.00000637, selected 0.555556.
- cap 0.0040 soft replacement: score -0.007122, dbpp +0.002899, selected 1.000000, win 0.888889, worst score +0.001513.
- cap 0.0040 plus 8-bit image signal: score -0.007115, signal bpp 0.00000637, selected 1.000000, win 0.888889.

Interpretation: the E283/E284 signal-overhead conclusion survives the CLIC tail codec-loop run. Coarse image-level signal cost is negligible relative to the replacement margin. The design conclusion remains unchanged: cap 0.0035 is safer and fully wins on selected rows; cap 0.0040 is a stronger mean candidate but includes a positive tail row and must remain an aggressive branch with failure analysis.

Next action: rerun or aggregate the Kodak counterpart with the same signal rows, then update the pooled CLIC33 + Kodak20 accounting table with no-signal, 1-bit, and 8-bit variants.

Artifacts:

- experiments/analysis/e285_glc_signal_accounted_replacement_rows_clicpro16held32_k4_parts01_t16_e16_s12.{md,json,csv}
- experiments/analysis/e285_glc_signal_accounted_replacement_rows_clicpro16held32_k4_parts01_t16_e16_s12.trace.csv

## E286 GLC Kodak-Held Signal-Accounted Replacement Run

Status: Done. E286 is the Kodak-held counterpart to E285 under the same current-code signal-accounted protocol. The run used CUDA_VISIBLE_DEVICES=0 and --device cuda:0 only, with OpenImages start8192 train-limit 16, Kodak eval-start-index 8/eval-limit 16, q=0, K=4, active parts 0/1, and 12 training steps. All 16 Kodak rows were finite; no NaN/nonfinite rows were observed.

Key trained rows on the Kodak held-16 slice:

- trained_soft_gate: score -0.014267, dbpp +0.000212, selected 1.000000.
- trained_replacement_soft: score -0.012358, dbpp +0.002121, selected 1.000000.
- trained_replacement_all_on: score +0.049225, again confirming dense all-on replacement is the negative control.
- cap 0.0025 soft replacement: score -0.010885, dbpp +0.001547, selected 0.812500.
- cap 0.0035 soft replacement: score -0.012160, dbpp +0.001898, selected 0.937500, selected win 1.000000.
- cap 0.0035 plus 8-bit image signal: score -0.012140, signal bpp 0.00002035, selected 0.937500.
- cap 0.0040 soft replacement: score -0.012358, dbpp +0.002121, selected 1.000000, selected win 1.000000.
- cap 0.0040 plus 8-bit image signal: score -0.012337, signal bpp 0.00002035, selected 1.000000.

Interpretation: Kodak remains much easier than the CLIC tail. Both cap 0.0035 and cap 0.0040 are safe on this slice, and image-level signal cost is negligible. This supports the earlier caution that Kodak-only tuning would overselect the aggressive cap; CLIC is still the reliability bottleneck for a paper-facing controller.

Artifacts:

- experiments/analysis/e286_glc_signal_accounted_replacement_rows_kodak16held_k4_parts01_t16_e16_s12.{md,json,csv}

## E287 GLC Signal-Accounted Current-Subset Pooling

Status: Done. E287 adds a reusable analyzer for pooling current-code signal-accounted codec-loop rows and deriving a fixed-index reinterpretation for selected replacement rows. The input set is E285 CLIC tail-9 plus E286 Kodak held-16, for 25 images total. This is still a short-cycle subset, not a final full-training/full-evaluation claim, but it is now a cleaner controller-selection table.

Pooled 25-image focus:

- trained_soft_gate: score -0.012661, fixed score -0.010654, fixed win 0.960000.
- trained_replacement_soft: score -0.010473, fixed score -0.008466, win 0.960000, fixed win 0.880000.
- trained_replacement_all_on: score +0.045886, fixed score +0.047893.
- cap 0.0035: score -0.009606, fixed score -0.008262, selected 0.800000, selected win 1.000000, selected fixed win 0.950000.
- cap 0.0035 plus 8-bit image signal: score -0.009591, fixed score -0.008246, signal bpp 0.000015, selected win 1.000000, selected fixed win 0.950000.
- cap 0.0040: score -0.010473, fixed score -0.008466, selected 1.000000, selected win 0.960000, selected fixed win 0.880000.
- cap 0.0040 plus 8-bit image signal: score -0.010457, fixed score -0.008451, signal bpp 0.000015, selected win 0.960000, selected fixed win 0.880000.

Decision: cap 0.0035 remains the paper-safer controller for GLC because it preserves selected empirical win 1.000000 and selected fixed win 0.950000 after signal accounting. Cap 0.0040 is the aggressive performance branch: it has stronger mean score but admits CLIC-tail failures under both empirical and fixed-index views. The next EF-LIC/GLC full-training candidates should carry both tracks explicitly rather than hiding the reliability tradeoff.

Artifacts:

- tools/analyze_e287_glc_signal_accounted_current_subset.py
- experiments/analysis/e287_glc_signal_accounted_clictail9_kodak16_current_subset.{md,json}
- experiments/analysis/e287_glc_signal_accounted_clictail9_kodak16_current_subset.summary.csv

E287 strict-controller addendum: the analyzer now also derives cap 0.0030 rows from the measured trained_replacement_soft per-image rows. On the pooled 25-image subset, derived cap 0.0030 gives score -0.008977, fixed score -0.007900, selected 0.680000, selected win 1.000000, and selected fixed win 1.000000. With an 8-bit image signal, the score is -0.008962 and fixed score -0.007885. This is less aggressive than cap 0.0035/0.0040 but is currently the cleanest strict no-entropy/fixed-index controller candidate for EF-LIC-style claims.


## E288-E291 EF-LIC/GLC Promotion Audit After Signal-Accounted GLC

Status: Done. This block reconnects the prompt goal with the current VQ-LIC integration route. The core claim remains that hyperprior/context state should generate local quantizer geometry, not hidden side information. For EF-LIC, the insertion point is after `_mean_scale` and normalized slice construction and before the existing RVQ slice quantizer; the original `h_a/h_s`, representation-domain decorrelation support buffer, adaptor, context predictor, and sequential decoder loop remain in place.

EF-LIC controller wiring was re-smoked with E288 using `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0` only. The fallback-gated context smoke passed (`all_checks_passed=True`), trained eval loss was 1.724250, hard fallback exactness was 1.000000, and nonfinite records were 0. This is controller wiring evidence, not final codec RD evidence.

EF-LIC current-code codec-path contract smokes were then run on Kodak4 and CLIC Professional tail4 with the decoder-safe branch vocabulary. In both runs, the zero preset exactly preserved the baseline (`delta_bpp=0`, decode diff 0, nonfinite 0). Kodak4 showed mixed small PSNR responses: sparse_prev005 +0.008680, sparse_prev010 -0.004520, constant020 +0.025201, soft_support020 -0.026054. CLIC tail4 was stronger in this PSNR-only smoke: sparse_prev005 +0.002017, sparse_prev010 +0.002674, constant020 +0.019810, soft_support020 +0.018952, all with `delta_bpp=0`, decode diff 0, and nonfinite 0.

E291 consolidates these EF-LIC smokes with the E287 GLC signal-accounted controller table. The promotion decision is:

- GLC should promote selected replacement, not dense all-on quantization. Cap 0.0035 is the balanced paper-facing controller, cap 0.0030 is the strict fixed-index/no-entropy controller, and cap 0.0040 remains an aggressive branch with CLIC-tail failure analysis.
- EF-LIC should inherit the strict accounting discipline: HCG-RVQ may coexist with the EF-LIC no-entropy claim only when the local geometry/controller is decoder-safe or any non-reconstructible signal is explicitly charged.
- Fixed presets are useful ablations, but E235 showed weak held-out posthoc predictors; the paper-main EF-LIC route should be an in-codec learned fallback controller.
- The original codec objective should remain dominant. Auxiliary losses should be limited to VQ/index/rate and false-positive/fallback terms that are directly tied to measured codec accounting.

Next implementation target: build an EF-LIC `HCGBranchController` next to `_mean_scale`, preserving exact zero fallback and reporting selected fraction, index entropy, code usage, bpp/PSNR/MS-SSIM/perceptual metrics, intermediate geometry statistics, and failure cases on Kodak24 plus CLIC Professional. After this mid-scale check passes, promote EF-LIC and GLC to paper-aligned full training/full evaluation.

Artifacts:

- tools/build_e291_eflic_glc_hcg_promotion_audit.py
- experiments/analysis/e291_eflic_glc_hcg_promotion_audit.{md,json}
- experiments/analysis/e288_eflic_fallback_gate_context_after_glc_contract_t8_e8_s24.{md,json}
- experiments/analysis/e289_eflic_branch_controller_current_kodak4_contract_smoke.{md,json,csv}
- experiments/analysis/e290_eflic_branch_controller_current_clic_tail4_contract_smoke.{md,json,csv}


## E292-E294 EF-LIC Current Contract Scaling Audit

Status: Done. E292/E293 extend the EF-LIC current-code codec-path branch smoke from four-image probes to Kodak24 and CLIC Professional 16. All runs used `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0` only. The zero preset preserved the EF-LIC baseline exactly in both splits: `delta_bpp=0`, max decode diff 0, and nonfinite rows 0.

E292 Kodak24 results:

- `zero`: delta PSNR +0.000000, contract exact.
- `sparse_prev005`: -0.002251, win 0.375000.
- `sparse_prev010`: -0.001141, win 0.416667.
- `constant020`: +0.000791, win 0.500000, worst -0.104394 on `kodim19.png`, best +0.153662 on `kodim02.png`.
- `soft_support020`: +0.003360, win 0.541667, worst -0.099141 on `kodim19.png`, best +0.130510 on `kodim10.png`.

E293 CLIC Professional 16 results:

- `zero`: delta PSNR +0.000000, contract exact.
- `sparse_prev005`: -0.000703, win 0.437500.
- `sparse_prev010`: -0.000066, win 0.500000.
- `constant020`: -0.018294, win 0.187500, worst -0.060643 on `dogancan-ozturan-395.png`.
- `soft_support020`: -0.002556, win 0.500000, worst -0.036084 on `allef-vinicius-109434.png`.

E294 pooled conclusion over Kodak24 + CLIC16:

- `soft_support020` is the only fixed nonzero branch with positive pooled mean, but it is tiny (+0.000994 dB), only wins 0.525000 of images, and has a -0.099141 dB worst case.
- `constant020` is direct evidence against dense/all-position HCG as the main EF-LIC policy: pooled -0.006843 dB, CLIC16 -0.018294 dB, and win fraction only 0.375000 pooled.
- Sparse previous-context branches are safer in magnitude but too weak and mixed to be final methods by themselves.
- The correct EF-LIC paper-main direction is a learned decoder-safe fallback controller next to `_mean_scale`, not a fixed unconditional branch.

Interpretation: HCG-RVQ remains compatible with EF-LIC because the geometry states are decoder-reproducible and preserve the no-entropy path under zero fallback. However, unconditional geometry is not reliable enough. The next EF-LIC implementation should learn when to use local geometry while keeping the original EF-LIC objective dominant and using only directly accountable VQ/index/rate and false-positive/fallback regularizers.

Artifacts:

- tools/analyze_e294_eflic_current_contract_scaling.py
- experiments/analysis/e292_eflic_branch_controller_current_kodak24_contract.{md,json,csv}
- experiments/analysis/e293_eflic_branch_controller_current_clicpro16_contract.{md,json,csv}
- experiments/analysis/e294_eflic_current_contract_scaling.{md,json}


## E295 EF-LIC HCG Branch-Controller Integration Smoke

Status: Done. E295 inserts the new decoder-safe `EFLICHCGBranchController` into the actual EF-LIC slice/RVQ loop immediately after `_mean_scale` and before the existing slice quantizer. The original EF-LIC `h_a/h_s`, representation-domain decorrelation/support-buffer loop, adaptor/context path, and fixed-length/no-entropy payload contract remain unchanged. All runs used `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0` only. No NaN/nonfinite rows were observed.

Kodak2 results:

- `force_zero`: delta bpp +0.000000, delta PSNR +0.000000, max decode diff 0, max baseline diff 0, payload equal 1.000000.
- `init_hard`: delta bpp +0.000000, delta PSNR +0.000000, max decode diff 0, max baseline diff 0, payload equal 1.000000.
- `init_soft`: delta bpp +0.000000, delta PSNR -0.001176, max decode diff 0, nonfinite rows 0, alpha mean 0.00000012, gate mean 0.002961.

CLIC Professional 2 results:

- `force_zero`: delta bpp +0.000000, delta PSNR +0.000000, max decode diff 0, max baseline diff 0, payload equal 1.000000.
- `init_hard`: delta bpp +0.000000, delta PSNR +0.000000, max decode diff 0, max baseline diff 0, payload equal 1.000000.
- `init_soft`: delta bpp +0.000000, delta PSNR +0.003396, max decode diff 0, nonfinite rows 0, alpha mean 0.00000009, gate mean 0.002949.

Interpretation: the EF-LIC HCG controller is now wired into the codec path with exact zero fallback and conservative hard initialization. This does not yet prove performance; it proves the implementation contract needed for the next learned-controller experiment. The next step is to train/calibrate the controller using E234/E235/E292-E294 branch labels and evaluate it on Kodak24 plus CLIC Professional with PSNR/MS-SSIM/bpp, index usage, codebook/per-stage stats, and hard failure rows.

Artifacts:

- hcg_rvq/eflic_local_controller.py
- tools/run_e295_eflic_hcg_branch_controller_integration_smoke.py
- experiments/analysis/e295_eflic_hcg_branch_controller_integration_smoke_kodak2.{md,json,csv}
- experiments/analysis/e295_eflic_hcg_branch_controller_integration_smoke_clicpro2.{md,json,csv}


## E296-E297 EF-LIC Learned Controller Handoff Smoke

Status: Done. E296 trains the integrated `EFLICHCGBranchController` on the saved E242 decoder-safe teacher context tensors and saves a reloadable controller checkpoint. E297 then loads that checkpoint back into the E295 EF-LIC codec-loop smoke. All runs used `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0` only.

E296 trainability smoke used 4 Kodak + 4 CLIC Professional train records, 2 + 2 eval records, and 16 steps. It is not final RD evidence. The checkpoint was saved to `experiments/analysis/e296_eflic_hcg_controller_context_train_smoke_t4_e2_s16.pth`. Summary:

- Train: loss 1.964144, target active fraction 0.625022, predicted alpha mean 0.003067, soft gate mean 0.316592, hard gate mean 0.425841, force-zero alpha max 0, nonfinite records 0.
- Eval: loss 2.611457, target active fraction 0.375000, predicted alpha mean 0.002755, soft gate mean 0.289042, hard gate mean 0.369121, force-zero alpha max 0, nonfinite records 0.

E297 reload codec-loop smoke on Kodak2:

- `force_zero`: exact baseline reproduction, delta bpp 0, delta PSNR 0, decode diff 0, nonfinite 0.
- `trained_hard`: delta PSNR -0.000745, delta bpp 0, decode diff 0, nonfinite 0, gate mean 0.447266.
- `trained_soft`: delta PSNR +0.024214, delta bpp 0, decode diff 0, nonfinite 0, gate mean 0.281267.

E297 reload codec-loop smoke on CLIC Professional 2:

- `force_zero`: exact baseline reproduction, delta bpp 0, delta PSNR 0, decode diff 0, nonfinite 0.
- `trained_hard`: delta PSNR -0.004265, delta bpp 0, decode diff 0, nonfinite 0, gate mean 0.360510.
- `trained_soft`: delta PSNR +0.000305, delta bpp 0, decode diff 0, nonfinite 0, gate mean 0.257679.

Interpretation: the learned-controller handoff path now works end to end: context training, checkpoint save, checkpoint reload, EF-LIC codec-loop execution, decoder agreement, and zero fallback are all verified. The result is intentionally mixed and too small for performance claims. The hard branch still shows false-positive harm on CLIC, so the next real experiment should improve calibration with stricter false-positive/rate weighting, then evaluate Kodak24 + CLIC Professional with full RD and intermediate statistics.

Artifacts:

- tools/run_e296_eflic_hcg_controller_context_train_smoke.py
- experiments/analysis/e296_eflic_hcg_controller_context_train_smoke_t4_e2_s16.{md,json,trace.csv,pth}
- experiments/analysis/e297_eflic_hcg_trained_controller_reload_kodak2.{md,json,csv}
- experiments/analysis/e297_eflic_hcg_trained_controller_reload_clicpro2.{md,json,csv}


## E298-E302 EF-LIC Controller Calibration and Geometry Direction

Status: Done. E298 first tried a stricter EF-LIC learned-controller calibration: `max_alpha=0.01`, false-positive weight 12, alpha weight 80, 8 Kodak + 8 CLIC train records, 4 + 4 eval records, and 48 steps. It stayed finite and reloadable, but it was too conservative for hard decisions and too weakly helpful for soft decisions. E299 confirmed this in the codec loop:

- Kodak4 strict `trained_hard`: delta PSNR +0.000000, gate mean 0.000000.
- Kodak4 strict `trained_soft`: delta PSNR -0.009334, gate mean 0.008956.
- CLIC Professional 4 strict `trained_hard`: delta PSNR -0.000006, gate mean 0.001604.
- CLIC Professional 4 strict `trained_soft`: delta PSNR -0.001226, gate mean 0.012104.

E300 then swept the earlier E296 checkpoint at stricter hard thresholds. With the original `mean` geometry direction:

- Kodak8, threshold 0.60: delta PSNR -0.005422, gate mean 0.364339.
- Kodak8, threshold 0.80: delta PSNR -0.005448, gate mean 0.157796.
- Kodak8, threshold 0.95: delta PSNR -0.001072, gate mean 0.012614.
- CLIC Professional 8, threshold 0.80: delta PSNR -0.004139, gate mean 0.108855.
- CLIC Professional 8, threshold 0.95: delta PSNR +0.001437, gate mean 0.016994.

E301 changed only the HCG direction source from `mean` to `logscale` under the same E296 checkpoint and threshold 0.95. This materially reduced the Kodak tail while preserving the CLIC gain:

- Kodak8 `logscale`: delta PSNR -0.000032, win 0.250000, non-negative 0.875000, worst -0.000636.
- CLIC Professional 8 `logscale`: delta PSNR +0.001491, win 0.375000, non-negative 0.625000, worst -0.002002.

E302 expanded that `logscale` threshold-0.95 controller:

- Kodak24: delta PSNR -0.000292, win 0.125000, non-negative 0.750000, worst -0.006093, best +0.000387, gate mean 0.007541.
- CLIC Professional 16: delta PSNR +0.000792, win 0.250000, non-negative 0.750000, worst -0.002002, best +0.013149, gate mean 0.010425.
- All runs preserved delta bpp 0, decode diff 0, and nonfinite rows 0.

Interpretation: strict retraining alone makes the controller safe but inert. For EF-LIC, the more useful update is geometry-direction selection: `logscale` is safer than `mean` for the current HCG branch. The gains are still too small for final claims, and Kodak has a remaining hard tail (`kodim18`). The next controller should learn direction/family or include a decoder-safe image/slice risk signal rather than relying only on one active threshold.

Artifacts:

- experiments/analysis/e298_eflic_hcg_controller_strict_context_train_t8_e4_s48.{md,json,trace.csv,pth}
- experiments/analysis/e299_eflic_hcg_strict_controller_reload_kodak4.{md,json,csv}
- experiments/analysis/e299_eflic_hcg_strict_controller_reload_clicpro4.{md,json,csv}
- experiments/analysis/e300_eflic_hcg_e296_threshold0{60,80,95}_*.{md,json,csv}
- experiments/analysis/e301_eflic_hcg_e296_threshold095_logscale_*.{md,json,csv}
- experiments/analysis/e302_eflic_hcg_e296_threshold095_logscale_kodak24.{md,json,csv}
- experiments/analysis/e302_eflic_hcg_e296_threshold095_logscale_clicpro16.{md,json,csv}


## E303-E304 EF-LIC Fixed-Direction HCG Check

Status: Done. E303/E304 test the third existing HCG direction source, `fixed`, using the same E296 learned controller, threshold 0.95, and hard codec-loop mode. This isolates whether the E301/E302 gain came from scale/logscale geometry specifically or from avoiding the harmful `mean` direction.

E303 small check:

- Kodak8 `fixed`: delta PSNR -0.000086, win 0.125000, non-negative 0.750000, worst -0.000471, best +0.000020, gate mean 0.012614.
- CLIC Professional 8 `fixed`: delta PSNR +0.002085, win 0.375000, non-negative 0.625000, worst -0.000028, best +0.016579, gate mean 0.016983.

E304 expanded check:

- Kodak24 `fixed`: delta PSNR -0.000046, win 0.125000, non-negative 0.750000, worst -0.000597, best +0.000349, gate mean 0.007541, alpha mean 0.00007871.
- CLIC Professional 16 `fixed`: delta PSNR +0.001030, win 0.187500, non-negative 0.687500, worst -0.000185, best +0.016579, gate mean 0.010425, alpha mean 0.00009241.
- All runs preserved delta bpp 0, decode diff 0, and nonfinite rows 0.

Comparison to E302 `logscale` on the same splits:

- Kodak24 `logscale`: -0.000292, worst -0.006093.
- CLIC Professional 16 `logscale`: +0.000792, worst -0.002002.

Interpretation: `fixed` is currently the safest EF-LIC HCG direction among tested options. It removes the large `kodim18` logscale tail and improves CLIC16 more than logscale, although the absolute gain remains small. The next EF-LIC controller should treat direction/family choice as a learnable or explicitly ablated variable instead of hard-coding `mean`.

Artifacts:

- experiments/analysis/e303_eflic_hcg_e296_threshold095_fixed_kodak8.{md,json,csv}
- experiments/analysis/e303_eflic_hcg_e296_threshold095_fixed_clicpro8.{md,json,csv}
- experiments/analysis/e304_eflic_hcg_e296_threshold095_fixed_kodak24.{md,json,csv}
- experiments/analysis/e304_eflic_hcg_e296_threshold095_fixed_clicpro16.{md,json,csv}


## E305-E306 EF-LIC Direction/Fallback Oracle Audit

Status: Done. E305 filled the missing `mean` direction rows for the same E296
learned controller used in E302/E304, with hard activation threshold 0.95 and
GPU0-only execution. E306 then aggregates matched `mean`, `logscale`, and
`fixed` codec-loop CSVs and computes an oracle upper bound for a future
decoder-safe direction/fallback selector. This is not a deployable selector and
not final paper performance evidence; it tells us whether the next controller
should learn direction/fallback instead of hard-coding one geometry source.

Single-direction results:

- Kodak24 `mean`: delta PSNR -0.000488, worst -0.004282, win 0.000000,
  non-negative 0.625000.
- Kodak24 `logscale`: delta PSNR -0.000292, worst -0.006093, win 0.125000,
  non-negative 0.750000.
- Kodak24 `fixed`: delta PSNR -0.000046, worst -0.000597, win 0.125000,
  non-negative 0.750000.
- CLIC Professional 16 `mean`: delta PSNR +0.000760, worst -0.000191,
  win 0.250000, non-negative 0.750000.
- CLIC Professional 16 `logscale`: delta PSNR +0.000792, worst -0.002002,
  win 0.250000, non-negative 0.750000.
- CLIC Professional 16 `fixed`: delta PSNR +0.001030, worst -0.000185,
  win 0.187500, non-negative 0.687500.

Oracle headroom:

- Kodak24 `oracle_with_fallback`: mean +0.000046, worst +0.000000,
  non-negative 1.000000, choices fallback 20 / logscale 3 / fixed 1.
- Kodak24 `oracle_nonfallback`: mean +0.000003, worst -0.000471,
  choices mean 16 / logscale 5 / fixed 3.
- CLIC Professional 16 `oracle_with_fallback`: mean +0.001144,
  worst +0.000000, non-negative 1.000000, choices fallback 10 / fixed 2 /
  mean 1 / logscale 3.
- CLIC Professional 16 `oracle_nonfallback`: mean +0.001142,
  worst -0.000028, choices mean 9 / fixed 3 / logscale 4.

Interpretation: `fixed` remains the best single hard-coded EF-LIC HCG direction
in this controller family, but the oracle shows that direction/fallback choice
is the next useful control variable. On Kodak, fallback is mostly a safety
mechanism; on CLIC Professional, a direction selector has small but real
headroom beyond `fixed`. The next implementation should therefore add a
decoder-safe direction/fallback selector or train the existing family head to
serve that role. This is preferable to adding more auxiliary losses: the
current bottleneck is conditional geometry choice and hard-image rejection, not
loss complexity.

Artifacts:

- tools/analyze_e306_eflic_direction_oracle.py
- experiments/analysis/e305_eflic_hcg_e296_threshold095_mean_kodak24.{md,json,csv}
- experiments/analysis/e305_eflic_hcg_e296_threshold095_mean_clicpro16.{md,json,csv}
- experiments/analysis/e306_eflic_direction_oracle.{md,json,csv}


## E307 EF-LIC Selector Proxy Threshold Check

Status: Done. E307 tests whether a simple monotonic threshold on decoder-safe
summary statistics (`gate_mean` or `alpha_mean`) can safely choose the E304
`fixed` HCG branch versus fallback. It uses the E306 matched per-image rows, so
it is a diagnostic/proxy analysis rather than a deployable learned selector.

Results:

- CLIC Professional 16 all-`fixed`: mean +0.001030, worst -0.000185,
  win 0.187500, non-negative 0.687500.
- CLIC Professional 16 best safe `gate_mean` threshold: select 1/16, mean
  +0.001036, worst +0.000000, non-negative 1.000000.
- CLIC Professional 16 best safe `alpha_mean` threshold: select 2/16, mean
  +0.001039, worst +0.000000, non-negative 1.000000.
- Kodak24 all-`fixed`: mean -0.000046, worst -0.000597, win 0.125000,
  non-negative 0.750000.
- Kodak24 best safe `gate_mean`/`alpha_mean` threshold: select 0/24, mean
  +0.000000, worst +0.000000, non-negative 1.000000.
- Pooled 40 all-`fixed`: mean +0.000385, worst -0.000597, win 0.150000,
  non-negative 0.725000.
- Pooled best safe threshold: select 1/40, mean +0.000414, worst +0.000000,
  non-negative 1.000000.

Interpretation: simple magnitude thresholding can make EF-LIC HCG safe, but only
by selecting almost nothing. It cannot recover the E306 direction/fallback oracle
because Kodak positive and negative active rows are not separable by monotonic
`gate_mean` or `alpha_mean`. The next EF-LIC implementation should therefore
use richer decoder-safe context and direct direction/fallback labels, not just a
single scalar threshold and not extra unrelated losses.

Artifacts:

- experiments/analysis/e307_eflic_selector_proxy_threshold.{md,json,csv}

## E308-E309 EF-LIC Direction Selector Trainability Check

Status: Done. E308 trains a tiny decoder-safe context head on E242 context
tensors to predict the E306 image-level oracle choice among fallback, mean,
logscale, and fixed. E309 post-processes the learned probabilities with
confidence gates to test whether the selector can safely recover non-fallback
rows.

Main E308/E309 findings:

- Strict safe training collapsed to fallback on all 40 pooled images. This is
  safe but gives no gain: mean delta PSNR +0.000000, worst +0.000000.
- Capacity-biased training can output non-fallback choices, but naive deployment
  is not safe: pooled mean +0.000157 with worst -0.006093, due mainly to Kodak
  logscale false positives.
- Probability gating can recover a safe same-pool policy: capacity
  `pred_choice_nonfallback_conf >= 0.79` selects 3/40 images and gives mean
  +0.000340 with worst +0.000000.
- However, cross-dataset rows do not retain useful safe gains. Leave-dataset-out
  CLIC/Kodak best-safe policies fall back to all images, while unsafe CLIC
  thresholds can improve mean but introduce negative tails.

Interpretation: image-level direction/fallback labels are too sparse and
condition-dependent to become the EF-LIC paper-main controller by themselves.
They are still useful as an oracle/headroom diagnostic, but the next EF-LIC
implementation should generate slice/local labels or a directly codec-trained
selector from decoder-available context. This keeps the HCG-RVQ story aligned
with `prompt.txt`: hyperprior/context should control local quantizer geometry,
not merely select an image-level mode after the fact.

Artifacts:

- tools/train_e308_eflic_direction_selector_from_contexts.py
- experiments/analysis/e308_eflic_direction_selector_context_train_cpu.{md,json,summary.csv,predictions.csv}
- experiments/analysis/e308_eflic_direction_selector_context_train_capacity_cpu.{md,json,summary.csv,predictions.csv}
- experiments/analysis/e309_eflic_selector_probability_gate_sweep.{md,json,csv}
- experiments/analysis/e309_eflic_selector_probability_gate_best.csv

## E310 EF-LIC Controller Granularity Decision

Status: Done. E310 consolidates the new E308-E309 direction/fallback selector
results with earlier E244-E245 cross-slice/local activation audits. It is a
design-decision artifact before spending full-training budget.

Key evidence:

- E309 same-pool safe probability gate can select 3/40 non-fallback rows, mean
  +0.000340 dB, worst +0.000000. This is safe but small and pool-dependent.
- E309 leave-dataset CLIC mean-seeking threshold can reach +0.000795 dB, but
  worst drops to -0.002002, so it is not paper-safe.
- E244/E245 local activation features contain signal, but naive operating points
  are unstable: best-F1 rows often activate almost everywhere with high false
  positives, while low-FPR rows miss most active targets.

Decision: do not promote an image-level direction/fallback selector as the
EF-LIC main method. The next EF-LIC HCG-RVQ controller should be two-stage and
local: first binary fallback/activation from decoder-safe context, then
conditional family/direction/strength inside activated regions. The missing
artifact is slice/local direction or residual-headroom labels, not another
image-level classifier or a pile of auxiliary losses.

Artifacts:

- experiments/analysis/e310_eflic_controller_granularity_decision.{md,json,csv}

## E311 EF-LIC Slice/Candidate Policy Signal Audit

Status: Done. E311 audits whether the existing E236 per-image/slice/candidate
statistics are strong enough to choose useful EF-LIC HCG local policies. This is
diagnostic only: E236 candidate statistics include policy-specific outcomes, so
they are not automatically a decoder-side controller.

Main results:

- The audit covers 520 valid codec-loop rows and 117 scalar global/slice
  features.
- Same-dataset separability is real. On Kodak24, sparse-union usefulness is
  separated by residual-error features with oriented AUC about 0.76
  (`avg_residual_error_rms_max4`, `slice0_avg_residual_error_rms`,
  `stage0_residual_error_rms_max4`). On CLIC Professional 41, useful sparse
  rows are more tied to geometry/index features, with top oriented AUC around
  0.72 for `avg_geometry_delta_rms_min4` and around 0.70 for index-entropy or
  index-use features.
- Cross-dataset single-feature rules are not paper-safe. A Kodak-trained
  best-mean residual rule transfers to CLIC with test mean +0.000380 score but
  worst +0.007176, so it improves some rows while causing a large tail. A
  CLIC-trained safe rule transfers to Kodak only by selecting zero rows.
- This confirms that E236 contains useful local-policy signal, but not in a
  form that should be deployed as a single global threshold.

Decision: keep E311 as evidence for the next controller design, not as a final
selector. The next EF-LIC HCG-RVQ implementation should build true local or
slice-level residual/headroom labels from decoder-available context, then train
a two-stage controller: conservative fallback/activation first, and
family/direction/strength only inside activated regions. This is better aligned
with the `prompt.txt` claim than another image-level classifier.

Artifacts:

- tools/analyze_e311_eflic_slice_policy_signal.py
- experiments/analysis/e311_eflic_slice_policy_signal.{md,json}
- experiments/analysis/e311_eflic_slice_policy_signal.feature_separation.csv
- experiments/analysis/e311_eflic_slice_policy_signal.rules.csv

## E312 EF-LIC Slice-Isolation Probe

Status: Initial Kodak1 probe done. E312 extends the E295 EF-LIC HCG codec-loop
probe with `--active-slices`, which zeroes HCG alpha on disabled y-slices in both
compress and decompress. This keeps the decoder contract symmetric and lets us
measure slice-isolated and leave-one-out HCG effects before building local
controller labels.

Implementation updates:

- `tools/run_e295_eflic_hcg_branch_controller_integration_smoke.py` now accepts
  `--active-slices all|none|0|1|2|3|0,1|...`.
- Disabled slices use exact zero-alpha fallback after the controller decision,
  so the original EF-LIC RVQ path is preserved for those slices.
- Row outputs now record `active_slices` and `y_slice_enabled`; summaries report
  `slice_enabled_frac`.
- `tools/analyze_e312_eflic_slice_isolation_probe.py` aggregates the resulting
  E295 CSVs into rows/summary/json/md artifacts.

Kodak1 probe condition: `kodim01.png`, `trained_soft`, E296 controller state,
`direction_source=fixed`, `active_threshold=0.95`, `CUDA_VISIBLE_DEVICES=0`,
`device=cuda:0`. All runs had delta bpp 0, decode max 0, and nonfinite 0.

Results:

- all slices: delta PSNR +0.008304.
- single slices: slice0 +0.003518, slice1 +0.006996, slice2 -0.004120,
  slice3 -0.001348.
- slices 0,1 only: +0.003748, so single-slice gains are not additive.
- leave-one-out: leave0 +0.006927, leave1 +0.012344, leave2 +0.004359,
  leave3 +0.008406.
- Best tested subset is `0,2,3`, improving over all by +0.004039 dB on this
  image; contract_ok_frac is 1.0.

Interpretation: this is a strong local-controller design signal, not a final RD
claim. Slice effects are context-dependent because EF-LIC updates the support
buffer sequentially. A slice can be helpful in isolation but harmful in the
all-slice context, so E312 labels should combine leave-one-out/contextual
marginals with residual/headroom statistics. The next controller should remain
sequential and decoder-safe rather than treating slice gates as independent
additive switches.

Artifacts:

- tools/run_e295_eflic_hcg_branch_controller_integration_smoke.py
- tools/analyze_e312_eflic_slice_isolation_probe.py
- experiments/analysis/e312_eflic_slice_isolation_kodak1_*.{csv,json,md}
- experiments/analysis/e312_eflic_slice_isolation_kodak1_summary.{md,json,rows.csv,summary.csv}

## E313 EF-LIC Multi-Image Slice-Isolation Sweep

Status: Done for Kodak first 4 images. E313 turns the E312 one-image probe into
a reusable multi-image sweep that evaluates several active-slice subsets in one
EF-LIC model/controller load. This is still a small-cycle diagnostic, not final
RD evidence, but it is now strong enough to guide the local controller label
design.

Condition: Kodak24 images `kodim01.png` to `kodim04.png`, `trained_soft`,
E296 controller state, `direction_source=fixed`, `active_threshold=0.95`,
`CUDA_VISIBLE_DEVICES=0`, `device=cuda:0`. All rows had delta bpp 0,
decode max 0, contract_ok_frac 1.0, and nonfinite 0.

Mean results over 4 images:

- `all`: mean delta PSNR +0.030334, worst -0.002477, win fraction 0.75.
- single slice0: +0.022691 mean, worst -0.009230, win fraction 0.75.
- single slice1: +0.007593 mean, worst +0.000692, win fraction 1.0.
- single slice2: -0.002461 mean, worst -0.007137, win fraction 0.50.
- single slice3: -0.000314 mean, worst -0.001628, win fraction 0.25.
- best fixed subset by mean remains `all`, but best per-image subset is not
  always `all`.

Per-image best subsets:

- `kodim01.png`: best `0,2,3`, +0.012344, +0.004039 over `all`.
- `kodim02.png`: best `all`, +0.046543.
- `kodim03.png`: best `0,2,3`, +0.079252, +0.010286 over `all`.
- `kodim04.png`: best `2`, +0.000806, +0.003283 over `all`; all-slice
  activation is slightly harmful at -0.002477.

Interpretation: `all` is a reasonable strong baseline on this small Kodak
subset, but it is not the correct final policy. Three of four images have a
better subset than `all`, and the safe/unsafe slice identity changes by image.
This confirms that EF-LIC HCG-RVQ needs context-aware local activation rather
than a global all-on switch. Slice1 is the safest single slice in this probe,
while slice2 and slice3 are only useful under the right surrounding context.

Next action: build local/sequential labels from E313-style contextual marginal
effects plus decoder-available residual/headroom features, then train a small
two-stage slice/context controller. Keep `all` as a baseline in the next
checkpoint/full-eval comparison, but do not treat it as the final HCG-RVQ
design.

Artifacts:

- tools/run_e313_eflic_slice_isolation_sweep.py
- experiments/analysis/e313_eflic_slice_isolation_sweep_kodak4.{md,json}
- experiments/analysis/e313_eflic_slice_isolation_sweep_kodak4.{rows,by_set,by_image}.csv

## E314 EF-LIC Slice Label Audit From E313

Status: Done. E314 converts the E313 subset sweep into slice-level label
diagnostics for the next sequential controller. It does not train a policy and
is not final RD evidence; it tells us what supervision would be safe or unsafe.

Source: `experiments/analysis/e313_eflic_slice_isolation_sweep_kodak4.rows.csv`.
The audit produces 16 slice labels from 4 images and 4 y-slices.

Main findings:

- `all` fixed policy: mean delta PSNR +0.030334, worst -0.002477.
- best-per-image oracle over tested E313 subsets: mean +0.034736, worst
  +0.000806, mean +0.004402 over `all`.
- single-slice fixed policies are not enough: slice0 mean +0.022691 but worst
  -0.009230; slice1 is safest with worst +0.000692 but mean only +0.007593;
  slice2/slice3 are negative on average.
- Oracle active fraction is 0.687500. Single/context sign agreement is 0.750000,
  but oracle/single agreement is only 0.562500. Oracle/context agreement is
  0.812500.
- Top tiny-sample correlations with oracle active are weak and diagnostic only:
  lower index entropy, lower risk score, and lower residual RMS tend to align
  with oracle-active slices, but the sample is too small for a deployable rule.

Interpretation: a naive label such as "turn on slices whose single-slice delta is
positive" would miss too many best-subset decisions. The next controller should
therefore learn sequential/contextual labels: use the support-buffer state and
residual/headroom features, not only single-slice signs. E314 also quantifies the
available small-cycle headroom over all-on, which justifies building the local
controller before launching larger full-training runs.

Artifacts:

- tools/build_e314_eflic_slice_label_audit.py
- experiments/analysis/e314_eflic_slice_label_audit_kodak4.{md,json}
- experiments/analysis/e314_eflic_slice_label_audit_kodak4.{slice_labels,policy_summary,feature_correlations}.csv

## E315 EF-LIC Kodak24 Slice-Isolation Sweep

Status: Done. E315 extends the E313 slice-isolation sweep from Kodak4 to the full Kodak24 set while keeping the same EF-LIC codec loop and GPU0-only evaluation. This is still a small-cycle controller-label diagnostic, not final RD evidence.

Condition: Kodak24, `trained_soft`, E296 controller state, `direction_source=fixed`, `active_threshold=0.95`, `CUDA_VISIBLE_DEVICES=0`, `device=cuda:0`. All tested rows had delta bpp 0, decode max 0, contract_ok_frac 1.0, and nonfinite 0.

Main fixed-policy results:

- `all`: mean delta PSNR +0.006391, worst -0.030184, win fraction 0.541667.
- single slice0: +0.007630 mean, worst -0.017223, win fraction 0.583333.
- single slice1: -0.002200 mean.
- single slice2: -0.001561 mean.
- single slice3: -0.000627 mean.

Interpretation: HCG activation is real but fragile. `all` improves the mean, but its tail is too risky for a paper-main policy. Several images are harmed by all-on activation, while other images benefit strongly. This confirms that the next EF-LIC HCG-RVQ controller must have an exact fallback path and local/slice selection, rather than claiming a global all-on switch.

Artifacts:

- experiments/analysis/e315_eflic_slice_isolation_sweep_kodak24.{md,json}
- experiments/analysis/e315_eflic_slice_isolation_sweep_kodak24.{rows,by_set,by_image}.csv

## E316 EF-LIC Kodak24 Slice Label Audit

Status: Done. E316 converts the E315 rows into slice-level labels and feature correlations.

Main findings:

- `all`: mean +0.006391, worst -0.030184.
- best-per-image oracle over E315 tested subsets: mean +0.015620, worst -0.006606, mean +0.009229 over `all`.
- Oracle active fraction: 0.572917.
- Single/context sign agreement: 0.656250.
- Oracle/single agreement: 0.604167.
- Oracle/context agreement: 0.739583.
- Top correlations are weak but suggest useful families: local score, zero-probability, strength, index usage, and geometry delta.

Interpretation: E316 says the original subset list still lacked a proper fallback option. The oracle improves the mean but can still be negative because some images prefer not to use any HCG perturbation. Therefore the next sweep must include `none` and all slice subsets before using the rows as teacher labels.

Artifacts:

- experiments/analysis/e316_eflic_slice_label_audit_kodak24.{md,json}
- experiments/analysis/e316_eflic_slice_label_audit_kodak24.{slice_labels,policy_summary,feature_correlations}.csv

## E317/E318 EF-LIC Kodak24 Powerset Sweep With Fallback

Status: Done. E317 evaluates the full 16-way slice powerset, including exact fallback `none`, on Kodak24. E318 converts those rows into slice labels.

Condition: same as E315, with `--slice-sets none 0 1 2 3 0,1 0,2 0,3 1,2 1,3 2,3 0,1,2 0,1,3 0,2,3 1,2,3 all`. All rows had delta bpp 0, decode max 0, contract_ok_frac 1.0, and nonfinite 0.

Key results:

- `all`: mean +0.006391, worst -0.030184.
- best fixed subset by mean: `0,3`, mean +0.008229, worst -0.020546.
- best all/none upper bound, computed in E320: mean +0.011310, worst 0.0.
- full best-per-image powerset oracle: mean +0.016323, worst 0.0, +0.009932 over `all`.
- E318 oracle active fraction drops to 0.468750 once `none` is available.
- E318 oracle/single agreement is 0.687500 and oracle/context agreement is 0.718750.
- Top E318 correlations point to lower geometry-delta RMS, lower active logit, lower alpha/gate/strength, and lower index-used fraction as signals of safer activation, but these are diagnostic rather than deployable rules.

Interpretation: This is the strongest current EF-LIC result. HCG-RVQ has a clear improvement upper bound under the exact EF-LIC fixed-payload contract, and fallback removes the negative tail in the oracle. The remaining problem is not codec validity or bpp accounting; it is learning a reliable controller that approaches the oracle without using held-out outcomes.

Artifacts:

- experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.{md,json}
- experiments/analysis/e317_eflic_slice_isolation_powerset_kodak24.{rows,by_set,by_image}.csv
- experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.{md,json}
- experiments/analysis/e318_eflic_slice_label_audit_powerset_kodak24.{slice_labels,policy_summary,feature_correlations}.csv

## E319/E320 EF-LIC Simple Policy CV

Status: Done. E319 and E320 ask whether the E317/E318 oracle headroom can be recovered by very simple threshold policies under leave-one-image-out cross-validation.

E319 slice-level single-feature threshold:

- `all`: mean +0.006391, worst -0.030184.
- LOOCV threshold policy: mean +0.002686, worst -0.023577, -0.003704 vs `all`.
- full powerset oracle: mean +0.016323, worst 0.0.

E320 image-level all/none gate, with outcome columns removed from candidate features:

- `all`: mean +0.006391, worst -0.030184.
- best all/none oracle: mean +0.011310, worst 0.0.
- LOOCV image gate: mean +0.005202, worst -0.030184, -0.001189 vs `all`.
- full powerset oracle: mean +0.016323, worst 0.0.

Interpretation: simple hand-made thresholds are not enough. This is a useful negative result because it prevents overclaiming from E317/E318. The correct next step is to use E317/E318 as teacher supervision for a learned decoder-available controller: conservative fallback/activation first, then slice/family/direction/strength inside activated regions. The EF-LIC codec loss should remain simple and original-RD dominated; do not add a large stack of auxiliary losses just to patch weak threshold rules.

Artifacts:

- tools/analyze_e319_eflic_slice_policy_cv.py
- tools/analyze_e320_eflic_image_gate_cv.py
- experiments/analysis/e319_eflic_slice_policy_cv_kodak24.{md,json,folds.csv,summary.csv}
- experiments/analysis/e320_eflic_image_gate_cv_kodak24.{md,json,folds.csv,summary.csv}

## E321/E322 EF-LIC Learned/Fixed Slice Policy CV

Status: Done. E321 tests whether the E317/E318 oracle headroom can be recovered
by a small learned slice gate rather than a one-feature threshold. E322 adds a
train-fold fixed-subset baseline so the learned-policy failure can be separated
from the simpler question of whether a globally selected slice subset is already
enough.

E321 setup: leave-one-image-out cross-validation on Kodak24. For each fold, a
regularized logistic slice gate is trained on E318 oracle-active labels from the
23 training images. Outcome columns such as delta PSNR, best subset, contextual
margins, and oracle-agreement labels are excluded from features. Hyperparameters
and the active-probability threshold are selected only on train-fold RD outcomes
using the E317 powerset lookup.

E321 main results:

- `all`: mean +0.006391 dB, worst -0.030184.
- best fixed subset by full-pool mean: `0,3`, mean +0.008229, worst -0.020546.
- best all/none oracle: mean +0.011310, worst 0.0.
- LOOCV logistic slice gate: mean +0.001912, worst -0.017709, -0.004478 vs
  `all`.
- full subset oracle: mean +0.016323, worst 0.0.

E321 interpretation: a shallow learned classifier over image/slice summary
features is still insufficient. It improves the tail relative to `all`, but it
loses too much mean and misses high-gain images. The predicted subsets are also
unstable (`0`, `0,2,3`, `all`, `2`, `none`, etc.), which suggests the E317/E318
oracle cannot be recovered reliably from aggregated statistics alone.

E322 setup: same E317 powerset rows, but no learned features. For each held-out
image, select one fixed slice subset on the 23 training images using
`mean + tail_weight * worst` and evaluate that same subset on the held-out image.

E322 main results:

- tail weight 0: mean +0.007360, worst -0.020546, +0.000969 vs `all`;
  selected subsets are mostly `0,3`.
- tail weight 0.25: mean +0.007236, worst -0.020546, +0.000845 vs `all`;
  selected subsets are mostly `0`.
- tail weight 0.5: mean -0.000718, worst -0.017223.
- tail weight 1.0: exact fallback everywhere, mean 0.0, worst 0.0.

E322 interpretation: fixed slice selection is a better baseline than the current
logistic gate and slightly improves mean over all-on activation, but it still
does not remove the negative tail unless it collapses to fallback. Therefore the
next EF-LIC controller should move below image/slice summary features: train on
decoder-available spatial context maps, preserve exact fallback, and make
local/sequential activation decisions after each EF-LIC support-buffer update.

Artifacts:

- tools/analyze_e321_eflic_slice_policy_logistic_cv.py
- tools/analyze_e322_eflic_fixed_subset_cv.py
- experiments/analysis/e321_eflic_slice_policy_logistic_cv_kodak24.{md,json,summary.csv,folds.csv,slice_probs.csv,coefs.csv}
- experiments/analysis/e322_eflic_fixed_subset_cv_kodak24.{md,json,summary.csv,folds.csv}

## E323/E324 EF-LIC Teacher Provenance Audit

Status: Done. E323 checks whether the older E242 spatial teacher labels can be
used directly for the newer E317/E318 EF-LIC powerset oracle. E324 rebuilds a
teacher artifact that keeps the decoder-available E242 context maps but replaces
the stale labels with E318 fallback-aware oracle labels.

E323 result:

- E242 mean active fraction: 0.614583.
- E318 fallback-aware oracle active fraction: 0.468750.
- Correlation between E242 active fraction and E318 oracle-active fraction: 0.230389.
- Correlation between E242 active fraction and single-slice delta PSNR: 0.040746.
- Best E242-derived threshold policy: mean +0.005827 dB, worst -0.019544 dB, still -0.000564 dB below E317 all-on activation.

E324 result:

- Built 24 aligned context-teacher tensors under `experiments/analysis/e324_eflic_e318_aligned_context_teacher_kodak24/`.
- Missing oracle keys: 0.
- Mean active fraction: 0.468750.
- Mean active slice count: 1.875.

Interpretation: the old E242 context maps are still useful as decoder-available
features, but the old labels are over-active and should not be used as paper-main
supervision. The correct teacher path is E318-aligned: exact fallback is part of
the target, and activation should be learned as a selective reliability problem.

Artifacts:

- tools/analyze_e323_eflic_e242_vs_e318_teacher_alignment.py
- tools/build_e324_eflic_e318_aligned_context_teacher.py
- experiments/analysis/e323_eflic_e242_vs_e318_teacher_alignment_kodak24.{md,json,slice_alignment.csv,threshold_alignment.csv,policy_summary.csv,image_policy.csv}
- experiments/analysis/e324_eflic_e318_aligned_context_teacher_kodak24/

## E325-E328 EF-LIC E318-Teacher Controller Smoke

Status: Done. E325 trains a small decoder-safe HCG branch controller from the
E324 E318-aligned teacher. E326/E327 evaluate that controller inside the real
EF-LIC codec loop on held-out Kodak images 17-24. E328 compares the result to
E317 powerset headroom.

E325 setup:

- Train/eval split: first 16 Kodak images for controller training, last 8 for held-out controller diagnostics.
- Steps: 96.
- False-positive weight: 4.0.
- Missed-active weight: 1.0.
- Device: `cuda:0`.
- Nonfinite rows: 0.

E325 controller diagnostics:

- Eval target active fraction: 0.437500.
- Eval predicted alpha mean: 0.000900.
- Eval soft gate mean: 0.0920.
- Eval hard gate mean: 0.0050.
- Exact `force_zero` alpha max: 0.

Codec-loop results:

- E326 hard gate at threshold 0.5: mean +0.002152 dB, worst -0.000903 dB, delta bpp 0, decode max 0, nonfinite 0.
- E326 soft gate: mean -0.003984 dB, worst -0.019705 dB.
- E327 hard gate at threshold 0.25: mean -0.004379 dB, worst -0.023009 dB.
- E328 eval8 E317 all-on headroom: +0.000559 dB.
- E328 eval8 E317 powerset oracle headroom: +0.012646 dB.
- E326 hard threshold 0.5 recovers 17.0% of that eval8 oracle headroom.

Interpretation: the controller wiring is valid and exact fallback is preserved,
but the first controller is too conservative. Simply lowering the active
threshold activates more regions and makes the result worse, so the bottleneck
is selectivity, not lack of perturbation strength.

Artifacts:

- tools/run_e325_eflic_hcg_controller_e318_teacher_train.py
- tools/analyze_e328_eflic_controller_eval_summary.py
- experiments/analysis/e325_eflic_hcg_controller_e318_teacher_train_kodak24_t16_e8_s96.{md,json,trace.csv,pth}
- experiments/analysis/e326_eflic_e325_controller_codec_eval_kodak24_eval8_thr050.{md,json,csv}
- experiments/analysis/e327_eflic_e325_controller_codec_eval_kodak24_eval8_thr025.{md,json,csv}
- experiments/analysis/e328_eflic_e325_controller_vs_e317_headroom_eval8.{md,json,summary.csv,per_image.csv}

## E329-E337 EF-LIC Balanced Controller and Risk Fallback

Status: Done as a design-discovery run. E329 retrains the same controller with a
more balanced active/fallback loss. E330-E337 evaluate codec-loop behavior under
different reliability gates. These are not final paper metrics yet because risk
thresholds were selected after seeing held-out diagnostics and the full24 table
contains controller-training images.

E329 setup:

- Steps: 128.
- False-positive weight: 2.0.
- Missed-active weight: 2.0.
- Score weight: 0.1.
- Alpha weight: 20.0.
- Family weight: 0.1.
- Device: `cuda:0`.
- Nonfinite rows: 0.

E329 controller diagnostics:

- Eval target active fraction: 0.437500.
- Eval predicted alpha mean: 0.002528.
- Eval soft gate mean: 0.2606.
- Eval hard gate mean: 0.3779.
- Exact `force_zero` alpha max: 0.

Eval8 codec-loop results:

- E330 risk 0.0, hard gate: mean +0.006544 dB, worst -0.058768 dB.
- E330 risk 0.0, soft gate: mean +0.004725 dB, worst -0.015886 dB.
- E332 max-risk -0.05: mean +0.011092 dB, worst -0.030843 dB.
- E333 max-risk -0.10: mean +0.011763 dB, worst -0.027929 dB.
- E334 max-risk -0.15: mean +0.019047 dB, worst -0.017333 dB.

Full Kodak24 codec-loop results:

- E335 max-risk -0.15: mean +0.012898 dB, worst -0.038376 dB, positive 15/24, negative 9/24, delta bpp 0, decode max 0, nonfinite 0.
- E336 comparison: E317 all-on full24 mean is +0.006391 dB and E317 powerset oracle is +0.016323 dB, so E335 recovers 79.0% of the oracle headroom and is +0.006508 dB above all-on.
- E337 max-risk -0.20: mean +0.007246 dB, delta bpp 0, decode max 0, nonfinite 0. This is safer in some cases but too conservative in mean.

Interpretation: the best current EF-LIC route is a balanced decoder-safe
controller plus risk fallback. It improves the full Kodak24 mean over all-on and
keeps the fixed-payload/no-side-bit contract intact, which is aligned with the
prompt target. The remaining paper-risk is validation protocol: E335 is a strong
candidate, but not yet an independent paper claim. Next, lock a train/validation/test
or cross-validation protocol for risk thresholds, then promote only thresholds
selected without looking at the test split.

Failure analysis:

- The main bad cases are false-positive activation on images where the E317 powerset oracle prefers fallback or a different sparse subset.
- Negative examples include `kodim05`, `kodim19`, and `kodim22`, where E317 best was `none` but the E329 controller still activated some geometry.
- The large positive examples show that learned alpha/risk control is more than a fixed slice subset: E335 can exceed the E317 fixed-powerset row on some images because its continuous geometry strength is not identical to E317 fixed branch.

Next actions:

- Build a paper-safe risk-selection protocol using only training/validation images, then evaluate the selected controller once on a held-out split.
- Add a no-op/fallback classifier for images or local regions whose E317 oracle selects `none`.
- Repeat the same controller-risk protocol on an independent split such as CLIC professional or a Kodak fold.
- Start the analogous GLC controller audit from the existing GLC RVQ/rate probes.

Artifacts:

- experiments/analysis/e329_eflic_hcg_controller_e318_teacher_train_balanced_fp2_miss2_s128.{md,json,trace.csv,pth}
- experiments/analysis/e330_eflic_e329_balanced_controller_codec_eval_kodak24_eval8_thr050.{md,json,csv}
- experiments/analysis/e331_eflic_controller_training_variants_vs_e317_headroom_eval8.{md,json,summary.csv,per_image.csv}
- experiments/analysis/e332_eflic_e329_balanced_controller_codec_eval_kodak24_eval8_thr050_riskm005.{md,json,csv}
- experiments/analysis/e333_eflic_e329_balanced_controller_codec_eval_kodak24_eval8_thr050_riskm010.{md,json,csv}
- experiments/analysis/e334_eflic_e329_balanced_controller_codec_eval_kodak24_eval8_thr050_riskm015.{md,json,csv}
- experiments/analysis/e335_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm015.{md,json,csv}
- experiments/analysis/e336_eflic_e329_balanced_controller_vs_e317_headroom_full24.{md,json,summary.csv,per_image.csv}
- experiments/analysis/e337_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm020.{md,json,csv}

## E338-E342 EF-LIC Full24 Risk Grid and Split Selection

Status: Done. E338-E340 fill the missing full Kodak24 risk-grid rows for the E329
balanced EF-LIC controller. E341 compares the full grid to E317 powerset
headroom. E342 checks whether the risk threshold can be selected using only the
first 16 Kodak images and then evaluated on the held-out last 8 images.

Full24 risk-grid results:

- E338 risk none: mean +0.013690 dB, worst -0.058768 dB, delta bpp 0, decode max 0, nonfinite 0.
- E339 max-risk -0.05: mean +0.012766 dB, worst -0.039950 dB, delta bpp 0, decode max 0, nonfinite 0.
- E340 max-risk -0.10: mean +0.010966 dB, worst -0.038873 dB, delta bpp 0, decode max 0, nonfinite 0.
- E335 max-risk -0.15: mean +0.012898 dB, worst -0.038376 dB, delta bpp 0, decode max 0, nonfinite 0.
- E337 max-risk -0.20: mean +0.007246 dB, worst -0.026403 dB, delta bpp 0, decode max 0, nonfinite 0.

E341 comparison to E317:

- E317 all-on mean: +0.006391 dB.
- E317 full powerset oracle: +0.016323 dB.
- Risk none has the highest full24 mean, +0.013690 dB, recovering 83.9% of the E317 oracle headroom, but its tail is unsafe at -0.058768 dB.
- Risk -0.15 is the best mean/tail compromise among the manually inspected full24 rows: +0.012898 dB mean, -0.038376 dB worst, 79.0% oracle headroom recovered.
- Risk -0.20 is safer in worst case but gives up too much mean.

E342 split-selection result:

- Candidate split: train/select on `kodim01`-`kodim16`, evaluate on `kodim17`-`kodim24`.
- The controller itself was trained on the first 16 images, so E342 is a risk-threshold protocol check, not final paper evidence.
- Train mean-only and moderate tail penalties select risk none, giving eval mean +0.006544 dB and eval worst -0.058768 dB.
- A very tail-heavy objective selects risk -0.20, giving eval mean +0.016443 dB and eval worst -0.024418 dB.
- The strongest eval8 row, risk -0.15, gives +0.019047 dB and worst -0.017333 dB, but is not selected by the current train16 objectives.

Interpretation: HCG-RVQ with the balanced controller is clearly useful in the
EF-LIC codec loop, but risk thresholding is not yet paper-safe. The next
improvement should not be post-hoc threshold tuning. It should be a learned
no-op/fallback classifier or validation-selected policy that predicts the `none`
oracle cases before applying local geometry.

Artifacts:

- experiments/analysis/e338_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_risknone.{md,json,csv}
- experiments/analysis/e339_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm005.{md,json,csv}
- experiments/analysis/e340_eflic_e329_balanced_controller_codec_eval_kodak24_full24_thr050_riskm010.{md,json,csv}
- experiments/analysis/e341_eflic_e329_risk_grid_vs_e317_headroom_full24.{md,json,summary.csv,per_image.csv}
- tools/analyze_e342_eflic_risk_grid_split_selection.py
- experiments/analysis/e342_eflic_e329_risk_grid_split_selection_kodak16_8.{md,json,summary.csv}


## E343-E344 EF-LIC No-op/Fallback Feature Audit

Status: Done. E343 joins the E317 slice-subset oracle with E329 controller
codec-loop outputs and audits only pre-decision/controller summary features. An
initial draft accidentally allowed outcome/oracle columns as candidate features;
that leak was removed before the results below were recorded.

E343 full24 optimistic feature thresholds:

- Risk none raw full24: mean +0.013690 dB, worst -0.058768 dB, 9 negative images.
- Best decoder-feature fallback for risk none: `slice_risk_score_mean_max <= 0.080215`, mean +0.022118 dB, worst -0.008581 dB, 1 negative image, suppressing 12/24 images. It suppresses 8 negative and 4 positive controller outputs.
- Risk -0.15 raw full24: mean +0.012898 dB, worst -0.038376 dB, 9 negative images.
- Best decoder-feature fallback for risk -0.15: `slice_risk_score_mean_max <= 0.073706`, mean +0.019077 dB, worst -0.010641 dB, 2 negative images, suppressing 10/24 images.

E344 first16-to-last8 split selection:

- Risk none raw eval8: mean +0.006544 dB, worst -0.058768 dB, 3 negative images.
- Selecting on the first 16 images chooses `slice_risk_score_mean_max <= 0.074003`; eval8 becomes +0.016563 dB mean, -0.022600 dB worst, 1 negative image, 4 suppressed images.
- Risk -0.005 raw eval8: +0.011092 dB mean, -0.030843 dB worst. A tail-heavy split objective chooses `z_index_used_frac <= 0.065430`; eval8 becomes +0.011970 dB mean, 0 worst, 0 negative images, 5 suppressed images.
- Risk -0.010 raw eval8: +0.011763 dB mean, -0.027929 dB worst. Tail-heavy selection again uses `z_index_used_frac <= 0.065430`; eval8 becomes +0.012193 dB mean, 0 worst, 0 negative images, 5 suppressed images.
- Risk -0.15 raw eval8 is already strongest at +0.019047 dB mean and -0.017333 dB worst. Split-selected fallback suppresses too much and drops eval8 to +0.008940 dB, although it removes the negative tail.

Interpretation: decoder-visible features contain a real no-op/fallback signal, so
the next method improvement should not be just a global max-risk sweep. However,
single feature thresholds are not stable enough to become the paper-main policy:
they can improve the risk-none tail but over-suppress the already-good risk -0.15
row. The next implementation target is a learned fallback/no-op head trained on
separate teacher or validation images, with the codec-loop contract checked on a
held-out split.

Artifacts:

- tools/analyze_e343_eflic_none_oracle_feature_audit.py
- experiments/analysis/e343_eflic_none_oracle_feature_audit_kodak24.{md,json,joined.csv,features.csv,policies.csv,summary.csv}
- tools/analyze_e344_eflic_noop_feature_split_selection.py
- experiments/analysis/e344_eflic_noop_feature_split_selection_kodak16_8.{md,json,csv}

## E345 EF-LIC Learned No-op Linear Probe

Status: Done. E345 turns the E343/E344 no-op/fallback observation into a small
learned diagnostic. A logistic image-level fallback classifier is trained on the
first 16 Kodak images from non-leaky, decoder-visible controller features only;
the fallback threshold is selected on the same train split and evaluated on the
held-out last 8 Kodak images. This is CPU-only analysis over saved CSVs, so no
GPU device was used.

Raw eval baselines from the same split:

- Risk none: +0.006544 dB mean, -0.058768 dB worst, 3 negative images.
- Risk -0.010: +0.011763 dB mean, -0.027929 dB worst, 3 negative images.
- Risk -0.015: +0.019047 dB mean, -0.017333 dB worst, 2 negative images.
- Risk -0.020: +0.016443 dB mean, -0.024418 dB worst, 4 negative images.

Best train-selected linear no-op policies:

- Risk none improves only to +0.006827 dB mean and keeps the -0.058768 dB worst case. It suppresses 4 eval images but misses the main tail failure.
- Risk -0.010 improves slightly to +0.012102 dB mean but still has -0.027929 dB worst.
- Risk -0.015 over-suppresses: eval mean drops from +0.019047 dB raw to +0.008108 dB, with the same -0.017333 dB worst.
- Risk -0.020 becomes tail-safe, +0.012173 dB mean with worst 0 and 0 negative images, but gives up mean relative to the raw +0.016443 dB row.

Interpretation: a naive global image-level learned no-op head is not the right
paper-main controller. It can perfectly clean the train split, but it does not
generalize better than E344's scalar feature thresholds and often suppresses
beneficial HCG activations. This is useful evidence: the remaining failure mode
is local/sequential and codec-gain dependent, not just an image-level difficulty
classification problem.

Decision: do not promote this linear no-op probe to the method. The next EF-LIC
controller should be a local/sequential no-op or family head trained against
actual codec-gain/fallback labels on independent calibration data. Full EF-LIC or
GLC performance training should wait until this selector is fixed and
codec-loop-valid, otherwise full training would test a moving controller rather
than the HCG-RVQ method.

Artifacts:

- tools/analyze_e345_eflic_noop_linear_probe.py
- experiments/analysis/e345_eflic_noop_linear_probe_kodak16_8.{md,json,csv}

## E346-E348 EF-LIC Codec-Gain Risk Teacher and Controller Grid

Status: Done. E346 rebuilds the EF-LIC controller teacher so the risk score is
calibrated from actual codec-gain labels rather than from an active-like proxy.
It reuses decoder-safe E242 context tensors, takes activation from E318
`contextual_positive` slice labels, and stores `risk_target =
-contextual_margin_psnr * 5.0` clipped to +/-0.1. The teacher has 24 Kodak
images, mean active fraction 0.541667, mean active slice count 2.166667, mean
contextual margin PSNR 0.001692, mean risk target -0.004090, and no missing
label keys.

E346 first training smoke was intentionally conservative and became nearly
inert: the 128-step false-positive-heavy run produced eval `hard_gate_mean=0`.
This confirmed that the teacher path is loadable and finite, but not useful as a
controller.

E347 reran the same teacher with a balanced objective
(`false-positive-weight=2`, `missed-active-weight=2`, 256 steps) on GPU0 only.
The controller became active while staying finite:

- Train hard gate mean: 0.377482; eval hard gate mean: 0.356771.
- Train pred alpha mean: 0.002646; eval pred alpha mean: 0.002571.
- `force_zero_alpha_max=0` and `nonfinite_records=0` on both splits.

E347/E348 codec-loop risk grid on held-out Kodak17-24, with exact fixed-payload
checks, gives:

| max risk | mean dPSNR | worst dPSNR | negative | max abs dBPP | decode max | mean gate | mean alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | +0.013984 | -0.026088 | 4 | 0.000000 | 0.000e+00 | 0.357992 | 0.003436 |
| -0.02 | +0.013841 | -0.026132 | 4 | 0.000000 | 0.000e+00 | 0.311523 | 0.002975 |
| -0.04 | +0.019086 | -0.020061 | 3 | 0.000000 | 0.000e+00 | 0.242106 | 0.002295 |
| -0.06 | +0.017634 | -0.016126 | 3 | 0.000000 | 0.000e+00 | 0.161702 | 0.001541 |
| -0.08 | +0.021775 | -0.022029 | 2 | 0.000000 | 0.000e+00 | 0.094238 | 0.000912 |
| -0.10 | +0.008476 | -0.011862 | 4 | 0.000000 | 0.000e+00 | 0.051595 | 0.000512 |

Interpretation: codec-gain risk targets are a real improvement over the previous
active-like risk signal. On the same held-out Kodak17-24 split, `max_risk=-0.08`
has the best mean at +0.021775 dB, while `max_risk=-0.06` offers a stronger
mean/tail compromise at +0.017634 dB mean and -0.016126 dB worst. Every evaluated
risk point preserves delta bpp 0, decode max 0, and nonfinite 0.

Decision: do not start final full-training performance claims yet. The next
paper-safe step is to freeze the E346 codec-gain teacher design and select the
risk/controller threshold on independent calibration data, then evaluate the
unchanged policy on Kodak/CLIC held-out splits. If that passes, the EF-LIC branch
is ready to move from controlled codec-loop evidence to longer/full training or
full evaluation.

Artifacts:

- tools/build_e346_eflic_codec_gain_context_teacher.py
- tools/analyze_e348_eflic_codec_gain_risk_grid.py
- experiments/analysis/e346_eflic_codec_gain_context_teacher_kodak24/manifest_kodak24_n24.{md,json,csv}
- experiments/analysis/e347_eflic_codec_gain_controller_train_balanced_kodak24_t16_e8_s256.{md,json,trace.csv,pth}
- experiments/analysis/e347_eflic_codec_gain_controller_codec_eval_kodak17_24_thr050_riskm{020,040,060,080,100}_fixed.{md,json,csv}
- experiments/analysis/e348_eflic_codec_gain_risk_grid_kodak17_24.{md,json,summary.csv,per_image.csv}


## E349 EF-LIC Kodak-to-CLIC Frozen Codec-Gain Controller Transfer

Status: Done. E349 freezes the E347 codec-gain controller trained on Kodak and
transfers it directly to CLIC Professional 41 images. This is intentionally not a
new fit on CLIC: the goal is to test whether the short-cycle Kodak controller
learned a usable EF-LIC/HCG reliability signal or only Kodak-specific artifacts.
All runs used CUDA_VISIBLE_DEVICES=0 and --device cuda:0; device 1 was not used.

The fixed-payload/no-entropy contract still holds. The force_zero sanity row on
CLIC41 has mean/worst dPSNR 0, max abs dBPP 0, decode max 0, and nonfinite 0.
The active transfer rows also keep max abs dBPP 0, decode max 0, and nonfinite 0
for every risk point.

| max risk | n | mean dPSNR | median | worst | best | negative | mean gate | mean alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.06 | 41 | +0.004621 | +0.003606 | -0.036221 | +0.078319 | 15 | 0.079639 | 0.000727 |
| -0.08 | 41 | +0.003411 | +0.001989 | -0.014399 | +0.047159 | 16 | 0.037216 | 0.000340 |
| -0.10 | 41 | +0.002682 | +0.000869 | -0.029681 | +0.049844 | 15 | 0.017148 | 0.000159 |

Risk -0.08 is the best CLIC transfer compromise: it gives smaller mean gain
than -0.06, but cuts the worst failure from -0.036221 dB to -0.014399 dB.
Risk -0.10 is more conservative on average gate/alpha, but is not monotonic in
tail safety because alejandro-escamilla-6.png worsens to -0.029681 dB.

The per-image no-op/risk oracle over {noop, -0.06, -0.08, -0.10} is much
stronger: full CLIC41 oracle mean is +0.010452 dB with worst 0 and 0 negative
images. The oracle choices are riskm060:19, riskm080:9, riskm100:7, and noop:6.
On the held split eval_last20, the oracle reaches +0.014428 dB with worst 0.
This says the HCG edit has real external-data headroom, but the current global
risk scalar is not yet the paper-main selector.

Decision: do not launch EF-LIC full training as the final performance claim yet.
E349 is positive transfer evidence, but the next paper-safe step is a
decoder-visible local/sequential reliability controller or CLIC-split codec-gain
teacher that learns when to choose no-op, risk -0.06, -0.08, or -0.10. Full
training becomes justified after that frozen policy remains positive on a
held-out split with controlled worst-case degradation. For GLC, the same lesson
reinforces the current selected-replacement route: avoid dense/all-on activation
and keep explicit signal/index accounting.

Artifacts:

- tools/analyze_e349_eflic_clic_transfer_grid.py
- experiments/analysis/e349_eflic_e347_codec_gain_controller_transfer_clicpro41_thr050_riskm060_fixed.{md,json,csv}
- experiments/analysis/e349_eflic_e347_codec_gain_controller_transfer_clicpro41_thr050_riskm080_fixed.{md,json,csv}
- experiments/analysis/e349_eflic_e347_codec_gain_controller_transfer_clicpro41_thr050_riskm100_fixed.{md,json,csv}
- experiments/analysis/e349_eflic_e347_clicpro41_transfer_grid_summary.{md,json,risk_summary.csv,split_summary.csv,oracle_summary.csv,failures.csv}


## E350-E351 EF-LIC Generative/Perceptual Metric Protocol

Status: Done. The EF-LIC HCG branch had been using dPSNR as the main short-cycle
signal, but this is not paper-safe for the generative/perceptual low-bitrate
setting. E350 therefore extends the EF-LIC codec-loop smoke/eval tool with
optional MS-SSIM, LPIPS, and DISTS reporting. PSNR is kept as a codec-health
and distortion diagnostic, not as the primary generative-compression selector.
All runs below used CUDA_VISIBLE_DEVICES=0 and --device cuda:0; device 1 was not
used.

A two-image smoke with `force_zero` and `trained_hard` confirmed the new metric
path is finite and the no-op contract is exact. The full CLIC Professional 41
perceptual evaluation then compared the frozen E347 codec-gain controller at
risk -0.06, -0.08, and -0.10 using the score
`delta_DISTS + 3 * delta_LPIPS` (lower is better). The bpp/decode contract holds
for every row: max abs delta bpp is 0, decode max is 0, and nonfinite rows are 0.

| risk | n | dPSNR | dMS-SSIM | dLPIPS | dDISTS | perceptual score | score wins | triple wins | mean gate | mean alpha | worst dPSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.06 | 41 | +0.004621 | +0.000113 | -0.000049 | -0.000111 | -0.000257 | 23 | 11 | 0.079639 | 0.000727 | -0.036221 |
| -0.08 | 41 | +0.003411 | +0.000088 | -0.000040 | +0.000002 | -0.000118 | 19 | 11 | 0.037216 | 0.000340 | -0.014399 |
| -0.10 | 41 | +0.002682 | +0.000054 | -0.000057 | -0.000037 | -0.000208 | 25 | 8 | 0.017148 | 0.000159 | -0.029681 |

The fixed-risk conclusion changes slightly under perceptual metrics. Risk -0.06
is best by mean perceptual score and mean PSNR, but has the worst PSNR tail.
Risk -0.10 has the most score-win images and the mildest worst perceptual score,
but its mean gain is weaker and it does not guarantee PSNR-tail safety. Risk
-0.08 remains the best PSNR-tail compromise, but it is not the best perceptual
choice.

The no-op/risk oracle over {noop, -0.06, -0.08, -0.10} gives a much stronger
perceptual upper bound: mean score -0.000913 with worst score 0. Choices are
noop:9, risk -0.06:17, risk -0.08:8, and risk -0.10:7. This means the HCG edit
has real external-data headroom, but a single fixed risk scalar is not the final
paper method.

Decision: do not select the EF-LIC full-training policy using dPSNR alone. The
next EF-LIC step is a metric-aware reliability selector trained/evaluated under a
split protocol, with PSNR retained only as a diagnostic/tail-safety metric. Full
training becomes justified after a frozen selector keeps the fixed-payload
contract and improves LPIPS/DISTS/MS-SSIM on held-out data.

Artifacts:

- tools/run_e295_eflic_hcg_branch_controller_integration_smoke.py
- tools/analyze_e350_eflic_perceptual_protocol.py
- tools/analyze_e351_eflic_perceptual_risk_grid.py
- experiments/analysis/e350_eflic_e347_perceptual_metric_smoke_clicpro2_riskm080.{md,json,csv}
- experiments/analysis/e350_eflic_e347_perceptual_metric_clicpro41_riskm080.{md,json,csv}
- experiments/analysis/e350_eflic_e347_perceptual_metric_clicpro41_riskm080_analysis.{md,json,csv}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm060.{md,json,csv}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm060_analysis.{md,json,csv}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm100.{md,json,csv}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_riskm100_analysis.{md,json,csv}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_risk_grid.{md,json}
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_risk_grid_risk_summary.csv
- experiments/analysis/e351_eflic_e347_perceptual_metric_clicpro41_risk_grid_oracle_choices.csv


## E352 EF-LIC Perceptual Selector Split Audit

Status: Done. E352 tests whether the E351 no-op/risk perceptual oracle can be
approximated by simple decoder-visible thresholds before moving to full EF-LIC
training. The split is intentionally small but honest: CLIC Professional images
are sorted by name, the first 20 images are used for calibration, and the last 21
images are held out for evaluation. The objective remains the perceptual score
`delta_DISTS + 3 * delta_LPIPS`; PSNR is reported as a diagnostic/tail metric.

The calibration-selected policy is a simple threshold on `y_strength_mean` for
risk -0.06. It transfers in the right direction but is far from the oracle:

| policy | cal score | cal worst | cal dPSNR | eval score | eval worst | eval dPSNR | eval worst dPSNR | eval wins | choices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| risk -0.06 if `y_strength_mean >= 0.0199847` | -0.000343 | +0.001717 | +0.002805 | -0.000121 | +0.003001 | +0.002664 | -0.032229 | 7/21 | risk -0.06:14, noop:7 |

Held-out oracle remains much stronger: eval oracle score is -0.001106, score wins
15/21, mean dPSNR +0.007875, with choices risk -0.06:9, risk -0.08:2, risk
-0.10:4, noop:6. Fixed-risk diagnostics on the same held split are useful:
risk -0.06 gives eval score -0.000327 and dPSNR +0.004959 but has a -0.036221
dPSNR tail; risk -0.10 gives eval score -0.000239 and dPSNR +0.006604 with a
much safer -0.008392 dPSNR tail; risk -0.08 is weaker by score (-0.000033) but
has an intermediate tail (-0.012491).

Decision: the EF-LIC branch is not ready for final full-training claims with a
simple global threshold. The next meaningful implementation step is a learned
local/sequential selector trained against perceptual codec-gain labels on an
independent split. It should choose among noop and a small family of risk
strengths while keeping the fixed-payload/decode-exact contract. Full training is
still justified soon, but only after this selector policy is frozen and transfers
under perceptual metrics.

Artifacts:

- tools/analyze_e352_eflic_perceptual_selector_split.py
- experiments/analysis/e352_eflic_e347_perceptual_selector_split_clicpro41_first20_last21.{md,json}
- experiments/analysis/e352_eflic_e347_perceptual_selector_split_clicpro41_first20_last21_all_policies.csv
- experiments/analysis/e352_eflic_e347_perceptual_selector_split_clicpro41_first20_last21_top_eval_diagnostic.csv

## E353 EF-LIC Perceptual Learned Selector Split Audit

Status: Done. E353 tests whether the E351 no-op/risk perceptual oracle can be
approximated by a tiny learned selector before moving to EF-LIC full training.
The selector is deliberately simple: ridge regression predicts the perceptual
score `delta_DISTS + 3 * delta_LPIPS` for each candidate among noop, risk -0.06,
risk -0.08, and risk -0.10 using decoder-visible candidate features. PSNR is
kept as a diagnostic/tail metric only.

Two feature sets were evaluated on the same CLIC Professional split used in
E352: first 20 images for calibration and last 21 images for held-out evaluation.
Both preserve the fixed-payload contract: max delta bpp is 0, decode max is 0,
and nonfinite rows are 0.

| feature set | LOOCV lambda | margin | cal selected score | eval selected score | eval worst score | eval dPSNR | eval worst dPSNR | eval score wins | eval choices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| global_slice | 1.0 | 0.00000 | -0.000533 | +0.000157 | +0.002179 | +0.004378 | -0.019746 | 4/21 | noop:8, risk -0.06:3, risk -0.08:7, risk -0.10:3 |
| global | 0.1 | 0.00025 | -0.000429 | +0.000210 | +0.003001 | +0.001461 | -0.003937 | 2/21 | noop:15, risk -0.06:2, risk -0.08:1, risk -0.10:3 |

The calibration rows look stronger than fixed-risk baselines, but both learned
selectors flip to positive perceptual score on held-out images. This is a useful
negative result: the current image-level learned selector is too small and too
split-specific to be promoted as the paper method. It should not be used to
launch final full-training claims.

The HCG mechanism itself remains promising. On the same held-out split, fixed
risk -0.06 gives score -0.000327, fixed risk -0.10 gives score -0.000239 with a
safer PSNR tail, and the held oracle remains much stronger at score -0.001106.
That means the quantizer-geometry edit has real perceptual headroom, but the
selector needs more independent teacher data and/or a local sequential design.

Decision: do not freeze the E353 learned selector for paper-main full training.
The next EF-LIC step is to build a larger perceptual teacher set across Kodak
and CLIC Professional, then evaluate a frozen selector under disjoint
train/calibration/eval splits. Full training becomes justified when the frozen
selector beats the best fixed-risk row on held-out LPIPS/DISTS/MS-SSIM while
keeping bpp/decode exactness and controlled diagnostic PSNR tails.

Artifacts:

- tools/analyze_e353_eflic_perceptual_learned_selector_split.py
- experiments/analysis/e353_eflic_e347_perceptual_learned_selector_split_clicpro41_first20_last21.{md,json,csv}
- experiments/analysis/e353_eflic_e347_perceptual_learned_selector_split_clicpro41_first20_last21_global.{md,json,csv}

## E354-E356 EF-LIC Mixed Perceptual Teacher and Selector Audits

Status: Done. E354 extends the EF-LIC perceptual protocol from CLIC Professional
to Kodak24 using the frozen E347 codec-gain controller. All runs used
`CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`; device 1 was not used. Every row
keeps the fixed-payload contract: max delta bpp is 0, decode max is 0, and
nonfinite rows are 0.

Kodak24 is a useful correction to the CLIC-only picture. Fixed risks are not
paper-safe by perceptual mean even when some PSNR diagnostics improve:

| risk | n | dPSNR | dMS-SSIM | dLPIPS | dDISTS | score | score wins | triple wins | mean gate | mean alpha | worst PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| risk -0.06 | 24 | +0.010798 | +0.000270 | +0.000044 | +0.000641 | +0.000772 | 10 | 3 | 0.165799 | 0.001605 | -0.047523 |
| risk -0.08 | 24 | +0.003780 | +0.000016 | +0.000170 | +0.000776 | +0.001286 | 8 | 2 | 0.091417 | 0.000891 | -0.047246 |
| risk -0.10 | 24 | -0.005944 | -0.000051 | +0.000058 | +0.000444 | +0.000619 | 10 | 3 | 0.045709 | 0.000450 | -0.054539 |

The Kodak no-op/active oracle is strong: mean perceptual score -0.001610 with
worst score 0 and choices noop:9, risk -0.06:7, risk -0.08:4, risk -0.10:4.
This confirms the main lesson from CLIC under a different direction: fixed
always-on HCG is the wrong final method, while selective HCG has real headroom.

E355 then builds a mixed-domain selector teacher: CLIC first20 + Kodak24 are used
for calibration (`cal-count=44`), and CLIC last21 remains held out. With all three
risk strengths, the `global_slice` ridge selector becomes useful again on held
CLIC, while `global` alone remains harmful:

| feature set | candidates | eval score | eval worst score | eval dPSNR | eval worst dPSNR | eval score wins | eval choices |
|---|---|---:|---:|---:|---:|---:|---|
| global_slice | noop/-0.06/-0.08/-0.10 | -0.000278 | +0.002043 | -0.000299 | -0.032229 | 6/21 | noop:10, risk -0.06:7, risk -0.08:1, risk -0.10:3 |
| global | noop/-0.06/-0.08/-0.10 | +0.000125 | +0.002734 | +0.005058 | -0.007900 | 5/21 | noop:9, risk -0.06:3, risk -0.08:3, risk -0.10:6 |

E356 tests a simpler candidate family, noop/risk -0.06/risk -0.10. This does not
improve the held score:

| feature set | candidates | eval score | eval worst score | eval dPSNR | eval worst dPSNR | eval score wins | eval choices |
|---|---|---:|---:|---:|---:|---:|---|
| global_slice | noop/-0.06/-0.10 | -0.000039 | +0.003001 | +0.004201 | -0.019746 | 5/21 | noop:10, risk -0.06:5, risk -0.10:6 |
| global | noop/-0.06/-0.10 | -0.000102 | +0.000815 | +0.003839 | -0.006069 | 4/21 | noop:12, risk -0.06:2, risk -0.10:7 |

Decision: mixed-domain teacher data helps, but the current image-level selector
still does not pass the full-training gate. The best learned row is E355
`global_slice`: it improves over E353 and is a reasonable mean/tail compromise,
but it still does not beat the held fixed risk -0.06 mean score (-0.000327). The
next step should be a local/sequential or robust-objective selector trained on a
larger perceptual teacher set, not final EF-LIC full training yet.

Artifacts:

- tools/build_e355_eflic_mixed_perceptual_teacher_csv.py
- experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_riskm{060,080,100}.{md,json,csv}
- experiments/analysis/e354_eflic_e347_perceptual_metric_kodak24_risk_grid.{md,json}
- experiments/analysis/e355_eflic_e347_mixed_kodak24_clic20_to_clic21_teacher.{md,json}
- experiments/analysis/e355_eflic_e347_mixed_kodak24_clic20_to_clic21_learned_selector_global_slice.{md,json,csv}
- experiments/analysis/e355_eflic_e347_mixed_kodak24_clic20_to_clic21_learned_selector_global.{md,json,csv}
- experiments/analysis/e356_eflic_e347_mixed_kodak24_clic20_to_clic21_learned_selector_global_slice_noop060100.{md,json,csv}
- experiments/analysis/e356_eflic_e347_mixed_kodak24_clic20_to_clic21_learned_selector_global_noop060100.{md,json,csv}


## E357-E358 EF-LIC Perceptual Selector Weighting and Oracle Feature Audit

Status: Done. E357 tests whether the E355 mixed-domain teacher can be improved by
source weighting rather than changing the controller architecture. The objective
is still `delta_DISTS + 3 * delta_LPIPS`; PSNR is reported only as a diagnostic
and is not used for policy selection. The first parallel run was stopped because
unbounded BLAS threading made the tiny CSV audit consume several hundred percent
CPU per process. The reruns use `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and `MKL_NUM_THREADS=1`.

Weighted fitting does not pass the full-training gate. Held CLIC eval results are:

| feature set | source weights | eval score | eval worst score | eval score wins | eval choices |
|---|---|---:|---:|---:|---|
| global_slice | CLIC 2.0 / Kodak 0.5 | +0.000039 | +0.002043 | 4/21 | noop:11, -0.06:2, -0.08:3, -0.10:5 |
| global_slice | CLIC 3.0 / Kodak 0.5 | +0.000123 | +0.002043 | 1/21 | noop:16, -0.06:2, -0.08:2, -0.10:1 |
| global_slice | CLIC 2.0 / Kodak 0.25 | +0.000049 | +0.002043 | 4/21 | noop:11, -0.06:4, -0.08:2, -0.10:4 |
| global_slice | CLIC 1.0 / Kodak 0.5 | -0.000023 | +0.002043 | 3/21 | noop:14, -0.06:4, -0.08:1, -0.10:2 |
| global | CLIC 2.0 / Kodak 0.5 | +0.000077 | +0.003001 | 2/21 | noop:17, -0.06:3, -0.08:1 |

The best of these is only `-0.000023`, far weaker than the held fixed risk -0.06
row (`-0.000327`) and fixed risk -0.10 (`-0.000239`). Source weighting therefore
is not the missing ingredient. It reduces some calibration overfitting, but it
also makes the policy too conservative or mis-ranked on the held CLIC split.

E358 then audits the mixed-domain no-op/risk oracle directly. The oracle remains
strong across both domains while preserving bpp/decode exactness:

| scope | fixed -0.06 | fixed -0.10 | oracle | oracle choices |
|---|---:|---:|---:|---|
| CLIC Professional 41 | -0.000257 | -0.000208 | -0.000913 | noop:9, -0.06:17, -0.08:8, -0.10:7 |
| Kodak24 | +0.000772 | +0.000619 | -0.001610 | noop:9, -0.06:7, -0.08:4, -0.10:4 |
| mixed 65 | +0.000122 | +0.000097 | -0.001171 | noop:18, -0.06:24, -0.08:12, -0.10:11 |
| held CLIC 21 | -0.000327 | -0.000239 | -0.001106 | noop:6, -0.06:9, -0.08:2, -0.10:4 |

The feature audit explains why the current selector stalls. The strongest
correlations with oracle score are moderate (`y_gate_mean` r=-0.475,
`y_alpha_active_frac` r=-0.475, `y_avg_geometry_delta_rms` r=-0.429), while the
best oracle-active AUC is only about 0.61 (`slice0_strength_mean`, slice1/2 index
entropy/perplexity, and `y_strength_mean`). Global/slice summary features see the
right direction, but not enough to recover the oracle.

Decision: EF-LIC is still progressing, but not ready for final full training from
an image-level selector. The consistent positive signal is selective,
decoder-reproducible HCG geometry with no payload change. The consistent failure
is dense/all-on or coarse image-level gating. Next implementation should return
to local/sequential reliability control: choose HCG risk at the slice/block level
from local residual, index-usage, and active-state statistics, keep the original
EF-LIC Representation-domain Decorrelation and no-entropy payload intact, and use
LPIPS/DISTS/MS-SSIM/bpp as the main reporting metrics.

Artifacts:

- tools/analyze_e357_eflic_weighted_mixed_perceptual_selector.py
- tools/analyze_e358_eflic_perceptual_oracle_feature_audit.py
- experiments/analysis/e357_eflic_e347_weighted_mixed_selector_*.{md,json,csv}
- experiments/analysis/e358_eflic_e347_mixed_perceptual_oracle_feature_audit.{md,json,per_image.csv,feature_rows.csv,feature_audit.csv,summaries.csv}


## E359-E360 EF-LIC Image-Level Threshold Limit Audit

Status: Done. E359 reruns the simple threshold selector on the mixed perceptual teacher split (CLIC calibration plus Kodak, held CLIC eval). The calibration-selected threshold is `thr_riskm060_y_mismatch_ge_129`: calibration score -0.000402 but held score only -0.000006. This is far weaker than the held fixed risk -0.06 row (-0.000327) and fixed risk -0.10 (-0.000239), while the held oracle remains -0.001106.

E360 tests whether robust calibration objectives fix this: mean plus worst-score penalties, minimum calibration-win constraints, and calibration-worst caps. They do not. Worst-score penalties choose an almost all-no-op policy with held score +0.000097, while mean/win criteria remain weaker than fixed risk. Diagnostic rows ranked by held score can reach about -0.000586, but they are not selectable from calibration, so they are not paper evidence.

Decision: image-level thresholding is exhausted for now. The EF-LIC branch should move to local/sequential reliability control using decoder-visible local residual, slice, active-state, and index/codebook statistics. PSNR remains only a diagnostic; LPIPS/DISTS/MS-SSIM/bpp drive the paper-facing decision.

Artifacts:

- experiments/analysis/e359_eflic_e347_mixed_perceptual_threshold_selector.{md,json}
- experiments/analysis/e359_eflic_e347_mixed_perceptual_threshold_selector_all_policies.csv
- experiments/analysis/e359_eflic_e347_mixed_perceptual_threshold_selector_top_eval_diagnostic.csv
- experiments/analysis/e360_eflic_e347_threshold_objective_transfer_audit.{md,json,csv}


## E361-E364 EF-LIC PSNR Teacher Audit and Perceptual Slice Isolation

Status: Done. E361 audits the earlier PSNR-oriented EF-LIC teacher under the corrected generative/perceptual protocol. The answer is important: some PSNR-positive HCG choices are not perceptually good. On Kodak24, fixed risk -0.06 has mean dPSNR +0.010798 but perceptual score +0.000772, with 6 images where PSNR improves while score worsens. On CLIC Professional 41, the same risk has mean score -0.000257 but still includes 10 PSNR-win / score-loss images. Therefore the PSNR teacher is no longer a paper-main controller target; it remains useful only as a codec-health and context-diagnostic artifact.

E362-E364 then rerun EF-LIC slice isolation with LPIPS/DISTS/MS-SSIM enabled, using score = delta_DISTS + 3 * delta_LPIPS, where lower is better. All evaluated rows preserve the EF-LIC fixed-payload contract: mean delta bpp 0, decode max 0, nonfinite 0, and contract_ok_frac 1.0.

Kodak24 E364 set-level summary:

| active slices | score | dPSNR | score wins | PSNR wins | triple perceptual wins |
|---|---:|---:|---:|---:|---:|
| all | +0.000717 | +0.009803 | 11/24 | 12/24 | 4/24 |
| 0 | +0.000721 | +0.011736 | 13/24 | 13/24 | 3/24 |
| 1 | -0.000090 | +0.000653 | 11/24 | 14/24 | 4/24 |
| 2 | -0.000128 | +0.000886 | 17/24 | 13/24 | 3/24 |
| 3 | -0.000004 | -0.000052 | 11/24 | 6/24 | 4/24 |
| 1,2,3 | -0.000187 | +0.000727 | 15/24 | 11/24 | 4/24 |

The per-image oracle over the tested slice sets is much stronger: mean score -0.001198 with 24/24 perceptual wins. By contrast, a PSNR oracle over the same rows has mean dPSNR +0.020597 but mean perceptual score +0.000473, and its selected slice set differs from the perceptual oracle on 22/24 images. The score-oracle improves mean score over all-on by -0.001916, showing that local/slice selection is the real headroom, not dense activation.

Top PSNR traps include kodim01: slice0 gives dPSNR +0.012420 but score +0.008809; all-on gives dPSNR +0.001873 but score +0.008613. This is exactly why PSNR must not drive the generative EF-LIC/GLC branch.

Decision: PSNR-positive EF-LIC HCG variants are not automatically valid. The safe paper-facing branch is now perceptual local/sequential reliability control: keep EF-LIC's Representation-domain Decorrelation and fixed no-entropy payload unchanged, use HCG as a decoder-reproducible local quantizer-geometry edit, and select/fallback by LPIPS/DISTS/MS-SSIM with bpp/decode/nonfinite as hard constraints. Full EF-LIC training should wait until a frozen local perceptual controller beats fixed risk/no-op on held-out perceptual metrics; all-on and PSNR-teacher policies should not be promoted.

Artifacts:

- tools/analyze_e361_psnr_teacher_perceptual_audit.py
- tools/run_e313_eflic_slice_isolation_sweep.py
- experiments/analysis/e361_eflic_psnr_teacher_perceptual_audit.{md,json,joined.csv,feature_audit.csv,threshold_policies.csv,fixed_summary.csv}
- experiments/analysis/e362_eflic_perceptual_slice_isolation_smoke_kodak1_riskm060.{md,json,rows.csv,by_set.csv,by_image.csv}
- experiments/analysis/e363_eflic_perceptual_slice_isolation_kodak4_riskm060.{md,json,rows.csv,by_set.csv,by_image.csv}
- experiments/analysis/e364_eflic_perceptual_slice_isolation_kodak24_riskm060.{md,json,rows.csv,by_set.csv,by_image.csv}


## E365 EF-LIC Perceptual Slice Oracle Audit

Status: Done. E365 turns the E364 slice-isolation table into a reusable audit. It compares the perceptual oracle against the PSNR oracle and emits per-image rows plus PSNR-trap rows.

Key outputs confirm the metric correction:

- contract_ok: true for all 240 rows.
- all-on score: +0.000717 with dPSNR +0.009803.
- perceptual oracle score: -0.001198 with dPSNR +0.005307.
- PSNR oracle score: +0.000473 with dPSNR +0.020597.
- PSNR/perceptual oracle choices disagree on 22/24 images.
- PSNR-positive but perceptual-bad rows: 55.
- perceptual-good but PSNR-negative rows: 63.

Decision: the EF-LIC/GLC generative branch must treat PSNR as diagnostic only. The full-training gate should be based on LPIPS/DISTS/MS-SSIM/bpp and exact payload consistency. The previous PSNR teacher is now demoted to a diagnostic artifact, not a training target.

Artifacts:

- tools/analyze_e365_eflic_perceptual_slice_oracle_audit.py
- experiments/analysis/e365_eflic_perceptual_slice_oracle_audit_kodak24_riskm060.{md,json,per_image.csv,psnr_traps.csv}

## E366 EF-LIC Perceptual Candidate Policy LOO

Status: Done. E366 asks whether the E364 perceptual slice oracle can be partially
recovered from decoder-visible candidate/context statistics rather than selected
with PSNR. It uses leave-one-image-out ridge policies over the tested slice-set
candidates plus an explicit no-op candidate. PSNR is diagnostic only.

Key results on Kodak24:

| policy | mean score | worst score | score wins | mean dPSNR | choices |
|---|---:|---:|---:|---:|---|
| perceptual oracle | -0.001198 | -0.000015 | 24/24 | +0.005307 | mixed local choices |
| CV best fixed | -0.000041 | +0.001222 | 14/24 | +0.000896 | mostly 1,2,3 |
| fixed 1,2,3 | -0.000187 | +0.001222 | 15/24 | +0.000727 | all images |
| fixed 2 | -0.000128 | +0.001095 | 17/24 | +0.000886 | all images |
| ridge alpha 0.01 | -0.000522 | +0.001118 | 6/24 | +0.006931 | noop-heavy mixed choices |
| PSNR oracle | +0.000472 | +0.008800 | 14/24 | +0.020600 | PSNR-driven choices |

Decision: this is a useful promotion signal, not a final controller. A very
simple perceptual candidate policy already beats the fixed slice policies by
mean score, which supports the HCG-RVQ hypothesis that local/context-conditioned
geometry selection has real value. The weakness is that the policy is
noop-heavy and has only 6/24 score wins, so a full EF-LIC training run should
still wait for a tail/win-constrained local controller. The next controller
should optimize the perceptual mean while constraining worst score and minimum
win rate, rather than chasing PSNR or dense all-on activation.

Artifacts:

- tools/analyze_e366_eflic_perceptual_candidate_policy_loo.py
- experiments/analysis/e366_eflic_perceptual_candidate_policy_loo_kodak24_riskm060.{md,json,per_image.csv}

## E367 Perceptual-Only EF-LIC/GLC Parallel Status

Status: Done. E367 consolidates the current EF-LIC and GLC tracks into one
PSNR-free decision artifact. The decision metric is the perceptual score
`delta_DISTS + 3 * delta_LPIPS` with bpp/decode consistency and nonfinite rows as
hard checks. PSNR is intentionally excluded from the decision tables.

EF-LIC remains active in parallel, but it is not ready for final all-on or full
training. The current E366 Kodak24 signal is:

| policy | mean score | worst score | score wins |
|---|---:|---:|---:|
| perceptual oracle | -0.001198 | -0.000015 | 24/24 |
| learned LOO ridge | -0.000522 | +0.001118 | 6/24 |
| fixed slice 1,2,3 | -0.000187 | +0.001222 | 15/24 |
| fixed slice 2 | -0.000128 | +0.001095 | 17/24 |
| all-on | +0.000717 | +0.008613 | 11/24 |

The EF-LIC interpretation is stable: the useful signal is selective,
decoder-visible HCG geometry; the bottleneck is robust local reliability
selection. Full training should wait for a held-out local controller that beats
fixed slice/risk baselines in perceptual mean without expanding positive tails,
while preserving exact bpp/decode behavior and zero nonfinite rows.

GLC is also active in parallel and is currently closer to a larger perceptual
full-evaluation promotion. Its current signal-accounted selected-replacement
summary over the E285 CLIC-tail9 + E286 Kodak16 subset is:

| controller | score | fixed score | selected | selected wins | selected fixed wins | worst score | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict cap 0.0030 | -0.008977 | -0.007900 | 0.680000 | 1.000000 | 1.000000 | +0.000000 | 0 |
| balanced cap 0.0035 | -0.009606 | -0.008262 | 0.800000 | 1.000000 | 0.950000 | +0.000000 | 0 |
| aggressive cap 0.0040 | -0.010473 | -0.008466 | 1.000000 | 0.960000 | 0.880000 | +0.001513 | 0 |
| dense all-on | +0.045886 | +0.047893 | 1.000000 | 0.000000 | 0.000000 | +0.084652 | 0 |

Decision: EF-LIC and GLC should continue in parallel, but their HCG-RVQ designs
do not need to be identical. EF-LIC should preserve its original
representation-domain decorrelation and fixed-payload/no-entropy identity while
adding a local fallback controller. GLC should promote selected replacement with
explicit signal/index accounting: cap 0.0035 is the balanced paper-facing
controller, cap 0.0030 is the stricter fixed-index/no-entropy candidate, and cap
0.0040 remains an aggressive performance branch with tail failure analysis.

Artifacts:

- tools/build_e367_perceptual_parallel_status.py
- experiments/analysis/e367_perceptual_only_parallel_status.{md,json}

## E368 EF-LIC Perceptual Abstention Gate Audit

Status: Done. E368 tests whether the E366 learned EF-LIC policy can be made
paper-safer by a scalar abstention threshold on the predicted perceptual score.
This uses only `delta_DISTS + 3 * delta_LPIPS`; PSNR is excluded.

Best mean thresholds preserve the E366 learned-policy mean score `-0.000522`, but
still have a positive worst score `+0.001118` and only `6/24` score wins. The best
no-positive-tail thresholds avoid all positive tails, but drop to only `1-2/24`
selected improvements and mean scores around `-0.000238` to `-0.000220`.

Decision: image-level abstention is not enough for EF-LIC. It can make the policy
safe only by discarding most useful HCG decisions. The next EF-LIC implementation
should move the reliability decision closer to the actual codec operation:
slice/block-level local controller, explicit no-op fallback, perceptual labels,
and hard bpp/decode/nonfinite constraints.

Artifacts:

- tools/analyze_e368_eflic_perceptual_abstention_gate.py
- experiments/analysis/e368_eflic_perceptual_abstention_gate_kodak24.{md,json}

## E369 GLC Kodak24 Perceptual Signal-Accounted Expansion

Status: Done. E369 expands the current-code GLC signal-accounted replacement
audit from the smaller held subset to full Kodak24 while keeping GPU fixed to
device 0. The protocol matches the recent GLC selected-replacement audits:
OpenImages start8192 train-limit 16, q=0, K=4, active parts 0/1, active groups
1/7/10/15, 12 codec-loop steps, replacement caps 0.0025/0.0035/0.0040, and
1-bit/8-bit image-level signal rows. The aggregation artifact is intentionally
PSNR-free and uses LPIPS/DISTS/MS-SSIM/bpp plus nonfinite checks.

Kodak24-only summary:

| controller | score | fixed score | selected | wins | fixed wins | worst score | worst fixed | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trained soft gate | -0.016910 | -0.013487 | 1.000000 | 1.000000 | 1.000000 | -0.006990 | -0.003170 | 0 |
| replacement soft | -0.015299 | -0.013487 | 1.000000 | 1.000000 | 1.000000 | -0.005303 | -0.003170 | 0 |
| cap 0.0035 | -0.015055 | -0.013355 | 0.958333 | 0.958333 | 0.958333 | +0.000000 | +0.000000 | 0 |
| cap 0.0040 | -0.015299 | -0.013487 | 1.000000 | 1.000000 | 1.000000 | -0.005303 | -0.003170 | 0 |
| dense all-on | +0.038617 | +0.040429 | 1.000000 | 0.041667 | 0.041667 | +0.093912 | +0.095723 | 0 |

Pooled CLIC-tail9 + Kodak24 summary:

| controller | score | fixed score | selected | wins | fixed wins | selected wins | selected fixed wins | worst score | worst fixed | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cap 0.0030 | -0.012200 | -0.010834 | 0.787879 | 0.787879 | 0.787879 | 1.000000 | 1.000000 | +0.000000 | +0.000000 | 0 |
| cap 0.0035 | -0.012330 | -0.010831 | 0.848485 | 0.848485 | 0.818182 | 1.000000 | 0.964286 | +0.000000 | +0.000404 | 0 |
| cap 0.0040 | -0.013069 | -0.011065 | 1.000000 | 0.969697 | 0.909091 | 0.969697 | 0.909091 | +0.001513 | +0.004568 | 0 |
| dense all-on | +0.038981 | +0.040985 | 1.000000 | 0.030303 | 0.030303 | 0.030303 | 0.030303 | +0.093912 | +0.095723 | 0 |

Decision: GLC selected replacement is no longer only a subset artifact. It should
move to the next larger perceptual evaluation tier before full training:
broader/full CLIC Professional and a low-rate q-index curve. The promoted design
is still not dense RVQ. The stable story is decoder-reproducible selected local
replacement with explicit index/signal accounting. cap 0.0030 is the strict
fixed-index/no-positive-selected controller, cap 0.0035 is the balanced
paper-facing controller, and cap 0.0040 is the aggressive performance branch
that needs tail-failure reporting.

Artifacts:

- tools/analyze_e369_glc_perceptual_signal_pool.py
- experiments/analysis/e369_glc_signal_accounted_replacement_rows_kodak24_full_k4_parts01_t16_e24_s12.{md,json,csv}
- experiments/analysis/e369_glc_perceptual_signal_pool_kodak24_full.{md,json,summary.csv}
- experiments/analysis/e369_glc_perceptual_signal_pool_clictail9_kodak24.{md,json,summary.csv}

## E370 GLC CLIC Professional Full Perceptual Expansion

Status: Done. E370 is the next larger GLC evaluation tier after E369. It runs the
same current-code selected-replacement protocol on all 41 CLIC Professional
validation images with GPU fixed to device 0: OpenImages start8192 train-limit
16, q=0, K=4, active parts 0/1, active groups 1/7/10/15, 12 codec-loop steps,
replacement caps 0.0025/0.0035/0.0040, and 1-bit/8-bit image-level signal rows.
The decision artifacts remain PSNR-free and aggregate only LPIPS/DISTS/MS-SSIM,
bpp, fixed-index reinterpretation, signal cost, and nonfinite checks.

CLIC Professional 41 summary:

| controller | score | fixed score | selected | wins | fixed wins | selected wins | selected fixed wins | worst score | worst fixed | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trained soft gate | -0.012522 | -0.007510 | 1.000000 | 1.000000 | 0.804878 | 1.000000 | 0.804878 | -0.002311 | +0.006166 | 0 |
| replacement soft | -0.010191 | -0.007510 | 1.000000 | 0.926829 | 0.804878 | 0.926829 | 0.804878 | +0.001214 | +0.006166 | 0 |
| cap 0.0030 | -0.008761 | -0.007340 | 0.658537 | 0.658537 | 0.609756 | 1.000000 | 0.925926 | +0.000000 | +0.001377 | 0 |
| cap 0.0035 | -0.009890 | -0.007893 | 0.829268 | 0.829268 | 0.756098 | 1.000000 | 0.911765 | +0.000000 | +0.001700 | 0 |
| cap 0.0040 | -0.010177 | -0.007599 | 0.975610 | 0.902439 | 0.804878 | 0.925000 | 0.825000 | +0.001214 | +0.006166 | 0 |
| dense all-on | +0.030343 | +0.033025 | 1.000000 | 0.048780 | 0.024390 | 0.048780 | 0.024390 | +0.079387 | +0.080772 | 0 |

Pooled CLIC41 + Kodak24 summary:

| controller | score | fixed score | selected | wins | fixed wins | selected wins | selected fixed wins | worst score | worst fixed | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trained soft gate | -0.014143 | -0.009717 | 1.000000 | 1.000000 | 0.876923 | 1.000000 | 0.876923 | -0.002311 | +0.006166 | 0 |
| replacement soft | -0.012077 | -0.009717 | 1.000000 | 0.953846 | 0.876923 | 0.953846 | 0.876923 | +0.001214 | +0.006166 | 0 |
| cap 0.0030 | -0.011085 | -0.009561 | 0.769231 | 0.769231 | 0.738462 | 1.000000 | 0.960000 | +0.000000 | +0.001377 | 0 |
| cap 0.0035 | -0.011797 | -0.009910 | 0.876923 | 0.876923 | 0.830769 | 1.000000 | 0.947368 | +0.000000 | +0.001700 | 0 |
| cap 0.0040 | -0.012068 | -0.009773 | 0.984615 | 0.938462 | 0.876923 | 0.953125 | 0.890625 | +0.001214 | +0.006166 | 0 |
| dense all-on | +0.033398 | +0.035759 | 1.000000 | 0.046154 | 0.030769 | 0.046154 | 0.030769 | +0.093912 | +0.095190 | 0 |

Decision: this is the strongest GLC promotion signal so far. The CLIC41 result
confirms that E369 was not Kodak-specific. Dense all-on remains an explicit
negative control, while selected local replacement keeps large perceptual gains
with counted rate/signal cost. cap 0.0035 is now the best balanced default for
paper-facing selected replacement: it has stronger mean than cap 0.0030, no
empirical positive selected tail, and better fixed-index selected reliability
than cap 0.0040. cap 0.0030 remains the strict safest branch; cap 0.0040 remains
the aggressive high-mean branch with tail reporting.

Next: move GLC to a low-rate q-index curve and then decide whether the selected
replacement controller is strong enough to justify matched fine-tuning/full
training. EF-LIC remains parallel but should focus on local slice/block control
before full training.

Artifacts:

- experiments/analysis/e370_glc_signal_accounted_replacement_rows_clicpro41_full_k4_parts01_t16_e41_s12.{md,json,csv}
- experiments/analysis/e370_glc_perceptual_signal_pool_clicpro41_full.{md,json,summary.csv}
- experiments/analysis/e370_glc_perceptual_signal_pool_clicpro41_kodak24.{md,json,summary.csv}


## E371-E374 GLC q-Curve Perceptual Expansion

Status: Done. E371-E374 extend the GLC selected-replacement evidence from the
single q=0 setting to a low-rate q-index curve on CLIC Professional 41 and
Kodak24. PSNR is intentionally excluded from the decision protocol. The score is
`delta_DISTS + 3 * delta_LPIPS + delta_bpp`, where lower is better; fixed score
uses the fixed-index cost reinterpretation when a row is selected. All runs used
`CUDA_VISIBLE_DEVICES=0` and `cuda:0`; no nonfinite rows were observed.

Protocol: official GLC checkpoint, OpenImages start8192 train-limit 16,
q-indexes 0/1/2/3, K=4, one RVQ stage, active parts 0/1, active groups
1/7/10/15, 12 codec-loop steps, replacement rows with explicit signal/index
accounting, and evaluation on CLIC Professional 41 plus Kodak24.

CLIC Professional 41 q-curve summary:

| controller | rows | score | fixed score | win frac | fixed win frac | worst score | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| replacement soft | 164 | -0.007837 | -0.004995 | 0.865854 | 0.701220 | +0.007873 | 0 |
| cap 0.0035 | 164 | -0.007679 | -0.005008 | 0.829268 | 0.682927 | +0.007873 | 0 |
| dense all-on | 164 | +0.029691 | +0.032533 | 0.048780 | 0.018293 | +0.081461 | 0 |

Kodak24 q-curve summary:

| controller | rows | score | fixed score | win frac | fixed win frac | worst score | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| replacement soft | 96 | -0.011288 | -0.009359 | 0.989583 | 0.968750 | +0.000519 | 0 |
| cap 0.0035 | 96 | -0.011193 | -0.009292 | 0.979167 | 0.958333 | +0.000519 | 0 |
| dense all-on | 96 | +0.043035 | +0.044964 | 0.031250 | 0.020833 | +0.095126 | 0 |

Pooled CLIC Professional 41 + Kodak24 summary:

| controller | rows | score | fixed score | win frac | fixed win frac | worst score | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| replacement soft | 260 | -0.009111 | -0.006606 | 0.911538 | 0.800000 | +0.007873 | 0 |
| cap 0.0035 | 260 | -0.008977 | -0.006590 | 0.884615 | 0.784615 | +0.007873 | 0 |
| dense all-on | 260 | +0.034618 | +0.037123 | 0.042308 | 0.019231 | +0.095126 | 0 |

Feature-risk audits show that decoder-side reliability can remove the positive
tail on selected rows, but the current thresholds are still diagnostic and
in-sample. On CLIC41, `index_entropy_mean >= 1.77551` selects 50/164 rows with
score_all -0.003679, selected win fraction 1.0, and selected worst score
-0.001221. On Kodak24, `index_entropy_mean >= 1.64554` selects 87/96 rows with
score_all -0.010717, selected win fraction 1.0, and selected worst score
-0.000619. On pooled CLIC41+Kodak24, `index_entropy_mean >= 1.78232` selects
79/260 rows with score_all -0.003947, selected win fraction 1.0, and selected
worst score -0.001221.

Decision: GLC is now the strongest near-term promotion path. The result survives
the corrected generative/perceptual protocol and confirms that the useful design
is not dense VQ/RVQ activation. It is sparse, local, decoder-reproducible
replacement with explicit rate/signal accounting. The next GLC implementation
target is a q-aware and index-entropy-aware reliability controller, followed by
matched fine-tuning/full training if held-out reliability remains favorable.
EF-LIC remains active in parallel, but should wait for a local slice/block
controller that preserves its fixed-payload/no-entropy contract.

Artifacts:

- tools/analyze_e371_glc_qcurve_perceptual.py
- tools/analyze_e372_glc_qcurve_feature_risk.py
- experiments/analysis/e371_glc_signal_accounted_replacement_rows_clicpro41_qcurve_k4_parts01_t16_e41_s12.{md,json,csv}
- experiments/analysis/e371_glc_perceptual_qcurve_clicpro41.{md,json,summary.csv}
- experiments/analysis/e372_glc_qcurve_feature_risk_clicpro41.{md,json}
- experiments/analysis/e373_glc_signal_accounted_replacement_rows_kodak24_qcurve_k4_parts01_t16_e24_s12.{md,json,csv}
- experiments/analysis/e373_glc_perceptual_qcurve_kodak24.{md,json,summary.csv}
- experiments/analysis/e373_glc_perceptual_qcurve_clicpro41_kodak24.{md,json,summary.csv}
- experiments/analysis/e373_glc_qcurve_feature_risk_kodak24.{md,json}
- experiments/analysis/e374_glc_qcurve_feature_risk_clicpro41_kodak24.{md,json}


## E375 GLC q-Curve Feature Transfer Audit

Status: Done. E375 checks whether the q-curve reliability signal transfers across
datasets instead of only fitting the dataset where the threshold is selected.
The audit is PSNR-free and uses the same perceptual score as E371-E374:
`delta_DISTS + 3 * delta_LPIPS + delta_bpp`. Policies are selected on one source
set and evaluated on the other target set; unselected rows contribute zero.

Best CLIC41-to-Kodak24 source-safe policy: `index_entropy_mean >= 1.773657`.
It selects 51/164 CLIC rows with source score_all -0.003728 and source selected
worst score -0.001221. Transferred to Kodak24, it selects 40/96 rows with target
score_all -0.005449, selected win fraction 1.0, selected fixed win fraction 1.0,
and target selected worst score -0.003992.

Best broad Kodak24-to-CLIC policies by source score do not control the CLIC tail.
For example, `active_scalar_mse >= 0.005471` has strong Kodak source score_all
-0.011004, but transferred to CLIC it leaves target selected worst score
+0.007873. However, high-entropy policies do transfer safely in both directions:
`index_entropy_mean >= 1.772773` selected on Kodak24 chooses 41/96 source rows
with source score_all -0.005455 and source worst -0.000619; transferred to CLIC41,
it selects 51/164 rows with target score_all -0.003728, selected win fraction
1.0, and target selected worst score -0.001221.

Decision: the next GLC controller should not be a broad scalar-MSE or all-on
controller. It should be q-aware and index-entropy/reliability-first, using high
index entropy as the conservative default and treating wider active-residual
coverage as an aggressive branch. This is enough evidence to implement the
controller before moving to long matched fine-tuning/full training.

Artifacts:

- experiments/analysis/e375_glc_qcurve_feature_transfer_clic41_kodak24.{md,json}
- experiments/analysis/e375_glc_qcurve_feature_transfer_clic41_kodak24.clic_to_kodak.csv
- experiments/analysis/e375_glc_qcurve_feature_transfer_clic41_kodak24.kodak_to_clic.csv

## E376 GLC q-Aware Reliability Controller Spec

Status: Done. E376 turns the E371-E375 GLC finding into a reusable controller
spec audit. The protocol is perceptual-only: PSNR columns are ignored, the main
score is `delta_DISTS + 3 * delta_LPIPS + delta_bpp`, and MS-SSIM is only side
reported. Policies are fit on a source split and evaluated on the opposite split;
unselected rows contribute exactly zero, matching explicit fallback behavior.

CLIC41-to-Kodak24 gives the strongest evidence for q-aware entropy control. The
best paper-facing candidate is `score+fixed-tail q-aware index_entropy_mean >=
[q0:1.63703,q1:1.73144,q2:1.79641,q3:1.80155]`: it selects 54/96 Kodak rows,
gets target score_all -0.007759, fixed-index score_all -0.006857, selected win
fraction 1.0, selected fixed-index win fraction 1.0, target worst score
-0.003628, and target worst fixed score -0.001543. A less strict q-aware
entropy profile selects 58/96 rows and remains tail-safe as well.

Kodak24-to-CLIC41 is more conservative. Source-fit q-aware policies can over-cover
CLIC and leave positive tails, so the safer cross-dataset default is global high
entropy. `score+fixed-tail global index_entropy_mean >= 1.77911` selects 46/164
CLIC rows, gets target score_all -0.003417, selected empirical win fraction 1.0,
and target worst score -0.001221, although fixed-index worst remains slightly
positive at +0.000370. A very conservative `active_mse_ratio >= 2.85176` removes
both empirical and fixed-index tails, but selects only 10/164 rows and is too low
coverage for the main controller.

Decision: the earlier "high index entropy removes the positive tail" observation
is real but needs precise wording. It is strong for conservative high-entropy
fallback and very promising for CLIC-to-Kodak q-aware entropy thresholds; however,
source-fit q-aware thresholds are not yet paper-safe in the reverse direction.
The next GLC branch should implement an entropy-first reliability controller with
explicit fallback, then add a held-out calibration split before long matched
fine-tuning/full training. EF-LIC remains active in parallel, but its promotion
path is still local slice/block reliability under the fixed-payload/no-entropy
contract, not an all-on branch.

Artifacts:

- tools/analyze_e376_glc_qaware_reliability_controller_spec.py
- hcg_rvq/reliability_index_controller.py
- experiments/analysis/e376_glc_qaware_reliability_controller_spec.md
- experiments/analysis/e376_glc_qaware_reliability_controller_spec.json
- experiments/analysis/e376_glc_qaware_reliability_controller_spec.clic_to_kodak.csv
- experiments/analysis/e376_glc_qaware_reliability_controller_spec.kodak_to_clic.csv

## E377 GLC q-Aware Held-Out Reliability Calibration

Status: Done. E377 converts the E376 q-aware entropy finding into an image-disjoint held-out calibration audit. It pools the CLIC Professional 41 and Kodak24 q-curve rows, fits reliability thresholds on calibration folds, and evaluates held-out folds using the perceptual-only score `delta_DISTS + 3 * delta_LPIPS + delta_bpp`; PSNR is ignored.

The strict pooled q-aware `index_entropy_mean` policy selects 110/260 held-out rows with score -0.005434 and fixed-index score -0.004732. Empirical positives are 0, but fixed-index positives remain 2 with worst fixed score +0.000370. Kodak24 is fully safe under this policy; the small remaining tail comes from CLIC41. This means the q-aware entropy signal is real, but the no-margin policy is not yet paper-safe enough for the main branch.

Artifacts:

- tools/analyze_e377_glc_qaware_heldout_calibration.py
- experiments/analysis/e377_glc_qaware_heldout_calibration.{md,json,summary.csv,folds.csv}

## E378 GLC Entropy Safety-Margin Held-Out Sweep

Status: Done. E378 adds conservative threshold margins to the E377 strict high-entropy policies and re-evaluates them on the same image-disjoint held-out folds. The best paper-facing pooled policy is q-aware `index_entropy_mean` with margin 0.02: it selects 92/260 rows, reaches score -0.004717 and fixed-index score -0.004147, keeps both empirical and fixed-index win fractions at 1.0, and removes all positive tails. Worst empirical score is -0.002180 and worst fixed-index score is -0.000325.

A simpler global entropy policy with margin 0.01 is also safe but weaker: it selects 59/260 rows with score -0.002900 and fixed-index score -0.002599. The decision is to promote GLC to a matched fine-tuning/full-training branch with the q-aware high-entropy controller plus margin 0.02, keep the global margin 0.01 policy as a simple ablation, and keep no-margin q-aware policies as diagnostic/high-gain ablations only.

Artifacts:

- tools/analyze_e378_glc_entropy_margin_heldout.py
- experiments/analysis/e378_glc_entropy_margin_heldout.{md,json,summary.csv,folds.csv}

## E379 GLC q-Aware Entropy-Margin Deployment Spec

Status: Done. E379 turns the E378 promotion decision into an execution artifact for the next GLC matched fine-tuning/full-training branch. Unlike E378, this is not a validation audit; it fits the selected controller on all currently available CLIC41+Kodak24 calibration rows and exports the deterministic spec to pass into the codec branch.

The main deployment spec is q-aware `index_entropy_mean >=` with margin 0.02 and thresholds `{0: 1.6570271146297455, 1: 1.747296993136406, 2: 1.8121935766935349, 3: 1.8215530556440354}`. Same-row diagnostics remain tail-safe: pooled selects 92/260 rows with score -0.004709, fixed-index score -0.004141, empirical/fixed win fractions 1.0, and positive rows 0/0. The global entropy margin policy is exported as the simple ablation.

Decision: GLC is now ready for a first matched long-run branch using this q-aware entropy-margin controller, scalar fallback, and perceptual-only evaluation. The paper claim should cite E378 held-out safety for validation and E379 only as the deployment configuration.

Artifacts:

- tools/build_e379_glc_qaware_entropy_margin_deployment_spec.py
- experiments/analysis/e379_glc_qaware_entropy_margin_deployment_spec.{md,json,csv}

## E380 GLC q-Aware Controller Pilot Integration Smoke

Status: Done. E380 wires the E379 deployment spec into the GLC codec-loop pilot
instead of keeping it as an offline analysis-only artifact. The pilot can now
load an exported q-aware/global threshold JSON, emit replacement rows selected
by decoder-reproducible `index_entropy_mean` thresholds, and optionally account
for an explicit image-level selection signal.

This is an implementation smoke, not a paper claim. It ran on GPU0 only with
`q=0`, one OpenImages training crop, one Kodak evaluation image, one optimization
step, and perceptual-only decision score `delta_DISTS + 3 * delta_LPIPS +
delta_bpp`. The q-aware margin-0.02 controller selected `kodim01.png` because
its `index_entropy_mean` was 1.785650/1.786400, above the q=0 threshold
1.657027. The global-margin ablation did not select it because the global
threshold is 1.816405.

The smoke behaved as intended: dense all-on remained bad (`trained_all_on`
score +0.134986), q-aware replacement was beneficial (`trained_qaware_q_aware
...` score -0.016558; -0.016556 with one signal bit), and global fallback stayed
exactly at baseline except for the optional signal-bit row. There were no
nonfinite rows. This confirms the next longer GLC run can use the same script
with the E379 JSON, while E378 remains the validation evidence for the claim.

Artifacts:

- tools/run_e263_glc_fallback_gate_codec_loop_pilot.py
- experiments/analysis/e380_glc_qaware_controller_smoke.{md,csv}


## E381/E382 GLC q-Aware Entropy-Margin q-Curve Validation

Status: Done. E381/E382 move the E379 q-aware deployment spec from one-image smoke to
matched q-curve codec-loop probes on Kodak24 and CLIC Professional 41. Both runs
used GPU0 only, q=0/1/2/3, 16 OpenImages training crops from start8192, 12 pilot
steps, active parts 0/1 and groups 1/7/10/15, with perceptual-only scoring
`delta_DISTS + 3 * delta_LPIPS + delta_bpp`. PSNR columns remain in legacy CSVs
but are ignored for decisions and paper-facing claims. No nonfinite rows appeared
in either run.

Kodak24 (E381) confirms the controller behavior seen in held-out calibration.
The q-aware `index_entropy_mean` margin-0.02 policy selects 45/96 rows, gets
score -0.006762, fixed-index score -0.006048, selected win fraction 1.0,
selected fixed-index win fraction 1.0, and zero empirical/fixed positive tails.
The global-margin ablation is safe but weaker: 25/96 selected, score -0.003633,
fixed score -0.003325. Dense all-on is harmful (+0.041926 score), while raw
soft replacement has a stronger mean (-0.011319) but leaves 3 empirical and 4
fixed positive selected rows.

CLIC Professional 41 (E382) gives the more important stress test. The same
q-aware controller selects 46/164 rows, gets score -0.003405, fixed-index score
-0.002932, selected win fraction 1.0, selected fixed-index win fraction 1.0,
and zero empirical/fixed positive tails. The global ablation again remains safe
but weaker: 21/164 selected, score -0.001449, fixed score -0.001277. Raw soft
replacement has a stronger mean (-0.007914) but leaves 21 empirical and 48
fixed positive selected rows; all-on is strongly harmful (+0.029270).

Decision: GLC should be promoted to a longer matched fine-tuning/full-training
branch with q-aware high-entropy reliability, exact scalar fallback, explicit
index/signal accounting, and checkpoint/codebook monitoring. E378 remains the
paper-facing held-out validation evidence; E381/E382 show that the exported E379
controller behaves correctly inside the codec-loop path on both Kodak and CLIC.

Artifacts:

- experiments/analysis/e381_glc_qaware_entropy_margin_kodak24_qcurve.{md,csv}
- experiments/analysis/e381_glc_qaware_entropy_margin_kodak24_qcurve_tail_summary.{md,csv}
- experiments/analysis/e382_glc_qaware_entropy_margin_clicpro41_qcurve.{md,csv}
- experiments/analysis/e382_glc_qaware_entropy_margin_clicpro41_qcurve_tail_summary.{md,csv}

## E383/E384 EF-LIC Perceptual Feature-Gate Split Diagnostics

Status: Done. E383/E384 keep the EF-LIC branch active in parallel with GLC, but evaluate it under the corrected generative/perceptual protocol. The diagnostic uses only `delta_DISTS + 3 * delta_LPIPS`; PSNR is excluded from selection and interpretation. It searches simple policies of the form "use one fixed slice candidate if a decoder-visible feature passes a threshold, otherwise noop" on a Kodak train split and evaluates the selected threshold on the held-out split.

This is not yet full evidence, but it improves the EF-LIC readout over E366/E368. The earlier image-level predicted-score abstention could remove positive tails only by keeping 1-2/24 images, with best no-positive-tail mean score -0.000238. E383 finds held-out tail-safe feature gates with better useful coverage: training on Kodak01-16 and evaluating Kodak17-24, the best safe policy selects 3/8 held-out images with score -0.000851, worst score +0.000000, and zero positive rows (`0,1,3` gated by `slice3_family_zero_prob_mean <= 0.743521`). E384 reverses the split direction by training on Kodak09-24 and evaluating Kodak01-08; it remains tail-safe but lower coverage, selecting 1/8 held-out image with score -0.000782 and zero positive rows.

Decision: EF-LIC should not move to dense all-on or image-level policy training. There is real perceptual oracle headroom, and E383/E384 show that decoder-visible local reliability features can find safe regions, but coverage is still too small for a paper-main claim. The next EF-LIC step is a local/slice-block reliability controller that combines `family_zero_prob`, slice risk, and local geometry/error features, then validates on CLIC Professional as well as Kodak before any longer matched training.

Artifacts:

- tools/analyze_e383_eflic_perceptual_feature_gate_split.py
- experiments/analysis/e383_eflic_perceptual_feature_gate_split_kodak16_8.{md,json}
- experiments/analysis/e384_eflic_perceptual_feature_gate_split_kodak09_24_to_01_08.{md,json}

## E385-E388 EF-LIC CLIC Professional Perceptual Slice Reliability

Status: Done. E385 extends the EF-LIC perceptual slice-isolation protocol from
Kodak24 to CLIC Professional 41 using the same fixed-payload HCG branch,
`CUDA_VISIBLE_DEVICES=0`, `cuda:0`, LPIPS-vgg, and the paper-facing perceptual
score `delta_DISTS + 3 * delta_LPIPS`. PSNR is emitted by legacy code only as a
codec-health diagnostic and is ignored for decisions. All rows preserved the
EF-LIC contract: `contract_ok_frac=1.0`, `mean_delta_bpp=0`, decode max 0, and
no nonfinite rows.

CLIC confirms the main EF-LIC risk pattern. Dense/all-on activation is not safe
despite a slightly negative mean: all-on has mean score -0.000257 but worst
score +0.003001. Single/fixed slice choices are also mixed; the best fixed
mean is `0,1` at -0.000281, but its worst score is still +0.002700. Therefore
EF-LIC should not advance as dense all-on or fixed global slice activation.

E386/E387 run CLIC-only train/eval split audits with a train positive-tail
constraint of zero. The first split finds tail-safe policies such as activating
`0,1` when `slice1_avg_geometry_delta_rms >= 0.000608126`, selecting 2/21
held-out images with score -0.000263 and zero positives. The reverse split is
more conservative: the best safe policies select 1-2/20 held-out images with
scores around -0.000106 to -0.000085 and zero positives.

E388 tests full dataset transfer between Kodak24 and CLIC41. Tail-safe policies
do transfer, but coverage remains small. CLIC -> Kodak selects 2/24 images with
score -0.000321 and zero positives using `slice0_stage0_geometry_delta_min`.
Kodak -> CLIC selects 5/41 images with score -0.000199 and zero positives. The
best non-safe Kodak -> CLIC mean (-0.000285) selects 16/41 images but leaves
6 positive rows, so it is not paper-main.

Decision: EF-LIC remains a valid direct RVQ plug-in branch, but its next step is
not full training yet. The evidence says HCG-RVQ must be converted from
per-image fixed-action gating into a slice/block-local reliability controller
that increases safe coverage while keeping EF-LIC's Representation-domain
Decorrelation, no-entropy/fixed-payload contract, exact fallback, and
perceptual-only objective. GLC remains the branch that has already passed the
promotion gate for a longer matched run.

Artifacts:

- tools/analyze_e388_eflic_cross_dataset_feature_gate.py
- experiments/analysis/e385_eflic_perceptual_slice_isolation_clicpro41_riskm060.{md,json,rows.csv,by_set.csv,by_image.csv}
- experiments/analysis/e386_eflic_perceptual_feature_gate_split_clic20_21.{md,json}
- experiments/analysis/e387_eflic_perceptual_feature_gate_split_clic21_20.{md,json}
- experiments/analysis/e388_eflic_cross_dataset_feature_gate_kodak_clic.{md,json,summary.csv}
