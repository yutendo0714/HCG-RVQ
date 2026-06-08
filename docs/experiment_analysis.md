# HCG-RVQ Experiment Analysis

Date: 2026-05-29

## Current Re-Evaluation Note

Important correction: the latest audit separates two different current-code val4096 slices. Files named `*_reeval_current.csv` were run without `--start-index`, so they evaluate the first 4096 OpenImages images (`start_index=0`). They are useful current-code sanity checks, but they are not the paper-facing holdout. The paper-facing current audit is now the true holdout4096 sweep with `start_index=4096`, `max_images=4096`.

The older val4096 CSVs that reported RD around 2.9 for the same seed/method names are stale relative to the current code/checkpoints and should remain historical diagnostics only.

True holdout4096 current-code checkpoint-selected 3-seed results are now:

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

Aggregate:

| method | mean RD | mean delta vs HCS | wins vs HCS | mean bpp | mean bpp_y | mean bpp_z | mean PSNR | mean MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HCS-RVQ | 2.263705 | +0.000000 | 0/3 | 0.312820 | 0.082688 | 0.230132 | 21.374270 | 0.779006 |
| old gate0.25 | 2.233542 | -0.030164 | 2/3 | 0.313678 | 0.083546 | 0.230132 | 21.448400 | 0.780873 |
| min090 inverse/detached risk | 2.230108 | -0.033597 | 2/3 | 0.312351 | 0.082219 | 0.230132 | 21.467035 | 0.779186 |

Interpretation:

- The true holdout4096 audit confirms a real positive pilot signal: both HCG-RVQ-H gate variants beat HCS on the 3-seed mean.
- min090 is numerically best in this audit by a small margin (`-0.003434` RD vs old gate0.25), but the margin is too small and the seed behavior is too different to declare it final paper-main.
- old gate0.25 wins seeds 1234/2345 and loses seed3456; min090 rescues seed3456 and still wins seed2345 but damages seed1234. This is stronger evidence for selective reliability control than for either global rule as a final answer.
- The per-image/oracle and intermediate-feature analysis has now been completed; the next method action is a selective reliability controller that can choose old-like geometry on images/seeds where it helps and min090-like suppression where old gate harms.
- Do not claim SOTA comparison yet. These remain MeanScaleHyperprior/RVQ pilot ablations, not multi-rate full-dataset comparisons against RDVQ, MambaIC, HPCM, DCAE, ELIC/MLIC/TCM, or official pretrained baselines.

Summary artifact: `experiments/analysis/gate025_min090_multiseed_val4096_holdout4096_current_summary.md`.

Seed3456 true-holdout per-image and feature diagnostics were also refreshed because this is the seed where old gate0.25 fails and min090 succeeds:

- old gate0.25 vs HCS: mean delta RD `+0.022359`, wins `1883/4096`.
- min090 vs HCS: mean delta RD `-0.035673`, wins `2356/4096`.
- min090 vs old: mean delta RD `-0.058031`; it beats old on `0.879883` of images.
- The previous start0/stale feature narrative should not be reused verbatim. On the true holdout seed3456 best checkpoints, `s_q_mean` is `0.448851 -> 0.449871`, raw gate is `0.292674 -> 0.296461`, and effective gate is `0.292674 -> 0.281185` due to the risk multiplier mean `0.947354`.
- Quartiles by HCS RD show min090 improves all buckets relative to old, but Q4 remains positive vs HCS: Q1 old/risk `+0.006029 -> -0.045590`, Q4 old/risk `+0.101288 -> +0.030999`.
- The co-adaptation concern remains plausible but not proven by these true-holdout numbers. Detached risk prevents direct risk-signal gradient pressure on `s_q`, while training can still jointly move `s_q`, raw gate, effective gate, and RVQ fit through the RD objective.

Tail/feature artifact: `experiments/analysis/gate025_min090_seed3456_tail_features_val4096_holdout4096_current.md`.


Old/min090 per-image oracle and feature analysis has now been completed for seeds 1234/2345/3456 on the same true holdout4096 slice:

| scope | HCS RD | old delta | min090 delta | oracle delta | oracle gain vs best fixed | old selected | min090 selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | -0.018379 | +0.016937 | -0.052511 | -0.034131 | 61.84% | 38.16% |
| 2345 | 2.316296 | -0.094470 | -0.082056 | -0.095886 | -0.001415 | 86.77% | 13.23% |
| 3456 | 2.263345 | +0.022359 | -0.035673 | -0.040231 | -0.004558 | 12.01% | 87.99% |
| ALL | 2.263705 | -0.030164 | -0.033597 | -0.062876 | -0.029278 | 53.54% | 46.46% |

Interpretation update:

- The aggregate fixed-rule difference is tiny: min090 is only `-0.003434` RD better than old gate0.25.
- The per-image oracle is much stronger: `-0.062876` RD vs HCS, and `-0.029278` RD better than the best fixed global rule.
- The oracle chooses old gate0.25 on 53.54% of images and min090 on 46.46%, so the next paper-track method should be selective reliability control rather than another single global risk floor.
- Seed1234 explains the danger of min090 as a fixed main method: `s_q_mean` drops `0.569798 -> 0.459787`, effective strength rises `0.257688 -> 0.279810`, Householder delta RMS rises `0.062684 -> 0.100800`, and latent quantization MSE rises `0.056989 -> 0.112679`.
- Seed3456 explains the danger of old gate0.25 as a fixed main method: min090 beats old on 87.99% of images and changes the seed mean from `+0.022359` to `-0.035673` delta RD.
- Seed2345 is the stable-control case: both variants beat HCS, old is better, and min090 mostly suppresses effective geometry without changing latent quantization MSE.

Oracle/feature artifact: `experiments/analysis/gate025_min090_oracle_feature_reading_val4096_holdout4096_current.md`.


A current-holdout single-feature selector analysis was also run with `tools/analyze_gate_selector.py --protocol current_holdout`:

| policy | mean delta RD | vs old gate0.25 | min090 fraction |
|---|---:|---:|---:|
| old gate0.25 | -0.030164 | +0.000000 | 0.000000 |
| min090 risk | -0.033597 | -0.003434 | 1.000000 |
| oracle old/min090 | -0.062876 | -0.032712 | 0.464600 |
| best single feature: old_raw_gate_mean >= 0.260788 | -0.049500 | -0.019336 | 0.356445 |

This threshold result is useful but not final: it recovers a large part of the oracle gap, yet remains `+0.013376` RD worse than the oracle and partly reflects seed-level calibration differences. It supports constrained selective reliability control rather than another global gate floor.

A fold-calibrated selector check was added to guard against over-reading a threshold chosen on the same holdout images. With 4 image-index folds, each fold selected `old_raw_gate_mean >= 0.2608` or `>= 0.2768`; the held-out mean delta was `-0.049023` RD, only `+0.000477` worse than the full-data threshold (`-0.049500`) and `-0.018859` better than old gate0.25. Seed deltas were also stable: seed1234 `-0.017248`, seed2345 `-0.094277`, seed3456 `-0.035545`. This makes the constrained selector direction more credible than the failed free reliability head below.

Selector artifacts: `experiments/analysis/gate025_min090_selector_val4096_holdout4096_current_summary.md` and `experiments/analysis/gate025_min090_selector_cv_val4096_holdout4096_current.md`.

The detached-input learnable reliability controller was then trained and evaluated on the same true holdout4096 protocol for seeds 1234/2345/3456:

| seed | HCS RD | old gate0.25 RD | min090 RD | detached reliability RD | delta vs HCS |
|---:|---:|---:|---:|---:|---:|
| 1234 | 2.211475 | 2.193095 | 2.228412 | 2.844109 | +0.632634 |
| 2345 | 2.316296 | 2.221826 | 2.234240 | 2.863034 | +0.546738 |
| 3456 | 2.263345 | 2.285704 | 2.227673 | 2.920301 | +0.656956 |
| mean | 2.263705 | 2.233542 | 2.230108 | 2.875814 | +0.612109 |

This learned detached reliability branch is not a paper-main candidate. It loses to HCS, old gate0.25, and min090 on every seed. Intermediate-feature diagnostics make the failure more specific: the reliability multiplier remains near initialization (`0.990675`, `0.990519`, `0.990340`), raw/effective Householder strength stays near the old gate0.25 regime, and latent quantization MSE stays high around `0.075-0.077`. So the failure is not "reliability suppressed geometry too much"; it is more consistent with the extra free reliability branch pushing training into a bad quantizer/decoder basin.

The conclusion is therefore updated: selective reliability control remains the main next direction, but not as this naive free RD-trained detached head. The next implementation should be a constrained selector that approximates the old/min090 oracle or the deployable raw-gate threshold while keeping the quantizer evidence distribution fixed or strongly regularized.

Detached reliability artifact: `experiments/analysis/reliability_detach_min05_init099_val4096_holdout4096_current_summary.md`.

## Executive Read

The current pilot evidence supports the prompt central thesis: the hyperprior can improve compression not only by predicting entropy parameters, but also by generating local quantizer geometry. Under the true holdout4096 current-code checkpoint-selected protocol, both tested HCG-RVQ-H gate variants improve over HCS-RVQ on the 3-seed mean.

Current paper-facing pilot claim:

- HCS-RVQ remains the stable internal baseline over Global RVQ because hyper-conditioned shift/scale and index-prior entropy coding make RVQ usable at this rate.
- HCG-RVQ-H old gate0.25 improves HCS on the true holdout4096 3-seed mean: RD `2.233542` vs `2.263705`, delta `-0.030164`, wins `2/3`.
- HCG-RVQ-H min090 inverse/detached risk is numerically best in the current audit: RD `2.230108` vs HCS `2.263705`, delta `-0.033597`, wins `2/3`.
- The min090 margin over old gate0.25 is small (`-0.003434` RD), and the seed profiles are complementary rather than settled: old gate0.25 wins seeds 1234/2345, while min090 rescues seed3456 but damages seed1234.
- Therefore the current scientific conclusion is not simply "min090 is the final method". The stronger conclusion is that geometry is useful, but needs selective reliability control.
- The detached-input learnable reliability branch has now been retested and should remain diagnostic only. It fails badly on the true holdout4096 3-seed protocol, so the next method should use constrained/oracle-approximating selection rather than a free RD-trained reliability head.

Interpretation for the proposal:

- The core geometry mechanism is real enough to keep: both gate variants improve the true holdout4096 3-seed mean and improve PSNR means.
- The main unresolved scientific question is selective reliability: how to preserve old-gate gains on seeds/images where geometry helps while applying min090-like suppression where old gate harms.
- The old/min090 per-image oracle has now been estimated on the true holdout. The next method action is to train or design a reliability controller that approximates that decision without moving the hyperprior frame.
- Do not claim comparison to latest/SOTA LIC or VQ-GIC yet. Current results are small MeanScaleHyperprior/RVQ pilot ablations, not multi-rate full-dataset comparisons against RDVQ, MambaIC, HPCM, DCAE, ELIC/MLIC/TCM, or official pretrained baselines.

Immediate next actions:

1. Drop the naive free detached reliability branch from the paper-main path.
2. Promote the fold-validated raw-gate selector into the next paper-track constrained policy, then test whether a frozen-evidence lightweight classifier closes the remaining oracle gap.
3. Evaluate the selector on another slice/dataset and then rerun checkpoint-selected comparisons against HCS, old gate0.25, min090, the best threshold, and the old/min090 oracle.
4. Only after the selector is stable, move to multi-rate paper curves and external comparisons.

## Result Table

| Experiment | Steps | Checkpoint | bpp | bpp_y | bpp_z | PSNR | MS-SSIM | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Scalar baseline | 500 | latest | 0.283923 | 0.058857 | 0.225066 | 19.152838 | 0.625586 | Low rate, poor quality |
| Global RVQ direct | 500 | latest | 0.356681 | - | - | 16.706942 | 0.604188 | Commit loss exploded |
| Global RVQ init | 0 | init | 0.361784 | 0.136719 | 0.225066 | 19.802803 | 0.675989 | Scalar-initialized codebook |
| Global RVQ frozen | 500 | latest | 0.361784 | 0.136719 | 0.225066 | 21.047032 | 0.723126 | Stable but high fixed index rate |
| HCS-RVQ frozen | 500 | latest | 0.316533 | 0.091467 | 0.225066 | 21.152914 | 0.750836 | Main positive HCS result |
| HCG-RVQ-H no-transform | 500 | latest | 0.316533 | 0.091467 | 0.225066 | 21.153417 | 0.750906 | Matched geometry control |
| HCG-RVQ-H frozen | 500 | latest | 0.317002 | 0.091937 | 0.225066 | 21.252954 | 0.752188 | Small geometry-positive signal |
| HCG-RVQ-H staged h_s | 1000 | latest | 0.332727 | 0.107661 | 0.225066 | 16.559856 | 0.442196 | h_s unfreeze collapse |
| HCG-RVQ-H staged h_s slow | 1000 | latest | 0.330909 | 0.105844 | 0.225066 | 18.108051 | 0.654894 | Better but still unstable |
| HCG-RVQ-H staged slow + drift reg | 1000 | latest | 0.319786 | 0.094720 | 0.225066 | 19.583592 | 0.653792 | Controls collapse, hurts warmup |
| HCG-RVQ-H staged slow + after500 reg | 500 | step_500 | 0.316976 | 0.091910 | 0.225066 | 21.285610 | 0.751540 | Best pilot checkpoint, before reg active |
| HCG-RVQ-H staged slow + after500 reg | 1000 | latest | 0.329011 | 0.103945 | 0.225066 | 19.789570 | 0.648762 | Still degrades after unfreeze |
| HCG-RVQ-H staged h_s anchor after500 | 750 | step_750 | 0.325913 | 0.100847 | 0.225066 | 20.428828 | 0.725705 | Slightly better mid-unfreeze RD, higher rate |
| HCG-RVQ-H staged h_s anchor after500 | 1000 | latest | 0.331253 | 0.106188 | 0.225066 | 18.781318 | 0.650649 | Anchor not enough for long unfreeze |
| HCS-RVQ frozen 1000 | 500 | step_500 | 0.316531 | 0.091466 | 0.225066 | 21.150005 | 0.750917 | Reproduces 500-step result |
| HCG-RVQ-H frozen 1000 | 500 | step_500 | 0.317023 | 0.091957 | 0.225066 | 21.236728 | 0.751724 | Reproduces small HCG-H gain |
| HCS-RVQ frozen 1000 | 1000 | latest | 0.321632 | 0.096566 | 0.225066 | 21.041694 | 0.738647 | Generalization degrades |
| HCG-RVQ-H frozen 1000 | 1000 | latest | 0.322877 | 0.097811 | 0.225066 | 20.833721 | 0.735942 | Generalization degrades more |

## Rate Accounting

The fixed hyperlatent rate is currently dominant and constant at about `0.225066 bpp`, so most method differences are visible through `bpp_y` and quality.

Global RVQ with `group_size=64`, `num_stages=1`, and `codebook_size=128` has a fixed latent index cost of `0.136719 bpp`. HCS/HCG reduce effective `bpp_y` to about `0.0915 bpp` on Kodak because the learned index prior entropy-codes the RVQ indices.

This means the Global RVQ to HCS/HCG gain is not just distortion-side adaptation. It is also an entropy-modeling gain:

- Global frozen 500: `bpp_y=0.136719`
- HCS frozen 500: `bpp_y=0.091467`
- HCG-H frozen 500: `bpp_y=0.091937`

For publication, the paper should report `bpp_y`, `bpp_z`, and total bpp separately. Otherwise the contribution of index entropy can be confused with geometry.

## Quality And RD

At 500 steps, the ranking is:

1. HCG-RVQ-H: highest PSNR/MS-SSIM, slightly higher rate.
2. HCS-RVQ and no-transform: effectively tied.
3. Global RVQ frozen: lower quality and higher rate.
4. Scalar: lower rate but much lower quality.

HCG-H 500 vs HCS 500:

- Rate: `+0.000469 bpp`
- PSNR: `+0.100040 dB`
- MS-SSIM: `+0.001352`

HCG-H step_500 in the 1000-step run vs HCS step_500:

- Rate: `+0.000492 bpp`
- PSNR: `+0.086723 dB`
- MS-SSIM: `+0.000807`

The geometry signal is repeatable across two 500-step checkpoints, but its size is small. The no-transform control shows that the geometry-specific lift is only the gap from HCS/no-transform to HCG-H, not the full gap from Global RVQ to HCG-H.

## Stability

The most fragile component is adaptation of the hyper-decoder `h_s`. Normal `h_s` unfreezing collapses the latent/codebook alignment:

- HCG-H frozen 500: `21.252954 dB / MS-SSIM 0.752188`
- HCG-H staged `h_s` normal: `16.559856 dB / MS-SSIM 0.442196`
- HCG-H staged `h_s` slow: `18.108051 dB / MS-SSIM 0.654894`

The W&B commit-loss summaries explain the failure mode:

- HCG-H frozen 500: `0.14980`
- Staged `h_s` normal: `55.87057`
- Staged `h_s` slow: `4.68124`

This points to a moving-coordinate problem: changing `h_s` changes the conditional quantizer frame faster than the codebook/decoder can follow. Two drift-regularized variants sharpen the diagnosis:

- Always-on drift regularization controls the worst collapse but hurts the healthy warmup solution: best checkpoint is only `0.306870 bpp / 20.577316 dB / MS-SSIM 0.728218`.
- After-500 drift regularization preserves the warmup and reaches `0.316976 bpp / 21.285610 dB / MS-SSIM 0.751540` at step 500, but still degrades to `19.789570 dB` by step 1000.
- Step-500 frame anchoring slightly improves the mid-unfreeze checkpoint over after500 shrinkage (`RD 2.558084` vs `2.611545`) and lowers latent quantization MSE at step 750 (`0.732994` vs `0.900671`), but it still degrades badly by step 1000.

So the next `h_s` experiment should not just shrink conditioning magnitude, and the first anchor is not sufficient either. Treat `h_s` adaptation as a secondary research thread; the main paper path should remain validation-selected frozen/staged-step500 evidence until multi-seed confidence is available.

## Generalization And Checkpoint Selection

The 1000-step frozen extensions show that `checkpoint_latest` is currently unsafe:

- HCG-H latest vs step_500: `+0.005854 bpp`, `-0.403007 dB`, `-0.015782 MS-SSIM`
- HCS latest vs step_500: `+0.005101 bpp`, `-0.108311 dB`, `-0.012270 MS-SSIM`

This is likely pilot-data overfitting or decoder/codebook over-specialization under a small subset. A conference submission needs validation-selected checkpointing before running larger sweeps.

Minimum action:

- Hold out a validation split from the 4096-image pilot subset or use a fixed external validation set.
- Evaluate every saved checkpoint.
- Select by validation RD score, not by final step.
- Report Kodak only as test.

The checkpoint-sweep tool now exists at `tools/evaluate_checkpoints.py`. Running it on the frozen 1000-step HCS/HCG-H experiments and the no-transform control selects `checkpoint_step_500.pth.tar` by RD score, matching the manual observation.

## Larger Holdout And Feature Diagnostics

The OpenImages holdout was expanded from 256 to 1024 images for the 3-seed HCS/no-transform/HCG-H comparison. Consolidated outputs:

- `experiments/analysis/multiseed_hcg_geometry_val1024_checkpoint_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_val1024_checkpoint_means.csv`
- `experiments/analysis/multiseed_hcg_geometry_val1024_summary.md`
- `experiments/analysis/feature_seed3456_val1024_summary.md`
- `experiments/analysis/per_image_seed3456_hcs250_vs_hcgh500_val1024_summary.md`

Best-checkpoint mean over seeds:

| Method | RD mean | RD std | bpp mean | PSNR mean | MS-SSIM mean |
|---|---:|---:|---:|---:|---:|
| HCS-RVQ | 2.911486 | 0.026273 | 0.311889 | 20.3094 | 0.749730 |
| NoTransform | 2.911905 | 0.025550 | 0.311898 | 20.3081 | 0.749830 |
| HCG-RVQ-H | 2.892321 | 0.027518 | 0.310686 | 20.3878 | 0.747841 |

Seed3456 remains the critical counterexample:

- HCS step250: `RD=2.907870`, `bpp_y=0.087921`, `PSNR=20.3188`, `MS-SSIM=0.748333`.
- NoTransform step250: `RD=2.911528`, `bpp_y=0.087945`, `PSNR=20.3089`, `MS-SSIM=0.748484`.
- HCG-H step500: `RD=2.931155`, `bpp_y=0.085704`, `PSNR=20.2559`, `MS-SSIM=0.742348`.

Feature diagnostics for seed3456 show that HCG-H has nontrivial learned geometry (`s_q_mean≈0.45`, Householder delta RMS `0.101297`) and recovers from its weak step250 checkpoint, but it still loses perceptual/reconstruction quality to the simpler step250 checkpoints.

A new per-image comparison tool, `tools/per_image_compare.py`, compared seed3456 HCS step250 against HCG-H step500. HCG-H improves RD on `428/1024` images and worsens it on `596/1024`; mean deltas are `+0.023284 RD`, `-0.002217 bpp`, `-0.062853 dB`, and `-0.005984 MS-SSIM` for HCG-H minus HCS. This means the geometry is conditionally useful rather than globally collapsed.

The follow-up visualization tool, `tools/visualize_comparison_cases.py`, saved top loss/gain panels under `experiments/analysis/visual_seed3456_hcs250_vs_hcgh500_val1024/`. The selected largest HCG-H losses average `+1.665789 RD`, `-0.9148 dB`, and `-0.041555 MS-SSIM`; the largest HCG-H gains average `-1.383967 RD` and `+1.8094 dB`, but still `-0.030552 MS-SSIM`. Decoder activation-difference magnitudes are comparable between selected losses and gains, so a global decoder-drift magnitude threshold is unlikely to separate failures. The next method action is to test geometry-strength gating or regularization that can reduce Householder impact where it damages structure.

A reduced-strength diagnostic (`householder_strength=0.25`) was added and evaluated on the same seed3456 val1024 slice. It selects step250 with `RD=2.934233`, `bpp_y=0.088366`, `PSNR=20.2480`, and `MS-SSIM=0.747560`. This restores most of the MS-SSIM loss relative to full HCG-H step500 (`0.747560` vs `0.742348`) and halves Householder delta RMS (`0.054287` vs `0.101297`), but it still trails HCS step250 in mean RD (`2.934233` vs `2.907870`) and wins only `382/1024` images against HCS. Against full HCG-H step500 it wins `531/1024` images and improves MS-SSIM by `+0.005212`, while mean RD is `+0.003072` worse. The result argues against a fixed global strength as the final method and favors learned/per-location geometry gating or a structure-aware regularizer.

The learned gate0.25 diagnostic was then evaluated on the same seed3456 val1024 slice. It selects step250 with `RD=2.903028`, `bpp_y=0.087848`, `PSNR=20.2335`, and `MS-SSIM=0.751104`, beating HCS step250 in RD and MS-SSIM. Feature diagnostics show similar average geometry magnitude to fixed strength0.25 (`H_delta_rms=0.054329`, mean strength `0.254163`), but with modest learned variation (`std=0.007656`, min/max about `0.243/0.279`). Per-image comparison against HCS is still mixed: gate improves mean RD by `-0.004843` and MS-SSIM by `+0.002771`, but wins only `425/1024` images. Against fixed strength0.25 and full HCG-H it improves mean RD by `-0.031206` and `-0.028127`, respectively. This motivated a multi-seed check before turning the gate into the main paper-track variant.

The gate0.25 diagnostic was then repeated on seed1234 and seed2345 with the same frozen 500-step protocol and validation-selected checkpointing. The 3-seed mean now favors gate0.25: `RD=2.859852`, beating HCS by `-0.051634`, no-transform by `-0.052053`, and full HCG-H by `-0.032469`. Per-image RD wins are `1709/3072`, with strong seed1234/2345 gains and a weaker but still mean-positive seed3456 result. Feature diagnostics are stable across seeds: mean gate strength stays near `0.256`, strength std near `0.008`, index empirical bpp near `0.1326`, RVQ perplexity near `61/128`, and dead-code ratio near `0.197`. This upgrades learned geometry gating from a single-seed rescue to the current main HCG geometry variant, while the seed3456 tail failures still require targeted analysis.

The first tail analysis shows gate improvements concentrate more on high-HCS-RD images, while seed3456 low-HCS-RD images remain fragile (`+0.137167` mean RD in Q1, `52/256` wins on val1024; `+0.130779` on val4096). Top-loss overlap is low across seeds, so the next regularizer should target geometry reliability rather than a fixed image subset. A first `s_q` risk-aware gate on seed3456 val4096 reduces the mean regression and improves image-level win rate, but it reverses the tail by improving Q1/Q2 and hurting Q4. Per-image features make the failure concrete: `s_q`/risk multiplier are lower on harder images, and Q4 latent quantization MSE more than doubles. A post-hoc inverse-risk control is much worse overall, so geometry-strength schedules must be trained with the quantizer/decoder rather than swapped after the fact. The trained calibrated inverse/detached risk gate fixes this specific issue on seed3456: Q4 returns to `-0.095130` RD and the seed mean beats HCS by `-0.015211`, but Q1 remains positive at `+0.096617`.

## Novelty Attribution

The ablation ladder should be framed as:

1. Scalar Gaussian latent baseline.
2. Global RVQ with scalar-initialized codebooks.
3. HCS-RVQ: hyper-conditioned shift/scale, plus index entropy.
4. HCG-RVQ-H no-transform: parameter/control check.
5. HCG-RVQ-H: Householder local geometry.

Current attribution:

- Strongly supported: scalar-initialized conditional RVQ can be trained stably with frozen encoder/hyperprior.
- Strongly supported: HCS/index entropy improves rate-quality over Global RVQ.
- Strongly supported as current pilot: learned Householder geometry gating improves the 3-seed mean over HCS, no-transform, and full HCG-H under validation-selected checkpoints. The seed3456 tail failure is now analyzed in RD and intermediate-feature space. Calibrated inverse/detached risk proves the failure is controllable, but its 3-seed mean is weaker than old gate0.25, so it is a diagnostic branch rather than the main variant.
- Not supported yet: unfreezing `h_s` improves performance.

## Planned Comparisons And Final Training

Planned evidence should be layered rather than jumping straight to all external baselines:

1. Internal attribution: scalar MeanScaleHyperprior, Global RVQ, HCS-RVQ, entropy-only/HVQ-CGIC-like index-prior control, HCG-H no-transform, HCG-H, and geometry-strength/stability variants.
2. VQ/GIC-family comparisons from the prompt: RDVQ, HVQ-CGIC, Adaptive LVQ, ProGIC, PQ-MIM/LVQ-VAE, plus FSQ/RFSQ/LFQ-style stability references if collapse remains a central issue.
3. Strong LIC backbones: CompressAI MeanScaleHyperprior and joint autoregressive hyperprior, Cheng-style attention, ELIC/MLIC/MLIC++, TCM, MambaIC, HPCM, and DCAE where official implementations or reliable reported numbers are available.
4. Perceptual/generative baselines only if the final claim includes perceptual or generative compression behavior.

For the final paper, HCG-RVQ itself should be trained under a full, reproducible schedule across multiple lambdas. The main recipe should remain stable scalar-compatible initialization plus warmup/checkpoint selection unless a true scratch schedule becomes stable. Full scratch training is useful as an ablation and sanity check, but the current evidence says it should not be the main protocol yet.

## Paper Readiness

Ready to write as pilot evidence:

- Motivation and method.
- Implementation details.
- Global RVQ vs HCS vs HCG-H ablation under the frozen schedule.
- Failure analysis of naive `h_s` unfreezing.
- No-transform, fixed-strength, learned-gate, first risk-aware controls, and calibrated inverse/detached risk as pilot evidence, including the 3-seed gate0.25 validation, seed3456 tail diagnosis, and 3-seed calibrated-risk repeat.

Not ready for final submission claims:

- Robust multi-seed/larger-validation evidence that removes or explains the current seed sensitivity.
- Multi-rate RD curves.
- Strong LIC and VQ-LIC baselines.
- Validation-selected checkpoint protocol on larger runs and reporting datasets.
- Larger-validation and multi-rate confirmation for gate0.25 after the seed3456 reliability schedule is fixed or convincingly explained.

## Recommended Next Actions

1. Keep warmup and stable scalar-checkpoint initialization as the default protocol; direct VQ insertion and late checkpoints have repeatedly shown unstable latent/codebook alignment.
2. Keep validation-selected checkpointing mandatory. Use OpenImages holdout for model selection and reserve Kodak/CLIC-style sets for reporting.
3. Treat HCG-RVQ-H gate0.25 as the current main geometry variant: it is still the strongest 3-seed pilot. The first `s_q` risk-aware gate and the calibrated inverse/detached gate are diagnostic reliability controls, not the main variant yet.
4. Do not prioritize naive `h_s` unfreezing for the main paper path. Keep it as a diagnosis track until a stricter frame-anchoring or isolated `h_s`-only release is designed.
5. Before multi-rate curves, test a selective/milder reliability control that does not globally shrink geometry: candidates include a higher risk floor, annealed risk after warmup, a gate residual on top of old gate0.25, or validation-selected fallback between old gate and risk gate. Then rerun the 3-seed val4096 gate comparison and only promote it if it beats old gate0.25, not merely HCS.
6. Add official-baseline comparisons using official repositories where practical, or cite official numbers when reproduction cost is too high.
7. For paper tables, always report total bpp, `bpp_y`, `bpp_z`, PSNR, MS-SSIM, selected checkpoint step, codebook dead-code/perplexity, and feature diagnostics.


## Retrospective Checkpoint Sweep

A retrospective sweep was run for the older experiments that previously had only partial or latest-checkpoint analysis. The consolidated outputs are:

- `experiments/analysis/checkpoint_summary.csv`
- `experiments/analysis/checkpoint_best_summary.md`

Best checkpoints by RD score on Kodak:

| Experiment | Step | RD score | bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|
| Scalar | 500 | 3.159538 | 0.283923 | 19.152838 | 0.625586 |
| Global RVQ direct norm | 500 | 5.537114 | 0.356681 | 16.706942 | 0.604188 |
| Global RVQ g32-l2-k256 | 500 | 5.395954 | 0.872583 | 17.240326 | 0.547151 |
| Global RVQ frozen | 500 | 2.284964 | 0.361784 | 21.047032 | 0.723126 |
| HCS-RVQ frozen | 500 | 2.197908 | 0.316533 | 21.152914 | 0.750836 |
| HCG-RVQ-H no-transform | 500 | 2.197728 | 0.316533 | 21.153418 | 0.750906 |
| HCG-RVQ-H frozen | 500/latest | 2.164484 | 0.317002 | 21.252953 | 0.752188 |
| HCG-RVQ-H staged | 500 | 2.167059 | 0.317054 | 21.255815 | 0.750419 |
| HCG-RVQ-H staged slow | 500 | 2.208772 | 0.317124 | 21.159279 | 0.742786 |
| HCG-RVQ-H staged slow + drift reg | 500 | 2.436334 | 0.306870 | 20.577316 | 0.728218 |
| HCG-RVQ-H staged slow + after500 reg | 500 | 2.154112 | 0.316976 | 21.285610 | 0.751540 |
| HCG-RVQ-H staged h_s anchor after500 | 750 | 2.558084 | 0.325913 | 20.428828 | 0.725705 |
| HCS-RVQ frozen 1000 | 500 | 2.198715 | 0.316531 | 21.150005 | 0.750917 |
| HCG-RVQ-H frozen 1000 | 500 | 2.170041 | 0.317023 | 21.236730 | 0.751724 |

The key correction is that staged `h_s` unfreeze should be judged at multiple checkpoints. Both normal and slow unfreeze are healthy at step 500 and degrade after `h_s` is released. Normal unfreeze drops from `21.255815 dB` at step 500 to `16.322948 dB` at step 750. Slow unfreeze drops more gradually, from `21.159279 dB` to `20.491187 dB` to `18.108026 dB` across steps 500/750/1000. The after500-regularized run gives the best step-500 pilot point (`21.285610 dB`) but still falls to `20.281686 dB` at step 750 and `19.789570 dB` at step 1000. The anchor-after500 run improves the step750 RD score to `2.558084` but then falls to `18.781318 dB` at step 1000. This is useful failure analysis, not a solved adaptation recipe.

## Intermediate Feature Analysis

A new diagnostic tool, `tools/feature_distribution_analysis.py`, was added and run on representative checkpoints. The consolidated outputs are:

- `experiments/analysis/feature_summary.csv`
- `experiments/analysis/feature_summary.md`

Selected feature diagnostics:

| Experiment | bpp | PSNR | y_error_rms | latent quant MSE | index empirical bpp | Householder delta RMS |
|---|---:|---:|---:|---:|---:|---:|
| Scalar 500 | 0.283923 | 19.152838 | 0.107369 | - | - | - |
| Global direct 500 | 0.356681 | 16.706942 | 14.360433 | 4588.053716 | 0.094593 | - |
| Global frozen 500 | 0.361784 | 21.047032 | 0.105462 | 0.378932 | 0.108479 | - |
| HCS 500 | 0.316533 | 21.152915 | 0.102659 | 0.068028 | 0.129182 | - |
| no-transform 500 | 0.316533 | 21.153418 | 0.102662 | 0.068025 | 0.129181 | - |
| HCG-H 500 | 0.317002 | 21.252954 | 0.103787 | 0.072757 | 0.128864 | 0.072190 |
| staged 750 | 0.320266 | 16.322948 | 0.397742 | 43.273949 | 0.093756 | 2.268818 |
| staged slow 1000 | 0.330910 | 18.108026 | 0.115122 | 4.066104 | 0.126805 | 0.799914 |
| staged slow reg 1000 | 0.319786 | 19.583592 | 0.138047 | 4.704360 | 0.125224 | 0.669120 |
| staged slow reg after500 500 | 0.316976 | 21.285609 | 0.103736 | 0.073027 | 0.128881 | 0.072146 |
| staged slow reg after500 1000 | 0.329011 | 19.789570 | 0.119411 | 4.289382 | 0.126473 | 0.709898 |
| staged h_s anchor after500 750 | 0.325913 | 20.428828 | 0.110665 | 0.732994 | 0.130551 | 0.210435 |
| staged h_s anchor after500 1000 | 0.331253 | 18.781318 | 0.115586 | 3.682138 | 0.126573 | 0.774070 |
| HCG-H 1000 latest | 0.322877 | 20.833721 | 0.106234 | 0.363311 | 0.128382 | 0.178468 |

Interpretation by viewpoint:

- Latent stability: direct Global RVQ fails because the encoder latent scale explodes (`y_std=14.439488`) and RVQ cannot track it. Freezing `g_a/h_a/h_s` keeps `y_std=0.170745` and makes RVQ trainable.
- Quantization quality: HCS/no-transform/HCG-H reduce latent quant MSE from Global frozen `0.378932` to roughly `0.068-0.073`, explaining the jump over Global RVQ.
- Entropy: HCS/HCG have higher marginal empirical index entropy than Global frozen, but lower model-coded `bpp_y` because the index prior is conditioned on hyper-features. This supports reporting conditional index coding as a real contribution.
- Geometry: HCG-H has slightly worse latent MSE than HCS/no-transform but better image PSNR. The likely mechanism is not lower latent error magnitude; it is more useful error direction after decoder adaptation.
- Collapse: after `h_s` unfreeze, `hyper_features_std`, `householder_delta_rms`, latent quant MSE, and dead-code ratio all increase. Normal unfreeze at step 750 has `hyper_features_std=3.395880`, `householder_delta_rms=2.268818`, `rvq_latent_quant_mse=43.273949`, and `dead_code_ratio=0.403320`.
- Regularization: simple magnitude penalties reduce the most extreme transform explosion but do not preserve RD. The after500 regularized run moves from `rvq_latent_quant_mse=0.073027` / `householder_delta_rms=0.072146` at step 500 to `4.289382` / `0.709898` at step 1000.
- Anchoring: the first step500-frame anchor reduces step750 quantization MSE versus after500 shrinkage (`0.732994` vs `0.900671`) and gives a slightly better RD score, but by step1000 it still has `rvq_latent_quant_mse=3.682138` and `householder_delta_rms=0.774070`.

Paper implication:

The current evidence supports the HCS/index-prior mechanism cleanly. The Householder mechanism is plausible and average-positive over three seeds, but it needs a controlled stability story: more robust seed/validation behavior, multi-rate curves, and a more careful `h_s` adaptation design. Naive magnitude shrinkage and the first anchor are analysis tools, not yet final method components.


## Multi-Seed Validation-Holdout Geometry Check

A true holdout was added from the OpenImages training directory by evaluating images after the 4096-image pilot subset (`start_index=4096`, `max_images=256`, `patch_size=256`). Kodak remains a test-style split. The consolidated outputs are:

- `experiments/analysis/multiseed_hcg_geometry_checkpoint_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_checkpoint_means.csv`
- `experiments/analysis/multiseed_hcg_geometry_feature_summary.csv`
- `experiments/analysis/multiseed_hcg_geometry_summary.md`

Mean validation-best results over seeds 1234/2345/3456:

| Split | Method | RD mean+-std | bpp mean+-std | PSNR mean+-std | MS-SSIM mean+-std | Selected steps |
|---|---|---:|---:|---:|---:|---|
| OpenImages holdout | HCS-RVQ | 3.040305+-0.045972 | 0.311772+-0.001287 | 20.117441+-0.103738 | 0.749681+-0.001022 | 500/250/250 |
| OpenImages holdout | NoTransform | 3.040674+-0.044819 | 0.311781+-0.001293 | 20.116747+-0.102200 | 0.749798+-0.000972 | 500/250/250 |
| OpenImages holdout | HCG-RVQ-H | 3.015330+-0.034915 | 0.311474+-0.001343 | 20.200736+-0.123357 | 0.749007+-0.001223 | 500/500/250 |
| Kodak | HCS-RVQ | 2.206217+-0.030258 | 0.318314+-0.001588 | 21.141664+-0.092295 | 0.749433+-0.001312 | 500/250/250 |
| Kodak | NoTransform | 2.189912+-0.013770 | 0.317552+-0.001785 | 21.210728+-0.050294 | 0.750700+-0.001793 | 500/500/250 |
| Kodak | HCG-RVQ-H | 2.162222+-0.018052 | 0.317176+-0.000296 | 21.264001+-0.071777 | 0.749293+-0.005310 | 500/500/500 |

Feature diagnostics reinforce two points. First, HCS/no-transform are nearly identical, so the large Global-RVQ-to-HCS gain should be attributed to hyper-conditioned shift/scale plus index entropy. Second, HCG-H sometimes improves image RD while not reducing latent quantization error; on seed 1234 it has higher latent MSE than HCS/no-transform but better PSNR. This supports the idea that geometry changes the useful direction of quantization error rather than simply minimizing latent MSE.

The critical caveat is seed sensitivity. HCG-H wins OpenImages holdout for seeds 1234 and 2345, but loses for seed 3456. That makes the current claim publishable only as a careful pilot/ablation unless more seeds, stronger regularization, or a larger validation subset make the geometry gain more stable.

## Current Submission Stance

The project is moving in the right direction, but the safest international-conference story today is not "HCG-H is already SOTA." It is:

"Hyperprior-conditioned RVQ substantially improves a scalar-initialized RVQ latent codec, and a geometry-conditioned Householder extension gives a small but repeatable pilot gain. We identify stability and generalization bottlenecks and provide matched controls for attribution."

To make the stronger claim that HCG geometry itself is a reliable contribution, the immediate requirement is no longer simply running multi-seed evidence; it is making the average-positive geometry gain robust to seed and validation choice.

## Start0 Selector Transfer Recheck

A current-code start0 recheck was added after discovering that older `reeval_current` checkpoint-summary CSVs were stale. The trusted start0 numbers now come from regenerated per-image diagnostics and independent checkpoint reevaluation on CUDA device 0.

Transfer checkpoint plan, matching the paper-facing holdout checkpoint choices, gives:

| policy | mean RD | delta vs HCS | vs old gate0.25 |
|---|---:|---:|---:|
| HCS | 2.931295 | +0.000000 | n/a |
| old gate0.25 | 2.910176 | -0.021119 | +0.000000 |
| min090 risk | 2.915664 | -0.015631 | +0.005488 |
| oracle old/min090 | 2.873311 | -0.057984 | -0.036865 |
| fixed holdout raw-gate selector | 2.898147 | -0.033148 | -0.012028 |
| start0 CV deployable selector | 2.895861 | -0.035434 | -0.014315 |

This is encouraging but not a free pass. The fixed holdout selector does transfer and improves old gate0.25 on start0, but the seed3456 transfer checkpoint is bad on this slice: old step500 is `+0.068447` RD worse than HCS and min090 step500 is `+0.023008` worse. The selector chooses the less bad risk checkpoint, so it improves old-vs-risk selection but cannot replace validation-selected checkpointing.

The slice-best diagnostic, where seed3456 uses step250 instead of the holdout-selected step500, changes the picture: old gate0.25 improves to `2.891076` mean RD and the fixed holdout threshold is slightly worse than old by `+0.003118` RD, while start0 CV gives a small improvement (`-0.003381` vs old). This separates two effects: reliability selection is real, but checkpoint selection is a first-order variable.

Paper implication: the main direction should be a constrained old/min090 reliability selector under a rigorous validation-selected checkpoint protocol. Do not claim that one global raw-gate threshold is universal, and do not promote the free learned reliability head. The robust claim is that frozen intermediate evidence can identify when geometry should be weakened, and that this can improve a geometry-conditioned RVQ when checkpoint selection is controlled.

## Start0 Artifact Consistency Audit

A provenance audit was added after the start0 recheck showed that legacy `*_reeval_current.csv` checkpoint-summary files can disagree with current-code reruns. The audit compares those legacy summary rows against regenerated per-image/debug references and writes `experiments/analysis/start0_artifact_consistency_audit.{md,json}`.

The result is decisive: all 9 checked legacy start0 checkpoint-summary rows are excluded. Mean legacy-vs-reference gaps are `-0.663705` RD for HCS, `-0.655822` RD for old gate0.25, and `-0.661289` RD for inverse/detached min090 risk. The available independent debug CSV references match their per-image JSON summaries within `1e-4`, and all trusted JSON references contain finite numeric values.

This locks the interpretation from the previous section: start0 claims should use regenerated per-image/debug artifacts and the selector summaries, not the stale `pilot_*_openimages_val4096_reeval_current.csv` checkpoint summaries. The old/min090 selector remains the main constrained reliability direction, but it must be reported with explicit checkpoint selection and artifact provenance.

## Validation-Calibrated Reporting Selector Protocol

A reporting-protocol selector analysis was added to separate threshold calibration from reporting evaluation. The policy is selected once on the OpenImages `start_index=4096` current holdout and applied unchanged to OpenImages `start_index=0` current recheck.

The calibrated deployable policy is `old_raw_gate_mean >= 0.260788`. On the calibration split it improves old gate0.25 by `-0.019336` RD and closes `0.591113` of the old/min090 oracle gap. On the start0 transfer reporting protocol it still improves old gate0.25 by `-0.012028` RD and closes `0.326284` of the oracle gap. This supports the central selector claim without using reporting labels to choose the threshold.

The caveat is equally important: under the start0 slice-best checkpoint protocol, the same calibrated policy is `+0.003118` RD worse than old gate0.25. Same-split analysis there prefers `risk_y_error_rms <= 0.145200`, not the validation raw-gate threshold. This confirms that reliability selection is real, but checkpoint choice is still a first-order variable. The paper-safe wording is therefore: validation-calibrated reliability evidence can improve HCG geometry when checkpoint selection is controlled; however, the current old/min090 switch is a multi-checkpoint diagnostic, not yet a unified single-codec method. A universal threshold is not established.

## Selector Claim-Readiness / Single-Codec Audit

A claim-readiness audit now separates the selector evidence from the final method claim. The old gate0.25 and min090 variants share the HCG-RVQ-H architecture family and initialization, but they do not share the same gate policy or checkpoint. Therefore, the calibrated old/min090 selector is best treated as a multi-checkpoint diagnostic showing reliability-control headroom, not as the final single-codec method.

This does not weaken the mechanism finding. On the reporting transfer split, the calibrated selector improves old gate0.25 by `-0.012028` RD and recovers `0.326284` of the old/min090 oracle gap. It does, however, change the paper plan: the final main-row method should be a unified single-checkpoint reliability controller. Until then, old gate0.25 remains the safest single-model HCG geometry result, min090 is a useful deterministic ablation, and the selector is evidence for the next controller design.

## Related-Work Recheck and Current Research Health

The 2026-05-30 literature/code recheck did not weaken the HCG-RVQ thesis. It clarified the claim boundary.

The strongest related methods now define three comparison axes:

- HVQ-CGIC-like entropy-only VQ: tests whether hyperprior-conditioned index probabilities explain the gain without geometry.
- RDVQ-like RD-aware VQ training: tests whether differentiable VQ rate optimization can match or exceed HCG without local geometry.
- Adaptive-LVQ / strong-backbone LIC: tests whether the HCG strategy improves an already capable quantizer/backbone over its own baseline.

The current evidence is therefore good and promising, but should be reported honestly:

- Solid: HCS/index-prior improves the scalar-initialized RVQ path and gives a clean mechanistic baseline.
- Promising: HCG-H geometry improves average pilot RD over multiple seeds/splits, but remains sensitive to checkpoint and image regime.
- Strong diagnostic: old/min090 selection shows reliability-control headroom and transfers under a calibration/reporting protocol.
- Not final yet: the selector is currently multi-checkpoint, so the final paper method should become a single-checkpoint controller.

Updated next action:

1. Build/evaluate a unified reliability-controlled HCG-RVQ checkpoint.
2. Compare it against HCS, old gate0.25, min090 risk, and the old/min090 oracle/selector headroom.
3. Add entropy-only HVQ-style and RDVQ-style comparisons after the controller is stable.
4. Only then move to DCAE/HPCM/MLIC/MambaIC-style strong-backbone integration.

This keeps the project aligned with `prompt.txt`: prove that hyperprior-conditioned quantizer geometry is useful, then show that the strategy can also improve stronger codecs.


## Posthoc Single-Checkpoint Controller Audit

The posthoc old-weights/min090 test is now resolved as a negative result. The superficially good posthoc deltas were caused by an HCS checkpoint mismatch in the comparison CSVs; the HCS side there was around `+0.63` to `+0.68` RD worse than the trusted current-holdout HCS artifacts. After matching the posthoc feature RD to trusted HCS rows by image path, posthoc min090-on-old-weights is `+0.651546` RD worse than HCS and `+0.681709` worse than old gate0.25 on the 3-seed mean.

This is an important design constraint. Reliability control cannot simply be applied after training to a geometry-conditioned checkpoint. The decoder/codebook appear coupled to the gate policy used during training, so changing Householder strength posthoc creates a large mismatch even when the mean risk multiplier is only around `0.92` to `0.95`.

The main paper path should therefore remain:

1. use old gate0.25 as the safest current single-checkpoint HCG-H result;
2. use trained min090 and old/min090 selector/oracle as reliability-headroom diagnostics;
3. implement a unified reliability controller that is present during training and evaluated under the trusted checkpoint-selection protocol;
4. keep absolute RD, path-matched per-image deltas, checkpoint provenance, and feature distributions as mandatory reporting checks.

Artifact: `experiments/analysis/posthoc_single_checkpoint_controller_val4096_holdout4096_current.md`.


## Old-Checkpoint Risk Fine-Tune Probe

A 250-step fine-tune from the old gate0.25 seed3456 checkpoint into the min090 inverse/detached risk rule did not rescue the posthoc mismatch. It worsened the trusted holdout4096 seed3456 delta from posthoc `+0.742438` to `+0.800023` RD vs HCS, while the separately trained min090 checkpoint remains `-0.035673` RD vs HCS.

The feature distribution explains why this is not just a decoder needing a few updates. During the fine-tune, `s_q` dropped from `0.436136` to `0.347017`, Householder strength rose from `0.281135` to `0.311987`, and latent quantization MSE jumped from `0.164911` to `0.370800`. The adaptation moves into a worse scale/geometry regime.

Design implication: stable-checkpoint warm-start is not automatically safe for VQ geometry control. The next unified controller should use a conservative schedule or regularization, not a direct risk-gate fine-tune from the already fragile old seed3456 checkpoint.

Artifact: `experiments/analysis/ftold500_risk_min090_seed3456_val4096_holdout4096_current.md`.

## Conservative Risk Floor min095 Audit

The trained `householder_gate_risk_min=0.95` variant is a clear negative result. On the trusted OpenImages holdout4096 path-matched protocol, the 3-seed mean is `2.972039` RD, which is `+0.708334` worse than HCS, `+0.738497` worse than old gate0.25, and `+0.741931` worse than trained min090. The failure is not seed-local: seed1234/2345/3456 are all around `+0.70` RD worse than HCS.

The intermediate features are the important part. The risk multiplier is conservative at about `0.974`, so this is not a failure caused by excessive multiplicative suppression. Instead, the learned raw gate stays high at about `0.297`, effective Householder strength rises to about `0.289`, and latent quantization MSE rises to about `0.154`. The Q4 hard-image quartile is especially bad, with min095-HCS deltas from `+1.017285` to `+1.110146` RD.

This changes the next design target. Plain floor tuning is not the right stabilization knob. The controller needs a direct constraint on geometry itself: lower LR, explicit Householder delta/strength regularization, or a staged schedule that pins scale and geometry near the validated old gate0.25 regime before allowing adaptation. Until that exists, old gate0.25 remains the safest single-checkpoint HCG-H result, while min090 and the old/min090 selector remain reliability-headroom diagnostics rather than the final single-codec method.

Artifact: `experiments/analysis/risk_floor_min095_val4096_holdout4096_current.md`.

## Direct Householder Delta Regularization Probe

The `rho_householder_delta=0.10` seed3456 probe is now a strong negative result, and it is useful precisely because it separates two hypotheses.

Hypothesis A was that the failing reliability controllers simply used too much geometry. If true, shrinking the Householder displacement should improve the fragile seed3456 behavior. This did not happen. The direct delta penalty reduced Householder delta RMS from the useful old/min090 range of about `0.099` to `0.038`, and reduced effective strength to `0.256979`, but RD worsened to `3.079296`. That is `+0.815950` worse than HCS, `+0.793592` worse than old gate0.25, and `+0.851623` worse than trained min090 on the same path-matched holdout.

Hypothesis B is therefore more consistent with the data: HCG geometry is useful only within a compatible decoder/codebook/scale regime. Collapsing the geometry toward zero breaks that regime instead of making it safer. The intermediate features support this interpretation: delta_reg010 has much smaller delta RMS, but latent quantization MSE rises to `0.206089`, worse than old gate0.25 (`0.120477`), trained min090 (`0.119785`), and min095 (`0.162526`). Dead-code ratio also rises to `0.216028`.

The HCS-difficulty quartiles make the failure mode especially clear. On the hardest Q4 images, old gate0.25 is `+0.101288` RD vs HCS, min090 is only `+0.030999`, but delta_reg010 is `+1.117536`. The regularizer does not selectively repair the tail; it severely damages it.

Design implication: the next single-checkpoint controller should not minimize Householder displacement. It should keep geometry near a validated operating range, for example with a target/anchor around the old/min090 `householder_delta_rms` or strength regime, or with a much weaker/staged penalty that preserves latent quantization MSE. This keeps the paper story aligned with the prompt: the claim is not that geometry should disappear, but that hyperprior-conditioned geometry must be generated under reliable local control.

Artifact: `experiments/analysis/delta_reg010_seed3456_val4096_holdout4096_current.md`.

## Householder Delta Target Probe

The target-loss probe answers the question left open by the direct delta shrinkage experiment. It was possible that E048 failed only because the regularizer collapsed the transform too far. To test that, `rho_householder_delta_target=5.0` with `householder_delta_target=0.095` was added, explicitly aiming at the old/min090 displacement regime rather than zero.

The result is again negative, but more informative. The target objective did what it was asked to do: `householder_delta_rms` became `0.092294`, close to old gate0.25 (`0.099553`) and trained min090 (`0.099314`). Yet the RD score worsened to `3.332857`, which is `+1.069512` vs HCS, `+1.047153` vs old gate0.25, and `+1.105184` vs trained min090.

This means the failure is not only the magnitude of Householder displacement. The model can hit the desired delta RMS while moving the rest of the quantization state into a poor regime: `s_q` rises to `0.488443`, effective strength drops to `0.245312`, latent quantization MSE rises to `0.293593`, and dead-code ratio rises to `0.261705`. The hardest quartile is especially bad at `+1.502881` RD vs HCS.

Design implication: scalar geometry controls are insufficient. The next controller should anchor a fuller state, especially `log_s_q`/`u` or latent quantization error/codebook-utilization behavior, against a validated checkpoint. This is stronger evidence for a reliability-controlled HCG geometry story: the geometry is useful, but only when the hyperprior-generated conditioning, codebook use, and decoder-compatible latent scale remain jointly controlled.

Artifact: `experiments/analysis/delta_target095_rho5_seed3456_val4096_holdout4096_current.md`.

## Old-Conditioning Anchor Probe

The full old-frame conditioning anchor is now resolved as a negative result. This probe was designed after the scalar Householder delta experiments failed: instead of controlling only geometry magnitude, the risk-gated model was trained with an anchor to the validated old gate0.25 checkpoint conditioning state (`mu_q`, `log_s_q`, and `u`). The implementation added `train.anchor_config` so the anchor model can be built from the old gate0.25 config while the current model uses the min090 inverse/detached risk config.

Mechanically, the anchor did what it was meant to do. On seed1234 holdout4096, the best anchored checkpoint has `s_q=0.566673`, latent qMSE `0.059710`, index bpp `0.117619`, and dead-code ratio `0.035179`, all close to old gate0.25 (`s_q=0.569798`, qMSE `0.056989`, index bpp `0.117960`, dead-code `0.034916`) and much safer than the trained min090 risk checkpoint on qMSE.

The RD result is still bad: best anchored RD is `2.375139`, which is `+0.163664` vs HCS, `+0.182044` vs old gate0.25, and `+0.146727` vs trained min090 on the trusted seed1234 holdout4096 comparison. MS-SSIM also drops to `0.749647`, far below old gate0.25 (`0.779345`) and trained min090 (`0.778710`). The key feature clue is that effective Householder strength is suppressed to `0.231738` and delta RMS to `0.054844`, below the old gate0.25 operating point.

Design implication: preserving quantization error and codebook statistics is not sufficient. The HCG controller also has to preserve decoder-compatible geometry strength and perceptual structure. A full `u` anchor plus risk multiplier over-constrains the Householder geometry; it makes the quantizer look numerically safe while weakening the geometric transform that the decoder expects. The next controller should therefore avoid full direction anchoring and instead try a weaker/staged scale anchor, such as `log_s_q` only, combined with a light strength or delta floor/target that prevents useful HCG geometry from collapsing.

Artifact: `experiments/analysis/per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_anchor_old_u05_logs01_seed1234_step250_val4096_holdout4096_current.json`.

## Scale-Only Old-Anchor Probe

The scale-only old-anchor probe is now resolved as a negative result. After the full old-conditioning anchor suppressed useful geometry, the next test removed the `u` anchor and kept only `rho_anchor_log_s`, first at `0.10` and then at `1.00`. This was meant to preserve the quantizer scale while leaving the Householder direction free.

The result improves over the full anchor but is still not competitive. On seed1234 holdout4096, `log_s`-only 0.10 reaches RD `2.365288`, and `log_s`-only 1.00 reaches RD `2.356073`. The stronger scale anchor is only `0.009215` RD better than 0.10 and remains `+0.144598` worse than HCS, `+0.162978` worse than old gate0.25, and `+0.127661` worse than trained min090.

The feature distributions explain why. `rho_anchor_log_s=1.00` only moves `s_q` to `0.465048`, compared with `0.454945` at 0.10, `0.459787` for trained min090, and `0.569798` for old gate0.25. Meanwhile Householder strength (`0.279093`) and delta RMS (`0.099012`) stay in the min090-like regime, not the old gate0.25 regime. So scale-only anchoring does not actually recover the old operating point; it mostly adds an optimization constraint while leaving the risky geometry pattern intact.

The per-image analysis is the most important part. In the hardest HCS quartile, old gate0.25 improves RD by `-0.103524` vs HCS, trained min090 is roughly neutral at `-0.001850`, but `log_s`-only 1.00 worsens by `+0.204068` and is `+0.205917` worse than min090. Degradation correlates with Householder delta RMS (`r=+0.707003`), strength (`r=+0.543714`), and risk multiplier (`r=+0.533519`), while higher `s_q` is protective (`r=-0.522559`).

Design implication: neither full direction anchoring nor global scale-only anchoring is the right reliability controller. The final single-checkpoint method should be selective and local: in high-risk/high-delta regions it should bound Householder amplitude or delta, while preserving enough geometry to keep the old gate0.25 hard-tail benefit. The next probe should therefore combine a mild scale floor/anchor with a detached risk-conditioned strength or delta cap, not a global anchor on the whole latent space.

Artifact: `experiments/analysis/old_anchor_scale_only_seed1234_val4096_holdout4096_current.md`.



## Local Delta Control and Reproducibility Boundary

The local Householder delta control probe changed the immediate conclusion in two ways.

First, under the current code state, local cap080/rho1 is provisionally the best seed1234 row among the rechecked methods: RD 2.828428, compared with current HCS 2.889062, current old gate0.25 2.843577, and current min090 2.891932. This means the local reliability-control direction is not dead.

Second, and more importantly for a serious conference submission, the May29 trusted baseline CSVs are not reproducible with the current code: the same checkpoint paths now evaluate much worse. This makes mixed-protocol comparisons invalid. Until the code/evaluation state is pinned or restored, the paper-facing claim should not use local cap vs May29 HCS/old/min090 numbers.

The feature analysis says scalar geometry magnitude is not enough. band063+cap080 restores global delta RMS near old gate0.25 but still fails because effective strength drops, latent qMSE rises to 0.152749, and dead-code ratio rises to 0.234999. The next reliable controller must preserve scale, strength, local-delta tails, latent qMSE, and codebook usage together.

Artifact: experiments/analysis/local_delta_controls_current_code_consistency_seed1234_holdout4096.md.


## 3-Seed Local-Cap Result and Corrected Protocol Interpretation

The protocol audit corrected the earlier E052 concern. The trusted historical holdout rows are reproducible by direct path-aligned probes; the invalid artifacts are the recheck/localstats files, not the trusted HCS/old/min090 rows. This matters because it restores a clean comparison base for local cap080/rho1.

Under that corrected protocol, local cap080/rho1 becomes the strongest current unified-controller candidate on OpenImages holdout4096. Its 3-seed mean RD is `2.221143`, beating HCS by `-0.042562`, old gate0.25 by `-0.012398`, and trained min090 by `-0.008965`. It also has per-image wins over HCS/old/min090 of `6821/12288`, `6238/12288`, and `6913/12288`, respectively.

The mechanism is plausible. The best local-cap checkpoints keep `s_q` high (`0.569045`), reduce Householder strength (`0.232035`) and local delta (`0.022989` mean), and keep latent qMSE low (`0.059622`). Step500 in every seed is worse and is accompanied by lower `s_q` and much higher qMSE, so the checkpoint-selection result is not arbitrary.

The strongest evidence is the HCS-difficulty split. Local cap is worse on easy Q1 images (`+0.050885` vs HCS), but it is much better on the hard Q4 tail (`-0.136153` vs HCS, `-0.067987` vs old gate0.25, `-0.082493` vs min090). This turns the paper story from a generic gate sweep into a more specific claim: local reliability control of hyperprior-generated geometry protects hard images and stabilizes fragile seeds.

Remaining risk: the result is still one OpenImages holdout protocol. It is strong enough to promote local cap080/rho1 to the next paper-main candidate, but not final until Kodak/secondary split checks and a clean ablation table confirm the same pattern.

Artifact: `experiments/analysis/local_cap080_rho1_multiseed_holdout4096_trusted_protocol.md`.


## Kodak Transfer Check for Local Cap080/rho1

The OpenImages holdout4096 result for local cap080/rho1 is strong, but the Kodak secondary split adds an important caution. Using the same checkpoint selections as OpenImages, local cap080/rho1 reaches Kodak RD `2.214630`, which is `+0.008413` worse than HCS, `+0.028446` worse than old gate0.25, and `+0.019581` worse than min090. There were no nonfinite rows, and all runs used GPU 0 with exact-inverse direct probes.

This does not invalidate the OpenImages hard-tail result, but it changes the paper positioning. Local cap080/rho1 should be treated as the leading research direction, not yet the final single-codec method. The Kodak failure is concentrated in seed3456: seeds 1234 and 2345 improve HCS by `-0.024608` and `-0.031932`, while seed3456 worsens by `+0.081778`.

The checkpoint sweep explains why this is not a simple implementation failure. On Kodak, local step500 improves the 3-seed local mean from `2.214633` to `2.198032`, beating HCS by about `-0.008185`; however, it remains worse than old gate0.25 and min090, and seed3456 still worsens from step250 to step500. This is split/checkpoint sensitivity, not NaN or device instability.

Mechanistically, step250 is very conservative on Kodak (`s_q=0.580781`, strength `0.231294`, local delta mean `0.017240`, qMSE `0.041158`). Step500 lowers scale and increases geometry (`s_q` around `0.48`, strength around `0.256`, qMSE around `0.07`). That helps Kodak seeds 1234/2345 but hurts seed3456. The next useful method is therefore not simply more or less geometry globally; it should be selective/easy-safe, preserving the OpenImages Q4 benefit while avoiding Q1/Kodak over-suppression.

Paper implication: the safe claim is currently that hyperprior-conditioned local geometry is useful and that local reliability control can strongly improve hard-tail behavior, but the final controller still needs secondary-split stabilization before claiming robust SOTA-style dominance.

Artifacts: `experiments/analysis/local_cap080_rho1_multiseed_kodak_trusted_protocol.md` and `experiments/analysis/local_cap080_rho1_kodak_checkpoint_sweep.md`.



## Excess-Risk Local Cap: Stronger Fixed-Step Candidate

The excess-risk local cap080/rho1 probe is the first result that improves both the trusted OpenImages holdout4096 average and the Kodak secondary split under a simple fixed checkpoint rule. This matters for paper safety: the per-seed-best result is slightly better, but fixed step500 is cleaner to defend.

OpenImages holdout4096:

| method/checkpoint rule | mean RD | vs HCS | vs old gate0.25 | vs min090 | vs previous local best |
|---|---:|---:|---:|---:|---:|
| excess-risk step250 fixed | 2.223519 | -0.040186 | -0.010023 | -0.006589 | +0.002376 |
| excess-risk step500 fixed | 2.218792 | -0.044913 | -0.014750 | -0.011316 | -0.002351 |
| excess-risk per-seed best | 2.216082 | -0.047623 | -0.017459 | -0.014026 | -0.005061 |

Kodak:

| method/checkpoint rule | mean RD | vs HCS | vs old gate0.25 | vs min090 | vs previous local step500 |
|---|---:|---:|---:|---:|---:|
| excess-risk step250 fixed | 2.199571 | -0.006646 | +0.013387 | +0.004522 | +0.001539 |
| excess-risk step500 fixed | 2.132528 | -0.073689 | -0.053656 | -0.062521 | -0.065504 |

Mechanistic read:

- The previous local cap080/rho1 result was strong on OpenImages but did not transfer cleanly to Kodak. The excess-risk variant fixes that secondary-split failure in the fixed step500 checkpoint.
- Step500 enters a stronger-geometry regime than step250: on OpenImages, `s_q` drops from `0.568803` to `0.465916`, Householder strength rises from `0.233568` to `0.261636`, local delta mean rises from `0.025209` to `0.029470`, and qMSE rises from `0.059102` to `0.110920`. The important change is that this no longer corresponds to the old step500 collapse; RD improves instead.
- Kodak shows the same direction with milder qMSE: step500 reaches `s_q=0.485199`, strength `0.256944`, local delta mean `0.023213`, and qMSE `0.068965`, producing a large RD gain.
- OpenImages seed2345 still prefers step250 by `0.008128` RD, so the method is not fully solved. But the fixed step500 average is already stronger than HCS, old gate0.25, min090, and previous local cap080/rho1.

Paper implication:

The main claim can now move from “local reliability control is promising but split-sensitive” to “excess-risk local control yields a fixed-step HCG geometry variant that improves both the main OpenImages validation protocol and Kodak transfer.” This is materially stronger, while still honest: the next evidence needed is difficulty-quartile/per-image analysis and an additional secondary split or a held-out test protocol.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_holdout4096_checkpoint_sweep.md`
- `experiments/analysis/excessrisk090_local_cap080_rho1_kodak_checkpoint_sweep.md`



## Excess-Risk Tail Correction

The per-image tail analysis changes the interpretation of excess-risk fixed step500. The result remains positive as an average-RD and Kodak-transfer candidate, but it is not the same mechanism as previous local cap080/rho1 step250.

On OpenImages holdout4096, fixed step500 improves the 3-seed mean to `2.218792`, beating HCS by `-0.044913`, old gate0.25 by `-0.014750`, min090 by `-0.011316`, and previous local step250 by `-0.002351`. Its win rate is also higher: `0.658854` vs HCS and `0.554199` vs previous local step250.

The quartile split is the key nuance:

| quartile | previous local-HCS | excess500-HCS | excess500-prev local |
|---|---:|---:|---:|
| Q1 easy | +0.050885 | -0.054984 | -0.105868 |
| Q2 | -0.013253 | -0.059420 | -0.046167 |
| Q3 | -0.071727 | -0.056804 | +0.014923 |
| Q4 hard | -0.136153 | -0.008446 | +0.127707 |

So the story is bifurcated:

- Previous local cap080/rho1 step250 is the stronger evidence for hard-tail reliability control.
- Excess-risk local cap080/rho1 fixed step500 is the stronger fixed-checkpoint average and Kodak-transfer candidate.
- A final paper-main controller should try to combine these: keep the fixed step500 easy/Q2 gain while preserving the step250 Q4 hard-tail benefit.

Artifact: `experiments/analysis/excessrisk090_local_cap080_rho1_tail_holdout4096.md`.

## Selector Headroom From Complementary Regimes

The previous-local step250 and excess-risk fixed step500 results are complementary rather than redundant. The fixed step500 row is the stronger average/Kodak-transfer candidate, while the previous-local step250 row remains the stronger hard-tail reliability-control evidence.

A per-image headroom diagnostic on OpenImages holdout4096 confirms this. The oracle that chooses the better of previous-local step250 and excess-risk fixed step500 per image reaches RD `2.147093`, which is `-0.116613` vs HCS, `-0.074051` vs previous-local step250, and `-0.071699` vs excess-risk fixed step500.

More importantly, a simple deployability-shaped proxy already captures much of that headroom: choosing excess-risk fixed step500 when `rvq_householder_delta_rms <= 0.045256` and previous-local step250 otherwise reaches RD `2.170055`. This is `-0.093650` vs HCS, `-0.063486` vs old gate0.25, `-0.060053` vs min090, `-0.051088` vs previous-local step250, and `-0.048737` vs excess-risk fixed step500.

This should not be written as a final method result, because it switches between two separately trained checkpoints. It should instead guide the next single-model design. The target controller should learn or impose the same behavior inside one checkpoint: use the stronger step500 geometry in low-delta/easy regions, and cap or soften that geometry when local Householder delta RMS enters the high-risk regime where previous-local step250 is safer.

Paper implication: this gives a clear path for strengthening HCG-RVQ without leaving the main thesis. The thesis is still hyperprior-generated local quantizer geometry; the refinement is reliability-aware geometry amplitude control, validated by a measurable intermediate feature rather than by ad hoc checkpoint selection.

Artifact: `experiments/analysis/excessrisk090_prevlocal_selector_headroom_holdout4096.md`.


## Cap080-to-Cap060 Is a Negative Schedule Control

The cap080-to-cap060 schedule answers an important design question: simply tightening the local Householder cap after step250 is not the right way to combine the previous local hard-tail result with the excess-risk step500 average result.

It gives a small diagnostic improvement at step250 and a better per-seed-best ceiling, but fixed step500 is worse on OpenImages (`2.242705`) and worse on Kodak (`2.155794`) than the earlier excess-risk cap080 step500. The tail split explains why: step500 improves Q1/Q2, but Q4 becomes roughly neutral vs HCS (`+0.000706`) and much worse than the local step250 hard-tail result.

The lesson is that the failure is not only excessive local Householder delta. Step500 still enters a low-scale/high-qMSE operating regime, so a useful controller needs to preserve the latent quantization/codebook state as well as the geometry amplitude.

Artifact: `experiments/analysis/excessrisk090_local_cap080to060_rho1_tail_holdout4096.md`.

## Beta-Commit Guard Becomes the Main Fixed-Checkpoint Candidate

The beta-commit guard after step250 is now the strongest fixed-checkpoint HCG-RVQ candidate in the current protocol. It keeps the excess-risk local cap080/rho1 geometry control, but raises `beta_commit` from `0.01` to `0.05` after step250 to resist the latent quantization drift seen in previous step500 runs.

The 3-seed OpenImages result is a large jump: fixed step500 reaches RD `2.173688`, compared with HCS `2.263705`, old gate0.25 `2.233542`, min090 `2.230108`, previous local step250 `2.221143`, and earlier excess-risk step500 `2.218792`. This is not a per-seed checkpoint oracle; every seed selects the same fixed step500 rule.

Kodak transfer also strengthens rather than weakening: beta-commit guard step500 reaches `2.100549`, improving the earlier excess-risk step500 `2.132528` by `-0.031980` RD. Nonfinite rows are zero on both OpenImages and Kodak, with all GPU work pinned to physical GPU 0.

The tail result is especially useful for paper positioning:

| HCS quartile | beta step500-HCS | previous local-HCS | interpretation |
|---|---:|---:|---|
| Q1 easy | -0.075655 | +0.050885 | beta fixes easy-image damage |
| Q2 | -0.094492 | -0.013253 | beta dominates |
| Q3 | -0.105463 | -0.071727 | beta dominates |
| Q4 hard | -0.084461 | -0.136153 | beta improves HCS, but previous local is still the hard-tail specialist |

This changes the main paper strategy. The safest current claim is no longer only that local reliability control helps the hard tail. The stronger claim is that hyperprior-conditioned geometry plus a latent-commitment guard yields a fixed-step codec variant that improves average RD, transfers to Kodak, and retains positive hard-tail behavior. The previous local step250 row should remain as an ablation showing the mechanism of hard-tail control.

The feature distribution supports the interpretation. Beta step500 still uses the stronger geometry/low-scale regime (`s_q=0.463063`, `strength=0.263283`, `delta RMS=0.040208`), but keeps qMSE and codebook use better controlled than the failed schedules (`qMSE=0.106819`, `dead code=0.031892`, `perplexity=68.286368`). Degradation still correlates with delta RMS and qMSE, so the next refinement should tune `beta_commit` or add a validation-selected guard rather than blindly increasing geometry.

Artifact: `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.md`.

## Beta-Commit Strength Is Not Monotonic

The seed3456 boundary check shows that the beta-commit guard is not a simple "larger is better" knob. On the main OpenImages holdout4096 protocol, the after-step250 strengths `0.03`, `0.05`, and `0.07` give step500 RD values of `2.206961`, `2.156416`, and `2.197884`, respectively. The current E059 value `0.05` is therefore clearly best on the most fragile seed, beating `0.03` by `-0.050545` RD and `0.07` by `-0.041468` RD.

Kodak gives a useful but more cautious signal: on seed3456, `0.07` reaches RD `2.093526`, slightly better than `0.05` at `2.106310`. This suggests a possible transfer-tuned variant, but it is not enough to move the paper-main setting because it lacks full 3-seed OpenImages, tail, and checkpoint-selection evidence.

The feature means also warn against over-interpreting the boundary result from a single scalar. For seed3456 step500 on OpenImages, `0.03/0.05/0.07` have very similar operating statistics: `s_q=0.458535/0.456233/0.456295`, delta RMS `0.040277/0.040308/0.040199`, and qMSE `0.111946/0.112377/0.111984`. The large RD advantage of `0.05` is therefore not explained by a simple monotonic shift in average scale, average geometry, or average qMSE. A later full 3-seed check should include per-image tail and correlation analysis before treating the beta value itself as a mechanism.

The practical conclusion is that E059 `beta_commit=0.05` remains the main fixed-checkpoint candidate. It has completed 3-seed evidence, large OpenImages average gains, Kodak transfer, zero nonfinite rows, and positive hard-tail behavior. The `0.07` setting should be kept as a future validation-selected or transfer-oriented follow-up, not as the current headline result.

Artifacts:

- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit003_after250_holdout4096_checkpoint_sweep.md`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit003_after250_kodak_checkpoint_sweep.md`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit007_after250_holdout4096_checkpoint_sweep.md`
- `experiments/analysis/excessrisk090_local_cap080_rho1_betacommit007_after250_kodak_checkpoint_sweep.md`

## Beta005 Guard Transfers to an Unselected OpenImages Slice

The transfer-split audit strengthens the E059 paper-main candidate. Checkpoints were chosen only by the trusted holdout4096 protocol and then evaluated unchanged on another OpenImages slice (`start_index=8192`, `max_images_per_seed=512`). This is not yet the final full-split table, but it directly tests whether the beta005 gain is tied to the original checkpoint-selection slice.

The result is clearly positive in this pilot: beta005 guard reaches RD `2.130398`, compared with HCS `2.223952`, old gate0.25 `2.198195`, and min090 `2.195762`. The deltas are `-0.093554` vs HCS, `-0.067798` vs old gate0.25, and `-0.065365` vs min090.

The per-seed result is also cleaner than earlier controllers. Against same-seed HCS, beta005 improves seed1234 by `-0.051852`, seed2345 by `-0.119462`, and seed3456 by `-0.109347`. In contrast, old gate0.25 still damages seed3456 on this split (`+0.023630` vs HCS), and min090 damages seed1234 (`+0.022554` vs HCS). This matters for a conference claim because the method is no longer only a fragile-seed rescue or a single-seed accident.

The HCS-difficulty quartiles are uniformly favorable:

| quartile | beta005-HCS | win vs HCS |
|---|---:|---:|
| Q1 easy | -0.079099 | 0.783854 |
| Q2 | -0.098651 | 0.812500 |
| Q3 | -0.105755 | 0.747396 |
| Q4 hard | -0.090709 | 0.588542 |

The hard-tail nuance remains important. Q4 improves by a large amount, but its win rate is still lower than Q1-Q3, and the earlier previous-local step250 result remains the sharpest hard-tail-control ablation. The safer story is therefore: beta005 is the current main average/transfer candidate, while previous-local step250 explains the hard-tail reliability-control mechanism.

The feature distribution supports the proposed mechanism rather than contradicting it. Beta005 has lower `s_q` (`0.463094`) and higher qMSE (`0.112163`) than HCS, so it is using a stronger quantized latent regime. However, its Householder delta RMS stays low (`0.040389`) compared with old gate0.25 (`0.073651`) and min090 (`0.085345`). This is exactly the desirable operating point: keep hyperprior-conditioned geometry active, but prevent uncontrolled geometry amplitude from damaging reliability.

Paper implication: if the full 4096-image transfer audit holds, the main claim can be framed as a fixed-checkpoint HCG-RVQ variant that improves the original OpenImages validation protocol, Kodak transfer, and an unselected OpenImages split. That is substantially safer than relying only on the original holdout4096 slice.

Artifact: `experiments/analysis/beta005_transfer_openimages_start8192_n512.md`.

## Full Transfer Split Confirms Beta005 Is Not a Slice Artifact

The full OpenImages transfer audit confirms the E061 pilot result at `max_images_per_seed=4096`. The protocol is strict enough for paper reasoning: checkpoints were selected only by the trusted holdout4096 protocol, then applied unchanged to an unselected OpenImages slice starting at `start_index=8192`.

The headline result remains strong: beta005 guard reaches RD `2.135355`, compared with HCS `2.222382`, old gate0.25 `2.197166`, and min090 `2.193251`. That is `-0.087027` vs HCS, `-0.061811` vs old gate0.25, and `-0.057895` vs min090. The n512 pilot gave `-0.093554` vs HCS, so the effect is stable when scaled to the full 4096-image-per-seed audit.

The seed behavior is especially useful for the paper. Beta005 beats same-seed HCS for all three seeds: seed1234 `-0.049906`, seed2345 `-0.105958`, and seed3456 `-0.105216`. By contrast, old gate0.25 still hurts seed3456 (`+0.024182`), and min090 still hurts seed1234 (`+0.017731`). This turns beta005 from a promising controller into the current safest fixed-checkpoint HCG-RVQ variant.

The quartile result is uniformly positive:

| quartile | beta005-HCS | win vs HCS |
|---|---:|---:|
| Q1 easy | -0.079769 | 0.809896 |
| Q2 | -0.100822 | 0.801432 |
| Q3 | -0.103274 | 0.751628 |
| Q4 hard | -0.064243 | 0.615234 |

The hard-tail caveat still matters: Q4 is positive, but the improvement is smaller than Q1-Q3 and less sharp than the earlier previous-local step250 hard-tail specialist. This does not weaken the main claim; it clarifies the roles. Beta005 is the paper-main fixed-checkpoint average/transfer method, while previous-local step250 remains the ablation that explains selective hard-tail reliability control.

The intermediate features support the intended mechanism. Beta005 operates at lower `s_q` (`0.463857`) and higher qMSE (`0.107123`) than HCS, so it is not simply avoiding quantization. But it keeps Householder delta RMS low (`0.040056`), whereas old gate0.25 and min090 have much larger delta RMS (`0.073179` and `0.084711`) and much smaller RD gains. This is the useful story: the beta-commit guard lets HCG use a stronger quantized latent regime while preventing uncontrolled geometry amplitude.

Paper implication: the HCG-RVQ claim is now substantially safer than before. We have three aligned evidence sources for the same fixed-step candidate: trusted holdout4096, Kodak transfer, and full unselected OpenImages transfer. The next research action should move from proving that beta005 is real to making the manuscript claim tight: ablate HCS vs geometry vs reliability guard, include checkpoint-selection protocol explicitly, and add one more external-style robustness check if time allows.

Artifact: `experiments/analysis/beta005_transfer_openimages_start8192_n4096.md`.

## CLIC External Audit Strengthens the Fixed-Checkpoint Claim

The CLIC mobile/professional audit adds an important robustness layer. Unlike the OpenImages transfer split, CLIC is a separate benchmark-style distribution. The split sizes are small (`61` mobile images and `41` professional images per seed), so these rows should not be oversold as full benchmark dominance. But they are very useful because the checkpoint rule is fixed by OpenImages holdout4096 and no CLIC-specific checkpoint selection is used.

The results are aligned with the OpenImages story. On CLIC mobile valid, beta005 reaches RD `1.736549`, compared with HCS `1.821108`, old gate0.25 `1.809649`, and min090 `1.796757`. The deltas are `-0.084559` vs HCS, `-0.073099` vs old, and `-0.060208` vs min090.

On CLIC professional valid, beta005 reaches RD `1.872124`, compared with HCS `2.002328`, old gate0.25 `1.970373`, and min090 `1.955854`. The deltas are even larger: `-0.130204` vs HCS, `-0.098249` vs old, and `-0.083730` vs min090.

The per-seed pattern is paper-useful. Beta005 improves all three seeds on both CLIC splits. Old gate0.25 and min090 still have seed-specific damage patterns: on CLIC mobile, old damages seed1234 (`+0.013525`) and min090 damages seed1234 (`+0.017331`); on CLIC professional, old damages seed1234 (`+0.005535`) and min090 damages seed1234 (`+0.020893`). Beta005 avoids that failure and improves seed1234 as well.

The quartile pattern is also favorable. Beta005 improves all HCS-difficulty quartiles on both CLIC splits. Mobile Q4 still has the smallest gain (`-0.039919`) but remains positive, while professional Q4 is strong (`-0.118604`). This matches the broader conclusion: beta005 is the safest fixed-checkpoint average/transfer method, and previous-local step250 remains the sharper hard-tail ablation rather than the paper-main codec.

Mechanistically, CLIC repeats the same signature seen on OpenImages. Beta005 lowers `s_q` and raises qMSE relative to HCS, but keeps Householder delta RMS low (`0.032413` on mobile and `0.034772` on professional). Old/min090 have larger delta RMS (`0.065083/0.074184` on mobile and `0.069964/0.080232` on professional) with weaker RD gains. This supports the claim that the beta-commit guard stabilizes usable hyperprior-conditioned geometry instead of merely suppressing it.

The manuscript-ready evidence table is now captured in `experiments/analysis/beta005_paper_evidence_summary.md`. After the later Kodak fixed-protocol audit, the current strongest claim is: a fixed-checkpoint HCG-RVQ variant with reliability-aware geometry control and beta-commit stabilization improves HCS, old geometry gating, and min090 risk control across trusted OpenImages validation, unselected OpenImages transfer, Kodak, and CLIC mobile/professional external-style splits.

Artifact: `experiments/analysis/beta005_paper_evidence_summary.md`.

## Beta005 Claim Matrix Freezes the Current Paper Boundary

The beta005 evidence is now organized into a manuscript-facing claim matrix. The current safe headline is a fixed-checkpoint prototype-codec claim: beta005 improves over HCS, old gate0.25, and min090 on trusted OpenImages holdout4096, unselected OpenImages transfer, Kodak, CLIC mobile valid, and CLIC professional valid. The deltas vs HCS are -0.090018, -0.087027, -0.105668, -0.084559, and -0.130204 RD respectively, with zero nonfinite rows.

This does not mean we should claim broad SOTA dominance yet. The correct boundary is narrower and stronger: under a controlled MeanScaleHyperprior/RVQ prototype, hyperprior-conditioned geometry becomes reliable when paired with local geometry control and beta-commit stabilization. That is exactly aligned with the prompt thesis that hyperprior should generate quantizer geometry, but it still needs stronger-backbone or official-baseline integration before making a SOTA-facing statement.

The supporting evidence should be split in the paper as follows:

| evidence type | role |
|---|---|
| HCS -> old gate0.25 -> min090 -> beta005 | main ablation path |
| previous local cap080/rho1 step250 | hard-tail mechanism ablation |
| old/min090 selector | diagnostic headroom only, because it is multi-checkpoint |
| posthoc min090, min095, delta010, cap080-to-cap060 | negative controls showing the gain is not a trivial shrinkage rule |

The next analysis step is to turn `experiments/analysis/beta005_paper_claim_matrix.md` into paper tables: one compact main ablation table and one appendix guardrail table. GPU time should next go to a stronger-backbone plug-in or a final official-baseline comparison only after this prototype claim table is frozen.

## Paper Table Split: Main Result vs Guardrails

The generated `beta005_paper_tables.md` now gives two separate paper-table candidates. The main table should carry the positive story: HCS-RVQ baseline, old Householder geometry, min090 risk control, previous local hard-tail control, and beta005 fixed-checkpoint stabilization. The appendix table should carry the rejected controls: posthoc min090 on old weights, min095 risk floor, delta_reg010, and cap080-to-cap060.

This split matters for the paper because it prevents the narrative from sounding like arbitrary hyperparameter search. The positive row is beta005 because it wins the 3-seed holdout by `-0.090018` RD vs HCS and keeps the Q4 hard-tail positive at `-0.084461`. The local cap080/rho1 step250 row remains valuable because it is even stronger on Q4 (`-0.136153`), but it is not the average/transfer paper-main codec. The rejected rows show that simply changing the risk threshold, applying a posthoc risk gate, or shrinking Householder displacement does not reproduce the beta005 behavior.

So the current paper structure should be: main claim table first, transfer/external table second, guardrail ablations in appendix, and stronger-backbone integration only after the prototype table is fixed.

## Kodak Fixed-Protocol Row Strengthens the Prototype Claim

Kodak has now been audited with the same fixed-checkpoint protocol as the OpenImages transfer and CLIC rows. This matters because the earlier Kodak evidence was beta-focused checkpoint-sweep evidence, while the new row directly compares HCS, old gate0.25, min090, and beta005 under the same checkpoint rule.

The result is strong: beta005 reaches RD `2.100549`, compared with HCS `2.206217`, old gate0.25 `2.186182`, and min090 `2.195049`. That is `-0.105668` vs HCS, `-0.085633` vs old, and `-0.094501` vs min090. It improves every seed and every HCS-difficulty quartile, with zero nonfinite rows.

Mechanistically, Kodak repeats the same reliable-HCG pattern seen on the larger splits. Beta005 has a low Householder delta RMS (`0.030612`) compared with old/min090 (`0.055672`/`0.062015`) while retaining strong RD gains. Its `s_q` is lower (`0.482487`) and qMSE is higher (`0.065989`) than HCS, so the method is not avoiding VQ; it is stabilizing a stronger quantized-latent operating point.

The paper implication is useful: the fixed-checkpoint prototype claim now has five aligned evidence rows: OpenImages trusted holdout4096, OpenImages transfer start8192, Kodak, CLIC mobile, and CLIC professional. This is strong enough for the controlled prototype claim. SOTA plug-in should still be pursued, but it should be framed as the next stage rather than a prerequisite for trusting the current HCG-RVQ mechanism.

## Parallel Track Decision After the Beta005 Table Freeze

Re-reading the prompt confirms that the target is not simply to lower RD with an arbitrary controller. The paper thesis is that the hyperprior can generate local quantizer geometry, and the research question is how to make that geometry reliable enough for learned image compression.

The current evidence supports two active tracks:

1. Paper-claim hardening. Beta005 now has five aligned fixed-checkpoint rows: OpenImages trusted holdout4096, OpenImages transfer start8192, Kodak, CLIC mobile, and CLIC professional. It improves over HCS, old gate0.25, and min090 on every row, with zero nonfinite outputs. This is ready as the controlled MeanScaleHyperprior/RVQ prototype claim, while still stopping short of SOTA dominance.
2. Method improvement. The selector-headroom audit shows that previous-local step250 and excess-risk step500 are complementary: the oracle reaches RD `2.147093`, and a simple Householder delta-RMS threshold reaches RD `2.170055`. Because this uses two checkpoints, it should not be a final method. It should guide a single-checkpoint reliability controller that keeps beta005's average/transfer strength while recovering more of the hard-tail behavior.

This means the two activities are genuinely parallel. The paper track is freezing the claim boundary and tables; the method track is using those same diagnostics to choose the next higher-upside experiment. Strong-backbone/SOTA plug-in work is now justified, but the first step should be an official-baseline and interface audit rather than a broad GPU-heavy integration, because the present novelty still has to be attributable to HCG geometry rather than to an external architecture.

The immediate method priority is therefore a narrow single-controller experiment, not a random sweep: take the measured delta-RMS reliability signal suggested by selector headroom, implement it inside one checkpoint, and promote it only if it beats beta005 on both mean RD and hard-tail behavior under the same fixed protocol.

Artifact: `experiments/analysis/hcg_rvq_parallel_next_plan.md`.

## Rel075 Reliability Probe Outcome

The first single-checkpoint reliability-controller probe after the beta005 table freeze is a negative but useful result. `rel075` adds a detached reliability multiplier on top of the beta005 guard, but on the fragile seed3456 OpenImages holdout4096 protocol it reaches RD `2.210781`, which is `+0.054365` worse than beta005 seed3456 step500 (`2.156416`). Step250 is also worse (`2.243897`, `+0.087481`). There were no NaN/nonfinite outputs, and the run was pinned to physical GPU 0.

The feature distribution explains why this should not be promoted. The learned reliability multiplier is almost an identity factor: mean `0.992290`, min `0.990848`, max `0.994993`, and std `0.000657`. Meanwhile, the main geometry and quantization statistics nearly match beta005: `s_q_mean` `0.456180` vs `0.456233`, latent qMSE `0.112445` vs `0.112377`, Householder delta RMS `0.039998` vs `0.040308`, and dead-code ratio `0.030619` vs `0.030539`. The failure is therefore not excessive gate suppression, codebook collapse, or geometry explosion. It is closer to an inactive extra head that fails to reproduce beta005 image-domain distortion.

Paper implication: beta005 remains the paper-main prototype row. Method implication: do not spend GPU on 3-seed `rel075`. The next reliability controller should not be another free multiplier near the gate; it should use an explicit measured reliability signal or teacher target derived from the selector-headroom analysis, while preserving beta005 fixed-checkpoint protocol.

Artifact: `experiments/analysis/betacommit005_rel075_seed3456_probe.md`.

## Decoder-Safe Selector Points to the Next Single-Checkpoint Controller

After `rel075` failed, the next question was whether the previous-local/beta005 complementarity is only visible through non-deployable diagnostics or whether the hyperprior-generated conditioning already contains a usable reliability signal. The decoder-safe selector audit answers this positively, but still as headroom rather than a final method.

The oracle switch between previous-local step250 and beta005 step500 reaches RD `2.121089`, which is `-0.142617` vs HCS and `-0.052599` vs beta005. More importantly, a single decoder-safe feature already captures a meaningful part of that headroom: selecting beta005 when `rvq_householder_gate_raw <= 0.284059` and otherwise falling back to previous-local reaches RD `2.158794`, `-0.104912` vs HCS and `-0.014894` vs beta005. This uses beta005 on about `86.9954%` of images.

The best diagnostic threshold is still stronger: `rvq_householder_delta_rms <= 0.047937` reaches RD `2.145928`, `-0.027760` vs beta005. But that feature depends on the pre-quantization latent/geometry outcome, so it is better treated as a teacher signal or target for training rather than as a decoder-side rule.

This changes the method-improvement plan in a good way. The next single-checkpoint controller should be raw-gate-informed selective geometry backoff: preserve beta005 behavior for the large low-raw-gate region, and regularize or train high-raw-gate cases toward previous-local-like safer geometry. This is still aligned with the prompt because the hyperprior-generated geometry remains central; the new piece is reliability control over when that generated geometry should be trusted.

Artifact: `experiments/analysis/beta005_previous_local_decoder_safe_selector.md`.

## Raw-Gate Backoff Is Too Blunt for the Selector Headroom

The E069 selector audit showed real headroom: a decoder-safe raw-gate threshold could choose between previous-local and beta005 and beat beta005 in a diagnostic two-checkpoint switch. E070 tests the obvious single-checkpoint translation of that idea: multiply Householder geometry by a smooth raw-gate backoff factor. The result is negative and clarifies the design boundary.

The posthoc check is the cleanest causal test. Applying the rawbackoff controller to the existing beta005 seed3456 step500 checkpoint worsens RD from `2.156416` to `2.180182` (`+0.023766`). That means the continuous multiplier itself damages beta005, even without retraining. Training with the multiplier worsens further to `2.228286` (`+0.071870`), which suggests co-adaptation around the raw-gate signal rather than recovery of selector-like behavior.

The feature distribution supports this interpretation. The posthoc multiplier is `0.854361`, and the trained multiplier is `0.810320`; both suppress the geometry across too much of the distribution. Householder strength falls from beta005 `0.264690` to `0.225906` posthoc and `0.219493` after training. The trained model also increases latent qMSE from `0.112377` to `0.115936`. This is not a nonfinite/collapse failure; it is a reliability-control design failure: a soft scalar multiplier does not approximate the discrete previous-local/beta005 complementarity.

Paper implication: beta005 remains the current paper-main controlled prototype row. The raw-gate selector audit should be treated as headroom/teacher evidence, while `rawbackoff065_t0284` should go into guardrails or internal notes as a negative control. The next method-improvement path should not be another unconstrained multiplier. It should use a teacher/supervised target or distribution-aware reliability objective that preserves beta005 geometry for low-risk images and selectively moves high-risk cases toward previous-local-like safer geometry.

Artifact: `experiments/analysis/betacommit005_rawbackoff065_t0284_seed3456_probe.md`.

## Raw-Tail Probe Updates the Method-Improvement Path

The raw-gate tail regularizer is a clean negative result. It was designed to be less blunt than rawbackoff by penalizing only the high raw-gate image-mean tail above the decoder-safe selector threshold `0.284059`. The numerical outcome is stable (`nonfinite_rows=0`) and improves over rawbackoff, but it still loses to beta005 on seed3456 holdout4096: rawtail step500 RD is `2.193294`, while beta005 is `2.156416`.

The important mechanism is not collapse. Rawtail step500 has almost beta005-identical geometry and codebook statistics: `s_q_mean` `0.455570` vs `0.456233`, latent qMSE `0.113065` vs `0.112377`, Householder delta RMS `0.040086` vs `0.040308`, strength `0.264145` vs `0.264690`, and dead-code ratio `0.030680` vs `0.030539`. The model can therefore satisfy or partially route around the raw-gate-tail pressure while returning to the same broad geometry regime and worse RD.

The per-image result makes this sharper. Rawtail step500 beats beta005 on only `32.8369%` of images. In beta005 high raw-gate Q4, rawtail is `+0.101462` RD worse and wins only `15.4297%`. So raw gate remains a useful diagnostic/proxy in selector audits, but it is too endogenous to be the only training target.

Decision: beta005 remains the paper-main controlled prototype. Rawtail joins rawbackoff as a guardrail negative control: a raw-gate-derived reliability signal has headroom as a two-checkpoint selector, but simple single-checkpoint penalties/multipliers do not recover that headroom. The next credible method-improvement experiment should use a detached or train-split teacher target built from diagnostic delta RMS or the explicit per-image beta005-vs-previous-local labels, then verify one checkpoint under the same fixed OpenImages/Kodak/CLIC protocol.

Artifact: `experiments/analysis/betacommit005_rawtail_t0284_rho100_seed3456_probe.md`.

## Leave-One-Seed-Out Audit Makes the Next Teacher Target Safer

The beta005/previous-local selector headroom is not only an all-data artifact. A leave-one-seed-out audit chooses a fallback policy on two seeds and evaluates it on the third. Decoder-safe conditioning features retain a small held-out average gain (`-0.006046` RD vs beta005), but the gain is seed-dependent and fails on held-out seed3456 (`+0.017255`). Diagnostic delta-RMS features are stronger and more stable, with mean held-out `-0.024650` RD and positive gains on all three held-out seeds.

This matters for the paper path because it separates three things that should not be mixed: diagnostic headroom, trainable teacher construction, and deployable inference. Raw-gate is decoder-safe but too endogenous when directly penalized. Delta-RMS is not a deployable decoder-side switch, but it is a better teacher signal for split-generated supervision. The next controller should therefore learn from diagnostic-delta or explicit beta005-vs-previous-local labels on a training/teacher split, then be evaluated as one fixed checkpoint with no validation-time switching.

Artifact: `experiments/analysis/beta005_teacher_target_loso.md`.

## Delta-Tail Probe Closes the Direct-Penalty Branch

The `deltatail_t0479_rho100_betacommit005` result is an important negative control because it tests the most literal translation of the diagnostic delta-RMS selector into a single-checkpoint loss. The LOSO audit showed that delta-RMS is a strong teacher signal, but this run shows that directly penalizing the image-level delta-RMS tail does not recover the selector headroom.

The key distinction is that the loss changes the representation, not just the reliability decision. Step500 does reduce the measured delta-RMS mean (`0.040308 -> 0.038047`) and the high-delta tail fraction (`0.242432 -> 0.184082`), so the penalty is active. But the model pays for that by increasing latent qMSE (`0.112377 -> 0.126267`) and worsening RD by `+0.069162` against beta005. This is the co-adaptation pattern we wanted to detect: the diagnostic signal is moved, but the image-domain rate-distortion behavior gets worse.

The quartile breakdown makes the failure sharper. In beta005's high delta-RMS Q4, deltatail step500 is `+0.136271` RD worse and wins only `9.9609%` of images. Therefore the result should not be interpreted as evidence that delta-RMS is a bad teacher. It is evidence that direct end-to-end tail penalization is the wrong mechanism for using that teacher.

The updated method-improvement direction is now clearer: train a deployable reliability controller from split-generated labels or distillation targets, not by directly shrinking raw gate or delta-RMS tails. The paper-main claim remains beta005, because it is the only fixed-checkpoint prototype row with stable multi-split evidence. Rawbackoff, rawtail, and deltatail should be kept as guardrail ablations showing that the gain is not reproduced by simple geometry suppression or naive reliability penalties.

Artifact: `experiments/analysis/betacommit005_deltatail_t0479_rho100_seed3456_probe.md`.

## Teacher-Label Reliability Controller Turns Selector Headroom Into a Trainable Branch

The latest action moved the method-improvement track from direct feature suppression to supervised reliability control. This is the right direction after the negative controls: rawbackoff, rawtail, and deltatail were numerically stable but each made RD worse than beta005. Their common failure mode was co-adaptation: the training objective changed the measured signal, but did not reproduce the per-image beta005/previous-local complementarity.

The new teacher-label export keeps the useful part of the selector analysis while making the next experiment deployable as one checkpoint. Across 12288 audited OpenImages rows, previous-local wins on `34.7087%` of images even though beta005 has much better average RD (`2.173688` vs `2.221143`). The oracle switch reaches RD `2.121089`, so there is still `0.052599` RD of headroom beyond beta005. That is too large to ignore, but it must be captured by a learned controller rather than by validation-time checkpoint switching.

The smoke implementation is intentionally conservative. It supervises the existing `householder_reliability_multiplier` rather than adding a new codec path, converts the multiplier into a binary keep probability, and leaves the paper-main beta005 evidence untouched. The two-step GPU smoke only proves that labels, dataset paths, conditioning tensors, and teacher loss are wired correctly. It is not a publishable result because the labels come from the audited holdout artifacts.

The next publishable-grade experiment should use split discipline: build teacher labels on a training/teacher slice, train the reliability controller there, and evaluate the resulting single checkpoint on the already frozen trusted holdout/Kodak/CLIC protocol. Promotion criteria should be strict: beat beta005 mean RD, avoid the Q4 hard-tail regression seen in direct penalties, keep zero nonfinite rows, and show in feature analysis that reliability changes are selective rather than broad geometry shrinkage.

This keeps the paper claim aligned with `prompt.txt`: the proposal is still hyperprior-generated local quantizer geometry, with reliability control deciding where that geometry is trustworthy. Beta005 remains the current manuscript-safe fixed-checkpoint method; the teacher-label branch is the higher-upside method-improvement track.


## Transfer-Label Controller Branch: Split Headroom vs Deployable Control

The transfer8192 teacher-label audit is useful because it separates teacher construction from holdout4096 evaluation. On transfer8192, previous-local beats beta005 on 33.7484% of rows, and the beta005/previous-local oracle reaches RD 2.079763 versus beta005 2.135355. This means the complementary behavior is real on a split not used for final holdout scoring.

The first deployable controller trained from the scalar initialization does not convert that headroom into a usable checkpoint. On seed3456 holdout4096, its best step is step500 RD 2.244678, which is +0.088262 worse than beta005 and nearly the previous-local operating point. The feature distribution is stable but not successful: reliability remains high, while s_q/qMSE/geometry shift away from the beta005 checkpoint. This rejects the from-scalar controller as a paper-main candidate.

The beta005-initialized head-only probe gives the key diagnostic. Freezing the whole codec and training only the reliability head preserves beta005 almost exactly: step250 RD 2.156985 and step500 RD 2.157168 versus beta005 2.156416, with zero nonfinite rows. This proves the implementation path can add a reliability controller without destroying the fixed-checkpoint evidence. However, reliability remains near identity, so it does not recover the oracle/selector headroom.

Current conclusion: beta005 remains the paper-main fixed-checkpoint method. The method-improvement branch should continue, but only in the beta005-initialized head-only regime or an equally conservative regime. The next experiment should strengthen the teacher objective or reshape it into soft margin/ranking supervision, rather than training the whole HCG branch from scalar initialization.


## Stronger Teacher Weight Confirms That Head Safety Is Solved, Not Head Selectivity

The rho0.50 head-only teacher run is a useful negative result because it tests the simplest explanation for E076: maybe rho0.05 was just too weak. The answer is no. Raising the teacher weight by 10x keeps the beta005-initialized frozen-codec path numerically stable, but does not turn the reliability head into a useful selector.

The checkpoint comparison is very tight. Rho0.50 step250 reaches RD 2.157003 and rho0.50 step500 reaches 2.157284 on seed3456 holdout4096, compared with beta005 2.156416. Both checkpoints still beat HCS by about -0.106 RD and have zero nonfinite rows, so the controller is safe. But both are slightly worse than beta005, and step500 is also slightly worse than the rho0.05 step500 probe.

The intermediate features explain the failure mode. Rho0.50 step500 does reduce the reliability mean to 0.985157, compared with 0.987704 for rho0.05 and near-identity 1.0 for beta005. It also slightly reduces Householder delta RMS and strength. However, s_q_mean stays fixed at 0.456233, latent qMSE stays at 0.112389, raw gate stays at 0.279697, and dead-code ratio stays at 0.030436. In other words, the head is moving, but it is mostly applying a weak global attenuation around the beta005 operating point rather than learning a strong image-conditional fallback policy.

The per-image reliability-signal audit sharpens that conclusion. On holdout4096, previous-local beats beta005 on 25.2686% of seed3456 images. Rho0.50 step500 assigns lower reliability to that fallback-needed group, 0.983433 versus 0.985740 for keep-like images, and the low-reliability AUC for identifying fallback-needed images is 0.717021. The direction is therefore not random. The problem is magnitude and objective shape: the separation is too small to recover the beta005/previous-local oracle headroom, and the resulting RD delta remains +0.000868 versus beta005.

The research implication is clean. The paper-main claim should remain beta005: hyperprior-generated local quantizer geometry plus local control and beta-commit stabilization is already supported across the fixed protocol. The method-improvement track should not spend more GPU on plain binary-teacher weight scaling. The next credible experiment should keep the same beta005-initialized head-only safety shell, but replace the binary BCE-like supervision with a shaped target tied to excess RD, margin/ranking against beta005, or soft reliability values derived from the teacher split. That would directly ask the head to spend its limited control budget where beta005 is measurably risky.

Artifacts: experiments/analysis/teacher_transfer8192_rel075_rho050_headonly_from_beta005_seed3456_probe.md and experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.md.


## Margin-Balanced Weighting Exposes VQ Assignment Fragility

The margin-weighted head-only teacher experiment answers the next natural question after E077. If the unweighted teacher target is directionally correct but too weak, can we make it useful by balancing fallback images and upweighting large RD-margin labels? The measured answer is no, at least without an explicit preservation guardrail.

The average RD result is much worse than beta005: step250 RD 2.850224 and step500 RD 2.850449, both about +0.694 RD relative to beta005. This is not a NaN or checkpoint-file failure. Both evaluations have zero nonfinite rows, and a direct checkpoint drift audit shows that the only tensors changed relative to beta005 are householder_reliability_head.weight and householder_reliability_head.bias. The rest of the codec is byte-level stable within the audit threshold.

The intermediate features reveal the mechanism. The reliability mean still looks near identity, 0.990860 at step250 and 0.982952 at step500, but latent qMSE jumps from beta005 scale 0.112377 to about 0.1534 and dead-code ratio jumps from 0.030539 to about 0.1836. The Householder strength and delta RMS do not explode. Instead, a small reliability-head shift changes the effective local geometry enough to put the residual quantizer into a poor assignment regime.

This is an important negative control for the paper path. It supports the claim that HCG-RVQ is not merely a soft scalar gate story; the generated geometry interacts sharply with the RVQ codebook. It also says that future reliability controllers need a codebook-usage or beta005-assignment preservation term, not just a better teacher label. The controller must decide when to trust geometry while keeping the discrete quantizer in a stable operating region.

The method branch should therefore pivot from stronger BCE weighting to constrained selectivity: preserve beta005 assignments/perplexity/dead-code ratio unless the teacher signal is both strong and locally safe. Candidate mechanisms are an assignment-distribution KL guardrail, a small residual reliability parameterization around beta005, or a ranking objective with an explicit qMSE/dead-code penalty. Beta005 remains the paper-main fixed-checkpoint method.

Artifacts: experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_headonly_from_beta005_seed3456_probe.md and experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.md.


## Low-Amplitude Reliability Does Not Fix Margin-Weighted BCE

The rel095 low-LR follow-up closes an important loophole in the E078 interpretation. If the margin-weighted teacher failed only because the reliability head moved too far, then raising the reliability floor to `0.95` and reducing the reliability-head LR to `0.25x` should have restored beta005-like codebook usage. It did not. Step250 and step500 both stay near RD `2.8499`, about `+0.6935` worse than beta005, with zero nonfinite rows.

The mechanism is sharper than before: reliability is almost identity (`0.998840` at step250, `0.998645` at step500), and a checkpoint drift audit confirms that only the two `householder_reliability_head` tensors changed, but latent qMSE remains `0.153383` and dead-code ratio remains `0.183668`. These are essentially the E078 failure values, not beta005 values (`0.112377` qMSE and `0.030539` dead code). Therefore the damaging part is not a large average reliability suppression. The weighted BCE appears to push a spatial/group reliability pattern that crosses fragile RVQ assignment boundaries even under a very small multiplier range.

The fallback-signal audit agrees. The rel095 variant has almost no fallback/keep separation (`-0.000054` at step500), AUC around `0.669`, and still damages both keep and fallback-needed images (`+0.602076` and `+0.963835` RD vs beta005 respectively). That means the controller is not spending reliability capacity selectively; it is changing the quantizer operating point in a way that the image-domain RD objective cannot recover.

The method lesson is concrete. Future reliability control should not be another binary BCE weighting variant unless it includes a preservation term for beta005 assignments, codebook usage, qMSE, or a distillation target on the quantized latent. For the paper narrative, this is a useful guardrail: HCG-RVQ's gain is geometry/codebook interaction, so reliability control must preserve the discrete quantizer state while deciding when to trust geometry. Beta005 remains the manuscript-safe single-checkpoint result.

Artifacts: experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_probe.md, experiments/analysis/teacher_transfer8192_rel095_marginw_rho050_lrm025_headonly_from_beta005_seed3456_checkpoint_drift.md, and experiments/analysis/teacher_transfer8192_headonly_reliability_signal_audit.md.

## Correction: E078-E080 Need Current-Code Re-Anchoring

The E078/E079 interpretation above is now corrected by a direct current-code beta005 audit. The earlier conclusion treated historical `variant500_rd` from `excessrisk090_local_cap080_rho1_betacommit005_after250_tail_holdout4096.csv` as interchangeable with the E078-E080 current-code evaluations. That was false. When the beta005 checkpoint is evaluated directly under the same current config/probe path as E078-E080, it gives RD `2.850149`, not `2.156416`. With the original beta005 config, the same checkpoint gives RD `2.849824`. The original-config delta to the historical reference is `+0.693408`, which explains almost all of the apparent `+0.694` failure in E078/E079/E080 and rules out a simple reliability-config mismatch.

This changes the scientific interpretation. E078 and E080 are nearly identical to current-code beta005 (`+0.000075` to `+0.000301` RD), while E079 is slightly better than current-code beta005 (`-0.000257` at step250 and `-0.000246` at step500). The E079 gain is far too small to promote, but the old "catastrophic assignment collapse" explanation is not supported.

The anchor-preservation audit makes the correction stronger. E080 step250 has mean RVQ `index_match=0.999681` against the beta005 anchor and mean `y_hat_mse=2.20e-06`; E080 step500 has mean `index_match=0.999085` and mean `y_hat_mse=7.28e-06`. A controller that preserves almost all RVQ assignments and the quantized latent cannot be the cause of a `+0.694` RD jump. The jump is an artifact/protocol mismatch.

Immediate paper-safety rule: do not mix historical beta005 CSVs with current-code controller CSVs. Any future claim table must be regenerated inside one frozen protocol, with explicit config path, checkpoint path, split start index, image paths, device, and probe inverse mode. The older beta005 table can remain as a historical paper-main candidate only if its full code/protocol state is restored or revalidated. Until then, current-code method-improvement rows are diagnostic, not paper-main evidence.

Next action priority:

1. Pin the current protocol or restore the historical one.
2. Regenerate HCS, old gate0.25, min090, beta005, and the best controller row under the same state.
3. Re-run checkpoint and intermediate-feature audits for the regenerated rows.
4. Promote only effects that survive this re-anchor by more than numerical noise and improve both mean RD and hard-tail behavior.

Artifacts: `experiments/analysis/teacher_transfer8192_current_beta005_reanchor_audit.md`, `experiments/analysis/beta005_after250_seed3456_original_config_direct_step500_val4096_holdout4096_current.md`, `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step250_anchor_drift.md`, and `experiments/analysis/teacher_transfer8192_rel075_marginw_rho050_yhatanchor100_headonly_from_beta005_seed3456_step500_anchor_drift.md`.



## Full-Image Protocol Re-Anchor Corrects the E078-E080 Interpretation

The previous current-code re-anchor section was directionally right to distrust mixed artifacts, but the more precise cause is now identified: the direct current-code probes that produced RD near `2.85` used `patch_size=256` center crops, while the historical beta005 paper-facing artifact uses full-image evaluation (`patch_size=None`). `ImageFolderDataset` returns the whole image when `patch_size` is `None` and deterministic center crops otherwise, so those rows are different evaluation tasks.

A direct full-image probe of the same beta005 seed3456 step500 checkpoint reproduces the historical row: RD `2.156416023`, mean per-image RD difference `9.75e-08` against historical `variant500_rd`, and zero nonfinite rows. This rules out current-code collapse, stale checkpoint overwrite, and reliability-config damage as explanations for the `+0.694` discrepancy.

Under the corrected full-image protocol, E079 `rel095_marginw_rho050_lrm025` is preservation-only: step250 is `+0.000083` RD and step500 is `+0.000092` RD vs beta005. Feature statistics also match beta005: qMSE `0.112378`, `s_q_mean` `0.456233`, dead-code ratio `0.030533`. Therefore the branch is not a catastrophic failure, but it is not an improvement. The paper-safe claim remains beta005 as the fixed-checkpoint HCG-RVQ prototype, and future controller claims must be evaluated and reported with explicit `patch_size`, split, checkpoint, device, and inverse mode.

Artifact: `experiments/analysis/teacher_transfer8192_fullimage_protocol_reanchor_audit.md`.


## E082 Analysis: Main Claim vs Method-Improvement Track

The E082 controller result narrows the conclusion after the full-image re-anchor. Reliability-head training with margin labels is not dangerous when interpreted under the correct protocol, but it is not yet useful. E079 is almost exactly beta005-preserving, and E080 with an explicit `y_hat` anchor is also stable but slightly worse. This means the bottleneck is not numerical stability or anchor preservation; it is selectivity. The head can stay near beta005, but it has not learned a strong enough policy to recover the beta005/local-cap or beta005/previous-local headroom.

The local-cap cross-split audit prevents over-pivoting. Local cap080/rho1 has a very attractive hard-tail profile on OpenImages, with much lower local geometry displacement and qMSE than beta005, but it loses to beta005 on average and fails to dominate Kodak. This supports a hybrid story rather than a replacement story: beta005 is the robust paper-main operating point, while local cap supplies a conservative fallback behavior for hard or geometry-risky regions.

Next experiment implication: do not spend more GPU on plain margin-weighted BCE or global local-cap substitution. The next serious method-improvement experiment should train a beta005-initialized controller with split-generated labels and an explicit preservation guardrail (`y_hat`, RVQ index match/distribution, qMSE/dead-code, or a small residual reliability parameterization). Promotion should require mean RD improvement over beta005 plus no Q4/hard-tail regression and stable intermediate features.


## E083 Analysis: Signal Exists, Image-Mean Reliability Is Too Weakly Selective

E083 separates two questions that had been easy to mix together: whether the local-cap/teacher signal has real headroom, and whether the current head-only reliability controller can recover it. The answer is now split cleanly. The signal is real. On seed3456 holdout4096, beta005 is RD `2.156416`, previous-local/local-cap is worse by itself at `2.245225`, but the per-image oracle of beta005 and previous-local is `2.114464` (`-0.041952` vs beta005). On transfer8192 over 3 seeds, the oracle is `2.079763` vs beta005 `2.135355` (`-0.055592`).

The current controller does not recover that headroom. Across E076-E080 full-image rows, the best mean RD is E079 rel095 low-LR step250 at `2.156499`, only `+0.000083` worse than beta005. Stronger rho, margin weighting, and y-hat anchoring are all stable but still average-worse. The feature distributions show preservation rather than collapse: qMSE remains around `0.11238`, `s_q_mean` remains `0.456233`, and dead-code ratio stays near `0.0304`.

The most useful mechanism clue is the quartile split by HCS difficulty. All controller variants improve Q4 hard images slightly, while hurting easier quartiles. For example, E078 margin-weighted step500 has Q4 delta `-0.001002` but Q1/Q2/Q3 deltas `+0.002246`, `+0.001639`, and `+0.000851`. This explains why the mean does not improve: the controller is directionally finding hard images, but the image-mean head-only supervision is too blunt.

Research implication: keep beta005 as paper-main, and move the method-improvement track from global/image-mean BCE labels to locality-aware selective control. A publishable strengthening path is to show that HCG-RVQ already supplies useful local geometry, then add a controller that keeps beta005 assignments/codebook usage unless a local teacher/ranking signal predicts hard-image benefit.

Artifact: `experiments/analysis/e083_fullimage_controller_family_closure.md`.


## E084/E085 Analysis: Selectivity Is the Bottleneck, Not Stability

E084 clarified why the reliability-controller branch kept looking directionally useful but average-worse. The E076-E080 controllers are not harmful on every image. When E078 margin-weighted step500 is applied posthoc only to the high beta005 `rvq_householder_delta_rms` subset around threshold `0.052714`, the mixed row improves beta005 by about `-0.000610` RD, and the selected subset itself improves by about `-0.004062` RD. Its quartile pattern is exactly the kind of behavior we want: almost no Q1 easy damage and a much stronger Q4 hard-image improvement. This is diagnostic headroom rather than a valid paper result, because the threshold is tuned on holdout4096, but it gives a concrete mechanism target.

E085 tested the first trainable version of that idea by keeping the beta005-initialized head-only safety shell and weighting the scalar teacher loss on local `householder_delta_rms_map`. The result is a clean negative. It is stable, uses GPU0 only, and produces zero nonfinite rows, with qMSE/dead-code still near beta005. But it does not improve RD: step250 is `+0.000594` and step500 is `+0.000881` vs beta005. It also does not improve the selective headroom: applying E085 step500 only to the same high delta-RMS subset gives `-0.000571`, slightly weaker than E080 (`-0.000593`) and E078 (`-0.000609`).

The important interpretation is that local weighting of an image-level label is too indirect. E085 still has the same average failure profile as E078/E080: Q4 hard images improve slightly, while Q1-Q3 lose more than enough to erase the gain. Its reliability high-low separation is also weaker than E080 at step500 (`-0.005178` vs `-0.005560`), so the local weighting did not make the controller more selective in the desired direction.

This refines the method-improvement plan. The next credible branch should not be another global BCE/margin-weight variant. It should explicitly train a deployable selector or keep policy: preserve beta005 on low-risk images/locations, supervise suppression only on independently chosen high-risk regions from transfer8192, and keep the `y_hat`/qMSE/dead-code/RVQ-assignment guardrails. That is the cleanest way to convert the observed beta005/local-cap complementarity into a single-checkpoint HCG-RVQ improvement without contaminating the paper-facing holdout.

Artifacts: `experiments/analysis/e084_selective_controller_threshold_headroom.md` and `experiments/analysis/e085_localdelta_weighted_teacher_audit.md`.

## E086 Analysis: Keep-Target Reduces Damage but Also Removes the Tail Gain

E086 tested the sharper version of the E085 hypothesis. Instead of merely weighting an image-level teacher on local `householder_delta_rms_map`, it locally changes the teacher target: low-risk locations are explicitly pushed to keep beta005-like reliability, and only transfer8192-selected high-risk locations receive the previous-local teacher signal. This is a cleaner protocol because the threshold `0.045151` comes from the transfer split rather than holdout4096.

The result is stable but not promotable. E086 step250 reaches RD `2.156993` (`+0.000577` vs beta005), and step500 reaches `2.157220` (`+0.000804`), both with zero nonfinite rows. These are small improvements over E085/E080, so the keep target does reduce some easy-image damage. But the improvement is too small and still does not beat beta005.

The more important mechanism result is that E086 also weakens the hard-tail benefit we were trying to preserve. On the transfer-derived high-delta selector, E086 step500 mixed apply-selected-only delta is `-0.000387`, compared with E085 `-0.000444`, E080 `-0.000459`, and E078 `-0.000471`. On the stricter E084 holdout diagnostic selector, E086 step500 is `-0.000500`, again weaker than the earlier controller rows. Quartiles tell the same story: Q1-Q3 damage is smaller, but Q4 hard-image improvement shrinks to `-0.000811` at step500.

This closes a useful branch. BCE-style reliability supervision is safe under the beta005-initialized frozen-codec shell, but it mostly trades off easy damage against hard-tail gain and never converts the transfer/oracle headroom into a beta005-beating single checkpoint. The next credible strengthening path should stop treating reliability as the direct target. It should train a deployable selector or mixture policy against RD/ranking outcomes, with explicit preservation of beta005 `y_hat`, RVQ assignment distribution, qMSE, and dead-code behavior on low-risk regions.

Paper implication: beta005 remains the manuscript-safe fixed-checkpoint prototype for the HCG-RVQ geometry claim. E076-E086 become supporting evidence that the geometry signal has selective headroom, that the system is stable on GPU0/full-image protocol, and that a stronger controller needs outcome-level selection rather than another teacher BCE variant.

Artifact: `experiments/analysis/e086_selector_keep_target_audit.md`.

## E087 Analysis: Outcome-Level RD Head Is Safer but Too Conservative

E087 tested the main lesson from E086 directly: stop supervising reliability as a BCE target and let the image RD objective move only the reliability head, while `y_hat` anchoring keeps the beta005 quantized-latent path close. This is a cleaner outcome-level probe than E080/E085/E086 because the controller is no longer asked to imitate a scalar teacher; it is asked to avoid RD damage under the actual compression objective.

The result is the best controller row in the current branch, but still not a beta005-beating method. E087 step500 reaches RD `2.156626`, only `+0.000210` worse than beta005, improving over E086 step500 by `-0.000594`, E085 step500 by `-0.000671`, and E080 step500 by `-0.000703`. It is numerically clean: `nonfinite_rows=0`, qMSE `0.112378`, `s_q_mean=0.456233`, and dead-code `0.030529`.

The mechanism is also clear. E087 preserves beta005 much better than the BCE teacher variants, but it loses their small hard-tail benefit. E080/E085/E086 still have negative deltas on high beta005 delta-RMS subsets, whereas E087 step500 has transfer-selected delta `+0.000041` and mixed apply-selected-only delta `+0.000013`. Its HCS difficulty quartiles are all small positive deltas, including Q4 hard `+0.000080`. Reliability is almost identity and even slightly higher on the transfer high-delta group (`high-low=+0.000437`), so the learned head is not selecting fallback-risk regions; it is mainly learning a conservative beta005-preserving correction.

This is a useful narrowing result. The branch is no longer about numerical instability or VQ collapse. The RD/y-hat anchored shell is stable and preferable to BCE-style reliability labels, but the missing ingredient is explicit selectivity. A publishable strengthening path should keep E087 as the safety baseline and add a deployable selector or pairwise ranking objective against beta005/local-cap outcomes, with constraints on beta005 reconstruction, RVQ assignment/code usage, qMSE, `s_q`, and dead-code statistics.

Paper implication: beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ geometry result. E087 should be described as method-improvement evidence: outcome-level control is safer than teacher BCE, but a single checkpoint still needs a selective/ranking mechanism before it can become a stronger HCG-RVQ variant.

Artifact: `experiments/analysis/e087_rdonly_head_probe_audit.md`.

## E088 Analysis: Transfer-Safe Selectivity Is Real

E088 is the first result in this branch that changes the next implementation priority in a positive way. The earlier E083-E087 experiments showed that beta005 and previous-local/local-cap are complementary, but the trainable reliability head either damaged easy images or became too conservative. E088 asks a simpler question first: if we train the selector on a separate transfer split, does the beta005/previous-local decision generalize to holdout at all?

The answer is yes. A decoder-safe logistic selector trained only on transfer8192 improves holdout4096 beta005 by `-0.027586` RD, closing about `52.45%` of the beta005-to-oracle gap. This is much larger than the sub-`0.001` controller effects seen in E080-E087. The selected fraction stays moderate (`24.29%`), precision is high (`0.7283`), and the result is positive on all three seeds. That matters because seed3456 had been the fragile seed; here it still improves by `-0.018749` versus beta005.

The HCS difficulty quartiles show the mechanism we wanted but did not get from global BCE supervision. The selector almost never switches on easy images (`2.25%` selected in Q1) and pays only `+0.000768` RD there. It switches aggressively on hard images (`59.60%` selected in Q4) and improves Q4 by `-0.089238` RD. This is exactly the missing selectivity: keep beta005 where the generated geometry is already reliable, and borrow the conservative/local behavior where beta005-side hyperprior features indicate high geometry risk.

The diagnostic feature upper bound is only slightly better (`-0.028083` versus beta005), so the useful information is not locked behind unavailable latent outcome features. The strongest decoder-safe weights are on `rvq_householder_v_abs_mean`, `rvq_mu_q_abs_mean`, `rvq_mu_q_std`, risk multiplier, and `s_q` statistics. That aligns with the HCG-RVQ thesis: hyperprior-generated conditioning carries information about local quantizer geometry and reliability.

The limitation is equally important. E088 is still a multi-checkpoint switch between beta005 and previous-local rows, so it cannot be claimed as the final proposed method. It also uses image-level selection, while the actual method should become a single checkpoint and preferably local/spatial or group-wise. Therefore beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ prototype, and E088 becomes a strong design target for the next branch.

The next paper-strengthening experiment should distill the E088 decoder-safe policy into a beta005-initialized single model: train a selector/ranking controller from transfer labels, preserve beta005 `y_hat` and RVQ assignment/code usage on low-risk regions, and evaluate full-image holdout4096 with qMSE, `s_q`, dead-code, quartile, seed, and checkpoint analyses. Promotion should require a mean RD improvement over beta005, a Q4 hard-image gain without Q1 damage, and stable intermediate feature distributions.

Artifact: `experiments/analysis/e088_transfer_learned_selector.md`.



## E089/E090 Analysis: Selector Labels Are Directional but Too Indirect

E089 answers whether the strong E088 transfer-learned selector can be turned into a single checkpoint by supervising the reliability head with image-level keep labels. The answer is no, but it is an informative no. The checkpoint is numerically clean and preserves the beta005 operating regime: qMSE, `s_q_mean`, dead-code, perplexity, and Householder strength all stay close to beta005, with zero nonfinite rows. Therefore the failure is not VQ collapse or CUDA/device instability.

The problem is magnitude and locality. The E088 decoder-safe switch improves seed3456 holdout beta005 by `-0.018749` RD and selects `17.4561%` of images. A successful distillation should recover a large fraction of that selected-image gain while leaving easy images alone. E089 does neither strongly enough. Step250 is globally `+0.000581` vs beta005, and step500 is `+0.000788`. The selected subset moves in the right direction (`-0.000638` at step250, `-0.001469` at step500), but this is tiny compared with the multi-checkpoint switch headroom, and easy-image damage dominates.

E090 tested a plausible rescue: keep the E089 step250 selector movement, then polish by RD-only training with the beta005 `y_hat` anchor. It reduces the mean damage to `+0.000427`, but it also washes out the hard-tail gain. Q4 goes from E089 step250 `-0.000247` to essentially neutral `+0.000002`, while Q1-Q3 remain positive. This mirrors E087: RD-only training is safer, but it becomes conservative and does not recover selectivity.

The reliability alignment metrics explain the mechanism. E089 reliability is directionally aligned with the E088 score (`score/suppression corr` around `+0.61` to `+0.64`, and selected images get lower reliability), but the actual selected-unselected reliability gap is only about `-0.0009` to `-0.0028`. E090 increases score correlation to `+0.752`, yet the selected-unselected gap collapses to `-0.000100`. So the head is learning the sign of the selector, but not a strong enough or localized enough action.

Research implication: E088 remains strong evidence that HCG-RVQ has useful hyperprior-side reliability information, but E089/E090 show that scalar/image-mean reliability supervision is not the right implementation path. The next serious branch should expose an explicit selection mechanism or ranking/margin objective: keep beta005 by default, apply local-cap-like weak geometry only where a transfer-derived policy predicts benefit, and preserve `y_hat`, RVQ assignment/code usage, qMSE, `s_q`, and dead-code on low-risk regions. Beta005 remains the manuscript-safe fixed-checkpoint HCG-RVQ row.

Artifact: `experiments/analysis/e089_e088_selector_distill_audit.md`.

## E091/E092 Analysis: Distortion-Dominated Headroom Needs More Than Image-Level Reliability

E091 resolves a design question that mattered after E089/E090. The strong E088 switch is not mainly a rate-model artifact. Its gain is overwhelmingly distortion-side. On holdout4096, the E088 selected subset improves previous-local over beta005 by `-0.113561` RD, with only `-0.004229` from bpp and `-0.109332` from the distortion term. The same pattern appears on transfer8192. This makes a selected-image distortion/ranking objective a reasonable minimal experiment before adding per-sample index-rate supervision.

E092 tested that minimal experiment. It added a selected-image distortion margin against the beta005 anchor while keeping the beta005-initialized frozen-codec shell and a lower `y_hat` anchor. The result is a safe but negative probe. Mean RD improves relative to E089/E090 (`+0.000310` vs beta005), and intermediate features remain beta005-like with zero nonfinite rows, so the implementation is not unstable. But the selector headroom is still not distilled: E092 selected-subset delta is `-0.000029`, Q4 is `+0.000071`, and reliability alignment reverts to RD-only-like conservative behavior rather than a useful E088 selector.

The interpretation is now sharper. The problem is not that E088 headroom is rate-only, and not that selected distortion is irrelevant. The problem is capacity and action path: a single image-mean reliability multiplier has too little leverage or too much anchor pressure to create the previous-local-like benefit on selected high-risk cases. The next credible method-improvement step should introduce an explicit local selector/mixture or a local trainable action that can spend capacity in selected regions, while protecting beta005 assignments/code usage and feature distributions elsewhere.

Paper implication: beta005 remains the fixed-checkpoint main row. E088 and E091 are strong mechanistic evidence that hyperprior-generated conditioning contains reliability information and that the useful fallback behavior is distortion-driven. E092 becomes the guardrail showing that simply adding a distortion-margin loss to the existing image-level head is insufficient.

Artifact: `experiments/analysis/e091_selector_gain_decomposition.md`, `experiments/analysis/e092_e088sel_distmargin_yhatanchor25_headonly_from_beta005_seed3456_step250_fullimage_holdout4096_current.md`, and `experiments/analysis/e089_e088_selector_distill_audit.md`.

## E093/E094 Analysis: Capacity Alone Breaks Geometry; Scalar Range Is Too Blunt

E093 and E094 close the two most obvious explanations for why E092 did not distill the E088 transfer selector into one checkpoint. E093 says the bottleneck is not simply trainable capacity. When the conditioning geometry heads are allowed to move, the model can change the quantizer distribution, but it changes it in the wrong way: RD worsens by `+0.115637`, qMSE jumps from the beta005 regime around `0.112377` to `0.154673`, and `s_q_mean` drops from `0.456233` to `0.418462`. Because `nonfinite_rows=0`, this is a real distributional failure, not a numerical crash.

E094 says the bottleneck is also not just that the scalar reliability range was too narrow. Widening the reliability multiplier to `0.50..1.0` creates a little of the desired selectivity: the E088-selected subset improves by `-0.000909`, and Q4 improves by `-0.000223`. But this costs too much on easy and medium images, producing Q1/Q2/Q3 deltas of `+0.002253`, `+0.001809`, and `+0.001182`, so the global RD is still `+0.001255` worse than beta005. The intermediate features remain beta005-like, which means E094 is safe but underpowered and too global.

The combined interpretation is useful for the paper and the next implementation. E088/E091 show that hyperprior-side features contain a real, distortion-dominated reliability signal. E089/E092/E094 show that an image-mean reliability scalar can learn the direction but cannot localize action enough to beat beta005. E093 shows that broadly moving the geometry heads is too risky without a tighter parameterization. Therefore the next credible HCG-RVQ strengthening path is a bounded selector/mixture module: beta005 as the exact default, local-cap/weak-geometry behavior as a controlled branch, and transfer-derived selection/ranking supervision with explicit preservation of reconstruction, RVQ assignments, qMSE, `s_q`, and code usage on low-risk regions.

Paper implication: beta005 remains the fixed-checkpoint manuscript-safe HCG-RVQ prototype. E088, E091, E093, and E094 should be used to justify the proposed reliability-control design space and to explain why the next method branch must be explicit local selection rather than scalar reliability suppression or unconstrained conditioning-head adaptation.

Artifact: `experiments/analysis/e089_e088_selector_distill_audit.md`.

## E095 Analysis: Local BCE Helps Slightly, But Does Not Create Enough Action

E095 tested the most conservative local version of the E088 distillation idea: keep the E094 safe reliability range (`0.50..1.0`), but apply the teacher target locally using the transfer-derived high-delta threshold `0.045151`. This is closer to the desired selector behavior than the image-mean target, because low local-risk areas are explicitly kept near beta005 even inside an image selected by E088.

The result is informative but negative. E095 is stable and beta005-like in intermediate features (`qMSE=0.112387`, `s_q_mean=0.456233`, dead-code `0.030436`, `nonfinite_rows=0`), and it slightly improves E094 on mean RD (`+0.001208` vs `+0.001255`). However, the useful action is still too small: E088-selected images improve only `-0.000836`, and Q4 improves only `-0.000189`. Q1-Q3 remain worse by `+0.002156`, `+0.001730`, and `+0.001137`, which dominates the mean.

The key interpretation is that locality in the loss is not enough if the action path is still a single reliability multiplier on the Householder gate. It can learn directional suppression, but it cannot express a strong enough beta005/default versus local-cap/weak-geometry branch. This makes the next design step clearer: implement a bounded selector/mixture path with exact beta005 default and a controlled alternative geometry action, rather than continuing to tune scalar reliability BCE losses.

Paper implication: E095 strengthens the negative evidence around the scalar-head family and makes the future method claim cleaner. The manuscript-safe fixed-checkpoint result remains beta005; E088/E091 show the available selector headroom; E093/E095 show why the single-checkpoint controller needs an explicit selector/mixture parameterization.

Artifact: `experiments/analysis/e089_e088_selector_distill_audit.md`.


## E096/E097 Analysis: Exact-Default Selector Works, Precision Is the Bottleneck

E096/E097 answer the next question after E095: whether an explicit selector parameterization can preserve beta005 by construction and still recover part of the E088 transfer-selector headroom. The implementation side is successful. The residual selector has exact beta005 default, loads the beta005 checkpoint with only selector-head tensors missing, trains only 965 parameters, and preserves the core intermediate distribution. This is important because E093 showed that broadly unfreezing conditioning heads can break the geometry distribution, while E089-E095 showed that scalar reliability heads are too blunt.

The E096 evidence is encouraging but not sufficient for a paper-main claim. It is path-matched to beta005 on full-image holdout4096 and has zero nonfinite rows. Its mean RD is `+0.000271` worse than beta005, but the direction is right on the subsets that motivated the branch: E088-selected images improve `-0.000874`, the high beta005 delta-RMS tail improves `-0.001709`, and Q4 hard images improve `-0.000486`. Selector prob is also meaningfully aligned with the diagnostics: correlation `+0.622207` with E088 score and `+0.812003` with beta delta RMS. Because qMSE, `s_q`, dead-code, perplexity, and gate/raw geometry remain beta005-like, this is not collapse; it is a controlled but underpowered selective action.

E097 clarifies that simply making the selector stronger is not enough. Increasing max suppression and teacher strength improves the intended tail more than E096, but the easy quartiles deteriorate more as well. The mean becomes `+0.000471` worse than beta005 even though Q4 improves `-0.000711` and the strict delta-RMS tail improves `-0.002606`. This is a clean precision failure: the branch can spend useful action in hard regions, but it still spends too much small action where beta005 should remain untouched.

The research implication is now sharper. The exact-default residual selector should remain the base controller mechanism because it is safer than unfreezing geometry heads and more expressive than the old scalar reliability multiplier. But the next branch should not merely raise `rho` or `selector_max`. It needs a more selective objective or action path: pairwise/ranking supervision against beta005 vs previous-local outcomes, stronger no-op preservation on Q1-Q3/easy images, or a bounded alternative geometry branch that is only available under confident selection. Promotion should require mean RD below beta005, Q4/tail gain without Q1 damage, and unchanged qMSE/`s_q`/dead-code/code usage.

Paper implication: beta005 remains the current manuscript-safe fixed-checkpoint HCG-RVQ prototype. E088/E091 provide the large diagnostic headroom; E096/E097 provide an implementation path that can safely expose that control inside one checkpoint, but also define the next bottleneck: precision, not stability or raw action magnitude.

Artifacts: `experiments/analysis/e096_residual_selector_audit.md` and `experiments/analysis/e089_e088_selector_distill_audit.md`.

## E098 Analysis: No-Op Regularization Helps the Right Failure Mode, But Too Weakly

E098 directly tested the precision hypothesis from E096/E097. If E097 improves selected/tail images but hurts easy quartiles, then a local no-op loss on low householder_delta_rms_map regions should reduce the unwanted easy action. The implementation is conservative: it leaves the exact-default selector parameterization intact, trains only 965 selector-head parameters, uses the same transfer8192 E088 teacher labels, and evaluates on path-matched full-image holdout4096 with exact inverse mode on physical GPU0.

The result confirms the hypothesis only weakly. E098 is numerically clean (nonfinite_rows=0) and keeps the beta005-like feature distribution: qMSE 0.112384, s_q_mean=0.456233, dead-code 0.030464, delta RMS 0.039784, and strength 0.262350. Relative to E097, the low-risk no-op term reduces Q1/Q2/Q3 damage from +0.001293/+0.000893/+0.000410 to +0.001230/+0.000847/+0.000387. But the same regularizer also trims the useful action: E088-selected gain weakens from -0.001322 to -0.001256, and Q4 weakens from -0.000711 to -0.000679. Mean RD improves only from E097 +0.000471 to E098 +0.000446 versus beta005, still worse than E096 +0.000271 and worse than beta005.

This is a useful negative, not a failure of the broader line. It says the exact-default residual selector is stable and its easy-damage failure mode is real, but local low-delta no-op regularization alone cannot supply the missing precision. The next credible strengthening step should use outcome/ranking information or a controlled alternative geometry branch so that selected high-risk regions receive a more distinct action while low-risk regions remain exact-default.

Paper implication: E098 should not be promoted. It supports the claim that HCG-RVQ has a stable, bounded local-control primitive and that the remaining bottleneck is policy precision rather than numerical stability, VQ collapse, or lack of raw suppression magnitude.

Artifact: experiments/analysis/e089_e088_selector_distill_audit.md.


## E100-E105 Analysis: Dead-Zone Converts Selector Precision Into Mean RD Gain

The E096-E099 residual-selector family had the right mechanism but a precision problem: it improved selected or hard images while leaving small positive damage on easy images. The E100-E105 dead-zone sweep directly tests that bottleneck without retraining the checkpoint. Low selector probabilities become exact no-op, so any gain must come from removing low-confidence action while retaining high-confidence hard-tail action.

The result is the strongest single-checkpoint evidence in the branch so far. On seed3456 holdout4096, increasing the dead-zone from 0.010 to 0.018 moves the mean beta005 delta from `+0.000126` to `-0.000423`. At 0.018, the HCS difficulty quartiles are all non-positive: Q1 `-0.000018`, Q2 `-0.000200`, Q3 `-0.000481`, and Q4 `-0.000993`. The E088-selected subset also improves by `-0.001359`. This is exactly the desired pattern: keep easy images at beta005 behavior and retain extra local geometry control on hard images.

The intermediate-feature audit supports the interpretation that this is not collapse or a hidden distribution shift. E104 keeps qMSE around `0.112380`, `s_q_mean=0.456233`, dead-code around `0.030527`, and `nonfinite_rows=0`; delta RMS and Householder strength stay in the same beta005-like regime. The change is therefore best read as policy calibration of an existing local geometry controller, not as a new quantizer distribution.

The start8192 check is important for paper discipline. On that slice, 0.018 also performs best among the tested thresholds: dz014 `-0.000258`, dz016 `-0.000380`, dz018 `-0.000425`, and dz020 `-0.000380`. Because dz020 drops back on the calibration side, I did not spend another holdout run on it. This gives a defensible stopping point and avoids turning the holdout into a threshold-search target.

Paper implication: beta005 remains the safe fixed-checkpoint row until multi-seed confirmation, but E104/deadzone018 is now the most promising candidate for a stronger HCG-RVQ variant. The next required action is not another hand threshold on seed3456; it is to lock the threshold selection protocol, then rerun path-matched seed1234/2345/3456 and checkpoint step250/500 with the same feature-distribution and quartile audits. Promotion should require mean RD below beta005, Q1 near zero or better, Q4/selected gains retained, and unchanged qMSE, `s_q`, dead-code, perplexity, and nonfinite counts.

Artifact: `experiments/analysis/e089_e088_selector_distill_audit.md`.

## E106 Analysis: Dead-Zone Residual Selector Is Now Multi-Seed Positive

The important change from E100-E105 is reproducibility. E104/deadzone018 was originally promising on seed3456 because it converted the exact-default residual selector from a tail-only diagnostic into a mean RD gain. The new seed1234/2345 evaluations show that this was not just the fragile-seed rescue pattern. Under the same direct fixed-checkpoint protocol, all three seeds improve over beta005.

The 3-seed aggregate is now `2.171120` RD for E104 versus `2.173688` for beta005, a `-0.002568` mean gain over 12,288 full-image holdout4096 evaluations. The per-seed deltas are seed1234 `-0.002459`, seed2345 `-0.004822`, and seed3456 `-0.000423`. Seed3456 remains the weakest effect, but it is no longer the only positive seed; seed1234/2345 provide the stronger evidence that the dead-zone controller is a real method improvement in this protocol.

The image-level distribution is also healthy. E104 wins on `76.30%` of images overall, with median delta `-0.000895`. The q95 delta is only `+0.000591`, while q05 is `-0.011170`, so most residual damage is small and the beneficial tail is larger. Sorting by beta005 RD, all aggregate quartiles improve: easy Q1 `-0.001294`, Q2 `-0.002117`, Q3 `-0.002881`, and hard Q4 `-0.003980`. This fixes the earlier recurring problem where hard-tail gains came with easy-image damage.

The feature audit is the key reason this can be promoted as a controlled HCG-RVQ branch rather than dismissed as a lucky RD shift. qMSE changes by only `+0.000002`, `s_q_mean` is unchanged to numerical precision, dead-code ratio decreases by `-0.000025`, delta RMS decreases by `-0.000257`, and Householder strength decreases by `-0.000654`. The controller is therefore not co-adapting `s_q` or breaking RVQ code usage; it is using an exact-default gate to suppress low-confidence geometry action and preserve higher-confidence local geometry where it helps.

Paper implication: E104 is now the strongest candidate for the proposed single-checkpoint HCG-RVQ variant. The claim should still be phrased carefully: not yet final SOTA, but a reproducible fixed-checkpoint improvement over the beta005 HCG-RVQ prototype with stable intermediate distributions. The next decisive experiments are checkpoint step250/500 selection, an independent threshold-selection rule, and then broader comparisons to HCS/no-transform and SOTA-compatible plug-in settings.

Artifact: `experiments/analysis/e104_multiseed_deadzone018_audit.md`.

## E107 Analysis: Dead-Zone Gain Is Not Holdout-Only

E107 is important because E106 made E104/deadzone018 look strong on holdout4096, but a conference-facing claim also needs to avoid threshold overfitting. I therefore evaluated the same fixed checkpoints on the independent start8192 transfer split and compared the resulting gains against the holdout deltas.

The transfer result is almost perfectly aligned with holdout. The 3-seed transfer aggregate is beta005 RD `2.135355` versus E104 RD `2.132781`, delta `-0.002574`, while the E106 holdout delta is `-0.002568`. The difference is only `-0.000006`. Per seed, seed1234 is `-0.002485` on transfer versus `-0.002459` on holdout, seed2345 is `-0.004813` versus `-0.004822`, and seed3456 is `-0.000425` versus `-0.000423`. This is a much cleaner result than merely saying the same threshold was positive once on another split.

The feature-distribution story also transfers. qMSE changes only by `+0.000001` to `+0.000003`; `s_q_mean` is unchanged to numerical precision; dead-code, delta RMS, and Householder strength all decrease. This repeats the core interpretation: the dead-zone does not improve RD by destabilizing VQ or by co-adapting `s_q`; it improves by suppressing low-confidence residual-selector geometry action while preserving useful high-confidence local geometry.

Paper implication: E104/deadzone018 should now be treated as the main manuscript-candidate branch under the current MeanScaleHyperprior/RVQ prototype. The remaining blockers are procedural rather than mechanistic: pre-declare how the dead-zone threshold is selected, audit checkpoint step250/500 selection, and then move to broader comparisons and later SOTA plug-in. The result is still not a final SOTA claim, but it is now a defensible controlled-method claim.

Artifact: `experiments/analysis/e106_deadzone018_transfer_vs_holdout_audit.md`.

## E108/E109 Analysis: Transfer-Selected Dead-Zone Is Mean-RD Stronger, With a Tail-Safety Trade-Off

E108 and E109 close the main protocol gap left by E106/E107. E106 showed that deadzone018 improves the 3-seed holdout mean, and E107 showed that this gain transfers almost exactly to an independent start8192 split. The remaining question was whether the threshold itself was chosen by looking at holdout. E108 answers that by using start8192 as the threshold-selection split. Under that pre-declared transfer rule, dz014 is selected: dz014 `-0.002827`, dz016 `-0.002762`, dz018 `-0.002574`, and dz020 `-0.002253` versus beta005.

E109 then tests the selected dz014 on holdout4096 without changing the rule. The confirmation is positive: dz014 reaches RD `2.170857` versus beta005 `2.173688`, for a 3-seed gain of `-0.002830`. This is slightly stronger than dz018 (`2.171120`, delta `-0.002568`). On the same images, dz014 minus dz018 has mean `-0.000262`, median `-0.000147`, and dz014 is better on `62.7197%` of images. Therefore dz014 is not a holdout-only accident; it is the threshold independently preferred by transfer and confirmed by holdout.

The important nuance is that dz014 and dz018 optimize different risk profiles. Dz014 is better for mean RD and improves all difficulty quartiles more: Q1/Q2/Q3/Q4 are `-0.001489`/`-0.002351`/`-0.003231`/`-0.004250`, compared with dz018 `-0.001253`/`-0.002049`/`-0.002940`/`-0.004031`. However, dz018 has a higher win rate (`0.763021` vs `0.729329`) and lower q95 damage (`+0.000591` vs `+0.001343`). This means dz014 should be the mean-RD manuscript candidate, while dz018 should remain a conservative/safety ablation.

The feature-distribution evidence supports the HCG-RVQ interpretation. Both thresholds keep qMSE, `s_q_mean`, dead-code, delta RMS, and Householder strength in the same controlled regime, with zero nonfinite rows. Dz014's gain is not coming from moving `s_q` or collapsing the RVQ; it comes from a calibrated exact-default residual-selector policy that removes low-confidence geometry action while retaining useful local hyperprior-conditioned geometry.

Paper implication: the current strongest claim is now stronger than E106. We can say that an independently selected dead-zone reliability controller improves a fixed-checkpoint HCG-RVQ prototype over beta005 on two splits and three seeds, with stable intermediate feature distributions. The next blocker is no longer threshold overfitting; it is checkpoint/protocol wording. Because the current residual-selector configs are explicitly 250-step runs, the manuscript should either predeclare step250 as the controller-training budget or run a separate long-step checkpoint audit before presenting checkpoint selection as part of the method.

Artifacts: `experiments/analysis/e108_deadzone_transfer_threshold_selection_audit.md` and `experiments/analysis/e109_deadzone014_holdout_confirmation_audit.md`.

## E110 Analysis: Max500 Improves Mean RD, But Reopens Seed Fragility

E110 answers the checkpoint-budget question that remained after E108/E109. The dead-zone threshold protocol was already clean: dz014 was selected on start8192 and confirmed on holdout4096. What was not clean was whether the residual-selector checkpoint should be fixed at step250 or whether longer controller training should be used. I therefore trained the same exact-default residual-selector configs to max500 for all three seeds and evaluated dz014/dz018 on the same holdout4096 direct-probe protocol.

The mean-RD result is strong. Step250 dz014 had beta005 delta `-0.002830`, while max500 dz014 reaches `-0.007877`; step250 dz018 had `-0.002568`, while max500 dz018 reaches `-0.007901`. On the same images, max500 improves over step250 by `-0.005047` for dz014 and `-0.005333` for dz018. This says the residual-selector controller is not merely a short-training artifact: with more training, the learned local policy can extract substantially more average RD gain.

The reason this is not immediately paper-main is the risk profile. Max500's aggregate win rate drops to about `0.684`, and q95 damage increases to about `+0.0061` to `+0.0063`, compared with step250's much smaller q95 damage (`+0.001343` for dz014 and `+0.000591` for dz018). Per seed, max500 is very strong for seed1234 and seed2345 but fails the fragile seed3456 beta-reference test: dz014 is `+0.000619` and dz018 is `+0.000522` on seed3456. Thus max500 improves the mean by creating a stronger action path, but it sacrifices the clean all-seed-positive story that made step250 dz014/dz018 attractive.

The feature distribution makes the interpretation useful rather than confusing. Max500 does not collapse the RVQ: qMSE remains around `0.106832`, `s_q_mean` remains `0.463063`, dead-code is slightly lower (`0.031743`), and all runs have `nonfinite_rows=0`. The main behavioral change is policy strength: selector probability rises from the sparse step250 regime (`0.0046`-`0.0073`) to about `0.030`-`0.032`, and the residual-selector multiplier drops accordingly. This is controlled extra action, not numerical instability or `s_q` co-adaptation.

Paper implication: max500 should be treated as a high-mean checkpoint candidate and a useful strengthening direction, not yet as the final manuscript rule. The safest current narrative is to present step250 dz014 as the independently selected and holdout-confirmed threshold result, include dz018 as the conservative safety ablation, and use E110 to motivate either an independent checkpoint-selection rule or a stabilization experiment that preserves the max500 mean gain while removing seed3456 regression and q95 damage. This is exactly the kind of checkpoint evaluation the conference story needs: it shows where the method has more headroom, and also prevents accidental checkpoint cherry-picking.

Artifact: `experiments/analysis/e110_residual_selector_max500_checkpoint_audit.md`.

## E111 Analysis: Independent Transfer Confirms Max500 Headroom and Tail Risk

E111 checks whether the E110 max500 checkpoint result was a holdout-only checkpoint artifact. It was not. On the independent start8192 transfer split, max500 again improves the 3-seed mean by about `-0.005` RD relative to step250: dz014 moves from `-0.002827` to `-0.007890`, and dz018 moves from `-0.002574` to `-0.007913` versus the beta005 reference. This is strong evidence that longer residual-selector training exposes real additional HCG-RVQ headroom rather than merely fitting the holdout split.

The same caveat also repeats, which is just as important for a serious conference claim. Seed1234 and seed2345 benefit substantially, but seed3456 remains slightly worse than beta005 at max500 (`+0.000592` for dz014 and `+0.000496` for dz018). The aggregate q95 tail also stays larger than step250. Therefore max500 is not a clean all-seed-positive manuscript rule yet; it is a high-mean candidate whose risk must be handled by an independent checkpoint-selection rule or a stabilization experiment.

The feature distribution makes the diagnosis reliable. There are zero nonfinite rows, qMSE and `s_q_mean` remain in the same regime as step250, and dead-code is stable. So the max500 issue is not CUDA/device instability, VQ collapse, or `s_q` co-adaptation. It is a policy-strength trade-off: more selector action improves average RD, but also reopens fragile-seed and positive-tail damage.

Paper implication: step250 dz014 remains the cleanest independently selected threshold result, with dz018 as the conservative safety ablation. Max500 should be reported as a supported strengthening/checkpoint-budget direction across both holdout and transfer, but promoted to paper-main only if the checkpoint rule is predeclared or if the next stabilization run removes the seed3456 regression without losing the mean gain.

Artifact: `experiments/analysis/e111_residual_selector_max500_transfer_checkpoint_audit.md`.

## E112 Analysis: Selector Cap Reduces Max500 Fragility, But Does Not Solve It

E112 tests the most direct explanation for the E110/E111 caveat: max500 may have learned useful stronger action on average, but the residual-selector action may be too strong for the fragile seed3456 cases. This was tested without retraining by lowering the deploy-time selector cap for the max500 deadzone018 seed3456 checkpoint and choosing the cap on the independent start8192 transfer split before checking holdout4096.

The result supports the diagnosis but does not fully fix the branch. Lowering the cap monotonically reduces the transfer damage: cap0.50 is `+0.000496` versus beta005, cap0.45 is `+0.000413`, cap0.35 is `+0.000278`, and cap0.25 is `+0.000167`. The selected cap0.25 also reduces holdout damage to `+0.000178`. That is a real reduction of the seed3456 regression, but both splits remain slightly above beta005, so this is not enough to claim that max500 is now all-seed-positive.

The feature audit is clean. qMSE, `s_q_mean`, and dead-code remain essentially unchanged, and all rows have zero nonfinite values. Therefore the cap is not improving by changing the quantizer distribution; it is simply limiting the amount of selector-controlled geometry action. This strengthens the interpretation that max500's failure mode is action calibration/tail risk, not numerical instability or latent distribution collapse.

Paper implication: E112 makes the max500 story more credible as a future strengthening path, because the fragile-seed failure can be reduced with a simple, interpretable control knob. But it also protects the manuscript from overclaiming: selector-cap tuning alone does not make max500 paper-main. The safer main claim remains the independently selected step250 dz014/dz018 dead-zone controller, while max500 should be described as additional headroom that needs a predeclared checkpoint/cap rule or a training-time stabilization objective.

Artifact: `experiments/analysis/e112_max500_selector_cap_seed3456_probe.md`.

## E113 Analysis: Global Selector Cap Is Not the Max500 Promotion Rule

E113 asks whether the E112 seed3456 fix scales into a legitimate max500 rule. The protocol is intentionally stricter than a one-seed rescue: select one deploy-time selector cap on the independent start8192 transfer split across all three seeds, then report the corresponding holdout row. This prevents the paper from quietly optimizing cap for the fragile seed while throwing away the broader mean-RD gain.

The answer is clear. The transfer-selected cap is the original cap0.50, with aggregate delta `-0.007913`. Lower caps are monotonically worse in aggregate: cap0.45 is `-0.007220`, cap0.35 is `-0.005774`, and cap0.25 is `-0.004244`. The selected holdout row remains the E110 max500 deadzone018 result, delta `-0.007901` with zero nonfinite rows.

This does not invalidate E112; it refines it. Cap reduction really does reduce seed3456 damage, but it also suppresses useful selector action for seed1234 and seed2345. In other words, max500 problem is not that the whole branch should simply be weaker. The branch has real high-mean headroom, but the policy needs conditional reliability control: reduce action on fragile images or regions while keeping strong action where it is already beneficial.

The intermediate-feature evidence keeps the diagnosis clean. qMSE, `s_q_mean`, and dead-code remain stable across caps, and all rows have zero nonfinite values. Therefore the trade-off is not VQ collapse, CUDA/device instability, or `s_q` co-adaptation. It is a calibration problem in the residual-selector action policy.

Paper implication: max500 should not be promoted via a global selector-cap sweep. E113 is still valuable because it blocks an unsafe shortcut and points to the next credible strengthening direction: a predeclared learned/per-image cap, a training-time stabilization objective, or a checkpoint rule that preserves the cap0.50 mean gain while controlling seed3456/q95 damage. Until that exists, the paper-main branch remains step250 dz014, with dz018 as the conservative safety setting and max500 as supported headroom rather than the central claim.

Artifact: `experiments/analysis/e113_max500_selector_cap_multiseed_audit.md`.

## E114 Analysis: Conditional Cap Control Has Headroom, But Needs Learning

E114 follows directly from the E113 negative control. E113 showed that a global cap reduction cannot promote max500 because it helps seed3456 while weakening seed1234/2345. E114 asks the more important question: if cap were chosen per image, is there enough headroom to justify implementing a learned reliability/cap controller?

The answer is yes, but not enough for a hand-crafted rule. The per-image oracle over caps `0.25`, `0.35`, `0.45`, and `0.50` improves the max500 dz018 transfer mean from `-0.007913` to `-0.008545`, a further `-0.000632` RD. It also turns seed3456 from a slight regression (`+0.000496`) into a gain (`-0.000606`). This is exactly the desired qualitative behavior: keep the strong cap0.50 action for most images, but back off on fragile cases.

The single-threshold audit is the guardrail. In-sample, the best threshold gives a small extra gain (`-0.000150`), but leave-one-seed CV is slightly worse than cap0.50 by `+0.000020`. Therefore the current evidence does not support adding a manually tuned per-image threshold to the paper method. It supports a learned controller with a separate teacher/selection split, or a multi-feature calibrated policy whose selection protocol is fixed before holdout reporting.

The feature correlations explain why learning is plausible. Lower-cap benefit is most aligned with the existing residual-selector probability/multiplier statistics (`|r|` about `0.42`), followed by local Householder-delta statistics. This means the signal is present in the model diagnostics, but it is not cleanly separable by one scalar threshold across seeds.

Paper implication: E114 strengthens the max500 roadmap without changing the current main claim. The paper-safe result remains step250 dz014/dz018. Max500 should be described as validated high-mean headroom, and E114 motivates the next method-development step: a learned conditional reliability/cap controller trained and selected on an independent split, then confirmed on holdout with the same feature-distribution checks.

Artifact: `experiments/analysis/e114_max500_per_image_cap_headroom.md`.

## E115 Analysis: Learned Cap Selection Improves Slightly, But Not Enough Yet

E115 is the stricter follow-up to E114. E114 showed that per-image cap selection has meaningful oracle headroom, but a single scalar threshold does not generalize. E115 replaces the hand threshold with a small learned ridge selector over diagnostic features, using nested leave-one-seed CV so that feature set and regularization are not picked on the held seed.

The result is directionally positive but not decisive. Cap0.50 gives transfer delta `-0.007913`, the oracle gives `-0.008545`, and the learned nested-CV selector gives `-0.008009`. That is a real gain over cap0.50 (`-0.000096`) and better than the single-threshold CV from E114, but it leaves most oracle headroom unrecovered.

The seed breakdown explains the limitation. The learned selector mainly fixes the known fragile seed3456, moving it from `+0.000496` to `+0.000167`, but seed2345 becomes slightly worse by `+0.000044` and seed1234 is essentially unchanged. So the model has learned a recognizable reliability signal, but it still acts too bluntly: it backs off action where seed3456 needs help, yet cannot preserve all of the strong cap0.50 benefit on the already-good seeds.

Paper implication: E115 supports the claim that max500 failure is a conditional policy-calibration problem rather than collapse, and it gives a principled path for future method development. But it does not change the main manuscript result. A deployable learned cap controller should only be promoted after it recovers a larger share of the oracle headroom under held-seed or held-split validation and then confirms on holdout with the same nonfinite/qMSE/`s_q`/dead-code audits.

Artifact: `experiments/analysis/e115_max500_learned_cap_selector_cv.md`.

## E116 Analysis: Claim Boundary Is Now Conference-Shaped, But Not Final

E116 turns the accumulated experiments into a submission-readiness view. The main conclusion is that the research is genuinely going well for a serious international-conference path, but the claim should stay layered. The safe prototype claim is broad because `beta005 guard` is already supported on OpenImages holdout/transfer plus Kodak and CLIC mobile/professional. The newer dead-zone branch is more interesting and stronger, but until E117 it only had OpenImages-style evidence.

The important paper judgment is not to collapse all positive rows into one headline. `deadzone014` is the transfer-selected and holdout-confirmed mean-RD candidate, `deadzone018` is the conservative safety ablation, and max500 is high-mean headroom with a known seed3456/tail-risk caveat. This protects the manuscript from overclaiming while still making the method look alive: HCG-RVQ has a safe broad row, a stronger controlled row, and a clear next mechanism for improvement.

Paper implication: the current story is now coherent enough to start shaping tables and ablations. The remaining pre-SOTA work is to confirm the dead-zone branch on external splits, then freeze the prototype evidence before plugging the same HCG quantizer strategy into stronger LIC backbones.

Artifact: `experiments/analysis/e116_hcg_rvq_submission_readiness_package.md`.

## E117 Analysis: External Splits Promote the Dead-Zone Branch

E117 addresses the main gap identified by E116: whether the step250 dead-zone controller generalizes beyond OpenImages when compared against the already external-checked beta005 guard. The answer is yes. On Kodak, CLIC mobile valid, and CLIC professional valid, both dz014 and dz018 have negative aggregate RD deltas versus seed-matched beta005 references, with zero nonfinite rows.

This is a meaningful strengthening of the paper claim. Weighted over all external images, dz014 gives `-0.001650` RD and dz018 gives `-0.001473` versus beta005. The split-level results are also consistently negative: Kodak `-0.001115`/`-0.001033`, CLIC mobile `-0.001493`/`-0.001295`, and CLIC professional `-0.002196`/`-0.001995` for dz014/dz018. This means the dead-zone controller is no longer only an OpenImages improvement; it is now externally confirmed on the same small but standard-style splits used to support beta005.

The result also preserves the safety story. Dz014 is the mean-RD winner, but dz018 has the more conservative tail: all-external win rate `0.724868` and q95 `+0.000300`, versus dz014 win rate `0.671958` and q95 `+0.000930`. The remaining weakness is tiny seed3456 positivity for dz014 on all three external splits and for dz018 on Kodak only. These values are small, but they justify keeping dz018 as a safety ablation rather than pretending dz014 strictly dominates every reliability view.

The intermediate-feature audit supports a controlled-method interpretation. qMSE, `s_q_mean`, dead-code, and nonfinite counts stay stable across the external runs. So the gain is not coming from a broken evaluation path or numerical instability. It is the intended residual-selector geometry control producing a small but repeatable RD improvement over the beta005 guard.

Paper implication: the main prototype table can now be upgraded. Dz014 should become the stronger external-confirmed HCG-RVQ branch, dz018 should remain the conservative safety ablation, beta005 should be described as the earlier broad guard baseline, and max500 should remain a headroom branch. This is a much healthier conference posture than before E117: the novelty claim now has both protocol-clean OpenImages evidence and external split confirmation.

Artifact: `experiments/analysis/e117_deadzone_external_fixed_protocol_audit.md`.

## E118 Analysis: The Prototype Table Can Now Be Frozen

E118 converts the E117 result into a manuscript table. This matters because the project no longer needs to speak vaguely about many positive variants: the current prototype evidence has a natural order. Dz014 is the main mean-RD branch, dz018 is the lower-tail-risk ablation, beta005 is the broad historical guard baseline, and max500 is high-mean headroom that is not yet safe enough for the main row.

The key table result is that both dz014 and dz018 improve over beta005 on all five reporting splits: OpenImages transfer start8192, OpenImages holdout4096, Kodak, CLIC mobile valid, and CLIC professional valid. Dz014 has mean split delta `-0.002092` and worst split delta `-0.001115`; dz018 has mean split delta `-0.001893` and worst split delta `-0.001033`. Both have zero nonfinite rows.

The remaining nuance is useful rather than damaging. Dz014 wins the mean-RD headline, while dz018 is safer: dz018 has higher mean win rate (`0.726444` vs `0.674802`) and lower worst q95 damage (`+0.000641` vs `+0.001539`). This gives the paper a stronger and more honest ablation story: the proposed residual-selector geometry policy can be tuned either toward mean RD or conservative tail reliability.

Paper implication: this is now a credible prototype-level international-conference story, provided the manuscript is careful about scope. It is not yet a SOTA claim, and it still needs stronger-backbone plug-in comparisons. But the core HCG-RVQ claim is no longer only a same-dataset or single-split observation; it has a frozen, externally confirmed table with checkpoint and feature-distribution audits behind it.

Artifact: `experiments/analysis/e118_hcg_rvq_prototype_main_table_package.md`.

## E119 Analysis: SOTA Plug-In Should Wait for an Adapter Boundary

E119 is a planning audit, but it changes the research priority in an important way. After E118, the prototype claim is strong enough that the project can start preparing SOTA/backbone comparisons. However, doing a large external integration immediately would be risky because the HCG quantizer logic is still embedded inside `HCGMeanScaleHyperprior._conditioned_rvq`.

The correct next method step is therefore not to jump straight to DCAE/MambaIC/HPCM. It is to define and smoke-test an HCG quantizer adapter: inputs `y`, `hyper_features`, and `image_hw`; outputs `y_hat`, indices, commitment loss, RVQ stats, and conditioning tensors. Once this boundary is bit/RD-equivalent to the current model on a known checkpoint, it can be attached to stronger backbones without muddying the novelty claim.

The local CompressAI path is the safest bridge. CompressAI is already installed and exposes stronger candidates such as `JointAutoregressiveHierarchicalPriors`, `mbt2018_mean`, and `Cheng2020Attention`. These are not the final SOTA comparison, but they are the lowest-risk proof that HCG-RVQ is a plug-in quantizer strategy rather than a one-off MeanScaleHyperprior artifact.

Paper implication: the project should now proceed on two lanes. The paper lane builds an explicit component ablation table and later multi-rate curves. The method lane extracts the adapter, then runs a local strong-backbone smoke, then considers official external repos. This is exactly the safer order for a serious international-conference submission.

Artifact: `experiments/analysis/e119_sota_plugin_readiness_audit.md`.

## E121 Analysis: Component Ablation Is Now Readable

E121 turns the existing evidence into a component table aligned with the prompt. The result is useful because it separates the method story into increasingly specific pieces: HCS-RVQ gives the shift/scale + index entropy baseline, old gate0.25 adds raw Householder geometry, min090 adds conservative risk control, beta005 adds the broad stabilization guard, and dz014/dz018 add the residual-selector dead-zone reliability controller.

The summary supports the current main claim. Raw geometry already improves HCS (`-0.023766` mean split delta), but the stabilized/reliability-controlled HCG variants are much stronger. Beta005 reaches `-0.099495` versus HCS, dz014 reaches `-0.101587`, and dz018 reaches `-0.101388`. Dz014 is the best mean component, while dz018 remains the safer tail ablation from E117/E118.

This also keeps the paper honest. The current table is not a complete HVQ-CGIC-style ablation, because there is no pure entropy-only final row; HCS includes shift/scale plus index entropy. Multi-rate curves and strong-backbone plug-ins are also still missing. These are not contradictions, but they should be explicit in the manuscript plan.

Paper implication: the prototype ablation section is now strong enough to draft. The missing pieces are well defined: entropy-only if time allows, multi-rate before final submission, and adapter-based strong-backbone evidence after E120.

Artifact: `experiments/analysis/e121_component_ablation_table.md`.

## E120 Analysis: Adapter Boundary Is Numerically Safe to Extract

E120 closes the main engineering risk raised by E119. The HCG quantizer logic is still embedded in `HCGMeanScaleHyperprior._conditioned_rvq`, so before plugging the method into stronger CompressAI or SOTA-style backbones, we need to know that the quantizer boundary can be called independently without changing the model behavior.

The smoke test gives that evidence. A manual route through the intended adapter boundary reproduces the full forward path within a strict `1e-4` tolerance on GPU0. The largest image-space difference is only `4.87e-05`, latent difference is `3.20e-05`, index-rate difference is `1.49e-08`, and all loss/stat differences are near zero. There are zero nonfinite values.

The exact-zero flag intentionally remains false, because GPU repeated-convolution order produces tiny numerical differences even when the graph is functionally equivalent. That is useful diagnostic information, not a failure. For the next phase, the paper-relevant contract should be tolerance-based: if adapter extraction preserves E120 within tolerance, any later strong-backbone result can be attributed to the same HCG-RVQ quantizer mechanism rather than a hidden rewrite.

Paper implication: the project can now move from planning to implementation for adapter extraction. This directly supports the prompt goal because it prepares HCG-RVQ to become a portable quantizer strategy, while E118/E121 keep the current prototype claim and ablation story intact.

Artifact: `experiments/analysis/e120_hcg_adapter_contract_smoke.md`.

## E122 Analysis: Adapter Extraction Preserves the Prototype Claim

E122 turns the E119/E120 plan into code. The important constraint was checkpoint compatibility: if adapter extraction changed parameter ownership or state-dict names, the existing E118/E121 evidence could become hard to interpret. The extraction therefore keeps all parameters on `HCGMeanScaleHyperprior` and moves only the quantizer-boundary procedure into `hcg_rvq/quantizers/hcg_adapter.py`.

The validation result is clean. The extracted model loads the dz014 seed1234 step250 checkpoint with no missing or unexpected keys, and the E120 forward/manual contract still passes under `1e-4` tolerance on GPU0 with zero nonfinite values. The remaining exact-zero failure is the same harmless repeated-GPU-kernel numerical difference already documented in E120, not a semantic change.

Paper implication: this reduces a real integration risk. Strong-backbone experiments can now reuse the same quantizer boundary rather than copy-pasting the full MeanScaleHyperprior method. That helps attribution: later improvements or failures can be tied to the HCG quantizer strategy itself, while the current prototype claim remains anchored by the frozen E118 main table and E121 component ablation.

Artifact: `experiments/analysis/e122_hcg_adapter_extraction_smoke.md`.

## E123 Analysis: Strong-Backbone Plug-In Needs Flexible Hyper Channels

E123 checks the first local strong-backbone targets after adapter extraction. The good news is that local CompressAI backbones have the right structural boundary: g_a creates a latent y, h_a and entropy bottleneck create z_hat, h_s creates hyper-conditioned features, and g_s can decode a latent-shaped tensor.

The important implementation catch is channel semantics. In the current HCG-RVQ prototype, h_s emits an N-channel feature map that the quantizer heads consume. In CompressAI mbt2018_mean and cheng2020_attn, h_s emits Gaussian-parameter channels: 384 for a 192-channel latent and 256 for a 128-channel latent. So a direct copy of the prototype heads would silently assume the wrong feature dimension.

Paper implication: this is exactly why E122 was needed before SOTA work. The next implementation should introduce a configurable HCG adapter module with latent_channels and hyper_channels, then test it first with mbt2018_mean. If that smoke is clean, cheng2020_attn adds context-model complexity as the next step. External DCAE/MambaIC/HPCM repos should still wait until this local plug-in contract is stable.

Artifact: experiments/analysis/e123_local_compressai_backbone_contract_audit.md.

## E124 Analysis: Local Strong-Backbone Plug-In Is Executable, Not Yet a Quality Claim

E124 validates the immediate implementation question raised by E123. The standalone `HCGQuantizerAdapter` can consume CompressAI hyper-decoder outputs with explicit `hyper_channels`, produce latent-shaped `y_hat`, and feed the original CompressAI decoder `g_s` without shape or finite-value failures.

The smoke result is clean on GPU0. `mbt2018_mean` and `cheng2020_attn` both pass with zero nonfinite values. The conditioning heads start in the intended identity regime: zero shift, unit scale, and zero Householder delta. This matters because it means the adapter can be inserted without an immediate geometric perturbation before training, which is the right starting point for a controlled plug-in experiment.

The limitation is equally important. This is not an RD result and should not be reported as method quality. The adapter codebooks are random and the test uses one random 64x64 input. The high dead-code ratios are expected under that smoke setting and do not diagnose training collapse. The value of E124 is architectural: it turns strong-backbone integration from a speculative plan into an executable local experiment.

Paper implication: the project now has two active lanes. The paper lane remains E118/E121 for the prototype claim and ablation. The method-strengthening lane can now run a small `mbt2018_mean` adapter-training pilot, selected and analyzed with the same checkpoint, RD, nonfinite, qMSE, scale, Householder-delta, and codebook-usage audits used for the prototype.

Artifact: `experiments/analysis/e124_local_compressai_hcg_adapter_smoke.md`.

## E125 Analysis: The NaN Was Loss Plumbing; Geometry Needs Explicit Activation

E125 is important because it corrects an overly pessimistic interpretation of the first local strong-backbone training pilot. The old direct `hcg_rvq_h` run did produce NaNs, but the pattern was not a CUDA device-1 issue and not strong evidence that HCG geometry is inherently unstable. It used physical GPU0, the first forward row was finite, and the first failure signal was `grad_norm=NaN` before the model outputs became nonfinite.

The code audit identified a concrete mechanism: zero-weight conditioning losses were still connected to the graph. In particular, `householder_delta_rms` is a square-root statistic that is exactly zero at identity initialization. Even with coefficient zero, keeping that `sqrt(0)` path in the backward graph can poison gradients. Guarding zero coefficients in `RateDistortionLoss` removes this failure mode without changing any nonzero regularizer behavior.

The recheck is clean. With the loss guard, direct `hcg_rvq_h` completes 30 steps on GPU0 with zero nonfinite outputs, zero nonfinite gradients, and no skipped optimizer steps. This means the previous NaN should be documented as an implementation bug, not as evidence against HCG-RVQ. The same diagnostic also validates the new gradient/nonfinite guard in the E125 harness.

The remaining method issue is different: geometry is stable but inactive. In the loss-guarded direct HCG run, `householder_delta_rms=0` and `householder_v_abs_mean=0` through step 30. In other words, the adapter learns the HCS/RVQ path, but zero-initialized Householder directions do not naturally activate. This is a useful design result, because it says the geometry path needs a controlled activation mechanism rather than a blind full-Householder start.

The safe activation pilot gives that mechanism. Small Householder gate plus nonzero direction bias trains finitely and produces active geometry (`householder_delta_rms` about `2.60e-04` at step 30). Its RD is worse than the inactive/HCS branch in this random-backbone smoke, so it should not enter the paper as a quality row. But it is the right engineering bridge for the next strong-backbone experiment: first make the adapter stable, then turn on geometry gently, then evaluate whether learned geometry helps after meaningful warmup.

Paper implication: E125 strengthens the SOTA plug-in plan without changing the current prototype claim. The E118/E121 MeanScaleHyperprior evidence remains the paper-facing result. E125 adds a clean implementation lesson for the next lane: use loss guards, checkpoint/gradient audits, HCS warmup, and gated geometry activation before making any strong-backbone or SOTA claims.

Artifact: `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_summary.md`.

## E126 Analysis: Staged Geometry Is Viable, Usage Control Is the Next Bottleneck

E126 directly tests the staged plan implied by E125. Instead of comparing scratch HCS and scratch gated-HCG runs, it loads the actual HCS warmup checkpoint and then activates Householder geometry with a small post-load direction/gate reset. This matters because the prompt's claim is not just that geometry can exist, but that hyperprior-conditioned geometry can be introduced as a controlled quantizer mechanism.

The result is encouraging in the narrow trainability sense. The staged HCS30 -> gated HCG run is fully finite on GPU0, loads the adapter checkpoint without key mismatches, activates nonzero geometry, and slightly improves the eval smoke RD from `38.172090` to `38.171590`. qMSE also improves from `0.000488` to `0.000437`. This is the first local strong-backbone-lane result where the staged geometry path is both active and connected to a previous stable checkpoint.

The important caveat is codebook usage. The staged run improves qMSE and RD, but dead-code ratio rises from `0.506836` to `0.579102`, and the half-amplitude variant still raises dead-code to `0.564453`. That means the improvement may partly come from narrowing the effective index set, which would be dangerous for a paper claim if it later hurts entropy/index diversity or tail images.

The half-amplitude run is especially useful because it rules out a simple explanation. If the dead-code issue were only caused by too-large geometry, halving the gate/bias should have fixed it. It did not: qMSE improved, geometry stayed active, but RD stopped improving and dead-code still worsened. So the next lever should be usage-aware reliability control, not just smaller Householder strength.

Paper implication: E126 strengthens the engineering route toward SOTA/backbone plug-in, but does not change the main evidence table. It says the staged method is plausible and stable; it also identifies the exact next analysis/implementation target needed before claiming active geometry on stronger backbones: checkpoint-level RD plus feature distribution plus index-usage protection.

Artifact: `experiments/analysis/e125_mbt2018_hcg_adapter_trainability_summary.md`.

## E127 Analysis: Corrected Per-Image Audit Shows Real Staged Geometry Signal

E127 is the first per-image audit of the local strong-backbone staged geometry path. It also caught a methodological issue that matters for every future local CompressAI pilot: because the frozen backbone is randomly initialized and not saved inside the adapter checkpoint, all compared models must be rebuilt with the same model seed. Without that, checkpoint comparisons mix adapter behavior with different random backbones. The corrected E127 audit resets the model seed before every case and should be the protocol template going forward.

With that correction, the staged gate `0.01` checkpoint is genuinely better than the HCS warmup checkpoint on the 8-image Kodak smoke: mean RD improves by `-0.000496`, every image wins, and q95 damage is zero. qMSE also improves, and Householder geometry is active. This supports the staged route identified in E126: HCS warmup first, then small gated geometry.

The result is not yet paper-quality evidence because the backbone is still a random frozen local `mbt2018_mean`, not a trained/pretrained compression model. But it is a meaningful engineering/research signal: active hyperprior-conditioned geometry can improve the local adapter objective after warmup when evaluated under a path-matched protocol.

The limiting factor is now precise. Geometry improves RD and qMSE, but it reduces index diversity: dead-code rises by about `+0.073` and perplexity drops by about `-6.28`. The half-gate run shows that lower amplitude does not automatically fix this; it still worsens dead-code and loses mean RD. So the next mechanism should not just be smaller geometry. It should be usage-aware geometry control.

The selector analysis gives a concrete direction. If HCG is only allowed when it wins RD and keeps dead-code delta under `0.05`, 3/8 images can still use geometry and the mean RD improves by `-0.000083` with much smaller dead-code penalty. This is small, but it is the right kind of controllable trade-off for a conference paper: geometry has image-dependent value, and a reliability/usage controller can preserve safe applications.

Paper implication: the project remains on track at the prototype and method-development level. The current publishable core is still E118/E121. E127 strengthens the future SOTA/plug-in lane by showing the next missing ingredient exactly: usage-aware gating or a differentiable proxy for codebook diversity.

Artifact: `experiments/analysis/e127_staged_geometry_per_image_audit.md`.

## E128 Analysis: Usage-Aware Geometry Control Needs Candidate-Side Signals

E128 asks whether the E127 staged-geometry gain can be made safer by a simple usage-aware selector. This is important because E127 showed a real full-gate RD gain (`-0.000496`, 8/8 wins) but also a clear usage cost (`+0.073242` dead-code ratio, `-6.28` perplexity). A paper-safe controller should not merely improve the mean; it should limit per-image index-usage damage.

The answer is encouraging but not finished. For `staged_gate001`, strict per-selected dead-code caps still leave positive RD headroom. With cap `0.05`, candidate-forward `hcg_latent_quant_mse` selects 1/8 image and improves `-0.000073` with selected max dead-code increase `0.039063`; the posthoc diagnostic upper bound selects 3/8 and improves `-0.000083`. With cap `0.075`, simple latent-qMSE rules select 2/8 and improve `-0.000165`, while the posthoc upper bound selects 4/8 and improves `-0.000175`. This means the safe subset is small but real.

The half-gate alternative is weaker under strict safety. Although it can look competitive under mean dead-code budgets, the strict-cap table shows that at cap `0.025`/`0.05` it mostly selects cases that worsen RD. It only becomes useful at cap `0.075`, and even then its gain (`-0.000096`) is below the full-gate branch. This argues against making smaller geometry amplitude the main next mechanism.

The feature story points to candidate-side usage control. Candidate HCG perplexity, stage entropy, dead-code, and latent qMSE carry stronger signal than baseline-only features on this smoke split. Baseline-only rules can sometimes select safe cases, but they are too small-sample and unstable for a manuscript method. A learned reliability head or deterministic candidate-forward guard is a more credible next step, as long as it is selected on an independent split and evaluated on holdout.

Paper implication: E128 does not change the current paper-main evidence table. It strengthens the roadmap from E127: active geometry is useful, usage damage is controllable on a subset, and the next strong-backbone experiment should be a protocol-clean usage-aware controller rather than another hand-tuned gate amplitude.

Artifact: `experiments/analysis/e128_usage_aware_gate_feature_audit.md`.

## E129 Analysis: Full Kodak Confirms Mean Gain and Exposes the Real Controller Requirement

E129 expands the staged-geometry audit from an 8-image smoke to the full 24-image Kodak split. This was the right next check because E127's 8/8 win was encouraging but too small to decide whether staged geometry is generally reliable.

The result is substantially more informative. Full-gate staged geometry (`staged_gate001_step30`) still wins on average: mean RD improves by `-0.000325`, win rate is `0.791667`, nonfinite rows remain zero, qMSE improves, and Householder geometry is active. But the larger split reveals nonzero tail damage (`q95=0.000569`, max `0.000626`) and a consistent usage cost (`+0.068034` dead-code, `-4.143783` perplexity). So the true statement is not “geometry always helps”; it is “geometry has a real average signal and a controllable subset, but usage/tail gating is necessary.”

The half-gate checkpoint is no longer a plausible main branch. It worsens mean RD by `+0.000252` despite improving qMSE and still damages codebook usage. This is exactly the failure pattern we wanted to detect: smaller geometry can lower latent quantization error while hurting the RD objective and tail images.

The selector headroom is the most paper-relevant part of E129. A usage-safe oracle at dead-code cap `0.05` selects 6/24 images and keeps a small RD gain (`-0.000098`) with low mean usage damage (`+0.010417`). At cap `0.075`, it selects 13/24 and improves `-0.000247`; at cap `0.10`, it selects 16/24 and improves `-0.000298`. This proves that the problem is not lack of useful geometry, but choosing where to apply it.

The 24-image feature audit sharpens the implementation path. Candidate-forward mean-budget rules are promising: `hcg_latent_quant_mse` under a mean dead budget of `0.05` selects 16/24 images and improves about `-0.000342`, and leave-one-image checking remains directionally positive. However, strict per-selected dead-code caps are much harder: at cap `0.10`, candidate-forward latent qMSE selects 8/24 and improves `-0.000178`; at cap `0.05`, the simple candidate-forward rule shrinks to 1/24 with negligible gain.

Paper implication: E129 improves confidence in the strong-backbone plug-in lane, but it also prevents overclaiming. The next publishable method upgrade should be a learned or protocol-clean usage-aware controller selected on an independent split, then confirmed on holdout. For the current manuscript structure, E118/E121 stay as the main prototype evidence, and E129 becomes the motivation for the next controlled plug-in experiment.

Artifact: `experiments/analysis/e129_staged_geometry_kodak24_audit.md` and `experiments/analysis/e129_usage_aware_gate_feature_audit.md`.

## E130 Analysis: Split Selection Makes the Usage Controller Direction Credible

E130 is the protocol check that E129 implied. E129 showed that active staged geometry has average gain but usage/tail risk, and that same-set feature thresholds can find useful subsets. E130 asks the stricter question: if the threshold is selected on one subset of Kodak images, does it still help on a disjoint subset?

The answer is yes for the full gate, and no for the half gate. `staged_gate001` candidate-forward policies win all four split protocols under mean dead budgets. The budget `0.05` result is the cleanest near-term target: it selects about `8.25/12` held-out images and improves RD by `-0.000318`, with mean dead-code delta `+0.049967`. At budgets `0.075`/`0.10`, the gain grows to `-0.000370`. This means the signal is not merely a same-set threshold artifact.

The stricter safety view is smaller but still useful. With strict selected-dead cap `0.075`, candidate-forward `staged_gate001` improves `-0.000096` and wins all four protocols while selecting `2.5/12` held-out images on average. With cap `0.10`, it improves `-0.000116` and selects `4.25/12`. This is exactly the trade-off expected from a reliability controller: conservative settings recover less mean RD but give a safer subset.

The half-gate result is a negative control. `staged_gate0005` does not generalize under mean-budget selection and is mostly no-op or unstable under strict caps. This is valuable because it prevents the project from spending more cycles on the appealing but wrong fix of simply shrinking geometry amplitude.

Paper implication: E130 strengthens the strong-backbone plug-in story without overclaiming. The result still comes from a random/frozen local `mbt2018_mean` adapter lane, not from a trained SOTA backbone. But it gives a concrete next implementation target: a candidate-forward usage/reliability controller for full-gate HCG geometry, selected on an independent split and confirmed on holdout with checkpoint, qMSE, dead-code, perplexity, and nonfinite audits.

Artifact: `experiments/analysis/e130_usage_controller_split_protocol.md`.

## E131 Analysis: Decision Package Turns Usage Control Into a Concrete Implementation Target

E131 is not a new experiment; it is the decision layer over E129/E130. That matters because the recent strong-backbone lane had several superficially plausible choices: half-gate geometry, full-gate geometry with oracle safety, single-feature guards, and a possible learned reliability head. The decision package makes the priority explicit.

The main conclusion is that the full `staged_gate001` branch should remain the active branch. It is the only one with both a positive checkpoint result on full Kodak and a split-protocol controller path. The recommended default is a candidate-forward expected-usage controller with mean dead-code budget `0.05`: it wins `4/4` split protocols, selects `8.25/12` held-out images, improves RD by `-0.000318`, and keeps mean dead-code delta at `+0.049967`. This is close to the desired trade-off: most of the full-gate gain, with usage cost explicitly budgeted.

The package also clarifies how to present ablations. The `0.075` mean-budget controller gives a stronger mean RD gain (`-0.000370`) and should be kept as a mean-performance ablation. The strict cap `0.075` controller is smaller but safer (`-0.000096`, `2.5/12` selected, mean dead `+0.017253`) and should be used to show that the gain is not only an unconstrained usage trade.

The feature-stability result prevents overclaiming. Different split protocols select different candidate-forward signals, mostly `hcg_dead_code_ratio`, `hcg_householder_delta_rms`, and `hcg_latent_quant_mse`. So the paper should not claim that one scalar explains reliability. The stronger claim is that candidate-side geometry/usage statistics contain enough signal to control where HCG geometry should be applied.

Paper implication: the strong-backbone lane now has a disciplined next implementation target. E118/E121 remain the current paper-main prototype evidence, but E131 gives the bridge experiment needed to make the SOTA plug-in story credible later.

Artifact: `experiments/analysis/e131_usage_controller_decision_package.md`.

## E132 Analysis: Teacher Labels Are Plausible But Safety Labels Are Scarce

E132 converts the E129 per-image audit into labels that can train or validate a reliability controller. This is important because E130's one-feature split rules are useful, but the final method should ideally have a controller that can be learned on independent teacher data and then frozen for holdout evaluation.

The label counts are encouraging but small. `rd_win` has `19/24` positives, so broad HCG usefulness is common on Kodak. Safety-labeled positives are more selective: `6/24` at dead cap `0.05`, `12/24` at `0.075`, and `15/24` at `0.10`. This supports using `safe_win_dead_le_0.075` as the first trainable-controller target: it is strict enough to reduce usage damage but balanced enough to learn from.

The feature separation is consistent with the mechanism story. Winning images tend to have lower candidate `hcg_householder_delta_rms` and lower `hcg_latent_quant_mse`; safety-positive images also separate on candidate entropy/perplexity and dead-code signals. Baseline-only difficulty signals also matter, but the candidate-forward statistics are more directly tied to the quantizer geometry that the prompt asks us to control.

Paper implication: E132 does not create a new result row. It prepares a cleaner next experiment: train or tune a usage/reliability controller on one split, then evaluate checkpoint RD, qMSE, dead-code, perplexity, and tail damage on a held-out split.

Artifact: `experiments/analysis/e132_usage_controller_teacher_labels.md`.

## E133 Analysis: Learned Controller Signal Exists, But Deterministic Guard Is Still Safer As The Default

E133 tests whether a tiny supervised controller can use E132 labels under the same split discipline as E130. The result is useful because it separates two questions: whether a learned head has upside, and whether it is ready to replace the simpler deterministic guard.

There is a real learned signal. A compact candidate-feature logistic probe trained on `rd_win` improves held-out RD by `-0.000342` at mean budget `0.075`, winning all four protocols. At budget `0.05`, the same family improves `-0.000324`, also winning all four protocols. These numbers are in the same range as the E131/E130 single-feature controller, so the idea of a trainable reliability head is not speculative.

The safety story is not finished. The budget `0.05` learned probe slightly exceeds the held-out mean dead-code target (`+0.051432`), and strict-cap probes can improve RD while still violating the held-out max selected dead-code cap. For example, the combined `safe_win_dead_le_0.075` strict-cap `0.075` probe improves `-0.000183` but has held-out mean max selected dead `0.085938`. That is a promising coverage increase, but not yet a strict guarantee.

The practical decision is therefore conservative. Use E131's candidate-forward deterministic guard at mean budget `0.05` as the next implementation default, because it is easier to reason about and already protocol-clean. Develop the E133 learned reliability head in parallel as the higher-upside route, but only promote it after it is trained on independent teacher labels and passes held-out usage/tail checks.

Paper implication: this is a good sign for the future method strength, but it should not weaken the current paper discipline. The manuscript should present the controlled geometry story first, then use learned reliability as an extension if the independent split confirms it.

Artifact: `experiments/analysis/e133_usage_controller_supervised_probe.md`.

## E134 Analysis: Cross-Fit Guard Is Reproducible, But It Is Expected-Usage Control

E134 takes the E130 split-selected policies and expands them into per-image decisions. This matters because a conference paper cannot rely only on averaged threshold tables; it needs to show that the controller is reproducible, that the selected images can be inspected, and that the feature distributions explain the trade-off.

The positive result is clean. The main candidate-forward mean-budget guard at `0.05` reproduces the E130/E131 summary exactly: `4/4` protocols win, mean selected count is `8.25/12`, mean RD improves `-0.000318`, mean dead-code delta is `+0.049967`, and q95 damage is `0.000113`. That is a credible controller prototype for the strong-backbone lane.

The limitation is equally important. The selected max dead-code delta is `0.128906`, and only about `54%` of selected decisions are positive under the `safe_win_dead_le_0.075` teacher. This means the default guard controls expected usage cost, not individual-image usage safety. The strict-cap variants reduce average dead-code damage but are still train-split constraints; their held-out selected max dead can exceed the nominal cap. So the paper should not use language like hard safety guarantee yet.

The feature contrast gives a mechanism. Selected decisions have substantially lower HCG latent qMSE (`0.000366` vs `0.000574`) and lower Householder delta RMS (`0.000265` vs `0.000365`) than rejected decisions. At the same time, selected decisions show higher HCG dead-code ratio and lower perplexity. In plain terms, the guard is finding cases where geometry gives stable RD/qMSE improvement despite narrower code usage. That is useful, but the next method improvement must decouple geometry benefit from codebook-usage narrowing.

Vote stability is not perfect: for the main guard, 13/24 images are selected by both held-out policies, 7/24 by one, and 4/24 by neither. This supports the broader E131/E133 conclusion: deterministic one-feature guards are good enough for a reproducible baseline, but the final method should likely use a small multi-feature or learned reliability controller trained on independent teacher labels.

Paper implication: E134 strengthens the controller story by making it auditable. It also protects against overclaiming. The safe claim is now: hyperprior-conditioned geometry has measurable average benefit, and candidate-side geometry/usage statistics can control expected usage cost under split protocol. The remaining gap for a strong international-conference submission is a decoder-reproducible controller that preserves this gain while reducing per-image usage tails.

Artifact: `experiments/analysis/e134_usage_guard_crossfit_package.md`.


## E135 Analysis: The Strong Guard Needs Either Signaling Or Proxy Distillation

E135 resolves an important deployability ambiguity in the usage-controller story. E130/E134 showed that candidate-forward statistics can control expected codebook-usage damage while preserving RD gain, but those statistics are not all equally available to the decoder before a geometry-use decision. A controller that uses latent quantization error, Householder displacement, dead-code ratio, or stage entropy is a valid diagnostic or teacher signal, but it is not automatically a no-side-bit decoder rule.

The best no-side-bit evidence is the `hyper_preindex` tier. It uses only hyperprior-side geometry summaries (`s_q`, `mu_q`, Householder direction magnitude) and wins all four split protocols at mean dead budget `0.05`, with RD delta `-0.000116`, selected count `3.00/12`, mean dead delta `+0.016276`, and q95 damage `0.000041`. This is valuable because it is plausibly decoder-reproducible before index decisions. However, it is conservative and recovers much less gain than the full candidate-forward controller.

The stronger controller evidence comes from features that need an implementation mechanism. At budget `0.05`, `all_candidate_forward` gives `-0.000318` RD, and `encoder_candidate_error` gives `-0.000304` with lower q95 damage (`0.000042`). Those are the numbers that make the strong-backbone lane interesting, but they rely on candidate quantization/error features. `candidate_index_usage` is similarly strong (`-0.000322`) but exceeds the nominal held-out mean dead target (`+0.055176`) and requires candidate indices or an explicit signal.

This changes the paper plan in a good way. The safe claim is not simply "we found a decoder-side threshold". The safer and stronger claim is: hyperprior-conditioned geometry produces measurable gain; candidate geometry/usage statistics reveal where it should be applied; and a deployable controller must either use decoder-known hyperprior summaries for a smaller no-side-bit gain, or learn/distill/signpost the stronger candidate-forward reliability signal. This keeps the work aligned with the prompt's central thesis while avoiding a hidden encoder-decoder mismatch.

The next experiment should therefore be bifurcated. Keep `hyper_preindex` as a conservative deterministic baseline, and in parallel train a small reliability head/proxy from E132/E135 teacher labels so that the stronger `encoder_candidate_error`/`candidate_index_usage` signal becomes decoder-reproducible or explicitly signaled. Both branches must be evaluated with the same checkpoint, split, RD, qMSE, dead-code, perplexity, q95 damage, and nonfinite audits.

Artifact: `experiments/analysis/e135_decoder_reproducible_guard_audit.md`.


## E136 Analysis: Existing Hyper Summaries Are Too Coarse For The Main Controller

E136 tests the most natural follow-up to E135: if a single hyper-preindex threshold is too conservative, can a tiny learned proxy over the same decoder-known hyper summaries recover more of the candidate-forward guard gain? The answer is no, at least with the current scalar summaries.

At mean dead budget `0.05`, the best deployable `hyper_preindex` probe is trained on `safe_win_dead_le_0.100`. It wins `4/4` split protocols and reaches delta RD `-0.000116` with mean dead `+0.016276` and q95 damage `0.000041`. That is useful, but it is the same operating point found by the E135 deterministic hyper-preindex guard. Other labels are weaker or less stable: `rd_win` and `safe_win_dead_le_0.075` only win `2/4`, and `safe_win_dead_le_0.050` wins `3/4`.

The budget `0.075` hyper-preindex result should not be mistaken for a solved controller. It selects every held-out image and therefore reproduces global staged-gate usage: delta RD `-0.000325`, mean dead `+0.068034`, and q95 damage `0.000360`. This is a valid global-use reference, but it does not provide reliability selection.

The reference probes explain what is missing. Baseline-diagnostic and candidate-reference feature sets can reach the `-0.00030` to `-0.00034` range under split protocol, but those features are not pure decoder-preindex signals. The current hyper summaries are too low-dimensional and nearly collapse to the same threshold behavior, while the strong decisions need information about image difficulty, candidate geometry displacement, or code usage.

Paper implication: the no-side-bit story is now safer but smaller. We can claim a conservative decoder-known guard, but the stronger controller needs either explicit signaling/two-pass evaluation/proxy distillation, or richer decoder-known local summaries generated by the hyperprior. This actually sharpens the method story: the proposed hyperprior-conditioned geometry should not only generate geometry, it should also expose enough local reliability information for deciding where that geometry should be trusted.

Artifact: `experiments/analysis/e136_decoder_proxy_supervised_probe.md`.

## E140 Analysis: Active Low-Rate Geometry Is Repeating, But Usage Control Still Matters

E140 advances the method-strengthening lane rather than replacing the current paper-main claim. At `lambda_rd=0.0018`, active HCG `bias010` improves over matched HCS on both completed seeds: seed1234 improves by `-0.021121` RD and seed2345 improves by `-0.011443` RD. The two-seed mean is scalar `1.723806`, HCS `1.200226`, and HCG `bias010` `1.183944`, so HCG improves mean RD by `-0.016282` vs HCS and wins `2/2` seeds.

The result is real enough to keep pursuing because the Householder path is active and checkpoint-selected HCG improves RD under the same training/evaluation protocol. It is not yet a submission-level rate curve because seed3456, holdout4096, and at least one additional lambda point are still missing. The checkpoint audit also matters: on seed2345, step500 drifts by `+0.071768` for HCG and `+0.083574` for HCS relative to step250, so reporting only final checkpoints would be misleading.

The intermediate-feature analysis gives the correct caution. Active geometry increases dead-code by about `+0.036133` on average and lowers perplexity by `-2.240702` relative to HCS. Mean qMSE improves by `-0.002610`, but seed2345 is nearly neutral on qMSE while still improving RD. This means the low-rate gain is not a simple collapse-free free lunch; it is a promising active-geometry result that still needs usage control, holdout confirmation, and multi-rate repetition.

Paper implication: keep E139 entropy-only holdout as the stronger component-ablation evidence for the central claim, and keep E140 as the parallel path for making HCG-RVQ itself stronger. The two lanes are complementary: E139 protects the claim that local geometry adds beyond index entropy, while E140 suggests that active geometry initialization can improve the actual method at a lower rate point.

Artifacts: `experiments/analysis/e140_multirate_lambda0018_two_seed_package.md` and related CSV/JSON files.

## E140 Analysis Update: Three Seeds Reveal A Fragile Low-Rate Geometry Gain

After adding seed3456, the active low-rate HCG `bias010` result should be treated as promising but fragile. It wins seed1234 by `-0.021121` RD and seed2345 by `-0.011443` RD, but loses seed3456 by `+0.029050` RD. The three-seed mean is still slightly better than HCS (`-0.001171` RD), but that number is too small and too seed-sensitive to serve as a paper-main rate claim.

The failure mode is not a dead geometry path. On seed3456, Householder geometry is active (`householder_delta_rms=0.040809`), but HCG worsens y-error by `+0.004033`, qMSE by `+0.000218`, and index bpp by `+0.001209` relative to HCS. This suggests the low-rate variant needs reliability or usage control, not simply larger geometry magnitude.

Paper implication: E139 remains the safer component-ablation evidence for the central claim that local HCG geometry adds beyond index entropy. E140 remains a parallel method-strengthening track: it identifies a useful active-geometry initialization and a concrete fragile-seed failure that can motivate a controlled geometry gate or selector.

Artifact: `experiments/analysis/e140_multirate_lambda0018_three_seed_package.md`.

## E141/E142 Analysis: Low-Rate Geometry Has Selection Headroom, But Local s_q Shrinkage Is The Wrong Control

E141 changes the low-rate interpretation from a weak mean result into a reliability-control problem. At `lambda_rd=0.0018`, fixed active-HCG `bias010` barely beats HCS on Kodak24 across three seeds (`-0.001171` RD), but per-image oracle switching between HCS and HCG reaches `-0.024997` RD vs HCS. The oracle uses HCG on `33/72` images, so the geometry is beneficial on a substantial subset rather than being a seed-only artifact.

The best simple leave-one-seed-out selector is high image-level `hcg_rvq_s_q_mean`: it improves the held-out mean by `-0.007173` RD vs HCS and nearly neutralizes the fragile seed3456 damage (`+0.001559` instead of fixed HCG's `+0.029050`). This matters because `s_q` is a hyperprior-side, decoder-known signal. It gives a plausible path to a deployable reliability controller, but only if the selection mechanism is implemented consistently inside one fixed codec or signaled cleanly.

E142 prevents the tempting but wrong shortcut. A posthoc local `s_q` risk multiplier over the Householder gate does not reproduce the image-level selector. The best local-multiplier row is still `+0.000231` vs HCS and worsens seed3456 to `+0.035453`. In other words, `s_q` is useful as an image-level reliability signal, but directly shrinking local geometry from local `s_q` changes the operating regime and can increase the fragile-seed failure.

Paper/method implication: E139 remains the safer component-ablation evidence for the manuscript claim that geometry adds beyond entropy-only conditioning. E141 strengthens the method-improvement lane by showing large low-rate selector headroom with a decoder-known signal. E142 says the next implementation should be an image-level or learned reliability controller, not a naive local continuous `s_q` multiplier.


## E143 Analysis: Holdout4096 Turns Low-Rate Geometry Into A Stronger Method Lane

E143 is the first low-rate `bias010` result that deserves serious method-level attention. On OpenImages holdout4096 across seeds `1234/2345/3456`, fixed HCG reaches RD `1.182335` versus HCS `1.200992`, a `-0.018656` improvement. This is much stronger than the Kodak E141 mean (`-0.001171`) and shows that active hyperprior-conditioned geometry is not merely a small Kodak fluctuation.

The result is still not a finished codec claim because the seed behavior is asymmetric. HCG improves seed1234 by `-0.081201` and seed2345 by `-0.010787`, but worsens seed3456 by `+0.036019`. The q95 positive damage is `0.142735`, so tail risk remains visible. The correct interpretation is therefore not "always use bias010 geometry", but "bias010 geometry creates real holdout gain and needs reliability control."

The difficulty analysis gives a clean mechanism. Fixed HCG hurts the easiest HCS quartile (`Q1`: `+0.020572`) and is near neutral on `Q2` (`-0.001161`), while improving `Q3` by `-0.019836` and `Q4` by `-0.074200`. This aligns with the central HCG story: local geometry is most useful when fixed HCS quantization is stressed, but it can over-transform easy images.

The selector result is the strongest part. A leave-one-seed-out rule using low `hcg_rvq_householder_strength` gives mean held-out delta `-0.030576` vs HCS. It keeps seed1234 large gain (`-0.081201`), keeps seed2345 positive (`-0.010551`), and almost eliminates seed3456 damage (`+0.000023`). The raw correlation is also interpretable: `hcg_rvq_householder_strength` correlates positively with HCG-HCS damage (`r=0.260907`), while higher HCS RD correlates with larger HCG benefit (`r=-0.369136`).

Paper implication: E143 strengthens both parallel lanes. For the manuscript safe claim, it reinforces that quantizer geometry has measurable effect beyond entropy-only controls. For making HCG-RVQ stronger, it gives a concrete reliability-controller target: geometry should be applied more conservatively when predicted Householder strength is high, especially for easy images. The next experiment should convert this LOSO diagnostic into one fixed decoder-reproducible controller and evaluate it on holdout4096 without validation-time switching, with the same RD, q95 damage, qMSE, codebook usage, and nonfinite audits.

Artifact: `experiments/analysis/e143_lowrate_bias010_holdout4096_selector.md`.

## E144 Analysis: Independent-Split Controller Confirms The Reliability Signal

E144 makes the E143 selector result much safer. Instead of choosing a threshold on holdout4096, it trains the threshold on an independent OpenImages transfer split (`start_index=8192`) and applies that threshold unchanged to holdout4096. This is still a diagnostic switch between separately evaluated HCS and HCG checkpoints, but it is no longer a same-split threshold artifact.

The transfer split repeats the fixed-HCG behavior almost exactly. Fixed HCG `bias010` improves transfer RD by `-0.018676` vs HCS, compared with `-0.018656` on holdout4096. The seed pattern also repeats: two strong/positive seeds and one fragile seed. This is important for the paper because it says the low-rate geometry effect and its failure mode are stable across OpenImages splits.

The transfer-trained controller is strikingly aligned with the E143 diagnostic: the best transfer-to-holdout rule is low `hcg_rvq_householder_strength`, with threshold `0.271352783`. On holdout4096 it reaches mixed RD `1.170368`, which is `-0.030624` vs HCS and `-0.011968` vs fixed HCG. It keeps seed1234 and seed2345 at their full fixed-HCG gains (`-0.081201`, `-0.010787`) while reducing seed3456 from fixed-HCG `+0.036019` to `+0.000115`.

The feature distribution gives a useful mechanism rather than just a leaderboard number. Selected rows are harder under HCS (`1.219957` vs rejected `1.162894`) and have lower predicted Householder strength (`0.259484` vs `0.283398`). HCG improves the selected subset by `-0.045869` with win fraction `0.682228`, but hurts the rejected subset by `+0.036009` with win fraction `0.229677`. This directly supports the claim that hyperprior-generated geometry needs reliability control, and that the reliability signal is present in the geometry head itself.

Paper implication: E144 is a strong controlled-evidence result for an international submission. It should be presented carefully: not as a deployed codec result, but as independent-split evidence that a decoder-known geometry summary can decide where HCG should be trusted. For the method-strengthening lane, this becomes the next implementation target: one HCG checkpoint with a built-in reliability gate derived from `householder_strength`, evaluated on transfer/holdout with RD, q95 damage, qMSE, dead-code, perplexity, and checkpoint audits. For the SOTA/backbone plug-in lane, this gives the reliability control that should be ported, rather than porting raw fixed geometry.

Artifact: `experiments/analysis/e144_lowrate_bias010_transfer_to_holdout_controller.md`.

## E145/E146 Analysis: The Reliability Signal Is Real, But Gate Multiplication Is Not Enough

E145 and E146 are important because they separate the reliability claim from the deployed-method claim. E144 already showed a strong independent-split result: a transfer-trained `householder_strength` rule improves holdout4096 by `-0.030624` RD vs HCS and `-0.011968` vs fixed HCG when it can switch between HCS and HCG outputs. E145/E146 ask the stricter question: can we get that benefit inside one HCG checkpoint by multiplying down the geometry gate?

The deterministic answer is no for the current backoff. E145 uses the E144 threshold inside the same HCG checkpoint, but the resulting RD is `1.215586`, which is worse than HCS by `+0.014594` and worse than fixed HCG by `+0.033251`. The failure is strongest on seed3456 (`+0.107861` vs HCS). This means the E144 controller should not be described as a simple amplitude shrinkage rule. The HCG checkpoint has been trained around its geometry/codebook operating point, and posthoc shrinkage does not recreate a matched HCS model.

The learned head answer is also no in the first setting, and the transfer-fit audit explains why. On the seed3456 transfer split, the teacher says keep HCG on only `0.244873` of images. If the reliability head had learned the label, keep-label images should have higher reliability than suppress-label images. Instead, step250 gives reliability `0.983337` on keep labels and `0.983483` on suppress labels, with AUC `0.467288`; step500 gives `0.972033/0.972502`, AUC `0.468973`. So E146 does not fail primarily because of holdout transfer. It fails before that: the head-only BCE configuration is underpowered or poorly coupled to the deployed gate.

This is a useful negative result, not wasted work. It protects the paper from overclaiming E144 as a final codec and gives a sharper method target. The controlled-evidence lane should present E144 as independent-split diagnostic evidence that the geometry head contains a reliability signal. The method-strengthening lane should now test either stronger teacher-head calibration, with enough LR/rho to actually fit transfer labels, or an explicit fallback/selector branch that chooses between HCG-like and HCS-like quantization states without perturbing a trained HCG geometry continuously.

Paper implication: the safest current story is two-layered. First, HCG geometry adds something beyond entropy-only conditioning (E139) and shows strong low-rate holdout benefit with a reliable subset structure (E143/E144). Second, deployable reliability control remains an active method component: naive strength backoff and weak head-only training are negative controls, so the final paper should only promote a controller after it passes transfer-fit, holdout RD, checkpoint, feature-distribution, q95 damage, codebook usage, and nonfinite audits.

Artifacts: `experiments/analysis/e145_lowrate_strength_backoff_single_checkpoint.md`, `experiments/analysis/e146_lowrate_bias010_teacher_headonly_holdout4096.md`, and `experiments/analysis/e146_lowrate_bias010_teacher_headonly_transfer8192_fit.md`.

## E147 Analysis: Stronger Reliability Learning Exposes The Wrong Deployment Mechanism

E147 tests the most important alternative explanation for E146: perhaps the learned reliability head failed only because it was too weak or too softly weighted. I therefore increased the seed3456 transfer teacher-head setting to `rho=20.0` and `lr=5.0e-3`, still training only the reliability head from the fixed low-rate HCG `bias010` checkpoint. This produces a useful diagnostic because it separates "can the head fit labels?" from "does this deployable gate improve RD?"

The label-fit answer improves. On the same independent transfer split used for training, step500 reaches AUC `0.640196`, correlation `+0.205475`, and reliability means `0.532626` on keep labels versus `0.456699` on suppress labels. That is a clear movement away from E146's saturated near-constant head, so the negative result is not simply "teacher labels are impossible to learn."

The RD answer is strongly negative. Step500 teacher-head RD is `1.532191`, which is `+0.392222` worse than HCS and `+0.358362` worse than fixed HCG on the transfer split. Step250 is also bad (`+0.363879` vs HCS), and q95 damage is close to or above one RD unit. This means that using the learned reliability as a multiplicative geometry suppressor does not recreate the safe HCS branch; it creates a third, poorly matched operating point.

This sharpens the two-track plan. The controlled-evidence track should keep E144 as an independent-split diagnostic: it proves that the geometry head contains a usable reliability signal and that switching between HCS/HCG outputs could improve holdout RD by `-0.030624` vs HCS. The method-strengthening track should no longer spend cycles on small variants of posthoc Householder gate shrinkage. The next serious design should preserve fallback states explicitly, for example by training a mixture/selector with an HCS-like identity branch and an HCG branch, distilling the transfer teacher into a selector head, or signaling an image-level branch choice when the no-side-bit proxy is not strong enough.

Paper implication: this is a good negative result for an international submission because it prevents an overclaim. It says the reliability signal is real, the label is partly learnable, but the current single-checkpoint multiplicative deployment is the wrong mechanism. That gives a principled next implementation target for the HCG-strengthening/SOTA-plug-in lane while keeping the manuscript's controlled evidence intact.

Artifact: `experiments/analysis/e147_lowrate_bias010_teacher_headonly_rho20_lr005_transfer8192_fit.md`.

## E148/E149 Analysis: Continuous Suppression Is Not The Deployable Low-Rate Controller

E148 and E149 close the next important branch after E147. E147 showed that a stronger reliability head can partly fit independent transfer labels, but using that head as a multiplicative geometry gate destroys RD. E148/E149 ask whether the safer exact-default residual-selector branch avoids that failure at low rate.

The answer is split but decisive. E148 preserves the checkpoint but does not learn. Its mean selector probability is `0.006506`, predicted suppress fraction at `p>=0.5` is `0.000000`, and AUC is below random (`0.462855`). The RD remains almost identical to fixed HCG (`-0.000505` vs fixed), so this is a stable no-op, not a useful controller.

E149 makes the selector learnable enough to test the action path. At step500, suppress-label images receive higher probability than keep-label images (`0.511250` vs `0.463237`), AUC reaches `0.580217`, and BCE falls to `0.719839`. That is still not a strong classifier, but it is enough movement to test whether suppressing Householder geometry inside one HCG checkpoint can mimic the E144 HCS/HCG switch.

It cannot. E149 step500 reaches RD `1.306368` on transfer, which is `+0.166399` worse than HCS and `+0.132538` worse than fixed HCG. The q95 damage rises to `0.536466`, while wins vs HCS fall to only `499/4096`. Intermediate features are consistent with over-suppression: selector multiplier falls to `0.750254`, Householder strength drops to `0.211414` from E148s `0.282330`, delta RMS drops to `0.030344`, but latent qMSE worsens to `0.021121`. Reducing geometry amplitude inside the trained HCG operating point is not the same as falling back to an HCS-trained quantizer state.

Paper implication: keep the controlled evidence clean. E139 still supports the component claim that HCG geometry adds beyond entropy/index conditioning, and E143/E144 show that low-rate geometry has strong split-stable reliability structure. But do not present E145-E149 as final deployed codec rows. For the method-strengthening lane, the next credible implementation should preserve discrete states: an explicit HCS/HCG branch, a jointly trained mixture with an identity/fallback path, or a signaled image-level branch if no-side-bit proxy control remains too weak. This is also the safer design to port later into SOTA/backbone plug-in experiments.

Artifacts: `experiments/analysis/e148_lowrate_bias010_residualselector_transfer8192_suppress_rho100_yhatanchor50_deadzone014_fit.md` and `experiments/analysis/e149_lowrate_bias010_residualselector_transfer8192_suppress_rho20_lr005_noanchor_deadzone014_fit.md`.

## E150 Analysis: The Next Stronger Method Should Preserve States, Not Shrink Geometry

E150 ties together the controller story in a way that is useful for both paper writing and next implementation. The best current controlled result is not a continuous gate, but a state-preserving branch between HCS-like and HCG-like behavior. On holdout4096, HCS is `1.200992`, fixed low-rate HCG `bias010` is `1.182335`, and the transfer-trained `householder_strength < 0.271352783` branch is `1.170368`. That is `-0.030624` vs HCS and `-0.011968` vs fixed HCG. The branch improves beyond fixed HCG because it keeps the useful geometry rows while removing much of the fragile tail.

The side-bit audit makes the branch path more credible than it initially looked. Even if the final codec must signal an image-level branch bit, the measured cost on the actual OpenImages rows is only `0.000001362`, and a conservative `256x256` assumption is `0.000015259`. These are tiny compared with the branch gain. Therefore, the next deployable low-rate design does not need to force a no-side-bit continuous suppressor if that suppressor breaks RD. A signaled branch or a decoder-reproducible branch are both plausible research paths.

The contrast with E145-E149 is the key diagnosis. E145, E147, and E149 all reduce or alter the Householder path inside one trained HCG operating point. They do not recreate an HCS-trained state. E145 becomes worse than HCS (`+0.014594`), E147 step500 becomes much worse on seed3456 transfer (`+0.392222`), and E149 step500 is also clearly worse (`+0.166399`). E148 only avoids damage by doing almost nothing. This pattern is too consistent to treat as a hyperparameter accident.

The paper-safe implication is: HCG geometry is useful, and the hyperprior-generated geometry statistics contain a split-stable reliability signal, but continuous posthoc suppression is not the right final mechanism. The stronger-method implication is: the next implementation should explicitly preserve two states. The lowest-risk version is an image-level HCS/HCG branch with one optional side bit. The more elegant version is a jointly trained two-state mixture or branch module with an identity/HCS-like path and an HCG path. This is also the mechanism that should be ported to SOTA/backbone plug-in later, rather than raw fixed HCG or continuous gate shrinkage.

Artifact: `experiments/analysis/e150_branch_vs_continuous_controller_audit.md`.

## E151 Analysis: Direct Evaluation Confirms The Branch Evidence

E151 answers the most important provenance question left after E150. E150 already showed that the transfer-trained `householder_strength < 0.271352783` branch was much better than fixed HCG, but it was a consolidated audit over existing artifacts. E151 reruns the matched HCS and HCG checkpoints directly under the current evaluation path and then applies the same branch rule. This removes the stale-CSV concern for the low-rate branch result.

The direct result is strong and internally consistent. HCS is `1.200992`, fixed HCG `bias010` is `1.182335`, and the branch is `1.170368`. Therefore the branch improves over HCS by `-0.030624` RD and over fixed HCG by `-0.011968` RD. With actual one-bit image-level signaling cost included, the branch is `1.170369`, so the rate cost changes the result by only `0.000001362`.

The seed breakdown explains why this is the right mechanism. Fixed HCG already works well for seed1234 and seed2345, so the branch keeps HCG on those rows and preserves their gains (`-0.081201` and `-0.010787` vs HCS). Fixed HCG fails on seed3456 (`+0.036019` vs HCS), and the branch nearly removes that failure by selecting HCG on only `0.002930` of seed3456 rows; seed3456 becomes only `+0.000115` worse than HCS. The result is therefore not just a better mean. It specifically removes the fragile tail that made fixed HCG hard to claim safely.

The paper-safe implication is now clearer. HCG-RVQ has controlled evidence that hyperprior-generated geometry is useful, that the useful subset is visible through a geometry statistic, and that a state-preserving HCS/HCG branch can exploit this signal with negligible side cost. The remaining limitation should be stated honestly: E151 is a two-state branch protocol over matched checkpoints, not a single integrated codec. That is acceptable as controlled evidence and as a design target, but a final main-method row still needs either a signaled integrated branch or a jointly trained two-state mixture.

The SOTA/backbone implication is also clearer. Jumping immediately to a large MambaIC/HPCM/DCAE-style backbone with raw HCG would risk mixing backbone effects with the known HCG reliability failure. The better plug-in path is to port the state-preserving branch interface first: add a minimal HCS-like fallback state and an HCG state to a strong backbone, then evaluate whether the branch improves that backbone over itself. This is the safer SOTA-facing claim than asking the current MeanScaleHyperprior prototype to beat all modern LIC models outright.

Artifact: `experiments/analysis/e151_signaled_branch_direct_eval.md`.

## E152 Analysis: The Two-Track Plan Is Now Operational

E152 makes the two-track plan concrete. The controlled-evidence branch is no longer just a hardcoded E151 script: it now has a manifest and a generic evaluator. The 2-image GPU0 smoke is deliberately small, but it verifies the important implementation contract: the manifest can load the matched HCS/HCG checkpoints, compute branch features with zero nonfinite rows, apply the same low-strength Householder rule, and account for the one-bit branch signal. This means future split checks and rate points can reuse the same branch protocol instead of creating new one-off scripts.

The full E151 result remains the meaningful number: branch RD `1.170368`, `-0.030624` vs HCS and `-0.011968` vs fixed HCG; signaled branch RD `1.170369`; nonfinite rows `0`. The E152 smoke is not a new quality claim. Its value is reproducibility and portability.

The SOTA/backbone scout changes the practical priority. DCAE, MambaIC, LIC-HPCM, and RDVQ are now cloned locally under `third_party/`. DCAE is the best first plug-in smoke because it has a straightforward CompressAI-style `y -> sliced y_hat -> g_s` boundary and real pretrained/eval/compress scripts. MambaIC is compelling but has heavier VMamba/selective-scan setup risk. HPCM is highly relevant for stage-context design but too entangled with progressive coding to be the first branch transplant. RDVQ is a must-cite and must-compare paper, but its official repo is not executable yet.

The key judgment is that SOTA experiments should start earlier than a final paper polish phase, but not as raw blind large-scale trials. The first SOTA-facing question should be: "Does a state-preserving HCS/HCG branch improve a strong backbone over itself?" That aligns with the observed HCG failure mode and avoids confusing a backbone gain with the proposed quantizer-geometry mechanism.

Artifact: `experiments/analysis/e152_branch_manifest_sota_package.md`.



## E153 Analysis: SOTA Plug-in Should Start With Boundary Validity, Then Quality

E153 directly addresses the scale concern: small controlled ablations can succeed while a larger backbone exposes incompatible assumptions. The first DCAE plug-in smoke did expose such assumptions. A 128x128 image failed inside the DCAE window-attention partitioning, not inside HCG, so the valid smoke size must respect the DCAE internal feature-grid constraints. More importantly, connecting HCG to DCAE exposed a missing adapter helper that had not been exercised by the current simple-backbone path: `HCGQuantizerAdapter` did not implement `_householder_gate_strength_backoff_multiplier`, while the shared quantizer runner and model-side HCG implementation expected it.

After adding the helper, the 256x256 DCAE smoke passed on GPU0 with `0` nonfinite rows. The shapes are exactly the kind needed for a real plug-in: DCAE produces `y` with shape `[1, 320, 16, 16]`, hyper features with shape `[1, 640, 16, 16]`, and HCG/HCS adapter outputs `y_hat` with shape `[1, 320, 16, 16]`, which DCAE `g_s` decodes back to `[1, 3, 256, 256]`. The active-HCG adapter also exposes finite Householder diagnostics: strength `0.250000`, delta RMS `0.026319`, and finite conditioning tensors.

This result should be interpreted conservatively. It uses randomly initialized DCAE weights unless a pretrained checkpoint is explicitly supplied later, so qMSE/perplexity here are only sanity statistics, not paper-facing compression numbers. The paper-relevant value is that the SOTA/backbone track is now operational in code, and future pretrained or trained DCAE-over-itself comparisons can ask the right question: does the state-preserving HCS/HCG branch improve a strong model over its own baseline?

The two-track policy remains the safest path. Controlled evidence keeps isolating the novelty and claim validity. SOTA plug-in experiments now proceed as staged external-validity checks: boundary smoke, pretrained baseline reproduction, HCS/HCG adapter training or branch integration, then self-improvement comparison against the same backbone. This avoids both failure modes: relying only on tiny experiments, or burning large GPU budget on an ambiguous raw transplant.

Artifact: `experiments/analysis/e153_dcae_hcg_adapter_smoke.md`.


## E154 Analysis: Large-Backbone Experiments Need Same-Backbone Claims

E154 answers the practical question behind the SOTA/backbone concern. Yes, short controlled experiments can fail to scale. But the right response is not to run a large raw transplant first. The right response is to make the large experiment answer a clean question: does the proposed quantizer-geometry branch improve a strong backbone over that same backbone, under matched checkpoints, images, rate accounting, and feature diagnostics?

The DCAE provenance check is reassuring. The existing clone and the README official clone target have the same commit, so the E153 adapter smoke used the same code state as the official target. This matters because the next DCAE result must be defensible as a plug-in experiment, not a fork artifact. DCAE also exposes both an evaluation script and low-rate MSE checkpoints, making it the best first SOTA path.

The comparison design should be staged. First reproduce DCAE baseline with one checkpoint and a path-fixed image subset, reporting bpp, PSNR, MS-SSIM, runtime if available, and nonfinite rows. Second insert the HCS/HCG state-preserving branch at the `y_hat -> g_s` boundary and compare against the same DCAE checkpoint family. Third expand to full Kodak or OpenImages holdout and add per-image tails, checkpoint selection, feature distributions, and branch side-bit accounting.

This also clarifies the paper strategy. HPCM and MambaIC are important SOTA baselines, but they are not the cleanest first plug-in because their codec paths introduce more confounders. RDVQ is a central VQ-related comparison for the literature framing, but it is not executable locally yet. Therefore, controlled evidence and DCAE-first plug-in are complementary rather than competing priorities.

Artifact: `experiments/analysis/e154_sota_backbone_reproduction_audit.md`.


## E155 Analysis: The SOTA Lane Has Moved From Boundary Smoke To Baseline Reproduction

E155 is the first step where the SOTA/backbone lane becomes empirical rather than only architectural. E153 showed that DCAE exposes a compatible boundary for HCG local geometry. E154 showed that the DCAE remote-name mismatch does not invalidate that boundary because the mirrored and README official clones share the same commit. E155 now shows that an official pretrained DCAE checkpoint can be downloaded, loaded, and evaluated on this machine with finite outputs.

The full Kodak baseline is useful even though it is only one rate point: bpp `0.109397`, PSNR `29.245565`, MS-SSIM dB `11.753408`, `0` nonfinite rows. More importantly for HCG-RVQ, the artifact now contains per-image intermediate distributions for `y`, predicted means, predicted scales, and likelihoods. That gives a concrete feature baseline for deciding whether a DCAE HCS/HCG branch changes the latent geometry in a controlled way or simply damages the pretrained operating point.

The correct next DCAE experiment is therefore not to claim SOTA improvement immediately. It is to keep the checkpoint, image paths, metric code, and feature diagnostics fixed, then insert a state-preserving HCS/HCG branch at the DCAE latent boundary. The comparison should be DCAE vs DCAE plus proposed branch, not HCG prototype vs DCAE. This is the same principle as E151, transferred to a stronger backbone.

This directly addresses the concern that small ablations may not scale. We are no longer staying only in small cycles; we have begun official-checkpoint SOTA reproduction. But the comparison remains controlled enough that a failure or gain can be attributed to the quantizer strategy rather than to a different backbone, split, or metric path.

Artifact: `experiments/analysis/e155_dcae_lambda0018_kodak24_baseline.md`.

## 2026-06-03 E159/E160 VQ-SOTA Plug-in Analysis

The current SOTA-facing VQ-LIC direction is now sharper. EF-LIC is the primary plug-in target because it already uses an RVQ bottleneck, and E158/E160 show that HCG geometry can be inserted at EF-LIC's projected 8-D RVQ boundary without bitstream drift. GLC remains important as a low-bitrate generative benchmark and risk hedge, but its main `y` path is scalar-rounded latent transform coding rather than RVQ, so it is a larger surgery for the direct HCG-RVQ claim.

E160's active projected-HCG smoke should be interpreted as an insertion and reliability-control result, not as a final quality row. Kodak24 `mean/alpha=0.05` is bitstream-valid (`0` nonfinite, `max_decode_diff=0`) and unchanged-bpp, but always-on untrained geometry only improves force0 DISTS on average and degrades higher-rate DISTS slightly. The useful result is the per-image headroom: a DISTS oracle can improve all force points by selectively turning active geometry on for 25-50% of images. This motivates a trained projected-HCG head or teacher-label controller for EF-LIC.

## 2026-06-03 E161 Selector Label Analysis

E161 turns the E160 EF-LIC active-geometry result into a controller-ready label set. The key finding is that active projected-HCG is not globally reliable, but it is not random either: DISTS-positive labels appear at every rate, from `6/24` images at force1 to `12/24` images at force0 and force3. The best next experiment is therefore a rate-conditioned reliability controller or a signaled image-level selector, rather than a fixed threshold shared across all rates.

The feature manifest matters for codec validity. Decoder-safe context features can support a no-side-bit controller, but they are weaker. Encoder/active diagnostic features can be much richer, but they require signaling the branch decision; the one-bit cost is small enough to keep this path viable. This mirrors the earlier controlled-evidence branch conclusion in the simpler HCG-RVQ backbone.

## 2026-06-03 E162/E163 GLC Main-Track Analysis

E162 changes the status of GLC from a planned hedge to an active main track. The official GLC image checkpoint loads strictly and evaluates on Kodak24 with `0` nonfinite rows. The q0-q3 BPP range is `0.0243` to `0.0370`, which is the ultra-low-bitrate regime relevant to GLC's paper. This makes GLC a real second main target for the low-bitrate generative compression part of the HCG-RVQ story.

The main caveat is metric protocol. GLC's official metric code uses LPIPS-alex, while EF-LIC's paper/evaluator path uses LPIPS-vgg. Therefore, source-method reproduction should follow each paper's protocol, but any cross-method table must standardize LPIPS backbone. DISTS is less affected by this particular mismatch and is useful as a common perceptual metric.

The training strategy should be staged. Pretrained reproduction and frozen smoke are necessary for protocol and bitstream validity. They are not enough for the final claim. Final paper rows should come from matched training or matched fine-tuning: EF-LIC baseline vs EF-LIC+HCG and GLC baseline vs GLC+HCG under the same data, optimizer, q/rate schedule, and checkpoint selection.



## E164 Analysis: GLC Evaluation Provenance Is Now Locked

E164 removes a major source of risk in the GLC track. The E162 evaluator was manually expanded from the official `test()` path so it could expose `y`, `z`, `z_hat`, priors, residuals, quantized latents, bit components, and codebook usage. Manual expansion is useful for HCG analysis but dangerous unless it is proven identical to the official path.

The full Kodak24 identity check shows exact equality: `max_abs_xhat_diff=0`, `mean_abs_xhat_diff=0`, and all bit-component differences are `0` for every image and q-index. Therefore the GLC intermediate-feature analysis can be used as a reliable baseline for future HCG deltas. This is especially important because GLC main compression path is not an RVQ module: future `z_vq` geometry and `y`-path HCG-RVQ replacement experiments need precise attribution between official codec behavior, measurement code, and the new quantizer geometry.

Research implication: EF-LIC and GLC can both remain main tracks, but with different evidence roles. EF-LIC is the clean direct RVQ plug-in track. GLC is a stronger low-bitrate generative compression track whose HCG version will require either a safe `z_vq` geometry branch or a larger `y` latent coding redesign. E164 makes that larger GLC redesign experimentally accountable instead of speculative.


## E165 Analysis: GLC Main Track Requires `y`-Path HCG, Not Only `z_vq`

E165 clarifies how to keep GLC as a true main track without weakening the HCG-RVQ claim. The `z_vq` module is convenient because it is explicit VQ, but it is upstream of the hyperprior output. A geometry transform conditioned on raw `z` would not be decoder-reproducible without extra signaling, and a fixed or q-conditioned transform would not fully express the core proposal that hyperprior/context generates local quantizer geometry.

The stronger GLC route is the `y` residual path in `forward_four_part_prior()`. At each masked part, the decoder already has `z_hat`, fused hyperprior parameters, spatial-prior context, q-index, deterministic masks, and previously decoded `y_hat_so_far`. That is the right place to replace scalar rounding with HCG-RVQ and a learned index-prior bit model. This is heavier than EF-LIC, but it is the GLC route that matches `docs/prompt.txt` rather than a superficial VQ transplant.

For final claims, pretrained-only forward modifications remain diagnostic. The paper-facing GLC comparison should be original GLC vs GLC+HCG under matched Stage II/III fine-tuning or retraining on the paper schedule. Full Stage I scratch training is optional later evidence, not the first gate.


## E166 Analysis: GLC Main `y` Boundary Is Ready For HCG

E166 is the GLC equivalent of the earlier EF-LIC identity-boundary work. It replaces `forward_four_part_prior()` with an HCG-ready identity implementation and proves that the rewritten path is exactly equivalent to official GLC on Kodak24. The important numbers are all zero: `x_hat` diff, `ref_latent` diff, `bit_y` diff, `bit_z` diff, and total-bit diff. Nonfinite rows are also zero.

This is a stronger GLC foundation than a `z_vq` smoke alone. The `y` path is where GLC actually performs transform coding in the generative latent space. Since E166 exposes `y_res`, `y_q`, part-wise masks, predicted means/scales, and combined reconstruction while preserving official behavior, future HCG-RVQ changes can be attributed to the quantizer design rather than to a rewritten GLC evaluation path.

The residual statistics already inform design. Combined residual std rises from `0.0765` at q0 to `0.1440` at q3, and the four masked parts have decreasing residual scale. A GLC HCG-RVQ branch should therefore normalize per part or use a part-aware head; one global RVQ setting risks being too aggressive for later parts or too weak for early parts.

## E167 Analysis: Past HCG Results Should Shape The Plug-in Design

E167 consolidates the prior HCG-RVQ experiments into transfer rules. The most important result is not merely that HCG geometry can help. It is that HCG helps selectively, and the deployable mechanism should preserve fallback/active states. E143/E144 show that geometry helps harder images and that `householder_strength` is a useful reliability signal. E150/E151 show that a state-preserving HCS/HCG branch improves beyond fixed HCG with negligible one-bit side cost. E145/E147/E149 show that continuous gate shrinkage is the wrong deployment mechanism.

This changes the EF-LIC/GLC plan. For EF-LIC, the next serious path is original projected RVQ as fallback plus projected-HCG active state, with a trained or signaled selector. For GLC, the next serious path is not a standalone `z_vq` transform and not raw active `y` geometry. It is an E166-based `y`-path HCS-like fallback and HCG-RVQ active branch, with index-prior or explicit side-bit accounting for paper-facing rows.


## E168 Analysis: GLC y Residuals Are Sparse But Heavy-Tailed

E168 refines the GLC transfer plan from a generic `y`-path replacement into a part/group-aware HCG-RVQ design. The audit is read-only at the E166 identity boundary, so it does not introduce a new codec result. Its value is that it measures the exact residual distribution currently handled by GLC scalar rounding and Gaussian bit estimation.

The active residuals are not well described by a single global scale. Earlier masked parts have larger residual std, and higher q-indexes make the p99 tail much larger, but p95 remains small. This means most active elements are tiny, while a small number of elements and channel groups carry the difficult residual mass. The scalar symbols are also extremely sparse: around `98.9%` to `99.6%` of active rounded residuals are zero in the part averages. Therefore a naive dense RVQ replacement would spend capacity and signaling on regions that the baseline already codes as zero.

The most important design cue is the group tail concentration. Groups `1`, `7`, `10`, and `15` dominate the largest tails in early parts; for q3 part0 group1 reaches `res_abs_p95=0.33593` and `res_abs_p99=1.87256`. This supports a GLC HCG-RVQ branch that is part/group aware, sparse, and state-preserving. The fallback should keep the original scalar/HCS-like path for zero-heavy regions, while the active HCG-RVQ state targets heavy-tail groups where a local hyperprior-conditioned codebook can plausibly reduce distortion or improve perceptual quality.

The difficulty-quartile view is weaker than the group view. DISTS quartiles change scale means slightly and q-zero fraction monotonically, but residual p95 is not a clean image-level selector. This suggests the first deployable GLC selector should not be only image-level difficulty; it should use decoder-safe local features such as part id, group id, predicted scale, residual-energy proxies, and eventually HCG geometry strength.

Artifact: `experiments/analysis/e168_glc_y_res_distribution_kodak24.md`.


## E169 Analysis: A Compact Decoder-Known Branch Captures Most GLC y-Residual Energy

E169 turns the E168 residual audit into a concrete GLC branch interface. The active subset is static and decoder-known: parts `[0, 1, 2]` and channel groups `[1, 7, 10, 15]` with group size `16`. Since both active and fallback states still run original scalar rounding, the correct result is exact equality with official GLC. The full Kodak24 run achieves that: max `x_hat`, ref-latent, and bit differences are all zero, with `0` nonfinite rows.

The coverage numbers are the important finding. Only `18.75%` of valid residual positions and `4.69%` of all latent elements are marked active, yet this subset covers `79.5%` to `82.3%` of residual energy, `65.1%` to `66.3%` of scalar rounding-error energy, and `78.6%` to `82.0%` of nonzero scalar symbols across q0-q3. This is a much cleaner target than dense GLC y-path replacement.

This gives the next GLC experiment a sharp hypothesis: HCG-RVQ should be applied first to a sparse, part/group-selected active state that carries the residual tail, while the original scalar path remains an exact fallback for zero-heavy regions. That design matches the prior HCG lesson from E151/E167 and also respects GLC bit accounting because the branch mask is static/decoder-known at this stage.

Artifact: `experiments/analysis/e169_glc_y_tail_branch_identity_kodak24.md`.

## E170/E171 Analysis: GLC Needs Local Codebooks And Index Priors

E170/E171 convert the E168/E169 GLC distribution evidence into a quantizer design decision. These are not final codec rows, because codebooks are trained leave-one-image-out on Kodak active residuals and image reconstruction is not changed yet. The value is diagnosis.

The most important negative result is that a shared active codebook fails. Shared K=8 is still worse than scalar rounding at every q, and the MSE ratio worsens as q increases: `1.6410`, `2.4966`, `4.2712`, and `6.4520`. This directly supports the HCG-RVQ motivation: the GLC residual tails are not compatible with one global codebook geometry.

The positive result is local but rate-limited. Part/group K=8 reduces active residual MSE for all q, reaching ratios `0.2458`, `0.3500`, `0.5402`, and `0.8122`. Part/group K=4 is a low-rate candidate for q0/q1 but not enough for q2/q3. K=2 multi-stage RVQ confirms that residual stages help, but not enough by themselves: K=2/L3 is strong at q0/q1 and near break-even at q2, but still worse at q3.

The design implication is precise. GLC should not receive a dense VQ replacement, and it should not receive a shared codebook active state. The next active branch should preserve scalar fallback, operate only on the E169 sparse active subset, use part/group-local codebooks or HCG-generated local geometry, and include categorical index-rate modeling. This also gives a clean paper story: HCG-RVQ is useful because it targets nonstationary heavy-tail residual states that scalar rounding and shared VQ handle poorly, while preserving the easy zero-heavy states.

Artifact: `experiments/analysis/e172_glc_tail_vq_rvq_design_decision.md`.

## E173/E174 Analysis: Integrated GLC Branch Exposes Distortion-Perceptual Mismatch

E173 is the first GLC active-branch experiment that actually changes the reconstructed image. It is still diagnostic because codebooks are trained leave-one-image-out on Kodak residuals, but it closes an important gap between residual MSE and final image metrics.

K=8 gives a clear positive signal for distortion: PSNR improves at all q and active residual MSE ratios stay below `1` at all q. This means the E168/E169 active subset is genuinely influential, not just an analysis artifact. But K=8 also increases rate and worsens DISTS almost everywhere. The branch is therefore not paper-ready as a fixed VQ replacement.

K=4 gives the complementary negative result. It reduces rate pressure, especially at high q, but it is too small for the active tail: q2/q3 active MSE ratios rise above `1`, and image metrics degrade. This rules out a simple low-capacity VQ branch as the next main method.

The diagnosis is important for the HCG-RVQ story. GLC is a perceptual/generative codec, so minimizing residual MSE in the latent prior space can move the decoder away from its preferred generative manifold. The next GLC method must train the active HCG-RVQ branch with downstream-aware losses and an index prior, while preserving scalar fallback for unreliable regions. This is aligned with the earlier HCG lesson from E151/E167: state-preserving control is not optional.

Artifact: `experiments/analysis/e174_glc_integrated_tail_vq_diagnostic_analysis.md`.

## 2026-06-04: E175/E176 GLC Decoder-Aware Active Branch Lesson

E175/E176 sharpen the GLC design after E173/E174. Fixed residual-MSE VQ proved that the sparse active subset has distortion headroom, but it damaged DISTS. The new trainable diagnostic freezes pretrained GLC and optimizes only the active part/group codebooks through the downstream decoder/generator.

The result is not paper-quality because it trains on a tiny evaluated Kodak subset, but it is a useful mechanism check. DISTS-aware optimization recovers much of the perceptual damage: q0 one-image DISTS moves from fixed VQ `0.16740` to trained `0.13440` against baseline `0.12375`; q0 two-image average moves from `0.15863` to `0.13823` against baseline `0.12772`; q3 two-image moves from `0.11542` to `0.10963` against baseline `0.10424`. All runs used GPU0 only and had `nonfinite=0`.

The important causal signal is that active residual MSE can worsen while perceptual quality improves. On q0 one image, active MSE ratio worsens from `0.2939` to `0.6129`, while DISTS improves from `0.16740` to `0.13440` and LPIPS improves from `0.22776` to `0.21694`. Therefore, the GLC paper branch should not be trained as a residual-MSE quantizer. It needs decoder/perceptual-aware training, an index prior or bit-aware loss, q-dependent capacity/stage gating, and a reliability selector that preserves scalar fallback on unsafe regions.

## 2026-06-04: E177/E178 GLC Split-Train Lesson

E177/E178 move the GLC active-tail branch beyond Kodak-overfit diagnostics. Active codebooks trained on OpenImages 256 crops transfer partially to held-out Kodak images: in q0 OI8 -> Kodak4, the trained branch improves PSNR, MS-SSIM, and LPIPS on all four images versus original GLC, and increases DISTS wins from `1/4` to `2/4` compared with the fixed init branch. Average DISTS still trails baseline (`0.12227` vs `0.11448`), and empirical bpp remains too high (`+0.014176`).

This updates the GLC hypothesis: train-split decoder-aware active quantization is viable, but always-on K=8/L1 is not sufficient as the paper method. The next GLC implementation should keep scalar fallback, learn/select active reliability, add q-dependent stages or capacity, and introduce index-prior or bit-aware training. The final novelty should be HCG-style local quantizer generation, not merely external part/group codebooks.

## 2026-06-04: GLC Active Branch Needs Metric-Aware Reliability, Not Always-On Use

E179/E181/E183 update the GLC conclusion. The OpenImages-trained active branch transfers consistently for PSNR, MS-SSIM, and LPIPS, but it is not DISTS-safe. On q0 OpenImages16 -> Kodak8, trained_eval improves LPIPS on `8/8` images (`0.20276 -> 0.19299`) and PSNR/MS-SSIM on `8/8`, but DISTS worsens from `0.11899` to `0.13134` and wins only `1/8`. DISTS-only training reduces the branch DISTS slightly to `0.13106` but still wins only `1/8`.

The selector audit is the important design signal. A DISTS oracle uses the branch on only `12.5%` of Kodak8 images and gives a tiny DISTS gain, while LOOCV threshold selectors still worsen DISTS. Therefore the paper-facing GLC path should not claim that the current active K8/L1 VQ is generally better. It should claim and implement controlled HCG-RVQ: preserve scalar fallback, learn or signal reliability, make the active branch bit-aware, and generate local shift/scale/geometry from decoder-safe context.

A metric-protocol note was also fixed: E177/E181 summaries report PSNR via mean-MSE aggregation, while E179 selector tables report arithmetic per-image PSNR. This must be stated explicitly in paper tables; DISTS/LPIPS/MS-SSIM/bpp conclusions are unaffected.

## EF-LIC Selector Reliability After E184

The E184 audit narrows the EF-LIC path. E160 proved that projected HCG can be inserted into EF-LIC's RVQ bottleneck with exact active/decode consistency (`max_decode_diff=0`, `nonfinite=0`), but always-on quality was mixed. E184 asks whether the per-image wins are predictable from features that would be available in a real codec.

The answer is: yes, but only in a conservative regime. For force0, a decoder-safe DISTS-target LOOCV threshold gives `dDISTS=-0.000592` without side bits, while in-sample thresholding reaches `-0.001325` and metric oracle reaches `-0.001695`. For force1-4, decoder-safe LOOCV becomes positive even when metric oracle remains negative. This means the branch has real headroom, but most force settings are not yet reliably selectable with a simple deployable controller.

The paper-safe interpretation is not "projected HCG universally improves EF-LIC now." The safer interpretation is "EF-LIC's RVQ bottleneck contains local geometry cases where a hyperprior-conditioned projected geometry helps, and reliability control is necessary to avoid harmful cases." The next experiment should therefore train or tune a force0-like weak HCG branch under an EF-LIC-compatible protocol, keep scalar/original fallback exact, and evaluate whether the decoder-safe reliability signal survives a larger split and full Kodak evaluation.

This also guides the VQ-LIC transfer design: use E184 as controlled evidence for state-preserving selective geometry, not as final low-bitrate performance evidence. Final claims still need full training/full evaluation under the EF-LIC/GLC paper settings or a carefully documented reproduction when official training code is unavailable.

## EF-LIC Global Predecision Selector After E185/E186

E185 strengthens the reliability-controller story by auditing feature provenance. The key correction is that not every `decoder_safe_context` feature is usable for a whole-image no-side-bit branch decision. Later-slice mean/scale statistics are decoder-reproducible only after earlier y slices have already been decoded under a chosen path. Therefore, E185 splits features into `global_predecision_context`, `sequential_context`, `legacy_decoder_safe_context`, and `encoder_active_diagnostic`.

The stricter result is still positive for the main EF-LIC candidate. Under `global_predecision_context`, force0 DISTS-target LOOCV gives `dDISTS=-0.000641`, and LPIPS-target LOOCV gives both `dDISTS=-0.000761` and `dLPIPS=-0.000281`. This means the force0 branch is not merely an artifact of later active-path diagnostics; z/slice0 hyperprior context already carries a deployable reliability signal.

E186 validates this in the actual evaluation path with the fixed rule `slice0_mean_abs_mean <= 0.455596`. The result matches the intended behavior: encoder and decoder decisions match on 24/24 Kodak images, no side bit is required, active decode is exact, and there are no nonfinite rows. The selected path improves DISTS from `0.09227` to `0.09103` and LPIPS from `0.27228` to `0.27217` at unchanged bpp `0.035645`. Always-active improves DISTS but hurts LPIPS, so the selector is doing real reliability control rather than simply applying active geometry everywhere.

Paper interpretation: this is controlled implementation evidence for hyperprior-conditioned local quantizer geometry plus reliability control at an existing VQ-LIC bottleneck. It is not yet final performance evidence, because the threshold was fit and evaluated on Kodak. The next step should be independent-split controller fitting, then EF-LIC-compatible full evaluation. If the signal survives, this becomes a much safer paper claim than raw projected-HCG smoke results.

## EF-LIC Selector Generalization After E187/E188

E187 is the first anti-overfit check for the EF-LIC global predecision selector. It does not provide final paper evidence because the data are still Kodak24, but it separates threshold fitting from held-out evaluation and uses only features that are available before a whole-image active/fallback decision.

The DISTS-target split result is positive but not sufficient. Held-out DISTS improves in `3/4` splits and averages `dDISTS=-0.000402`, but LPIPS worsens on average (`+0.000206`) and the selected feature changes across splits. This means the same-table E186 DISTS result is not purely fake, but a DISTS-only scalar threshold is probably too brittle for a paper-facing controller.

The LPIPS-target split result is more useful for the next design. It averages `dDISTS=-0.000677` and `dLPIPS=-0.000128` on held-out halves, improving both metrics in `3/4` splits. This points toward a multi-metric reliability objective: use active HCG geometry where it is perceptually safe, preserve the original EF-LIC branch where the active transform risks LPIPS/DISTS damage, and avoid claiming universal always-on gains.

E188 confirms that the LPIPS-balanced rule is deployable in the actual EF-LIC forward path, not just in an offline table. With `slice0_mean_min >= -10.7448`, encoder and decoder decisions match on all 24 Kodak images, no side bit is needed, bpp is unchanged, and both DISTS and LPIPS improve (`dDISTS=-0.000870`, `dLPIPS=-0.000468`). Compared with E186, E188 trades away the larger DISTS gain (`-0.001238`) for a much safer LPIPS improvement. That is a better conference-paper mechanism result because it demonstrates reliability control rather than metric cherry-picking.

The research decision is therefore: keep EF-LIC force0 projected-HCG as the main direct VQ-LIC plug-in candidate, but do not freeze the paper story around a Kodak-fit scalar rule. The next controller should be learned or fit on an independent split, preferably from OpenImages/CLIC-style validation images under the EF-LIC protocol. The final claim should be framed as state-preserving hyperprior-conditioned quantizer geometry with decoder-reproducible reliability control.

## EF-LIC Paper-Protocol Bridge After E189

E189 turns the E185-E188 selector evidence into a reproducible paper-protocol bridge. The key improvement is not another same-Kodak metric row; it is that the selector fit is now a standalone artifact that can be run on an independent fit split and applied through the direct EF-LIC forward path on a held-out split.

This matters for the international-conference story. The current EF-LIC evidence supports the mechanism claim that hyperprior/early-slice context can decide when HCG projected geometry is beneficial without a side bit. But the E187 instability means a hand-picked scalar threshold is not enough. E189 defines the next falsifiable test: fit the global-predecision rule on non-Kodak validation images, then evaluate on held-out Kodak/Tecnick/DIV2K/CLIC with exact encoder/decoder decision matching and unchanged bpp.

The near-term mainline should use the LPIPS-target rule because it is multi-metric safer; the DISTS-target rule is retained as an ablation showing the DISTS/LPIPS tradeoff. If independent fit/eval preserves both DISTS and LPIPS gains, EF-LIC becomes a strong direct VQ-LIC plug-in result. If not, the next implementation should move from scalar thresholds to a learned decoder-side reliability head trained on the same E161-style labels.

## EF-LIC Multi-Objective Selector After E190

E190 strengthens the EF-LIC line by making the reliability objective match the paper risk more closely. E186 showed the largest DISTS gain, but its DISTS-heavy controller had a known LPIPS risk. E188 showed a safer LPIPS-balanced direct rule. E190 asks whether that safer rule is an accident of one target metric or whether a joint DISTS/LPIPS criterion naturally selects it.

The answer is favorable. Under the deployable `global_predecision_context` feature class, the multi-objective search with `DISTS=1`, `LPIPS=3`, and positive penalty `20` again selects `slice0_mean_min >= -10.7447786`. The same-table Kodak diagnostic is identical to E188 (`dDISTS=-0.000870`, `dLPIPS=-0.000468`, unchanged bpp), but the split evidence is stronger than the previous single-target story: LOOCV improves both DISTS and LPIPS, and the four split-fit/eval audits average `dDISTS=-0.000282`, `dLPIPS=-0.000171`, with both metrics improving in `4/4` splits.

This changes the main EF-LIC claim. The paper-facing mechanism should not be framed as simply maximizing DISTS gain. It should be framed as a state-preserving HCG branch controlled by a multi-objective perceptual reliability signal. The DISTS-only E186 rule remains a useful ablation showing the DISTS/LPIPS tradeoff; the E190 rule is the better default for independent validation because it is less likely to be rejected as metric cherry-picking.

The limitation is still important: E190 is a Kodak diagnostic built from the existing active-vs-baseline label table. It is not final evidence. The next falsifiable step is independent controller fitting, then direct held-out EF-LIC evaluation with exact encoder/decoder selector matching. If that survives, HCG-RVQ has a clean VQ-LIC plug-in story: hyperprior/early-slice context predicts when local projected geometry improves a pretrained RVQ bottleneck at no extra bpp.

## EF-LIC Selector Failure Modes After E191/E192

E191 makes the E190 selector easier to defend scientifically. The E190 rule does not merely improve an average metric; it selects all images that are beneficial under the `DISTS + 3*LPIPS` diagnostic objective and avoids the worst fallback cases. Its remaining weakness is not missed opportunity but false positives: `5` selected images are still harmful under the weighted objective. This is a good failure mode for a paper method because it suggests a precise next model improvement, namely a learned decoder-side reliability head that filters selected-bad cases.

The comparison with the E186 DISTS-oriented rule is important. E186 is attractive if only DISTS is emphasized, but under the same multi-objective criterion it doubles selected-bad cases (`10` vs `5`). That explains why the main claim should not be largest DISTS gain. The stronger claim is reliability-controlled HCG geometry: active HCG helps a subset of images, and hyperprior/early-slice context can choose that subset without signaling.

E192 tests a tempting next step: add a second scalar threshold. It improves the same-table Kodak row, but it is not robust enough under LOOCV and split-fit/eval. This is useful negative evidence. It prevents us from overfitting a fragile two-threshold rule, while still identifying candidate signals (`slice0_mean_abs_mean`, `z_index_entropy`, `slice0_scale_abs_mean`, `z_hat_abs_mean`) for a learned controller. The current research posture is therefore conservative but stronger: use E190 primary for held-out/full-eval, and treat E192 as design evidence for the next trainable reliability module.

## EF-LIC Learned Reliability Head After E193

E193 tests the natural next step after E191/E192: replace hand-built scalar refinements with a tiny learned decoder-side reliability head. The result is deliberately conservative. On the same Kodak diagnostic table, the head does exactly what we hoped mechanistically: it keeps all 10 beneficial active cases, reduces selected-bad cases from 5 to 2, and improves the multi-objective score from E190 primary -0.002274 to -0.002506. This supports the hypothesis that global predecision context contains more reliability information than a single threshold can express.

But the generalization checks do not support making this Kodak-trained head the paper method. LOOCV selects only 5 of the 10 beneficial cases and leaves 6 false positives; split-fit/eval becomes unstable, sometimes falling back to no active branch and sometimes selecting harmful active rows. Stronger L2 regularization does not remove the instability. This is exactly the failure mode we wanted to detect before moving to full experiments: a higher-capacity controller can make the same-table row look better while losing the robust split behavior that made E190 useful.

The paper-facing interpretation is therefore: E190 remains the controlled default, and E193 supplies design evidence for a learned controller trained on an independent fit split. This is a stronger research posture than chasing the best Kodak24 number. The next EF-LIC experiment should generate E161-style active/fallback labels on non-Kodak validation images, train either the E190 scalar rule or this tiny reliability head there, and evaluate directly on held-out sets. If the learned head then filters E190 false positives without split collapse, it can become the main HCG-RVQ controller. Until then, it is a mechanism and headroom result.

## EF-LIC Direct Reliability-Head Deployment After E194

E194 closes the gap between the offline E193 learned reliability-head audit and an actual EF-LIC codec path. The new direct probe trains the same tiny logistic controller from the E161-style label table, then recomputes all selector features from quantities that are available on both encoder and decoder sides before the branch decision. This is important because a learned selector is only publishable as a codec mechanism if the decoder can reproduce it without side information.

The implementation found and fixed one meaningful consistency issue. The E193 feature set can include `z_index_*` statistics, but the direct path initially only exposed z and slice0 mean/scale statistics. E194 adds `z_index` entropy, perplexity, usage, and related stats from the transmitted z indices on both sides. The full Kodak24 self-check then gives exact encoder/decoder selector agreement `24/24`, zero probability difference, zero active decode difference, and no nonfinite rows.

Mechanistically, this is a positive result: the learned controller can be integrated into EF-LIC as a deterministic reliability module, and it reproduces the E193 same-table metric row inside the true branch path (`dDISTS=-0.000881`, `dLPIPS=-0.000542`, unchanged bpp). It also improves the small first4 smoke (`dDISTS=-0.001836`, `dLPIPS=-0.000771`). These numbers confirm deployability and headroom, not final generalization.

The limitation remains the same as E193. The current head is fit from Kodak-derived labels and evaluated on Kodak-derived images, so it should not replace E190 as the paper-facing default. The full self-check still selects only `7/24` DISTS wins even though average DISTS improves, which means the controller is optimizing a multi-objective average and still misses some active-good cases. The correct next experiment is independent fit/eval: generate active/fallback labels on non-Kodak data, fit both E190 scalar and E193/E194 learned controllers, and evaluate them directly on held-out Kodak/Tecnick/CLIC.

Paper interpretation: E194 strengthens the implementation story for HCG-RVQ. It shows that reliability-controlled hyperprior-conditioned geometry can be deployed with exact decoder reproducibility and no extra bpp. It does not yet prove that a learned controller generalizes; that proof needs independent labels and held-out evaluation.

## VQ-LIC Transfer Readiness After E195

E195 reorganizes the current research posture into a two-track paper plan. The first track is the existing HCG-RVQ prototype claim: beta005 remains the broad fixed-checkpoint evidence package and should not be discarded when moving to EF-LIC/GLC. It has aligned gains across OpenImages trusted holdout4096, OpenImages transfer start8192, Kodak, and CLIC mobile/professional, with stable low Householder displacement and zero nonfinite rows.

The second track is VQ-LIC plug-in evidence. EF-LIC is the cleaner immediate target because the official checkpoint and RVQ-like bottleneck path are locally available, and E160-E194 already show exact active/decode consistency plus decoder-reproducible reliability control. The current safe hierarchy is: E190 scalar multi-objective selector as the controlled default, E193 as learned-head headroom with overfit warning, and E194 as the direct codec-path implementation of that head. The learned head should only become the main method after independent fit/eval.

GLC remains a main candidate, but E168-E183 make the design stricter: the target is not dense VQ replacement. GLC needs sparse active residual states, scalar fallback, local or hyperprior-generated quantizer geometry, index-rate accounting, and decoder/perceptual-aware training. This makes it a stronger low-bitrate generative benchmark, but a heavier implementation path than EF-LIC.

The practical blocker is data, not the EF-LIC direct code. The workspace currently has only Kodak image directories under `experiments/data`; independent non-Kodak fit/eval directories are absent. Therefore, the next paper-facing EF-LIC experiment should first populate or mount an independent fit split, generate E161-style active/fallback labels there, fit both E190 scalar and E194 learned controllers, and evaluate on held-out Kodak/Tecnick/CLIC with exact encoder/decoder matching.


## EF-LIC Independent Transfer After E196-E201

E196-E201 move the EF-LIC evidence from same-Kodak diagnostics toward a real VQ-LIC plug-in protocol. The conceptual result is sharper than just another metric row: fixed projected-HCG geometry is not safe when applied everywhere, but it creates useful per-image alternatives that a decoder-reproducible controller can select.

The active branch itself is mixed on independent OpenImages labels. On OpenImages64, always-active HCG worsens the average (`dDISTS=+0.000582`, `dLPIPS=+0.000153`), but there are many useful cases (`29/64` DISTS-positive, `31/64` LPIPS-positive, `16/64` both-positive). This supports the HCG-RVQ hypothesis in a controlled way: local geometry can help a subset of images, but low-bitrate perceptual compression needs reliability control rather than unconditional quantizer perturbation.

The strongest positive transfer result is E198. A learned reliability head trained on OpenImages16 labels and evaluated through the actual EF-LIC path on Kodak24 improves both DISTS and LPIPS (`dDISTS=-0.000350`, `dLPIPS=-0.000036`) at the same bpp, while preserving exact encoder/decoder selector agreement and exact active decode reproduction. This is important because it is not a same-table Kodak selector. It is still too small for a final paper claim, but it validates the fit-on-non-Kodak/apply-to-Kodak protocol.

E199/E200 explain why the next step must include calibration. With more OpenImages labels, the head becomes too conservative when transferred to Kodak (`branch_share=0.042`) and misses many active-good Kodak cases. E201 shows that a lower threshold would improve both DISTS and LPIPS on Kodak, so the score is not useless; the threshold is miscalibrated under distribution shift. This is a useful failure mode: the publishable method should have an independent calibration split or a calibrated probability/temperature objective, not a threshold frozen only by OpenImages training loss.

The updated paper strategy is therefore conservative and stronger. EF-LIC should be presented as a VQ-LIC plug-in where HCG-RVQ supplies a state-preserving active geometry branch and a hyperprior/codebook-statistics reliability controller chooses when to use it. The immediate evidence is not yet final SOTA performance, but it is mechanistically aligned, codec-valid, and now has a clear path to paper-facing evaluation: train on OpenImages, calibrate on disjoint OpenImages/CLIC, evaluate on Kodak/Tecnick/CLIC, and report active/fallback failures by feature distribution and per-image metric tradeoff.


## EF-LIC Calibration Transfer After E202-E207

E202-E207 test the next obvious fix after E200: choose the reliability threshold on an independent calibration split instead of using the threshold learned from the OpenImages fit labels. This is exactly the right scientific protocol, but the result is negative for the current controller.

For the OpenImages64-trained head, calibration on disjoint OpenImages64 works on its own split. The raw threshold already gives slight DISTS/LPIPS improvement, and the calibration best-score threshold strengthens DISTS while keeping LPIPS near zero. However, applying that calibrated threshold to Kodak24 loses the benefit (`dDISTS=+0.000034`, `dLPIPS=+0.000071`). The same pattern holds for the OpenImages16 head: calibration chooses a conservative threshold, but applying it to Kodak selects harmful cases and gives `dDISTS=+0.000221`, `dLPIPS=+0.000022`.

This updates the interpretation of E198. E198 is a real and useful positive smoke because it is non-Kodak fit to Kodak eval with exact codec reproducibility. But the three-way protocol shows it is not yet robust enough for a paper claim. The likely issue is not bitstream validity or NaN instability; those checks are all clean. The issue is reliability transfer under domain shift: the probability ranking and threshold learned/calibrated on OpenImages do not consistently identify the same active-good cases on Kodak.

The next research move should be diagnostic rather than another blind threshold. Compare probability and feature distributions between OpenImages fit, OpenImages calibration, Kodak, and ideally CLIC/Tecnick. The candidate signals are already visible in the learned coefficients (`z_index_*`, `z_hat_*`, `slice0_mean/scale_*`). If the distributions shift, use feature normalization or domain-mixed calibration. If rankings are weak even after normalization, the controller objective should change: stronger false-positive penalties, LPIPS-specific labels, or a small calibrated head trained with held-out calibration loss rather than only label classification.

This is still progress toward an international-conference claim because it prevents over-claiming a lucky split and keeps the method story precise. The claim should not be “active HCG always improves EF-LIC.” It should be “HCG creates a valid alternative quantizer geometry, and robust reliability control is required to harvest it in VQ-LIC low-bitrate compression.” The next positive result must come from a controller whose calibration transfers across datasets, not from a same-domain threshold.


## EF-LIC Reliability Shift And Active-State Retuning After E208-E210

E208 resolves the ambiguity left by E202-E207. The OpenImages-trained reliability head is not failing because the EF-LIC HCG branch is non-deterministic, and not because the machine produced NaNs. It is failing because the decoder-safe features and probability calibration shift across datasets. The proof is concrete: OpenImages fit has `prob_mean=0.422`, OpenImages calibration has `prob_mean=0.410`, but Kodak drops to `prob_mean=0.310`; Kodak ranking is random (`AUC=0.500`) and even slightly points the wrong way under the score correlation. The selected Kodak row is harmful. Meanwhile, encoder/decoder selector decisions match exactly and active decode diff is zero.

This changes the controller story. Independent calibration on a same-domain OpenImages split is not enough if the final eval domain has lower latent/context statistics. The controller either needs domain-mixed calibration, feature normalization tied to codec statistics, or a more conservative objective that explicitly penalizes false positives under LPIPS as well as DISTS. A learned head should not become the paper method until it survives this transfer audit.

E209 is the more constructive result. It shows that the active HCG geometry itself can be made more stable. The previous `mean/alpha=0.05` branch gave useful Kodak DISTS but weak LPIPS. Sweeping `alpha` and direction reveals that smaller `mean` geometry is more transfer-safe. `mean/alpha=0.02` improves both DISTS and LPIPS on OpenImages calibration32 and Kodak24 while preserving unchanged bpp, exact decode reproducibility, and zero nonfinite rows. This is a better candidate for a full VQ-LIC plug-in row than the old `alpha=0.05` branch.

E210 then prevents over-claiming. On the OpenImages64 fit split, `mean/alpha=0.02` still has mixed always-active behavior (`dDISTS=+0.000173`, `dLPIPS=-0.000017`) despite good oracle headroom. A head trained on those labels transfers to Kodak worse than all-active. This means the active-state update is real, but the current reliability head is still not the right publishable controller. The next reliable improvement path is: first validate `mean/alpha=0.02` over more held-out sets; second, train reliability control with domain-mixed or normalized features; third, only then move to full EF-LIC/GLC training or paper-level RD curves.

The international-conference claim is therefore stronger but more disciplined: HCG-RVQ is not yet a final SOTA row in EF-LIC, but it has codec-valid active geometry, measurable low-bitrate perceptual gains on held-out data, and a diagnosed bottleneck. The next result needed for a clean paper claim is not another Kodak-tuned threshold; it is a robust reliability mechanism that preserves the E209 active-state gains across Kodak, Tecnick, CLIC, and OpenImages-style validation.


## EF-LIC Cross-Dataset Active-State And Strength Headroom After E211-E213

E211 is an important reality check for the stronger EF-LIC active branch found in E209. The implementation result is clean: all additional datasets ran with GPU0 fixed, nonfinite rows are zero, bpp is unchanged, and active encode/decode reconstruction matches exactly. Therefore the remaining differences should be interpreted as quantizer-geometry effects, not CUDA/GPU instability or bitstream mismatch.

The performance result is nuanced. `mean/alpha=0.02` is not a universal always-active solution. It improves CLIC mobile on both perceptual metrics and improves DISTS on DIV2K/OpenImages, but it is average-harmful on CLIC professional and Tecnick at the same alpha. This does not invalidate the HCG-RVQ hypothesis. Instead, it sharpens it: hyperprior-conditioned geometry creates useful alternate RVQ states, but low-bitrate compression needs a reliability/strength policy to avoid images where the local geometry perturbation hurts perceptual reconstruction.

E212 separates two possible failure explanations. If the failure were simply “alpha too large,” then smaller alpha should rescue both harmful domains. That only happens partly. Tecnick becomes beneficial at `alpha=0.01`, but CLIC professional remains harmful for every tested alpha. So a fixed global alpha is too weak as a method. The controller should be image- or feature-conditioned and should include both strength selection and fallback.

E213 quantifies why this is worth doing. On the same failure splits, a per-image oracle over `alpha in {0, 0.005, 0.01, 0.015, 0.02, 0.03}` gives strong headroom: CLIC professional can reach `dDISTS=-0.000976`, `dLPIPS=-0.000333`; Tecnick can reach `dDISTS=-0.001016`, `dLPIPS=-0.000474`. The active fraction around `0.708` on both splits means the ideal policy is neither always-off nor always-on. This is exactly aligned with the HCG-RVQ claim: hyperprior geometry should be generated and then reliability-controlled, not blindly applied.

The updated paper strategy should therefore use E211-E213 as the bridge from smoke tests to a stronger method. The next publishable EF-LIC plug-in should be a decoder-reproducible strength controller: input features from z and early slice hyperprior context, output either `alpha=0` fallback or one of a small predeclared alpha values. It must be trained/calibrated on mixed non-final domains and evaluated on held-out domains. The ablation table should include original EF-LIC, always-active `mean/alpha=0.02`, best fixed alpha per dataset as diagnostic only, binary active/fallback controller, and strength controller. The expected paper claim is not just a metric gain; it is that HCG-RVQ exposes local geometry alternatives and a conservative controller harvests them at unchanged bpp.


## EF-LIC Mixed-Domain Strength Controller After E214-E216

E214-E216 answer an important design question left open by E211-E213: is the remaining problem simply choosing a better global alpha, or do we need a richer local reliability mechanism? The evidence points to the latter.

The good news is the HCG-RVQ mechanism still has strong headroom. Across the mixed 128-image pool, the per-image alpha oracle reaches `dDISTS=-0.001220` and `dLPIPS=-0.000471`, with a joint score of `-0.002631`. This is much stronger than any fixed-alpha row and shows that the local geometry perturbation is often useful. The active fraction is `0.773`, so the best policy is not a trivial all-off solution.

The limiting result is that current whole-image global predecision control is too weak. The best pooled non-oracle result is the decision-stump policy with score `-0.000399`, only a small improvement over fixed `alpha=0.01` (`-0.000281`) and far from the oracle. Leave-one-domain-out makes the limitation clearer: policies trained on other domains do not reliably rescue CLIC professional or Tecnick, and different domains prefer different simple policies. This means the issue is not merely linear-vs-nonlinear classifier capacity. It is that the available whole-image features are not carrying enough stable information to decide a single strength for every slice of an image.

This updates the main HCG-RVQ implementation plan. A paper-facing method should not be a global Kodak-style threshold, nor the current 22-feature whole-image controller. The next credible controller should be local and sequential: for each slice or RVQ stage, use the decoder-available local context immediately before the quantizer decision to choose fallback/strength, with the same deterministic rule on encoder and decoder. That would use the information that E160-style diagnostics show is most related to geometry and residual error, while remaining no-sidebit and aligned with the prompt thesis that hyperprior context generates quantizer geometry.

For the conference claim, E214-E216 should be written as controlled evidence and design selection: HCG projected geometry creates valid alternatives; fixed strength is not enough; oracle selection demonstrates substantial room; and whole-image global control is the wrong abstraction. The next ablations should compare global always-active, whole-image controller, local/slice controller, and oracle upper bound on the same EF-LIC low-bitrate protocol.

## EF-LIC Slice-Wise Strength Control After E217

E217 tests whether the E216 failure of whole-image strength control can be partially solved by a simpler intermediate mechanism: fixed per-slice HCG strengths. The implementation is intentionally conservative. The schedule is deterministic and shared by encoder and decoder, so it adds no side bits and remains bitstream-valid. It is a diagnostic bridge between scalar always-active geometry and a future learned local controller.

The result is useful because it separates three facts. First, the codec path is stable: both CLIC professional and Tecnick probes report zero nonfinite rows and zero active-forward/decode difference. Second, slice location matters. On Tecnick, late/local schedules improve both perceptual metrics (`slice2only020`: `dDISTS=-0.000218`, `dLPIPS=-0.000213`; `late020`: `dDISTS=-0.000200`, `dLPIPS=-0.000214`). On CLIC professional, stronger early/all schedules can improve DISTS but tend to hurt LPIPS, while `slice3only020` is safest but only tiny. Third, the oracle gap remains large: pooled fixed `late020` reaches only score `-0.000210`, while the per-image schedule oracle reaches score `-0.002382` (`dDISTS=-0.001016`, `dLPIPS=-0.000456`).

This is a stronger design result than simply saying “try another alpha.” Different domains and images prefer different slice locations, and no fixed schedule consistently captures the headroom. That supports the main HCG-RVQ thesis: hyperprior/context should generate or control local quantizer geometry. For EF-LIC, the next credible method is a no-sidebit local controller that chooses per-slice fallback/strength from decoder-available local context before the quantizer decision. The ablation ladder should now be original EF-LIC, scalar all-active, fixed slice schedules, local/context controller, and per-image oracle upper bound.

E217 should not be used as a final quality claim. It should be used as paper-facing mechanism evidence: HCG projected geometry exposes useful alternatives, fixed global control is inadequate, and local slice-aware reliability control is the right abstraction to harvest the low-bitrate perceptual gains.

## EF-LIC Local Slice Controller Diagnostics After E218-E220

E218-E220 sharpen the EF-LIC plug-in story in an important way. The positive part is clear: local HCG/RVQ alternatives have real headroom. Across 5 domains and 240 marginal slice decisions, the single-slice oracle reaches `score=-0.000735` with both DISTS and LPIPS improving on average. This is not a bitstream artifact: all newly generated E219 probes used GPU0, had no nonfinite rows, and had exact active/decode reproduction. The result remains aligned with `prompt.txt`: useful behavior appears when the quantizer geometry is controlled locally, not when one global image-level switch is applied everywhere.

The limiting part is equally clear. Simple decoder-safe controllers do not yet transfer well enough. The best pooled stump and ridge rows look reasonable on the same table, but leave-dataset-out exposes domain dependence. The ridge controller is especially diagnostic: same-table it can approach the oracle, but when trained on other domains it becomes harmful on CLIC professional, DIV2K, OpenImages, and Tecnick. That means the current small marginal labels are enough to reveal signal, but not enough to define the paper method.

This updates the research direction. The next serious HCG-RVQ strengthening step should not be another hand-tuned threshold. It should be a learned local HCG geometry/strength module with explicit fallback, trained on mixed non-final domains and evaluated on held-out domains. The controller should use only decoder-reproducible context, but it likely needs richer local inputs than global z statistics plus current slice summary scalars. Candidate inputs are local mean/scale maps, support/context buffers, z-index priors, slice id, and possibly stage residual statistics that are available sequentially at both encoder and decoder. The ablation ladder should then compare original EF-LIC, fixed global alpha, fixed slice schedule, hand-built local controller, trained local controller, and oracle upper bounds.

For paper claims, E218-E220 are controlled evidence rather than final performance rows. They support the mechanism claim that HCG geometry creates useful alternatives at no extra side bits, and they justify why the final method must be learned/local rather than fixed/global. They also protect the project from over-claiming: a same-table local controller is not enough for an international-conference result unless it survives independent mixed-domain training and full held-out evaluation.

## EF-LIC Spatial Local Quant-MSE Control After E221-E222

E221-E222 move the EF-LIC plug-in analysis from image-level and slice-level switches to spatial local decisions inside the quantizer. This is closer to the central HCG-RVQ claim in `docs/prompt.txt`: the hyperprior/context should generate useful local quantizer geometry, not merely choose a single global active state.

The positive evidence is strong. On 24,576 sampled spatial positions across CLIC professional, CLIC mobile, Tecnick, and OpenImages, projected HCG has a large local oracle opportunity even though applying it everywhere is harmful. Pooled all-on dMSE is `+0.000063`, but the local oracle is `-0.000458`, and every domain has an oracle around `-0.00043` to `-0.00048`. This means the HCG geometry is not just producing random perturbations; there are many local positions where the alternate geometry expresses the normalized residual better than the fixed EF-LIC RVQ state.

The negative evidence is also important. Decoder-safe hand-built features and a ridge score controller can exploit this signal only in same-table diagnostics. Same-table pooled ridge reaches `-0.000042` dMSE, but leave-dataset-out remains harmful on all held-out domains and leave-image-out remains positive. The feature correlations are weak and domain-dependent, so a scalar threshold or post-hoc linear policy would be a fragile paper claim.

The updated method direction should therefore be more ambitious and more faithful to HCG-RVQ. E221-E222 justify a trained local geometry/strength head: a no-sidebit module that uses decoder-reproducible local maps (`mean`, `scale`, support context, z/index priors, slice/stage position) to choose fallback or HCG strength before each local quantizer decision. This head should be trained on independent mixed-domain data with an explicit false-positive cost and evaluated with held-out full codec metrics. In the paper, E221-E222 should appear as mechanism/headroom evidence and as motivation for why fixed/global control is insufficient, not as final performance rows.

## EF-LIC Spatial Normalization Probe After E223

E223 tests whether the E221-E222 transfer failure is just a feature-normalization problem. The answer is mostly no. Image-relative and image/slice-relative features can change the same-table capacity row, and raw-plus-relative features produce the best same-table dMSE (`-0.000054`), but they do not produce robust held-out behavior. Leave-dataset-out remains harmful for almost every domain and feature mode, and leave-image-out remains positive for all tested modes.

This matters for the research direction. If relative normalization had rescued LODO, a simple deterministic controller might have been enough for the first EF-LIC paper row. Instead, E223 says the local decision boundary itself is not stable under post-hoc linear policies. The correct next step is not another scalar threshold, nor a slightly more elaborate normalized ridge. It is a trainable local HCG module that changes the quantizer geometry or strength during codec training, with explicit fallback and false-positive control.

For the paper, E223 should be framed as a negative but useful ablation: hand-built deterministic local controllers are insufficient even when decoder-safe normalization is allowed. This justifies the proposed learned hyperprior-conditioned geometry head and helps separate the main contribution from a fragile heuristic selector.

## EF-LIC Learned Spatial Teacher Head After E224

E224 tests whether the next step can be solved by simply replacing the post-hoc linear spatial controller with a small MLP trained on the E221 teacher labels. The result is useful but not sufficient for a paper method. Same-table rows improve, with the best feature mode reaching `-0.000050` normalized quantization-MSE versus the harmful all-on `+0.000063`. This means the local feature maps and labels contain learnable structure.

However, the transfer result is the deciding evidence. Leave-dataset-out evaluation is still mostly harmful: `raw_plus_image_rel` is positive on all four held-out domains, and `raw_plus_image_slice_rel` improves only CLIC professional while hurting CLIC mobile, OpenImages, and Tecnick. The failure is not a NaN/codec-validity issue; E224 ran as an offline diagnostic on finite E221 samples. The failure is generalization of the teacher-label decision boundary.

This updates the HCG-RVQ strengthening plan. The project should not spend more effort trying to make a standalone teacher-label selector paper-main through small architectural tweaks. The better path is to move the learned module into the codec training loop: a decoder-reproducible local head after the EF-LIC mean/scale prediction that outputs fallback or a small HCG strength, trained with the actual codec objective and an explicit penalty for false-positive active decisions. E224 therefore supports the mechanism claim and narrows the next implementation target, while also protecting the paper from over-claiming a same-table learned controller.

## EF-LIC Spatial Alpha-Map Codec Path After E225

E225 is the first result after E224 that puts local HCG strength control back into the actual EF-LIC encode/decode path. This matters because the previous teacher-label controllers could diagnose local headroom but failed transfer. E225 instead asks a stricter implementation question: can encoder and decoder both derive the same spatial fallback/strength map from already available local context, with no side bits and no reconstruction mismatch?

The answer is yes. The `zero` mode exactly reproduces the EF-LIC baseline, all tested key rows have unchanged bpp, and `max_decode_diff=0` on both CLIC professional and Kodak. This means the scaffold is bitstream-valid and suitable for the next learned codec module. It also directly matches the prompt thesis: hyperprior/context is being used to control local quantizer geometry, not merely to predict entropy parameters or an offline selector.

The metric result is small but directionally important. On the full 41-image CLIC professional validation split, the previous scalar-style all-on row is not ideal: it slightly hurts PSNR and LPIPS while improving DISTS only weakly. The decoder-safe local `prev_rms_top` rule improves PSNR and DISTS with nearly neutral LPIPS (`dPSNR=+0.003486`, `dDISTS=-0.000103`, `dLPIPS=+0.000015`) at unchanged bpp. It also reduces the actual y-index mismatch fraction from about `0.197` in all-on to about `0.0278`, which is exactly the kind of conservative false-positive control the E221-E224 diagnostics said was necessary.

Kodak shows the limitation of a fixed local rule. There, all-on `constant` remains best among the tested fixed policies (`dDISTS=-0.000835`, `dLPIPS=-0.000070`), while `prev_rms_top` is weaker though still slightly perceptually positive. This is not a contradiction; it is a design cue. Different splits/images need different strength policies, so the paper method should not hard-code `prev_rms_top`. The method should learn a decoder-reproducible local controller that can choose between fallback, all-on, local-sparse, and smaller strength.

The updated EF-LIC path is therefore:

1. Keep E225 alpha-map as the codec-valid implementation scaffold.
2. Train a local HCG head after EF-LIC `_mean_scale(support_buf, i)` using decoder-safe maps: current mean/scale, previous decoded support, support-buffer summaries, slice id, and hyper/index features.
3. Include explicit false-positive control and fallback because all-on is useful on Kodak but harmful on harder CLIC-style images.
4. Evaluate with CLIC professional as the main CLIC split, while treating CLIC mobile as robustness or appendix evidence rather than the main paper table.

For the conference story, E225 is stronger than another smoke test: it closes the gap between offline headroom and a codec-valid mechanism. It does not yet claim a large RD gain. It establishes that the next trained HCG-RVQ module can be placed in the bitstream path and that local geometry control can improve a difficult CLIC professional split without changing bpp.

## EF-LIC Spatial Strength Selection After E226

E226 checks whether the E225 CLIC professional improvement was tied to a single arbitrary strength. It is not. Weakening the local `prev_rms_top` rule from `alpha=0.02` to `alpha=0.01` makes the CLIC professional row better under the joint perceptual score: `dDISTS=-0.000193`, `dLPIPS=-0.000009`, and `score=-0.000220` at unchanged bpp and exact decoder reproduction. The y-index mismatch fraction also drops to about `0.016`, so the improvement is consistent with the false-positive-control story from E221-E224.

The same change does not simply dominate everywhere. On Kodak24, `constant/alpha=0.02` remains the best tested fixed policy under `DISTS + 3*LPIPS`, while the weaker local rule is beneficial but smaller. This is an important positive/negative pair: HCG geometry can improve low-bitrate EF-LIC, but the desired geometry strength is dataset/image dependent. A fixed universal local-sparse heuristic would be too weak for Kodak-like cases and too risky in all-on form for CLIC professional.

The paper-facing design implication is now clear. The EF-LIC plug-in should be framed as learned hyperprior/context-conditioned strength selection, not as one handcrafted alpha map. The current scaffold already proves that such a selector can live in the codec path without side bits. The next implementation should therefore train a local head to choose among fallback, weak local geometry, and stronger all-on/local geometry using decoder-safe features. The ablation ladder should include original EF-LIC, all-on scalar HCG, fixed local alpha maps, learned strength/local controller, and oracle upper bounds.

For dataset protocol, CLIC mobile should not drive the main claim. CLIC professional is the more relevant CLIC stress test for paper-facing validation here, while CLIC mobile can remain an appendix robustness split. Kodak stays as the standard small benchmark and as evidence that HCG can give perceptual gains when aggressive geometry is safe.

## EF-LIC Candidate Oracle and Selector Limits After E227-E228

E227-E228 make the current EF-LIC plug-in story sharper. Adding `alpha=0.005` shows that CLIC professional benefits from very conservative local geometry: `prev_rms_top@0.005` is the best fixed CLIC row among the tested rules, with both DISTS and LPIPS improving and a very low y-index mismatch fraction. Kodak shows the opposite boundary: the same weak local control is too small, and stronger all-on `constant@0.02` remains best. This is a clean empirical reason not to turn any one fixed rule into the proposed method.

The candidate oracle is now large enough to matter for a paper direction. Over CLIC professional and Kodak, choosing the best codec-valid candidate per image reaches score `-0.003292`, far beyond the best fixed pooled candidate `-0.000234`. The oracle choices are diverse: constant geometry, weak/strong previous-support local maps, support-rms local maps, and occasional fallback are all selected. This is exactly the HCG-RVQ hypothesis in operational form: the hyperprior/context should generate and select local quantizer geometry conditioned on the image, not apply one global geometry everywhere.

The negative result is equally important. A same-table stump can harvest some of the headroom (`-0.001039`), but leave-dataset-out transfer remains fragile and even harmful on CLIC professional. This means a post-hoc hand-built selector is not the right method. The right next step is a codec-trained local head, because the useful decision boundary appears to depend on local residual/context behavior and perceptual consequences that do not transfer from a tiny offline table.

For the conference claim, E227-E228 should be used as mechanism and design-selection evidence. They support the following safe claim structure: HCG-RVQ creates multiple valid quantizer-geometry states at unchanged bpp; fixed states already improve some splits; per-image oracle shows large headroom; and a learned local controller is required to robustly harvest that headroom. The next publishable experiment is therefore not another smoke test but a trained decoder-reproducible strength/local controller evaluated on held-out CLIC professional, Kodak, Tecnick, and ideally an independent training/calibration split.

## EF-LIC Continuous Alpha-Map Candidate Analysis After E229

E229 intentionally challenges the current sparse top-k design. Instead of activating exactly the strongest 25% positions, it tests smooth alpha maps whose strength is proportional to decoder-reproducible local context. This is closer to what a learned HCG head might output, and it probes whether the fixed sparse map was too rigid.

The result is a useful split. As a fixed policy, smooth all-position local maps are not reliable enough. On CLIC professional, the soft support/previous-support maps often improve DISTS but introduce LPIPS regressions, so their joint score is positive/harmful. On Kodak, `support_over_scale_top_soft@0.02` is promising for DISTS (`-0.000888`) and improves the score (`-0.000828`), but still does not beat the known all-on `constant@0.02` fixed row (`-0.001047`). Therefore, simply making the alpha map continuous is not the missing final method.

As a candidate branch, however, E229 is clearly positive. The soft candidates increase the per-image oracle from `-0.003292` to `-0.003613` pooled over CLIC professional and Kodak. The gains appear on both datasets: CLIC professional improves to `-0.002664`, and Kodak improves to `-0.005233`. Oracle choices include soft candidates on 14 of 65 images, including `support_over_scale_top_soft`, `support_rms_top_soft`, and `prev_rms_top_soft`. This means smooth hyperprior/context-conditioned strength adds genuinely useful geometry states that the earlier fixed candidate set did not contain.

The paper-facing interpretation should be disciplined: E229 strengthens the argument for a learned local controller, not for another handcrafted rule. A strong HCG-RVQ/EF-LIC module should expose multiple decoder-safe quantizer geometry states and learn when to use each. The next controller should include at least these branch families: zero fallback, weak sparse previous-support, stronger all-on, sparse support, and soft support/scale-conditioned strength. The objective should penalize LPIPS/DISTS false positives because the same soft branches that help oracle headroom can hurt when applied everywhere.

## EF-LIC Weak Continuous Alpha-Map Analysis After E230

E230 tests whether the E229 soft-alpha failure as a fixed policy was mainly a strength problem. The answer is partly yes. On CLIC professional, reducing soft alpha from `0.02` to `0.01` turns `prev_over_scale_top_soft` into a useful fixed candidate: it improves both DISTS and LPIPS at unchanged bpp and exact decoder reproduction. This is significant because it shows smooth local strength maps are not merely oracle artifacts; at the right scale they can be safe on the harder CLIC professional split.

The result is not a universal fixed-rule solution. Kodak24 moves in the opposite direction: the weak soft maps do not beat the existing strong all-on `constant@0.02` row. The useful geometry state depends on image/domain risk. This keeps the paper-facing claim disciplined: HCG-RVQ should not be presented as one fixed alpha schedule. It should be presented as a hyperprior/context-conditioned quantizer-geometry controller that learns when to use weak local geometry, smooth local geometry, stronger all-on geometry, or fallback.

The oracle result strengthens that design. Adding `alpha=0.01` soft branches raises the pooled oracle score to `-0.003649`, with CLIC professional at `-0.002696` and Kodak at `-0.005277`. The newly selected weak soft branches show that smooth strength is a real candidate family, while the continued dominance of different candidates across images shows why hand-coded fixed selection is inadequate.

The next publishable implementation step should therefore be a codec-trained decoder-reproducible local HCG head after EF-LIC `_mean_scale(support_buf, i)`. It should output a small strength or mixture over the candidate families, with explicit false-positive regularization for perceptual metrics. A short-cycle training run can first use the E225/E230 branch set as fixed basis states, but the paper-main claim will require held-out full codec evaluation and, if promising, full training or faithful fine-tuning under EF-LIC/GLC protocol.

## EF-LIC Lower Soft Alpha Analysis After E231

E231 closes the immediate soft-strength question. Reducing continuous local geometry to `alpha=0.005` reduces risk but also removes much of the useful DISTS movement. On CLIC professional, `prev_over_scale_top_soft@0.005` is LPIPS-safe but weaker than the same branch at `0.01`. On Kodak24, the weak soft branches are consistently too small compared with stronger all-on geometry.

This reinforces the current HCG-RVQ design thesis rather than weakening it. The useful geometry states form a library with different risk profiles: sparse previous-support at very low active rate is the safest CLIC fixed rule; smooth previous-support at `0.01` is a competitive continuous branch; stronger constant/support branches are needed for Kodak-like images; and very weak soft states only help selected images. A hand-coded universal policy cannot capture this structure.

The paper-facing next step should therefore move from branch discovery to branch control. The controller should be decoder-reproducible, trained in the codec loop, and regularized to avoid false-positive perceptual damage. The candidate branch set after E231 should include: zero fallback, sparse `prev_rms_top@0.005/0.01`, strong `constant@0.02`, support-rms sparse branches, and soft previous/support-over-scale branches at `0.005/0.01/0.02`. The full evaluation claim should be made only after this controller is trained and tested on held-out splits with the same codec-valid checks used here.


## EF-LIC Branch Library Audit After E232

E232 converts the E225-E231 codec-valid alpha-map experiments into a branch-library design audit. The important conclusion is not that the oracle itself is publishable; it is that the useful HCG-RVQ geometry states are diverse and non-redundant. Pooled over CLIC professional 41 and Kodak24, the best fixed candidate is only `-0.000234` under `DISTS + 3*LPIPS`, while the per-image oracle over the same no-sidebit codec-valid branches reaches `-0.003725`. This gap is too large to ignore, but it also cannot be claimed with a post-hoc oracle.

The family analysis explains why a single fixed policy keeps failing. CLIC professional first wants safer support-conditioned local behavior: greedy family selection starts with `soft_support` and the strongest fixed CLIC row is the conservative `prev_rms_top@0.005`. Kodak first wants stronger all-on geometry: greedy selection starts with `constant`, and the best fixed Kodak row is `constant@0.02`. The risk correlations reinforce the same point. On CLIC, more y-index mismatch and larger geometry perturbation correlate with worse score; on Kodak, they correlate with better score. This is exactly the false-positive-control problem that the trained HCG head must solve.

The leave-one-family-out table gives the branch vocabulary for the next implementation. Removing `constant` hurts pooled oracle by `+0.000483`; removing `soft_support` hurts by `+0.000257`; removing `sparse_support`, `soft_prev`, and `sparse_prev` still causes measurable losses. Therefore the next EF-LIC module should not be another scalar alpha sweep. It should be a decoder-reproducible controller after `_mean_scale(support_buf, i)` that can choose among zero fallback, aggressive constant geometry, soft support/scale local geometry, soft previous-context geometry, sparse support, and sparse previous-context.

For the paper, E232 strengthens the mechanism and design-selection story. HCG-RVQ creates multiple codec-valid local quantizer geometry states at unchanged bpp, and the right state depends on decoder-available context and image/domain risk. The remaining publishable step is to train this branch controller in the codec loop, with explicit false-positive regularization and held-out full evaluation. Short-cycle branch audits should now serve as ablations and safety checks, not as the final performance claim.


## EF-LIC Decoder-Safe Branch Readiness After E233

E233 is an important guardrail before implementing the learned EF-LIC HCG branch controller. The central question was not whether a richer post-hoc classifier can overfit the E232 oracle; it was whether the branch decisions are readable from information that the decoder can also reproduce before quantization. The answer is yes, but not in a form that is safe enough as a final method.

The positive evidence is that decoder-safe predecision context is informative. Using only z-hat/z-index statistics and per-slice mean/scale/support/previous-support summaries, same-table ridge score prediction reaches `-0.003251` under `DISTS + 3*LPIPS`, close to the E232 oracle score `-0.003725` and far better than the best pooled fixed branch `-0.000234`. This says the branch library is not random and the right HCG geometry state leaves a measurable signature in hyperprior/context features. The feature-separation table also points to plausible mechanism variables: early support RMS dispersion, support tails, scale statistics, and later support-over-scale or prev-over-scale maps. These are all consistent with the prompt's thesis that hyperprior/context should control local quantizer geometry.

The negative evidence prevents over-claiming. The same ridge predictor falls to `-0.000122` in pooled leave-one-image-out, and leave-dataset-out transfer is not robust: training on CLIC professional and testing Kodak is about neutral/harmful, while training Kodak and testing CLIC gives only a small improvement. Even a true-family upper diagnostic is asymmetric: if the oracle family were known but the best family member were selected from the other dataset, it helps CLIC (`-0.001466`) but barely helps Kodak (`-0.000139`). This shows that both family selection and strength selection are domain/image dependent.

The design implication is now quite sharp. A hand-built image-level selector, even with safe features, is not a conference-safe main method. The next credible HCG-RVQ/EF-LIC implementation should train a decoder-reproducible local controller inside the codec loop, not fit a small table after the fact. It should expose the E232 branch vocabulary: zero fallback, aggressive constant geometry, soft support/scale local geometry, soft previous-context geometry, sparse support, and sparse previous-context. It should also include explicit false-positive control because CLIC and Kodak react oppositely to geometry perturbation.

For the paper story, E233 should be used as mechanism and implementation-safety evidence. It proves that the proposed controller can be fed only decoder-safe context, and it explains why a learned local branch/strength head is necessary. It should not be used as a final performance row.

## EF-LIC Branch-Controller Scaffold After E234

E234 is the first EF-LIC step that turns the E232 branch-library conclusion into
a reusable controller interface. Its value is implementation safety rather than
new final performance. The script exposes a compact no-sidebit HCG vocabulary
inside the same sequential support-buffer loop used by EF-LIC compression and
decompression: zero fallback, strong constant geometry, sparse previous/support
local geometry, and smooth previous/support-over-scale geometry. All decisions
are functions of decoder-reproducible context, so this is the right interface
for a trained branch/strength head.

The contract checks are strong. On Kodak24 and CLIC professional 41, every E234
row has unchanged bpp, exact decoder reproduction (`max_decode_diff=0`), and no
nonfinite rows; GPU0 was used exclusively. The zero preset exactly reproduces
the EF-LIC baseline. This matters because it rules out a class of false
explanations: the observed gains or losses are not caused by side bits, decoder
drift, CUDA device-1 NaNs, or evaluation-path mismatch.

The fixed-preset behavior also preserves the core mechanism diagnosis. Kodak24
favors stronger geometry (`constant020` score `-0.001047`, `soft_support020`
score `-0.000828`), while CLIC professional favors conservative local
previous-context geometry (`sparse_prev005` score `-0.000248`, `soft_prev010`
score `-0.000222`). CLIC's all-on `constant020` is harmful under the joint score
(`+0.000435`). This reinforces the paper-safe interpretation: HCG geometry is
useful, but the method must learn when to activate which geometry state. A
universal all-on rule would be too risky, and a universal weak rule leaves Kodak
headroom unused.

The E234 summary makes the controller need quantitative on the executable
vocabulary itself. The pooled best fixed preset is only `-0.000234`, but the
per-image oracle over the same seven presets reaches `-0.002815` with a
`0.954` score-win fraction. The oracle selects all active families plus zero
fallback (`constant=18`, `soft_prev=13`, `soft_support=14`, `sparse_prev=11`,
`sparse_support=6`, `zero=3` pooled), so the compact branch set is already
nontrivial enough for a learned controller.

The next publishable step is therefore not another preset sweep. E235 should
train a decoder-reproducible local controller after `_mean_scale(support_buf, i)`
that outputs this same vocabulary or a differentiable relaxation of it. The loss
should include actual reconstruction/perceptual terms plus an explicit
false-positive/fallback cost, because the harmful CLIC rows show that local
geometry mistakes are the main failure mode. E234 becomes the ablation and
safety scaffold for that method.

## EF-LIC Compact Controller Readiness After E235

E235 tightens the transition from branch discovery to branch control. Unlike
E233, it uses only the compact E234 preset vocabulary that already passed the
codec-valid no-sidebit contract. This makes the result directly relevant to the
next implementation interface: `zero`, aggressive constant geometry, sparse
previous/support geometry, and soft previous/support-over-scale geometry.

The good news is that decoder-safe context contains strong signal. In pooled
resubstitution, ridge score prediction over all safe features reaches score
`-0.002548`, close to the compact oracle `-0.002815` and much stronger than the
best fixed preset `-0.000234`. The leading family-separating features are not
arbitrary dataset IDs: they are support/previous-support RMS statistics,
mean-RMS statistics, and z-index usage. This is mechanistically aligned with the
HCG thesis that hyperprior/context should control local quantizer geometry.

The bad news is exactly the guardrail needed for a conference-safe method. The
same selector does not survive held-out stress: pooled leave-one-image-out is
only about `-0.000116` at best, and cross-domain transfer is asymmetric and
fragile. Training on Kodak then testing CLIC is weak or harmful even with
fallback, while training on CLIC then testing Kodak recovers only part of the
headroom. Therefore, the observed E234 oracle cannot be converted into a paper
claim by a small post-hoc image-level selector.

The most useful diagnostic is the true-family upper bound. If the oracle family
is known and only the within-family preset is chosen from training images, the
score is close to oracle (`-0.002721` vs `-0.002815` pooled). That means the
main unsolved problem is reliable family/activation selection, not fine
selection inside a family. This points directly to a learned local controller
with false-positive control: the controller must learn where geometry is safe,
when to fall back to zero, and whether a position resembles the aggressive
Kodak-like regime or the conservative CLIC-like regime.

Paper implication: E235 should be cited as a negative-result design audit, not
as a performance row. It strengthens the narrative that HCG-RVQ is not merely a
static alpha trick. The publishable method should train a decoder-reproducible
local branch/strength head in the codec loop after EF-LIC `_mean_scale`, using
actual reconstruction/perceptual losses and fallback regularization. Full
performance claims still require held-out full evaluation and eventually
faithful EF-LIC/GLC training or fine-tuning.

## EF-LIC Local Controller-Map Analysis After E236

E236 is a useful pressure test for the current HCG-RVQ direction because it
moves beyond the E234 fixed preset interface without yet claiming a trained
controller. The implementation composes local alpha maps from decoder-safe
support/mean/scale context inside the actual EF-LIC codec loop. The important
contract is clean: Kodak24 and CLIC professional 41 both completed with
unchanged bpp, exact forward/decode reconstruction (`max_decode_diff=0`), and no
nonfinite rows on GPU0 only. This means the observed improvements and failures
are quantizer-geometry effects, not bitstream, decoder-drift, or CUDA-device
artifacts.

The fixed-policy result sharpens the all-on false-positive story. On CLIC
professional, aggressive all-on `constant020` is harmful on average
(`score=+0.000435`) even though its DISTS term alone is slightly negative. The
LPIPS penalty dominates. Guarding the same alpha by decoder-safe support/scale
top-25% turns it into a positive fixed CLIC row (`-0.000263`) while cutting
mismatch from roughly `0.197` to `0.073`. So the CLIC failure is not that HCG
geometry is intrinsically bad; it is that uncontrolled geometry is applied in
places where the perceptual decoder is fragile.

Kodak24 shows the other half of the tradeoff. `constant020` remains the best
fixed policy (`-0.001047`), while the best local/guarded policy
`guarded_support020_top50` reaches only `-0.000485`. In other words, the same
false-positive guard that rescues CLIC removes useful Kodak geometry strength.
This is exactly the image/domain-dependent regime split seen in E232-E235, now
confirmed with a richer local controller-map interface.

The E236 per-image oracle is the strongest design signal. It reaches
`-0.002913` pooled over 65 images, slightly above the E234 compact oracle
(`-0.002815`). The selected policies are diverse: all-on constant is still
chosen 20 times, but guarded constant is chosen 13 times, guarded support 9
times, soft blends 11 times, sparse union 6 times, hybrid once, and zero 5
times. Therefore E236 adds useful basis states, but a fixed hand-coded policy is
still not enough. The paper-main method should be a learned decoder-reproducible
local branch/strength controller, not a manually chosen local map.

The risk correlations make the training objective clearer. On CLIC,
`y_mismatch_frac`, `y_avg_index_entropy`, and `y_avg_index_used_frac` correlate
positively with the score, meaning more perturbation tends to be worse. On
Kodak, `y_mismatch_frac`, alpha mean, and geometry RMS correlate negatively,
meaning stronger perturbation often helps. A single monotonic risk rule cannot
serve both domains. The learned controller should include a zero/fallback path,
a false-positive regularizer, and possibly a domain-agnostic confidence loss
that penalizes LPIPS/DISTS regressions more strongly than it rewards small
index-entropy or PSNR movement.

For the conference story, E236 should be used as implementation and mechanism
evidence. It proves that richer local HCG basis states can be represented
without side bits and that guarded geometry can rescue CLIC-like failures. It
also prevents over-claiming: until a learned controller harvests the oracle
headroom on held-out splits, the fixed E236 rows are ablations rather than final
performance claims. The next experiment should train this local controller in
the codec loop, using E236 policies as basis states or differentiable relaxed
branches, then evaluate on Kodak24, CLIC professional, Tecnick, and at least one
independent calibration/validation split.

## EF-LIC Local Policy Controller Split Analysis After E237

E237 is the cleanest split-aware diagnostic so far for the E236 local-policy
vocabulary. It keeps the codec-valid E236 contract unchanged and asks a narrow
question: can a shallow controller choose among already executable local HCG
policies using only decoder-safe information? The answer is mostly no for a
post-hoc image-level regressor, but the failure mode is informative rather than
discouraging.

The shallow long-ridge selector fails the conference-safety bar. Same-table
resubstitution gives small gains over the best fixed policy, but leave-one-image
out turns positive (`+0.000505` to `+0.000575` for the fallback rows highlighted
in the report), and cross-dataset transfer is either neutral or harmful. This is
the same warning seen in E235, now repeated on the richer E236 local map space:
image-level feature fitting can see some same-table signal but does not produce
a robust method. It should not be promoted as HCG-RVQ.

The true-family upper bound changes the interpretation. When the oracle family
is supplied and only the within-family policy is chosen from training images,
the method nearly recovers the E236 oracle: pooled LOIO `-0.002675` vs oracle
`-0.002913`, CLIC->Kodak `-0.003587` vs `-0.004095`, and Kodak->CLIC
`-0.002114` vs `-0.002220`. The family labels are coarse and paper-meaningful:
zero fallback, aggressive constant geometry, guarded constant, guarded support,
soft blend, sparse union, and hybrid. Therefore the bottleneck is not the exact
alpha within a known family; it is robustly deciding which geometry family is
safe and useful.

This reinforces the main HCG-RVQ thesis. Hyperprior/context features such as
`z_index_used_frac`, `z_index_perplexity`, scale-RMS maxima, and mean/scale
distribution tails separate the oracle families, so the side information does
contain quantizer-geometry control signal. But a global image-level linear score
model is too weak and too domain-biased. The controller needs to operate at the
local slice/map level, consume the actual decoder-side support/mean/scale maps,
and be trained with an explicit asymmetric loss that penalizes false-positive
geometry activation more heavily than missed small gains.

For the paper path, E237 is a valuable negative-result ablation and a design
selection result. It justifies an ablation ladder of original EF-LIC, fixed
all-on HCG, guarded/fixed local policies, shallow split selector, learned local
family/strength head, and E236 oracle. The learned head should be evaluated with
independent fit/calibration/eval splits before final Kodak/Tecnick/CLIC full
evaluation.


Nearest-family classification confirms that the issue is not only ridge score
regression. Even when the selector directly predicts the coarse HCG family and
then uses the best train-side policy within that family, global image-level
features do not generalize: pooled LOIO remains positive (`+0.000222` for
nearest-family top16, `+0.000315` for top64), and cross-dataset transfer remains
far from the true-family upper bound. Therefore E237 narrows the design target
further: the paper-main controller should not be a global classifier over image
summary features. It should be a local neural head over support/mean/scale maps,
trained to output family/strength logits with zero fallback at slice or spatial
resolution.

## EF-LIC Teacher Label Margin Analysis After E238

E238 answers a different question from E237: not whether a shallow selector can
already solve the task, but whether the E236 oracle can be converted into sane
supervision for a learned in-codec controller. The answer is yes, with one
important caveat: the supervision should be confidence-gated and fallback-heavy.

The positive evidence is that most oracle wins are not microscopic. Pooled
oracle score is `-0.002913`, and `83.1%` of images have at least `5e-4`
improvement over zero. This is large enough to justify a learned HCG branch,
especially because the true-family positive control in E237 already showed that
coarse family selection nearly recovers the oracle. The HCG state is therefore
not just a brittle table lookup; there is a meaningful family/activation signal
to learn.

The caution is that family labels are not always clean. Pooled mean family
margin is `0.001077`, but `36.9%` of images have family margin below `5e-4`.
Those cases should not drive the first training stage. If high-gain and
moderate-margin labels are used first, the headroom remains high: using only
images with gain >= `5e-4` and family margin >= `5e-5` activates `73.8%` of
pooled images while preserving `91.2%` of oracle headroom. This is exactly the
shape wanted for a paper method: a conservative controller can still capture
most of the benefit while reducing harmful activations.

Wrong-family costs define the loss. True-zero images are especially sensitive:
activating hybrid, soft-blend, or constant families gives roughly `+0.0045` to
`+0.0051` positive score. True soft-blend and sparse-union images also suffer
when forced into constant geometry. Therefore the learned head should not be
trained with ordinary symmetric multiclass cross entropy alone. It needs an
asymmetric objective: explicit zero/fallback target, high penalty for nonzero
false positives, lower penalty for falling back on low-margin gains, and
possibly family-pair costs from `e238_eflic_teacher_label_margins.family_costs.csv`.

This connects cleanly to the HCG-RVQ thesis. Hyperprior/context should generate
quantizer geometry, but only when the local evidence supports it. E236 provides
codec-valid geometry basis states, E237 shows family selection is the bottleneck,
and E238 gives the training curriculum and cost structure for the learned local
head. The next result that can become paper-main is not another post-hoc
selector; it is an EF-LIC forward/decompress implementation of that local
confidence-gated family/strength head.

## EF-LIC Trainable Local Head Plan After E239

E239 is the first concrete bridge from oracle audits to a trainable EF-LIC
module. The new local head consumes only information available at the decoder
side immediately after `_mean_scale`: current mean/scale maps, support-prefix
statistics, previous decoded-y statistics, and slice id. This matters for the
paper claim because the controller is not allowed to inspect raw encoder
residuals, oracle policy labels, or validation-time distortion signals at
inference.

The label design is conservative but promising. Using E238 thresholds of
`gain >= 5e-4` and `family_margin >= 5e-5`, the E239 target manifest preserves
`91.2%` of pooled oracle headroom while leaving `26.2%` of images at explicit
zero fallback. This is a better first objective than all-on HCG: it captures
most of the measurable opportunity while directly encoding the empirical fact
that false-positive geometry is expensive.

The class distribution also guides the next ablations. Constant and zero are
well represented; guarded constant, guarded support, and soft blend are present
but smaller; sparse union is rare; hybrid has no high-confidence labels under
the current curriculum. Therefore the first learned-head result should report
both performance and class-behavior diagnostics: zero activation rate, family
confusion, false-positive nonzero rate on low-margin images, and per-family RD
impact. If hybrid is never selected, that is not a failure; it may simply be an
oracle-only diagnostic family rather than a paper-main branch.

The next paper-facing test should be strict. First, train only
`LocalHCGFamilyHead` while EF-LIC and the HCG basis states remain frozen. Then
run codec-contract checks on GPU0: forward/decompress agreement, finite outputs,
unchanged or explicitly accounted index-rate behavior, and per-image RD/LPIPS/
DISTS deltas. Only after the frozen head has held-out false-positive control
should the experiment move to spatial labels, strength regression, or full
fine-tuning.

## EF-LIC Context Export Analysis After E240

E240 closes an important implementation gap. Before this point, E239 had a
trainable head definition and an image-level teacher manifest, but no exported
codec-path local tensors. The new exporter proves that the exact EF-LIC
insertion point can provide finite decoder-safe context maps on real Kodak data.
This is stronger than a synthetic head smoke because it exercises `g_a`, `h_a`,
z quantization, `h_s`, EF-LIC support-buffer construction, `_mean_scale`, and
original decoded-prefix updates.

The first context tensor has shape `[4, 11, 16, 24]`, matching four EF-LIC
slices and the current 11-channel local context design. The label is still
image-level, so it is not yet a final training signal; it is a broadcastable
smoke target for checking loss plumbing, class imbalance, and zero-biased
initialization. The analysis implication is clear: if frozen-head training fails
on these tensors, the issue is likely label granularity, objective weighting, or
family vocabulary, not inability to reproduce decoder-side context.

For conference-grade evidence, E240 should be treated as infrastructure rather
than performance. Its value is that it makes the next ablation executable:
original EF-LIC, E236 oracle/fixed policies, E239 label curriculum, and learned
local head can now be connected through the same decoder-safe context path.
Full claims still require full Kodak/CLIC/Tecnick evaluation and preferably
spatial/slice labels after the broadcast-label smoke is stable.

## EF-LIC Frozen Local Head Analysis After E241

E241 is a useful negative result because it separates implementation plumbing
from supervision quality. The E240 full Kodak24 export proves that real EF-LIC
local context maps can be collected at the intended decoder-safe insertion
point, but the first supervised training audit shows that E239 image-level
labels are too coarse for a local map controller.

The broadcast-label objective is the clearest failure. It asks every slice and
spatial position in an image to take the same family label, so the easiest
solution is to predict a frequent nonzero family everywhere. Increasing the
false-positive penalty and zero-class weight did not fix this; validation
false-positive nonzero rate remained `1.0`. This directly recreates the danger
identified by E238: nonzero HCG geometry on true-zero images is expensive, and a
controller that cannot reliably fall back is not paper-main material.

The image-level pooled objective is a better diagnostic but still not enough.
It can fit the train split with low family cost, yet four-fold validation over
Kodak24 has only `0.125` mean accuracy and false-positive nonzero rate `1.0`.
This means the local context tensors contain useful signal for the training
images, but the current labels do not teach a transferable zero-vs-active
boundary. The limitation is supervision granularity and calibration, not the
availability of decoder-safe EF-LIC state.

The next HCG-RVQ/EF-LIC experiment should therefore be E242: build local
slice/spatial teachers from the E236 policy outputs or local residual-error
statistics, then train a two-stage head. Stage one should predict conservative
activation versus zero fallback with a high false-positive penalty. Stage two
should predict family/strength only inside activated regions. That keeps the
main thesis intact: hyperprior/context generates quantizer geometry, but only
where local evidence supports it.

## EF-LIC Spatial Teacher Analysis After E242

E242 confirms that the E241 failure was not just a bad optimizer setting. The
new export gives a real local supervision object: `target_map` marks active HCG
regions according to the exact E236 policy alpha map and keeps inactive
positions as zero/fallback. On Kodak24, the mean active target fraction is
`0.614583`, so the target is neither all-zero nor all-active. This is the right
kind of label for the HCG-RVQ thesis: geometry is local and confidence-gated.

The frozen-head training result exposes the next bottleneck. A single
multiclass head cannot simultaneously learn zero calibration and family
selection under the current tiny Kodak-only split. With conservative weights it
predicts almost all zero and misses active regions. With stronger active weights
it predicts nearly all active and false-positives zero regions. This sharp flip
is exactly what E238 warned about: activation is a separate reliability problem,
not just another family class.

Therefore the next paper-main candidate should not be a monolithic seven-way
family classifier. It should be a two-stage controller: a binary activation head
with an explicit calibrated threshold and false-positive-heavy loss, followed by
a conditional family/strength head trained only where the teacher alpha is
active. The first reportable metrics for that stage are activation AUROC/PR,
false-positive rate at fixed recall, missed-active rate, and then codec-loop RD
once activation is stable.

## EF-LIC Activation Calibration Analysis After E243

E243 gives the first clean answer to the question raised by E242: is activation
separable enough to train before family selection? The answer is currently
"weakly, but not reliably." A binary head is a better abstraction than the
seven-way family classifier because it exposes the actual tradeoff: thresholds
that keep false positives low miss most active regions, while F1-oriented
thresholds recover active regions only by activating too much zero/fallback
area.

The rem0 global-summary result is useful because it shows the direction is not
hopeless. Broadcast global statistics from decoder-safe context maps improve
held-out recall at similar FPR. But the four-fold result prevents overclaiming:
val AUROC averages only `0.4595`, with strong fold dependence. This means the
current Kodak24 supervision and small local/global-summary head are not enough
to produce a deployable controller.

The design implication is important for the full-training path. The next method
should not add many auxiliary losses directly to EF-LIC's final R-D objective.
Instead, activation should first be made reliable as a pretraining/calibration
module, then inserted into codec-loop training with the original R-D/perceptual
loss dominant. If the activation head remains weak, full training may simply
learn around it or collapse to all-active/all-zero behavior, giving ambiguous
results and weak paper attribution.

This also connects to the GLC track. GLC diagnostics already showed that sparse
active states can improve some metrics but are DISTS/bit-rate fragile without a
reliability selector. EF-LIC E243 says the same thing in a cleaner RVQ setting:
HCG-RVQ's next improvement is not more always-on geometry, but stronger
decoder-safe reliability modeling plus bit-aware/perceptual-aware training.

## EF-LIC Activation Signal Analysis After E244/E245

E244 and E245 are important because they prevent a tempting but weak conclusion.
E244 made the activation head larger and more globally aware by stacking all
four EF-LIC slice contexts as `[44, H, W]`, but the four-fold result overfits:
train AUROC/AUPRC reaches `0.958515` / `0.974339`, while validation AUROC/AUPRC
falls to `0.269167` / `0.501585`. This is not a head-capacity success; it is a
small-teacher-split warning.

E245 then removes the neural head and asks whether any single decoder-safe
context channel has a robust threshold. The answer is also no for paper-main
activation. F1-oriented thresholds recover active regions only by activating
nearly all zero regions (`best_f1` validation FPR `0.978462`). Conservative
thresholds reduce false positives but miss almost all useful active regions
(min-weighted-risk validation recall `0.027936`). Therefore the current
mean/scale/support context maps are not sufficient to train a reliable HCG
activation controller from Kodak24 teacher labels alone.

This does not invalidate HCG-RVQ. It narrows the next implementation target:
activation reliability needs richer decoder-known state and/or independent fit
labels before full codec-loop training. The original `prompt.txt` thesis remains
unchanged: hyperprior/context should generate quantizer geometry. The updated
claim is more precise: geometry generation must be gated by reliable,
bit-aware, decoder-reproducible evidence; otherwise all-on HCG can hurt even if
small oracle/local policies show headroom.

## CLIC Extension After E244/E245

The CLIC professional extension is important because it separates a Kodak-only
failure from a general controller-design issue. E242 CLIC export produced 41
finite spatial-teacher tensors with a balanced active fraction (`0.487835`) and
richer family mix than Kodak, so it is a reasonable independent fit source.
However, both E245 scalar-feature CV and E244 cross-slice heads show the same
activation tradeoff: high recall requires nearly all-active predictions, while
false-positive control misses most active regions.

This strengthens the research judgment. The next EF-LIC paper-main step should
not be “full-train the current all-on or current E244 head and hope it fixes
itself.” Full training may improve calibration if the R-D/perceptual objective
is dominant, but if the controller input lacks a stable bit-aware signal, the
model can simply learn around the branch or collapse to dataset-specific
activation. Therefore full training is still needed eventually, but only after
adding richer decoder-reproducible signals or a cleaner fit/calibration split.

## EF-LIC Decoder-Safe Feature-Group Analysis After E246

E246 closes the most obvious follow-up to E244/E245: maybe the controller simply
needed z-prior/index summaries and richer decoder-safe local statistics. The
answer is no for the current frozen-teacher protocol. Rich E233 feature groups
can memorize the 65-image Kodak+CLIC teacher table, but they do not generalize
under leave-one-image-out or Kodak/CLIC transfer. The best-looking pooled LOIO
activation row is still a high-FPR compromise, and family accuracy remains near
chance for a six-family target.

This is an important refinement, not a dead end. E237 showed that true-family
or oracle local policies have real headroom; E246 says the missing piece is not
just appending more global summaries to a small supervised head. The bottleneck
is robust, decoder-reproducible reliability control: deciding where HCG geometry
should be active and which local geometry family should be trusted without
using validation-time teacher information.

The practical consequence is that the next EF-LIC full-training run should not
be an auxiliary-loss-heavy classifier experiment. The safer paper-grade design
is an R-D/perceptual-dominant codec objective with a conservative HCG branch:
start from zero/scalar fallback, initialize or weakly regularize activation from
teacher labels, and let the original EF-LIC objective decide whether local
geometry is useful. Ablations must include original EF-LIC, all-on HCG, fixed
guarded HCG, oracle/teacher upper bounds, and the learned controller. If the
learned controller wins only because auxiliary labels dominate, the claim is
weak; if it wins under the original codec objective and improves codebook/error
statistics, it supports the HCG-RVQ thesis.

For GLC, E246 reinforces the same message as the sparse active-state probes:
local residual VQ/RVQ can have headroom, but index-rate and perceptual
reliability must be part of the design. A dense or all-on branch that improves
residual MSE while harming bitrate or perceptual quality is not enough for an
international-conference claim.

## Loss Objective Analysis After E247

E247 answers the loss-design risk directly. The current `RateDistortionLoss`
keeps a clean core objective, `bpp + lambda * mse + beta_commit * commit_loss`,
but the repository now contains many exploratory configs with teacher,
selector, anchor, and geometry regularization terms. That is useful for
diagnosing failure modes, yet dangerous for paper claims if the auxiliary terms
become the source of the improvement.

The audit found `76` RD/commit-only configs out of `151`, while `46` configs use
teacher/selector or anchor losses and `72` use geometry/gate regularizers. This
means the experimental record can be separated cleanly: teacher-heavy runs are
mechanism probes or warmup candidates; paper-main codec rows should use a
dominant original EF-LIC/GLC objective, with auxiliary terms either absent or
shown by ablation to be weak.

This matters for the next full-training step. If HCG-RVQ only wins when a
teacher or anchor loss is large, the claim becomes "the auxiliary target helped"
rather than "hyperprior-conditioned quantizer geometry improves compression."
The stronger claim requires the learned HCG branch to improve RD/perceptual
metrics, codebook usage, residual error, and bitrate accounting under the same
codec objective as the baseline.

## Full-Training Candidate Analysis After E248

E248 is the bridge from controlled evidence to full-training selection. It does
not make the final performance claim. Its value is that it prevents two common
mistakes: promoting a teacher-heavy selector because oracle rows look strong,
and promoting a GLC residual-MSE branch because the active residual error falls
while bpp or DISTS regress.

For EF-LIC, the evidence is promising but conditional. E234/E236 show real local
HCG headroom: fixed policies produce small but nonzero improvements, while
per-image oracle/local policies are much larger. E235/E237 further show that if
the correct family is known, much of the oracle can be recovered. The failure is
not the quantizer-geometry idea itself; the failure is current frozen
feature-based reliability control. E246 shows that decoder-safe feature groups
can memorize the small Kodak/CLIC teacher table but do not transfer well enough
for a paper-main controller. Therefore the next EF-LIC claim should be: local
hyperprior/context-conditioned geometry has headroom, and it must be learned in
the codec loop with conservative fallback under the original objective.

For GLC, E170/E171 confirm that active residual VQ/RVQ has strong
representational headroom, especially q0 part-group codebooks. E181 is the
important warning: decoder-aware training improves PSNR, MS-SSIM, and LPIPS on
Kodak8 after OI16 training, but the empirical index-rate increase and DISTS
regression mean that residual MSE alone is not the submission story. The next
GLC design must explicitly account for index bpp and perceptual quality, then
scale from q0 to later q/stage branches only after this conflict is controlled.

The loss-policy implication from E247 remains central. Full-training candidates
should be simple: original codec R-D/perceptual objective, VQ commitment/index
terms if needed, and only weak/warmup auxiliary terms. If the branch only wins
with a large teacher, selector, anchor, or geometry penalty, the improvement is
not clean evidence for HCG-RVQ. This keeps the model design closer to the kind
of simple, interpretable mechanism that is easier to defend in an international
conference paper.

## GLC Bit-Aware Score Analysis After E249

E249 quantifies why the GLC track is promising but not yet paper-main. The
trained E181 branch clearly improves several perceptual/quality indicators:
PSNR, MS-SSIM, LPIPS, and the combined `DISTS + 3*LPIPS` score before bpp is
charged. That means the q0 active residual branch is not just a numerical MSE
artifact; it changes decoded quality in a useful direction.

The same table also explains why full-training the current branch unchanged is
risky. Empirical bpp rises by about `+0.014548`, and DISTS still worsens by
about `+0.012346`. The break-even bpp weight is `1.166984`, so the apparent
perceptual gain survives only under a mild rate penalty. If the final low-rate
objective or benchmark places a stronger cost on extra bits, the branch becomes
harmful.

This is the right kind of negative/positive split for research progress. It
supports the HCG-RVQ thesis that local residual geometry has usable headroom,
but it rejects the naive all-on or residual-MSE-only implementation. The next
GLC implementation should make rate part of the branch design: explicit index
entropy accounting, activation gating, smaller/fewer codebooks where the gain is
weak, or a differentiable proxy that discourages high-entropy assignments.

## GLC Soft-Index Bit-Aware Analysis After E250

E250 tested the most direct response to E249: add a differentiable soft
codebook-usage entropy excess penalty to the GLC q0 active tail VQ branch. The
result is now a successful mid-scale candidate gate, but still not a final paper-main row.

The best E250 variant is currently the OpenImages8 -> Kodak8 K=8 part-group row
with `lpips_weight=1.0` and `soft_index_weight=1.0`: `dPSNR=+0.588917`,
`dMS-SSIM=+0.024542`, `dLPIPS=-0.010767`, `dbpp=+0.013677`,
`score0=-0.020319`, and `score1=-0.006642`. More importantly, the matched
OpenImages16 -> Kodak8 rerun also beats the E181 reference after bpp is charged:
`dPSNR=+0.583724`, `dMS-SSIM=+0.024469`, `dLPIPS=-0.010576`, `dbpp=+0.014432`,
`score0=-0.018201`, and `score1=-0.003769`. DISTS is still the limiting metric:
matched E250 has `dDISTS=+0.013527`, slightly worse than E181's `+0.012346`.

The ablations explain the failure mode. K=4 and shared-codebook variants can
reduce empirical index rate, but they do it by removing too much local residual
capacity. K=4 leaves the active residual MSE ratio near `0.898600`; shared K=8
is worse (`1.770163`). Both damage PSNR/LPIPS/DISTS, so the problem is not that
GLC merely needs a smaller dictionary. It needs a way to spend the K=8 local
capacity selectively where the perceptual/RD gain pays for the index bits.

The design implication is sharper than E249: soft entropy pressure plus an
explicit perceptual term is no longer just a smoke test. After the memory fix,
the matched OI16 -> Kodak8 gate also improves the bpp-charged score over E181.
This makes E250 the first GLC-side HCG-RVQ branch worth scaling. The remaining
research risk is DISTS: the overall score wins because LPIPS and bpp improve,
but DISTS itself is still slightly worse than E181. The next full-Kodak/CLIC
run should therefore keep the same branch and add DISTS-sensitive activation or
index-prior analysis rather than changing the whole design again.

## GLC Activation Gate Analysis After E251

E251 resolves the main ambiguity left by E250. The E250 K=8 part-group branch is
not weak because local residual geometry is useless; it is weak on full Kodak24
because all-on activation spends index bits and worsens DISTS on images where the
LPIPS/PSNR gain is not large enough. The all-on score is slightly harmful
(`+0.002014`) even though PSNR, MS-SSIM, and LPIPS all improve.

The per-image oracle is the key evidence: selecting only `11/24` Kodak images
turns the same branch into a clear win (`-0.005773`) under
`DISTS + 3*LPIPS + bpp`, with one bit/image of side overhead included. A
single-feature leave-one-out threshold still stays negative (`-0.001565`), so
this is not only a perfectly overfit oracle table. It is a signal that the next
model should predict activation or index priors from codec-side statistics.

This also sharpens the HCG-RVQ claim. The proposed geometry branch should not be
presented as "more VQ everywhere is better." The safer and stronger claim is:
hyperprior-conditioned quantizer geometry creates a useful extra local coding
mode, and the compression model must learn when that mode is worth its rate.
That is closer to the prompt central thesis than dense/all-on residual VQ.

The immediate risk is that the best explanatory in-sample rule uses base PSNR,
which is not a practical decoder-side signal. A codec-safe-only threshold audit
finds a weak in-sample rule from empirical bpp delta (`<=0.015252`, score
`-0.001625`), but the same safe-feature family does not survive leave-one-out
(`+0.006953`). Therefore the next implementation must translate the observed
selection headroom into learned or predicted features available before or during
coding: active residual energy, predicted index entropy, local scale, codebook
usage/uncertainty, or a small hyperprior-generated gate. Base-quality features
should remain analysis-only.

## GLC CLIC Professional External Gate After E252/E253

E252/E253 add the first external-domain pressure test for the E250 GLC branch.
The result is negative but valuable. On CLIC Professional first-8, the same q0
K=8 branch still improves distortion-style and perceptual-adjacent metrics such
as PSNR, MS-SSIM, and LPIPS, but every image is harmful under the paper-facing
`DISTS + 3*LPIPS + bpp` score. The oracle selects `0/8` images for both the
original LPIPS+DISTS recipe and the DISTS-heavy variant.

This means the current GLC q0 tail branch is not merely under-gated on CLIC Pro.
For this domain, the local residual codebook state is producing a consistent
DISTS penalty that is larger than the LPIPS gain and far larger after index bpp
is charged. Increasing DISTS loss weight from 1 to 2 does not fix it in the
short gate; it slightly reduces some per-image DISTS harm but worsens the final
combined score.

The research interpretation is disciplined: Kodak evidence still supports the
existence of useful HCG/RVQ local coding modes, but CLIC Professional exposes a
domain/metric mismatch. The next GLC improvement should not be a larger all-on
run. It should learn a domain-mixed reliability or index-prior gate, ideally on
OpenImages plus CLIC-style calibration images, and it should use DISTS as an
explicit validation guard. This also argues for keeping EF-LIC and GLC parallel:
EF-LIC may supply a stronger codec-valid controller path while GLC supplies a
hard low-bitrate generative compression stress test.

## GLC Domain-Mixed Gate Readiness After E254

E254 answers the question that E251/E252 left open: can the same simple gate
rule carry the Kodak-positive local RVQ branch into CLIC Professional? The
answer is no for the current E250 branch-internal features.

The useful signal is still real. In the pooled primary set, all-on is harmful
(`+0.007638`), but an oracle that can silence the branch selects `11/32` images
and improves the score to `-0.004329`. Those 11 positives are all Kodak images,
which means the local residual geometry mode has measurable value but is not yet
domain-stable.

The failure mode is equally clear. The best in-sample branch-internal rule
(`active_rvq_mse >= 0.00188907`) reaches only `-0.001061`, and leave-domain-out
turns positive (`+0.001671`). Training on Kodak selects two CLIC images and
hurts (`+0.006684`); training on CLIC learns a silent rule and misses all Kodak
wins. That is exactly the pattern expected when the branch has a local coding
mode but no reliable hyperprior/index-prior controller.

The paper-facing claim should therefore not be that dense RVQ improves low-rate
GLC. The stronger and safer claim is that hyperprior-conditioned quantizer
geometry exposes an alternate local coding mode, and a learned reliability/index
controller is required to decide when the extra index rate and DISTS risk are
worth paying. This also supports keeping the loss simple: original codec
R-D/perceptual terms plus minimal VQ/index regularization, rather than masking a
weak controller with large auxiliary objectives.

## GLC Linear Controller Proxy After E255

E255 tests the natural next question after E254: whether the local-RVQ branch can
be controlled by a tiny learned model rather than a hand threshold. The answer is
encouraging only within the current Kodak-heavy evidence, not across domains.

Resubstitution shows capacity: branch-internal logistic selects `8/32` and scores
`-0.002896`, and the analysis-only upper bound reaches `-0.003081`. The best
proper leave-one-image-out row is branch-plus-rate score regression at
`-0.000865`. That is a real but small signal, and it is much weaker than the
oracle `-0.004329`.

The cross-domain result blocks promotion. Leave-domain-out branch/rate
controllers choose the safe silent policy, so they avoid CLIC harm but miss every
Kodak positive. Analysis-only domain transfer selects a few images but is
harmful. This means the issue is not just model capacity; the current calibration
rows do not contain a domain-stable reliability signal.

Paper implication: keep the controller simple, but train it on the right
evidence. The next GLC controller should be calibrated on CLIC-like or
domain-mixed branch labels and optimized against DISTS+bpp risk, while preserving
no-branch fallback.

## GLC CLIC Calibration and Controller Evidence After E256-E258

E256-E258 add the first non-overlapping CLIC Professional calibration evidence to
the E250/E254/E255 GLC branch. The result sharpens, rather than overturns, the
previous conclusion.

The CLIC calibration slice has one useful image but is mostly harmful. All-on is
strongly positive in the bad direction (`+0.040670`) because DISTS worsens by
`+0.030436` and empirical bpp rises by `+0.016525`. The oracle selecting `1/8`
images at `-0.000830` proves CLIC is not completely hopeless, but the headroom is
small and much harder to identify than on Kodak.

When pooled with Kodak24 and CLIC first-8, the 40-row oracle is `12/40` at
`-0.003630`; the extra CLIC-positive row helps only slightly. A branch-internal
threshold can still find an in-sample subset (`4/40`, `-0.000849`), but
leave-domain-out remains harmful (`+0.001337`). Tiny learned controllers tell the
same story: resubstitution has capacity (`-0.002728` for branch-internal score
regression), but deployable held-out rows are near zero or positive, and
domain-transfer becomes silent or harmful.

Paper implication: do not claim that dense local RVQ improves GLC. The current
claim should be that HCG-RVQ exposes a conditional local coding mode, but the
paper-main method must include an index/reliability controller trained on
sufficient domain-mixed evidence. This also reinforces the loss-design rule:
keep the codec objective simple and make DISTS+bpp the gate criterion, rather
than hiding a weak selector behind large auxiliary losses.

## Full-Training Readiness After E259

E259 is the current cross-track decision point. It combines the EF-LIC
decoder-safe controller audit, the loss-objective audit, and the updated GLC
CLIC calibration/controller results. The main result is that both EF-LIC and GLC
support the same architectural direction, but neither supports a dense all-on
promotion yet.

EF-LIC is promising because pooled features can perfectly separate useful
activation in resubstitution (`0.000000` min-risk), but it is not yet reliable:
LOIO aggregate min-risk remains `0.953846`, and cross-domain min-risk is
`0.916667` for CLIC->Kodak and `1.219512` for Kodak->CLIC. This is exactly the
pattern expected from a real local coding signal without a sufficiently trained
reliability controller.

GLC now has a sharper low-bitrate story. All-on local RVQ is bad
(`+0.014245`), but oracle selection is good (`-0.003630`). The best deployable
LOOCV proxy is only `-0.000171`, so the headroom exists but the current
controller is too weak to claim domain-stable improvement.

The paper-facing design should therefore stay simple and principled:
hyperprior/context generates local quantizer geometry and an index/reliability
decision; the original codec loss remains dominant; VQ commitment and index
entropy terms are added only when they directly support rate-aware quantization.
Large teacher/selector losses should not be part of the main claim unless an
ablation proves they are necessary and do not dilute RD optimization.

## Reliability/Index Controller Probe After E260

E260 separates two questions that were easy to blur: whether we have the right
controller contract, and whether the current offline evidence is enough to fit a
paper-main selector. The first answer is yes; the second answer is still no.

The new controller contract is correct for the HCG-RVQ story. It predicts both
activation and signed branch risk, initializes toward fallback, and can be used
as either a spatial EF-LIC head or a per-image GLC controller. The smoke test is
healthy: zero/fallback initialization selects no positions, and gradients are
finite.

The GLC row probe confirms the generalization blocker. Resubstitution reaches
the exact oracle (`12/40`, `-0.003630`), which means the feature/label set
contains enough information to memorize useful images. But LOOCV becomes
harmful (`+0.003572`) and leave-domain is also harmful (`+0.003267`). This
matches E246/E258: capacity is not the bottleneck; domain-stable reliability and
rate control are.

Paper implication: the final method should not be "train a bigger selector."
The main method should train a compact reliability/index controller with the
codec objective and domain-mixed calibration. This strengthens the loss-design
argument: keep the original RD/perceptual objective dominant and only add VQ,
index, and lightweight reliability terms that directly support fallback
selection.

## Domain-Robust Controller Calibration After E261

E261 confirms that the GLC local-RVQ headroom is not yet accessible through a
simple offline threshold over current diagnostics. The oracle remains useful
(`12/40`, `-0.003630`), and the best resubstitution threshold selects a small
subset (`4/40`, `-0.000849`). This proves the branch diagnostics carry some
signal, but held-out protocols erase it: LOOCV is slightly harmful (`+0.000015`),
leave-domain is harmful (`+0.001337`), and leave-variant is also harmful
(`+0.001337`). The selected held-out rows have `0.000000` win rate for
leave-domain/leave-variant.

The interpretation is important for the paper story. The current blocker is not
that HCG-RVQ lacks a useful local coding mode; all-on versus oracle still shows
that the mode exists. The blocker is that hand thresholds and tiny offline
controllers cannot robustly decide when the extra index/perceptual cost is worth
paying across Kodak and CLIC-like domains. This makes E261 a negative control
supporting the proposed reliability/index controller: the controller must be
trained in or adjacent to the codec objective, not fitted post hoc on a small
row table.

The next HCG-RVQ implementation should therefore keep the method simple but make
the decision learnable: compact local branch, explicit no-branch fallback,
rate-aware activation/risk head, original codec R-D/perceptual objective as the
main loss, and only minimal VQ/index terms. Full training is justified only for
that codec-loop controller path, not for dense/all-on or threshold-only variants.

## Controller Fallback-Mix Contract After E262

E262 implements the practical bridge from the E261 negative threshold result to
the next full-training candidate. The key contract is simple: the original codec
path is the default, the HCG-RVQ branch is blended in only through a compact
reliability/index gate, and hard evaluation can exactly recover the base path.

The smoke passed with conservative initialization. Spatial soft gate mean is
`0.057882`, MLP soft gate mean is `0.077224`, the hard gate sum is `0.000000`,
and hard fallback reconstructs the base tensor exactly. This is important
because it gives EF-LIC and GLC the same safe insertion mechanism: if the branch
or controller is bad, the model can stay near the original codec rather than
paying all-on index/perceptual cost.

Analysis implication: E262 does not prove performance, but it removes an
implementation blocker. The next evidence should come from mid-scale codec-loop
pilots that compare base, hard fallback, soft learned gate, all-on branch,
threshold negative control, and oracle/teacher upper bound under the same
checkpoint and domain split policy.

## GLC Fallback-Gate Codec-Loop Pilot After E263

E263 gives the first positive codec-loop evidence for the post-E259 design
change. The key shift is not "add RVQ everywhere"; it is "let a compact
reliability/index controller decide how much local RVQ geometry to spend." This
matches the central prompt thesis that hyperprior/context should generate local
quantizer geometry and reliability, not merely add a dense quantizer branch.

The pattern is stable across the small pilots. Dense all-on activation improves
some fidelity metrics, but pays too much index/perceptual cost and remains
harmful under the guarded score. In the 4-image trained run, all-on has score
`+0.012332` and dbpp `+0.014682`. The soft fallback gate has score `-0.019209`
with dbpp only `+0.001576`, while improving PSNR (`+0.388592`), MS-SSIM
(`+0.011077`), LPIPS (`-0.006471`), and DISTS (`-0.001372`) on average. Its
gate mean is only `0.107217`, which is exactly the desired behavior: harvest a
small useful residual correction without committing the codec to dense branch
usage.

This also explains why the earlier EF-LIC/GLC all-on intuition failed. The
branch can contain useful reconstruction information, but the branch is not
uniformly worth its extra index/rate cost. Full training might improve the
branch, but E259-E263 indicate that the final method should spend full-training
budget on the fallback-gated codec-loop design, not on dense/all-on activation.

The CLIC Professional first-8 run strengthens this beyond a Kodak-only finding.
On the same domain where earlier E250 all-on pilots paid roughly `+0.016` bpp
and worsened DISTS, E263 keeps the all-on negative control harmful
(`+0.034813`, win rate `0/8`) but makes the soft fallback gate useful
(`-0.010313`, win rate `8/8`). The gate stays small (`0.084116`) and the
effective dbpp is only `+0.001313`, which is the exact mechanism needed for a
low-bitrate paper claim: local RVQ residual information exists, but it must be
spent sparsely and rate-awarely.

Limitations remain clear. The current soft-gate bpp is an empirical diagnostic,
not final entropy-coded bit accounting. The hard gate still selects no images at
the default threshold, so an evaluation-time hard policy needs calibration or a
training schedule that makes selected cases robust. The next scale-up should use
more Kodak images and a held-out CLIC Professional slice, then port the same
controller contract to EF-LIC with spatial features and matched RD/perceptual
reporting.

## Held-Out GLC Slice and Rate-Accounting Audit After E263/E264

The held-out E263 runs reduce the chance that the result is an easy-slice
artifact. Kodak held-out and CLIC Professional held-out both preserve the same
qualitative behavior as the first slices: dense all-on local RVQ is harmful,
while the compact soft fallback gate improves every evaluated image under the
diagnostic score. This is exactly the low-bitrate mechanism HCG-RVQ needs:
local residual geometry contains useful information, but it should be spent
sparsely.

E264 adds the necessary caution. If every nonzero soft gate had to transmit the
full branch index stream, the aggregate soft-gate gain would not be paper-ready:
all soft rows move from `-0.011948` diagnostic score to `+0.002064` under
conservative full-branch-bpp charging. Kodak remains close to usable under that
upper bound, but CLIC Professional does not. Therefore the correct conclusion is
not "soft gating solves rate." The correct conclusion is "soft gating reveals a
stable sparse residual correction, and the next method component must make its
index/rate cost real."

This updates the full-training decision. A full run of dense/all-on HCG-RVQ is
still a poor use of budget. A full run of fallback-gated HCG-RVQ is justified
only after the branch has one of two paper-defensible accounting paths:
hard/annealed sparse activation with final selected-index bpp, or progressive
entropy-coded branch bits whose transmitted cost scales with the activated
portion. This keeps the final claim simple and honest: hyperprior/context
generates useful local quantizer geometry and a reliability/index decision, not
a free fractional bitstream.

## EF-LIC Controller Wiring After E265

E265 confirms that the EF-LIC path is no longer blocked at the artifact or
controller interface level. E242 context tensors have the right decoder-safe
shape (`context_maps`, `alpha_target`, `target_map`), and the shared
reliability/index controller can consume them with finite gradients. Hard
fallback exactly recovers no-branch output, which is the essential safety
property for low-bitrate integration.

The important boundary is also clear: E265 is not a codec result. It does not
measure bpp, PSNR, MS-SSIM, LPIPS, DISTS, or actual EF-LIC reconstruction after
inserting HCG-RVQ into the model. It only verifies that the next EF-LIC branch
can share the same control design as GLC: conservative initialization,
soft differentiable training, hard no-branch fallback, and an eventual
selected-index/rate report.

This keeps the two-backbone strategy coherent. GLC currently provides the
strongest performance-side signal, but EF-LIC now has a clean implementation
path. The next EF-LIC experiment should be a true codec-loop pilot: base EF-LIC,
all-on HCG/RVQ branch, soft fallback gate, hard fallback, and conservative
bit-accounting side by side on Kodak and CLIC Professional.

## Low-Rate Branch Lesson After E266/E267

E266 and E267 clarify the real bottleneck in the GLC path. The original E263
branch proved that local residual geometry is useful under soft fallback, but
E266 shows that the original branch is too expensive if every selected row pays
full branch bpp. The aggregate diagnostic score `-0.011948` turns into
`+0.002064` under full-branch-bpp charging, and CLIC Professional is the weak
case (`+0.005540` on trained CLIC rows). This means a simple hard threshold on
the original branch is not enough for a conference-grade claim.

E267 changes the conclusion in a useful way. Reducing the branch to `K=4` and
active parts `[0, 1]` keeps the soft fallback benefit while making the branch
cheap enough that conservative full-branch-bpp accounting stays negative. On
CLIC Professional held-out rows, the diagnostic soft score is `-0.009876` and
the selected-soft/full-bpp score remains `-0.007632`; on Kodak held-out rows,
those numbers are `-0.017813` and `-0.016357`. This is stronger than the E263
result because it directly addresses the E264/E266 rate objection.

The mechanism is also now cleaner. All-on low-rate output is still harmful, so
the method is not "more RVQ everywhere." The useful design is: a compact local
quantizer branch, conservative fallback, and a learned soft reliability/index
controller that spends a small amount of residual capacity only where the codec
benefits. That is well aligned with the prompt's thesis that hyperprior/context
should generate local quantizer geometry and its reliability, not just attach a
dense VQ module.

Remaining limitation: E267 still evaluates a soft tensor blend, and the
selected-soft/full-bpp audit is a conservative accounting proxy, not a complete
entropy-coded bitstream. The next paper-directed experiment should implement
actual selected-index or progressive branch accounting, then repeat the
Kodak/CLIC split with checkpoint-level and feature-distribution reports.

## First/Held-Out Confirmation After E268

E268 strengthens E267 from a held-out-only observation into a small but coherent
four-split result. The low-rate branch is still harmful when forced all-on, but
the fallback-gated reconstruction stays useful across Kodak first, Kodak held,
CLIC first, and CLIC held. This is important because earlier E250-E266 failures
were mostly caused by CLIC fragility and rate overpayment.

The combined trained-row audit is now hard to dismiss as an easy-slice artifact:
`-0.011600` selected-soft/full-bpp score across 24 rows, `0.958` selected win
rate, and negative means on both domains (`-0.017096` Kodak and `-0.008852`
CLIC). Cross-split threshold transfer is also negative in all three checks.
That does not yet prove the final codec, but it says the branch capacity/rate
scale is now in the right regime.

The main scientific interpretation is that HCG-RVQ should be framed as
rate-aware local residual geometry. A small RVQ branch can improve perceptual
and distortion metrics when softly blended through reliability control; a dense
branch worsens the codec because it spends residual indices everywhere. The
next claim-building step is therefore not another all-on run. It is actual
selected-index or progressive-rate accounting, then a larger checkpoint-level
run with codebook and residual-stage analysis.

## Progressive-Rate Margin Interpretation After E269

E269 makes the E267/E268 result substantially more interpretable. The useful
property is not simply that a soft local branch improves reconstructions. The
useful property is that the low-rate branch improves reconstructions while
leaving enough rate margin to survive conservative branch-bpp accounting.

The original E263 branch and the low-rate E267/E268 branch have comparable
no-bpp reconstruction gains, but their rate costs differ by almost an order of
magnitude. The original trained branch pays `+0.015265` bpp and becomes harmful
on average (`+0.001893`), especially on CLIC (`+0.005540`). The low-rate trained
branch pays only `+0.002202` bpp and stays negative (`-0.011600`), including
CLIC (`-0.008852`). This supports the current model story: HCG-RVQ should be a
small, reliability-controlled local residual geometry module, not dense RVQ
added everywhere.

The margin numbers also define the implementation target. For trained low-rate
rows, the mean max affordable rate fraction is `7.092` and p10 is `1.603`, so
most cases have enough headroom even under full observed branch bpp. The
remaining CLIC tail needs fallback or selected/progressive savings: the hardest
trained row, `casey-fyfe-999.png`, can afford only `0.877` of the observed full
branch bpp. This is exactly the kind of failure that a real activation policy
should solve.

This result should not be overstated as final RD evidence. It still evaluates a
soft tensor blend and uses accounting over existing rows. However, it is strong
design evidence for moving to an actual selected-index/progressive bitstream.
The next paper-facing claim should require: measured selected-index bpp, hard or
annealed fallback exactness, codebook utilization/perplexity, residual-stage
contribution, and larger Kodak/CLIC Professional evaluation before full-training
claims.

## Selected-Rate Policy Lesson After E270

E270 answers the immediate question left by E269: do we need a complex learned
selector to remove the low-rate CLIC tail, or is the remaining failure mostly a
rate-cap problem? In the current small splits, the answer is the latter. The
best deployable transfer policies all use only `full_branch_dbpp <= threshold`.

The most important protocol is Kodak-to-CLIC, because CLIC has been the fragile
domain throughout E250-E269. Training the threshold on Kodak chooses
`full_branch_dbpp <= 0.00242854`; applying it to trained CLIC selects `11/16`
rows, scores `-0.007675`, and the selected rows have win rate `1.000`. This is
slightly less negative than selecting all low-rate rows under E269
(`-0.008852`), but it removes the positive tail and creates a cleaner hard
fallback story.

This supports a simple model-design direction. The next HCG-RVQ codec should
not add many auxiliary losses to chase the tail. It should first implement a
small local branch with a measurable branch-rate cap, exact fallback, and only
minimal loss terms needed for VQ/index/rate consistency. If this simple policy
survives larger evaluation, it becomes much easier to explain why HCG-RVQ works:
hyperprior/context proposes local residual geometry, and the codec spends it
only when the local branch cost is low enough.

The limitation is that `full_branch_dbpp` is currently computed by the audit
from existing branch rows. For paper evidence, the same quantity must become an
actual encoder/bitstream quantity: selected branch indices, selected entropy or
fixed length, and a signaled fallback decision.

## Rate-Cap Codec-Loop Lesson After E271

E271 resolves an important ambiguity in E270. A simple branch-rate cap is useful,
but it is not sufficient by itself. When the capped branch is used with the
soft/progressive residual output, the score stays negative on both tested
domains: `-0.007255` on CLIC Professional held-out 8 and `-0.016177` on Kodak
held-out 4. When the same cap is used with all-on branch output, the result is
strongly harmful: `+0.035364` on CLIC and `+0.053973` on Kodak.

This means the current HCG-RVQ signal is not "activate RVQ whenever it is cheap
enough." The stronger interpretation is "use a small local residual geometry
branch, but spend it progressively and softly enough that the base codec is not
overwritten." That is consistent with the earlier all-on failures and explains
why a dense branch felt intuitively tempting but empirically failed: it changes
too much of the reconstruction manifold for too little controlled gain.

For the paper path, E271 is still a short-cycle result. It uses a soft tensor
blend and charges selected full branch bpp, not a final entropy-coded
progressive bitstream. But it gives a sharp implementation requirement for the
next stage: selected/progressive coding must make the soft branch's rate real,
while keeping all-on as the negative-control ablation. The EF-LIC port should
inherit this lesson immediately; starting EF-LIC with a dense all-on branch is
now low priority unless it is only used as a failure/ablation baseline.

## Gate-Signal Overhead Lesson After E272

E272 adds an implementation-level constraint to the E271 result. If the soft or
progressive branch needs only a scalar or very coarse gate signal, the side cost
is not the blocker. Scalar 8-bit signaling changes the pooled score from
`-0.010229` to only `-0.010220`, and `tile32_1bit` remains `-0.009408`.

However, dense gate maps are not free. On CLIC, `tile16_2bit` leaves only
`-0.001357` average score and drops the win fraction to `0.500`. That makes the
paper-facing design preference clearer: HCG-RVQ should make reliability mostly
decoder-predictable from hyperprior/context, or signal only a coarse selection
state. Sending a dense learned gate map would spend away the exact low-rate
advantage that E267-E271 recovered.

This fits the original prompt: hyperprior/context should generate quantizer
geometry and reliability, not require a large extra side-channel to decide where
quantization is useful. The next implementation should therefore measure
selected-index bpp plus at most scalar/coarse gate overhead, and keep the dense
map option as an ablation rather than the main method.

## Progressive-Enhancement Reality Check After E273/E274

E273/E274 prevent an important overclaim. The low-rate GLC HCG-RVQ branch gives
real reconstruction/perceptual gains under soft control, but those gains are not
automatically a valid bitstream result. If the original scalar base bitstream is
kept and all active RVQ indices are sent as an enhancement, the active extra
cost is about `0.0138` bpp. That is too large for CLIC held-out 8: the mean
score changes from the no-extra quality margin `-0.010160` to the full-extra
score `+0.003321`.

This does not invalidate HCG-RVQ. It narrows the winning design. The branch
should be framed as rate-aware local residual geometry with exact fallback, not
as dense enhancement appended to the base stream. Kodak has enough headroom to
survive full active enhancement weakly (`-0.003543`), but CLIC only supports a
reduced extra-index stream. E274 estimates that CLIC can afford roughly half of
the active RVQ extra bits on average (`half score -0.003420`) but has a severe
tail (`affordable fraction min 0.241`).

The strongest next hypothesis is therefore: HCG-RVQ should replace or selectively
encode local scalar residual information in active regions, rather than always
adding a second active index stream. This fits the original prompt's thesis that
the hyperprior/context should generate local quantizer geometry. It also fits
the repeated negative controls: all-on RVQ is harmful, dense gate maps spend too
much side information, and weak/soft reliability control is where the gains
appear.

Paper-facing implication: E263-E274 are still short-cycle evidence, not final
RD curves. But they now define a credible full-training candidate and a clear
failure criterion. A promoted GLC or EF-LIC run must report base, all-on,
soft/progressive branch, selected/replacement bpp, active scalar bpp, active RVQ
bpp, gate overhead, codebook utilization, residual-stage contribution, and CLIC
failure cases. It should keep the loss simple: original codec RD/perceptual
terms dominant, plus only VQ commitment/index/rate terms that directly make the
selected branch measurable.

## Replacement-Rate Interpretation After E275

E275 substantially strengthens the GLC path. E273 showed that naive
base-plus-full-active-RVQ enhancement overpays rate on CLIC. E275 shows that the
same measured quality gains become useful again if the active RVQ stream is
viewed as replacing active scalar residual bits rather than being appended to
them. Mean replacement scores are `-0.007644` on CLIC, `-0.016521` on Kodak,
and `-0.010603` pooled, with replacement win fractions `0.875`, `1.000`, and
`0.917`.

This is exactly the kind of result that makes the HCG-RVQ story cleaner. The
method should not be sold as adding a second latent stream everywhere. It should
be sold as a hyperprior/context-conditioned local quantizer that changes the
coding geometry where scalar residual coding is inefficient, while preserving
exact fallback elsewhere. The active RVQ stream only costs about `+0.002145` bpp
more than the active scalar stream on average, so the rate burden is plausible
if the implementation truly avoids double-sending scalar and RVQ residuals.

The remaining risk is tail control. `casey-fyfe-999.png` is still slightly
positive under replacement accounting (`+0.000500`) because its quality margin
is small and replacement dbpp is high (`+0.003726`). The cap sweep suggests a
simple initial policy: require replacement dbpp below about `0.0025` before
activating the branch. This keeps CLIC negative with selected win rate `1.000`
in the current held-out slice. That policy should be treated as a codec-design
prior, not as a final hand-tuned threshold; the full implementation should make
replacement dbpp an encoder/decoder-measurable quantity and report held-out
transfer.

## Direct Replacement-Row Interpretation After E276

E276 moves replacement accounting from an external audit into the actual GLC
pilot reporting path. This matters because future experiments can now compare
base, soft-gate, full progressive extra, replacement, and capped replacement in
the same table and under the same train/eval run.

The result preserves the main E275 signal. On CLIC, full active enhancement is
positive (`+0.003378`), but replacement soft is negative (`-0.007584`) and cap
replacement soft is almost identical (`-0.007291`) while selecting only `75%` of
images. On Kodak, replacement soft is `-0.015999`, close to the unconstrained
soft-gate proxy and far better than progressive-extra full accounting. This is
the cleanest current evidence that the HCG-RVQ branch should be a local
replacement mode rather than an appended extra stream.

The result also keeps our loss-design stance intact. The gain did not come from
adding a complex auxiliary objective; the pilot loss remained image/perceptual
plus direct gate-rate and weak gate sparsity terms. The next strengthening step
should therefore improve bitstream structure and active-index modeling before
adding new losses.

## Scaling Lesson After E277/E278

E277/E278 turn the previous replacement-rate hypothesis into a stronger design
signal. The important result is not that the soft branch can improve images for
free; that was already known and is not a valid codec claim. The stronger result
is that the same quality gain remains useful when the active RVQ stream is
charged as a replacement for active scalar residual bits across a larger mixed
held-out set. Pooled `trained_replacement_soft` is `-0.010195` over 44 images,
with win fraction `0.931818`, while the additive progressive interpretation is
positive (`+0.001212`) and all-on is very harmful (`+0.044610`).

This is now the cleanest story for HCG-RVQ in GLC: hyperprior/context does not
just decide an entropy prior; it creates a local residual quantizer mode that
can replace inefficient scalar residual coding in selected regions. The branch
should not be described as a universal enhancement layer. It is a selected local
coding mode with exact fallback.

The CLIC tail also explains why the method should stay simple. The worst rows
are not caused by dead code or entropy collapse. The active codebook uses all
codes in these pilots and dead-code fraction is `0.0`. Failures are concentrated
where the replacement delta-bpp is high (`~0.0037` to `0.0044`) and the quality
margin is small. That points to bit accounting and reliability gating, not to
adding many new losses. A cap near `0.0035` is currently the best conservative
candidate: it preserves selected win `1.0` over the 44-image audit while keeping
most of the replacement improvement.

Paper-facing status: still short-cycle evidence, not final RD curves. But it is
now enough to define the next full-training/full-evaluation candidate. The GLC
candidate should implement exact selected replacement and report `soft_gate`,
`progressive_extra`, `replacement`, `capped_replacement`, and `all_on` rows, plus
active scalar bpp, active RVQ bpp, replacement delta-bpp, gate overhead,
codebook usage, residual-stage contribution, and worst-case CLIC images.



## Multi-Cap Replacement Lesson After E279/E280

E279/E280 strengthen the GLC low-rate HCG-RVQ interpretation without changing
the core design. The result is not that dense RVQ should be turned on everywhere;
that remains strongly false. Over CLIC33 + Kodak20, all-on replacement is
`+0.043588`, while selected/soft replacement is `-0.009700`. The useful method
is therefore local replacement with fallback, not universal enhancement.

The larger CLIC evidence is important. CLIC33 replacement remains negative
(`-0.007603`) with win fraction `0.878788`, while Kodak20 is stronger
(`-0.013159`, win fraction `1.000000`). Additive progressive accounting remains
positive on CLIC (`+0.003339`), showing that the same branch must be made into a
replacement coding mode before it becomes a valid low-bitrate codec claim.

The cap tradeoff is now measurable. Cap `0.0035` has selected win `1.000000` on
all 53 images and score `-0.009155`, making it the conservative paper-facing
controller. Cap `0.0040` gives a slightly better pooled mean (`-0.009758`) but
admits positive CLIC tail rows, so it should be carried as an aggressive branch
rather than used alone for the main claim. The worst replacement rows remain
CLIC images with high replacement delta-bpp, e.g. `michael-durana-82941.png`
(`+0.002223`, dbpp `+0.004138`) and `stefan-kunze-26931.png` (`+0.001777`, dbpp
`+0.003702`). This points to reliability/rate control, not codebook collapse.

For HCG-RVQ strengthening, the next implementation should keep the model simple:
original codec RD/perceptual loss stays dominant, VQ/index/rate terms are used
only when they make the replacement mode measurable, and dense all-on/all-extra
branches are retained as ablations. The paper story is becoming: hyperprior and
context generate local quantizer geometry that can replace inefficient scalar
residual coding under a measurable rate guard.


## Fixed-Index Accounting Lesson After E281

E281 is a useful guardrail for turning the GLC replacement result into a real codec claim. The empirical-index replacement score over CLIC33 + Kodak20 remains strong (`-0.009700`, win fraction `0.924528`), but a conservative fixed-index reinterpretation reduces the pooled margin to `-0.007449` and drops win fraction to `0.773585`. This does not invalidate the method; it shows exactly which part of the claim must be made explicit.

The interpretation is now sharper. HCG-RVQ is not just gaining from a better reconstruction branch. It needs an actual selected replacement code path where active scalar residual bits are not also sent, and where active RVQ indices are charged honestly. If the branch can exploit empirical index statistics or coarse selected-index coding, the current CLIC/Kodak result has comfortable margin. If it must use fully fixed-length indices everywhere, Kodak is still robust but CLIC tail rows become the limiting factor.

The cap result gives the next controller contract. Cap `0.0035` is still the safe paper-facing choice: empirical selected win is `1.000000`, fixed-index selected win is `0.926829`, and the pooled fixed-score remains negative (`-0.007736`). Cap `0.0040` is a good aggressive candidate for performance search, but it admits CLIC rows that are positive under empirical accounting and many more that become positive under fixed-index accounting. So the paper should carry both: `0.0035` for controlled reliability, `0.0040` for performance/ablation with failure analysis.

This result also changes the EF-LIC transfer standard. EF-LIC intentionally avoids entropy coding, so an EF-LIC+HCG-RVQ experiment cannot lean on unreported entropy savings. The first EF-LIC pilot should preserve the official representation-domain decorrelation and RVQ structure, insert HCG only as a decoder-safe local replacement/geometry controller, and report both fixed-index and any optional coarse-index accounting from the start.

## Controller Transfer Lesson After E282

E282 separates performance-seeking cap selection from paper-safe controller
selection. The important result is not that one threshold is universally best.
The important result is that the threshold choice reflects the claim being made.

Kodak is too easy for controller tuning. A cap selected on Kodak chooses
`0.0040` under all tested policies, and transferring it to CLIC still gives a
negative empirical score (`-0.007697`). However, the selected fixed-index win on
CLIC is only `0.677419`. That means a Kodak-only tuning story would overstate
reliability for CLIC and for EF-LIC-style fixed-index/no-entropy claims.

CLIC-aware tuning supports the earlier conservative cap. CLIC-selected
`fixed_best` and `safe_empirical_best` both choose cap `0.0035`; pooled over
CLIC33 + Kodak20 this gives empirical score `-0.009155`, fixed score
`-0.007736`, selected empirical win `1.000000`, and selected fixed win
`0.926829`. This is currently the best balance for a paper-facing GLC claim if
we also report fixed-index accounting and failure rows.

If the paper claim must be fully fixed-index with no entropy/statistical index
help, cap `0.0030` is the safer controller. It is less aggressive, but selected
fixed win rises to `0.972222` pooled and stays `1.000000` on the CLIC tail-9
leave-one-source test. This is exactly the EF-LIC lesson: because the EF-LIC core
identity is no entropy coding, the first EF-LIC+HCG experiment should not start
from the aggressive Kodak-selected cap. It should either use the stricter
fixed-safe controller or explicitly add and account for a coarse/index coding
mechanism.

The broader research implication is encouraging. The GLC branch survives
cross-split audits under both empirical and fixed accounting, so the current
HCG-RVQ direction is not just a same-split spreadsheet artifact. But the
controller, not dense model capacity or extra losses, is now the bottleneck. The
next model work should turn this cap into a decoder-safe local rate/reliability
head and keep the loss simple: original codec objective dominant, with VQ/rate
terms only where they make the selected replacement bitstream measurable.

## Signal Overhead Lesson After E283

E283 removes one more ambiguity from the GLC replacement story. The current
selected-replacement result is not relying on an uncharged, dense selection map.
If the mode is signaled coarsely at image level, the overhead is essentially
zero at the current evaluation sizes: for cap `0.0035`, pooled empirical/fixed
scores move from `-0.009155`/`-0.007736` to `-0.009154`/`-0.007734` with a
1-bit image flag and to `-0.009144`/`-0.007724` with an 8-bit image signal.
The selected empirical/fixed win fractions remain `1.000000`/`0.926829`.

The audit also gives a useful negative constraint. Dense tile-level signaling
starts eating real margin. At cap `0.0035`, a selected `64x64` tile map is still
safe but weaker (`-0.008962`/`-0.007542` pooled), while a selected `32x32` map
falls to `-0.008393`/`-0.006973` and drops selected fixed win to `0.902439`.
So the next implementation should not turn the gate into a dense transmitted
map unless that map is compressed, predicted from decoder state, or justified by
larger quality gains.

This strengthens the paper-safe interpretation: HCG-RVQ should first be framed
as a decoder-safe coarse replacement mode where hyperprior/context supplies the
local quantizer geometry and a reliability/rate controller decides whether to
replace scalar residual bits. The model should stay simple: original codec
objective dominant, VQ/index accounting explicit, and dense local selection kept
as an ablation rather than the main bitstream claim.

## Codec-Loop Signal Rows After E284

E284 is a reproducibility and bitstream-accounting step, not a new performance claim. Its value is that the selected-replacement controller now reports the same cost that the decoder would need to know in a real codec. This prevents the GLC branch from drifting into an oracle-selector story where the selection map or mode choice is silently free.

The smoke result matches the E283 conclusion. Image-level signaling is tiny at Kodak resolution: one bit on kodim01.png is 2.543e-6 bpp and eight bits are 2.0345e-5 bpp. In the trained soft cap0p003 smoke row, the score changes only from -0.016858 to -0.016856 with 1 bit and -0.016838 with 8 bits. The key point is not that this one image proves the method; it proves that future short-cycle and full-evaluation runs can report a decoder-safe selected mode without post-hoc spreadsheet adjustments.

This also sharpens the HCG-RVQ/EF-LIC integration rule. HCG-RVQ should improve the VQ/RVQ quantizer geometry or replacement branch under explicit rate terms, while EF-LICs decorrelation/no-entropy-coding identity should remain intact. If the controller needs an image-level or coarse mode signal, it is acceptable only when the cost is charged and ablated. Dense spatial gates remain diagnostic unless their signal is predictable from decoder state or compressed explicitly.

Next analysis checkpoint: rerun the larger CLIC/Kodak GLC replacement audits with --replacement-signal-bits 1 8, then aggregate no-signal vs 1-bit vs 8-bit rows under empirical and fixed-index accounting. Only after that should the same contract be ported into EF-LIC controller experiments.

## CLIC-Tail Signal-Accounted Result After E285

E285 confirms that explicit image-level signaling does not change the CLIC-tail replacement decision qualitatively. The cap 0.0035 row stays conservative: it selects 5/9 images, keeps selected win 1.000000, and changes only from score -0.005066 without signal to -0.005059 with an 8-bit image signal. The mean signal charge is 0.00000637 bpp on this mixed-resolution tail slice.

The aggressive cap 0.0040 row remains the performance-seeking branch, not the paper-safe controller. It selects all 9 images and reaches score -0.007122 without signal and -0.007115 with 8-bit signaling, but its win fraction is 0.888889 and its worst per-image score is positive (+0.001513 without signal, +0.001516 with 8-bit signal). This mirrors the E279/E280 lesson: the extra mean gain comes from accepting fragile tail rows.

For the paper story, the useful claim is now narrower and stronger: HCG-RVQ can be a coarse, decoder-safe local replacement mode whose signaling overhead is explicit and small. The current bottleneck is not signal cost and not codebook collapse; it is reliability/rate control for the hard CLIC tail. The next pooled table should therefore separate cap 0.0035 controlled evidence from cap 0.0040 aggressive performance, and both should include no-signal, 1-bit, and 8-bit signal variants.

## Signal-Accounted Current-Subset Lesson After E286/E287

E286/E287 turn the E284 signal-row implementation into a current-code controller audit across two held slices: CLIC professional tail-9 and Kodak held-16. This is still not a final full-training RD curve, but it is stronger than a one-image smoke or offline spreadsheet calculation because every row comes from the same codec-loop output and carries selection_signal_bpp explicitly.

The first clear result is that the image-level signal cost is not the bottleneck. On the 25-image pooled subset, cap 0.0035 changes from score -0.009606 to -0.009591 when an 8-bit image-level signal is charged; cap 0.0040 changes from -0.010473 to -0.010457. The mean signal charge is only 0.000015 bpp. This supports a decoder-safe coarse replacement mode: the paper does not need to hide a free oracle selector if the controller is image-level or similarly coarse.

The second result is the same reliability split seen in earlier audits, now under current-code signal-accounted rows. Cap 0.0035 is the controlled-evidence candidate: pooled selected win is 1.000000 and selected fixed-index win is 0.950000. Cap 0.0040 is the aggressive performance candidate: pooled mean score is better (-0.010473 versus -0.009606), but selected win falls to 0.960000 and selected fixed-index win to 0.880000. CLIC tail remains the limiting domain; Kodak alone would incorrectly suggest that cap 0.0040 is universally safe.

The fixed-index reinterpretation is important for EF-LIC. EF-LICs core claim is entropy-coding-free compression, so EF-LIC+HCG-RVQ should not rely on unreported empirical index entropy. On this subset, cap 0.0035 has fixed score -0.008262 and selected fixed win 0.950000, while cap 0.0040 has fixed score -0.008466 but selected fixed win 0.880000. That means EF-LIC should begin from the conservative controller or explicitly introduce a charged coarse/index side signal; the aggressive cap can remain as a performance-search branch with failure analysis.

The negative control is also stable. Dense all-on replacement remains strongly harmful: pooled trained_replacement_all_on is +0.045886, while selected replacement rows are negative. This is useful for the paper story because it shows that HCG-RVQ is not merely adding a better decoder branch everywhere. The useful contribution is local geometry/replacement under reliability and rate control.

Design implication: the next full-training candidate should keep the objective simple. Original codec RD/perceptual terms should remain dominant, VQ/index/rate terms should be present only where they make the replacement bitstream measurable, and additional losses should not be added unless a specific failure mode is observed. The main unresolved issue is controller reliability on hard CLIC images, not signal overhead or codebook collapse.

Strict-controller addendum after E287: cap 0.0030 was derived from the same measured per-image replacement rows to avoid rerunning the codec loop. It is less performant than cap 0.0035 and 0.0040, but it has the strongest selected-row reliability: pooled selected empirical win and selected fixed-index win are both 1.000000, and the 8-bit image-level signal version remains negative (score -0.008962, fixed score -0.007885). This makes cap 0.0030 the best conservative starting point for EF-LIC/no-entropy experiments, while cap 0.0035 remains the balanced GLC paper-facing controller and cap 0.0040 remains the aggressive performance branch.



## EF-LIC Current-Path Scaling Lesson After E292-E294

E292/E293 are the first current-code EF-LIC branch-controller checks that move beyond tiny four-image smokes while still staying cheap enough for design iteration. The most important result is not the small average PSNR changes; it is the reliability pattern. Zero fallback exactly preserves the EF-LIC baseline on Kodak24 and CLIC Professional 16 (`delta_bpp=0`, decode diff 0, nonfinite rows 0), so the decoder-safe integration contract is intact. Nonzero geometry states are valid codec-path states, but fixed unconditional use is not reliable enough for the paper-main method.

The split disagreement is the useful signal. On Kodak24, `soft_support020` is mildly positive (+0.003360 dB) and `constant020` is near neutral (+0.000791 dB), but both have large negative worst cases around -0.10 dB. On CLIC Professional 16, `constant020` is clearly harmful (-0.018294 dB, win 0.187500), while `soft_support020` is also negative (-0.002556 dB). Pooled over 40 images, `soft_support020` is only +0.000994 dB with win 0.525000, and `constant020` is -0.006843 dB. This is direct evidence against a dense/all-on EF-LIC HCG policy even though such a policy can look plausible and occasionally improves easy images.

The correct EF-LIC direction is therefore a learned decoder-safe fallback controller next to `_mean_scale`, preserving the official `h_a/h_s`, representation-domain decorrelation/support-buffer loop, adaptor/context predictor, and no-entropy identity. Fixed branches should remain ablations and oracle/headroom vocabulary. The model should stay simple: original EF-LIC objective dominant, HCG geometry predicted from decoder-available context, and only directly accountable VQ/index/rate plus false-positive/fallback regularizers. This also answers the all-on/full-training concern: full training may make a controller better, but the current fixed all-on/constant branch has enough CLIC harm that it should not be scaled as the main policy without reliability control.


## EF-LIC Controller Contract Lesson After E295

E295 closes an implementation gap that was still open after E292-E294. Earlier EF-LIC smokes showed that fixed nonzero HCG geometry states are decoder-safe but not reliable enough as unconditional policies. E295 shows that the replacement strategy can instead be expressed as a decoder-safe controller inserted at the correct EF-LIC location: after `_mean_scale(support_buf, slice_id)` and before the existing RVQ slice quantizer. This preserves EF-LIC hyperprior, representation-domain decorrelation/support-buffer loop, adaptor/context path, and no-entropy-coding identity.

The important contract is exact fallback. On both Kodak2 and CLIC Professional 2, `force_zero` and the fallback-biased `init_hard` controller exactly reproduced the EF-LIC baseline: `delta_bpp=0`, `delta_psnr=0`, decode diff 0, baseline diff 0, payload equality 1.0, and no nonfinite rows. This means future learned-controller experiments can safely start from a state that is behaviorally identical to EF-LIC and only deviate where the controller activates.

The soft initialization is also useful diagnostically. It produced tiny nonzero alpha maps (`~1e-7` mean) and finite, decoder-reproducible outputs with decode diff 0. The observed small PSNR changes are not performance evidence; they simply confirm that the continuous controller branch can perturb the quantizer path without breaking encoder/decoder agreement.

The research implication is now clearer. EF-LIC+HCG-RVQ should not be presented as dense all-on geometry and should not rely on extra hidden signals. The paper-main route should be a learned, decoder-safe fallback controller whose loss remains simple: original EF-LIC objective dominant, with only directly accountable VQ/index/rate and false-positive/fallback regularizers. The next meaningful experiment is controller training/calibration and mid-scale evaluation on Kodak24 plus CLIC Professional; full-training claims should wait until that controller shows consistent selected-row gains and bounded hard-image failures.


## Learned EF-LIC Controller Handoff Lesson After E296-E297

E296-E297 are important because they move EF-LIC+HCG-RVQ from static branch probes to a real learned-controller artifact flow. The controller can now be trained on decoder-safe context tensors, saved, reloaded with its own architecture config, and executed inside the EF-LIC codec loop without breaking encoder/decoder agreement. This is the correct implementation shape for the paper-main EF-LIC route.

The encouraging part is the contract. Force-zero remains exact after training because fallback is explicitly hard-coded. In E297, both Kodak2 and CLIC Professional 2 show `delta_bpp=0`, decode diff 0, and nonfinite rows 0 across loaded-controller modes. This means HCG-RVQ can coexist with EF-LIC no-entropy coding as long as the controller decision is derived from decoder-available context or any extra signal is charged.

The caution is just as important. The trained controller is a tiny teacher-smoke artifact, not a calibrated codec model. It activates substantially (`hard_gate_mean` around 0.36 to 0.45 in E297) and already shows false-positive harm on hard cases: Kodak trained hard is slightly negative on average and CLIC trained hard is -0.004265 dB on two images. Trained soft is less harsh and positive on this tiny check, but this is not enough to claim performance.

The next design update should therefore be conservative and simple. Do not add many unrelated losses. Keep the original EF-LIC objective dominant, and for the controller add only directly motivated terms: false-positive penalty, alpha/rate strength penalty, optional index/accounting term, and teacher distillation only as a warm start. The next experiment should test whether stricter calibration reduces hard-image failures while preserving selected gains; if it does, then promote to Kodak24 + CLIC Professional mid-scale and then paper-aligned full training/full evaluation.


## EF-LIC Calibration and Direction Lesson After E298-E302

E298-E302 clarify why EF-LIC+HCG-RVQ should not simply become a stronger all-on branch or a heavily penalized always-silent controller. The stricter E298 training run was safe but mostly inert: hard decisions stayed near fallback, while tiny soft perturbations still degraded Kodak4 and CLIC Professional 4. This means false-positive weighting and alpha shrinkage alone do not create a useful paper method.

The more informative result is the direction-source split. With the E296 learned controller and threshold 0.95, the original `mean` direction still had a Kodak8 tail (-0.001072 dB mean, worst -0.004282). Switching only the HCG geometry direction to `logscale` reduced Kodak8 to almost neutral (-0.000032 dB, worst -0.000636) while keeping CLIC Professional 8 positive (+0.001491 dB). On the larger E302 check, `logscale` stayed codec-safe with delta bpp 0, decode diff 0, and no nonfinite rows; it was slightly negative on Kodak24 (-0.000292 dB) and positive on CLIC Professional 16 (+0.000792 dB).

This is not yet the large EF-LIC improvement needed for a final claim, but it is a useful design result: EF-LIC appears to prefer scale/logscale-derived local geometry over mean-derived geometry for the current HCG branch. The remaining failure mode is not signal overhead or decoder mismatch; it is controller calibration on hard images such as `kodim18`. The current `risk_score` also does not separate good and bad activations reliably, since both improved and degraded images can have similarly negative risk values.

Next EF-LIC design direction: keep the official EF-LIC path and loss dominant, keep zero fallback exact, and replace the single scalar active threshold with a decoder-safe controller that can choose direction/family or image/slice-level risk. The next candidate should explicitly test `mean` vs `logscale` vs mixed direction as an ablation, and should report worst-case tail reduction as a first-class metric, not just average PSNR.


## EF-LIC Direction Choice Lesson After E303-E304

E303-E304 make the direction-choice issue sharper. `fixed` direction is currently better than both the original `mean` and the E301/E302 `logscale` direction for the E296 controller at threshold 0.95. On Kodak24 it is almost neutral (-0.000046 dB) and has a much smaller worst case (-0.000597) than `logscale` (-0.006093). On CLIC Professional 16 it is positive (+0.001030 dB) and also has a smaller worst case (-0.000185) than `logscale` (-0.002002).

This result should not be oversold as a final performance claim: the mean gains are still tiny and the win fraction is low. But it is strong implementation guidance. The current HCG branch is useful only when its geometry direction is controlled; `mean` can create damaging geometry, `logscale` is safer, and `fixed` is the safest tested option. That suggests the next real EF-LIC design should use a small direction/family selector, or at least present direction source as an ablation in the paper.

The result also supports a simple-loss philosophy. The improvement came from changing the quantizer geometry direction under the same controller and codec objective, not from adding more auxiliary losses. For the next training pass, keep the original EF-LIC objective dominant and add only the minimum controller terms needed to prevent false positives and select direction/fallback.


## EF-LIC Direction/Fallback Selector Lesson After E305-E306

E305-E306 close the small ambiguity left after the fixed-direction result. The
current EF-LIC HCG branch should not be interpreted as a single universally good
geometry source. Under the same E296 controller and threshold 0.95, `fixed` is
the strongest single policy, but the matched oracle shows that some images still
prefer `mean` or `logscale`, and many Kodak images should simply fall back to the
original EF-LIC quantizer.

The Kodak24 oracle is mainly a safety result. `fixed` is already close to neutral
(-0.000046 dB), and `oracle_with_fallback` only reaches +0.000046 dB, choosing
fallback on 20/24 images. This means the next Kodak-facing controller should be
judged by worst-case tail removal and exact fallback, not by large average gains
from this small branch.

The CLIC Professional 16 oracle is a little more promising. `fixed` gives
+0.001030 dB, while `oracle_with_fallback` reaches +0.001144 dB with no negative
rows and `oracle_nonfallback` reaches +0.001142 dB with only -0.000028 worst
case. The headroom is not large enough for a final claim, but it is enough to
justify a decoder-safe direction/fallback selector as the next EF-LIC design
step.

The design implication is deliberately simple. Do not add a stack of auxiliary
losses to chase this. Keep the official EF-LIC RD/perceptual objective dominant,
keep the representation-domain decorrelation and no-entropy-coding identity
unchanged, and train a small decoder-available controller to choose fallback vs
mean/logscale/fixed geometry. If any direction or mode signal is not derivable
from decoder state, it must be explicitly signaled and charged.


## EF-LIC Selector Proxy Lesson After E307

E307 is a useful negative result. If we restrict the practical controller to a
single monotonic threshold on `gate_mean` or `alpha_mean`, the EF-LIC HCG branch
can be made safe, but only by selecting almost no images. On Kodak24 the best
safe threshold selects 0/24 images, giving exactly fallback behavior. On the
40-image pooled subset, the best safe threshold selects only 1/40 image and
reaches +0.000414 dB, which is safe but far below the direction/fallback oracle
story.

This explains why the next controller should not just tune the existing active
threshold. The useful Kodak rows and harmful Kodak rows are interleaved in gate
and alpha magnitude; magnitude is not the right decision variable. A real
selector needs richer decoder-safe context and direct labels for fallback vs
mean/logscale/fixed, or an explicitly signaled coarse mode whose cost is
charged.

For the paper, this is a clean ablation. It shows that HCG-RVQ is not merely
"use a bigger gate" or "turn on geometry harder". The contribution has to be
conditional local quantizer geometry with reliability-aware direction/fallback
control. The loss should remain simple: original EF-LIC objective dominant,
plus only the minimal terms needed to train this controller and account for any
VQ/index/mode cost.

## EF-LIC Selector Trainability Lesson After E308-E309

E308-E309 are a useful stopping point for the image-level selector idea. The
matched E306 oracle said that fallback/mean/logscale/fixed choice has headroom,
but directly training a small decoder-safe context head on those image-level
labels is not enough. The strict run collapses to fallback, while the capacity
run learns to activate but reintroduces the same Kodak tail that the controller
was supposed to avoid.

The best same-pool probability-gated row is safe and mildly positive: capacity
`pred_choice_nonfallback_conf >= 0.79` selects 3/40 images, mean +0.000340 dB,
and worst +0.000000. But this is not robust evidence for a paper-main method,
because leave-dataset-out policies either fall back completely or trade mean gain
for negative tails. That means image-level direction labels are too sparse and
possibly too dataset-specific for direct deployment.

The next EF-LIC route should move one level closer to the actual HCG-RVQ
hypothesis: learn local or slice-wise quantizer geometry decisions from
hyperprior/support-buffer context. In practical terms, the next labels should be
built from residual/headroom at the slice or spatial-block level, with fallback
exactly preserving EF-LIC. A simple objective is still preferred: original EF-LIC
loss dominant, one controller/rate penalty, and explicit accounting for any
mode/index signal. Adding unrelated losses is unlikely to fix the current
failure mode, which is decision granularity and false-positive control.

## EF-LIC Local-Controller Direction After E310

E310 connects the recent image-level direction selector failure with the older
local activation studies. The shared lesson is that the HCG-RVQ signal exists,
but it should not be expressed as one global mode label per image. A global
selector is too coarse: safe thresholds fall back almost everywhere, while
mean-seeking thresholds reintroduce hard-tail failures.

The local activation artifacts show why the next design should be two-stage.
Activation is partly separable from decoder-safe features, but a single
threshold has a poor precision/recall tradeoff. Therefore the controller should
not be a flat multiclass family/direction head. It should first learn a
conservative activation/fallback decision, then choose family/direction/strength
only where local context suggests quantizer-geometry headroom.

For full training, this is a useful risk reduction. The first full EF-LIC branch
should keep the official EF-LIC transform, hyperprior, support-buffer
representation-domain decorrelation, adaptor/context predictor, and no-entropy
identity intact. HCG-RVQ should only modify the RVQ quantizer geometry through a
small decoder-safe controller. The loss should stay simple: original codec loss
plus minimal false-positive and strength/rate terms. The next required
experiment is to create local direction/residual-headroom labels, then train the
two-stage controller and evaluate it with Kodak/CLIC split, checkpoint, and
intermediate-feature diagnostics.

## EF-LIC Slice/Candidate Policy Signal Lesson After E311

E311 confirms the useful part of the E236 local-policy oracle while also marking
its boundary. Candidate-level EF-LIC HCG policies are not random: residual-error
features separate useful sparse-union rows on Kodak24 with oriented AUC around
0.76, while CLIC Professional 41 exposes stronger geometry/index-use signals
with top AUC around 0.70-0.72. This supports the HCG-RVQ hypothesis that local
quantizer geometry should be conditioned on decoder-side context rather than
chosen as one global image setting.

The negative result is equally important. Single-feature rules trained on one
dataset do not transfer safely. The Kodak-trained best-mean residual rule still
has positive CLIC tail risk, and the CLIC-trained safe rule transfers to Kodak
only by falling back everywhere. Therefore E311 should not become a paper-main
selector. It is a label-design and feature-audit artifact: it tells us which
families of signals matter, not a deployable decision rule.

The next EF-LIC design should use E311 to build local or slice-level labels:
activation/fallback from residual or headroom, then family/direction/strength
only inside the activated region. That preserves the simple codec objective and
keeps EF-LIC intact: representation-domain decorrelation, hyperprior/support
buffer, adaptor/context predictor, and no-entropy-coding identity stay in place.
HCG-RVQ modifies the RVQ quantizer geometry only where decoder-available context
says there is headroom. Any extra mode/index signal must be decoder-derived or
charged. This is the strongest current route toward a full-training EF-LIC claim
because it addresses the observed failure mode: false positives and coarse
selection, not lack of auxiliary loss terms.

## EF-LIC Slice Interaction Lesson After E312

E312 is the first direct bridge from the E310/E311 design decision to an actual
codec-loop local-label mechanism. The one-image Kodak probe is small, but the
pattern is highly informative: HCG perturbations are not additive across EF-LIC
y-slices. On `kodim01.png`, slice1 alone improves PSNR by +0.006996 dB, yet
removing slice1 from the all-slice branch gives the best tested subset
(`0,2,3`) at +0.012344 dB, which is +0.004039 above all-slice activation.
Meanwhile slice2 is negative alone (-0.004120 dB) but can be part of the best
combined subset. All rows stayed codec-valid with delta bpp 0, decode max 0,
and no nonfinite outputs.

The important lesson is that EF-LIC HCG-RVQ should not use an independent
per-slice on/off label learned from single-slice deltas alone. EF-LIC updates
its support buffer after each decoded y-slice, so early geometry changes alter
the context seen by later slices. The controller therefore needs sequential
context awareness: conservative activation/fallback, then direction/family and
strength conditioned on the current support buffer and the already reconstructed
previous slices.

This strengthens the current HCG-RVQ hypothesis rather than weakening it. The
useful variable is not just whether a slice likes geometry perturbation in
isolation; it is whether the local quantizer geometry is beneficial under the
current hyperprior/support-buffer state. For the next EF-LIC training artifact,
labels should combine leave-one-out marginal effects, residual/headroom
features, and local/slice context. Full-training should keep the original EF-LIC
codec objective dominant and use only minimal controller/rate penalties so that
we are testing quantizer geometry, not an over-engineered loss stack.

## EF-LIC Multi-Image Slice Policy Lesson After E313

E313 broadens E312 from a single Kodak image to the first four Kodak images
while keeping the exact EF-LIC codec loop, fixed-length payload, and GPU0-only
evaluation. The contract is clean: every tested row has delta bpp 0, decode max
0, contract_ok_frac 1.0, and no nonfinite output. This makes the result useful
as controller-label evidence rather than an artifact of asymmetric
compress/decompress behavior.

The main result is a useful tension. The fixed `all` policy is strongest by
mean on this tiny subset (+0.030334 dB), so HCG-RVQ is not merely random
perturbation. However, `all` is not the best policy per image: `kodim01` and
`kodim03` prefer `0,2,3`, `kodim04` prefers slice2 only, and only `kodim02`
prefers `all`. Thus the correct EF-LIC integration is not "always turn HCG on"
and not "disable the weak slices globally". The policy must be conditional.

The single-slice behavior explains why. Slice1 is safest as a single slice
(positive on all four images, mean +0.007593 dB), while slice2 and slice3 are
negative on average. But slice2 participates in the best subset for
`kodim01`, `kodim03`, and `kodim04`. This means single-slice deltas are
insufficient labels: later EF-LIC slices are decoded under a support-buffer
state produced by earlier slices, so marginal value depends on current context
and on already-applied HCG geometry.

For the paper path, E313 supports a sharper and simpler claim: HCG-RVQ should
be a decoder-safe local quantizer-geometry controller, not an extra loss-heavy
side module. The official EF-LIC redundancy-reduction machinery should remain
intact, and HCG should learn where the local RVQ geometry has headroom. The
next implementation should train a two-stage sequential controller: first a
conservative activation/fallback decision, then slice/family/direction/strength
inside activated regions. The labels should use contextual leave-one-out
marginals from E313-style sweeps together with residual/headroom and
support-buffer features. This is the current best bridge from small-cycle
evidence to full-training EF-LIC.

## EF-LIC Sequential Label Lesson After E314

E314 makes the E313 result operational. It shows that the paper-safe next step
is not another global switch and not a single-slice sign rule. The best tested
per-image oracle over E313 subsets reaches +0.034736 dB mean, which is +0.004402
over all-slice HCG activation and has a nonnegative worst case on the four-image
probe. That is small but real headroom inside the already codec-valid branch.

The label agreement numbers explain the risk. Single/context sign agreement is
0.75, but oracle/single agreement is only 0.5625. In other words, a slice can be
positive alone yet excluded from the best subset, or negative alone yet included
in the best subset. This is exactly what a sequential EF-LIC controller should
model: earlier y-slice geometry changes the support buffer and therefore the
conditions under which later quantizer geometry helps.

The current feature correlations are deliberately treated as weak diagnostics:
lower index entropy, lower residual RMS, and lower risk score correlate mildly
with oracle-active slices on the tiny Kodak4 sample. This is not transfer
evidence. It is a feature shortlist for the next local-label training run. The
method claim should stay simple: preserve EF-LIC's representation-domain
decorrelation and no-entropy-coding path, then add a decoder-safe controller that
chooses local RVQ geometry where the hyperprior/support-buffer state predicts
headroom.

The next concrete experiment should use E313/E314 as teacher construction logic
rather than as the final policy: expand the sweep to a larger Kodak/CLIC subset,
construct sequential labels with margins, train a two-stage activation plus
slice/direction/strength controller, and evaluate it against `all`, exact
fallback, and the original EF-LIC branch. Promotion requires mean improvement,
no hard-tail regression, zero decode mismatch, zero nonfinite rows, and feature
analysis showing selective rather than blanket geometry changes.

## EF-LIC Kodak24 Controller Evidence After E315-E320

E315-E320 change the EF-LIC story from a one-image/local curiosity into a clearer controller problem. The positive result is that HCG-RVQ can improve the EF-LIC reconstruction under the original fixed-payload contract. On Kodak24, `all` activation gives mean +0.006391 dB with delta bpp 0, decode max 0, contract_ok_frac 1.0, and no nonfinite rows. This means the HCG branch is codec-valid and not merely exploiting an asymmetric evaluation artifact.

The negative tail is the central issue. The same `all` policy has worst -0.030184 dB, so it is not paper-safe as the main design. E317 resolves the correct upper-bound question by adding exact fallback `none` and the full 16-way slice powerset. The full per-image powerset oracle reaches mean +0.016323 dB with worst 0.0, and the simpler all/none oracle reaches mean +0.011310 dB with worst 0.0. This is strong evidence that useful HCG-RVQ headroom exists, but only if the method can decide where not to apply geometry perturbation.

E318 also changes the labels. Once `none` is available, the oracle active fraction is 0.468750 rather than a blanket-on regime. The best subsets differ by image: some images use all slices, some use a sparse subset, and some choose exact fallback. This supports the HCG-RVQ hypothesis in its local form: hyperprior/support-buffer context should generate quantizer geometry only where there is local headroom.

E319 and E320 are important guardrails. Simple leave-one-image-out threshold policies do not recover the oracle headroom. The slice-level threshold policy falls to mean +0.002686 dB, worse than `all`; the image-level all/none gate is +0.005202 dB, also worse than `all`, and still keeps the -0.030184 dB tail. This prevents an overclaim such as "one feature threshold solves reliability." The right conclusion is not that HCG-RVQ is weak; it is that the controller needs to be learned from teacher labels rather than hand-written from a single statistic.

The next EF-LIC implementation should therefore be a learned decoder-available controller trained from E317/E318-style teacher labels. Keep the codec objective simple and original-RD dominated. First learn a conservative fallback/activation decision, then learn slice/family/direction/strength inside activated regions. Promotion should require: mean improvement over `all`, reduced negative tail, zero bpp change under the fixed-length EF-LIC contract, zero decode mismatch, no nonfinite rows, and feature analysis showing selective geometry use rather than blanket activation.

This also clarifies the full-training path. Short-cycle experiments remain useful because they exposed the exact failure mode and produced teacher labels. They are not final performance evidence. Full EF-LIC/GLC training should start only after the learned controller beats `all` and simple gates on held-out Kodak-style diagnostics, then be rerun under the official low-bitrate settings.

## EF-LIC Controller Evidence After E321/E322

E321/E322 sharpen the EF-LIC conclusion. The HCG-RVQ branch still has real
headroom under the fixed-payload EF-LIC contract, but the currently available
image/slice summary features are not enough to learn a paper-safe controller.

The strongest upper bound remains the E317 full powerset oracle: +0.016323 dB
mean with worst 0.0, compared with all-on activation at +0.006391 dB mean and
-0.030184 dB worst. E321 tried the next natural step after simple thresholds: a
regularized logistic slice gate trained on E318 oracle-active labels under
leave-one-image-out CV. It excludes PSNR-delta and oracle-outcome columns, tunes
hyperparameters only on the training fold, and scores held-out predictions with
the E317 powerset lookup. This learned summary-feature controller reaches only
+0.001912 dB mean, with worst -0.017709 dB. It therefore does not recover the
oracle headroom and is worse than all-on by -0.004478 dB mean.

E322 shows that this failure is not simply because all-on is unbeatable. A
train-selected fixed subset can beat all-on slightly: LOOCV fixed subset with
mean-only train selection reaches +0.007360 dB mean, +0.000969 over all-on, but
still has a -0.020546 dB worst case. Tail-heavy selection eventually collapses
to exact fallback, which removes the tail but loses the mean gain. This gives a
clean hierarchy:

- full subset oracle: useful and safe upper bound;
- fixed subset: small mean gain, unsafe tail;
- simple threshold and summary-feature logistic controllers: not yet reliable;
- fallback-only: safe but no HCG benefit.

The design implication is important for the paper story. HCG-RVQ should not be
sold as a global EF-LIC switch, and not as a shallow classifier over aggregate
statistics. The evidence points to a local/sequential controller: use the same
decoder-available context maps that EF-LIC has after each support-buffer update,
train conservative spatial activation with exact fallback, then choose
family/direction/strength only inside activated regions. This keeps EF-LIC's
representation-domain decorrelation and no-entropy-coding identity intact while
moving HCG-RVQ closer to its actual hypothesis: hyperprior/support context
generates local quantizer geometry where there is local headroom.

For full training, the criterion is now clearer. Do not launch a paper-main
full EF-LIC run from the E321 summary-feature controller. First build a
spatial/sequential teacher or controller artifact that beats all-on and fixed
subset under held-out diagnostics, preserves zero bpp/decode mismatch, and
reduces the negative tail. The loss should remain original-RD dominated with at
most minimal activation/rate/strength calibration terms; the observed bottleneck
is decision granularity, not a need for many auxiliary losses.

## EF-LIC Controller Interpretation After E323-E337

E323-E337 move the EF-LIC branch from "there is oracle headroom" to "there is a
specific trainable controller route that already improves the codec loop, but
still needs a paper-safe validation protocol."

The first important correction is teacher provenance. E323 shows that the older
E242 spatial teacher is over-active relative to the newer fallback-aware
E317/E318 oracle: its active fraction is 0.614583 versus 0.468750, and its
correlation with the E318 oracle-active fraction is only 0.230389. This means
E242 labels should not be used as the final supervision source. E324 fixes the
provenance problem by keeping the decoder-available context maps and replacing
the labels with E318-aligned fallback-aware targets.

The second lesson is controller balance. The first E325 controller preserved
fallback exactly and was codec-valid, but it became too conservative: the eval
hard gate mean was only 0.0050. In codec-loop evaluation, its hard mode reached
only +0.002152 dB on the held-out eval8 split, recovering 17.0% of the eval8
E317 powerset headroom. Lowering the hard threshold to 0.25 worsened the result,
so the failure was not simply under-activation.

E329 changes the picture. With a more balanced false-positive/missed-active
training objective, the controller becomes genuinely useful: at risk 0.0 it
reaches +0.006544 dB on eval8, but with a dangerous -0.058768 dB tail. Adding a
simple max-risk fallback is therefore not cosmetic; it directly addresses the
observed failure mode. The best discovery setting so far is E335, the E329 hard
controller with max-risk -0.15 on full Kodak24:

- mean delta PSNR: +0.012898 dB;
- E317 all-on full24 mean: +0.006391 dB;
- E317 powerset oracle full24 mean: +0.016323 dB;
- oracle headroom recovered: 79.0%;
- delta bpp: 0;
- decode mismatch: 0;
- nonfinite rows: 0.

This is a strong EF-LIC result, but it is not yet a final paper claim. The
threshold was selected during exploration, and the full24 evaluation contains
both controller-training images and held-out diagnostics. The correct paper
interpretation is: HCG-RVQ has a promising controller design that can improve
EF-LIC under its fixed-payload/no-entropy-side-bit contract, and the next task is
to make the selection protocol independent.

The failure cases clarify the next design. E335 is mainly harmed by false
positive activation on images whose E317 oracle prefers exact fallback or a
different sparse subset. Examples include `kodim05`, `kodim19`, and `kodim22`,
where E317 selects `none` but the controller still activates geometry. E337
shows that a stricter risk threshold (-0.20) is too conservative: full24 mean
drops to +0.007246 dB. Therefore, the next controller should not merely lower
activation globally. It should learn a better no-op/fallback decision and then
apply local geometry only where the decoder-available context predicts headroom.

This fits the prompt core thesis cleanly. EF-LIC representation-domain
decorrelation and no-entropy-coding path remain intact; HCG-RVQ modifies the
quantizer geometry inside that path. The current evidence suggests the
redundancy-reduction claim of EF-LIC and the quantizer-geometry claim of
HCG-RVQ are compatible: EF-LIC makes the representation compact and fixed-rate,
while HCG-RVQ decides how to quantize the remaining local residual geometry.

Near-term promotion criteria:

- select risk thresholds on train/validation only;
- evaluate once on a held-out Kodak/CLIC-professional split;
- report mean, worst-case tail, positive/negative image counts, bpp, decode mismatch, and nonfinite rows;
- include intermediate statistics: alpha/gate distribution, y mismatch, geometry delta RMS, index entropy, and activation on `none`-oracle images;
- only promote a setting if it beats E317 all-on and fixed-subset baselines without relying on test-tuned thresholds.

## EF-LIC Risk-Threshold Lesson After E338-E342

E338-E342 clarify the strongest current EF-LIC opportunity and its main weakness.
The balanced E329 controller is not a small artifact: across full Kodak24, every
risk-grid row keeps delta bpp 0, decode max 0, and nonfinite 0. The branch is
therefore compatible with EF-LIC fixed-payload decoding. The real question is
how to choose activation safely.

The full24 grid is encouraging. Risk-none gives the largest mean improvement,
+0.013690 dB, recovering 83.9% of the E317 powerset oracle headroom. Risk -0.15
keeps nearly the same mean, +0.012898 dB, while reducing the worst case from
-0.058768 dB to -0.038376 dB. Risk -0.20 improves the worst case further to
-0.026403 dB, but its mean drops to +0.007246 dB. This is the expected
mean/tail tradeoff: HCG geometry is useful, but false positives are expensive.

E342 prevents overclaiming. When the risk threshold is selected using only
`kodim01`-`kodim16` and evaluated on `kodim17`-`kodim24`, ordinary mean or
moderate-tail objectives select risk-none, whose eval tail is still -0.058768
dB. A very tail-heavy objective selects risk -0.20 and obtains eval mean
+0.016443 dB with worst -0.024418 dB. The best eval8 row, risk -0.15, reaches
+0.019047 dB with worst -0.017333 dB, but the current train16 objectives do not
select it.

This means the present EF-LIC result is genuinely promising but not yet a final
conference result. The useful contribution is not a hand-picked max-risk value;
it is the evidence that a decoder-safe reliability controller can recover most
of the HCG-RVQ oracle headroom under EF-LIC no-side-bit constraints. The next
method improvement should turn this into a learnable decision: a no-op/fallback
classifier trained on E318/E317 `none`-oracle cases, followed by local geometry
activation for non-fallback regions.

Paper-safe next step:

- freeze the risk policy before seeing the test split;
- prefer a learned fallback/no-op head over a single global post-hoc threshold;
- evaluate on Kodak folds and an independent low-bitrate set such as CLIC professional;
- report risk-none, validation-selected risk, no-op classifier, E317 all-on, fixed subset, and E317 oracle as separate rows;
- keep the loss simple: original EF-LIC reconstruction/fixed-payload behavior remains dominant, with only minimal controller calibration.


## EF-LIC No-op/Fallback Signal After E343-E344

E343-E344 refine the EF-LIC controller story. The earlier risk-grid result was
not wrong, but it was incomplete: global `max_risk` is too blunt because the
learned risk score is partly a teacher-active signal, not a calibrated codec-gain
signal. In E343, lower `slice_risk_score_mean_max` is associated with negative
controller outcomes for risk-none (`corr_controller_negative = -0.571469`, AUC
for negative = 0.148148). That means the current sign/meaning of the risk score
is not enough for safety. It can identify a useful active pattern, but it does
not yet answer the paper question: should this image/slice receive HCG geometry
at all?

The encouraging result is that decoder-visible features do contain fallback
information. On full Kodak24, a non-leaky feature threshold on
`slice_risk_score_mean_max` improves risk-none from +0.013690 dB mean / -0.058768
worst to +0.022118 dB mean / -0.008581 worst. This is an optimistic full-set
upper bound, not a final result, but it shows that the HCG failures are not
random. They are predictable from controller/context statistics.

E344 is the colder check. Selecting thresholds on `kodim01`-`kodim16` and
evaluating on `kodim17`-`kodim24` still helps risk-none: eval mean improves from
+0.006544 dB to +0.016563 dB and worst improves from -0.058768 dB to -0.022600
with one remaining negative image. Tail-heavy selection on risk -0.005/-0.010
uses `z_index_used_frac <= 0.065430`, giving zero negative eval images but also a
conservative policy. For risk -0.15, the raw eval row is already best
(+0.019047 dB, worst -0.017333), while split-selected fallback over-suppresses and
drops mean to +0.008940 dB.

Design consequence: HCG-RVQ for EF-LIC should add a learned no-op/fallback head
or validation-selected reliability policy, but it should not blindly suppress
all low-confidence images. The head should be trained against actual codec gain
or held-out teacher labels, not only the current active-map labels. This keeps the
method simple and aligned with the prompt: EF-LIC's fixed-payload decorrelation
path remains unchanged, while HCG-RVQ learns when local quantizer geometry is
worth applying.

Paper-facing caution:

- E343 full24 feature thresholds are discovery evidence only.
- E344 split selection is a stronger diagnostic but still shares the same Kodak/controller-training ecosystem.
- The next paper-safe result needs train/validation/test separation, preferably with CLIC professional or another independent low-bitrate split.
- Report both mean gains and tail behavior; HCG-RVQ is only convincing if it improves RD without hiding fragile false positives.

## EF-LIC No-op/Fallback After E345

E345 is a useful negative result. The idea was conservative: if decoder-visible
summary features already contain fallback information, then a tiny learned
classifier should improve over single-feature thresholds. It does not, at least
not in the current image-level form. On the held-out Kodak split, risk-none moves
only from +0.006544 dB to +0.006827 dB and still keeps the -0.058768 dB failure.
Risk -0.015, the best raw eval row, is damaged by the learned fallback because it
suppresses useful activations. Risk -0.020 can be made tail-safe, but only by
becoming conservative and losing mean gain.

This changes the implementation priority. The main HCG-RVQ hypothesis is still
supported: EF-LIC's fixed-payload RVQ path accepts decoder-reproducible local
geometry, and E317/E338-E344 show real oracle and controlled-policy headroom. But
the controller cannot be a global image classifier trained from the same small
Kodak ecosystem. The decision has to move closer to the EF-LIC sequential slice
loop, where the decoder-visible context, support-buffer state, local residual
statistics, and family/direction choice are available.

Paper-facing implication: E343 full-set thresholds and E345 linear probes are
design evidence, not final rows. The next credible row should use a frozen
local/sequential selector trained on independent codec-gain labels, with exact
fallback to the original EF-LIC RVQ branch and unchanged/fixed-index bit
accounting. Only after that selector passes split and independent-set checks is
it worth launching full EF-LIC/GLC training as a performance claim.

Immediate next experiments:

- build codec-gain labels for local EF-LIC decisions from the E317/E318 slice
  powerset and E324 aligned context tensors;
- train a minimal no-op/family controller whose primary target is fallback safety,
  not just active-map reconstruction;
- evaluate with Kodak folds plus CLIC Professional, reporting mean, worst case,
  negative count, selected fraction, index/codebook usage, and nonfinite/decode
  exactness;
- keep the loss simple: original RD/perceptual codec objectives dominate in
  future full training, with only small VQ commitment and controller calibration
  terms.

## EF-LIC Codec-Gain Risk After E346-E348

E346-E348 changes the controller diagnosis in an important way. The older risk
signal behaved too much like an activation proxy: it could open or suppress HCG,
but it was not directly calibrated to whether a local geometry edit helps the
codec. E346 makes the target closer to the paper question by assigning negative
risk to slices with positive contextual codec margin. That means the risk head is
trained to answer: "is this HCG edit likely to improve the actual codec loop?"
rather than merely "is this slice active?"

The held-out Kodak17-24 grid is encouraging but not final evidence:

- Best mean: `max_risk=-0.08`, +0.021775 dB mean, -0.022029 dB worst, 2 negative images.
- Best mean/tail compromise: `max_risk=-0.06`, +0.017634 dB mean, -0.016126 dB worst, 3 negative images.
- Best tail: `max_risk=-0.10`, +0.008476 dB mean, -0.011862 dB worst, 4 small negative images.
- All rows keep delta bpp 0, decode max 0, and nonfinite 0.

Compared with the earlier E342 raw risk rows, codec-gain risk is more useful:
`max_risk=-0.08` exceeds the previous eval8 best raw row (`risk=-0.15`,
+0.019047 dB mean), and `max_risk=-0.06` gets close while improving tail. This is
not just a cosmetic threshold sweep; it supports the hypothesis that HCG needs a
codec-gain-calibrated reliability signal, not a generic no-op classifier.

However, the result is still controlled evidence. The controller is trained on
Kodak01-16 and the threshold grid is inspected on Kodak17-24, so a paper claim
would require an independent calibration/selection protocol. The next decision
boundary is:

1. Build the same codec-gain teacher on a non-Kodak calibration split, preferably
   CLIC professional or another validation set that matches the EF-LIC evaluation
   protocol.
2. Select `max_risk`, active threshold, and any fallback policy on that
   calibration split only.
3. Evaluate the frozen policy on Kodak and a disjoint CLIC/professional split,
   reporting PSNR/MS-SSIM/bpp, exact bpp contract, decode consistency,
   nonfinite rows, per-image wins/losses, and gate/alpha/risk distributions.
4. Move to full training only after the frozen policy remains positive under this
   protocol. Otherwise full training would test an unstable controller choice
   rather than HCG-RVQ itself.

For EF-LIC, the current best implementation direction is therefore simple and
interpretable: keep the original EF-LIC decorrelation/representation path and
fixed-payload claim intact, add a small hyperprior/context-conditioned HCG RVQ
branch, and control it with a codec-gain-calibrated reliability head. This keeps
the method compatible with EF-LIC's "no entropy coding" story: HCG changes the
quantizer geometry and reconstruction quality while preserving the transmitted
index payload in these smoke evaluations.


## EF-LIC External Transfer After E349

E349 is a useful full-training gate. It freezes the Kodak-trained E347
codec-gain controller and applies it directly to CLIC Professional 41 images, so
it tests transfer rather than in-domain threshold chasing. The result is mixed
but clearly informative. The HCG branch remains bitstream-compatible with
EF-LIC: delta bpp is 0, decode max is 0, and nonfinite rows are 0. This preserves
the EF-LIC entropy-coding-free/fixed-payload story while changing only the
quantizer geometry.

The performance signal transfers, but modestly. Risk -0.06 has the best mean
(+0.004621 dB) but a -0.036221 dB tail failure. Risk -0.08 is the best balance
(+0.003411 dB mean, -0.014399 dB worst). Risk -0.10 is not automatically safer
(+0.002682 dB mean, -0.029681 dB worst), which means the risk scalar controls
interaction strength but is not a monotone guarantee of codec gain.

The important headroom result is the no-op/risk oracle: choosing among noop and
the three risk points per image gives +0.010452 dB mean with worst 0 on CLIC41.
That supports the HCG-RVQ hypothesis more strongly than any single fixed risk:
the local geometry edits are beneficial on external data, but reliability
selection is the bottleneck. This aligns with the earlier GLC lesson that
selected replacement beats dense/all-on active quantization.

Paper-facing implication: the next EF-LIC method should stay simple and
explainable. Keep EF-LIC Representation-domain Decorrelation and original
fixed-index payload unchanged; add HCG-RVQ as a decoder-reproducible geometry
branch; train a small codec-gain-calibrated controller to choose no-op or a small
set of risk strengths. Do not add many auxiliary losses. Future full training
should primarily optimize the original RD/perceptual objective plus the minimal
VQ/controller terms needed for stable quantization and reliability calibration.

Full-training readiness is therefore conditional, not blocked indefinitely. The
EF-LIC branch can move to longer/full training once a frozen selector selected on
CLIC/Kodak calibration remains positive on held-out Kodak/CLIC with exact bpp and
controlled tails. Launching full training before that would risk spending compute
on a controller policy that E349 already shows is not fully reliable.


## EF-LIC Perceptual Metric Correction After E350-E351

The user's concern that this branch is generative/perceptual rather than pure
PSNR compression is correct. PSNR is still useful because it catches codec-health
regressions, large distortion failures, and unintended payload/decoder changes;
however, it must not be the main selector for a low-bitrate generative claim.
The paper-facing metric stack should treat LPIPS/DISTS/MS-SSIM and qualitative
comparisons as primary, with bpp and decode exactness as constraints.

E350 confirms why this matters. On CLIC Professional 41 at risk -0.08, mean dPSNR
is positive (+0.003411), but dPSNR has almost no correlation with the perceptual
score `delta_DISTS + 3 * delta_LPIPS` (corr -0.046446). There are 13 images where
PSNR improves while the perceptual score worsens, and 7 images where PSNR drops
while the perceptual score improves. A PSNR-only report would therefore select
or reject HCG edits for the wrong reason in several cases.

E351 expands this to the three risk strengths. Risk -0.06 is the best fixed
choice by mean perceptual score (-0.000257) and also has the highest mean dPSNR
(+0.004621), but its PSNR tail is unsafe (-0.036221). Risk -0.10 is more
conservative in gate/alpha and wins the perceptual score on 25/41 images, but it
has weaker average improvement and still a -0.029681 dPSNR tail. Risk -0.08 is a
middle point with the best PSNR worst case (-0.014399), but its perceptual score
is weakest among the three fixed risks.

The no-op/active oracle is the key scientific signal: choosing per image among
noop and the three risk strengths gives mean perceptual score -0.000913 with
worst score 0, substantially better than any fixed risk. This supports the
HCG-RVQ hypothesis in a sharper way than the earlier dPSNR-only transfer: the
geometry branch can improve external CLIC images without changing payload length
or decoder consistency, but the bottleneck is reliability selection under
perceptual metrics.

Implication for EF-LIC: keep EF-LIC's original Representation-domain
Decorrelation and fixed-payload/no-entropy-coding story intact. HCG-RVQ should be
introduced as a decoder-reproducible quantizer-geometry branch. The next method
component should be a small metric-aware reliability selector that decides noop
vs a small family of HCG strengths. It should be calibrated on a disjoint split
and evaluated frozen on held-out Kodak/CLIC. Full training should not be launched
as the final claim until this selector is stable under LPIPS/DISTS/MS-SSIM, bpp,
decode exactness, nonfinite checks, and PSNR tail diagnostics.

This does not make the short-cycle experiments meaningless. They are doing the
right job: they found that HCG geometry is payload-safe and has real oracle
headroom, and they also prevented us from making a misleading PSNR-only claim.
For full training, the loss should stay simple: the original codec objective
(RD/perceptual according to the target paper setting) plus only the minimal VQ
commitment and selector calibration needed for stable HCG behavior. Adding many
auxiliary losses would make the main contribution harder to optimize and harder
to explain.


## EF-LIC Selector Readiness After E352

E352 is a useful guardrail against over-interpreting the E351 oracle. The oracle
shows that HCG geometry edits can improve perceptual metrics, but a paper method
cannot use an oracle. A simple threshold selected on the first 20 CLIC images does
transfer with a negative held-out perceptual score (-0.000121), so the signal is
not random. However, it is weaker than the best fixed-risk held diagnostics and
far weaker than the held oracle (-0.001106). This means the current global
threshold family is not enough to claim a robust reliability controller.

The more interesting diagnostic policy in the held split is not the one selected
by calibration: a residual-error max-feature rule reaches eval score about
-0.000523 and mean dPSNR +0.00769 with choices spread across risk strengths.
This suggests the selector should not merely ask whether to turn HCG on; it
should use local residual/context information to choose the HCG strength/family.
That aligns with the main HCG-RVQ story: local geometry and local reliability are
both content dependent.

The next EF-LIC experiment should therefore move from hand thresholds to a small
learned selector, but still keep the model simple. A good target is:

- input: decoder-visible EF-LIC context/hyper features, local residual/proxy
  statistics, gate/alpha/risk summaries;
- output: noop / risk -0.06 / risk -0.08 / risk -0.10 or a small continuous risk
  strength;
- target: perceptual codec-gain label using DISTS+LPIPS, with PSNR tail as a
  constraint rather than the main objective;
- evaluation: split-selected frozen policy on held-out Kodak/CLIC, reporting
  bpp/decode exactness, LPIPS/DISTS/MS-SSIM, PSNR tail, nonfinite rows, and
  qualitative examples.

Full training should not be delayed forever, but E352 says what must be true
before it is worth spending that compute: the selector must transfer beyond a
single split and must optimize the metric family that matches the generative
compression claim. Once that is achieved, the full-training loss should remain
simple: original EF-LIC objective plus minimal HCG VQ/selector calibration.

## EF-LIC Learned Selector Readiness After E353

E353 prevents a premature full-training decision. The learned selector is
reasonable as a diagnostic probe: it uses only decoder-visible candidate
features, preserves exact bpp/decode behavior, and fits the calibration split
better than simple fixed-risk rows. But it does not transfer. The `global_slice`
selector changes from calibration score -0.000533 to held-out score +0.000157,
and the smaller `global` selector changes from -0.000429 to +0.000210. Lower is
better, so both held-out rows are harmful under the perceptual objective.

This result should not be read as "HCG-RVQ fails." The same held split still has
useful non-oracle fixed rows, especially risk -0.06 at score -0.000327 and risk
-0.10 at score -0.000239 with a safer PSNR diagnostic tail. The no-op/risk
oracle is much stronger at score -0.001106. The scientific signal is therefore
consistent with E350-E352: hyperprior/context-conditioned geometry can improve
external perceptual metrics without changing payload, but reliability selection
is the current bottleneck.

The paper-facing lesson is to keep the method simple but move the selector closer
to the actual local codec decision. A global image-level ridge model over 20 CLIC
calibration images is too weak and too noisy. The next selector should be trained
from a larger mixed teacher set, ideally Kodak plus CLIC Professional and, if
available, another low-bitrate validation source. It should use disjoint
train/calibration/eval splits and should report LPIPS, DISTS, MS-SSIM, bpp,
decode exactness, nonfinite rows, and PSNR only as a tail/codec-health
diagnostic.

Full EF-LIC training should wait for that frozen selector gate. Launching full
training with E353 would optimize around a selection policy that already
overfits. Launching full training after a mixed-split selector beats fixed-risk
baselines would be much more meaningful: then the full run tests whether
HCG-RVQ's local quantizer geometry improves the original EF-LIC objective, not
whether a small selector happened to match one split.

## EF-LIC Full-Training Gate After E354-E356

E354-E356 sharpen the full-training decision. Kodak24 shows why PSNR must remain
a diagnostic rather than the generative-compression objective: risk -0.06 has
positive mean dPSNR (+0.010798) but worsens the perceptual score (+0.000772).
All fixed-risk Kodak rows have positive perceptual score even though the
no-op/active oracle reaches -0.001610. This is strong evidence that HCG-RVQ should
not be deployed as a fixed always-on perturbation.

The positive result is that the oracle headroom transfers across both CLIC and
Kodak while bpp/decode exactness remains perfect. The HCG branch is not breaking
EF-LIC's fixed-payload/no-entropy-coding contract. It is changing local quantizer
geometry in a decoder-reproducible way, and some images benefit substantially.
That is exactly the scientific space HCG-RVQ is meant to occupy.

E355 also shows that more independent teacher data helps. Adding Kodak24 to the
CLIC first20 calibration set changes the held-out CLIC learned selector from
harmful in E353 to useful in E355: the `global_slice` selector reaches held score
-0.000278 with exact bpp/decode. This is not yet a paper-main result, because
fixed risk -0.06 is still slightly better by mean held score (-0.000327), but it
is a meaningful direction update. The selector is starting to transfer when it
sees more than one dataset.

E356 is a useful negative simplification. Removing risk -0.08 and using only
noop/risk -0.06/risk -0.10 does not improve the held score. This suggests the
problem is not merely too many candidate strengths; it is the coarseness of the
image-level selector and the calibration objective. The next controller should
move closer to the EF-LIC slice/local decision where the actual residual,
context, and active-state statistics are available.

Full EF-LIC training should therefore wait for one more selector stage. A good
promotion criterion is: a frozen selector trained on mixed calibration data must
beat the best fixed-risk mean score on held-out LPIPS/DISTS, keep worst score and
PSNR diagnostic tails controlled, and preserve max delta bpp 0, decode max 0,
and nonfinite 0. Once that criterion is met, full training becomes meaningful
because the experiment will test HCG-RVQ's local quantizer geometry rather than a
fragile global selection rule.

The immediate implementation direction is simple: keep EF-LIC's original
Representation-domain Decorrelation and no-entropy-coding payload unchanged;
keep the HCG branch small; train a local/sequential reliability head or a robust
objective selector from mixed-domain perceptual codec-gain labels. Avoid adding
many auxiliary losses. The full-training loss should remain the original
RD/perceptual codec objective plus minimal VQ commitment and selector
calibration, so the contribution stays interpretable.


## EF-LIC Selector Direction After E357-E358

E357 and E358 update the EF-LIC decision in a useful way. The paper objective for
this generative/low-bitrate branch should be perceptual: LPIPS, DISTS, MS-SSIM,
and bpp. PSNR can remain in the tables only as a diagnostic for distortion tails
or codec-health regressions, not as the selector or main claim metric.

The experiments are still順調 in the research sense, but not yet in the final
performance-claim sense. The good news is that HCG-RVQ has repeated headroom: the
mixed CLIC/Kodak oracle reaches score -0.001171, CLIC41 oracle reaches -0.000913,
Kodak24 oracle reaches -0.001610, and held CLIC21 oracle reaches -0.001106, all
with exact bpp/decode and no nonfinite rows. This says the quantizer-geometry edit
can improve perceptual behavior without violating EF-LIC's fixed-payload
contract.

The negative result is also clear: image-level source-weighted selectors do not
recover that headroom. The best E357 weighted row is only -0.000023 on held CLIC,
which is worse than fixed risk -0.06 (-0.000327) and fixed risk -0.10 (-0.000239).
E358 shows why: decoder-visible global/slice summary features have the correct
trend but weak separation. The best oracle-active AUC is only about 0.61, so the
selector is under-instrumented at the image level.

The designs that improve results are indeed coming from the same philosophy:

1. keep the original codec contract intact, including EF-LIC's decorrelation and
   no-entropy-coding payload;
2. make HCG a decoder-reproducible local quantizer-geometry edit, not a dense
   all-on perturbation;
3. keep a no-op/fallback path so harmful local states can avoid HCG;
4. use simple, interpretable objectives and avoid piling auxiliary losses onto
   the original RD/perceptual/VQ losses;
5. choose policies by perceptual metrics, with bpp/decode/nonfinite checks as
   hard constraints.

This also explains the repeated failures. Dense/all-on policies look intuitive
but hurt Kodak perceptual quality; coarse image-level learned selectors overfit;
and PSNR-oriented choices can disagree with the generative/perceptual objective.
The next EF-LIC method step should therefore be local/sequential reliability
control trained from mixed perceptual codec-gain labels. Full training becomes
worth starting when that frozen local policy beats the best fixed-risk row on
held-out LPIPS/DISTS while preserving exact bpp/decode behavior.


## EF-LIC Image-Level Selector Limit After E359-E360

E359-E360 make the EF-LIC selector bottleneck clearer. The experiments are still順調 in the research sense because the headroom and failure mode are now separated: HCG geometry edits have perceptual oracle gain under exact bpp/decode, but image-level selection cannot reliably recover it.

The calibration-selected E359 threshold transfers to held CLIC with only -0.000006 score, essentially no improvement and weaker than fixed risk -0.06 (-0.000327). E360 then shows that robust calibration objectives do not rescue the family. Penalizing worst score makes the policy almost all no-op and harmful; adding calibration-win constraints still stays below fixed-risk baselines.

There is an important nuance: held-ranked diagnostic thresholds can reach around -0.000586. That means the feature family contains some signal, but it is not stable enough to select at the image level from the available calibration data. This supports the next design step rather than invalidating HCG-RVQ: the reliability decision should be local/sequential, close to the slice/block where residual, active-state, and index-usage statistics are actually observed.

The designs that keep working share the same philosophy: preserve the original EF-LIC payload and decorrelation path, apply HCG as a decoder-reproducible local quantizer-geometry edit, keep a no-op fallback, and optimize/report perceptual metrics with bpp/decode/nonfinite checks as hard constraints. PSNR should not drive the generative-compression claim.


## EF-LIC Metric Correction After E361-E364

E361-E364 answer a key risk in the EF-LIC branch: yes, some methods that looked good under PSNR are not good under the generative/perceptual objective. This does not invalidate HCG-RVQ, but it changes which evidence is trustworthy.

The invalidated part is the PSNR-driven teacher/policy selection. Kodak fixed risk -0.06 is the clearest example: mean dPSNR is +0.010798, but perceptual score is +0.000772, so it is worse by the actual low-bitrate perceptual target. E364 makes the same point at slice level. The PSNR oracle and perceptual oracle choose different slice sets on 22/24 Kodak images. PSNR-best choices average +0.000473 perceptual score, while perceptual-best choices average -0.001198.

The surviving evidence is stronger and better aligned with the paper claim. Every E364 row preserves exact bpp/decode behavior, so the branch is still compatible with EF-LIC's no-entropy-coding payload. The per-image perceptual oracle reaches -0.001198 score and improves all 24 images among the tested slice sets, while all-on activation is +0.000717. This means the HCG geometry edit has real local headroom, but dense activation and PSNR-based selection expose the wrong part of it.

This also answers the full-training question. There is a reasonable expectation for a stronger HCG-RVQ EF-LIC result, but not from all-on training or PSNR teacher labels. The next full-training gate should be stricter: train or freeze a decoder-visible local controller from perceptual codec-gain labels, evaluate it on held-out images, and require improvement in LPIPS/DISTS/MS-SSIM at unchanged bpp/decode with no nonfinite rows. If that gate passes, full training is justified because it tests the real hypothesis: hyperprior/context-conditioned local quantizer geometry. If it does not pass, full training would likely spend compute optimizing a fragile selection rule rather than the HCG idea.

The design rule remains simple and matches the user's intuition: keep the loss and model interpretable. The core loss should remain the original codec objective plus perceptual/RD terms already appropriate for EF-LIC/GLC and minimal VQ/commitment/controller calibration. Do not add a large stack of auxiliary penalties to force a weak selector to work. The promising direction is simpler: local HCG geometry, exact fallback, perceptual labels, and hard payload consistency.


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

## EF-LIC Full-Training Gate After E366

E366 is the first positive sign after the PSNR metric correction that the
perceptual oracle may be recoverable by a simple policy. The leave-one-image-out
ridge policy reaches mean score -0.000522 on Kodak24, better than fixed 1,2,3
(-0.000187), fixed 2 (-0.000128), CV best fixed (-0.000041), no-op (0), all-on
(+0.000717), and the PSNR oracle (+0.000472). This means the PSNR correction did
not erase the HCG-RVQ direction. It removed the wrong evidence and exposed a
better one: perceptual local/candidate selection.

The result is not yet sufficient for the paper-main full-training branch. The
learned policy is noop-heavy and has only 6/24 score wins, while the oracle has
24/24 wins and score -0.001198. That gap says the next step should be a
constrained local controller, not dense all-on training. The controller should
select HCG only when decoder-visible local statistics predict perceptual gain,
while explicitly controlling worst-score tail and minimum win rate. If that gate
beats the fixed perceptual slice policies on held-out Kodak/CLIC with unchanged
bpp/decode and no nonfinite rows, EF-LIC full training becomes justified.

Paper implication: PSNR-based improvements should be rewritten as diagnostics,
not claims. The viable claim is that hyperprior/context-conditioned quantizer
geometry can improve perceptual low-bitrate compression under a fixed payload,
provided the HCG branch is local and reliability-controlled.

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

## GLC Perceptual Promotion After E369

E369 is the first current-code GLC selected-replacement expansion that makes the
next-step decision clear under perceptual metrics only. Full Kodak24 is strongly
positive with zero nonfinite rows: the trained soft gate has score -0.016910 and
fixed-index score -0.013487 with 24/24 empirical and fixed-index wins. The
replacement-soft row remains positive as well, while dense all-on is strongly
negative evidence for a dense RVQ transplant: score +0.038617 on Kodak24 and
+0.038981 on the pooled CLIC-tail9 + Kodak24 set.

The important design conclusion is not that GLC wants more VQ everywhere. It is
that the HCG-RVQ-style branch should be sparse, decoder-reproducible, and
reliability-controlled. On the pooled set, cap 0.0030 is the strict controller:
score -0.012200, fixed score -0.010834, selected fraction 0.787879, and both
selected empirical and selected fixed-index win fractions are 1.0. cap 0.0035 is
the balanced paper-facing setting: it selects 0.848485 of images and improves the
mean slightly more, but allows one tiny fixed-index positive tail. cap 0.0040 is
the aggressive setting: best mean score among the capped rows, but it has a
positive tail and should be reported as an ablation/performance branch rather
than the safest controller.

This moves GLC ahead of EF-LIC for the next larger evaluation. EF-LIC still has
clear perceptual oracle headroom, but image-level abstention cannot recover it
safely. GLC now has a controller family with explicit signal/index accounting,
zero nonfinite rows, and a full-Kodak24 confirmation. The next GLC action should
therefore be broader/full CLIC Professional and a low-rate q-index curve before
launching matched full fine-tuning. The next EF-LIC action remains local
slice/block-level reliability control that preserves the EF-LIC fixed-payload and
representation-domain decorrelation contract.

For the paper claim, E369 supports a simple statement: hyperprior/context-guided
quantizer replacement helps low-bitrate generative compression when it acts only
where the local residual state is reliable and when the side signal/index cost is
explicitly counted. It also supplies the negative control needed by reviewers:
dense all-on replacement fails even though the selected branch succeeds.

## GLC Full-CLIC Confirmation After E370

E370 upgrades the GLC evidence from selected subsets to the full CLIC
Professional validation set. The result is favorable under the corrected
perceptual-only protocol. On CLIC41, the trained soft gate is negative on every
image by empirical score, with mean score -0.012522 and zero nonfinite rows. The
selected replacement family remains positive after explicit bpp/signal
accounting, while dense all-on remains strongly bad: +0.030343 on CLIC41 and
+0.033398 on CLIC41+Kodak24.

The pooled CLIC41+Kodak24 table is now the best short-cycle decision artifact for
GLC. cap 0.0035 gives score -0.011797, fixed score -0.009910, selected fraction
0.876923, selected empirical win 1.0, selected fixed-index win 0.947368, and zero
nonfinite rows. cap 0.0030 is safer but leaves more mean gain unused. cap 0.0040
is slightly stronger on mean score, but its positive tail is larger and it should
stay an aggressive ablation rather than the default paper row.

This result strengthens the main HCG-RVQ claim because it repeats the same
mechanism across Kodak and CLIC: useful gains come from sparse, local,
reliability-controlled quantizer replacement, not from dense VQ/RVQ activation.
It also makes the full-training decision more concrete. GLC is now ready for a
larger low-rate curve audit and then matched fine-tuning/full training if the
q-index curve preserves the same cap ordering. EF-LIC should not be forced into
full training yet; its current bottleneck is still local reliability control
inside the fixed-payload/no-entropy EF-LIC contract.


## GLC q-Curve After E371-E374

E371-E374 are the first GLC expansion results that match the corrected
generative-compression evaluation stance: the promoted signal is perceptual-only
and excludes PSNR. The score is `delta_DISTS + 3 * delta_LPIPS + delta_bpp`, so
the evidence is aligned with the low-bitrate generative setting rather than with
pixel-fidelity improvements.

The main conclusion is positive. Selected local replacement is not a q=0 artifact
and not a Kodak-only artifact. Across q-indexes 0/1/2/3, CLIC Professional 41
gives replacement-soft score -0.007837 with win fraction 0.865854, Kodak24 gives
score -0.011288 with win fraction 0.989583, and the pooled CLIC41+Kodak24 result
is score -0.009111 with win fraction 0.911538. This remains true after explicit
signal/index accounting. The dense all-on control is consistently bad, with
pooled score +0.034618, which strengthens the interpretation that HCG-RVQ must
be reliability-controlled rather than activated everywhere.

The main limitation is also clear. The CLIC high-q tail still has positive
per-image cases, and simple rate caps do not remove that tail. This is not a
reason to drop the design; it is evidence that the controller must use local
decoder-side reliability signals. The strongest current diagnostic signal is
`index_entropy_mean`: on the pooled q-curve, `index_entropy_mean >= 1.78232`
selects 79/260 rows with score_all -0.003947, selected win fraction 1.0, and
selected worst score -0.001221. Because the threshold is selected in-sample, it
should be treated as design evidence rather than a paper claim.

The updated hypothesis is: HCG-RVQ helps GLC when the active residual state has
enough local index entropy/residual structure to benefit from a learned local
codebook replacement. It harms low-reliability or over-activated regions because
dense replacement adds perceptual distortion and index/signal cost. This is
consistent with the original prompt: hyperprior/context should generate local
quantizer geometry, but the generated geometry must include reliability and
fallback behavior.

Full-training timing: GLC is close to promotion, but one q-aware,
index-entropy-aware controller experiment should come before long matched
fine-tuning/full training. If that controller preserves the CLIC+Kodak
perceptual gains on held-out thresholds and keeps the selected positive tail
near zero, then full training is justified. EF-LIC should continue in parallel,
but its next step remains slice/block-level local control under the
fixed-payload/no-entropy EF-LIC contract rather than an all-on transplant.


## GLC Transfer Reliability After E375

E375 resolves an important risk in the E372-E374 feature-risk analysis: the
`index_entropy_mean` threshold could have been an in-sample artifact. The
cross-dataset transfer audit weakens that concern. A CLIC-selected high-entropy
threshold transfers cleanly to Kodak24, and a Kodak-selected high-entropy
threshold transfers cleanly back to CLIC41. In both directions the selected rows
have negative worst perceptual score on the target set.

The transfer result also says what not to do. Kodak-selected broad active-MSE
policies look strong on Kodak24 but allow positive CLIC tails. This matches the
previous dense-all-on failure mode: increasing replacement coverage improves mean
on easy data but is not paper-safe on the harder CLIC distribution. The more
credible HCG-RVQ story is therefore selective local geometry, not more
quantization everywhere.

The implementation implication is concrete. The next GLC branch should use a
q-aware reliability controller whose conservative mode is driven by high local
index entropy, with explicit fallback to the scalar/unchanged path outside that
region. Full training should start after this controller is implemented and
validated on held-out thresholds, because the evidence is now strong enough to
justify a longer run but still shows tail risk if the controller is too broad.

## GLC q-Aware Reliability After E376

E376 upgrades the GLC reliability result from an in-sample threshold observation
to a reusable controller design audit. The important change is that threshold
selection is q-aware and cross-dataset: CLIC41-selected policies are evaluated on
Kodak24, and Kodak24-selected policies are evaluated on CLIC41. This directly
tests whether the reliability signal is a real decoder-side condition rather
than a dataset-specific artifact.

The main evidence supports an entropy-first controller. High local
`index_entropy_mean` is the only feature family that repeatedly gives useful
coverage while suppressing selected positive tails. CLIC-to-Kodak q-aware entropy
thresholds are especially strong: the stricter fixed-tail profile selects 54/96
Kodak rows with negative empirical and fixed-index worst cases. This is exactly
the kind of behavior the HCG-RVQ paper needs: local geometry helps when the
active residual/index state has enough structure, and the fallback path protects
regions where the branch is unreliable.

The negative evidence is equally useful. Broad active-residual difficulty
features and source-fit q-aware thresholds can look excellent on Kodak but still
over-cover CLIC and leave positive tails. This matches the dense all-on failure
seen earlier. The HCG-RVQ story should therefore not be "use more VQ/RVQ". It
should be "use decoder-reproducible local reliability to decide where generated
quantizer geometry is safe." This wording aligns with the original prompt and is
more robust for a conference submission.

Full-training timing after E376: GLC is close, but one held-out calibration step
should precede long matched fine-tuning/full training. The controller to promote
is conservative high `index_entropy_mean` with explicit fallback; q-aware entropy
thresholds should be treated as the stronger variant once held-out calibration
keeps empirical and fixed-index tails non-positive. EF-LIC continues in parallel,
but its current evidence still points to slice/block-level control inside EF-LIC's
fixed-payload/no-entropy design rather than full-image or all-on activation.

## GLC Held-Out Promotion After E377-E378

E377/E378 change the GLC branch from a promising selector observation into a promotion candidate. The key mechanism is not a generic rate cap: high `index_entropy_mean`, especially with q-aware thresholds, identifies active residual states where the replacement branch can improve perceptual quality while paying its own index cost. The remaining risk after E377 was a small fixed-index positive tail on CLIC41, so E378 tested conservative threshold margins.

The q-aware entropy controller with a 0.02 margin is the first GLC policy in this sequence that is simultaneously perceptual-only, image-disjoint, pooled across CLIC41 and Kodak24, and fixed-tail safe. It keeps 92/260 held-out rows active with score -0.004717, fixed-index score -0.004147, and zero positive rows under both empirical and fixed-index accounting. This is strong enough to start a matched GLC fine-tuning/full-training branch, provided the codec path keeps exact scalar fallback for low-reliability states and the original GLC perceptual objective remains dominant.

The paper claim should be framed carefully: HCG-RVQ is useful when the hyperprior/decoder-visible state predicts local quantizer reliability, not as dense all-on replacement. For GLC, the next main implementation should be q-aware high-entropy activation with a safety margin, a scalar fallback, index-rate accounting, codebook-usage monitoring, and checkpoint evaluation by LPIPS/DISTS/MS-SSIM/bpp only. EF-LIC remains a parallel branch, but it has not yet passed the same promotion gate; its next step is still slice/block-level local perceptual reliability control under the no-entropy-coding contract.

## GLC Deployment Spec After E379

E379 resolves a practical gap after E378. The held-out sweep established that q-aware `index_entropy_mean` with a 0.02 margin is the safest high-gain GLC controller, but the fold-specific thresholds are validation artifacts. E379 fits the same controller on all current CLIC41+Kodak24 calibration rows and exports one deterministic deployment spec for the next matched GLC fine-tuning/full-training run.

The exported q-aware thresholds are `{0: 1.6570271146297455, 1: 1.747296993136406, 2: 1.8121935766935349, 3: 1.8215530556440354}`. These should be treated as the first main GLC run configuration. The global margin spec remains the simple ablation. The important methodological distinction is that E378 supports the claim; E379 supports reproducibility and implementation.

## GLC q-Aware Controller Integration After E380

E380 is a smoke-level implementation check for the GLC promotion path. The
important result is not the one-image score itself; it is that the E379 q-aware
entropy-margin spec is now executable inside the codec-loop pilot and produces
separate paper-main and ablation rows. The q-aware policy activates only when the
current q-index threshold is passed, while the global policy remains a simpler
conservative ablation.

The observed behavior matches the E378/E379 hypothesis. For `kodim01.png` at
q=0, `index_entropy_mean` is high enough for the q-aware threshold but below the
global threshold. The q-aware replacement row improves the perceptual score
(-0.016558 after one pilot step), dense all-on is strongly harmful (+0.134986),
and global fallback is exactly baseline aside from optional signal accounting.
This supports the implementation direction: HCG-RVQ should be deployed as
selective local quantizer replacement with scalar fallback, not as all-on RVQ.

Protocol note: the pilot table still carries legacy PSNR columns because older
analysis scripts share the writer, but PSNR is excluded from decisions and paper
claims for the generative low-bitrate track. The operative score remains
`delta_DISTS + 3 * delta_LPIPS + delta_bpp`, with MS-SSIM as a side metric.

Next decision: GLC is ready for a longer matched q-curve run using the E379 JSON
on GPU0. The first longer run should cover q=0/1/2/3, Kodak24 and CLIC
Professional, preserve exact fallback outside selected states, log signal/index
bpp, and monitor codebook usage/nonfinite rows. EF-LIC continues in parallel, but
its next promotion gate is still local slice/block reliability under the
fixed-payload no-entropy contract rather than a dense all-on transplant.


## GLC Promotion Evidence After E381/E382

E381/E382 materially strengthen the GLC branch. The earlier held-out result
(E378) said that q-aware high `index_entropy_mean` with a 0.02 margin can remove
positive tails when choosing where local RVQ replacement is safe. E381/E382 show
that this is not just an offline selector artifact: the exported E379 controller
now runs inside the codec-loop pilot and preserves the same safety profile on
both Kodak24 and CLIC Professional.

The main pattern is consistent across datasets. Dense all-on replacement is bad
because it applies local RVQ geometry to low-reliability residual states where
the added quantization/index burden hurts perceptual quality. Raw soft
replacement proves that the learned local branch has real capacity, but its mean
improvement hides positive tails, especially on CLIC and higher q. The q-aware
controller sacrifices coverage to make the branch paper-safe: Kodak24 keeps
45/96 rows and CLIC keeps 46/164 rows, with negative selected worst cases under
both empirical and fixed-index accounting.

This supports a clear HCG-RVQ claim for GLC: hyperprior/context-generated local
quantizer geometry helps low-bitrate generative compression when the decoder can
also infer reliability from the active residual/index state. The contribution is
not simply “more RVQ” and not a rate cap; it is selective, q-aware local geometry
with exact fallback. This framing is compatible with the prompt and with the
perceptual target of the GLC task.

Full-training timing: GLC has passed the promotion gate for a longer matched
run. The next GLC experiment should keep the q-aware entropy-margin controller as
the main branch, include the global-margin controller and raw soft replacement as
ablations, log LPIPS/DISTS/MS-SSIM/bpp only for paper decisions, and record
codebook usage, dead-code fraction, residual-stage contribution, selected-row
coverage, and fixed-index tail at every checkpoint. EF-LIC remains in parallel,
but it should use its own local slice/block reliability design rather than force
this GLC image-level entropy controller unchanged.

## EF-LIC Perceptual Controller Status After E383/E384

EF-LIC remains a viable parallel branch, but its evidence is qualitatively different from GLC. GLC now has a q-aware entropy controller that is safe on Kodak24 and CLIC Professional inside the codec-loop pilot. EF-LIC still has perceptual oracle headroom, but simple fixed-slice, all-on, and image-level learned policies are not paper-safe. E365/E366 showed the gap clearly: the perceptual oracle reaches mean score -0.001198 with a non-positive worst case, while the best learned image-level policy improves the mean to -0.000522 but leaves a +0.001118 worst case. E368 confirmed that scalar abstention can remove the positive tail only by discarding most useful decisions.

E383/E384 add a more promising design cue. When the controller is restricted to fixed actions gated by decoder-visible local features, held-out tail-safe improvements survive in both Kodak split directions. The best forward split uses `slice3_family_zero_prob_mean <= 0.743521` to activate `0,1,3`, selecting 3/8 held-out images with score -0.000851 and zero positive rows. The reverse split is more conservative, selecting 1/8 image with score -0.000782 and zero positive rows. This is not enough coverage for paper-main performance, but it is a stronger EF-LIC direction than image-level regression because the safety condition is local and decoder-reproducible.

The EF-LIC hypothesis should therefore be narrowed: HCG-RVQ should not be dense quantization replacement inside EF-LIC. It should be a local reliability controller layered on top of EF-LIC representation-domain decorrelation, preserving fixed payload/no entropy coding while activating HCG geometry only where local redundancy and zero-probability/risk features indicate perceptual benefit. The next EF-LIC experiment should combine the recurring feature families (`family_zero_prob`, slice risk, geometry delta, residual error) into a small local/slice-block controller and validate on CLIC Professional before promoting to longer matched training.

Paper implication: current strongest main branch is GLC q-aware entropy-margin. EF-LIC is still valuable as the second backbone, but it needs one more controller-design step before full training. The shared HCG-RVQ message remains coherent: generated quantizer geometry is useful only when paired with reliability/fallback, and the reliable signal differs by backbone.

## EF-LIC CLIC/Cross-Dataset Reliability After E385-E388

E385-E388 clarify the EF-LIC branch under the corrected perceptual/generative
protocol. The result is neither a failure nor a promotion pass. It is a sharper
local-control diagnosis.

The positive part is that HCG-RVQ projected geometry remains contract-safe in
EF-LIC on CLIC Professional 41: no bpp drift, no decode mismatch, and no nonfinite
rows. There is also real perceptual headroom. CLIC's per-image best slice sets
often differ from all-on, and cross-dataset E388 finds tail-safe rules in both
directions: CLIC -> Kodak reaches score -0.000321 with 2/24 selected rows, and
Kodak -> CLIC reaches score -0.000199 with 5/41 selected rows. These rules use
geometry/index features available to the decoder, so they are compatible with
EF-LIC's fixed-payload design.

The negative part is equally important. Dense all-on is still not safe: CLIC
all-on has mean score -0.000257 but worst score +0.003001. Broader fixed-action
rules improve the mean only by accepting positive tails. The best non-safe
Kodak -> CLIC rule selects 16/41 rows and reaches mean -0.000285, but leaves
6 positive rows. That is not a conference-paper mainline.

The EF-LIC conclusion is therefore: do not move EF-LIC to full training with an
image-level or fixed-slice controller yet. The next EF-LIC method should increase
coverage by making the controller spatial/slice-block local, while preserving the
original Representation-domain Decorrelation and no-entropy-coding contract. A
reasonable implementation target is a decoder-reproducible local reliability head
that consumes `family_zero_prob`, geometry-delta magnitude/tail, residual-error
statistics, slice risk, and index-usage features, with explicit zero/fallback as
the default state. This may be trained with perceptual labels, but the final loss
should remain simple: the original EF-LIC perceptual/R-D objective plus only the
minimal VQ/reliability terms needed for stability.

The GLC/EF-LIC split is now clean. GLC has already passed the longer-run promotion
gate via q-aware high-entropy reliability and exact scalar fallback. EF-LIC is
still a parallel RVQ plug-in branch, but needs one more local-controller design
step before matched full training. This is consistent with the HCG-RVQ thesis:
hyperprior-conditioned quantizer geometry is useful when paired with
backbone-specific decoder-visible reliability control, not when forced on densely.
