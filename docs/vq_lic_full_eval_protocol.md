# VQ-LIC Full Evaluation Protocol for HCG-RVQ

This note connects the short-cycle HCG-RVQ evidence to paper-facing EF-LIC/GLC experiments. Short runs remain implementation and design probes; final claims need independent split or matched full evaluation.

## Current Claim State

HCG-RVQ's core claim is: hyperprior/context should generate local quantizer geometry, not only entropy parameters. The strongest current VQ-LIC evidence is EF-LIC force0 projected-HCG with decoder-reproducible reliability control:

- E186 DISTS-oriented rule: `slice0_mean_abs_mean <= 0.455595642328`, Kodak diagnostic `dDISTS=-0.001238`, `dLPIPS=-0.000112`, unchanged bpp.
- E188 LPIPS-balanced rule: `slice0_mean_min >= -10.7447786331`, Kodak diagnostic `dDISTS=-0.000870`, `dLPIPS=-0.000468`, unchanged bpp.
- E190 multi-objective rule: `slice0_mean_min >= -10.7447786331` is rediscovered by `DISTS=1, LPIPS=3` search, with LOOCV `dDISTS=-0.000598`, `dLPIPS=-0.000393` and split-eval averages `dDISTS=-0.000282`, `dLPIPS=-0.000171` across four splits.
- E191/E192 failure-mode audit: E190 has fewer multi-objective selected-bad cases than E186 (`5` vs `10`). A two-stage hand rule improves same-Kodak metrics but is weaker under split-fit/eval, so it is a learned-controller target, not the default held-out rule.
- E187/E190 warning: the signal survives small held-out splits, but same-Kodak threshold rows are not final paper claims. Independent fit/eval is required before a paper table.

## EF-LIC Independent Selector Protocol

Prepare two image directories. The fit directory must not overlap the final held-out evaluation directory.

```text
experiments/data/eflic_selector_fit/      # e.g. OpenImages/CLIC/DIV2K validation crops or images
experiments/data/eflic_selector_eval/     # e.g. Kodak/Tecnick/DIV2K/CLIC test split
```

Run projected-HCG active-vs-baseline labels on the fit split. Keep GPU fixed to device 0.

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir experiments/data/eflic_selector_fit \
  --force-ind 0 \
  --alpha 0.05 \
  --direction-source mean \
  --output-prefix experiments/analysis/e190_eflic_selector_fit_active_labels
```

Build selector labels and feature manifest.

```bash
.venv/bin/python tools/build_e161_eflic_projected_hcg_selector_labels.py \
  --input-csv experiments/analysis/e190_eflic_selector_fit_active_labels.csv \
  --output-prefix experiments/analysis/e190_eflic_selector_fit_labels
```

Fit a no-side-bit global predecision selector. LPIPS target is currently the safer default because it improved both LPIPS and DISTS in E187/E188.

```bash
.venv/bin/python tools/fit_e189_eflic_global_selector_rule.py \
  --input-csv experiments/analysis/e190_eflic_selector_fit_labels.csv \
  --manifest-csv experiments/analysis/e190_eflic_selector_fit_labels_feature_manifest.csv \
  --output-prefix experiments/analysis/e190_eflic_selector_fit_rule_lpips \
  --target lpips \
  --force 0 \
  --feature-set global_predecision_context \
  --eval-dir-placeholder experiments/data/eflic_selector_eval
```

For the current mainline, also run the multi-objective E190 selector audit on the fit labels. The `DISTS=1, LPIPS=3` setting is the recommended default because it was the most stable Kodak diagnostic across LOOCV and split-fit/eval.

```bash
.venv/bin/python tools/analyze_e190_eflic_multiobjective_selector.py \
  --input-csv experiments/analysis/e190_eflic_selector_fit_labels.csv \
  --output-prefix experiments/analysis/e190_eflic_selector_fit_rule_multiobj_d1_l3 \
  --force 0 \
  --feature-set global_predecision_context \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --psnr-weight 0.0 \
  --positive-penalty 20.0
```

Run direct held-out EF-LIC evaluation with the fitted rule. Prefer the multi-objective rule if it improves both DISTS and LPIPS on the independent fit audit; otherwise fall back to the LPIPS-target E189 rule and report the disagreement as analysis. Use the printed command from the selected rule artifact; it will look like this:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e186_eflic_global_predecision_selector_probe.py \
  --device cuda:0 \
  --kodak-dir experiments/data/eflic_selector_eval \
  --force-ind 0 \
  --alpha 0.05 \
  --direction-source mean \
  --selector-feature FITTED_FEATURE \
  --selector-op FITTED_OP \
  --selector-threshold FITTED_THRESHOLD \
  --output-prefix experiments/analysis/e190_eflic_selector_heldout_eval_lpips
```

Paper-facing success condition:

- encoder/decoder selector match is exact;
- bpp is unchanged or explicitly side-bit accounted;
- DISTS and LPIPS improve on held-out data;
- PSNR/MS-SSIM changes are reported, even if not primary;
- per-image failures are analyzed by feature distribution and active/fallback decision.

## EF-LIC Rate Curve Extension

After force0 survives independent fit/eval, repeat the same pipeline for `force_ind=1..4`. E184/E185 currently say force1-4 are not stable with a scalar global selector, so treat them as ablations first, not as expected wins.

For a full RD curve, evaluate:

- official EF-LIC baseline at force0-4;
- always-active HCG at force0-4 as a failure/upper-risk diagnostic;
- selected HCG at force0-4 with independently fitted rules;
- oracle selector only as an analysis upper bound, never as a deployable row.

## GLC Parallel Protocol

GLC remains a main low-bitrate generative track, but it is heavier than EF-LIC. Current evidence says not to use dense always-on VQ. Use the E166/E169 path:

- preserve official `forward_four_part_prior()` behavior exactly as fallback;
- target the sparse active subset of `y` residuals rather than all residuals;
- use part/group-local HCG-RVQ or RVQ codebooks;
- add bit-aware/index-prior modeling before claiming rate gains;
- train with decoder/perceptual-aware losses because residual MSE and DISTS diverge.

Paper-facing GLC rows should use Stage-II/III-compatible matched fine-tuning or retraining. Fixed Kodak diagnostic codebooks are useful ablations, not final claims.

## Dataset/Training Reality Check

Current workspace data are not enough for final claims: `experiments/data` contains only Kodak24 and Kodak first4. EF-LIC official code here is inference-only. Therefore the immediate next blocker is data/protocol, not model code. Once independent fit/eval images are available, the EF-LIC commands above can run without changing code. Full EF-LIC retraining would require official training code or a local reconstruction of the ImageNet training recipe.

## E193 Reliability-Head Protocol Update

E193 adds a learned-controller option for EF-LIC, but it should not be reported as a final paper row when trained and evaluated on Kodak24. Same-table Kodak training reduces E190 false positives, but LOOCV and split-fit/eval are unstable. For paper-facing use, train the reliability head only on an independent fit split that has E161-style active/fallback labels, then apply it through the direct EF-LIC evaluation path on held-out data.

Recommended ordering:

1. Keep E190 DISTS=1, LPIPS=3 scalar primary as the default controlled rule for immediate held-out/full evaluation.
2. Generate independent active/fallback labels for EF-LIC force0 projected HCG.
3. Fit both the E190 scalar controller and the E193 logistic reliability head on that fit split.
4. Run direct held-out evaluation with exact encoder/decoder selector matching, unchanged bpp, and nonfinite checks.
5. Promote the learned head only if it improves held-out DISTS and LPIPS without increasing false-positive active selections.

## E194 Direct Reliability-Head Evaluation Command

E194 provides the direct EF-LIC path for a learned reliability controller. Use it after an independent fit split has produced E161-style active/fallback labels. It is acceptable as an implementation smoke on Kodak, but it becomes paper-facing only when the fit CSV and evaluation directory are disjoint.

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e194_eflic_reliability_head_selector_probe.py \
  --device cuda:0 \
  --fit-csv experiments/analysis/INDEPENDENT_FIT_LABELS.csv \
  --fit-manifest-csv experiments/analysis/INDEPENDENT_FIT_LABELS_feature_manifest.csv \
  --eval-dir experiments/data/eflic_selector_eval \
  --force-ind 0 \
  --alpha 0.05 \
  --direction-source mean \
  --feature-set global_predecision_context \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --positive-penalty 20.0 \
  --output-prefix experiments/analysis/e194_eflic_reliability_head_heldout_eval_d1_l3
```

Required checks before using the result in a paper table:

- encoder and decoder selector decisions match for every image;
- max selector probability difference is zero or numerically negligible;
- active decode difference is zero;
- nonfinite rows are zero;
- bpp is unchanged, unless side bits are explicitly accounted;
- DISTS and LPIPS are reported together, with PSNR and MS-SSIM as secondary distortion metrics;
- per-image active/fallback failures are analyzed rather than hidden by averages.

Current Kodak diagnostic status: first4 smoke selected branch share `0.750` with `dDISTS=-0.001836` and `dLPIPS=-0.000771`; full Kodak24 self-check selected branch share `0.500` with `dDISTS=-0.000881` and `dLPIPS=-0.000542`. Both have exact encoder/decoder matching and no nonfinite rows. These are deployability checks, not final generalization claims.


## Independent Fit/Calibration/Eval Update After E196-E201

E196-E201 clarify the paper-facing EF-LIC protocol. Non-Kodak images are available under `/dpl`, so the immediate EF-LIC track should use explicit fit/calibration/eval separation instead of same-Kodak diagnostics.

Generate independent active/fallback labels:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/openimages/open-images-v6/train/data \
  --start-index 8192 \
  --max-images 64 \
  --force-ind 0 \
  --alpha 0.05 \
  --direction-source mean \
  --output-prefix experiments/analysis/e199_eflic_openimages8192_fit64_active_labels
```

Build labels and fit/analyze controller candidates:

```bash
.venv/bin/python tools/build_e161_eflic_projected_hcg_selector_labels.py \
  --input-csv experiments/analysis/e199_eflic_openimages8192_fit64_active_labels.csv \
  --output-prefix experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels

.venv/bin/python tools/analyze_e190_eflic_multiobjective_selector.py \
  --input-csv experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels.csv \
  --manifest-csv experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels_feature_manifest.csv \
  --output-prefix experiments/analysis/e199_eflic_openimages8192_fit64_multiobj_selector \
  --force 0 \
  --feature-set global_predecision_context \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --positive-penalty 20.0
```

Direct held-out eval with a learned reliability head:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e194_eflic_reliability_head_selector_probe.py \
  --device cuda:0 \
  --fit-csv experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels.csv \
  --fit-manifest-csv experiments/analysis/e199_eflic_openimages8192_fit64_selector_labels_feature_manifest.csv \
  --eval-dir /dpl/kodak \
  --start-index 0 \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.05 \
  --direction-source mean \
  --feature-set global_predecision_context \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --positive-penalty 20.0 \
  --output-prefix experiments/analysis/e200_eflic_openimages64_head_to_kodak24_direct_probe
```

Calibration diagnostic, not a paper-facing selector unless threshold is chosen on a separate calibration split:

```bash
.venv/bin/python tools/analyze_e201_eflic_head_threshold_transfer.py \
  --input-csv experiments/analysis/e200_eflic_openimages64_head_to_kodak24_direct_probe.csv \
  --output-prefix experiments/analysis/e201_e200_openimages64_head_kodak24_threshold_audit \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --positive-penalty 20.0
```

Current evidence: OpenImages16-to-Kodak24 direct transfer improves both DISTS and LPIPS (`dDISTS=-0.000350`, `dLPIPS=-0.000036`) with exact encoder/decoder agreement. OpenImages64-to-Kodak24 needs threshold calibration: the raw threshold is too conservative, while an eval-threshold diagnostic gives `dDISTS=-0.000254`, `dLPIPS=-0.000296`. The next full protocol must add an independent calibration split before promoting this result.


## Three-Way Calibration Probe Status

The first fit/calibration/eval attempt is complete. Use this as a guardrail before any paper claim:

- fit: OpenImages64 from `start-index=8192`;
- calibration: disjoint OpenImages64 from `start-index=16384`;
- eval: Kodak24 under `/dpl/kodak`.

The implementation is codec-valid in all cases: encoder/decoder decisions match exactly, active decode diff is zero, bpp is unchanged, and nonfinite rows are zero. The current controller still fails to transfer after calibration. OpenImages64-calibrated threshold `0.324296` gives Kodak24 `dDISTS=+0.000034`, `dLPIPS=+0.000071`. OpenImages16-calibrated threshold `0.775014` gives Kodak24 `dDISTS=+0.000221`, `dLPIPS=+0.000022`.

Next protocol revision:

1. Add a feature/probability distribution-shift audit across OpenImages fit, OpenImages calibration, Kodak, CLIC, and Tecnick.
2. Calibrate with a dataset-mixed split that does not include the final evaluation images.
3. Try stricter false-positive penalties or LPIPS/DISTS joint labels before applying the threshold to Kodak/Tecnick/CLIC.
4. Report E198 as positive transfer smoke only, not as final evidence; report E204/E207 as guardrail negative evidence if needed.


## E208-E210 Protocol Revision

The EF-LIC paper-facing branch should now use `mean/alpha=0.02` as the active-state candidate for the next round, while treating the current learned reliability head as diagnostic only.

Rebuild active labels on an independent fit split:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/openimages/open-images-v6/train/data \
  --start-index 8192 \
  --max-images 64 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_active_labels

.venv/bin/python tools/build_e161_eflic_projected_hcg_selector_labels.py \
  --input-csv experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_active_labels.csv \
  --output-prefix experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_selector_labels
```

Direct held-out check with the learned head remains diagnostic:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e194_eflic_reliability_head_selector_probe.py \
  --device cuda:0 \
  --fit-csv experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_selector_labels.csv \
  --fit-manifest-csv experiments/analysis/e210_eflic_openimages8192_fit64_mean_alpha002_selector_labels_feature_manifest.csv \
  --eval-dir /dpl/kodak \
  --start-index 0 \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --feature-set global_predecision_context \
  --dists-weight 1.0 \
  --lpips-weight 3.0 \
  --positive-penalty 20.0 \
  --output-prefix experiments/analysis/e210_eflic_openimages64_mean_alpha002_head_to_kodak24_direct_probe
```

Current decision criteria before full training/full evaluation:

1. Run `mean/alpha=0.02` always-active on at least Kodak, Tecnick, CLIC mobile/professional, and an OpenImages held-out split. Promote it only if both DISTS and LPIPS are stable or if a predeclared safety selector preserves them.
2. Use `tools/analyze_e208_eflic_reliability_shift_audit.py` whenever fitting a head: any paper-facing controller must report probability AUC, feature shift, branch share, selected-good/bad, and exact encoder/decoder agreement.
3. Do not tune thresholds on final eval rows. Calibration must be chosen from disjoint data, preferably domain-mixed and excluding the final test images.
4. Full EF-LIC/GLC training should start after the active-state and reliability policy are fixed enough that the experiment tests a method rather than a moving target. Until then, short-cycle smokes are design selection, not final performance evidence.


## E211-E213 Cross-Dataset Active-State And Strength-Controller Protocol

Additional held-out always-active checks for the current EF-LIC active branch:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/openimages/open-images-v6/train/data \
  --start-index 24576 \
  --max-images 64 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e211_eflic_openimages24576_eval64_mean_alpha002_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/clic/mobile/valid \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e211_eflic_clic_mobile24_mean_alpha002_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/clic/professional/valid \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e211_eflic_clic_professional24_mean_alpha002_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/div2k/DIV2K_valid_HR \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e211_eflic_div2k_valid24_mean_alpha002_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/tecnick/SAMPLING/8BIT/RGB/1200x1200/B01R01 \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.02 \
  --direction-source mean \
  --output-prefix experiments/analysis/e211_eflic_tecnick_b01r01_24_mean_alpha002_active
```

Aggregate E211:

```bash
.venv/bin/python tools/analyze_e211_eflic_cross_dataset_active_state.py
```

Failure-split strength sweep:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/clic/professional/valid \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.005 0.01 0.015 0.02 0.03 \
  --direction-source mean \
  --output-prefix experiments/analysis/e212_eflic_clic_professional24_mean_alpha_sweep_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/tecnick/SAMPLING/8BIT/RGB/1200x1200/B01R01 \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.005 0.01 0.015 0.02 0.03 \
  --direction-source mean \
  --output-prefix experiments/analysis/e212_eflic_tecnick_b01r01_24_mean_alpha_sweep_active
```

Current decision rule for the next full protocol: do not promote `mean/alpha=0.02` as a universal always-active method. Use it as the active candidate inside a strength controller. The controller must include `alpha=0` fallback, be calibrated on disjoint mixed-domain data, and be evaluated on final held-out Kodak/Tecnick/CLIC/OpenImages without threshold tuning on final rows.


## E214-E216 Mixed-Domain Strength-Controller Diagnostic

Generate additional mixed-domain alpha labels, always on GPU0:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/clic/mobile/valid \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.005 0.01 0.015 0.02 0.03 \
  --direction-source mean \
  --output-prefix experiments/analysis/e215_eflic_clic_mobile24_mean_alpha_sweep_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/openimages/open-images-v6/train/data \
  --start-index 24576 \
  --max-images 32 \
  --force-ind 0 \
  --alpha 0.005 0.01 0.015 0.02 0.03 \
  --direction-source mean \
  --output-prefix experiments/analysis/e215_eflic_openimages24576_eval32_mean_alpha_sweep_active

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/div2k/DIV2K_valid_HR \
  --max-images 24 \
  --force-ind 0 \
  --alpha 0.005 0.01 0.015 0.02 0.03 \
  --direction-source mean \
  --output-prefix experiments/analysis/e215_eflic_div2k_valid24_mean_alpha_sweep_active
```

Run the mixed-domain controller probe:

```bash
.venv/bin/python tools/analyze_e214_eflic_strength_controller_probe.py \
  --output-prefix experiments/analysis/e216_eflic_mixed_domain_strength_controller_probe
```

Current decision rule: the mixed-domain alpha oracle is strong enough to justify continuing HCG-RVQ strengthening, but the current whole-image global predecision controller is not strong enough to become the method. Before full EF-LIC/GLC training claims, implement and test a local/sequential controller that uses decoder-available per-slice context and preserves exact encoder/decoder reproducibility without side bits.

## E217 Slice-Schedule Probe Commands

The slice-schedule probe extends the EF-LIC projected-HCG smoke with deterministic per-slice strengths:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_e160_eflic_projected_hcg_smoke.py \
  --device cuda:0 \
  --kodak-dir /dpl/clic/professional/valid \
  --max-images 12 \
  --force-ind 0 \
  --direction-source mean \
  --slice-alpha-schedule all010:0.01,0.01,0.01,0.01 \
  --slice-alpha-schedule all020:0.02,0.02,0.02,0.02 \
  --slice-alpha-schedule early020:0.02,0.02,0,0 \
  --slice-alpha-schedule late020:0,0,0.02,0.02 \
  --slice-alpha-schedule decay020:0.02,0.015,0.01,0.005 \
  --slice-alpha-schedule rise020:0.005,0.01,0.015,0.02 \
  --slice-alpha-schedule slice0only020:0.02,0,0,0 \
  --slice-alpha-schedule slice1only020:0,0.02,0,0 \
  --slice-alpha-schedule slice2only020:0,0,0.02,0 \
  --slice-alpha-schedule slice3only020:0,0,0,0.02 \
  --output-prefix experiments/analysis/e217_eflic_clic_professional12_slice_schedule_probe
```

Use the same command with `--kodak-dir /dpl/tecnick/SAMPLING/8BIT/RGB/1200x1200/C00R01` and output prefix `experiments/analysis/e217_eflic_tecnick12_slice_schedule_probe` for Tecnick. Aggregate with:

```bash
.venv/bin/python tools/analyze_e217_eflic_slice_schedule_probe.py \
  --input clic_professional=experiments/analysis/e217_eflic_clic_professional12_slice_schedule_probe.csv \
  --input tecnick=experiments/analysis/e217_eflic_tecnick12_slice_schedule_probe.csv \
  --output-prefix experiments/analysis/e217_eflic_slice_schedule_probe_summary
```

